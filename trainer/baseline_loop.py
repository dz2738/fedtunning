"""Train the prompt-only baselines on the shared frozen backbone."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from baselines.fedavg_prompt import FedAvgPrompt
from baselines.ifca_prompt import IFCAPrompt
from baselines.local_prompt import LocalPrompt, PromptOptimizationConfig
from baselines.per_fedavg_prompt import PerFedAvgConfig, PerFedAvgPrompt
from baselines.shared_local_prompt import SharedLocalPrompt
from client.state import InnerLoopConfig
from data.schema import DataSplit
from model.soft_prompt import initialize_prompt_
from trainer.evaluator import summarize_client_results

try:
    from scripts.train import (
        ExperimentRuntime,
        _append_jsonl,
        _prepare_output_dir,
        _run_metadata,
        plain_mapping,
    )
except ImportError:
    from train import (
        ExperimentRuntime,
        _append_jsonl,
        _prepare_output_dir,
        _run_metadata,
        plain_mapping,
    )


BASELINE_METHODS = (
    "local_prompt",
    "fedavg_prompt",
    "shared_local_prompt",
    "ifca_prompt",
    "per_fedavg_prompt",
)


def _optimization_config(inner: InnerLoopConfig) -> PromptOptimizationConfig:
    return PromptOptimizationConfig(
        steps=inner.steps,
        learning_rate=max(inner.coordinate_lr, 1.0e-8),
        weight_decay=inner.coordinate_weight_decay,
    )


def _blank_prompt(runtime: ExperimentRuntime) -> Tensor:
    prompt = torch.empty(
        runtime.server.prompt_length,
        runtime.server.hidden_size,
        device=runtime.backbone.runtime_device,
        dtype=runtime.backbone.compute_dtype,
    )
    initialize_prompt_(prompt, strategy="normal", init_std=runtime.server.prompt_init_std)
    return prompt


def _build_method(name: str, runtime: ExperimentRuntime) -> Any:
    initial = _blank_prompt(runtime)
    if name == "local_prompt":
        return LocalPrompt(initial)
    if name == "fedavg_prompt":
        return FedAvgPrompt(initial)
    if name == "shared_local_prompt":
        shared_length = max(1, runtime.server.prompt_length // 2)
        if shared_length >= runtime.server.prompt_length:
            raise ValueError("shared_local_prompt needs prompt_length >= 2")
        return SharedLocalPrompt(initial, shared_length=shared_length)
    if name == "ifca_prompt":
        clusters = runtime.server.grouper.config.target_num_groups or 4
        clusters = max(2, int(clusters))
        stacked = torch.stack(
            [_blank_prompt(runtime) for _ in range(clusters)],
            dim=0,
        )
        return IFCAPrompt(stacked)
    if name == "per_fedavg_prompt":
        return PerFedAvgPrompt(initial)
    raise ValueError(f"unsupported baseline {name!r}")


def _client_prompts(method: Any, client_ids: Sequence[str]) -> dict[str, Tensor]:
    prompts: dict[str, Tensor] = {}
    for client_id in client_ids:
        if isinstance(method, LocalPrompt):
            prompts[client_id] = method.prompt(client_id).detach()
        elif isinstance(method, FedAvgPrompt):
            prompts[client_id] = method.global_prompt.detach()
        elif isinstance(method, SharedLocalPrompt):
            prompts[client_id] = method.prompt(client_id).detach()
        elif isinstance(method, IFCAPrompt):
            # Last assignment is recovered by evaluating all clusters is expensive;
            # broadcast cluster 0 as a fallback and overwrite after the round.
            prompts[client_id] = method.cluster_prompts[0].detach()
        elif isinstance(method, PerFedAvgPrompt):
            prompts[client_id] = method.global_prompt.detach()
        else:
            raise TypeError(f"cannot read prompts from {type(method)!r}")
    return prompts


def _apply_round_prompts(
    method: Any,
    *,
    client_ids: Sequence[str],
    round_result: Any,
) -> dict[str, Tensor]:
    prompts = _client_prompts(method, client_ids)
    if isinstance(method, IFCAPrompt):
        for client_id, cluster_id in round_result.assignments.items():
            prompts[client_id] = method.cluster_prompts[cluster_id].detach()
    elif isinstance(method, PerFedAvgPrompt):
        prompts.update(
            {
                client_id: prompt.detach()
                for client_id, prompt in round_result.adapted_prompts.items()
            }
        )
    elif isinstance(method, LocalPrompt):
        for client_id in client_ids:
            prompts[client_id] = method.prompt(client_id).detach()
    return prompts


def evaluate_baseline_prompts(
    runtime: ExperimentRuntime,
    prompts: Mapping[str, Tensor],
    *,
    adaptation_steps: int,
    seed: int,
    split: DataSplit,
    method_name: str,
    round_number: int,
) -> Any:
    results = []
    for position, client_id in enumerate(sorted(prompts)):
        results.append(
            runtime.clients[client_id].evaluate_prompt(
                prompts[client_id],
                group_id=method_name,
                config=runtime.server.inner_loop,
                adaptation_steps=adaptation_steps,
                seed=seed + position,
                split=split,
            )
        )
    return summarize_client_results(
        results,
        clients=runtime.clients,
        round_number=round_number,
        adaptation_steps=adaptation_steps,
        split=split,
        worst_client_fraction=runtime.evaluator.worst_client_fraction,
    )


def run_baseline_training(
    config: DictConfig,
    runtime: ExperimentRuntime,
    *,
    method_name: str,
) -> dict[str, Any]:
    if method_name not in BASELINE_METHODS:
        raise ValueError(f"baseline {method_name!r} is not wired")
    output_dir = Path(str(config.output_dir)).resolve()
    checkpoint_config = plain_mapping(config.checkpoint)
    _prepare_output_dir(output_dir, resume_from=checkpoint_config.get("resume_from"))
    metadata = _run_metadata(config)
    metadata["baseline_method"] = method_name
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )

    experiment = plain_mapping(config.experiment)
    num_rounds = int(experiment["num_rounds"])
    eval_every = int(experiment.get("eval_every_rounds", 5))
    eval_steps = tuple(int(step) for step in experiment.get("eval_inner_steps", (0, 5)))
    selection_steps = int(experiment.get("selection_adaptation_steps", max(eval_steps)))
    inner = runtime.server.inner_loop
    opt_config = _optimization_config(inner)
    method = _build_method(method_name, runtime)
    client_ids = tuple(sorted(runtime.clients))
    latest_prompts = _client_prompts(method, client_ids)
    best_loss = float("inf")
    best_prompts: dict[str, Tensor] | None = None
    best_record: dict[str, Any] | None = None

    for round_number in range(1, num_rounds + 1):
        objectives = [
            runtime.clients[client_id].prompt_objective(
                config=inner,
                seed=int(config.seed) + round_number * 100_003 + index,
            )
            for index, client_id in enumerate(client_ids)
        ]
        if isinstance(method, PerFedAvgPrompt):
            round_result = method.run_round(
                objectives,
                PerFedAvgConfig(
                    inner=opt_config,
                    server_learning_rate=float(
                        plain_mapping(config.method.server_optimizer).get("lr", 0.01)
                    ),
                ),
            )
        elif isinstance(method, LocalPrompt):
            updates = [
                method.run_client(objective, opt_config) for objective in objectives
            ]
            round_result = type("LocalRound", (), {"client_updates": tuple(updates)})()
        else:
            round_result = method.run_round(objectives, opt_config)
        latest_prompts = _apply_round_prompts(
            method,
            client_ids=client_ids,
            round_result=round_result,
        )
        mean_train = None
        if hasattr(round_result, "mean_train_loss"):
            mean_train = float(round_result.mean_train_loss)
        elif hasattr(round_result, "mean_query_loss"):
            mean_train = float(round_result.mean_query_loss)
        elif hasattr(round_result, "client_updates"):
            updates = round_result.client_updates
            mean_train = sum(update.mean_train_loss for update in updates) / len(updates)
        _append_jsonl(
            output_dir / "rounds.jsonl",
            {
                "round_number": round_number,
                "method": method_name,
                "mean_train_loss": mean_train,
                "clients": list(client_ids),
            },
        )
        if round_number % eval_every != 0 and round_number != num_rounds:
            continue
        for steps in eval_steps:
            summary = evaluate_baseline_prompts(
                runtime,
                latest_prompts,
                adaptation_steps=steps,
                seed=int(config.seed),
                split=DataSplit.VALIDATION,
                method_name=method_name,
                round_number=round_number,
            )
            _append_jsonl(output_dir / "evaluations.jsonl", asdict(summary))
            if steps == selection_steps and summary.mean_test_loss < best_loss:
                best_loss = summary.mean_test_loss
                best_prompts = {
                    client_id: prompt.detach().cpu()
                    for client_id, prompt in latest_prompts.items()
                }
                best_record = {
                    "round_number": round_number,
                    "adaptation_steps": steps,
                    "mean_validation_loss": summary.mean_test_loss,
                }

    selected_prompts = (
        {
            client_id: prompt.to(
                device=runtime.backbone.runtime_device,
                dtype=runtime.backbone.compute_dtype,
            )
            for client_id, prompt in best_prompts.items()
        }
        if best_prompts is not None
        else latest_prompts
    )
    if best_record is not None:
        (output_dir / "best_validation.json").write_text(
            json.dumps(best_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    test_summaries = [
        evaluate_baseline_prompts(
            runtime,
            selected_prompts,
            adaptation_steps=steps,
            seed=int(config.seed),
            split=DataSplit.TEST,
            method_name=method_name,
            round_number=num_rounds,
        )
        for steps in eval_steps
    ]
    payload = {
        "method": method_name,
        "checkpoint": "best_validation" if best_prompts is not None else "final",
        "summaries": [asdict(summary) for summary in test_summaries],
    }
    (output_dir / "test_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
