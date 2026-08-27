"""Lightweight client that reuses the process-wide shared backbone."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from client.inner_loop import adapt_and_compute_feedback
from client.state import (
    ClientEvaluationResult,
    ClientRoundResult,
    ClientState,
    InnerLoopConfig,
    MetaGradientMode,
)
from data.federated_data import ClientData
from data.schema import DataSplit, TextExample
from metrics.task_metrics import compute_task_metric
from model.backbone import BackboneBatch, SharedBackbone
from model.prompt_subspace import PromptSubspace


@dataclass(frozen=True, slots=True)
class TextBatchConfig:
    max_source_length: int = 384
    max_target_length: int = 96


def _batched_examples(
    examples: Sequence[TextExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[tuple[TextExample, ...], ...]:
    if not examples:
        raise ValueError("client split cannot be empty")
    indices = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return tuple(
        tuple(examples[int(index)] for index in indices[start : start + batch_size])
        for start in range(0, len(indices), batch_size)
    )


def _clip_for_upload(residual: Tensor, max_norm: float | None) -> Tensor:
    """Detach and norm-clip a residual before it crosses the client boundary."""

    value = residual.detach()
    if max_norm is None:
        return value
    if max_norm <= 0:
        raise ValueError("residual_upload_max_norm must be positive")
    work = value.float()
    norm = torch.linalg.vector_norm(work)
    scale = torch.clamp(
        torch.as_tensor(max_norm, device=work.device, dtype=work.dtype) / norm.clamp_min(1.0e-12),
        max=1.0,
    )
    return value * scale.to(dtype=value.dtype)


class FederatedClient:
    """Client metadata, local data views, and local adaptation only."""

    def __init__(
        self,
        *,
        state: ClientState,
        data: ClientData,
        backbone: SharedBackbone,
        text_batch: TextBatchConfig,
    ) -> None:
        if state.client_id != data.client_id:
            raise ValueError("client state and data have different client IDs")
        if state.task_id != data.task.task_id:
            raise ValueError("client state and data have different task IDs")
        self.state = state
        self.data = data
        self.backbone = backbone
        self.text_batch = text_batch

    def _tokenize_batches(
        self,
        examples: Sequence[TextExample],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> tuple[BackboneBatch, ...]:
        batches = _batched_examples(
            examples,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return tuple(
            self.backbone.tokenize(
                [example.input_text for example in batch],
                [example.target_text for example in batch],
                max_source_length=self.text_batch.max_source_length,
                max_target_length=self.text_batch.max_target_length,
            )
            for batch in batches
        )

    def run_round(
        self,
        *,
        group_id: str,
        initial_coordinates: Tensor,
        subspace: PromptSubspace,
        config: InnerLoopConfig,
        round_seed: int,
        return_residual: bool,
        residual_upload_max_norm: float | None = None,
    ) -> ClientRoundResult:
        self.state.assign_group(group_id)
        support_examples = self.data.support
        query_examples = self.data.query
        support_batches = self._tokenize_batches(
            support_examples,
            batch_size=config.support_batch_size,
            shuffle=True,
            seed=round_seed,
        )
        query_batches = self._tokenize_batches(
            query_examples,
            batch_size=config.query_batch_size,
            shuffle=False,
            seed=round_seed,
        )

        def support_loss(prompt: Tensor, step: int) -> Tensor:
            batch = support_batches[step % len(support_batches)]
            return self.backbone(batch, prompt_embeddings=prompt).loss

        monitor_batch = support_batches[0]

        def support_monitor_loss(prompt: Tensor) -> Tensor:
            return self.backbone(monitor_batch, prompt_embeddings=prompt).loss

        def query_loss(prompt: Tensor) -> Tensor:
            weighted_losses: list[Tensor] = []
            total_examples = 0
            for batch in query_batches:
                batch_examples = int(batch.input_ids.shape[0])
                weighted_losses.append(
                    self.backbone(batch, prompt_embeddings=prompt).loss * batch_examples
                )
                total_examples += batch_examples
            return torch.stack(weighted_losses).sum() / total_examples

        adaptation = adapt_and_compute_feedback(
            support_loss=support_loss,
            support_monitor_loss=support_monitor_loss,
            query_loss=query_loss,
            subspace=subspace,
            initial_coordinates=initial_coordinates,
            initial_residual=None
            if config.reset_residual_each_round
            else self.state.deployment_residual,
            config=config,
        )
        residual_energy = adaptation.terminal_residual.float().square().sum().item()
        prompt_energy = (
            subspace(
                adaptation.terminal_coordinates,
                adaptation.terminal_residual,
            )
            .float()
            .square()
            .sum()
            .item()
        )
        if not config.reset_residual_each_round:
            self.state.deployment_residual = adaptation.terminal_residual.detach().cpu()
        uploaded_residual = (
            _clip_for_upload(
                adaptation.terminal_residual,
                residual_upload_max_norm,
            )
            if return_residual
            else None
        )
        result = ClientRoundResult(
            client_id=self.state.client_id,
            group_id=group_id,
            meta_gradient=config.meta_gradient,
            coordinate_feedback=adaptation.coordinate_feedback,
            initial_coordinates=adaptation.initial_coordinates,
            terminal_coordinates=adaptation.terminal_coordinates,
            terminal_residual=uploaded_residual,
            residual_energy=residual_energy,
            mean_support_loss=adaptation.mean_support_loss,
            query_loss=adaptation.query_loss,
            support_examples=len(support_examples),
            query_examples=len(query_examples),
            prompt_energy=prompt_energy,
            second_order_diagnostics=adaptation.second_order_diagnostics,
            support_losses=adaptation.support_losses,
            support_monitor_losses=adaptation.support_monitor_losses,
        )
        self.state.record_metrics(
            support_loss=result.mean_support_loss,
            query_loss=result.query_loss,
            residual_energy=result.residual_energy,
        )
        return result

    def evaluate_loss(
        self,
        *,
        group_id: str,
        initial_coordinates: Tensor,
        subspace: PromptSubspace,
        config: InnerLoopConfig,
        adaptation_steps: int,
        seed: int,
        split: DataSplit = DataSplit.TEST,
    ) -> ClientEvaluationResult:
        """Adapt on support data and evaluate one held-out split."""

        if adaptation_steps < 0:
            raise ValueError("adaptation_steps must be non-negative")
        support_examples = self.data.support
        evaluation_examples = self.data.split(split)
        support_batches = self._tokenize_batches(
            support_examples,
            batch_size=config.support_batch_size,
            shuffle=True,
            seed=seed,
        )
        evaluation_batches = self._tokenize_batches(
            evaluation_examples,
            batch_size=config.query_batch_size,
            shuffle=False,
            seed=seed,
        )

        def support_loss(prompt: Tensor, step: int) -> Tensor:
            batch = support_batches[step % len(support_batches)]
            return self.backbone(batch, prompt_embeddings=prompt).loss

        def evaluation_loss(prompt: Tensor) -> Tensor:
            weighted_losses: list[Tensor] = []
            total_examples = 0
            for batch in evaluation_batches:
                batch_examples = int(batch.input_ids.shape[0])
                weighted_losses.append(
                    self.backbone(batch, prompt_embeddings=prompt).loss * batch_examples
                )
                total_examples += batch_examples
            return torch.stack(weighted_losses).sum() / total_examples

        if adaptation_steps == 0:
            residual = torch.zeros_like(subspace.center)
            final_prompt = subspace(initial_coordinates.detach(), residual)
            value = evaluation_loss(final_prompt)
            mean_support_loss = None
            residual_energy = 0.0
        else:
            evaluation_config = replace(
                config,
                steps=adaptation_steps,
                reset_residual_each_round=True,
                meta_gradient=MetaGradientMode.FIRST_ORDER,
            )
            adaptation = adapt_and_compute_feedback(
                support_loss=support_loss,
                query_loss=evaluation_loss,
                subspace=subspace,
                initial_coordinates=initial_coordinates,
                initial_residual=None,
                config=evaluation_config,
            )
            final_prompt = subspace(
                adaptation.terminal_coordinates,
                adaptation.terminal_residual,
            )
            value = final_prompt.new_tensor(adaptation.query_loss)
            mean_support_loss = adaptation.mean_support_loss
            residual_energy = float(adaptation.terminal_residual.float().square().sum())

        predictions: list[str] = []
        with torch.no_grad():
            for batch in evaluation_batches:
                generated = self.backbone.generate(
                    batch,
                    prompt_embeddings=final_prompt.detach(),
                    max_new_tokens=self.text_batch.max_target_length,
                )
                predictions.extend(
                    self.backbone.tokenizer.batch_decode(
                        generated,
                        skip_special_tokens=True,
                    )
                )
        metric = compute_task_metric(
            self.data.task.metric,
            predictions,
            [example.target_text for example in evaluation_examples],
        )
        return ClientEvaluationResult(
            client_id=self.state.client_id,
            group_id=group_id,
            adaptation_steps=adaptation_steps,
            mean_support_loss=mean_support_loss,
            test_loss=float(value.detach()),
            residual_energy=residual_energy,
            support_examples=len(support_examples),
            test_examples=len(evaluation_examples),
            split=split,
            metric_name=metric.name,
            metric_value=metric.value,
        )
