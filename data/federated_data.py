"""Build a memory-efficient, single-process federated data view."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import (
    ClassLabel,
    load_dataset,
)
from datasets import (
    Sequence as HFSequence,
)
from huggingface_hub import hf_hub_download

from data.partition import SplitRatios, build_client_partitions, partition_indices
from data.preprocess import infer_label_names, preprocess_rows
from data.registry import TaskAdapterRegistry, build_default_registry
from data.schema import (
    ClientPartition,
    DataSplit,
    TaskDescription,
    TaskSpec,
    TextExample,
    validate_unique_task_ids,
)

CONLL2003_LABELS = [
    "O",
    "B-ORG",
    "B-MISC",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-ORG",
    "I-MISC",
    "I-LOC",
]


def _load_conll2003(
    *,
    cache_dir: str | None,
    offline: bool,
):
    """Load CoNLL-2003 from the tner JSON dump.

    Sequence-labeling NER remains supported for optional experiments, but it is
    not part of the default prototype mix.
    """
    cache_root = Path(cache_dir or ".cache/huggingface")
    raw_dir = cache_root / "raw" / "tner_conll2003"

    filenames = {
        "train": "dataset/train.json",
        "validation": "dataset/valid.json",
        "test": "dataset/test.json",
    }

    if offline:
        data_files = {
            split: str(raw_dir / filename)
            for split, filename in filenames.items()
        }

        missing = [
            path
            for path in data_files.values()
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "CoNLL2003 本地原始文件不存在。"
                "请先在联网模式下运行一次 prepare_data.py。"
                f" 缺少文件: {missing}"
            )
    else:
        data_files = {
            split: hf_hub_download(
                repo_id="tner/conll2003",
                repo_type="dataset",
                filename=filename,
                local_dir=raw_dir,
            )
            for split, filename in filenames.items()
        }

    dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
    )

    dataset = dataset.rename_column("tags", "ner_tags")
    dataset = dataset.cast_column(
        "ner_tags",
        HFSequence(
            ClassLabel(names=CONLL2003_LABELS)
        ),
    )

    return dataset

def _to_plain_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise TypeError("config must be a mapping when OmegaConf is unavailable") from error
    converted = OmegaConf.to_container(config, resolve=True)
    if not isinstance(converted, Mapping):
        raise TypeError("data config must resolve to a mapping")
    return converted


class ClientDatasetView(Sequence[TextExample]):
    """Zero-copy index view over one task-level example tuple."""

    def __init__(self, examples: Sequence[TextExample], indices: Sequence[int]) -> None:
        self._examples = examples
        self._indices = tuple(int(index) for index in indices)
        if any(index < 0 or index >= len(examples) for index in self._indices):
            raise IndexError("client view contains an out-of-range example index")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int | slice) -> TextExample | tuple[TextExample, ...]:
        if isinstance(index, slice):
            return tuple(self._examples[item] for item in self._indices[index])
        return self._examples[self._indices[index]]

    def __iter__(self) -> Iterator[TextExample]:
        return (self._examples[index] for index in self._indices)


@dataclass(frozen=True, slots=True)
class ClientData:
    client_id: str
    task: TaskSpec
    partition: ClientPartition
    examples: Sequence[TextExample]
    description: TaskDescription

    def split(self, split: DataSplit | str) -> ClientDatasetView:
        return ClientDatasetView(self.examples, self.partition.indices(split))

    @property
    def support(self) -> ClientDatasetView:
        return self.split(DataSplit.SUPPORT)

    @property
    def query(self) -> ClientDatasetView:
        return self.split(DataSplit.QUERY)

    @property
    def validation(self) -> ClientDatasetView:
        return self.split(DataSplit.VALIDATION)

    @property
    def test(self) -> ClientDatasetView:
        return self.split(DataSplit.TEST)


@dataclass(frozen=True, slots=True)
class FederatedData:
    """All clients reference shared per-task example collections."""

    tasks: Mapping[str, TaskSpec]
    clients: Mapping[str, ClientData]

    def client_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.clients))

    def clients_for_task(self, task_id: str) -> tuple[ClientData, ...]:
        return tuple(
            self.clients[client_id]
            for client_id in self.client_ids()
            if self.clients[client_id].task.task_id == task_id
        )

    def summary(self) -> dict[str, Any]:
        return {
            "num_tasks": len(self.tasks),
            "num_clients": len(self.clients),
            "num_examples": sum(client.partition.num_examples for client in self.clients.values()),
            "clients_per_task": {
                task_id: len(self.clients_for_task(task_id)) for task_id in sorted(self.tasks)
            },
        }


def _load_task_rows(
    spec: TaskSpec,
    *,
    cache_dir: str | None,
    streaming: bool,
    offline: bool,
    max_examples: int,
    row_offset: int,
) -> tuple[Any, Sequence[Mapping[str, Any]]]:
    
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as error:
        raise RuntimeError("install project dependencies before loading datasets") from error

    download_config = DownloadConfig(local_files_only=offline)
    if spec.dataset_path == "tner/conll2003":
        dataset = _load_conll2003(
            cache_dir=cache_dir,
            offline=offline,
        )
    else:
        dataset = load_dataset(
            spec.dataset_path,
            spec.dataset_name,
            split="train",
            cache_dir=cache_dir,
            streaming=streaming,
            download_config=download_config,
        )
    if row_offset < 0:
        raise ValueError("row_offset must be non-negative")
    if streaming:
        rows = tuple(islice(dataset, row_offset, row_offset + max_examples))
    else:
        stop = min(len(dataset), row_offset + max_examples)
        if row_offset >= stop:
            raise ValueError(
                f"row_offset={row_offset} leaves no rows in task {spec.task_id!r}"
            )
        rows = dataset.select(range(row_offset, stop))
    return dataset, rows


def build_federated_data(
    config: Any,
    *,
    seed: int,
    registry: TaskAdapterRegistry | None = None,
) -> FederatedData:
    """Load, normalize, partition, and split all configured tasks."""

    config = _to_plain_mapping(config)
    specs = tuple(TaskSpec.from_mapping(item) for item in config["tasks"])
    validate_unique_task_ids(specs)
    registry = registry or build_default_registry()

    split_config = config["split"]
    ratios = SplitRatios(
        support=float(split_config["support_ratio"]),
        query=float(split_config["query_ratio"]),
        validation=float(split_config["validation_ratio"]),
        test=float(split_config["test_ratio"]),
    )
    partition_config = config["partition"]
    num_clients = int(config["clients_per_dataset"])
    max_examples = sum(
        int(config[key])
        for key in (
            "max_train_examples_per_dataset",
            "max_validation_examples_per_dataset",
            "max_test_examples_per_dataset",
        )
    )
    template_config = config.get("text_template", {})

    tasks: dict[str, TaskSpec] = {}
    clients: dict[str, ClientData] = {}
    for task_number, spec in enumerate(specs):
        raw_dataset, rows = _load_task_rows(
            spec,
            cache_dir=config.get("cache_dir"),
            streaming=bool(config.get("streaming", False)),
            offline=bool(config.get("offline", False)),
            max_examples=max_examples,
            row_offset=int(config.get("row_offset", 0)),
        )
        label_names = infer_label_names(raw_dataset, spec.target_field)
        examples = preprocess_rows(
            rows,
            spec,
            registry=registry,
            label_names=label_names,
            include_task_prefix=bool(template_config.get("include_task_prefix", True)),
            max_examples=max_examples,
            row_index_offset=int(config.get("row_offset", 0)),
        )
        task_seed = seed + task_number * 10_000
        client_indices = partition_indices(
            num_examples=len(examples),
            num_clients=num_clients,
            strategy=str(partition_config["strategy"]),
            seed=task_seed,
            labels=[example.stratification_key for example in examples],
            dirichlet_alpha=float(partition_config.get("dirichlet_alpha", 0.5)),
            min_examples_per_client=int(
                partition_config.get("min_examples_per_client", 1)
            ),
            size_imbalance=float(partition_config.get("size_imbalance", 0.0)),
        )
        client_partitions = build_client_partitions(
            task_id=spec.task_id,
            task_indices_by_client=client_indices,
            ratios=ratios,
            seed=task_seed + 1_000,
        )
        tasks[spec.task_id] = spec
        for client_index, client_partition in enumerate(client_partitions):
            clients[client_partition.client_id] = ClientData(
                client_id=client_partition.client_id,
                task=spec,
                partition=client_partition,
                examples=examples,
                description=spec.client_description(client_index),
            )

    return FederatedData(tasks=tasks, clients=clients)

