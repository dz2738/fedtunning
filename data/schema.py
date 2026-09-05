"""Typed contracts shared by data loading, clients, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    """Task families supported by the first prototype."""

    CLASSIFICATION = "classification"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    BOOLEAN_QA = "boolean_qa"
    SEQUENCE_LABELING = "sequence_labeling"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"


class DataSplit(StrEnum):
    SUPPORT = "support"
    QUERY = "query"
    VALIDATION = "validation"
    TEST = "test"


DESCRIPTION_FIELDS = ("op", "in", "out", "dom", "lang")


def _require_non_empty(name: str, value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class TaskDescription:
    """Structured task record used by semantic grouping."""

    op: str
    input_object: str
    output_format: str
    domain: str
    language: str

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            _require_non_empty(name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskDescription:
        missing = [name for name in DESCRIPTION_FIELDS if name not in value]
        if missing:
            raise ValueError(f"task description is missing fields: {missing}")
        return cls(
            op=str(value["op"]),
            input_object=str(value["in"]),
            output_format=str(value["out"]),
            domain=str(value["dom"]),
            language=str(value["lang"]),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "op": self.op,
            "in": self.input_object,
            "out": self.output_format,
            "dom": self.domain,
            "lang": self.language,
        }

    def serialize(self) -> str:
        """Return a deterministic representation for the task encoder."""

        values = self.as_dict()
        return " | ".join(f"{field_name}: {values[field_name]}" for field_name in DESCRIPTION_FIELDS)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Configuration-independent description of one source dataset/task."""

    task_id: str
    dataset_path: str
    dataset_name: str | None
    task_type: TaskType
    input_fields: tuple[str, ...]
    target_field: str
    metric: str
    description: TaskDescription
    description_variants: tuple[TaskDescription, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("task_id", self.task_id)
        _require_non_empty("dataset_path", self.dataset_path)
        _require_non_empty("target_field", self.target_field)
        _require_non_empty("metric", self.metric)
        if not self.input_fields:
            raise ValueError("input_fields must contain at least one field")
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError(f"input_fields contains duplicates: {self.input_fields}")

    def client_description(self, client_index: int) -> TaskDescription:
        """Return this client's wording; variants cycle if there are fewer than N clients."""

        if client_index < 0:
            raise ValueError("client_index must be non-negative")
        variants = self.description_variants or (self.description,)
        return variants[client_index % len(variants)]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskSpec:
        description = TaskDescription.from_mapping(value["description"])
        raw_variants = value.get("description_variants") or ()
        variants = tuple(TaskDescription.from_mapping(item) for item in raw_variants)
        return cls(
            task_id=str(value["task_id"]),
            dataset_path=str(value["dataset_path"]),
            dataset_name=None
            if value.get("dataset_name") is None
            else str(value["dataset_name"]),
            task_type=TaskType(str(value["task_type"])),
            input_fields=tuple(str(item) for item in value["input_fields"]),
            target_field=str(value["target_field"]),
            metric=str(value["metric"]),
            description=description,
            description_variants=variants,
        )


@dataclass(frozen=True, slots=True)
class TextExample:
    """Normalized text-to-text example shared by every downstream module."""

    example_id: str
    task_id: str
    input_text: str
    target_text: str
    stratification_key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("example_id", self.example_id)
        _require_non_empty("task_id", self.task_id)
        _require_non_empty("input_text", self.input_text)
        _require_non_empty("target_text", self.target_text)
        _require_non_empty("stratification_key", self.stratification_key)


@dataclass(frozen=True, slots=True)
class ClientPartition:
    """Indices into one shared task-level example collection."""

    client_id: str
    task_id: str
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_non_empty("client_id", self.client_id)
        _require_non_empty("task_id", self.task_id)
        split_sets = {
            DataSplit.SUPPORT: set(self.support_indices),
            DataSplit.QUERY: set(self.query_indices),
            DataSplit.VALIDATION: set(self.validation_indices),
            DataSplit.TEST: set(self.test_indices),
        }
        for split_name, indices in split_sets.items():
            if len(indices) != len(self.indices(split_name)):
                raise ValueError(f"{split_name} contains duplicate indices")
        splits = tuple(split_sets)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                if split_sets[left] & split_sets[right]:
                    raise ValueError(f"{left} and {right} indices overlap")

    def indices(self, split: DataSplit | str) -> tuple[int, ...]:
        split = DataSplit(split)
        if split is DataSplit.SUPPORT:
            return self.support_indices
        if split is DataSplit.QUERY:
            return self.query_indices
        if split is DataSplit.VALIDATION:
            return self.validation_indices
        return self.test_indices

    @property
    def num_examples(self) -> int:
        return (
            len(self.support_indices)
            + len(self.query_indices)
            + len(self.validation_indices)
            + len(self.test_indices)
        )


def validate_unique_task_ids(specs: Sequence[TaskSpec]) -> None:
    task_ids = [spec.task_id for spec in specs]
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate task_id values: {duplicates}")

