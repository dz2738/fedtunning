"""Normalize heterogeneous raw datasets into text-to-text examples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from data.registry import TaskAdapterRegistry, build_default_registry
from data.schema import TaskSpec, TextExample


def infer_label_names(dataset: Any, target_field: str) -> tuple[str, ...] | None:
    """Best-effort extraction of Hugging Face ClassLabel names.

    The implementation uses duck typing so the schema layer stays independent
    of a concrete datasets version.
    """

    features = getattr(dataset, "features", None)
    if features is None or target_field not in features:
        return None
    feature = features[target_field]
    while hasattr(feature, "feature"):
        feature = feature.feature
    names = getattr(feature, "names", None)
    if names is None:
        return None
    return tuple(str(name) for name in names)


def format_task_prefix(spec: TaskSpec) -> str:
    return f"task: {spec.description.op}"


def normalize_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    spec: TaskSpec,
    registry: TaskAdapterRegistry,
    label_names: Sequence[str] | None = None,
    include_task_prefix: bool = True,
) -> TextExample:
    adapter = registry.get(spec.task_type)
    input_text = adapter.format_input(row, spec)
    if include_task_prefix:
        input_text = f"{format_task_prefix(spec)} | {input_text}"
    target_text = adapter.format_target(row, spec, label_names)
    stratification_key = adapter.stratification_key(row, spec, label_names)
    raw_id = row.get("id", row.get("idx", row_index))
    return TextExample(
        example_id=f"{spec.task_id}:{raw_id}",
        task_id=spec.task_id,
        input_text=input_text,
        target_text=target_text,
        stratification_key=stratification_key,
        metadata={"raw_id": raw_id, "row_index": row_index},
    )


def preprocess_rows(
    rows: Iterable[Mapping[str, Any]],
    spec: TaskSpec,
    *,
    registry: TaskAdapterRegistry | None = None,
    label_names: Sequence[str] | None = None,
    include_task_prefix: bool = True,
    max_examples: int | None = None,
    row_index_offset: int = 0,
) -> tuple[TextExample, ...]:
    """Materialize a bounded collection of normalized examples."""

    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    if row_index_offset < 0:
        raise ValueError("row_index_offset must be non-negative")
    registry = registry or build_default_registry()
    examples: list[TextExample] = []
    for row_index, row in enumerate(rows):
        if max_examples is not None and row_index >= max_examples:
            break
        examples.append(
            normalize_row(
                row,
                row_index=row_index + row_index_offset,
                spec=spec,
                registry=registry,
                label_names=label_names,
                include_task_prefix=include_task_prefix,
            )
        )
    if not examples:
        raise ValueError(f"task {spec.task_id!r} produced no examples")
    return tuple(examples)

