from pathlib import Path

import pytest

from data.partition import SplitRatios, split_client_indices
from data.registry import ClassificationAdapter, SequenceLabelingAdapter, build_default_registry
from data.schema import DataSplit, TaskDescription, TaskSpec, TaskType
from metrics.task_metrics import compute_task_metric, span_micro_f1
from scripts.train import _prepare_output_dir


def test_four_way_client_split_is_disjoint_and_complete() -> None:
    source = tuple(range(40))
    support, query, validation, test = split_client_indices(
        source,
        SplitRatios(
            support=0.5,
            query=0.2,
            validation=0.1,
            test=0.2,
        ),
        seed=42,
    )

    splits = {
        DataSplit.SUPPORT: set(support),
        DataSplit.QUERY: set(query),
        DataSplit.VALIDATION: set(validation),
        DataSplit.TEST: set(test),
    }
    assert tuple(map(len, splits.values())) == (20, 8, 4, 8)
    assert set().union(*splits.values()) == set(source)
    assert sum(len(values) for values in splits.values()) == len(source)


def test_task_metrics_cover_all_configured_metric_families() -> None:
    assert compute_task_metric("accuracy", ["positive"], ["positive"]).value == 1.0
    assert compute_task_metric("squad_f1", ["the blue car"], ["blue car"]).value == 1.0
    assert (
        compute_task_metric(
            "rouge_l",
            ["red blue green"],
            ["red blue"],
        ).value
        == 0.8
    )
    assert (
        compute_task_metric(
            "seqeval_f1",
            ["PER: Alice; LOC: Paris"],
            ["PER: Alice; LOC: Paris"],
        ).value
        == 1.0
    )


def _task_spec(task_type: TaskType, **overrides: object) -> TaskSpec:
    description = TaskDescription(
        op="test_op",
        input_object="input",
        output_format="label",
        domain="news",
        language="en",
    )
    values = {
        "task_id": "task",
        "dataset_path": "dummy",
        "dataset_name": None,
        "task_type": task_type,
        "input_fields": ("sentence",),
        "target_field": "label",
        "metric": "accuracy",
        "description": description,
        **overrides,
    }
    return TaskSpec(**values)  # type: ignore[arg-type]


def test_sequence_labeling_gold_spans_score_one_but_freeform_scores_zero() -> None:
    spec = _task_spec(
        TaskType.SEQUENCE_LABELING,
        input_fields=("tokens",),
        target_field="ner_tags",
        metric="seqeval_f1",
    )
    adapter = SequenceLabelingAdapter()
    labels = ["O", "B-ORG", "B-MISC", "B-PER", "I-PER", "B-LOC"]
    gold = adapter.format_target(
        {"tokens": ["EU", "rejects", "German", "call"], "ner_tags": [1, 0, 2, 0]},
        spec,
        labels,
    )
    assert gold == "ORG: EU ; MISC: German"
    assert compute_task_metric("seqeval_f1", [gold], [gold]).value == 1.0
    assert span_micro_f1(["EU is an organization"], [gold]) == 0.0
    assert span_micro_f1(["none"], [gold]) == 0.0


def test_nli_and_boolean_qa_use_short_class_labels() -> None:
    registry = build_default_registry()
    nli_spec = _task_spec(
        TaskType.NATURAL_LANGUAGE_INFERENCE,
        input_fields=("sentence1", "sentence2"),
    )
    bool_spec = _task_spec(
        TaskType.BOOLEAN_QA,
        input_fields=("question", "passage"),
        target_field="answer",
    )
    nli_adapter = registry.get(TaskType.NATURAL_LANGUAGE_INFERENCE)
    bool_adapter = registry.get(TaskType.BOOLEAN_QA)
    assert isinstance(nli_adapter, ClassificationAdapter)
    assert nli_adapter.format_target(
        {"sentence1": "a", "sentence2": "b", "label": 0},
        nli_spec,
        ("entailment", "not_entailment"),
    ) == "entailment"
    assert (
        bool_adapter.format_target(
            {"question": "q", "passage": "p", "answer": True},
            bool_spec,
            None,
        )
        == "yes"
    )
    assert (
        bool_adapter.format_target(
            {"question": "q", "passage": "p", "answer": False},
            bool_spec,
            None,
        )
        == "no"
    )
    assert compute_task_metric("accuracy", ["entailment", "yes"], ["entailment", "yes"]).value == 1.0


def test_new_run_refuses_to_append_existing_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "rounds.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to append"):
        _prepare_output_dir(output_dir, resume_from=None)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "round_000001.pt"
    checkpoint.touch()
    _prepare_output_dir(output_dir, resume_from=checkpoint)
