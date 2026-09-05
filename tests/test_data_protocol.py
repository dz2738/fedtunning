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
    assert compute_task_metric("accuracy", ["yes"], ["entailment"]).value == 1.0
    assert compute_task_metric("accuracy", ["_entailment"], ["entailment"]).value == 1.0
    assert compute_task_metric("accuracy", ["text_entailment"], ["entailment"]).value == 1.0
    assert compute_task_metric("accuracy", ["no"], ["not_entailment"]).value == 1.0
    assert compute_task_metric("accuracy", ["yes"], ["not_entailment"]).value == 0.0
    assert compute_task_metric("accuracy", ["entailment"], ["yes"]).value == 1.0
    assert compute_task_metric("accuracy", ["not_entailment"], ["no"]).value == 1.0
    assert compute_task_metric("accuracy", ["positive"], ["yes"]).value == 1.0
    assert compute_task_metric("accuracy", ["acceptable"], ["yes"]).value == 1.0
    assert compute_task_metric("accuracy", ["negative"], ["no"]).value == 1.0
    assert compute_task_metric("accuracy", ["pos"], ["positive"]).value == 1.0
    assert compute_task_metric("accuracy", ["not_contextual_entailment"], ["not_entailment"]).value == 1.0
    assert compute_task_metric("accuracy", [""], ["entailment"]).value == 0.0
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
    ) == "yes"
    assert nli_adapter.format_target(
        {"sentence1": "a", "sentence2": "b", "label": 1},
        nli_spec,
        ("entailment", "not_entailment"),
    ) == "no"
    assert nli_adapter.format_input(
        {"sentence1": "a", "sentence2": "b", "label": 0},
        nli_spec,
    ) == "Answer yes or no. sentence1: a | sentence2: b"
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
    assert bool_adapter.format_input(
        {"question": "q", "passage": "p", "answer": True},
        bool_spec,
    ) == "Answer yes or no. question: q | passage: p"
    assert compute_task_metric("accuracy", ["yes", "no"], ["yes", "no"]).value == 1.0


def test_sentiment_and_cola_use_yes_no_verbalizer() -> None:
    adapter = ClassificationAdapter()
    sst_spec = _task_spec(
        TaskType.CLASSIFICATION,
        description=TaskDescription(
            op="sentiment_classification",
            input_object="sentence",
            output_format="categorical_label",
            domain="movie_review",
            language="en",
        ),
    )
    cola_spec = _task_spec(
        TaskType.CLASSIFICATION,
        description=TaskDescription(
            op="grammaticality_classification",
            input_object="sentence",
            output_format="categorical_label",
            domain="linguistics",
            language="en",
        ),
    )
    ag_spec = _task_spec(
        TaskType.CLASSIFICATION,
        description=TaskDescription(
            op="topic_classification",
            input_object="news_article",
            output_format="categorical_label",
            domain="news",
            language="en",
        ),
    )
    assert adapter.format_target({"sentence": "great", "label": 1}, sst_spec, ("negative", "positive")) == "yes"
    assert adapter.format_target({"sentence": "bad", "label": 0}, sst_spec, ("negative", "positive")) == "no"
    assert adapter.format_input({"sentence": "great", "label": 1}, sst_spec) == "Answer yes or no. sentence: great"
    assert adapter.format_target({"sentence": "ok", "label": 0}, cola_spec, ("unacceptable", "acceptable")) == "no"
    assert adapter.format_target({"sentence": "ok", "label": 1}, cola_spec, ("unacceptable", "acceptable")) == "yes"
    assert adapter.format_target({"sentence": "news", "label": 2}, ag_spec, ("World", "Sports", "Business", "Sci/Tech")) == "business"
    assert adapter.format_target({"sentence": "news", "label": 3}, ag_spec, ("World", "Sports", "Business", "Sci/Tech")) == "tech"
    assert adapter.format_input({"sentence": "news", "label": 2}, ag_spec) == (
        "Answer with one topic: world, sports, business, or tech. sentence: news"
    )
    assert compute_task_metric("accuracy", ["Sci/Tech", "World"], ["tech", "world"]).value == 1.0


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


def test_task_spec_cycles_client_description_variants() -> None:
    spec = TaskSpec.from_mapping(
        {
            "task_id": "sst2_sentiment",
            "dataset_path": "dummy",
            "dataset_name": None,
            "task_type": "classification",
            "input_fields": ["sentence"],
            "target_field": "label",
            "metric": "accuracy",
            "description": {
                "op": "sentiment_classification",
                "in": "sentence",
                "out": "categorical_label",
                "dom": "movie_review",
                "lang": "en",
            },
            "description_variants": [
                {
                    "op": "sentiment_classification",
                    "in": "sentence",
                    "out": "categorical_label",
                    "dom": "movie_review",
                    "lang": "en",
                },
                {
                    "op": "review_polarity_classification",
                    "in": "review_text",
                    "out": "categorical_label",
                    "dom": "film_review",
                    "lang": "en",
                },
            ],
        }
    )

    first = spec.client_description(0)
    second = spec.client_description(1)
    assert first.op == "sentiment_classification"
    assert second.op == "review_polarity_classification"
    assert spec.client_description(2) == first
    assert first != second
