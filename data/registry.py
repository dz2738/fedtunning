"""Task-specific conversion from raw rows to a common text-to-text format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from data.schema import TaskSpec, TaskType


class TaskFormatError(ValueError):
    """Raised when a raw dataset row does not match its task specification."""


def _get_required(row: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in row:
        raise TaskFormatError(f"raw example is missing field {field_name!r}")
    return row[field_name]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(str(item) for item in value).strip()
    return str(value).strip()


class TaskAdapter(ABC):
    """Convert one task family into input, target, and partitioning labels."""

    @abstractmethod
    def format_input(self, row: Mapping[str, Any], spec: TaskSpec) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_target(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stratification_key(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        raise NotImplementedError


_YES_NO_TASK_TYPES = frozenset(
    {
        TaskType.NATURAL_LANGUAGE_INFERENCE,
        TaskType.BOOLEAN_QA,
    }
)
_YES_NO_OPS = frozenset(
    {
        "textual_entailment",
        "boolean_question_answering",
        "grammaticality_classification",
        "sentiment_classification",
    }
)
_YES_NO_VERBALIZER = {
    "entailment": "yes",
    "not_entailment": "no",
    "yes": "yes",
    "no": "no",
    "true": "yes",
    "false": "no",
    "acceptable": "yes",
    "unacceptable": "no",
    "positive": "yes",
    "negative": "no",
    "pos": "yes",
    "neg": "no",
}


def _label_key(text: str) -> str:
    return text.strip().casefold().replace("-", "_").replace(" ", "_").replace("/", "_")


def _yes_no_label(text: str) -> str:
    return _YES_NO_VERBALIZER.get(_label_key(text), text)


_TOPIC_VERBALIZER = {
    "world": "world",
    "sports": "sports",
    "sport": "sports",
    "business": "business",
    "sci_tech": "tech",
    "science_tech": "tech",
    "tech": "tech",
    "science": "tech",
}
_TOPIC_INSTRUCTION = "Answer with one topic: world, sports, business, or tech."


def _verbalize_class_label(text: str) -> str:
    mapped = _YES_NO_VERBALIZER.get(_label_key(text))
    if mapped is not None:
        return mapped
    mapped = _TOPIC_VERBALIZER.get(_label_key(text))
    if mapped is not None:
        return mapped
    return text


class ClassificationAdapter(TaskAdapter):
    def format_input(self, row: Mapping[str, Any], spec: TaskSpec) -> str:
        parts = [f"{name}: {_as_text(_get_required(row, name))}" for name in spec.input_fields]
        body = " | ".join(parts)
        if spec.task_type in _YES_NO_TASK_TYPES or spec.description.op in _YES_NO_OPS:
            return f"Answer yes or no. {body}"
        if spec.description.op == "topic_classification":
            return f"{_TOPIC_INSTRUCTION} {body}"
        return body

    def format_target(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        label = _get_required(row, spec.target_field)
        if isinstance(label, bool) or type(label).__name__ == "bool_":
            text = "yes" if bool(label) else "no"
        elif isinstance(label, int) and label_names is not None:
            if label < 0 or label >= len(label_names):
                raise TaskFormatError(f"label index {label} is outside label_names")
            text = str(label_names[label])
        else:
            text = _as_text(label)
        return _verbalize_class_label(text)

    def stratification_key(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        return self.format_target(row, spec, label_names)


class SequenceLabelingAdapter(TaskAdapter):
    def format_input(self, row: Mapping[str, Any], spec: TaskSpec) -> str:
        if len(spec.input_fields) != 1:
            raise TaskFormatError("sequence labeling expects exactly one token field")
        tokens = _get_required(row, spec.input_fields[0])
        if not isinstance(tokens, Sequence) or isinstance(tokens, str):
            raise TaskFormatError("sequence labeling input must be a sequence of tokens")
        return "tokens: " + " ".join(str(token) for token in tokens)

    def format_target(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        tokens = list(_get_required(row, spec.input_fields[0]))
        raw_tags = list(_get_required(row, spec.target_field))
        if len(tokens) != len(raw_tags):
            raise TaskFormatError("token and tag sequences have different lengths")
        tags = [self._tag_name(tag, label_names) for tag in raw_tags]
        spans = self._bio_to_spans(tokens, tags)
        return "none" if not spans else " ; ".join(f"{kind}: {text}" for kind, text in spans)

    def stratification_key(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        raw_tags = list(_get_required(row, spec.target_field))
        tags = [self._tag_name(tag, label_names) for tag in raw_tags]
        entity_types = sorted({tag.split("-", 1)[-1] for tag in tags if tag != "O"})
        return "+".join(entity_types) if entity_types else "no_entity"

    @staticmethod
    def _tag_name(tag: Any, label_names: Sequence[str] | None) -> str:
        if isinstance(tag, int) and label_names is not None:
            if tag < 0 or tag >= len(label_names):
                raise TaskFormatError(f"tag index {tag} is outside label_names")
            return str(label_names[tag])
        return str(tag)

    @staticmethod
    def _bio_to_spans(tokens: Sequence[Any], tags: Sequence[str]) -> list[tuple[str, str]]:
        spans: list[tuple[str, str]] = []
        current_type: str | None = None
        current_tokens: list[str] = []

        def flush() -> None:
            nonlocal current_type, current_tokens
            if current_type is not None:
                spans.append((current_type, " ".join(current_tokens)))
            current_type = None
            current_tokens = []

        for token, tag in zip(tokens, tags, strict=True):
            if tag == "O":
                flush()
                continue
            prefix, separator, entity_type = tag.partition("-")
            if not separator:
                prefix, entity_type = "B", tag
            if prefix == "B" or entity_type != current_type:
                flush()
                current_type = entity_type
            current_tokens.append(str(token))
        flush()
        return spans


class QuestionAnsweringAdapter(TaskAdapter):
    def format_input(self, row: Mapping[str, Any], spec: TaskSpec) -> str:
        values = {name: _as_text(_get_required(row, name)) for name in spec.input_fields}
        if "question" in values and "context" in values:
            return f"question: {values['question']} | context: {values['context']}"
        return " | ".join(f"{name}: {value}" for name, value in values.items())

    def format_target(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        answers = _get_required(row, spec.target_field)
        if isinstance(answers, Mapping):
            texts = answers.get("text", ())
            if texts:
                return _as_text(texts[0])
        if isinstance(answers, Sequence) and not isinstance(answers, str) and answers:
            return _as_text(answers[0])
        text = _as_text(answers)
        return text if text else "unanswerable"

    def stratification_key(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        length = len(self.format_target(row, spec, label_names).split())
        if length <= 2:
            return "answer_len_0_2"
        if length <= 6:
            return "answer_len_3_6"
        return "answer_len_7_plus"


class SummarizationAdapter(TaskAdapter):
    def format_input(self, row: Mapping[str, Any], spec: TaskSpec) -> str:
        return "document: " + " ".join(
            _as_text(_get_required(row, field_name)) for field_name in spec.input_fields
        )

    def format_target(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        return _as_text(_get_required(row, spec.target_field))

    def stratification_key(
        self,
        row: Mapping[str, Any],
        spec: TaskSpec,
        label_names: Sequence[str] | None,
    ) -> str:
        length = len(self.format_target(row, spec, label_names).split())
        if length <= 20:
            return "summary_len_short"
        if length <= 50:
            return "summary_len_medium"
        return "summary_len_long"


class TaskAdapterRegistry:
    """Explicit registry to keep dataset logic out of client/server modules."""

    def __init__(self) -> None:
        self._adapters: dict[TaskType, TaskAdapter] = {}

    def register(self, task_type: TaskType | str, adapter: TaskAdapter) -> None:
        key = TaskType(task_type)
        if key in self._adapters:
            raise KeyError(f"adapter already registered for {key.value}")
        self._adapters[key] = adapter

    def get(self, task_type: TaskType | str) -> TaskAdapter:
        key = TaskType(task_type)
        try:
            return self._adapters[key]
        except KeyError as error:
            raise KeyError(f"no adapter registered for {key.value}") from error


def build_default_registry() -> TaskAdapterRegistry:
    registry = TaskAdapterRegistry()
    classification = ClassificationAdapter()
    registry.register(TaskType.CLASSIFICATION, classification)
    registry.register(TaskType.NATURAL_LANGUAGE_INFERENCE, classification)
    registry.register(TaskType.BOOLEAN_QA, classification)
    registry.register(TaskType.SEQUENCE_LABELING, SequenceLabelingAdapter())
    registry.register(TaskType.QUESTION_ANSWERING, QuestionAnsweringAdapter())
    registry.register(TaskType.SUMMARIZATION, SummarizationAdapter())
    return registry

