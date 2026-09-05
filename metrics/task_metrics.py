"""Dependency-light text metrics for the prototype task families."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskMetricResult:
    name: str
    value: float
    num_examples: int


_NLI_POSITIVE = frozenset(
    {
        "entailment",
        "yes",
        "true",
        "entails",
        "text entailment",
        "textual entailment",
    }
)
_NLI_NEGATIVE = frozenset(
    {
        "not entailment",
        "no",
        "false",
        "contradiction",
        "neutral",
        "no entailment",
        "not",
    }
)
_BOOL_POSITIVE = frozenset({"yes", "true", "entailment", "positive", "pos", "acceptable"})
_BOOL_NEGATIVE = frozenset({"no", "false", "not entailment", "not", "negative", "neg", "unacceptable"})
_SENTIMENT_POSITIVE = frozenset({"positive", "pos", "yes"})
_SENTIMENT_NEGATIVE = frozenset({"negative", "neg", "no"})
_TOPIC_LABELS = {
    "world": "world",
    "sports": "sports",
    "sport": "sports",
    "business": "business",
    "tech": "tech",
    "sci tech": "tech",
    "science tech": "tech",
    "science": "tech",
    "sci": "tech",
}


def normalize_answer(text: str) -> str:
    text = str(text).casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _surface_label(text: str) -> str:
    first_line = str(text).splitlines()[0] if str(text) else ""
    return " ".join(first_line.casefold().replace("_", " ").replace("-", " ").split())


def _nli_class(label: str) -> str | None:
    if (
        "not entailment" in label
        or label.startswith("not ")
        or label in _NLI_NEGATIVE
    ):
        return "not_entailment"
    if label in _NLI_POSITIVE or label.endswith(" entailment"):
        return "entailment"
    return None


def _bool_class(label: str) -> str | None:
    if label in _BOOL_NEGATIVE:
        return "no"
    if label in _BOOL_POSITIVE:
        return "yes"
    return None


def _sentiment_class(label: str) -> str | None:
    if label in _SENTIMENT_NEGATIVE:
        return "negative"
    if label in _SENTIMENT_POSITIVE:
        return "positive"
    return None


def _topic_class(label: str) -> str | None:
    compact = " ".join(label.replace("/", " ").split())
    return _TOPIC_LABELS.get(compact)


def canonicalize_class_label(text: str, *, reference: str | None = None) -> str:
    """Map noisy classifier generations onto the gold verbalizer family."""

    surface = _surface_label(text)
    expected = _surface_label(reference) if reference is not None else surface
    if expected in {"entailment", "not entailment"}:
        mapped = _nli_class(surface)
        if mapped is not None:
            return mapped
    elif expected in {"yes", "no"}:
        mapped = _bool_class(surface)
        if mapped is not None:
            return mapped
    elif expected in {"positive", "negative"}:
        mapped = _sentiment_class(surface)
        if mapped is not None:
            return mapped
    elif expected in {"world", "sports", "business", "tech"}:
        mapped = _topic_class(surface)
        if mapped is not None:
            return mapped
    return surface


def label_match(prediction: str, reference: str) -> float:
    predicted = canonicalize_class_label(prediction, reference=reference)
    expected = canonicalize_class_label(reference, reference=reference)
    return float(predicted == expected)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not predicted and not expected:
        return 1.0
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = _lcs_length(predicted, expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _parse_spans(text: str) -> set[tuple[str, str]]:
    spans: set[tuple[str, str]] = set()
    if normalize_answer(text) == "none":
        return spans
    for item in str(text).split(";"):
        label, separator, value = item.partition(":")
        if separator and label.strip() and value.strip():
            spans.add((label.strip().casefold(), normalize_answer(value)))
    return spans


def span_micro_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    true_positive = 0
    predicted_total = 0
    reference_total = 0
    for prediction, reference in zip(predictions, references, strict=True):
        predicted = _parse_spans(prediction)
        expected = _parse_spans(reference)
        true_positive += len(predicted & expected)
        predicted_total += len(predicted)
        reference_total += len(expected)
    if predicted_total == 0 and reference_total == 0:
        return 1.0
    if true_positive == 0:
        return 0.0
    precision = true_positive / predicted_total
    recall = true_positive / reference_total
    return 2.0 * precision * recall / (precision + recall)


def _mean_pairwise(
    function: Callable[[str, str], float],
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    return sum(
        function(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(predictions)


def compute_task_metric(
    metric: str,
    predictions: Sequence[str],
    references: Sequence[str],
) -> TaskMetricResult:
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions and references must be non-empty and equally sized")
    metric = str(metric).lower()
    if metric == "accuracy":
        value = _mean_pairwise(label_match, predictions, references)
    elif metric == "exact_match":
        value = _mean_pairwise(exact_match, predictions, references)
    elif metric in {"squad_f1", "token_f1"}:
        value = _mean_pairwise(token_f1, predictions, references)
    elif metric in {"rouge_l", "rougel"}:
        value = _mean_pairwise(rouge_l_f1, predictions, references)
    elif metric in {"seqeval_f1", "span_f1"}:
        value = span_micro_f1(predictions, references)
    else:
        raise ValueError(f"unsupported task metric: {metric!r}")
    return TaskMetricResult(name=metric, value=float(value), num_examples=len(predictions))
