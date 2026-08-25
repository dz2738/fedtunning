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


def normalize_answer(text: str) -> str:
    text = str(text).casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def label_match(prediction: str, reference: str) -> float:
    predicted = " ".join(str(prediction).casefold().split())
    expected = " ".join(str(reference).casefold().split())
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
