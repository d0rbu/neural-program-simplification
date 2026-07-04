from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from beartype import beartype
from phantom import Phantom

from neural_program_simplification.task_datasets import (
    ExampleId,
    TokenizedTaskDataset,
)


def _is_non_empty(value: str) -> bool:
    return bool(value.strip())


class ModelId(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """Non-empty model or checkpoint identifier."""


@beartype
def parse_model_id(value: str) -> ModelId:
    return ModelId.parse(value)


@runtime_checkable
class NextTokenModel(Protocol):
    """Minimal causal-LM surface needed to score task-answer tokens."""

    @property
    def model_id(self) -> ModelId: ...  # pragma: no cover

    def next_token_logits(  # pragma: no cover
        self,
        token_id_batches: Sequence[Sequence[int]],
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TaskPrediction:
    """Per-example next-token result for the held-out final task token."""

    example_id: ExampleId
    answer_token_id: int
    predicted_token_id: int
    answer_logit: float
    predicted_logit: float
    answer_logprob: float
    answer_rank: int

    @property
    def is_correct(self) -> bool:
        return self.predicted_token_id == self.answer_token_id


@dataclass(frozen=True, slots=True)
class TaskRun:
    """Result of scoring a tokenized task dataset with one model."""

    dataset_id: str
    model_id: ModelId
    predictions: Sequence[TaskPrediction]

    def __post_init__(self) -> None:
        predictions = tuple(self.predictions)
        if not predictions:
            raise ValueError("task run must contain at least one prediction")

        object.__setattr__(self, "model_id", parse_model_id(str(self.model_id)))
        object.__setattr__(self, "predictions", predictions)

    @property
    def top1_accuracy(self) -> float:
        correct = sum(prediction.is_correct for prediction in self.predictions)
        return correct / len(self.predictions)


def _coerce_logits(logits: np.ndarray, expected_rows: int) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.ndim != 2:
        raise ValueError("next-token logits must have shape batch x vocab")
    if logits_array.shape[0] != expected_rows:
        raise ValueError("next-token logits batch size must match the dataset")
    if logits_array.shape[1] == 0:
        raise ValueError("next-token logits must include at least one vocabulary column")
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("next-token logits must be finite")
    return logits_array


def _answer_logprob(row: np.ndarray, answer_token_id: int) -> float:
    max_logit = float(np.max(row))
    log_denominator = max_logit + float(np.log(np.sum(np.exp(row - max_logit))))
    return float(row[answer_token_id] - log_denominator)


def _prediction_for_row(
    example_id: ExampleId,
    answer_token_id: int,
    row: np.ndarray,
) -> TaskPrediction:
    if answer_token_id >= row.shape[0]:
        raise ValueError("answer token id is outside the logits vocabulary")

    predicted_token_id = int(np.argmax(row))
    answer_logit = float(row[answer_token_id])
    return TaskPrediction(
        example_id=example_id,
        answer_token_id=answer_token_id,
        predicted_token_id=predicted_token_id,
        answer_logit=answer_logit,
        predicted_logit=float(row[predicted_token_id]),
        answer_logprob=_answer_logprob(row, answer_token_id),
        answer_rank=int(np.count_nonzero(row > answer_logit) + 1),
    )


@beartype
def run_task_dataset(
    tokenized_dataset: TokenizedTaskDataset,
    model: NextTokenModel,
) -> TaskRun:
    prefixes = tuple(example.prefix_token_ids for example in tokenized_dataset.examples)
    logits = _coerce_logits(model.next_token_logits(prefixes), len(prefixes))

    predictions = tuple(
        _prediction_for_row(
            example_id=example.example.example_id,
            answer_token_id=example.answer_token_id,
            row=logits[row_index],
        )
        for row_index, example in enumerate(tokenized_dataset.examples)
    )

    return TaskRun(
        dataset_id=str(tokenized_dataset.dataset_id),
        model_id=model.model_id,
        predictions=predictions,
    )
