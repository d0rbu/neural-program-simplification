from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from neural_program_simplification.model_execution import (
    ModelId,
    TaskRun,
    parse_model_id,
    run_task_dataset,
)
from neural_program_simplification.task_datasets import (
    TaskDataset,
    TaskExample,
    TokenizedTaskDataset,
    TokenizedTaskExample,
    parse_answer_text,
    parse_dataset_id,
    parse_example_id,
    parse_prompt_text,
)


@dataclass(frozen=True)
class StaticNextTokenModel:
    model_id: ModelId
    logits_by_prefix: Mapping[tuple[int, ...], np.ndarray]

    def next_token_logits(self, token_id_batches: Sequence[Sequence[int]]) -> np.ndarray:
        return np.vstack(
            [self.logits_by_prefix[tuple(token_id_batch)] for token_id_batch in token_id_batches]
        )


def _tokenized_dataset() -> TokenizedTaskDataset:
    examples = (
        TaskExample(
            example_id=parse_example_id("mcq-1"),
            prompt=parse_prompt_text("Question: 2+2?\nRESPONSE: B"),
            answer_text=parse_answer_text(" B"),
        ),
        TaskExample(
            example_id=parse_example_id("ioi-1"),
            prompt=parse_prompt_text("Alice gave the book to Bob. The indirect object is Bob"),
            answer_text=parse_answer_text(" Bob"),
        ),
    )
    dataset = TaskDataset(dataset_id=parse_dataset_id("toy-tasks"), examples=examples)

    return TokenizedTaskDataset(
        dataset=dataset,
        examples=(
            TokenizedTaskExample(example=examples[0], prompt_token_ids=(10, 20, 3), answer_token_id=3),
            TokenizedTaskExample(example=examples[1], prompt_token_ids=(11, 22, 4), answer_token_id=4),
        ),
    )


def test_run_task_dataset_scores_held_out_final_task_tokens() -> None:
    tokenized_dataset = _tokenized_dataset()
    model = StaticNextTokenModel(
        model_id=parse_model_id("toy/model"),
        logits_by_prefix={
            (10, 20): np.array([0.0, 1.0, 2.0, 8.0, -1.0], dtype=np.float64),
            (11, 22): np.array([0.0, 7.0, 4.0, 2.0, 3.0], dtype=np.float64),
        },
    )

    task_run = run_task_dataset(tokenized_dataset, model)

    assert task_run.dataset_id == "toy-tasks"
    assert task_run.model_id == parse_model_id("toy/model")
    assert task_run.top1_accuracy == pytest.approx(0.5)
    assert task_run.predictions[0].is_correct
    assert task_run.predictions[0].answer_rank == 1
    assert task_run.predictions[1].predicted_token_id == 1
    assert task_run.predictions[1].answer_rank == 3
    assert task_run.predictions[1].answer_logprob < 0.0


def test_run_task_dataset_rejects_bad_logit_shape() -> None:
    tokenized_dataset = _tokenized_dataset()

    @dataclass(frozen=True)
    class BadShapeModel:
        model_id: ModelId = parse_model_id("toy/bad-shape")

        def next_token_logits(self, token_id_batches: Sequence[Sequence[int]]) -> np.ndarray:
            return np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="batch x vocab"):
        run_task_dataset(tokenized_dataset, BadShapeModel())


def test_run_task_dataset_rejects_logit_batch_size_mismatch() -> None:
    tokenized_dataset = _tokenized_dataset()

    @dataclass(frozen=True)
    class MismatchedBatchModel:
        model_id: ModelId = parse_model_id("toy/mismatched")

        def next_token_logits(self, token_id_batches: Sequence[Sequence[int]]) -> np.ndarray:
            return np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="batch size"):
        run_task_dataset(tokenized_dataset, MismatchedBatchModel())


def test_run_task_dataset_rejects_empty_vocab_logits() -> None:
    tokenized_dataset = _tokenized_dataset()

    @dataclass(frozen=True)
    class EmptyVocabModel:
        model_id: ModelId = parse_model_id("toy/empty-vocab")

        def next_token_logits(self, token_id_batches: Sequence[Sequence[int]]) -> np.ndarray:
            return np.empty((2, 0), dtype=np.float64)

    with pytest.raises(ValueError, match="vocabulary"):
        run_task_dataset(tokenized_dataset, EmptyVocabModel())


def test_run_task_dataset_rejects_answer_outside_vocab() -> None:
    tokenized_dataset = _tokenized_dataset()
    model = StaticNextTokenModel(
        model_id=parse_model_id("toy/small-vocab"),
        logits_by_prefix={
            (10, 20): np.array([0.0, 1.0], dtype=np.float64),
            (11, 22): np.array([0.0, 1.0], dtype=np.float64),
        },
    )

    with pytest.raises(ValueError, match="outside the logits vocabulary"):
        run_task_dataset(tokenized_dataset, model)


def test_run_task_dataset_rejects_non_finite_logits() -> None:
    tokenized_dataset = _tokenized_dataset()
    model = StaticNextTokenModel(
        model_id=parse_model_id("toy/non-finite"),
        logits_by_prefix={
            (10, 20): np.array([0.0, np.nan, 1.0, 2.0, 3.0], dtype=np.float64),
            (11, 22): np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        },
    )

    with pytest.raises(ValueError, match="finite"):
        run_task_dataset(tokenized_dataset, model)


def test_task_run_rejects_empty_predictions() -> None:
    with pytest.raises(ValueError, match="at least one prediction"):
        TaskRun(
            dataset_id="empty",
            model_id=parse_model_id("toy/model"),
            predictions=(),
        )
