from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
import torch as t
from jaxtyping import TypeCheckError
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from neural_program_simplification.model_execution import (
    TaskBatch,
    TokenizationError,
    TokenizedTaskDataset,
    TokenizedTaskDocument,
    _pad_token_id,
    collate_task_dataset,
    masked_next_token_loss,
    run_task_dataset,
    tokenize_task_dataset,
    tokenize_task_document,
)
from neural_program_simplification.task_datasets import TaskDataset, TaskDocument
from neural_program_simplification.types import parse_task_text


class FakeTokenizer(PreTrainedTokenizerBase):
    def __init__(self, token_ids_by_text: Mapping[str, list[int]]) -> None:
        super().__init__()
        self._token_ids_by_text = token_ids_by_text
        self._pad_token_id: int | None = 0
        self._eos_token_id: int | None = 99

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value: int | None) -> None:
        self._pad_token_id = value

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @eos_token_id.setter
    def eos_token_id(self, value: int | None) -> None:
        self._eos_token_id = value

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> dict[str, t.Tensor]:
        assert not add_special_tokens
        assert return_tensors == "pt"
        return {"input_ids": t.tensor([self._token_ids_by_text[text]], dtype=t.long)}


class FakeOutputs:
    def __init__(self, logits: t.Tensor) -> None:
        self.logits = logits


class FakeCausalLM(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, logits: t.Tensor) -> None:
        super().__init__(PretrainedConfig())
        self._logits = logits
        self.observed_input_ids: t.Tensor | None = None
        self.observed_attention_mask: t.Tensor | None = None
        self._parameter = t.nn.Parameter(t.zeros(()))

    def forward(self, *, input_ids: t.Tensor, attention_mask: t.Tensor) -> object:
        self.observed_input_ids = input_ids
        self.observed_attention_mask = attention_mask
        return FakeOutputs(self._logits.to(device=input_ids.device))


def _dataset() -> TaskDataset:
    return TaskDataset(
        documents=(
            TaskDocument(text=parse_task_text("Question: 2+2?\nRESPONSE: B")),
            TaskDocument(
                text=parse_task_text("Alice gave the book to Bob."),
                behavior_token_indices=(2, 4),
            ),
        )
    )


def _tokenizer() -> FakeTokenizer:
    return FakeTokenizer(
        {
            "Question: 2+2?\nRESPONSE: B": [10, 11, 12],
            "Alice gave the book to Bob.": [20, 21, 22, 23, 24],
        }
    )


def test_tokenize_task_document_defaults_behavior_to_final_token() -> None:
    document = TaskDocument(text=parse_task_text("Question: 2+2?\nRESPONSE: B"))

    tokenized = tokenize_task_document(document, _tokenizer())

    assert tokenized.input_ids.dtype == t.long
    assert tokenized.input_ids.tolist() == [10, 11, 12]
    assert tokenized.behavior_mask.dtype == t.bool
    assert tokenized.behavior_mask.tolist() == [False, False, True]


def test_tokenize_task_document_uses_explicit_behavior_indices() -> None:
    document = TaskDocument(
        text=parse_task_text("Alice gave the book to Bob."),
        behavior_token_indices=(2, 4),
    )

    tokenized = tokenize_task_document(document, _tokenizer())

    assert tokenized.input_ids.tolist() == [20, 21, 22, 23, 24]
    assert tokenized.behavior_mask.tolist() == [False, False, True, False, True]


def test_tokenize_task_document_rejects_unscorable_or_out_of_bounds_indices() -> None:
    with pytest.raises(TokenizationError, match="index 0"):
        tokenize_task_document(
            TaskDocument(
                text=parse_task_text("Question: 2+2?\nRESPONSE: B"),
                behavior_token_indices=(0,),
            ),
            _tokenizer(),
        )

    with pytest.raises(TokenizationError, match="outside"):
        tokenize_task_document(
            TaskDocument(
                text=parse_task_text("Question: 2+2?\nRESPONSE: B"),
                behavior_token_indices=(3,),
            ),
            _tokenizer(),
        )


def test_tokenized_task_document_validates_tensor_shapes() -> None:
    document = TaskDocument(text=parse_task_text("Question: 2+2?\nRESPONSE: B"))

    with pytest.raises(TokenizationError, match="shape tokens"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([[1, 2]], dtype=t.long),
            behavior_mask=t.tensor([[False, True]]),
        )

    with pytest.raises(TokenizationError, match="torch.long"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([1.0, 2.0]),
            behavior_mask=t.tensor([False, True]),
        )

    with pytest.raises(TokenizationError, match="at least one token"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([], dtype=t.long),
            behavior_mask=t.tensor([], dtype=t.bool),
        )

    with pytest.raises(TokenizationError, match="must match"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([1, 2], dtype=t.long),
            behavior_mask=t.tensor([True], dtype=t.bool),
        )

    with pytest.raises(TokenizationError, match="torch.bool"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([1, 2], dtype=t.long),
            behavior_mask=t.tensor([0, 1], dtype=t.long),
        )

    with pytest.raises(TokenizationError, match="select at least one"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([1, 2], dtype=t.long),
            behavior_mask=t.tensor([False, False]),
        )

    with pytest.raises(TokenizationError, match="token 0"):
        TokenizedTaskDocument(
            document=document,
            input_ids=t.tensor([1, 2], dtype=t.long),
            behavior_mask=t.tensor([True, False]),
        )


def test_tokenize_and_collate_task_dataset() -> None:
    tokenized = tokenize_task_dataset(_dataset(), _tokenizer())
    batch = collate_task_dataset(tokenized, pad_token_id=0)

    assert batch.input_ids.tolist() == [
        [10, 11, 12, 0, 0],
        [20, 21, 22, 23, 24],
    ]
    assert batch.valid_token_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert batch.behavior_mask.tolist() == [
        [False, False, True, False, False],
        [False, False, True, False, True],
    ]


def test_collate_task_dataset_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        TokenizedTaskDataset(documents=())


def test_task_batch_validates_shapes_and_dtypes() -> None:
    input_ids = t.tensor([[1, 2]], dtype=t.long)
    bool_mask = t.tensor([[True, True]], dtype=t.bool)

    with pytest.raises(ValueError, match="batch x tokens"):
        TaskBatch(
            input_ids=t.tensor([1, 2], dtype=t.long),
            valid_token_mask=t.tensor([True, True]),
            behavior_mask=t.tensor([False, True]),
        )

    with pytest.raises(ValueError, match="torch.long"):
        TaskBatch(
            input_ids=t.tensor([[1.0, 2.0]]),
            valid_token_mask=bool_mask,
            behavior_mask=bool_mask,
        )

    with pytest.raises(ValueError, match="valid_token_mask"):
        TaskBatch(
            input_ids=input_ids,
            valid_token_mask=t.tensor([[True]], dtype=t.bool),
            behavior_mask=bool_mask,
        )

    with pytest.raises(ValueError, match="behavior_mask"):
        TaskBatch(
            input_ids=input_ids,
            valid_token_mask=bool_mask,
            behavior_mask=t.tensor([[False]], dtype=t.bool),
        )

    with pytest.raises(ValueError, match="valid_token_mask.*torch.bool"):
        TaskBatch(
            input_ids=input_ids,
            valid_token_mask=t.tensor([[1, 1]], dtype=t.long),
            behavior_mask=bool_mask,
        )

    with pytest.raises(ValueError, match="behavior_mask.*torch.bool"):
        TaskBatch(
            input_ids=input_ids,
            valid_token_mask=bool_mask,
            behavior_mask=t.tensor([[0, 1]], dtype=t.long),
        )


def test_masked_next_token_loss_scores_only_behavior_tokens() -> None:
    input_ids = t.tensor([[0, 1, 2]], dtype=t.long)
    behavior_mask = t.tensor([[False, True, True]], dtype=t.bool)
    logits = t.tensor(
        [
            [
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=t.float32,
    )

    per_token_loss = masked_next_token_loss(logits, input_ids, behavior_mask)

    assert per_token_loss[0, 0] == pytest.approx(0.0)
    assert per_token_loss[0, 1] > 8.0
    assert per_token_loss[0, 2] < 0.001


def test_masked_next_token_loss_rejects_bad_shapes() -> None:
    logits = t.zeros((1, 2, 3), dtype=t.float32)
    input_ids = t.zeros((1, 2), dtype=t.long)
    behavior_mask = t.tensor([[False, True]], dtype=t.bool)

    with pytest.raises(TypeCheckError):
        masked_next_token_loss(t.zeros((1, 3, 3), dtype=t.float32), input_ids, behavior_mask)

    with pytest.raises(TypeCheckError):
        masked_next_token_loss(logits, input_ids, t.zeros((1, 3), dtype=t.bool))

    with pytest.raises(ValueError, match="at least two"):
        masked_next_token_loss(
            t.zeros((1, 1, 3), dtype=t.float32),
            t.zeros((1, 1), dtype=t.long),
            t.zeros((1, 1), dtype=t.bool),
        )

    with pytest.raises(ValueError, match="token 0"):
        masked_next_token_loss(logits, input_ids, t.tensor([[True, False]], dtype=t.bool))


def test_run_task_dataset_uses_raw_huggingface_model_shape() -> None:
    logits = t.zeros((2, 5, 30), dtype=t.float32)
    logits[0, 1, 12] = 5.0
    logits[1, 1, 22] = 5.0
    logits[1, 3, 24] = 5.0
    model = FakeCausalLM(logits)

    task_run = run_task_dataset(_dataset(), model, _tokenizer())

    assert model.observed_input_ids is not None
    assert model.observed_input_ids.tolist() == [
        [10, 11, 12, 0, 0],
        [20, 21, 22, 23, 24],
    ]
    assert model.observed_attention_mask is not None
    assert model.observed_attention_mask.dtype == t.bool
    assert task_run.logits.shape == (2, 5, 30)
    assert task_run.mean_behavior_loss < 0.2
    assert task_run.per_token_loss[0, 0] == pytest.approx(0.0)


def test_run_task_dataset_requires_tensor_logits() -> None:
    class BadOutputs:
        logits = [[1.0]]

    class BadModel(FakeCausalLM):
        def forward(self, *, input_ids: t.Tensor, attention_mask: t.Tensor) -> BadOutputs:
            return BadOutputs()

    with pytest.raises(ValueError, match="tensor logits"):
        run_task_dataset(_dataset(), BadModel(t.zeros((2, 5, 30))), _tokenizer())


def test_run_task_dataset_requires_pad_or_eos_token() -> None:
    class TokenizerWithoutPadOrEos:
        pad_token_id = None
        eos_token_id = None

    with pytest.raises(ValueError, match="pad_token_id or eos_token_id"):
        _pad_token_id(cast(PreTrainedTokenizerBase, TokenizerWithoutPadOrEos()))


def test_type_aliases_accept_torch_tensors() -> None:
    # This is intentionally small: jaxtyping annotations should describe Torch tensors, not NumPy.
    assert isinstance(t.tensor([1, 2], dtype=t.long), t.Tensor)
