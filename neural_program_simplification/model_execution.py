from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch as t
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neural_program_simplification.task_datasets import TaskDataset, TaskDocument

TokenIds = Int64[t.Tensor, "tokens"]
BatchTokenIds = Int64[t.Tensor, "batch tokens"]
BatchLogits = Float[t.Tensor, "batch tokens vocab"]
BehaviorMask = Bool[t.Tensor, "tokens"]
BatchBehaviorMask = Bool[t.Tensor, "batch tokens"]
BatchLoss = Float[t.Tensor, "batch tokens"]


class TokenizationError(ValueError):
    """Raised when task documents cannot be tokenized into valid model inputs."""


@dataclass(frozen=True, slots=True)
class TokenizedTaskDocument:
    """A task document tokenized once with a behavior mask over its tokens."""

    document: TaskDocument
    input_ids: TokenIds
    behavior_mask: BehaviorMask

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise TokenizationError("input_ids must have shape tokens")
        if self.input_ids.dtype != t.long:
            raise TokenizationError("input_ids must use torch.long dtype")
        if self.input_ids.numel() == 0:
            raise TokenizationError("task document must tokenize to at least one token")
        if self.behavior_mask.shape != self.input_ids.shape:
            raise TokenizationError("behavior_mask must match input_ids shape")
        if self.behavior_mask.dtype != t.bool:
            raise TokenizationError("behavior_mask must use torch.bool dtype")
        if not bool(t.any(self.behavior_mask).item()):
            raise TokenizationError("behavior_mask must select at least one token")
        if bool(self.behavior_mask[0].item()):
            raise TokenizationError("behavior_mask cannot select token 0 for causal LM loss")


@dataclass(frozen=True, slots=True)
class TokenizedTaskDataset:
    """Task dataset after tokenization with model-behavior masks."""

    documents: Sequence[TokenizedTaskDocument]

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if not documents:
            raise ValueError("tokenized task dataset must contain at least one document")
        object.__setattr__(self, "documents", documents)


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Padded model inputs for a tokenized task dataset."""

    input_ids: BatchTokenIds
    attention_mask: BatchBehaviorMask
    behavior_mask: BatchBehaviorMask

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape batch x tokens")
        if self.input_ids.dtype != t.long:
            raise ValueError("input_ids must use torch.long dtype")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        if self.behavior_mask.shape != self.input_ids.shape:
            raise ValueError("behavior_mask must match input_ids shape")
        if self.attention_mask.dtype != t.bool:
            raise ValueError("attention_mask must use torch.bool dtype")
        if self.behavior_mask.dtype != t.bool:
            raise ValueError("behavior_mask must use torch.bool dtype")


@dataclass(frozen=True, slots=True)
class TaskRun:
    """Outputs from running a task dataset through a causal language model."""

    batch: TaskBatch
    logits: BatchLogits
    per_token_loss: BatchLoss

    def __post_init__(self) -> None:
        if self.logits.ndim != 3:
            raise ValueError("logits must have shape batch x tokens x vocab")
        if self.logits.shape[:2] != self.batch.input_ids.shape:
            raise ValueError("logits batch and token dimensions must match input_ids")
        if self.per_token_loss.shape != self.batch.input_ids.shape:
            raise ValueError("per_token_loss must match input_ids shape")

    @property
    def mean_behavior_loss(self) -> t.Tensor:
        return self.per_token_loss[self.batch.behavior_mask].mean()


def _model_device(model: PreTrainedModel) -> t.device:
    return next(model.parameters()).device


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer must define pad_token_id or eos_token_id")


def _model_logits(outputs: Any) -> t.Tensor:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, t.Tensor):
        raise ValueError("causal language model output must expose tensor logits")
    return logits


def _behavior_indices_for_document(document: TaskDocument, token_count: int) -> tuple[int, ...]:
    if document.behavior_token_indices is None:
        behavior_token_indices = (token_count - 1,)
    else:
        behavior_token_indices = tuple(document.behavior_token_indices)

    for index in behavior_token_indices:
        if index >= token_count:
            raise TokenizationError("behavior token index is outside the tokenized document")
        if index == 0:
            raise TokenizationError("behavior token index 0 cannot be scored by a causal LM")
    return behavior_token_indices


@beartype
def tokenize_task_document(
    document: TaskDocument,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: t.device | str | None = None,
    add_special_tokens: bool = False,
) -> TokenizedTaskDocument:
    encoded = tokenizer(
        str(document.text),
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0].to(device=device, dtype=t.long)
    behavior_mask = t.zeros_like(input_ids, dtype=t.bool)
    behavior_mask[list(_behavior_indices_for_document(document, input_ids.numel()))] = True

    return TokenizedTaskDocument(
        document=document,
        input_ids=input_ids,
        behavior_mask=behavior_mask,
    )


@beartype
def tokenize_task_dataset(
    dataset: TaskDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: t.device | str | None = None,
    add_special_tokens: bool = False,
) -> TokenizedTaskDataset:
    return TokenizedTaskDataset(
        documents=tuple(
            tokenize_task_document(
                document,
                tokenizer,
                device=device,
                add_special_tokens=add_special_tokens,
            )
            for document in dataset.documents
        )
    )


@beartype
def collate_task_dataset(
    dataset: TokenizedTaskDataset,
    *,
    pad_token_id: int,
) -> TaskBatch:
    max_tokens = max(document.input_ids.numel() for document in dataset.documents)
    batch_size = len(dataset.documents)
    device = dataset.documents[0].input_ids.device

    input_ids = t.full((batch_size, max_tokens), pad_token_id, dtype=t.long, device=device)
    attention_mask = t.zeros((batch_size, max_tokens), dtype=t.bool, device=device)
    behavior_mask = t.zeros((batch_size, max_tokens), dtype=t.bool, device=device)

    for row_index, document in enumerate(dataset.documents):
        token_count = document.input_ids.numel()
        input_ids[row_index, :token_count] = document.input_ids
        attention_mask[row_index, :token_count] = True
        behavior_mask[row_index, :token_count] = document.behavior_mask

    return TaskBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        behavior_mask=behavior_mask,
    )


@jaxtyped(typechecker=beartype)
def masked_next_token_loss(
    logits: BatchLogits,
    input_ids: BatchTokenIds,
    behavior_mask: BatchBehaviorMask,
) -> BatchLoss:
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits batch and token dimensions must match input_ids")
    if behavior_mask.shape != input_ids.shape:
        raise ValueError("behavior_mask must match input_ids")
    if logits.shape[1] < 2:
        raise ValueError("at least two tokens are required for causal LM loss")
    if bool(t.any(behavior_mask[:, 0]).item()):
        raise ValueError("behavior_mask cannot select token 0 for causal LM loss")

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape_as(shifted_targets)

    per_token_loss = t.zeros(input_ids.shape, dtype=logits.dtype, device=logits.device)
    per_token_loss[:, 1:] = shifted_loss
    return per_token_loss.masked_fill(~behavior_mask, 0.0)


@beartype
def run_task_dataset(
    dataset: TaskDataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: t.device | str | None = None,
    add_special_tokens: bool = False,
) -> TaskRun:
    if device is None:
        device = _model_device(model)

    tokenized = tokenize_task_dataset(
        dataset,
        tokenizer,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    batch = collate_task_dataset(tokenized, pad_token_id=_pad_token_id(tokenizer))

    with t.inference_mode():
        outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)

    logits = _model_logits(outputs)
    per_token_loss = masked_next_token_loss(logits, batch.input_ids, batch.behavior_mask)
    return TaskRun(batch=batch, logits=logits, per_token_loss=per_token_loss)
