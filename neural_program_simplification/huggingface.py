from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch as t
from beartype import beartype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from neural_program_simplification.types import ModelId, parse_model_id

DEFAULT_QWEN35_MODEL_ID = parse_model_id("Qwen/Qwen3.5-0.8B")


@dataclass(frozen=True, slots=True)
class LoadedCausalLM:
    """Loaded Hugging Face causal language model and matching tokenizer."""

    model_id: ModelId
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


@beartype
def load_huggingface_causal_lm(
    model_id: str | ModelId = DEFAULT_QWEN35_MODEL_ID,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: t.dtype | str | None = "auto",
    device_map: str | None = None,
    device: t.device | str | None = None,
) -> LoadedCausalLM:
    parsed_model_id = parse_model_id(str(model_id))

    tokenizer_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(str(parsed_model_id), **tokenizer_kwargs),
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(str(parsed_model_id), **model_kwargs),
    )
    if device is not None:
        cast(Any, model).to(device)
    model.eval()

    return LoadedCausalLM(
        model_id=parsed_model_id,
        model=model,
        tokenizer=tokenizer,
    )
