from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from beartype import beartype

from neural_program_simplification.model_execution import ModelId, parse_model_id

DEFAULT_QWEN35_MODEL_ID = parse_model_id("Qwen/Qwen3.5-0.8B")
_MODELS_EXTRA_MESSAGE = "Install model runtime dependencies with `uv sync --extra models`."


def _import_optional_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MODELS_EXTRA_MESSAGE) from exc


@dataclass(frozen=True, slots=True)
class HuggingFaceCausalLM:
    """Hugging Face adapter for the NextTokenModel protocol."""

    model_id: ModelId
    tokenizer: Any
    model: Any

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        return int(pad_token_id)

    def _model_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def next_token_logits(self, token_id_batches: Sequence[Sequence[int]]) -> np.ndarray:
        torch = _import_optional_module("torch")

        batches = tuple(tuple(int(token_id) for token_id in batch) for batch in token_id_batches)
        if not batches:
            raise ValueError("token_id_batches must not be empty")
        if any(not batch for batch in batches):
            raise ValueError("each token id batch must contain at least one token")
        if any(token_id < 0 for batch in batches for token_id in batch):
            raise ValueError("token ids must be non-negative")

        pad_token_id = self._pad_token_id()
        max_length = max(len(batch) for batch in batches)
        padded_input_ids = [
            [*batch, *([pad_token_id] * (max_length - len(batch)))] for batch in batches
        ]
        attention_masks = [
            [*(1 for _ in batch), *([0] * (max_length - len(batch)))] for batch in batches
        ]

        device = self._model_device()
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        row_indices = torch.arange(len(batches), device=outputs.logits.device)
        last_token_indices = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[row_indices, last_token_indices, :]
        return np.asarray(logits.detach().float().cpu().numpy(), dtype=np.float64)


@beartype
def load_huggingface_causal_lm(
    model_id: str | ModelId = DEFAULT_QWEN35_MODEL_ID,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | None = "auto",
    device_map: str | None = None,
    device: str | None = None,
) -> HuggingFaceCausalLM:
    """Load a Hugging Face causal LM and tokenizer for task-dataset scoring."""

    transformers = _import_optional_module("transformers")
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(parsed_model_id), **tokenizer_kwargs)
    model = transformers.AutoModelForCausalLM.from_pretrained(str(parsed_model_id), **model_kwargs)
    if device is not None:
        model.to(device)
    model.eval()

    return HuggingFaceCausalLM(
        model_id=parsed_model_id,
        tokenizer=tokenizer,
        model=model,
    )
