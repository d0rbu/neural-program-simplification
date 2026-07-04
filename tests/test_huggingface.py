from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import neural_program_simplification.huggingface as hf
from neural_program_simplification import DEFAULT_QWEN35_MODEL_ID


class FakeTensor:
    def __init__(self, values: Any, *, device: str = "cpu") -> None:
        self.values = np.asarray(values)
        self.device = device

    def sum(self, dim: int) -> FakeTensor:
        return FakeTensor(np.sum(self.values, axis=dim), device=self.device)

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return FakeTensor(self.values.astype(np.float32), device=self.device)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values)

    def __sub__(self, other: int) -> FakeTensor:
        return FakeTensor(self.values - other, device=self.device)

    def __getitem__(self, key: Any) -> FakeTensor:
        if isinstance(key, tuple):
            normalized_key = tuple(item.values if isinstance(item, FakeTensor) else item for item in key)
            return FakeTensor(self.values[normalized_key], device=self.device)
        return FakeTensor(self.values[key], device=self.device)


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values: Any, *, dtype: Any, device: str) -> FakeTensor:
        assert dtype == FakeTorch.long
        return FakeTensor(values, device=device)

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()

    @staticmethod
    def arange(count: int, *, device: str) -> FakeTensor:
        return FakeTensor(np.arange(count), device=device)


@dataclass(frozen=True)
class FakeTokenizer:
    pad_token_id: int | None = 99
    eos_token_id: int | None = 100


class FakeOutputs:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = FakeTensor(logits)


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.input_ids: np.ndarray | None = None
        self.attention_mask: np.ndarray | None = None

    def __call__(self, *, input_ids: FakeTensor, attention_mask: FakeTensor) -> FakeOutputs:
        self.input_ids = np.asarray(input_ids.values)
        self.attention_mask = np.asarray(attention_mask.values)
        logits = np.zeros((2, 3, 5), dtype=np.float64)
        logits[0, 1, :] = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        logits[1, 2, :] = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
        return FakeOutputs(logits)


def test_default_qwen35_model_id_is_small_prototyping_model() -> None:
    assert DEFAULT_QWEN35_MODEL_ID == "Qwen/Qwen3.5-0.8B"


def test_huggingface_adapter_pads_batches_and_extracts_last_context_logits(monkeypatch) -> None:
    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTorch)
    model = FakeModel()
    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(),
        model=model,
    )

    logits = bundle.next_token_logits(((1, 2), (3, 4, 5)))

    np.testing.assert_array_equal(model.input_ids, np.array([[1, 2, 99], [3, 4, 5]]))
    np.testing.assert_array_equal(model.attention_mask, np.array([[1, 1, 0], [1, 1, 1]]))
    np.testing.assert_array_equal(
        logits,
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_huggingface_adapter_uses_eos_as_fallback_pad_token(monkeypatch) -> None:
    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTorch)
    model = FakeModel()
    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(pad_token_id=None, eos_token_id=7),
        model=model,
    )

    bundle.next_token_logits(((1, 2), (3,)))

    np.testing.assert_array_equal(model.input_ids, np.array([[1, 2], [3, 7]]))


def test_huggingface_adapter_uses_parameter_device_when_model_has_no_device() -> None:
    class FakeParameter:
        device = "meta"

    class ParameterOnlyModel:
        def parameters(self) -> Any:
            yield FakeParameter()

    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(),
        model=ParameterOnlyModel(),
    )

    assert bundle._model_device() == "meta"


def test_huggingface_adapter_rejects_empty_batches(monkeypatch) -> None:
    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTorch)
    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        bundle.next_token_logits(())

    with pytest.raises(ValueError, match="at least one token"):
        bundle.next_token_logits(((),))


def test_huggingface_adapter_rejects_negative_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTorch)
    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    with pytest.raises(ValueError, match="non-negative"):
        bundle.next_token_logits(((1, -2),))


def test_huggingface_adapter_rejects_missing_pad_and_eos_token(monkeypatch) -> None:
    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTorch)
    bundle = hf.HuggingFaceCausalLM(
        model_id=hf.DEFAULT_QWEN35_MODEL_ID,
        tokenizer=FakeTokenizer(pad_token_id=None, eos_token_id=None),
        model=FakeModel(),
    )

    with pytest.raises(ValueError, match="pad_token_id or eos_token_id"):
        bundle.next_token_logits(((1,),))


def test_optional_import_error_mentions_models_extra(monkeypatch) -> None:
    def missing_module(module_name: str) -> Any:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(hf, "import_module", missing_module)

    with pytest.raises(ModuleNotFoundError, match="uv sync --extra models"):
        hf._import_optional_module("torch")


def test_load_huggingface_causal_lm_uses_transformers_factories(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (model_id, kwargs)
            return FakeTokenizer()

    class FakeLoadedModel:
        def __init__(self) -> None:
            self.device: str | None = None
            self.evaluated = False

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.evaluated = True

    loaded_model = FakeLoadedModel()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeLoadedModel:
            calls["model"] = (model_id, kwargs)
            return loaded_model

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoModelForCausalLM = FakeAutoModelForCausalLM

    def fake_import_module(module_name: str) -> type[FakeTransformers]:
        assert module_name == "transformers"
        return FakeTransformers

    monkeypatch.setattr(hf, "import_module", fake_import_module)

    bundle = hf.load_huggingface_causal_lm(
        "org/model",
        revision="abc123",
        trust_remote_code=True,
        device="cuda",
        device_map="auto",
    )

    assert bundle.model_id == "org/model"
    assert calls["tokenizer"] == (
        "org/model",
        {"trust_remote_code": True, "revision": "abc123"},
    )
    assert calls["model"] == (
        "org/model",
        {
            "trust_remote_code": True,
            "revision": "abc123",
            "torch_dtype": "auto",
            "device_map": "auto",
        },
    )
    assert loaded_model.device == "cuda"
    assert loaded_model.evaluated


def test_load_huggingface_causal_lm_omits_optional_model_kwargs(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (model_id, kwargs)
            return FakeTokenizer()

    class FakeLoadedModel:
        def eval(self) -> None:
            calls["evaluated"] = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeLoadedModel:
            calls["model"] = (model_id, kwargs)
            return FakeLoadedModel()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoModelForCausalLM = FakeAutoModelForCausalLM

    monkeypatch.setattr(hf, "import_module", lambda module_name: FakeTransformers)

    hf.load_huggingface_causal_lm("org/minimal", torch_dtype=None)

    assert calls["tokenizer"] == ("org/minimal", {"trust_remote_code": False})
    assert calls["model"] == ("org/minimal", {"trust_remote_code": False})
    assert calls["evaluated"]
