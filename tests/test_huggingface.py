from __future__ import annotations

from typing import Any

import torch as t

import neural_program_simplification.huggingface as hf


def test_default_qwen35_model_id_is_small_prototyping_model() -> None:
    assert hf.DEFAULT_QWEN35_MODEL_ID == "Qwen/Qwen3.5-0.8B"


def test_load_huggingface_causal_lm_uses_transformers_factories(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeTokenizer:
        pass

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (model_id, kwargs)
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.device: t.device | str | None = None
            self.evaluated = False

        def to(self, device: t.device | str) -> None:
            self.device = device

        def eval(self) -> None:
            self.evaluated = True

    loaded_model = FakeModel()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls["model"] = (model_id, kwargs)
            return loaded_model

    monkeypatch.setattr(hf, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(hf, "AutoModelForCausalLM", FakeAutoModelForCausalLM)

    loaded = hf.load_huggingface_causal_lm(
        "org/model",
        revision="abc123",
        trust_remote_code=True,
        torch_dtype=t.float16,
        device_map="auto",
        device="cuda",
    )

    assert loaded.model_id == "org/model"
    assert loaded.tokenizer.__class__ is FakeTokenizer
    assert loaded.model is loaded_model
    assert calls["tokenizer"] == (
        "org/model",
        {"trust_remote_code": True, "revision": "abc123"},
    )
    assert calls["model"] == (
        "org/model",
        {
            "trust_remote_code": True,
            "revision": "abc123",
            "torch_dtype": t.float16,
            "device_map": "auto",
        },
    )
    assert loaded_model.device == "cuda"
    assert loaded_model.evaluated


def test_load_huggingface_causal_lm_keeps_optional_kwargs_optional(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["tokenizer"] = (model_id, kwargs)
            return object()

    class FakeModel:
        def eval(self) -> None:
            calls["evaluated"] = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls["model"] = (model_id, kwargs)
            return FakeModel()

    monkeypatch.setattr(hf, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(hf, "AutoModelForCausalLM", FakeAutoModelForCausalLM)

    hf.load_huggingface_causal_lm("org/model", torch_dtype=None)

    assert calls["tokenizer"] == ("org/model", {"trust_remote_code": False})
    assert calls["model"] == ("org/model", {"trust_remote_code": False})
    assert calls["evaluated"]
