from __future__ import annotations

import json
from typing import cast

import pytest
import torch as t
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from neural_program_simplification import DEFAULT_QWEN35_MODEL_ID
from neural_program_simplification.model_execution import (
    TaskRun,
    collate_task_dataset,
    run_task_dataset,
    tokenize_task_dataset,
)
from neural_program_simplification.task_dataset_library import (
    BUILTIN_TASK_DATASET_NAMES,
    builtin_task_dataset_names,
    builtin_task_dataset_resource,
    iter_builtin_task_datasets,
    load_builtin_task_dataset,
)
from neural_program_simplification.task_datasets import TaskDataset


class FakeOutputs:
    def __init__(self, logits: t.Tensor) -> None:
        self.logits = logits


class NextTokenOracleModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self._parameter = t.nn.Parameter(t.zeros(()))
        self.calls = 0

    def forward(self, *, input_ids: t.Tensor, attention_mask: t.Tensor) -> object:
        self.calls += 1
        token_count = input_ids.shape[1]
        max_real_token_id = int(input_ids[attention_mask].max().item())
        logits = t.zeros(
            (*input_ids.shape, max_real_token_id + 1),
            dtype=t.float32,
            device=input_ids.device,
        )

        for batch_index in range(input_ids.shape[0]):
            for token_index in range(token_count - 1):
                if bool(attention_mask[batch_index, token_index + 1].item()):
                    target = int(input_ids[batch_index, token_index + 1].item())
                    logits[batch_index, token_index, target] = 32.0

        return FakeOutputs(logits)


@pytest.fixture(scope="session")
def default_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_QWEN35_MODEL_ID))
    return cast(PreTrainedTokenizerBase, tokenizer)


def test_builtin_dataset_names_are_unique_sorted_and_present() -> None:
    names = builtin_task_dataset_names()

    assert names == BUILTIN_TASK_DATASET_NAMES
    assert names == tuple(sorted(names))
    assert len(set(names)) == len(names)
    assert set(names) == {
        "arithmetic_multiple_choice",
        "arithmetic_multiple_choice_aug",
        "factual_recall",
        "factual_recall_aug",
        "indirect_object_identification",
        "indirect_object_identification_aug",
        "python_code_completion",
        "python_code_completion_aug",
        "sentiment_classification",
        "sentiment_classification_aug",
        "translation_en_fr",
        "translation_en_fr_aug",
    }


def test_builtin_dataset_resources_match_registry() -> None:
    for name in BUILTIN_TASK_DATASET_NAMES:
        resource = builtin_task_dataset_resource(name)
        assert resource.name == f"{name}.json"
        assert resource.is_file()


def test_builtin_dataset_resource_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown"):
        builtin_task_dataset_resource("not-a-dataset")


@pytest.mark.parametrize("name", BUILTIN_TASK_DATASET_NAMES)
def test_builtin_dataset_json_is_canonical_and_loadable(name: str) -> None:
    resource = builtin_task_dataset_resource(name)
    raw = json.loads(resource.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)
    assert raw["schema_version"] == 1
    assert raw["description"].strip()
    expected_document_count = 8 if name.endswith("_aug") else 4
    assert len(raw["documents"]) == expected_document_count

    dataset = load_builtin_task_dataset(name)

    assert isinstance(dataset, TaskDataset)
    assert dataset.to_json_dict() == raw
    assert dataset.description is not None
    assert "Hand-authored synthetic" in dataset.description
    assert all(document.behavior_token_indices is None for document in dataset.documents)
    assert all(str(document.text).strip() == str(document.text) for document in dataset.documents)


def test_iter_builtin_task_datasets_preserves_registry_order() -> None:
    loaded = iter_builtin_task_datasets()

    assert tuple(name for name, _dataset in loaded) == BUILTIN_TASK_DATASET_NAMES
    assert all(isinstance(dataset, TaskDataset) for _name, dataset in loaded)


@pytest.mark.parametrize("name", BUILTIN_TASK_DATASET_NAMES)
def test_builtin_datasets_tokenize_with_current_default_tokenizer(
    name: str,
    default_tokenizer: PreTrainedTokenizerBase,
) -> None:
    dataset = load_builtin_task_dataset(name)

    tokenized = tokenize_task_dataset(dataset, default_tokenizer)
    batch = collate_task_dataset(tokenized, pad_token_id=0)

    assert len(tokenized.documents) == len(dataset.documents)
    assert batch.input_ids.ndim == 2
    assert batch.valid_token_mask.shape == batch.input_ids.shape
    assert batch.behavior_mask.shape == batch.input_ids.shape
    assert int(batch.behavior_mask.sum().item()) == len(dataset.documents)
    assert not bool(t.any(batch.behavior_mask[:, 0]).item())
    assert bool(t.all(batch.valid_token_mask[batch.behavior_mask]).item())

    for document in tokenized.documents:
        assert document.input_ids.numel() >= 2
        assert int(document.behavior_mask.sum().item()) == 1
        assert bool(document.behavior_mask[-1].item())


@pytest.mark.parametrize("name", BUILTIN_TASK_DATASET_NAMES)
def test_builtin_datasets_run_with_current_default_tokenizer_and_fake_model(
    name: str,
    default_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = NextTokenOracleModel()
    dataset = load_builtin_task_dataset(name)

    task_run = run_task_dataset(dataset, model, default_tokenizer)

    assert isinstance(task_run, TaskRun)
    assert model.calls == 1
    assert task_run.batch.input_ids.shape[:2] == task_run.logits.shape[:2]
    assert task_run.per_token_loss.shape == task_run.batch.input_ids.shape
    assert task_run.mean_behavior_loss < 0.01
    assert float(task_run.per_token_loss[~task_run.batch.behavior_mask].sum().item()) == 0.0


def test_builtin_datasets_cover_distinct_task_families() -> None:
    descriptions = {
        name: load_builtin_task_dataset(name).description or "" for name in BUILTIN_TASK_DATASET_NAMES
    }

    expected_keywords = {
        "arithmetic_multiple_choice": "arithmetic",
        "arithmetic_multiple_choice_aug": "arithmetic",
        "factual_recall": "factual",
        "factual_recall_aug": "factual",
        "indirect_object_identification": "IOI-style",
        "indirect_object_identification_aug": "IOI-style",
        "python_code_completion": "Python",
        "python_code_completion_aug": "Python",
        "sentiment_classification": "sentiment",
        "sentiment_classification_aug": "sentiment",
        "translation_en_fr": "translation",
        "translation_en_fr_aug": "translation",
    }
    for name, keyword in expected_keywords.items():
        assert keyword in descriptions[name]


@pytest.mark.parametrize(
    "name",
    tuple(name for name in BUILTIN_TASK_DATASET_NAMES if name.endswith("_aug")),
)
def test_augmented_builtin_datasets_have_varied_prompt_formats(name: str) -> None:
    dataset = load_builtin_task_dataset(name)
    texts = [str(document.text) for document in dataset.documents]

    assert len(texts) == 8
    assert len(set(texts)) == len(texts)
    if name == "indirect_object_identification_aug":
        assert all("\n" not in text for text in texts)
        assert len({text.split()[0] for text in texts}) == len(texts)
    else:
        assert any("\n" in text for text in texts)
        assert any(separator in text for text in texts for separator in ("|", "=>", "->", "=", "{"))


def test_ioi_builtin_datasets_are_sentence_completions_not_qa_prompts() -> None:
    forbidden_fragments = (
        "Answer:",
        "answer",
        "Question:",
        "question",
        "Recipient:",
        "recipient",
        "Indirect object:",
        "indirect object",
        "Name:",
        "Who ",
        "Identify",
        "Choose",
    )

    for name in ("indirect_object_identification", "indirect_object_identification_aug"):
        dataset = load_builtin_task_dataset(name)
        for document in dataset.documents:
            text = str(document.text)
            assert "\n" not in text
            assert all(fragment not in text for fragment in forbidden_fragments)


def test_builtin_task_dataset_loader_rejects_non_object_resource(monkeypatch) -> None:
    class FakeResource:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return "[]"

    monkeypatch.setattr(
        "neural_program_simplification.task_dataset_library.builtin_task_dataset_resource",
        lambda name: FakeResource(),
    )

    with pytest.raises(ValueError, match="JSON object"):
        load_builtin_task_dataset("arithmetic_multiple_choice")
