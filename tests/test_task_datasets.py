from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest

from neural_program_simplification.task_datasets import (
    TaskDataset,
    TaskExample,
    TokenizationError,
    TokenizedTaskDataset,
    TokenizedTaskExample,
    load_task_dataset,
    parse_answer_text,
    parse_dataset_id,
    parse_example_id,
    parse_prompt_text,
    save_task_dataset,
    validate_task_dataset_tokenization,
)


class ScriptedTokenizer:
    def __init__(self, encodings: Mapping[str, Sequence[int]]) -> None:
        self._encodings = {text: tuple(token_ids) for text, token_ids in encodings.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]:
        assert not add_special_tokens
        return self._encodings[text]


def _mcq_example() -> TaskExample:
    return TaskExample(
        example_id=parse_example_id("mcq-1"),
        prompt=parse_prompt_text("Question: 2+2?\nA. 3\nB. 4\nRESPONSE: B"),
        answer_text=parse_answer_text(" B"),
        metadata={"kind": "multiple-choice"},
    )


def _dataset() -> TaskDataset:
    return TaskDataset(
        dataset_id=parse_dataset_id("arithmetic-mcq"),
        description="Tiny arithmetic multiple-choice prompts.",
        examples=(_mcq_example(),),
        metadata={"split": "dev"},
    )


def test_task_example_requires_prompt_to_end_with_answer_text() -> None:
    with pytest.raises(ValueError, match="end with answer_text"):
        TaskExample(
            example_id=parse_example_id("bad"),
            prompt=parse_prompt_text("Question: 2+2?\nRESPONSE: B"),
            answer_text=parse_answer_text(" A"),
        )


def test_task_dataset_rejects_duplicate_example_ids() -> None:
    example = _mcq_example()

    with pytest.raises(ValueError, match="unique"):
        TaskDataset(dataset_id=parse_dataset_id("duplicates"), examples=(example, example))


def test_task_dataset_metadata_is_immutable() -> None:
    dataset = _dataset()

    metadata = cast(Any, dataset.metadata)
    with pytest.raises(TypeError):
        metadata["new-key"] = "new-value"

    mutable_dataset = cast(Any, dataset)
    with pytest.raises(AttributeError):
        mutable_dataset.description = "changed"


def test_task_dataset_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        TaskDataset(
            dataset_id=parse_dataset_id("empty-description"),
            description=" ",
            examples=(_mcq_example(),),
        )


def test_task_example_rejects_invalid_constructor_metadata() -> None:
    with pytest.raises(ValueError, match="metadata keys"):
        TaskExample(
            example_id=parse_example_id("bad-key"),
            prompt=parse_prompt_text("RESPONSE: A"),
            answer_text=parse_answer_text(" A"),
            metadata={" ": "blank"},
        )

    with pytest.raises(ValueError, match="metadata values"):
        TaskExample(
            example_id=parse_example_id("bad-value"),
            prompt=parse_prompt_text("RESPONSE: A"),
            answer_text=parse_answer_text(" A"),
            metadata=cast(Any, {"kind": 1}),
        )


def test_task_dataset_json_round_trip(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "dataset.json"
    dataset = _dataset()

    save_task_dataset(dataset, destination)

    raw = json.loads(destination.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["examples"][0]["answer_text"] == " B"
    assert load_task_dataset(destination) == dataset


def test_save_task_dataset_refuses_to_overwrite_by_default(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "dataset.json"
    save_task_dataset(_dataset(), destination)

    with pytest.raises(FileExistsError):
        save_task_dataset(_dataset(), destination)

    save_task_dataset(_dataset(), destination, overwrite=True)


def test_validate_task_dataset_tokenization_accepts_single_final_answer_token() -> None:
    dataset = _dataset()
    tokenizer = ScriptedTokenizer(
        {
            str(dataset.examples[0].prompt): (101, 102, 303),
            str(dataset.examples[0].answer_text): (303,),
        }
    )

    tokenized = validate_task_dataset_tokenization(dataset, tokenizer)

    assert tokenized.dataset_id == dataset.dataset_id
    assert tokenized.examples[0].prompt_token_ids == (101, 102, 303)
    assert tokenized.examples[0].answer_token_id == 303
    assert tokenized.examples[0].answer_token_index == 2
    assert tokenized.examples[0].prefix_token_ids == (101, 102)


def test_validate_task_dataset_tokenization_rejects_multi_token_answer_text() -> None:
    dataset = _dataset()
    tokenizer = ScriptedTokenizer(
        {
            str(dataset.examples[0].prompt): (101, 102, 303),
            str(dataset.examples[0].answer_text): (30, 3),
        }
    )

    with pytest.raises(TokenizationError, match="exactly one token"):
        validate_task_dataset_tokenization(dataset, tokenizer)


def test_validate_task_dataset_tokenization_rejects_answer_that_is_not_final_token() -> None:
    dataset = _dataset()
    tokenizer = ScriptedTokenizer(
        {
            str(dataset.examples[0].prompt): (101, 102, 999),
            str(dataset.examples[0].answer_text): (303,),
        }
    )

    with pytest.raises(TokenizationError, match="final prompt token"):
        validate_task_dataset_tokenization(dataset, tokenizer)


def test_tokenized_task_example_requires_context_before_answer_token() -> None:
    with pytest.raises(TokenizationError, match="context"):
        TokenizedTaskExample(
            example=_mcq_example(),
            prompt_token_ids=(303,),
            answer_token_id=303,
        )


def test_task_dataset_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TaskDataset(dataset_id=parse_dataset_id("empty"), examples=())


def test_task_dataset_rejects_bad_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        TaskDataset.from_json_dict({"schema_version": 2, "dataset_id": "bad", "examples": []})


def test_task_dataset_rejects_non_list_examples() -> None:
    with pytest.raises(ValueError, match="examples"):
        TaskDataset.from_json_dict(
            {
                "schema_version": 1,
                "dataset_id": "bad",
                "examples": "not-a-list",
            }
        )


def test_task_dataset_rejects_non_object_examples() -> None:
    with pytest.raises(ValueError, match="each example"):
        TaskDataset.from_json_dict(
            {
                "schema_version": 1,
                "dataset_id": "bad",
                "examples": ["not-an-object"],
            }
        )


def test_task_example_rejects_non_string_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        TaskExample.from_json_dict(
            {
                "id": "bad-metadata",
                "prompt": "RESPONSE: A",
                "answer_text": " A",
                "metadata": {"kind": 1},
            }
        )


def test_task_example_rejects_missing_required_string() -> None:
    with pytest.raises(ValueError, match="answer_text"):
        TaskExample.from_json_dict(
            {
                "id": "missing-answer",
                "prompt": "RESPONSE: A",
            }
        )


def test_task_dataset_rejects_non_string_description() -> None:
    with pytest.raises(ValueError, match="description"):
        TaskDataset.from_json_dict(
            {
                "schema_version": 1,
                "dataset_id": "bad-description",
                "description": 3,
                "examples": [
                    {
                        "id": "example",
                        "prompt": "RESPONSE: A",
                        "answer_text": " A",
                    }
                ],
            }
        )


def test_task_dataset_rejects_non_object_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        TaskDataset.from_json_dict(
            {
                "schema_version": 1,
                "dataset_id": "bad-metadata",
                "metadata": [],
                "examples": [
                    {
                        "id": "example",
                        "prompt": "RESPONSE: A",
                        "answer_text": " A",
                    }
                ],
            }
        )


def test_load_task_dataset_rejects_non_object_json(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_task_dataset(path)


def test_tokenized_task_dataset_preserves_example_order() -> None:
    first = _mcq_example()
    second = TaskExample(
        example_id=parse_example_id("mcq-2"),
        prompt=parse_prompt_text("Question: 1+1?\nRESPONSE: A"),
        answer_text=parse_answer_text(" A"),
    )
    dataset = TaskDataset(dataset_id=parse_dataset_id("ordered"), examples=(first, second))

    with pytest.raises(TokenizationError, match="preserve dataset example order"):
        TokenizedTaskDataset(
            dataset=dataset,
            examples=(
                TokenizedTaskExample(example=second, prompt_token_ids=(10, 11), answer_token_id=11),
                TokenizedTaskExample(example=first, prompt_token_ids=(20, 21), answer_token_id=21),
            ),
        )


def test_tokenized_task_dataset_rejects_count_mismatch() -> None:
    dataset = _dataset()

    with pytest.raises(TokenizationError, match="count"):
        TokenizedTaskDataset(dataset=dataset, examples=())


def test_tokenized_task_example_rejects_negative_token_ids() -> None:
    with pytest.raises(TokenizationError, match="prompt token ids"):
        TokenizedTaskExample(
            example=_mcq_example(),
            prompt_token_ids=(10, -1),
            answer_token_id=-1,
        )

    with pytest.raises(TokenizationError, match="answer token id"):
        TokenizedTaskExample(
            example=_mcq_example(),
            prompt_token_ids=(10, 11),
            answer_token_id=-1,
        )

    with pytest.raises(TokenizationError, match="final prompt token"):
        TokenizedTaskExample(
            example=_mcq_example(),
            prompt_token_ids=(10, 11),
            answer_token_id=12,
        )


def test_tokenized_task_example_rejects_non_integer_token_ids() -> None:
    with pytest.raises(TokenizationError, match="integer"):
        TokenizedTaskExample(
            example=_mcq_example(),
            prompt_token_ids=cast(Any, (10, 11.5)),
            answer_token_id=cast(Any, 11.5),
        )
