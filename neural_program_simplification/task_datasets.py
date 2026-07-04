from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from beartype import beartype
from phantom import Phantom

TASK_DATASET_SCHEMA_VERSION = 1


def _is_non_empty(value: str) -> bool:
    return bool(value.strip())


class DatasetId(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """Non-empty identifier for a task dataset."""


class ExampleId(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """Non-empty identifier for one task example."""


class PromptText(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """Non-empty prompt text, including the final task token."""


class AnswerText(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """Non-empty textual suffix that should be one model token."""


@beartype
def parse_dataset_id(value: str) -> DatasetId:
    return DatasetId.parse(value)


@beartype
def parse_example_id(value: str) -> ExampleId:
    return ExampleId.parse(value)


@beartype
def parse_prompt_text(value: str) -> PromptText:
    return PromptText.parse(value)


@beartype
def parse_answer_text(value: str) -> AnswerText:
    return AnswerText.parse(value)


def _freeze_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    checked: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("metadata values must be strings")
        checked[key] = value
    return MappingProxyType(checked)


def _required_str(raw: Mapping[str, Any], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_str(raw: Mapping[str, Any], field_name: str) -> str | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when present")
    return value


def _metadata_from_json(raw: Mapping[str, Any]) -> Mapping[str, str]:
    value = raw.get("metadata", {})
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object when present")

    metadata: dict[str, str] = {}
    for key, metadata_value in value.items():
        if not isinstance(key, str) or not isinstance(metadata_value, str):
            raise ValueError("metadata must contain only string keys and values")
        metadata[key] = metadata_value
    return metadata


class TokenizationError(ValueError):
    """Raised when a task example is not well formed for a tokenizer."""


def _coerce_token_id(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TokenizationError(f"{field_name} must contain integer token ids")
    try:
        token_id = int(value)
    except (TypeError, ValueError) as exc:
        raise TokenizationError(f"{field_name} must contain integer token ids") from exc
    if token_id != value:
        raise TokenizationError(f"{field_name} must contain integer token ids")
    return token_id


@dataclass(frozen=True, slots=True)
class TaskExample:
    """One task prompt whose final token is the task answer."""

    example_id: ExampleId
    prompt: PromptText
    answer_text: AnswerText
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        example_id = parse_example_id(str(self.example_id))
        prompt = parse_prompt_text(str(self.prompt))
        answer_text = parse_answer_text(str(self.answer_text))
        if not prompt.endswith(answer_text):
            raise ValueError("prompt must end with answer_text")

        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "answer_text", answer_text)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.example_id),
            "prompt": str(self.prompt),
            "answer_text": str(self.answer_text),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> TaskExample:
        return cls(
            example_id=parse_example_id(_required_str(raw, "id")),
            prompt=parse_prompt_text(_required_str(raw, "prompt")),
            answer_text=parse_answer_text(_required_str(raw, "answer_text")),
            metadata=_metadata_from_json(raw),
        )


@dataclass(frozen=True, slots=True)
class TaskDataset:
    """Versioned collection of task prompts for a reference behavior check."""

    dataset_id: DatasetId
    examples: Sequence[TaskExample]
    description: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dataset_id = parse_dataset_id(str(self.dataset_id))
        examples = tuple(self.examples)
        if not examples:
            raise ValueError("task dataset must contain at least one example")

        example_ids = [example.example_id for example in examples]
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("task dataset example ids must be unique")

        if self.description is not None and not self.description.strip():
            raise ValueError("description must be non-empty when present")

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_DATASET_SCHEMA_VERSION,
            "dataset_id": str(self.dataset_id),
            "description": self.description,
            "metadata": dict(self.metadata),
            "examples": [example.to_json_dict() for example in self.examples],
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> TaskDataset:
        if raw.get("schema_version") != TASK_DATASET_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {TASK_DATASET_SCHEMA_VERSION}")

        examples_raw = raw.get("examples")
        if not isinstance(examples_raw, Sequence) or isinstance(examples_raw, str):
            raise ValueError("examples must be a list")

        examples: list[TaskExample] = []
        for example_raw in examples_raw:
            if not isinstance(example_raw, Mapping):
                raise ValueError("each example must be an object")
            examples.append(TaskExample.from_json_dict(example_raw))

        return cls(
            dataset_id=parse_dataset_id(_required_str(raw, "dataset_id")),
            description=_optional_str(raw, "description"),
            metadata=_metadata_from_json(raw),
            examples=tuple(examples),
        )


@runtime_checkable
class TaskTokenizer(Protocol):
    """Tokenizer surface needed to validate task-answer tokens."""

    def encode(  # pragma: no cover
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class TokenizedTaskExample:
    """Tokenizer-checked task example with the answer as the final prompt token."""

    example: TaskExample
    prompt_token_ids: Sequence[int]
    answer_token_id: int

    def __post_init__(self) -> None:
        prompt_token_ids = tuple(
            _coerce_token_id(token_id, "prompt token ids") for token_id in self.prompt_token_ids
        )
        answer_token_id = _coerce_token_id(self.answer_token_id, "answer token id")
        if len(prompt_token_ids) < 2:
            raise TokenizationError("prompt must include context before the answer token")
        if any(token_id < 0 for token_id in prompt_token_ids):
            raise TokenizationError("prompt token ids must be non-negative")
        if answer_token_id < 0:
            raise TokenizationError("answer token id must be non-negative")
        if prompt_token_ids[-1] != answer_token_id:
            raise TokenizationError("answer token id must be the final prompt token")

        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)
        object.__setattr__(self, "answer_token_id", answer_token_id)

    @property
    def answer_token_index(self) -> int:
        return len(self.prompt_token_ids) - 1

    @property
    def prefix_token_ids(self) -> tuple[int, ...]:
        return tuple(self.prompt_token_ids[:-1])


@dataclass(frozen=True, slots=True)
class TokenizedTaskDataset:
    """Task dataset validated against one tokenizer."""

    dataset: TaskDataset
    examples: Sequence[TokenizedTaskExample]

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if len(examples) != len(self.dataset.examples):
            raise TokenizationError("tokenized example count must match the dataset")

        dataset_example_ids = [example.example_id for example in self.dataset.examples]
        tokenized_example_ids = [example.example.example_id for example in examples]
        if tokenized_example_ids != dataset_example_ids:
            raise TokenizationError("tokenized examples must preserve dataset example order")

        object.__setattr__(self, "examples", examples)

    @property
    def dataset_id(self) -> DatasetId:
        return self.dataset.dataset_id


def _encode_without_specials(tokenizer: TaskTokenizer, text: str) -> tuple[int, ...]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return tuple(_coerce_token_id(token_id, "tokenizer output") for token_id in token_ids)


@beartype
def tokenize_task_example(
    example: TaskExample,
    tokenizer: TaskTokenizer,
) -> TokenizedTaskExample:
    prompt_token_ids = _encode_without_specials(tokenizer, str(example.prompt))
    answer_token_ids = _encode_without_specials(tokenizer, str(example.answer_text))

    if len(answer_token_ids) != 1:
        raise TokenizationError(
            f"example {example.example_id!s} answer_text must encode to exactly one token"
        )

    answer_token_id = answer_token_ids[0]
    if not prompt_token_ids or prompt_token_ids[-1] != answer_token_id:
        raise TokenizationError(
            f"example {example.example_id!s} answer_text must be the final prompt token"
        )

    return TokenizedTaskExample(
        example=example,
        prompt_token_ids=prompt_token_ids,
        answer_token_id=answer_token_id,
    )


@beartype
def validate_task_dataset_tokenization(
    dataset: TaskDataset,
    tokenizer: TaskTokenizer,
) -> TokenizedTaskDataset:
    return TokenizedTaskDataset(
        dataset=dataset,
        examples=tuple(tokenize_task_example(example, tokenizer) for example in dataset.examples),
    )


@beartype
def load_task_dataset(path: str | PathLike[str]) -> TaskDataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("task dataset file must contain a JSON object")
    return TaskDataset.from_json_dict(raw)


@beartype
def save_task_dataset(
    dataset: TaskDataset,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dataset.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
