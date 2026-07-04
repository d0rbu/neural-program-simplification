from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from beartype import beartype

from neural_program_simplification.types import TaskText, parse_task_text

TASK_DATASET_SCHEMA_VERSION = 1


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


def _coerce_token_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("behavior token indices must be integers")
    try:
        token_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior token indices must be integers") from exc
    if token_index != value:
        raise ValueError("behavior token indices must be integers")
    if token_index < 0:
        raise ValueError("behavior token indices must be non-negative")
    return token_index


def _coerce_behavior_token_indices(indices: Sequence[int] | None) -> tuple[int, ...] | None:
    if indices is None:
        return None

    coerced = tuple(_coerce_token_index(index) for index in indices)
    if not coerced:
        raise ValueError("behavior_token_indices must be non-empty when present")
    if len(set(coerced)) != len(coerced):
        raise ValueError("behavior_token_indices must not contain duplicates")
    return coerced


def _behavior_token_indices_from_json(raw: Mapping[str, Any]) -> tuple[int, ...] | None:
    if "behavior_token_indices" not in raw:
        return None

    value = raw["behavior_token_indices"]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("behavior_token_indices must be a list of integers")
    return _coerce_behavior_token_indices(value)


@dataclass(frozen=True, slots=True)
class TaskDocument:
    """One tokenizable task document with optional behavior-token positions."""

    text: TaskText
    behavior_token_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", parse_task_text(str(self.text)))
        object.__setattr__(
            self,
            "behavior_token_indices",
            _coerce_behavior_token_indices(self.behavior_token_indices),
        )

    def to_json_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {"text": str(self.text)}
        if self.behavior_token_indices is not None:
            raw["behavior_token_indices"] = list(self.behavior_token_indices)
        return raw

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> TaskDocument:
        return cls(
            text=parse_task_text(_required_str(raw, "text")),
            behavior_token_indices=_behavior_token_indices_from_json(raw),
        )


@dataclass(frozen=True, slots=True)
class TaskDataset:
    """Versioned collection of task documents."""

    documents: Sequence[TaskDocument]
    description: str | None = None

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if not documents:
            raise ValueError("task dataset must contain at least one document")
        if self.description is not None and not self.description.strip():
            raise ValueError("description must be non-empty when present")

        object.__setattr__(self, "documents", documents)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_DATASET_SCHEMA_VERSION,
            "description": self.description,
            "documents": [document.to_json_dict() for document in self.documents],
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> TaskDataset:
        if raw.get("schema_version") != TASK_DATASET_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {TASK_DATASET_SCHEMA_VERSION}")

        documents_raw = raw.get("documents")
        if not isinstance(documents_raw, Sequence) or isinstance(documents_raw, str):
            raise ValueError("documents must be a list")

        documents: list[TaskDocument] = []
        for document_raw in documents_raw:
            if not isinstance(document_raw, Mapping):
                raise ValueError("each document must be an object")
            documents.append(TaskDocument.from_json_dict(document_raw))

        return cls(
            description=_optional_str(raw, "description"),
            documents=tuple(documents),
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
