from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable

from beartype import beartype

from neural_program_simplification.task_datasets import TaskDataset

_TASK_DATASET_PACKAGE = "neural_program_simplification.data.task_datasets"

BUILTIN_TASK_DATASET_NAMES = (
    "arithmetic_multiple_choice",
    "factual_recall",
    "indirect_object_identification",
    "python_code_completion",
    "sentiment_classification",
    "translation_en_fr",
)


def builtin_task_dataset_names() -> tuple[str, ...]:
    return BUILTIN_TASK_DATASET_NAMES


@beartype
def builtin_task_dataset_resource(name: str) -> Traversable:
    if name not in BUILTIN_TASK_DATASET_NAMES:
        raise ValueError(f"unknown built-in task dataset: {name}")
    return files(_TASK_DATASET_PACKAGE).joinpath(f"{name}.json")


@beartype
def load_builtin_task_dataset(name: str) -> TaskDataset:
    raw = json.loads(builtin_task_dataset_resource(name).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("built-in task dataset resource must contain a JSON object")
    return TaskDataset.from_json_dict(raw)


def iter_builtin_task_datasets() -> tuple[tuple[str, TaskDataset], ...]:
    return tuple((name, load_builtin_task_dataset(name)) for name in BUILTIN_TASK_DATASET_NAMES)
