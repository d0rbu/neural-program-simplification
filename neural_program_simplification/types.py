from __future__ import annotations

from beartype import beartype
from phantom import Phantom


def _is_non_empty(value: str) -> bool:
    return bool(value.strip())


class NonEmptyStr(str, Phantom[str], predicate=_is_non_empty, bound=str):
    """String with at least one non-whitespace character."""


class ModelId(NonEmptyStr):
    """Hugging Face model id, local checkpoint path, or other model name."""


class TaskText(NonEmptyStr):
    """Text for one task document."""


@beartype
def parse_model_id(value: str) -> ModelId:
    return ModelId.parse(value)


@beartype
def parse_task_text(value: str) -> TaskText:
    return TaskText.parse(value)
