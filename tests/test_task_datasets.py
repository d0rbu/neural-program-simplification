from __future__ import annotations

import json
from typing import Any, cast

import pytest

from neural_program_simplification.task_datasets import (
    TaskDataset,
    TaskDocument,
    load_task_dataset,
    save_task_dataset,
)
from neural_program_simplification.types import TaskText, parse_task_text


def test_task_document_accepts_final_token_default_behavior() -> None:
    document = TaskDocument(text=parse_task_text("Question: 2+2?\nRESPONSE: B"))

    assert isinstance(document.text, TaskText)
    assert document.behavior_token_indices is None
    assert document.to_json_dict() == {"text": "Question: 2+2?\nRESPONSE: B"}


def test_task_document_accepts_arbitrary_behavior_token_indices() -> None:
    document = TaskDocument(
        text=parse_task_text("Alice gave the book to Bob."),
        behavior_token_indices=(2, 5),
    )

    assert document.behavior_token_indices == (2, 5)
    assert document.to_json_dict() == {
        "text": "Alice gave the book to Bob.",
        "behavior_token_indices": [2, 5],
    }


def test_task_document_rejects_empty_text() -> None:
    with pytest.raises((TypeError, ValueError)):
        TaskDocument(text=parse_task_text(" "))


@pytest.mark.parametrize(
    "indices",
    [
        (),
        (-1,),
        (1, 1),
        (True,),
        (1.5,),
    ],
)
def test_task_document_rejects_bad_behavior_indices(indices: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        TaskDocument(
            text=parse_task_text("Question: 2+2?\nRESPONSE: B"),
            behavior_token_indices=cast(Any, indices),
        )


def test_task_dataset_json_round_trip(tmp_path) -> None:  # noqa: ANN001
    dataset = TaskDataset(
        description="Tiny task documents.",
        documents=(
            TaskDocument(text=parse_task_text("Question: 2+2?\nRESPONSE: B")),
            TaskDocument(
                text=parse_task_text("Alice gave the book to Bob."),
                behavior_token_indices=(2, 5),
            ),
        ),
    )
    path = tmp_path / "task-dataset.json"

    save_task_dataset(dataset, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["documents"][1]["behavior_token_indices"] == [2, 5]
    assert load_task_dataset(path) == dataset


def test_task_dataset_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        TaskDataset(documents=())


def test_task_dataset_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        TaskDataset(
            description=" ",
            documents=(TaskDocument(text=parse_task_text("A task document.")),),
        )


def test_task_dataset_is_immutable() -> None:
    dataset = TaskDataset(documents=(TaskDocument(text=parse_task_text("A task document.")),))

    mutable_dataset = cast(Any, dataset)
    with pytest.raises(AttributeError):
        mutable_dataset.description = "changed"


def test_save_task_dataset_refuses_to_overwrite_by_default(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "task-dataset.json"
    dataset = TaskDataset(documents=(TaskDocument(text=parse_task_text("A task document.")),))
    save_task_dataset(dataset, path)

    with pytest.raises(FileExistsError):
        save_task_dataset(dataset, path)

    save_task_dataset(dataset, path, overwrite=True)


def test_task_dataset_rejects_bad_json_shapes(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "task-dataset.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_task_dataset(path)

    with pytest.raises(ValueError, match="schema_version"):
        TaskDataset.from_json_dict({"schema_version": 2, "documents": []})

    with pytest.raises(ValueError, match="documents"):
        TaskDataset.from_json_dict({"schema_version": 1, "documents": "bad"})

    with pytest.raises(ValueError, match="each document"):
        TaskDataset.from_json_dict({"schema_version": 1, "documents": ["bad"]})

    with pytest.raises(ValueError, match="text"):
        TaskDocument.from_json_dict({"behavior_token_indices": [1]})

    with pytest.raises(ValueError, match="description"):
        TaskDataset.from_json_dict(
            {
                "schema_version": 1,
                "description": 1,
                "documents": [{"text": "A task document."}],
            }
        )

    with pytest.raises(ValueError, match="behavior_token_indices"):
        TaskDocument.from_json_dict(
            {
                "text": "A task document.",
                "behavior_token_indices": "bad",
            }
        )
