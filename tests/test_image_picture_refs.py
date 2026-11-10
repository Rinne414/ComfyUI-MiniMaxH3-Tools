"""The prompt's <Picture n> references must be compared with the connected pictures, never enforced."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_picture_refs as refs  # noqa: E402


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Edit <Picture 1> and <Picture 2>.", [1, 2]),
        ("<picture 3> and <PICTURE 1>", [1, 3]),
        ("<Picture2> <Picture  2> < Picture 2 >", [2]),
        ("<Picture 10> <Picture 2>", [2, 10]),
        ("Picture 2 shows a cat", []),
        ("<Pictures 2> <Picture> <Video 1>", []),
        ("", []),
    ],
)
def test_references_are_read_in_every_spelling_the_templates_and_users_use(
    prompt: str, expected: list[int]
) -> None:
    assert refs.referenced_pictures(prompt) == expected


@pytest.mark.parametrize(
    "prompt,connected",
    [
        ("make the sky red", 0),
        ("make the sky red", 1),
        ("Edit <Picture 1>: make the sky red", 1),
        ("Put <Picture 2> into <Picture 1>", 2),
        ("", 0),
    ],
)
def test_consistent_prompts_get_no_note(prompt: str, connected: int) -> None:
    assert refs.picture_count_note(prompt, connected) == ""


def test_a_reference_beyond_the_connected_pictures_is_reported() -> None:
    assert (
        refs.picture_count_note("Use <Picture 1> and <Picture 3>", 2)
        == "prompt mentions Picture 3 but only 2 pictures are connected"
    )
    assert (
        refs.picture_count_note("Use <Picture 2>", 1)
        == "prompt mentions Picture 2 but only 1 picture is connected"
    )
    assert (
        refs.picture_count_note("Use <Picture 1>", 0)
        == "prompt mentions Picture 1 but no pictures are connected"
    )
    assert refs.picture_count_note("<Picture 3> <Picture 4>", 1) == (
        "prompt mentions Picture 3 and Picture 4 but only 1 picture is connected"
    )


def test_a_connected_picture_the_prompt_never_names_is_reported() -> None:
    assert (
        refs.picture_count_note("Edit <Picture 1>", 2)
        == "2 pictures connected but the prompt never mentions Picture 2"
    )
    assert refs.picture_count_note("make it red", 3) == (
        "3 pictures connected but the prompt never mentions Picture 1, Picture 2 and Picture 3"
    )


def test_a_missing_reference_wins_over_an_unnamed_picture() -> None:
    note = refs.picture_count_note("<Picture 3>", 2)
    assert note.startswith("prompt mentions Picture 3")


def test_the_note_is_added_to_a_report_only_when_there_is_one() -> None:
    assert refs.append_note("done", "make it red", 1) == "done"
    assert refs.append_note("done", "<Picture 2>", 1) == (
        "done; prompt mentions Picture 2 but only 1 picture is connected"
    )


def test_connected_pictures_follow_the_numeric_order_and_skip_empty_slots() -> None:
    slots = {f"image_{i}": f"p{i}" for i in (1, 2, 10, 11)}
    slots["image_3"] = None
    assert refs.connected_pictures(slots) == ["p1", "p2", "p10", "p11"]
    assert refs.connected_pictures(None) == []
