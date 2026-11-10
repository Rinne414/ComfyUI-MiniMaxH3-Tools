"""The expander reads its instructions from the user's installed ComfyUI templates (fake templates here)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_expand_instructions as instructions  # noqa: E402

SYSTEM = "FAKE SYSTEM: answer with one JSON object."
TASK = "FAKE TASK for {input text} -- write the caption."


def _string_node(title: str, text: str, node_id: int = 1) -> dict:
    return {
        "id": node_id,
        "type": "PrimitiveStringMultiline",
        "title": title,
        "widgets_values": [text],
    }


def _flat_template(system: str = SYSTEM, task: str = TASK) -> dict:
    return {
        "nodes": [
            _string_node("Text (system_prompt)", system, 1),
            _string_node("Text (task_prompt)", task, 2),
            _string_node("Text (user_prompt)", "", 3),
        ]
    }


def _subgraph_template() -> dict:
    return {
        "nodes": [{"id": 9, "type": "some-subgraph-id"}],
        "definitions": {
            "subgraphs": [
                {
                    "nodes": [
                        _string_node("Text (system_prompt)", SYSTEM, 17),
                        _string_node("Text (task_prompt)", TASK, 20),
                    ]
                }
            ]
        },
    }


def _write(folder: Path, name: str, template: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(template), encoding="utf-8")
    return path


def test_reads_the_pair_from_a_flat_template(tmp_path: Path) -> None:
    path = _write(tmp_path, instructions.TEMPLATE_FILES["edit"], _flat_template())
    found = instructions.find_instructions("edit", [tmp_path])
    assert (found.system, found.task, found.source) == (SYSTEM, TASK, str(path))


def test_reads_the_pair_from_a_subgraph(tmp_path: Path) -> None:
    _write(tmp_path, instructions.TEMPLATE_FILES["t2i"], _subgraph_template())
    found = instructions.find_instructions("t2i", [tmp_path])
    assert (found.system, found.task) == (SYSTEM, TASK)


def test_each_kind_reads_its_own_file(tmp_path: Path) -> None:
    _write(tmp_path, instructions.TEMPLATE_FILES["t2i"], _flat_template(task="T2I {input text}"))
    _write(tmp_path, instructions.TEMPLATE_FILES["edit"], _flat_template(task="EDIT {input text}"))
    assert instructions.find_instructions("t2i", [tmp_path]).task == "T2I {input text}"
    assert instructions.find_instructions("edit", [tmp_path]).task == "EDIT {input text}"


def test_the_first_folder_with_a_usable_template_wins(tmp_path: Path) -> None:
    broken, good = tmp_path / "a", tmp_path / "b"
    _write(broken, instructions.TEMPLATE_FILES["t2i"], {"nodes": []})
    _write(good, instructions.TEMPLATE_FILES["t2i"], _flat_template())
    assert instructions.find_instructions("t2i", [broken, good]).system == SYSTEM


def test_a_missing_template_gives_a_plain_message(tmp_path: Path) -> None:
    with pytest.raises(instructions.InstructionsNotFound) as error:
        instructions.find_instructions("t2i", [tmp_path])
    message = str(error.value)
    assert "not found" in message and "your prompt is used unchanged" in message
    assert "requirements.txt" in message


@pytest.mark.parametrize(
    "template",
    [
        {"nodes": []},
        _flat_template(task="no placeholder here"),
        _flat_template(system="   "),
        {"nodes": [_string_node("Text (system_prompt)", SYSTEM)]},
        {"nodes": [{"type": "PrimitiveStringMultiline", "title": "Text (task_prompt)"}]},
        ["not", "a", "template"],
    ],
)
def test_a_template_with_another_layout_is_reported_not_guessed(tmp_path: Path, template: object) -> None:
    _write(tmp_path, instructions.TEMPLATE_FILES["t2i"], template)
    with pytest.raises(instructions.InstructionsNotFound) as error:
        instructions.find_instructions("t2i", [tmp_path])
    assert "does not recognise" in str(error.value)
    assert "your prompt is used unchanged" in str(error.value)


def test_unreadable_json_is_treated_like_an_unrecognised_layout(tmp_path: Path) -> None:
    (tmp_path / instructions.TEMPLATE_FILES["t2i"]).write_text("{broken", encoding="utf-8")
    with pytest.raises(instructions.InstructionsNotFound):
        instructions.find_instructions("t2i", [tmp_path])


def test_a_template_saved_by_the_user_is_found_by_its_name(tmp_path: Path) -> None:
    saved = _write(tmp_path / "workflows" / "mine", "my_minimax_h3_image_edit copy.json", _flat_template())
    found = instructions.find_instructions("edit", [], extra_directories=[tmp_path / "workflows"])
    assert found.source == str(saved)
    with pytest.raises(instructions.InstructionsNotFound):
        instructions.find_instructions("t2i", [], extra_directories=[tmp_path / "workflows"])


def test_an_updated_file_is_read_again(tmp_path: Path) -> None:
    path = _write(tmp_path, instructions.TEMPLATE_FILES["t2i"], _flat_template(system="FIRST"))
    assert instructions.find_instructions("t2i", [tmp_path]).system == "FIRST"
    path.write_text(json.dumps(_flat_template(system="SECOND")), encoding="utf-8")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
    assert instructions.find_instructions("t2i", [tmp_path]).system == "SECOND"


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        instructions.find_instructions("video", [])


def test_package_directories_follow_the_installed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / instructions.PACKAGE_NAME
    (package / "templates").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert package / "templates" in instructions.package_directories()


def test_this_pack_ships_no_copy_of_the_official_instruction_text() -> None:
    """The instruction text is Comfy-Org's: it must be read at run time, never stored in this repository."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / "prompts").exists()
    for path in list(root.glob("*.py")) + list(root.glob("*.json")) + list(root.rglob("locales/*/*.json")):
        text = path.read_text(encoding="utf-8")
        assert "caption-expansion and canvas-selection" not in text, path.name
        assert "enhancing a short image-editing instruction" not in text, path.name
