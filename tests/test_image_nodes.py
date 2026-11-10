"""The five H3-Image nodes: schemas build, the expander never ships or guesses the official text."""

from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("comfy_extras.nodes_minimax_h3")

import image_expand  # noqa: E402
import image_expand_instructions as instructions  # noqa: E402
import image_nodes  # noqa: E402

EXPECTED_IDS = {
    "MiniMaxH3XImagePromptExpand",
    "MiniMaxH3XImageEncodeLowRAM",
    "MiniMaxH3XImageSize",
    "MiniMaxH3XSaveConditioning",
    "MiniMaxH3XLoadConditioning",
}
GOOD_REPLY = '{"expanded_caption": "A red fox on snow.", "aspect_ratio": "4:3"}'


def test_the_five_nodes_are_registered_with_their_saved_workflow_ids() -> None:
    nodes = image_nodes.get_image_nodes()
    assert {node.define_schema().node_id for node in nodes} == EXPECTED_IDS


@pytest.mark.parametrize("node", image_nodes.get_image_nodes(), ids=lambda node: node.__name__)
def test_schema_builds_with_aliases_and_outputs(node) -> None:
    schema = node.define_schema()
    assert schema.outputs
    assert "h3" in schema.search_aliases and "minimax" in schema.search_aliases
    assert any("\u4e00" <= c <= "\u9fff" for alias in schema.search_aliases for c in alias)
    assert not schema.is_experimental


def test_experimental_parts_say_so_in_their_tooltips() -> None:
    expand = image_expand.MiniMaxH3XImagePromptExpand.define_schema()
    assert expand.description.startswith("Experimental")
    from image_encode import MiniMaxH3XImageEncodeLowRAM

    clip = next(i for i in MiniMaxH3XImageEncodeLowRAM.define_schema().inputs if i.id == "clip_name")
    assert ".gguf" in clip.tooltip and "experimental" in clip.tooltip


def test_an_older_comfyui_keeps_the_rest_of_the_pack_loading(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "image_encode", None)  # makes the import fail like a missing core module
    with caplog.at_level(logging.WARNING, logger=image_nodes.LOGGER.name):
        assert image_nodes.get_image_nodes() == []
    assert "H3-Image nodes unavailable" in caplog.text and "0.45.0" in caplog.text


def test_expander_off_passes_the_prompt_through_without_looking_for_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(image_expand, "find_instructions", fail)
    monkeypatch.setattr(image_expand, "cached_generate", fail)
    output = image_expand.MiniMaxH3XImagePromptExpand.execute("a fox", False, "m.safetensors", 0)
    assert output.result[0] == "a fox"


def test_expander_without_the_official_template_passes_the_prompt_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args, **kwargs):
        raise instructions.InstructionsNotFound(instructions.NOT_FOUND)

    def fail(*args, **kwargs):
        raise AssertionError("no model may load when there are no instructions")

    monkeypatch.setattr(image_expand, "find_instructions", missing)
    monkeypatch.setattr(image_expand, "cached_generate", fail)
    prompt, ratio, report, raw = image_expand.MiniMaxH3XImagePromptExpand.execute(
        "a fox", True, "m.safetensors", 0
    ).result
    assert (prompt, ratio, raw) == ("a fox", "", "")
    assert "not found" in report and "used unchanged" in report


def test_expander_inserts_the_prompt_into_the_task_it_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def fake_find(kind, extra_directories=()):
        seen["kind"] = kind
        return instructions.Instructions("SYS", "TASK: {input text}", "fake.json")

    def fake_generate(model, task, system, seed, max_length, pictures):
        seen.update(task=task, system=system)
        return GOOD_REPLY, False

    monkeypatch.setattr(image_expand, "find_instructions", fake_find)
    monkeypatch.setattr(image_expand, "cached_generate", fake_generate)
    prompt, ratio, report, _ = image_expand.MiniMaxH3XImagePromptExpand.execute(
        "  a fox  ", True, "m.safetensors", 3
    ).result
    assert (prompt, ratio) == ("A red fox on snow.", "4:3")
    assert seen == {"kind": "t2i", "task": "TASK: a fox", "system": "SYS"}
    assert "expanded" in report


def test_expander_uses_the_edit_instructions_when_pictures_are_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    kinds: list[str] = []

    def fake_find(kind, extra_directories=()):
        kinds.append(kind)
        return instructions.Instructions("SYS", "{input text}", "fake.json")

    monkeypatch.setattr(image_expand, "find_instructions", fake_find)
    monkeypatch.setattr(image_expand, "cached_generate", lambda *a: (GOOD_REPLY, True))
    picture = torch.zeros(1, 8, 8, 3)
    image_expand.MiniMaxH3XImagePromptExpand.execute(
        "make it blue", True, "m.safetensors", 0, images={"image_1": picture}
    )
    assert kinds == ["edit"]


def _encode_with_fakes(monkeypatch: pytest.MonkeyPatch, prompt: str, images):
    import torch

    import image_encode

    cond = [[torch.zeros(1, 2, 3), {}]]
    monkeypatch.setattr(image_encode, "encode_once", lambda *args: cond)
    monkeypatch.setattr(image_encode, "_available_mb", lambda: 1000)
    return image_encode.MiniMaxH3XImageEncodeLowRAM.execute(
        "m.safetensors", prompt, use_cache=False, images=images
    )


def test_encode_report_warns_when_the_prompt_names_a_missing_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    picture = torch.zeros(1, 8, 8, 3)
    report = _encode_with_fakes(
        monkeypatch, "Put <Picture 3> into <Picture 1>", {"image_1": picture, "image_2": picture}
    ).result[2]
    assert report.endswith("prompt mentions Picture 3 but only 2 pictures are connected")


def test_encode_report_warns_about_an_unnamed_picture_and_stays_quiet_when_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    picture = torch.zeros(1, 8, 8, 3)
    two = {"image_1": picture, "image_2": picture}
    warned = _encode_with_fakes(monkeypatch, "Edit <Picture 1>", two).result[2]
    assert "never mentions Picture 2" in warned
    quiet = _encode_with_fakes(monkeypatch, "Put <Picture 2> into <Picture 1>", two).result[2]
    assert "mentions" not in quiet and "never" not in quiet
    text_to_image = _encode_with_fakes(monkeypatch, "a red fox", None).result[2]
    assert "mentions" not in text_to_image


def test_encode_still_runs_when_the_references_disagree(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _encode_with_fakes(monkeypatch, "<Picture 5>", None)
    assert output.result[0] and output.result[1]


def test_encode_report_warns_on_a_cache_hit_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    import image_encode

    cond = [[torch.zeros(1, 2, 3), {}]]
    monkeypatch.setattr(image_encode, "encoder_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(image_encode, "file_identity", lambda path: "te|1|2")
    monkeypatch.setattr(image_encode, "cache_folder", lambda: str(tmp_path))
    monkeypatch.setattr(image_encode, "cache_summary", lambda: "cache folder")
    monkeypatch.setattr(image_encode, "read_cache", lambda path: (cond, "cache hit"))
    report = image_encode.MiniMaxH3XImageEncodeLowRAM.execute(
        "m.safetensors", "Use <Picture 2>", use_cache=True, images=None
    ).result[2]
    assert report == "cache hit; cache folder; prompt mentions Picture 2 but no pictures are connected"
