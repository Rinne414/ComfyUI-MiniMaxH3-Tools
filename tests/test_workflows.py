"""Validate the committed ComfyUI example workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.generate_example_workflows import Workflow, WorkflowNode, build_workflows

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / "example_workflows"

EXPECTED_NODE_TYPES = {
    "MiniMax-H3_Planning_and_Prompt_Lint.json": {
        "MarkdownNote",
        "MiniMaxH3XCanvasSolver",
        "MiniMaxH3XPromptLint",
    },
    "MiniMax-H3_AV_Latent_Tools.json": {
        "MarkdownNote",
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3XSplitAV",
        "MiniMaxH3XUpscaleVideoLatent",
        "MiniMaxH3XJoinAV",
        "MiniMaxH3XTemporalTrim",
        "MiniMaxH3XLatentInfo",
    },
    "MiniMax-H3_Audio_Reroll_Experimental.json": {
        "MarkdownNote",
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
        "BasicGuider",
        "KSamplerSelect",
        "BasicScheduler",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "MiniMaxH3XAudioReroll",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
        "SaveAudio",
    },
}


def _load_workflow(path: Path) -> Workflow:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("%s must contain a JSON object" % path)
    required = {
        "id",
        "revision",
        "last_node_id",
        "last_link_id",
        "nodes",
        "links",
        "groups",
        "config",
        "extra",
        "version",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("%s is missing fields: %s" % (path, ", ".join(missing)))
    return cast(Workflow, raw)


def _node_by_type(workflow: Workflow, node_type: str) -> list[WorkflowNode]:
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def _assert_bilingual_text(text: str, context: str) -> None:
    assert "/" in text, "%s must separate English and Simplified Chinese with '/'" % context
    assert any(character.isascii() and character.isalpha() for character in text), (
        "%s must contain English text" % context)
    assert any("\u4e00" <= character <= "\u9fff" for character in text), (
        "%s must contain Simplified Chinese text" % context)


def _link_tuples(workflow: Workflow) -> set[tuple[int, int, int, int, int, str]]:
    parsed: set[tuple[int, int, int, int, int, str]] = set()
    for link in workflow["links"]:
        if len(link) != 6:
            raise ValueError("workflow link must have six fields, got %r" % link)
        link_id, source_id, source_slot, target_id, target_slot, link_type = link
        if not all(isinstance(value, int) for value in (
                link_id, source_id, source_slot, target_id, target_slot)):
            raise TypeError("workflow link IDs and slots must be integers: %r" % link)
        if not isinstance(link_type, str):
            raise TypeError("workflow link type must be a string: %r" % link)
        parsed.add((
            cast(int, link_id),
            cast(int, source_id),
            cast(int, source_slot),
            cast(int, target_id),
            cast(int, target_slot),
            link_type,
        ))
    return parsed


def _assert_graph_integrity(workflow: Workflow) -> None:
    nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
    if len(nodes_by_id) != len(workflow["nodes"]):
        raise ValueError("workflow contains duplicate node IDs")
    links = _link_tuples(workflow)
    link_ids = {link[0] for link in links}
    if len(link_ids) != len(links):
        raise ValueError("workflow contains duplicate link IDs")

    for link_id, source_id, source_slot, target_id, target_slot, link_type in links:
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise ValueError("link %d references a missing node" % link_id)
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        if source_slot >= len(source["outputs"]):
            raise ValueError("link %d references missing source slot" % link_id)
        if target_slot >= len(target["inputs"]):
            raise ValueError("link %d references missing target slot" % link_id)
        if source["outputs"][source_slot]["type"] != link_type:
            raise ValueError("link %d source type does not match" % link_id)
        if target["inputs"][target_slot]["type"] != link_type:
            raise ValueError("link %d target type does not match" % link_id)
        if target["inputs"][target_slot]["link"] != link_id:
            raise ValueError("link %d is missing from its target input" % link_id)
        if link_id not in source["outputs"][source_slot]["links"]:
            raise ValueError("link %d is missing from its source output" % link_id)

    assert workflow["last_node_id"] == max(nodes_by_id)
    assert workflow["last_link_id"] == max(link_ids, default=0)


@pytest.mark.parametrize("filename", sorted(EXPECTED_NODE_TYPES))
def test_generated_workflow_matches_committed_json(filename: str) -> None:
    generated = dict(build_workflows())[filename]
    committed = _load_workflow(WORKFLOW_DIRECTORY / filename)
    assert committed == generated


@pytest.mark.parametrize("filename", sorted(EXPECTED_NODE_TYPES))
def test_workflow_graph_is_complete_and_well_typed(filename: str) -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / filename)
    _assert_graph_integrity(workflow)
    node_types = {node["type"] for node in workflow["nodes"]}
    assert EXPECTED_NODE_TYPES[filename] <= node_types


def test_planning_workflow_has_no_model_loader() -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3_Planning_and_Prompt_Lint.json")
    node_types = {node["type"] for node in workflow["nodes"]}
    assert not node_types & {"UNETLoader", "CLIPLoader", "VAELoader"}


def test_av_tools_workflow_routes_all_stream_operations() -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / "MiniMax-H3_AV_Latent_Tools.json")
    links = _link_tuples(workflow)
    ids = {node["type"]: node["id"] for node in workflow["nodes"]}
    assert any(link[1:5] == (ids["EmptyMiniMaxH3LatentAV"], 0,
                              ids["MiniMaxH3XUpscaleVideoLatent"], 0) for link in links)
    assert any(link[1:5] == (ids["MiniMaxH3XJoinAV"], 0,
                              ids["MiniMaxH3XTemporalTrim"], 0) for link in links)
    assert any(link[1:5] == (ids["MiniMaxH3XTemporalTrim"], 0,
                              ids["MiniMaxH3XLatentInfo"], 0) for link in links)


def test_audio_reroll_uses_passthrough_video_and_new_audio() -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3_Audio_Reroll_Experimental.json")
    links = _link_tuples(workflow)
    reroll = _node_by_type(workflow, "MiniMaxH3XAudioReroll")[0]
    samplers = _node_by_type(workflow, "SamplerCustomAdvanced")
    video_decode = _node_by_type(workflow, "VAEDecode")[0]
    audio_decode = _node_by_type(workflow, "VAEDecodeAudio")[0]

    assert any(link[1] == reroll["id"] and link[2] == 2
               and link[3] == video_decode["id"] and link[4] == 0 for link in links)
    assert any(link[1] == reroll["id"] and link[2] == 1
               and link[3] in {node["id"] for node in samplers}
               and link[4] == 4 for link in links)
    assert any(link[1] in {node["id"] for node in samplers}
               and link[3] == audio_decode["id"] and link[4] == 0 for link in links)


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOW_DIRECTORY.glob("*.json")),
    ids=lambda path: path.name,
)
def test_workflow_titles_and_notes_are_bilingual(path: Path) -> None:
    workflow = _load_workflow(path)
    notes = _node_by_type(workflow, "MarkdownNote")
    assert notes, "%s must include at least one bilingual Markdown note" % path.name

    for node in workflow["nodes"]:
        title = node.get("title")
        assert isinstance(title, str), "%s node %d must have a bilingual title" % (
            path.name, node["id"])
        _assert_bilingual_text(title, "%s node %d title" % (path.name, node["id"]))

    for note in notes:
        values = note["widgets_values"]
        assert len(values) == 1 and isinstance(values[0], str), (
            "%s Markdown note %d must contain one text value" % (path.name, note["id"]))
        note_text = cast(str, values[0])
        assert "English" in note_text, "%s note %d must contain English" % (
            path.name, note["id"])
        assert "简体中文" in note_text, (
            "%s note %d must contain Simplified Chinese" % (path.name, note["id"]))


LOADER_TYPES = {"UNETLoader", "CLIPLoader", "VAELoader"}
MODEL_URL_PREFIX = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOW_DIRECTORY.glob("*.json")),
    ids=lambda path: path.name,
)
def test_loader_nodes_carry_a_verified_download_link(path: Path) -> None:
    workflow = _load_workflow(path)
    loaders = [node for node in workflow["nodes"] if node["type"] in LOADER_TYPES]
    for node in loaders:
        file_name = node["widgets_values"][0]
        models = node["properties"]["models"]
        assert models == [
            {
                "name": file_name,
                "url": "%s%s/%s" % (MODEL_URL_PREFIX, models[0]["directory"], file_name),
                "directory": models[0]["directory"],
            }
        ]
        assert models[0]["directory"] in {"diffusion_models", "text_encoders", "vae"}


def test_example_workflows_are_a_single_flat_folder() -> None:
    entries = list(WORKFLOW_DIRECTORY.iterdir())
    assert entries
    assert all(entry.is_file() and entry.suffix in {".json", ".jpg", ".png"} for entry in entries)
    assert not (REPOSITORY_ROOT / "workflows").exists()
