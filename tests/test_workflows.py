"""Validate the committed ComfyUI example workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.generate_example_workflows import (
    MODEL_DIRECTORIES,
    MODEL_SOURCES,
    model_url,
    Workflow,
    WorkflowNode,
    build_workflows,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / "example_workflows"

EXPECTED_NODE_TYPES = {
    "MiniMax-H3_Planning_and_Prompt_Lint.json": {
        "MarkdownNote",
        "MiniMaxH3XCanvasSolver",
        "MiniMaxH3XPromptLint",
        "PreviewAny",
    },
    "MiniMax-H3_AV_Latent_Tools.json": {
        "MarkdownNote",
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3XSplitAV",
        "MiniMaxH3XUpscaleVideoLatent",
        "MiniMaxH3XJoinAV",
        "MiniMaxH3XTemporalTrim",
        "MiniMaxH3XLatentInfo",
        "PreviewAny",
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
    assert "/" in text, (
        "%s must separate English and Simplified Chinese with '/'" % context
    )
    assert any(character.isascii() and character.isalpha() for character in text), (
        "%s must contain English text" % context
    )
    assert any("\u4e00" <= character <= "\u9fff" for character in text), (
        "%s must contain Simplified Chinese text" % context
    )


def _link_tuples(workflow: Workflow) -> set[tuple[int, int, int, int, int, str]]:
    parsed: set[tuple[int, int, int, int, int, str]] = set()
    for link in workflow["links"]:
        if len(link) != 6:
            raise ValueError("workflow link must have six fields, got %r" % link)
        link_id, source_id, source_slot, target_id, target_slot, link_type = link
        if not all(
            isinstance(value, int)
            for value in (link_id, source_id, source_slot, target_id, target_slot)
        ):
            raise TypeError("workflow link IDs and slots must be integers: %r" % link)
        if not isinstance(link_type, str):
            raise TypeError("workflow link type must be a string: %r" % link)
        parsed.add(
            (
                cast(int, link_id),
                cast(int, source_id),
                cast(int, source_slot),
                cast(int, target_id),
                cast(int, target_slot),
                link_type,
            )
        )
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
        if target["inputs"][target_slot]["type"] not in {link_type, "*"}:
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


IMAGE_NODE_TYPES = {
    "MiniMaxH3XImagePromptExpand",
    "MiniMaxH3XImageEncodeLowRAM",
    "MiniMaxH3XImageSize",
    "UNETLoader",
    "VAELoader",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "MarkdownNote",
}
# Edited by hand (converted from a ComfyUI export), not produced by the generator.
HAND_MADE_NODE_TYPES = {
    "MiniMax-H3-Image_T2I_Low_RAM.json": IMAGE_NODE_TYPES,
    "MiniMax-H3-Image_Edit_Low_RAM.json": IMAGE_NODE_TYPES | {"LoadImage"},
    "MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json": IMAGE_NODE_TYPES | {"LoadImage"},
}
ALL_NODE_TYPES = EXPECTED_NODE_TYPES | HAND_MADE_NODE_TYPES


@pytest.mark.parametrize("filename", sorted(ALL_NODE_TYPES))
def test_workflow_graph_is_complete_and_well_typed(filename: str) -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / filename)
    _assert_graph_integrity(workflow)
    node_types = {node["type"] for node in workflow["nodes"]}
    assert ALL_NODE_TYPES[filename] <= node_types


def test_planning_workflow_has_no_model_loader() -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3_Planning_and_Prompt_Lint.json"
    )
    node_types = {node["type"] for node in workflow["nodes"]}
    assert not node_types & {"UNETLoader", "CLIPLoader", "VAELoader"}


def test_av_tools_workflow_routes_all_stream_operations() -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / "MiniMax-H3_AV_Latent_Tools.json")
    links = _link_tuples(workflow)
    ids = {node["type"]: node["id"] for node in workflow["nodes"]}
    assert any(
        link[1:5]
        == (ids["EmptyMiniMaxH3LatentAV"], 0, ids["MiniMaxH3XUpscaleVideoLatent"], 0)
        for link in links
    )
    assert any(
        link[1:5] == (ids["MiniMaxH3XJoinAV"], 0, ids["MiniMaxH3XTemporalTrim"], 0)
        for link in links
    )
    assert any(
        link[1:5] == (ids["MiniMaxH3XTemporalTrim"], 0, ids["MiniMaxH3XLatentInfo"], 0)
        for link in links
    )


def test_audio_reroll_uses_passthrough_video_and_new_audio() -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3_Audio_Reroll_Experimental.json"
    )
    links = _link_tuples(workflow)
    reroll = _node_by_type(workflow, "MiniMaxH3XAudioReroll")[0]
    samplers = _node_by_type(workflow, "SamplerCustomAdvanced")
    video_decode = _node_by_type(workflow, "VAEDecode")[0]
    audio_decode = _node_by_type(workflow, "VAEDecodeAudio")[0]

    assert any(
        link[1] == reroll["id"]
        and link[2] == 2
        and link[3] == video_decode["id"]
        and link[4] == 0
        for link in links
    )
    assert any(
        link[1] == reroll["id"]
        and link[2] == 1
        and link[3] in {node["id"] for node in samplers}
        and link[4] == 4
        for link in links
    )
    assert any(
        link[1] in {node["id"] for node in samplers}
        and link[3] == audio_decode["id"]
        and link[4] == 0
        for link in links
    )


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
            path.name,
            node["id"],
        )
        _assert_bilingual_text(title, "%s node %d title" % (path.name, node["id"]))

    for note in notes:
        values = note["widgets_values"]
        assert len(values) == 1 and isinstance(values[0], str), (
            "%s Markdown note %d must contain one text value" % (path.name, note["id"])
        )
        note_text = cast(str, values[0])
        assert "English" in note_text, "%s note %d must contain English" % (
            path.name,
            note["id"],
        )
        assert "简体中文" in note_text, "%s note %d must contain Simplified Chinese" % (
            path.name,
            note["id"],
        )


LOADER_TYPES = {"UNETLoader", "CLIPLoader", "VAELoader"}
MODEL_NODE_WIDGET = {
    "UNETLoader": 0,
    "CLIPLoader": 0,
    "VAELoader": 0,
    "MiniMaxH3XImageEncodeLowRAM": 0,
    "MiniMaxH3XImagePromptExpand": 2,
}

# Every URL was checked against the repository's file listing and answered with a
# redirect to the file (HTTP 302) on 2026-11-11.
VERIFIED_MODEL_URLS = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "minimax_h3_video_vae_fp16.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_image_pruned_w4a8.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
    "minimax_h3_image_vae_fp16.safetensors": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
    "qwen3.5_9b_w4a8_gptq.safetensors": "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors",
}


def test_model_urls_match_the_verified_huggingface_layout() -> None:
    assert {name: model_url(name) for name in MODEL_SOURCES} == VERIFIED_MODEL_URLS
    assert {name: MODEL_DIRECTORIES[name] for name in MODEL_SOURCES} == {
        name: url.split("/resolve/main/")[1].split("/")[0]
        for name, url in VERIFIED_MODEL_URLS.items()
    }


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOW_DIRECTORY.glob("*.json")),
    ids=lambda path: path.name,
)
def test_model_nodes_carry_a_verified_download_link(path: Path) -> None:
    workflow = _load_workflow(path)
    nodes = [node for node in workflow["nodes"] if node["type"] in MODEL_NODE_WIDGET]
    for node in nodes:
        file_name = node["widgets_values"][MODEL_NODE_WIDGET[node["type"]]]
        assert file_name in VERIFIED_MODEL_URLS, (
            "%s uses an unverified model file" % file_name
        )
        directory = (
            VERIFIED_MODEL_URLS[file_name].split("/resolve/main/")[1].split("/")[0]
        )
        assert node["properties"]["models"] == [
            {
                "name": file_name,
                "url": VERIFIED_MODEL_URLS[file_name],
                "directory": directory,
            }
        ]


def test_the_research_generator_keeps_the_download_links() -> None:
    """Re-running research/gen_h3_workflow.py must not strip them from AV_Advanced."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_h3_workflow", REPOSITORY_ROOT / "research" / "gen_h3_workflow.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes = module.build()[0]
    generated = {
        node["id"]: node["properties"].get("models")
        for node in nodes
        if node["type"] in LOADER_TYPES
    }
    committed = {
        node["id"]: node["properties"].get("models")
        for node in _load_workflow(WORKFLOW_DIRECTORY / "MiniMax-H3_AV_Advanced.json")[
            "nodes"
        ]
        if node["type"] in LOADER_TYPES
    }
    assert generated and all(generated.values())
    assert generated == committed


def test_example_workflows_are_a_single_flat_folder() -> None:
    entries = list(WORKFLOW_DIRECTORY.iterdir())
    assert entries
    assert all(
        entry.is_file() and entry.suffix in {".json", ".jpg", ".png"}
        for entry in entries
    )
    assert not (REPOSITORY_ROOT / "workflows").exists()


OUTPUT_NODE_TYPES = {"PreviewAny", "SaveVideo", "SaveAudio", "SaveImage"}


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOW_DIRECTORY.glob("*.json")),
    ids=lambda path: path.name,
)
def test_every_example_has_an_output_node_so_run_does_not_fail(path: Path) -> None:
    """ComfyUI refuses a prompt without output nodes ("Prompt has no outputs")."""
    workflow = _load_workflow(path)
    assert {node["type"] for node in workflow["nodes"]} & OUTPUT_NODE_TYPES


def test_planning_previews_show_the_planner_report_and_lint_warnings() -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3_Planning_and_Prompt_Lint.json"
    )
    links = _link_tuples(workflow)
    planner = _node_by_type(workflow, "MiniMaxH3XCanvasSolver")[0]
    lint = _node_by_type(workflow, "MiniMaxH3XPromptLint")[0]
    previews = {node["id"] for node in _node_by_type(workflow, "PreviewAny")}
    assert any(
        link[1] == planner["id"]
        and link[2] == 8
        and link[3] in previews
        and link[5] == "STRING"
        for link in links
    )
    assert any(
        link[1] == lint["id"]
        and link[2] == 1
        and link[3] in previews
        and link[5] == "STRING"
        for link in links
    )


def test_av_tools_preview_shows_the_latent_info_report() -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / "MiniMax-H3_AV_Latent_Tools.json")
    links = _link_tuples(workflow)
    info = _node_by_type(workflow, "MiniMaxH3XLatentInfo")[0]
    preview = _node_by_type(workflow, "PreviewAny")[0]
    assert any(
        link[1] == info["id"] and link[2] == 0 and link[3] == preview["id"]
        for link in links
    )


def _picture_links(workflow: Workflow, target_type: str) -> dict[str, int]:
    """Map an encoder/expander picture input name to the LoadImage node feeding it."""
    nodes = {node["id"]: node for node in workflow["nodes"]}
    target = _node_by_type(workflow, target_type)[0]
    found: dict[str, int] = {}
    for link in workflow["links"]:
        if link[3] != target["id"]:
            continue
        slot = target["inputs"][link[4]]
        if slot["name"].startswith("images.") and nodes[link[1]]["type"] == "LoadImage":
            found[slot["name"]] = link[1]
    return found


@pytest.mark.parametrize(
    "target", ["MiniMaxH3XImageEncodeLowRAM", "MiniMaxH3XImagePromptExpand"]
)
def test_edit_workflow_links_its_picture_to_the_first_image_input(target: str) -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / "MiniMax-H3-Image_Edit_Low_RAM.json")
    loader = _node_by_type(workflow, "LoadImage")[0]
    assert _picture_links(workflow, target) == {"images.image_1": loader["id"]}


@pytest.mark.parametrize(
    "target", ["MiniMaxH3XImageEncodeLowRAM", "MiniMaxH3XImagePromptExpand"]
)
def test_two_picture_workflow_links_both_pictures(target: str) -> None:
    workflow = _load_workflow(
        WORKFLOW_DIRECTORY / "MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json"
    )
    loaders = sorted(node["id"] for node in _node_by_type(workflow, "LoadImage"))
    assert len(loaders) == 2
    links = _picture_links(workflow, target)
    assert sorted(links) == ["images.image_1", "images.image_2"]
    assert sorted(links.values()) == loaders
    # the canvas follows the first picture only
    size = _node_by_type(workflow, "MiniMaxH3XImageSize")[0]
    size_sources = {
        link[1]
        for link in workflow["links"]
        if link[3] == size["id"] and size["inputs"][link[4]]["name"] == "image"
    }
    assert size_sources == {links["images.image_1"]}


@pytest.mark.parametrize(
    "filename,expands",
    [
        ("MiniMax-H3-Image_T2I_Low_RAM.json", True),
        ("MiniMax-H3-Image_Edit_Low_RAM.json", False),
        ("MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json", False),
    ],
)
def test_expander_is_on_for_text_to_image_and_off_for_edits(
    filename: str, expands: bool
) -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / filename)
    expander = _node_by_type(workflow, "MiniMaxH3XImagePromptExpand")[0]
    assert expander["widgets_values"][1] is expands


def test_example_folder_holds_no_api_exports_or_personal_references() -> None:
    assert not list(WORKFLOW_DIRECTORY.glob("*.api.json"))
    for path in WORKFLOW_DIRECTORY.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "h3img_",
            "I:\\\\",
            "C:\\\\Users",
            "/Users/",
            "output/conditioning",
        ):
            assert forbidden not in text, (path.name, forbidden)


@pytest.mark.parametrize(
    "filename",
    [
        "MiniMax-H3-Image_Edit_Low_RAM.json",
        "MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json",
    ],
)
def test_edit_examples_load_the_stock_example_picture(filename: str) -> None:
    workflow = _load_workflow(WORKFLOW_DIRECTORY / filename)
    for loader in _node_by_type(workflow, "LoadImage"):
        assert loader["widgets_values"][0] == "example.png"


def test_thumbnails_sit_next_to_their_workflow_and_stay_small() -> None:
    """ComfyUI's Templates browser looks for <workflow name>.jpg beside the .json."""
    thumbnails = sorted(WORKFLOW_DIRECTORY.glob("*.jpg"))
    assert thumbnails
    for path in thumbnails:
        assert path.with_suffix(".json").is_file(), path.name
        data = path.read_bytes()
        assert data[:3] == b"\xff\xd8\xff", "%s is not a JPEG" % path.name
        assert len(data) < 150_000, path.name
        assert b"Exif" not in data[:64], "%s carries EXIF metadata" % path.name
