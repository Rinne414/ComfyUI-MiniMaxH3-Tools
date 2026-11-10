"""Validate the English schema text and the en/zh locale files that translate it."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from audio_reroll import MiniMaxH3XAudioReroll  # noqa: E402
from av_latent import (  # noqa: E402
    MiniMaxH3XJoinAV,
    MiniMaxH3XLatentInfo,
    MiniMaxH3XSplitAV,
    MiniMaxH3XTemporalTrim,
    MiniMaxH3XUpscaleVideoLatent,
)
from canvas_solver import MiniMaxH3XCanvasSolver  # noqa: E402
from nf4_loader import get_nf4_nodes  # noqa: E402
from prompt_lint import MiniMaxH3XPromptLint  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")

EXPECTED_DISPLAY_NAMES = {
    "MiniMaxH3XCanvasSolver": "MiniMax H3 Canvas Planner",
    "MiniMaxH3XPromptLint": "MiniMax H3 Prompt Lint",
    "MiniMaxH3XSplitAV": "MiniMax H3 Split AV Latent",
    "MiniMaxH3XJoinAV": "MiniMax H3 Join AV Latent",
    "MiniMaxH3XLatentInfo": "MiniMax H3 AV Latent Info",
    "MiniMaxH3XUpscaleVideoLatent": "MiniMax H3 Upscale Video Latent",
    "MiniMaxH3XTemporalTrim": "MiniMax H3 Temporal Trim",
    "MiniMaxH3XAudioReroll": "MiniMax H3 Audio Reroll (Experimental)",
    "MiniMaxH3XNF4ModelLoader": "MiniMax H3 NF4 Model Loader",
    "MiniMaxH3XNF4TextEncoderLoader": "MiniMax H3 NF4 Text Encoder Loader",
    "MiniMaxH3XNF4VideoVAELoader": "MiniMax H3 NF4 Video VAE Loader",
    "MiniMaxH3XNF4AudioVAELoader": "MiniMax H3 NF4 Audio VAE Loader",
}

NODES = (
    MiniMaxH3XCanvasSolver,
    MiniMaxH3XPromptLint,
    MiniMaxH3XSplitAV,
    MiniMaxH3XJoinAV,
    MiniMaxH3XLatentInfo,
    MiniMaxH3XUpscaleVideoLatent,
    MiniMaxH3XTemporalTrim,
    MiniMaxH3XAudioReroll,
    *get_nf4_nodes(),
)
SCHEMAS = {node.define_schema().node_id: node.define_schema() for node in NODES}

NODE_FILES = (
    REPOSITORY_ROOT / "canvas_solver.py",
    REPOSITORY_ROOT / "prompt_lint.py",
    REPOSITORY_ROOT / "av_latent.py",
    REPOSITORY_ROOT / "audio_reroll.py",
    REPOSITORY_ROOT / "nf4_loader.py",
)
LOCALES_DIRECTORY = REPOSITORY_ROOT / "locales"
LANGUAGES = ("en", "zh")


def _load_locale(language: str) -> dict:
    path = LOCALES_DIRECTORY / language / "nodeDefs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _key_structure(value: object) -> object:
    if isinstance(value, dict):
        return {key: _key_structure(child) for key, child in value.items()}
    return None


def _output_names(schema) -> list[str | None]:
    return [output.display_name for output in schema.outputs]


def _schema_text_literals(path: Path) -> list[str]:
    """Return every display_name, description and tooltip literal in a node file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"display_name", "description", "tooltip"}:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                literals.append(value.value)
    return literals


def test_schema_text_is_english_only() -> None:
    for path in NODE_FILES:
        literals = _schema_text_literals(path)
        assert literals, "%s must define schema text" % path.name
        for text in literals:
            assert not CJK.search(text), "%s: %r" % (path.name, text)
            assert " / " not in text, "%s still looks bilingual: %r" % (path.name, text)


def test_display_names_match_the_expected_english_names() -> None:
    assert {
        node_id: schema.display_name for node_id, schema in SCHEMAS.items()
    } == EXPECTED_DISPLAY_NAMES


def test_locale_files_exist_for_every_language() -> None:
    for language in LANGUAGES:
        assert (LOCALES_DIRECTORY / language / "nodeDefs.json").is_file()


def test_en_and_zh_have_identical_key_structure() -> None:
    assert _key_structure(_load_locale("en")) == _key_structure(_load_locale("zh"))


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_registered_node_has_a_locale_entry(language: str) -> None:
    assert set(_load_locale(language)) == set(SCHEMAS)


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_input_and_output_has_a_name(language: str) -> None:
    locale = _load_locale(language)
    for node_id, schema in SCHEMAS.items():
        entry = locale[node_id]
        assert entry["display_name"].strip()
        assert entry["description"].strip()
        for node_input in schema.inputs:
            assert entry["inputs"][node_input.id]["name"].strip(), (node_id, node_input.id)
        assert set(entry["inputs"]) == {node_input.id for node_input in schema.inputs}
        assert set(entry["outputs"]) == {str(index) for index in range(len(schema.outputs))}
        for index in range(len(schema.outputs)):
            assert entry["outputs"][str(index)]["name"].strip(), (node_id, index)


def test_zh_text_is_chinese_and_en_text_is_not() -> None:
    en, zh = _load_locale("en"), _load_locale("zh")
    for node_id in SCHEMAS:
        assert not CJK.search(json.dumps(en[node_id], ensure_ascii=False)), node_id
        assert CJK.search(zh[node_id]["display_name"]), node_id
        assert CJK.search(zh[node_id]["description"]), node_id
        for input_id, entry in zh[node_id]["inputs"].items():
            assert CJK.search(entry["name"]), (node_id, input_id)
            if "tooltip" in entry:
                assert CJK.search(entry["tooltip"]), (node_id, input_id)


def test_en_locale_matches_the_schema_text() -> None:
    """The English file must not drift from what the schema itself declares."""
    en = _load_locale("en")
    for node_id, schema in SCHEMAS.items():
        entry = en[node_id]
        assert entry["display_name"] == schema.display_name
        assert entry["description"] == schema.description
        for node_input in schema.inputs:
            tooltip = getattr(node_input, "tooltip", None)
            assert entry["inputs"][node_input.id].get("tooltip") == tooltip, (
                node_id,
                node_input.id,
            )


def test_every_tooltip_in_a_schema_is_translated() -> None:
    zh = _load_locale("zh")
    for node_id, schema in SCHEMAS.items():
        for node_input in schema.inputs:
            if getattr(node_input, "tooltip", None):
                assert "tooltip" in zh[node_id]["inputs"][node_input.id], (
                    node_id,
                    node_input.id,
                )


def test_locale_files_are_utf8_without_byte_order_mark() -> None:
    for language in LANGUAGES:
        raw = (LOCALES_DIRECTORY / language / "nodeDefs.json").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
