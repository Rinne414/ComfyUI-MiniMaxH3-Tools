"""Validate bilingual user-facing custom-node metadata."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DISPLAY_NAMES = {
    "MiniMaxH3XCanvasSolver": "MiniMax H3 Canvas Planner / 画布规划",
    "MiniMaxH3XPromptLint": "MiniMax H3 Prompt Lint / 提示词检查",
    "MiniMaxH3XSplitAV": "MiniMax H3 Split AV Latent / 拆分音视频潜空间",
    "MiniMaxH3XJoinAV": "MiniMax H3 Join AV Latent / 合并音视频潜空间",
    "MiniMaxH3XLatentInfo": "MiniMax H3 AV Latent Info / 音视频潜空间信息",
    "MiniMaxH3XUpscaleVideoLatent": "MiniMax H3 Upscale Video Latent / 放大视频潜空间",
    "MiniMaxH3XTemporalTrim": "MiniMax H3 Temporal Trim / 时长裁剪",
    "MiniMaxH3XAudioReroll": (
        "MiniMax H3 Audio Reroll (Experimental) / 音频重生成（实验性）"),
}

NODE_FILES = (
    REPOSITORY_ROOT / "canvas_solver.py",
    REPOSITORY_ROOT / "prompt_lint.py",
    REPOSITORY_ROOT / "av_latent.py",
    REPOSITORY_ROOT / "audio_reroll.py",
)


def _string_keyword(call: ast.Call, name: str, path: Path) -> str:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            raise TypeError("%s Schema.%s must be a string literal" % (path, name))
    raise ValueError("%s Schema is missing '%s'" % (path, name))


def _schema_metadata(path: Path) -> dict[str, tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    metadata: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "Schema":
            continue
        node_id = _string_keyword(node, "node_id", path)
        display_name = _string_keyword(node, "display_name", path)
        description = _string_keyword(node, "description", path)
        metadata[node_id] = (display_name, description)
    return metadata


@pytest.mark.parametrize("path", NODE_FILES, ids=lambda path: path.name)
def test_custom_node_metadata_is_bilingual(path: Path) -> None:
    metadata = _schema_metadata(path)
    assert metadata, "%s must define at least one node schema" % path
    for node_id, (display_name, description) in metadata.items():
        assert display_name == EXPECTED_DISPLAY_NAMES[node_id]
        assert " / " in description
        assert any("\u4e00" <= character <= "\u9fff" for character in description)


def test_all_expected_custom_nodes_have_bilingual_metadata() -> None:
    discovered = {
        node_id
        for path in NODE_FILES
        for node_id in _schema_metadata(path)
    }
    assert discovered == set(EXPECTED_DISPLAY_NAMES)
