"""Validate V3 schema metadata that the ComfyUI frontend reads."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

NF4_NODES = tuple(get_nf4_nodes())
ALL_NODES = (
    MiniMaxH3XCanvasSolver,
    MiniMaxH3XPromptLint,
    MiniMaxH3XSplitAV,
    MiniMaxH3XJoinAV,
    MiniMaxH3XLatentInfo,
    MiniMaxH3XUpscaleVideoLatent,
    MiniMaxH3XTemporalTrim,
    MiniMaxH3XAudioReroll,
    *NF4_NODES,
)
EXPERIMENTAL_NODES = (MiniMaxH3XAudioReroll, *NF4_NODES)


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


@pytest.mark.parametrize("node", ALL_NODES, ids=lambda node: node.__name__)
def test_every_node_has_english_and_chinese_search_aliases(node) -> None:
    aliases = node.define_schema().search_aliases
    assert "h3" in aliases
    assert "minimax" in aliases
    assert any(_has_chinese(alias) for alias in aliases)
    assert len(aliases) == len(set(aliases))


@pytest.mark.parametrize("node", ALL_NODES, ids=lambda node: node.__name__)
def test_experimental_flag_matches_the_documented_status(node) -> None:
    assert node.define_schema().is_experimental == (node in EXPERIMENTAL_NODES)


@pytest.mark.parametrize("node", NF4_NODES, ids=lambda node: node.__name__)
def test_nf4_loader_inputs_say_which_file_and_folder_they_expect(node) -> None:
    inputs = node.define_schema().inputs
    assert inputs
    for loader_input in inputs:
        tooltip = loader_input.tooltip or ""
        assert "nf4.safetensors" in tooltip
        assert "models/" in tooltip
