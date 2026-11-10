"""The report nodes show their text on the node as well as returning it."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from av_latent import MiniMaxH3XLatentInfo  # noqa: E402
from calibration import NO_PROFILE  # noqa: E402
from canvas_solver import MiniMaxH3XCanvasSolver  # noqa: E402
from prompt_lint import MiniMaxH3XPromptLint  # noqa: E402
from test_av_latent import make_latent  # noqa: E402


def _preview_text(output) -> str:
    payload = output.ui.as_dict()
    assert list(payload) == ["text"]
    (text,) = payload["text"]
    return text


def test_canvas_planner_shows_its_report_on_the_node() -> None:
    output = MiniMaxH3XCanvasSolver.execute(
        "16:9 (Widescreen)", 0.4, 5.0, 20, NO_PROFILE, 24.0, False
    )
    assert _preview_text(output) == output.result[8]


def test_prompt_lint_shows_its_warnings_on_the_node() -> None:
    output = MiniMaxH3XPromptLint.execute("a short prompt", "ref2va", 0.0, 0, 0, 0)
    assert output.result[1] != "clean -- no issues found"
    assert _preview_text(output) == output.result[1]


def test_prompt_lint_shows_a_clean_result_on_the_node() -> None:
    output = MiniMaxH3XPromptLint.execute("", "t2va", 0.0, 0, 0, 0)
    assert _preview_text(output) == output.result[1]


def test_av_latent_info_shows_its_report_on_the_node() -> None:
    output = MiniMaxH3XLatentInfo.execute(make_latent())
    assert _preview_text(output) == output.result[0]
