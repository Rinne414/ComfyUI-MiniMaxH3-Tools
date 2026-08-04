"""Tests for MiniMax-H3 geometry planning."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import NO_PROFILE  # noqa: E402
from canvas_solver import (  # noqa: E402
    ASPECT_RATIOS,
    MAX_PIXELS,
    MP_UNIT,
    UNAVAILABLE_ESTIMATE,
    MiniMaxH3XCanvasSolver,
    align_frame_count,
    solve_canvas,
    video_latent_t,
)

# Expected ResolutionSelector-compatible geometry. Hardware does not affect it.
EXPECTED_GEOMETRY = [
    (0.4, 5, 864, 480, 124, 37, 14985),
    (0.6, 5, 1056, 608, 124, 37, 23199),
    (0.8, 5, 1216, 672, 124, 37, 29526),
    (1.0, 5, 1376, 768, 124, 37, 38184),
    (0.4, 10, 864, 480, 243, 72, 29160),
    (0.6, 10, 1056, 608, 243, 72, 45144),
    (0.8, 10, 1216, 672, 243, 72, 57456),
    (1.0, 10, 1376, 768, 243, 72, 74304),
]


@pytest.mark.parametrize("mp,secs,w,h,frames,lt,tokens", EXPECTED_GEOMETRY)
def test_solver_reproduces_expected_geometry(mp, secs, w, h, frames, lt, tokens):
    got_w, got_h, _ = solve_canvas("16:9 (Widescreen)", mp, False)
    assert (got_w, got_h) == (w, h)
    got_frames = align_frame_count(round(secs * 24))
    assert got_frames == frames
    assert video_latent_t(got_frames) == lt
    assert lt * (h // 32) * (w // 32) == tokens


def test_one_mp_16x9_really_is_over_the_trained_cap():
    """The measured 1.0 MP row is 1376x768 = 1,056,768 px, over the 1,032,192
    cap. H3's generation path never calls adapt_canvas(), so it renders anyway.
    The solver must report this honestly instead of substituting a legal canvas.
    """
    w, h, over_cap = solve_canvas("16:9 (Widescreen)", 1.0, False)
    assert (w, h) == (1376, 768)
    assert w * h > MAX_PIXELS
    assert over_cap

    clamped_w, clamped_h, still_over = solve_canvas(
        "16:9 (Widescreen)", 1.0, True)
    assert clamped_w * clamped_h <= MAX_PIXELS
    assert still_over, "over_cap stays True to record that clamping happened"
    assert (clamped_w, clamped_h) != (w, h)


def test_frame_grid_is_17k_plus_5():
    for n in range(1, 400):
        aligned = align_frame_count(n)
        assert aligned % 17 == 5
        assert aligned >= n or n < 5
        assert aligned - n < 17


def test_frame_grid_is_idempotent():
    for n in (5, 22, 124, 243, 362):
        assert align_frame_count(n) == n


def test_latent_t_matches_core_formula():
    # mirrors comfy_extras/nodes_minimax_h3.py video_latent_t()
    for frames in (5, 22, 39, 124, 243, 362):
        expected = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        assert video_latent_t(frames) == expected


def test_area_cap_is_enforced_when_requested():
    for ratio in ASPECT_RATIOS:
        w, h, over_cap = solve_canvas(ratio, 1.4, True)
        assert w * h <= MAX_PIXELS, "%s produced %dx%d over the cap" % (ratio, w, h)
        assert over_cap


def test_over_cap_is_reported_not_silently_fixed():
    w, h, over_cap = solve_canvas("16:9 (Widescreen)", 1.4, False)
    assert over_cap
    assert w * h > MAX_PIXELS, "unclamped mode must return the real geometry"


def test_every_ratio_lands_on_the_32_grid():
    for ratio in ASPECT_RATIOS:
        for mp in (0.1, 0.4, 0.7, 1.0, 1.2):
            for clamp in (False, True):
                w, h, _ = solve_canvas(ratio, mp, clamp)
                assert w % 32 == 0 and h % 32 == 0, (
                    "%s @ %s clamp=%s -> %dx%d" % (ratio, mp, clamp, w, h))
                assert w >= 32 and h >= 32


def test_portrait_ratios_are_taller_than_wide():
    for ratio in ("9:16 (Portrait Widescreen)", "3:4 (Portrait Standard)",
                  "2:3 (Portrait Photo)"):
        w, h, _ = solve_canvas(ratio, 0.4, False)
        assert h > w, "%s gave %dx%d" % (ratio, w, h)


def test_small_requests_are_within_the_cap():
    w, h, over_cap = solve_canvas("16:9 (Widescreen)", 0.8, False)
    assert not over_cap
    assert w * h <= MAX_PIXELS


def test_one_megapixel_unit_matches_resolution_selector():
    # ResolutionSelector uses 1024*1024, not 1e6. Getting this wrong shifts
    # every predicted canvas by ~5%.
    assert MP_UNIT == 1048576


def test_runtime_estimate_is_disabled_by_default():
    output = MiniMaxH3XCanvasSolver.execute(
        "16:9 (Widescreen)",
        0.4,
        5.0,
        20,
        NO_PROFILE,
        24.0,
        False,
    ).result
    assert output[6] == UNAVAILABLE_ESTIMATE
    assert output[7] == UNAVAILABLE_ESTIMATE
    assert "not estimated" in output[8]


def test_runtime_estimate_requires_an_explicit_profile():
    output = MiniMaxH3XCanvasSolver.execute(
        "16:9 (Widescreen)",
        0.4,
        5.0,
        20,
        "rtx-3090-int8-convrot-xformers",
        24.0,
        False,
    ).result
    assert output[6] > 0.0
    assert output[7] > 0.0
    assert "Single RTX 3090" in output[8]
