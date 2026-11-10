"""No-pin context restore behaviour and closest aspect ratio (no ComfyUI needed)."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_pinning as pinning  # noqa: E402
import image_sizes as sizes  # noqa: E402


@pytest.mark.parametrize("initial", [False, True])
def test_pinning_is_off_inside_and_restored_after(initial):
    mm = types.SimpleNamespace(DISABLE_PINNED_MEMORY=initial)
    with pinning.pinning_disabled(mm):
        assert mm.DISABLE_PINNED_MEMORY is True
    assert mm.DISABLE_PINNED_MEMORY is initial


def test_pinning_is_restored_when_the_encode_raises():
    mm = types.SimpleNamespace(DISABLE_PINNED_MEMORY=False)
    with pytest.raises(RuntimeError, match="encode failed"):
        with pinning.pinning_disabled(mm):
            raise RuntimeError("encode failed")
    assert mm.DISABLE_PINNED_MEMORY is False


@pytest.mark.parametrize(
    "width,height,ratio",
    [
        (1024, 1024, "1:1"),
        (1920, 1080, "16:9"),
        (1080, 1920, "9:16"),
        (4000, 3000, "4:3"),
        (3000, 2000, "3:2"),
        (2560, 1080, "21:9"),
        (832, 1216, "2:3"),
        (1000, 990, "1:1"),
    ],
)
def test_closest_ratio_picks_the_nearest_preset(width, height, ratio):
    assert sizes.closest_ratio(width, height) == ratio


def test_closest_ratio_rejects_empty_pictures():
    with pytest.raises(ValueError):
        sizes.closest_ratio(0, 512)
