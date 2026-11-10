"""Size presets, file naming and zero conditioning (no ComfyUI needed)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conditioning_io as cond_io  # noqa: E402
import image_sizes as sizes  # noqa: E402


@pytest.mark.parametrize("ratio,size", sorted(sizes.OFFICIAL_16MP.items()))
def test_16mp_matches_model_card_table(ratio, size):
    assert sizes.canvas_size(ratio, 16) == size


@pytest.mark.parametrize("ratio", list(sizes.ASPECT_RATIOS))
@pytest.mark.parametrize("mp", list(sizes.MEGAPIXELS.values()))
def test_every_preset_is_a_multiple_of_32_near_the_area(ratio, mp):
    w, h = sizes.canvas_size(ratio, mp)
    assert w % 32 == 0 and h % 32 == 0
    assert abs(w * h / (mp * 1024 * 1024) - 1) < 0.03


def test_latent_shape_is_one_frame_24_channels_16x():
    assert sizes.latent_shape(4736, 3552) == [1, 24, 1, 222, 296]
    with pytest.raises(ValueError):
        sizes.latent_shape(1000, 1024)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        sizes.canvas_size("7:5", 16)
    with pytest.raises(ValueError):
        sizes.canvas_size("1:1", 0)


def test_clean_name_and_free_path(tmp_path):
    assert cond_io.clean_name(" my prompt/../x.safetensors ") == "my_prompt_.._x"
    assert cond_io.clean_name("...") == "conditioning"
    first = cond_io.free_path(str(tmp_path), "a")
    open(first, "wb").close()
    assert os.path.basename(cond_io.free_path(str(tmp_path), "a")) == "a_2.safetensors"


def test_zero_conditioning_keeps_tags_and_zeroes_embeddings():
    tags = torch.tensor([[1, 0, 1]])
    cond = [
        [
            torch.ones(1, 3, 4),
            {"minimax_token_tags": tags, "pooled_output": torch.ones(2)},
        ]
    ]
    ((zeros, extra),) = cond_io.zero_conditioning(cond)
    assert not zeros.any() and not extra["pooled_output"].any()
    assert extra["minimax_token_tags"] is tags
    assert cond[0][0].all(), "the input conditioning must not be modified"
