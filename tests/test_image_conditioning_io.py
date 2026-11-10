"""Round-trip tests for conditioning_io (encoded prompts saved as safetensors, no pickle)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conditioning_io as cond_io  # noqa: E402


def _h3_like_conditioning():
    embeds = torch.randn(1, 7, 16, dtype=torch.bfloat16)
    tags = torch.tensor([[1, 1, 0, 0, 1, 1, 1]], dtype=torch.long)
    ref = {
        "kind": "image",
        "latent_h": 64,
        "latent_w": 48,
        "latent": torch.randn(1, 24, 1, 64, 48),
    }
    extra = {
        "pooled_output": None,
        "minimax_token_tags": tags,
        "minimax_refs": [ref],
        "strength": 1.0,
        "__tensor__": "a stored key that looks like a marker",
    }
    return [[embeds, extra]]


def test_round_trip_keeps_tensors_and_structure(tmp_path):
    cond = _h3_like_conditioning()
    path = os.path.join(tmp_path, "c.safetensors")
    cond_io.save_conditioning(
        path, cond, {"prompt": "a fox", "text_encoder": "te.safetensors"}
    )
    loaded, info = cond_io.load_conditioning(path)

    assert info == {"prompt": "a fox", "text_encoder": "te.safetensors"}
    ((embeds, extra),) = loaded
    assert embeds.dtype == torch.bfloat16 and torch.equal(embeds, cond[0][0])
    assert torch.equal(extra["minimax_token_tags"], cond[0][1]["minimax_token_tags"])
    ref = extra["minimax_refs"][0]
    assert ref["kind"] == "image" and ref["latent_h"] == 64 and ref["latent_w"] == 48
    assert torch.equal(ref["latent"], cond[0][1]["minimax_refs"][0]["latent"])
    assert extra["pooled_output"] is None and extra["strength"] == 1.0
    assert extra["__tensor__"] == "a stored key that looks like a marker"


def test_shared_and_view_tensors_are_saved(tmp_path):
    base = torch.arange(12.0).reshape(3, 4)
    cond = [[base, {"same": base, "view": base[1:]}]]
    path = os.path.join(tmp_path, "c.safetensors")
    cond_io.save_conditioning(path, cond)
    ((t, extra),) = cond_io.load_conditioning(path)[0]
    assert (
        torch.equal(t, base)
        and torch.equal(extra["same"], base)
        and torch.equal(extra["view"], base[1:])
    )


def test_rejects_objects_it_cannot_store(tmp_path):
    with pytest.raises(TypeError):
        cond_io.save_conditioning(
            os.path.join(tmp_path, "c.safetensors"), [[torch.zeros(1), {"x": object()}]]
        )


def test_rejects_other_safetensors_files(tmp_path):
    from safetensors.torch import save_file

    path = os.path.join(tmp_path, "model.safetensors")
    save_file({"w": torch.zeros(2)}, path)
    with pytest.raises(ValueError):
        cond_io.load_conditioning(path)
