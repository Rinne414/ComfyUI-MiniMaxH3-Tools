"""The NF4 VAE loader must not make core build its NF4 layers in host RAM.

Under dynamic VRAM, core builds the H3 VAE's Linear layers lazily and fills a
weight that is missing from the state dict with a full float32 zeros tensor
(comfy/ops.py _lazy_load_from_state_dict). The NF4 weights are exactly the
ones the loader holds back until it has swapped in its NF4 layers, so for the
36-block video VAE that was 9 GiB of zeros. This test builds core's real
comfy.sd.VAE (decoder only, one transformer block) and measures what the NF4
layers hold at the moment the loader swaps them.
"""

from __future__ import annotations

import json

import pytest
import torch

import nf4_cpu_support  # noqa: F401  (sets up a CPU-only ComfyUI first)

import comfy.memory_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import nf4_loader  # noqa: E402
from nf4 import NF4_QUANT_STATE_SUFFIX, with_placeholder_weights  # noqa: E402

DIM = 2048  # core's H3 video VAE decoder: 32 heads x 64
NF4_WEIGHTS = {
    "decoder.transformer_blocks.0.attn.to_qkv.weight": (DIM * 3, DIM),
    "decoder.transformer_blocks.0.attn.to_out.weight": (DIM, DIM),
    "decoder.transformer_blocks.0.ff.w1.weight": (DIM * 8, DIM),
    "decoder.transformer_blocks.0.ff.w2.weight": (DIM, DIM * 4),
}


class _Stop(Exception):
    """Ends the load once the NF4 layers are swapped; the rest needs BitsAndBytes."""


class _DynamicPatcher(comfy.model_patcher.ModelPatcher):
    """Core's VAE loads with assign=True only for a dynamic patcher."""

    def is_dynamic(self) -> bool:
        return True


def _quant_state(shape: tuple[int, int]) -> torch.Tensor:
    payload = json.dumps({"quant_type": "nf4", "shape": list(shape)}).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _nf4_video_vae_state_dict() -> dict[str, torch.Tensor]:
    state_dict = {
        "decoder.transformer_blocks.0.scale1": torch.ones(DIM),
        "decoder.x_embedder.weight": torch.empty(DIM, 24),
        "decoder.proj_out.weight": torch.empty(3 * 4 * 16 * 16, DIM),
    }
    for key, (rows, columns) in NF4_WEIGHTS.items():
        state_dict[key] = torch.empty((rows * columns // 2, 1), dtype=torch.uint8)
        state_dict[key + NF4_QUANT_STATE_SUFFIX] = _quant_state((rows, columns))
    return state_dict


def _load_until_the_nf4_swap() -> None:
    try:
        nf4_loader._load_nf4_vae("vae.safetensors", _nf4_video_vae_state_dict())
    except _Stop:
        return
    except RuntimeError as error:
        # Older cores (the weekly CI job's pinned one) only detect a full H3 VAE
        # and always build all 36 blocks, too large for a unit test.
        if "VAE is invalid" in str(error):
            pytest.skip("this ComfyUI does not build a decoder-only H3 video VAE")
        raise
    pytest.fail("the loader never reached the NF4 swap")


def test_placeholders_keep_logical_shapes_and_cost_no_memory() -> None:
    state_dict = _nf4_video_vae_state_dict()
    keys = set(state_dict)

    probe = with_placeholder_weights(state_dict, NF4_WEIGHTS)

    assert set(state_dict) == keys
    assert not any(key.endswith(NF4_QUANT_STATE_SUFFIX) for key in probe)
    for key, shape in NF4_WEIGHTS.items():
        assert tuple(probe[key].shape) == shape
        assert probe[key].dtype == torch.float32
        assert probe[key].untyped_storage().nbytes() <= 4
    assert probe["decoder.proj_out.weight"] is state_dict["decoder.proj_out.weight"]


def test_core_vae_build_does_not_allocate_the_nf4_layers(monkeypatch) -> None:
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", _DynamicPatcher)
    monkeypatch.setattr(nf4_loader, "_require_bitsandbytes", lambda: None)
    held_bytes = {}
    anchors = []
    replace = nf4_loader._replace_quantized_linears

    def swap_and_stop(root, shapes):
        modules = dict(root.named_modules())
        for key in shapes:
            weight = modules[key[: -len(".weight")]].weight
            held_bytes[key] = weight.untyped_storage().nbytes()
        assert replace(root, shapes) == len(shapes)
        modules = dict(root.named_modules())
        anchors.extend(modules[key[: -len(".weight")]].load_anchor for key in shapes)
        raise _Stop

    monkeypatch.setattr(nf4_loader, "_replace_quantized_linears", swap_and_stop)

    _load_until_the_nf4_swap()

    assert set(held_bytes) == set(NF4_WEIGHTS)
    assert max(held_bytes.values()) <= 4, f"core built the NF4 layers: {held_bytes}"
    # The NF4 layers get the same device and dtype as before the change.
    assert all(a.device.type == "cpu" and a.dtype == torch.float32 for a in anchors)


def test_core_vae_build_still_works_without_dynamic_vram(monkeypatch) -> None:
    """Without dynamic VRAM core builds every Linear up front and copies the
    placeholders into them; the loader must still reach the NF4 swap."""
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    monkeypatch.setattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )
    monkeypatch.setattr(nf4_loader, "_require_bitsandbytes", lambda: None)
    replace = nf4_loader._replace_quantized_linears
    swapped = []

    def swap_and_stop(root, shapes):
        swapped.append(replace(root, shapes))
        raise _Stop

    monkeypatch.setattr(nf4_loader, "_replace_quantized_linears", swap_and_stop)

    _load_until_the_nf4_swap()
    assert swapped == [len(NF4_WEIGHTS)]
