"""Core rebuilds a loader's patcher through cached_patcher_init; the NF4 loaders
must accept every call core makes.

ModelPatcher.clone(disable_dynamic=True) calls factory(*args, disable_dynamic=True)
(TorchCompile, hooks, multi-GPU), and deepclone_multigpu calls factory(*args).
The tests drive core's real clone(); only file reading and model construction
below the loaders are replaced.
"""

from __future__ import annotations

import json

import pytest
import torch

import nf4_cpu_support  # noqa: F401  (sets up a CPU-only ComfyUI first)

import comfy.ldm.minimax.model  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
import nf4_loader  # noqa: E402
from nf4 import NF4_QUANT_STATE_SUFFIX  # noqa: E402

CPU = torch.device("cpu")


class _DynamicPatcher(comfy.model_patcher.ModelPatcher):
    """Stands in for ModelPatcherDynamic: clone() only asks is_dynamic()."""

    def is_dynamic(self) -> bool:
        return True


def _quant_state(shape: tuple[int, int]) -> torch.Tensor:
    payload = json.dumps({"quant_type": "nf4", "shape": list(shape)}).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _nf4_state_dict(weight_key: str, shape: tuple[int, int]) -> dict:
    return {
        weight_key: torch.zeros((shape[0] * shape[1] // 2, 1), dtype=torch.uint8),
        weight_key + NF4_QUANT_STATE_SUFFIX: _quant_state(shape),
    }


class _DiTModel(torch.nn.Module):
    def __init__(self, operations) -> None:
        super().__init__()
        attention = comfy.ldm.minimax.model.Attention(
            64, 4, 16, 1e-5, operations=operations
        )
        self.diffusion_model = torch.nn.Sequential(attention)
        self.assigned = None

    def load_model_weights(self, state_dict, prefix, assign=False) -> None:
        del state_dict, prefix
        self.assigned = assign


class _DiTConfig:
    unet_config = {"image_model": "minimax_h3"}
    custom_operations = None

    def set_inference_dtype(self, *args, **kwargs) -> None:
        del args, kwargs

    def get_model(self, state_dict, prefix) -> _DiTModel:
        del state_dict, prefix
        return _DiTModel(self.custom_operations)


@pytest.fixture
def no_files(monkeypatch):
    monkeypatch.setattr(nf4_loader, "_require_bitsandbytes", lambda: None)
    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", _DynamicPatcher)

    def use(state_dict: dict) -> None:
        monkeypatch.setattr(
            nf4_loader, "_load_state_dict", lambda path: (dict(state_dict), None)
        )

    return use


def _clone_non_dynamic(patcher):
    clone = patcher.clone(disable_dynamic=True)
    assert type(clone) is comfy.model_patcher.ModelPatcher
    assert clone.model is not patcher.model, "clone did not rebuild the model"
    return clone


def test_diffusion_model_factory_accepts_disable_dynamic(no_files, monkeypatch) -> None:
    no_files(_nf4_state_dict("blocks.0.attn.qkv_proj.weight", (192, 64)))
    monkeypatch.setattr(
        comfy.model_detection,
        "model_config_from_unet",
        lambda *args, **kwargs: _DiTConfig(),
    )

    patcher = nf4_loader.load_nf4_diffusion_model("dit.safetensors")
    assert patcher.is_dynamic() and patcher.model.assigned is True

    clone = _clone_non_dynamic(patcher)
    assert clone.model.assigned is False, (
        "the rebuilt model was loaded for dynamic VRAM"
    )
    factory, factory_args = patcher.cached_patcher_init
    assert type(factory(*factory_args)) is _DynamicPatcher


def test_text_encoder_factory_accepts_disable_dynamic(no_files, monkeypatch) -> None:
    no_files(
        _nf4_state_dict("model.language_model.layers.0.mlp.up_proj.weight", (8, 4))
    )
    calls = []

    class _Clip:
        def __init__(self, patcher) -> None:
            self.patcher = patcher

    def load_text_encoder_state_dicts(state_dicts, **kwargs):
        calls.append(kwargs)
        patcher_class = (
            comfy.model_patcher.ModelPatcher
            if kwargs.get("disable_dynamic")
            else _DynamicPatcher
        )
        return _Clip(patcher_class(torch.nn.Linear(4, 8), CPU, CPU))

    monkeypatch.setattr(
        comfy.sd, "load_text_encoder_state_dicts", load_text_encoder_state_dicts
    )

    clip = nf4_loader.load_nf4_text_encoder("encoder.safetensors")
    _clone_non_dynamic(clip.patcher)

    # A core older than disable_dynamic must never be passed the argument.
    assert "disable_dynamic" not in calls[0]
    assert calls[1]["disable_dynamic"] is True


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_vae_factories_accept_disable_dynamic(kind, no_files, monkeypatch) -> None:
    no_files(_nf4_state_dict("decoder.transformer_blocks.0.attn.to_qkv.weight", (8, 4)))

    class _Vae:
        def __init__(self) -> None:
            self.first_stage_model = torch.nn.Linear(4, 4)
            self.patcher = _DynamicPatcher(self.first_stage_model, CPU, CPU)

    monkeypatch.setattr(nf4_loader, "_load_nf4_vae", lambda path, state_dict: _Vae())
    monkeypatch.setattr(nf4_loader, "_replace_nf4_audio_attentions", lambda root: 1)
    loader = (
        nf4_loader.load_nf4_video_vae
        if kind == "video"
        else nf4_loader.load_nf4_audio_vae
    )

    vae = loader("vae.safetensors")
    _clone_non_dynamic(vae.patcher)
    factory, factory_args = vae.patcher.cached_patcher_init
    assert isinstance(factory(*factory_args), comfy.model_patcher.ModelPatcher)
