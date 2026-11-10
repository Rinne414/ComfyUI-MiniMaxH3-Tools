"""A LoRA on an NF4 model must fail with a plain message, never silently do nothing.

Core applies a LoRA three ways, depending on the patcher and on free VRAM:
dynamic VRAM attaches weight_lowvram_function, the legacy low-VRAM path attaches
weight_function, and a legacy full load merges it through set_weight. A LoRA
cannot be merged into 4-bit NF4 codes, so every path must refuse.
"""

from __future__ import annotations

import pytest
import torch

from nf4_cpu_support import BACKENDS, CPU, dynamic_patcher, make_case

import comfy.lora  # noqa: E402  (nf4_cpu_support set up the CPU first)
import comfy.model_patcher  # noqa: E402

MESSAGE = "cannot use LoRA"


def _lora_patches(features: int = 256) -> dict:
    generator = torch.Generator().manual_seed(9)
    return comfy.lora.load_lora(
        {
            "0.lora_up.weight": torch.randn(features, 4, generator=generator),
            "0.lora_down.weight": torch.randn(4, features, generator=generator),
        },
        {"0": "0.weight"},
    )


def _patched_model(backend, monkeypatch, patcher_factory):
    case = make_case(backend, monkeypatch)
    model = torch.nn.Sequential(case.module)
    patcher = patcher_factory(model)
    patched = patcher.add_patches(_lora_patches(), strength_patch=1.0)
    assert "0.weight" in patched, "core did not accept the LoRA for the NF4 layer"
    return case, model, patcher


@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamic_vram_lora_on_nf4_fails_loudly(backend, monkeypatch) -> None:
    case, model, patcher = _patched_model(
        backend, monkeypatch, lambda model: dynamic_patcher(model, monkeypatch)
    )
    patcher.load(CPU)
    assert case.module.weight_lowvram_function is not None

    with pytest.raises(RuntimeError, match=MESSAGE):
        model(torch.randn(2, 256))


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_full_load_lora_on_nf4_fails_loudly(backend, monkeypatch) -> None:
    _, model, patcher = _patched_model(
        backend,
        monkeypatch,
        lambda model: comfy.model_patcher.ModelPatcher(model, CPU, CPU),
    )
    try:
        with pytest.raises(RuntimeError, match=MESSAGE):
            patcher.patch_model(device_to=CPU)
            model(torch.randn(2, 256))
    finally:
        patcher.unpatch_model(device_to=CPU)


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_low_vram_lora_on_nf4_fails_loudly(backend, monkeypatch) -> None:
    case, model, patcher = _patched_model(
        backend,
        monkeypatch,
        lambda model: comfy.model_patcher.ModelPatcher(model, CPU, CPU),
    )
    try:
        patcher.patch_model(device_to=CPU, lowvram_model_memory=1)
        assert case.module.weight_function, "core did not take the low-VRAM path"
        with pytest.raises(RuntimeError, match=MESSAGE):
            model(torch.randn(2, 256))
    finally:
        patcher.unpatch_model(device_to=CPU)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nf4_without_lora_still_runs_after_a_lora_clone(backend, monkeypatch) -> None:
    case = make_case(backend, monkeypatch)
    model = torch.nn.Sequential(case.module)
    base = dynamic_patcher(model, monkeypatch)
    base.clone().add_patches(_lora_patches(), strength_patch=1.0)
    base.load(CPU)
    inputs = torch.randn(2, 256)

    torch.testing.assert_close(
        model(inputs), case.reference(inputs), rtol=0.0, atol=0.0
    )
