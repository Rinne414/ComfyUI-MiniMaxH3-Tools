"""The NF4 Linear under ComfyUI's model patchers, run on the CPU.

Each test goes through core's own classes: ModelPatcherDynamic.load() sets the
module up, comfy.model_prefetch stages its weights, and comfy.ops casts them.
The legacy ModelPatcher is covered too. See nf4_cpu_support for what is faked
and why.
"""

from __future__ import annotations

import pytest
import torch

from nf4_cpu_support import (
    BACKENDS,
    CPU,
    dynamic_patcher,
    make_case,
    offload_to_pinned_host,
    weight_sized_tensors,
)

import comfy.memory_management  # noqa: E402  (nf4_cpu_support set up the CPU first)
import comfy.model_patcher  # noqa: E402
import comfy.model_prefetch  # noqa: E402


def _loaded(backend, monkeypatch, offloaded):
    case = make_case(backend, monkeypatch)
    if offloaded:
        offload_to_pinned_host(case.module)
    model = torch.nn.Sequential(case.module)
    patcher = dynamic_patcher(model, monkeypatch)
    patcher.load(CPU)
    assert hasattr(case.module, "_v"), (
        "the dynamic patcher did not give the layer a VRAM slot"
    )
    return case, model, patcher


@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamic_prefetch_stages_nf4_linear_bytes(backend, monkeypatch) -> None:
    case, model, _ = _loaded(backend, monkeypatch, offloaded=False)

    comfy.model_prefetch.pin_modules([case.module], CPU, None)
    try:
        staged = case.module._prefetch["xfer_dest"]
        weight, _ = comfy.memory_management.interpret_gathered_like(
            [case.packed, case.module.bias], staged
        )
        assert torch.equal(weight, case.packed)
    finally:
        comfy.model_prefetch.cleanup_prefetched_modules(model, [case.module])
    assert not hasattr(case.module, "_prefetch")


@pytest.mark.parametrize("prefetched", [False, True], ids=["cast", "prefetched"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamic_forward_uses_core_staged_weight_without_host_copy(
    backend, prefetched, monkeypatch
) -> None:
    case, model, _ = _loaded(backend, monkeypatch, offloaded=True)
    module = case.module
    stored_weight = module.weight
    stored_quant_state = module.quant_state
    inputs = torch.randn(3, 256)
    expected = case.reference(inputs)

    if prefetched:
        comfy.model_prefetch.pin_modules([module], CPU, None)
    try:
        actual = model(inputs)
    finally:
        if prefetched:
            comfy.model_prefetch.cleanup_prefetched_modules(model, [module])

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert module.weight is stored_weight
    assert module.weight.device.type == "meta", "the forward pulled the weight into RAM"
    assert module.quant_state is stored_quant_state
    assert weight_sized_tensors(module, case.packed.nbytes) == []
    assert not hasattr(module, "_prefetch")


@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamic_forward_with_resident_weight_keeps_it(backend, monkeypatch) -> None:
    case, model, _ = _loaded(backend, monkeypatch, offloaded=False)
    stored_weight = case.module.weight
    stored_pointer = stored_weight.data_ptr()
    inputs = torch.randn(2, 256)

    torch.testing.assert_close(
        model(inputs), case.reference(inputs), rtol=0.0, atol=0.0
    )
    assert case.module.weight is stored_weight
    assert case.module.weight.data_ptr() == stored_pointer


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_patcher_forward_matches_reference(backend, monkeypatch) -> None:
    case = make_case(backend, monkeypatch)
    model = torch.nn.Sequential(case.module)
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=CPU, offload_device=CPU
    )
    patcher.patch_model(device_to=CPU)
    inputs = torch.randn(2, 256)
    try:
        torch.testing.assert_close(
            model(inputs), case.reference(inputs), rtol=0.0, atol=0.0
        )
    finally:
        patcher.unpatch_model(device_to=CPU)
