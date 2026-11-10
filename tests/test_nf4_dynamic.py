"""The NF4 Linear under ComfyUI's model patchers, run on the CPU.

Each test goes through core's own classes: ModelPatcherDynamic.load() sets the
module up, comfy.model_prefetch stages its weights, and comfy.ops casts them.
The legacy ModelPatcher is covered too. See nf4_cpu_support for what is faked
and why.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from nf4_cpu_support import (
    BACKENDS,
    CPU,
    dynamic_patcher,
    fake_aimdo,
    make_case,
    needs_file_backed_weights,
    needs_prefetch,
    offload_to_pinned_host,
    weight_sized_tensors,
)

import comfy.memory_management  # noqa: E402  (nf4_cpu_support set up the CPU first)
import comfy.model_patcher  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.pinned_memory  # noqa: E402


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


@needs_prefetch
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


@needs_prefetch
@pytest.mark.skipif(
    "fast_disk"
    not in inspect.signature(
        comfy.model_patcher.ModelPatcherDynamic.__init__
    ).parameters,
    reason="this ComfyUI has no fast-disk storage policy",
)
@pytest.mark.parametrize("fast_disk", [True, False], ids=["fast-disk", "slow-disk"])
def test_staging_copies_nf4_weights_into_pinned_ram_only_off_a_fast_disk(
    fast_disk, monkeypatch
) -> None:
    """Why the loader must pass core's storage policy: with a VRAM slot granted,
    core reads a fast disk straight into VRAM, otherwise it first copies the
    weight into a pinned host buffer (comfy/ops.py cast_modules_with_vbar)."""
    import comfy_aimdo.model_vbar

    case = make_case("fake", monkeypatch)
    model = torch.nn.Sequential(case.module)
    fake_aimdo(monkeypatch)
    pinned = []
    monkeypatch.setattr(
        comfy.pinned_memory,
        "pin_memory",
        lambda m, subset, size=None: pinned.append(subset),
    )
    monkeypatch.setattr(comfy_aimdo.model_vbar, "vbar_fault", lambda v: object())
    monkeypatch.setattr(
        comfy.ops.comfy_aimdo.torch,
        "aimdo_to_tensor",
        lambda v, device: torch.empty(v[2], dtype=torch.uint8),
    )
    patcher = comfy.model_patcher.ModelPatcherDynamic(
        model, CPU, CPU, fast_disk=fast_disk
    )
    patcher.load(CPU)

    comfy.model_prefetch.pin_modules([case.module], CPU, None)
    try:
        staged = case.module._prefetch["xfer_dest"]
        weight, _ = comfy.memory_management.interpret_gathered_like(
            [case.packed, case.module.bias], staged
        )
        assert torch.equal(weight, case.packed)
    finally:
        comfy.model_prefetch.cleanup_prefetched_modules(model, [case.module])
    assert pinned == ([] if fast_disk else ["weights-loaded"])


@needs_file_backed_weights
@pytest.mark.parametrize(
    "prefetched",
    [
        pytest.param(False, id="cast"),
        pytest.param(True, id="prefetched", marks=needs_prefetch),
    ],
)
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
