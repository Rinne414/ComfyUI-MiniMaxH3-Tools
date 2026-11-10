"""The NF4 Linear under ComfyUI's dynamic VRAM patcher, run on the CPU.

Each test goes through core's own classes: ModelPatcherDynamic.load() sets the
module up, comfy.model_prefetch stages its weights, and comfy.ops casts them.
See nf4_cpu_support for what is faked and why.
"""

from __future__ import annotations

import pytest
import torch

from nf4_cpu_support import BACKENDS, CPU, dynamic_patcher, make_case

import comfy.memory_management  # noqa: E402  (nf4_cpu_support set up the CPU first)
import comfy.model_prefetch  # noqa: E402


@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamic_prefetch_stages_nf4_linear_bytes(backend, monkeypatch) -> None:
    case = make_case(backend, monkeypatch)
    model = torch.nn.Sequential(case.module)
    patcher = dynamic_patcher(model, monkeypatch)
    patcher.load(CPU)
    assert hasattr(case.module, "_v"), (
        "the dynamic patcher did not give the layer a VRAM slot"
    )

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
