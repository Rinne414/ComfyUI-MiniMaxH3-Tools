"""CPU stand-ins for exercising the NF4 Linear against ComfyUI's dynamic VRAM path.

Two NF4 backends build the same kind of module:

- "fake": a stand-in for BitsAndBytes whose matmul checks that it received the
  exact stored packed bytes on the input's device, then multiplies by the dense
  matrix those bytes stand for. Needs nothing beyond torch, so CI runs it.
- "bitsandbytes": real NF4 quantization and matmul on the CPU, used when
  BitsAndBytes is installed.

`dynamic_patcher` runs core's real ModelPatcherDynamic.load() on the CPU. Only
the pieces that live in the native comfy-aimdo library (the VRAM address range,
host buffers and residency faults) are replaced, the same way core's own unit
tests fake them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.memory_management  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.pinned_memory  # noqa: E402

import nf4_loader  # noqa: E402

CPU = torch.device("cpu")
BACKENDS = ("fake", "bitsandbytes")


class FakeQuantState:
    """Holds the dense matrix and the packed bytes it was stored as."""

    def __init__(self, dense: torch.Tensor, packed: torch.Tensor) -> None:
        self.dense = dense
        self.packed = packed
        self.absmax = torch.ones(1, device=dense.device)
        self.shape = dense.shape

    def as_dict(self, packed: bool = False) -> dict[str, torch.Tensor]:
        del packed
        return {"dense": self.dense, "packed": self.packed}

    @classmethod
    def from_dict(cls, qs_dict: dict[str, torch.Tensor], device) -> FakeQuantState:
        return cls(qs_dict["dense"].to(device), qs_dict["packed"].to(device))


class FakeBitsAndBytes:
    """Only the part of the BitsAndBytes API the NF4 Linear calls."""

    def __init__(self) -> None:
        self.calls = 0

    def matmul_4bit(self, inputs, weight_t, bias=None, quant_state=None):
        self.calls += 1
        packed = weight_t.t()
        if packed.dtype != torch.uint8 or packed.device != inputs.device:
            raise AssertionError(
                f"packed weight reached matmul as {packed.dtype} on {packed.device}"
            )
        if quant_state.absmax.device != inputs.device:
            raise AssertionError("quant state is not on the input device")
        if not torch.equal(
            packed.reshape(-1), quant_state.packed.reshape(-1).to(packed.device)
        ):
            raise AssertionError(
                "matmul received bytes that differ from the stored weight"
            )
        return functional.linear(inputs, quant_state.dense.to(inputs.dtype), bias)


@dataclass
class NF4Case:
    backend: str
    module: torch.nn.Module
    packed: torch.Tensor
    reference: Callable[[torch.Tensor], torch.Tensor]
    bnb: object


def _fake_case(in_features: int, out_features: int, bias: bool) -> NF4Case:
    generator = torch.Generator().manual_seed(5)
    dense = torch.randn(out_features, in_features, generator=generator)
    packed = torch.randint(
        0,
        256,
        (out_features * in_features // 2, 1),
        dtype=torch.uint8,
        generator=generator,
    )
    module = nf4_loader._NF4Linear(in_features, out_features, bias, CPU, torch.float32)
    module.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
    module.quant_state = FakeQuantState(dense, packed.clone())
    module.is_nf4 = True
    bias_value = torch.randn(out_features, generator=generator) if bias else None
    if bias_value is not None:
        module.bias = torch.nn.Parameter(bias_value.clone(), requires_grad=False)
    module.register_parameter("load_anchor", None)

    def reference(inputs: torch.Tensor) -> torch.Tensor:
        bias_input = None if bias_value is None else bias_value.to(inputs.dtype)
        return functional.linear(inputs, dense.to(inputs.dtype), bias_input)

    return NF4Case("fake", module, packed, reference, FakeBitsAndBytes())


def _bitsandbytes_case(bnb, in_features: int, out_features: int, bias: bool) -> NF4Case:
    torch.manual_seed(5)
    source = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=bias,
        compute_dtype=torch.float32,
        compress_statistics=True,
        quant_type="nf4",
    )
    source.weight = bnb.nn.Params4bit(
        torch.randn(out_features, in_features),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    )
    source = source.to(CPU)
    state_dict = source.state_dict()
    module = nf4_loader._NF4Linear(in_features, out_features, bias, CPU, torch.float32)
    result = module.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"NF4 test module did not load: {result}")
    return NF4Case("bitsandbytes", module, state_dict["weight"].clone(), source, bnb)


def make_case(
    backend: str, monkeypatch, in_features=256, out_features=256, bias=True
) -> NF4Case:
    """Build an NF4 Linear on the CPU. 256x256 packs to 32 KB, above the 16 KB
    under which the dynamic patcher force-loads a module instead of staging it."""
    if backend == "fake":
        case = _fake_case(in_features, out_features, bias)
    else:
        import pytest

        bnb = pytest.importorskip("bitsandbytes")
        monkeypatch.setattr(nf4_loader, "_require_bitsandbytes", lambda: bnb)
        case = _bitsandbytes_case(bnb, in_features, out_features, bias)
    monkeypatch.setattr(nf4_loader, "_require_bitsandbytes", lambda: case.bnb)
    return case


class _FakeHostBuffer:
    """comfy-aimdo host buffers need the native library, which only loads with CUDA."""

    size = 0

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def truncate(self, *args, **kwargs) -> None:
        del args, kwargs


class _FakeVBar:
    """The VRAM address range: hands out (vbar, offset, size) slots like ModelVBAR."""

    base_addr = 0

    def __init__(self) -> None:
        self.offset = 0

    def alloc(self, size: int):
        slot = (self, self.offset, size)
        self.offset += size
        return slot

    def prioritize(self) -> None:
        return None

    def loaded_size(self) -> int:
        return 0

    def free_memory(self, size: int) -> int:
        del size
        return 0


def fake_aimdo(monkeypatch) -> None:
    """Fake the native pieces; every slot misses, so weights take the cast-buffer path."""
    import comfy_aimdo.host_buffer
    import comfy_aimdo.model_vbar

    vbar = _FakeVBar()
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", _FakeHostBuffer)
    monkeypatch.setattr(comfy_aimdo.model_vbar, "vbar_fault", lambda v: None)
    monkeypatch.setattr(
        comfy_aimdo.model_vbar, "vbar_signature_compare", lambda *a: False
    )
    monkeypatch.setattr(comfy_aimdo.model_vbar, "vbar_unpin", lambda v: None)
    monkeypatch.setattr(
        comfy.model_patcher.ModelPatcherDynamic,
        "_vbar_get",
        lambda self, create=False: vbar,
    )
    monkeypatch.setattr(
        comfy.model_management, "ensure_pin_registerable", lambda *a, **k: True
    )
    monkeypatch.setattr(comfy.pinned_memory, "pin_memory", lambda *a, **k: None)
    # ModelPatcherDynamic reroutes CPU load devices to the legacy patcher, and the
    # cast path treats a CPU compute device as "no VRAM"; core's tests do the same.
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)


def dynamic_patcher(model: torch.nn.Module, monkeypatch):
    """A real ModelPatcherDynamic on the CPU, before load()."""
    fake_aimdo(monkeypatch)
    patcher = comfy.model_patcher.ModelPatcherDynamic(model, CPU, CPU)
    assert patcher.is_dynamic()
    return patcher


def offload_to_pinned_host(module: torch.nn.Module) -> torch.Tensor:
    """Take the module's weights out of RAM the way dynamic VRAM holds them.

    The parameters become file-backed meta tensors and their bytes sit in the
    module's pinned host buffer, so a forward can only reach them through core's
    cast path. Call before ModelPatcherDynamic.load(). Returns the buffer.
    """
    tensors = [
        module.weight.detach(),
        None if module.bias is None else module.bias.detach(),
    ]
    pin = torch.empty(
        comfy.memory_management.vram_aligned_size(tensors), dtype=torch.uint8
    )
    comfy.model_management.cast_to_gathered(tensors, pin)
    for name, tensor in zip(("weight", "bias"), tensors):
        if tensor is None:
            continue
        offloaded = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        offloaded.untyped_storage()._comfy_tensor_file_slice = (
            comfy.memory_management.TensorFileSlice(
                None, threading.Lock(), 0, tensor.numel() * tensor.element_size()
            )
        )
        setattr(module, name, torch.nn.Parameter(offloaded, requires_grad=False))
    module._pins = {"weights": {"pin": pin, "registered": True}}
    return pin


def weight_sized_tensors(module: torch.nn.Module, nbytes: int) -> list[str]:
    """Names of module attributes, outside core's pinned buffer, holding a host
    tensor at least `nbytes` large: a second host copy of a weight shows up here."""
    found = []
    for name, value in vars(module).items():
        if name == "_pins":
            continue
        values = value.items() if isinstance(value, dict) else [(name, value)]
        for key, item in values:
            if (
                isinstance(item, torch.Tensor)
                and item.device.type == "cpu"
                and item.numel() * item.element_size() >= nbytes
            ):
                found.append(str(key))
    return found
