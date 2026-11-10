"""Core's fused linear helpers must run the NF4 Linear's own forward.

comfy.ops.linear_input_act (used by the H3 video VAE, DiT MLP and text encoder)
reads linear.weight as a dense matrix when --fast fp16_accumulation is on and
the activation is fp16 on CUDA. Here that condition is forced and the CUDA
kernel is replaced by the plain math it computes, so the test runs on the CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from nf4_cpu_support import BACKENDS, make_case

import comfy.ops  # noqa: E402  (nf4_cpu_support set up the CPU first)
import comfy.quant_ops  # noqa: E402

# Older cores (the weekly CI job's pinned one) have no fp16 fused path, so the
# defect these tests guard against cannot happen there.
pytestmark = pytest.mark.skipif(
    not hasattr(comfy.ops, "_fp16_linear_wanted"),
    reason="this ComfyUI has no fp16-accumulation fused linear path",
)


@pytest.fixture
def fp16_accumulation(monkeypatch):
    calls = []

    def fp16_linear(x, weight, bias, residual=None, residual_scale=None):
        calls.append(weight)
        out = functional.linear(x, weight, bias)
        return out if residual is None else torch.addcmul(residual, out, residual_scale)

    monkeypatch.setattr(comfy.ops, "_fp16_linear_wanted", lambda x: True)
    monkeypatch.setattr(comfy.quant_ops.ck, "fp16_linear", fp16_linear, raising=False)
    return calls


def test_fused_path_is_taken_for_a_dense_core_linear(fp16_accumulation) -> None:
    layer = comfy.ops.manual_cast.Linear(8, 4)
    inputs = torch.randn(2, 8)
    residual = torch.randn(2, 4)

    actual = comfy.ops.linear_input_act(
        layer, inputs, None, residual=residual, residual_scale=torch.ones(4)
    )

    expected = residual + functional.linear(inputs, layer.weight, layer.bias)
    torch.testing.assert_close(actual, expected)
    assert len(fp16_accumulation) == 1, "the forced fp16 path was not reached"


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp16_accumulation_runs_nf4_forward_with_activation_and_residual(
    backend, fp16_accumulation, monkeypatch
) -> None:
    case = make_case(backend, monkeypatch)
    inputs = torch.randn(3, 512)
    residual = torch.randn(3, 256)
    scale = torch.full((256,), 0.5)
    gate, up = inputs.chunk(2, dim=-1)
    expected = torch.addcmul(
        residual, case.reference(functional.silu(gate) * up), scale
    )

    actual = comfy.ops.linear_input_act(
        case.module, inputs, "swiglu", residual=residual, residual_scale=scale
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert fp16_accumulation == [], "the NF4 codes were fed to the dense fp16 GEMM"


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp16_accumulation_runs_nf4_forward_with_rms_norm(
    backend, fp16_accumulation, monkeypatch
) -> None:
    case = make_case(backend, monkeypatch)
    inputs = torch.randn(3, 256)
    norm_weight = torch.rand(256) + 0.5
    normed = functional.rms_norm(inputs, (256,), norm_weight, 1e-6)

    actual = comfy.ops.linear_input_act(
        case.module, inputs, "rms_norm", act_weight=norm_weight, act_eps=1e-6
    )

    torch.testing.assert_close(actual, case.reference(normed), rtol=1e-5, atol=1e-5)
    assert fp16_accumulation == []
