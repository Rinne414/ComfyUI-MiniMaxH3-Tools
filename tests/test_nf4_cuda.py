"""Optional CUDA integration tests for the BitsAndBytes NF4 Linear shell."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

bitsandbytes = pytest.importorskip("bitsandbytes")

from nf4_loader import (  # noqa: E402
    _NF4DiTAttention,
    _NF4Linear,
    _replace_nf4_audio_attentions,
    _replace_quantized_linears,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NF4 matmul")
def test_prequantized_nf4_roundtrip_and_cpu_offload() -> None:
    torch.manual_seed(7)
    source = bitsandbytes.nn.Linear4bit(
        64,
        32,
        bias=True,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    source.weight = bitsandbytes.nn.Params4bit(
        torch.randn(32, 64, dtype=torch.bfloat16),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    )
    source.bias = torch.nn.Parameter(
        torch.randn(32, dtype=torch.bfloat16), requires_grad=False
    )
    source = source.cuda()
    state_dict = {key: value.cpu() for key, value in source.state_dict().items()}

    restored = _NF4Linear(64, 32, True, torch.device("cpu"), torch.bfloat16)
    load_result = restored.load_state_dict(state_dict, strict=True)
    inputs = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)

    expected = source(inputs)
    actual = restored(inputs)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert restored.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NF4 restore")
def test_comfyui_model_patcher_moves_prequantized_nf4_weight() -> None:
    import comfy.model_patcher

    torch.manual_seed(13)
    source = bitsandbytes.nn.Linear4bit(
        64,
        32,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    source.weight = bitsandbytes.nn.Params4bit(
        torch.randn(32, 64, dtype=torch.bfloat16),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    )
    source = source.cuda()
    state_dict = {key: value.cpu() for key, value in source.state_dict().items()}

    restored = _NF4Linear(64, 32, False, torch.device("cpu"), torch.bfloat16)
    restored.load_state_dict(state_dict, strict=True)
    root = torch.nn.Sequential(restored)
    patcher = comfy.model_patcher.ModelPatcher(
        root,
        load_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
    )

    patcher.patch_weight_to_device("0.weight", device_to=torch.device("cuda"), force_cast=True)
    inputs = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)

    assert root[0].weight.device.type == "cuda"
    torch.testing.assert_close(root(inputs), source(inputs), rtol=0.0, atol=0.0)
    assert patcher.backup["0.weight"].weight.device.type == "cpu"
    patcher.unpatch_model(device_to=torch.device("cpu"))
    assert root[0].weight.device.type == "cpu"
    assert root[0].weight.quant_state.absmax.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NF4 restore")
def test_prequantized_nf4_rejects_wrong_logical_shape() -> None:
    source = bitsandbytes.nn.Linear4bit(
        64,
        32,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    source.weight = bitsandbytes.nn.Params4bit(
        torch.randn(32, 64, dtype=torch.bfloat16),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    )
    state_dict = {
        key: value.cpu() for key, value in source.cuda().state_dict().items()
    }
    restored = _NF4Linear(64, 31, False, torch.device("cpu"), torch.bfloat16)

    with pytest.raises(RuntimeError, match=r"logical shape \(32, 64\); expected \(31, 64\)"):
        restored.load_state_dict(state_dict, strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NF4 matmul")
def test_comfyui_audio_attention_uses_nf4_qkv_forward() -> None:
    import comfy.ldm.minimax.audio_vae
    import comfy.ops

    torch.manual_seed(11)
    source_qkv = bitsandbytes.nn.Linear4bit(
        64,
        192,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    source_qkv.weight = bitsandbytes.nn.Params4bit(
        torch.randn(192, 64, dtype=torch.bfloat16),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    )
    source_qkv = source_qkv.cuda()
    qkv_state = {key: value.cpu() for key, value in source_qkv.state_dict().items()}

    attention = comfy.ldm.minimax.audio_vae.CausalAttention(64, 32, 4)
    attention.q_bias.data.copy_(torch.randn(64))
    attention.v_bias.data.copy_(torch.randn(64))
    attention.zero_k_bias.zero_()
    attention.proj.weight.data.copy_(torch.randn(32, 32))
    attention.proj.bias.data.copy_(torch.randn(32))
    root = torch.nn.Sequential(attention)
    state_dict = root.state_dict()
    del state_dict["0.qkv.weight"]
    state_dict.update({f"0.qkv.{key}": value for key, value in qkv_state.items()})

    assert _replace_quantized_linears(root, {"0.qkv.weight": (192, 64)}) == 1
    load_result = root.load_state_dict(state_dict, strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert _replace_nf4_audio_attentions(root) == 1

    inputs = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16)
    attention.proj.to(device=inputs.device, dtype=inputs.dtype)
    bias = torch.cat(
        (attention.q_bias, attention.zero_k_bias, attention.v_bias)
    ).to(device=inputs.device, dtype=inputs.dtype)
    qkv = source_qkv(inputs) + bias
    q, k, v = (
        qkv.reshape(2, 5, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
    )
    attended = comfy.ops.scaled_dot_product_attention(q, k, v, is_causal=True)
    pooled = functional.adaptive_avg_pool1d(torch.mean(attended, dim=1), 32)
    expected = functional.linear(
        pooled,
        attention.proj.weight.to(device=inputs.device, dtype=inputs.dtype),
        attention.proj.bias.to(device=inputs.device, dtype=inputs.dtype),
    )

    actual = root(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert root[0].qkv.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for attention")
def test_nf4_dit_attention_reads_head_interleaved_qkv() -> None:
    import comfy.ldm.minimax.model
    from comfy.ldm.modules.attention import optimized_attention

    torch.manual_seed(17)
    source = comfy.ldm.minimax.model.Attention(
        64,
        4,
        16,
        1e-5,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        operations=torch.nn,
    ).eval()
    attention = _NF4DiTAttention(source).eval()
    inputs = torch.randn(11, 64, device="cuda", dtype=torch.bfloat16)

    qkv = source.qkv_proj(inputs).view(11, 4, 3, 16)
    query = source.q_norm(qkv[:, :, 0, :].contiguous())
    key = source.k_norm(qkv[:, :, 1, :].contiguous())
    value = qkv[:, :, 2, :].contiguous()
    attended = optimized_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        4,
        mask=None,
        skip_reshape=True,
        transformer_options={},
    )
    expected = source.out_proj(attended.squeeze(0))

    actual = attention(inputs, rope_freqs=None, transformer_options={})

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
