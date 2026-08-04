"""Validate MiniMax H3 NF4 checkpoint parsing without allocating model weights."""

from __future__ import annotations

import json

import pytest
import torch

from nf4 import (
    AUDIO_LATENTS_MEAN,
    AUDIO_LATENTS_STD,
    NF4_QUANT_STATE_SUFFIX,
    convert_minimax_audio_vae_state_dict,
    convert_minimax_text_encoder_state_dict,
    make_shape_probe,
    nf4_quantized_shapes,
    without_quantized_payload,
)


def _quant_state(shape: tuple[int, int]) -> torch.Tensor:
    payload = json.dumps(
        {
            "quant_type": "nf4",
            "blocksize": 64,
            "dtype": "bfloat16",
            "shape": list(shape),
        }
    ).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def test_nf4_shape_probe_restores_logical_matrix_without_allocating_storage() -> None:
    weight_key = "blocks.0.attn.qkv_proj.weight"
    state_dict = {
        weight_key: torch.zeros((12, 1), dtype=torch.uint8),
        weight_key + ".absmax": torch.ones(1, dtype=torch.uint8),
        weight_key + NF4_QUANT_STATE_SUFFIX: _quant_state((6, 4)),
    }

    shapes = nf4_quantized_shapes(state_dict)
    probe = make_shape_probe(state_dict, shapes)

    assert shapes == {weight_key: (6, 4)}
    assert probe[weight_key].device.type == "meta"
    assert tuple(probe[weight_key].shape) == (6, 4)
    assert tuple(state_dict[weight_key].shape) == (12, 1)


def test_nf4_parser_rejects_missing_and_mislabeled_quant_state() -> None:
    with pytest.raises(ValueError, match="contains no BitsAndBytes NF4"):
        nf4_quantized_shapes({"weight": torch.ones(1)})

    key = "layer.weight" + NF4_QUANT_STATE_SUFFIX
    invalid = json.dumps({"quant_type": "fp4", "shape": [2, 2]}).encode("utf-8")
    with pytest.raises(ValueError, match="expected 'nf4'"):
        nf4_quantized_shapes(
            {
                "layer.weight": torch.zeros((2, 1), dtype=torch.uint8),
                key: torch.tensor(list(invalid), dtype=torch.uint8),
            }
        )


def test_without_quantized_payload_keeps_full_precision_tensors() -> None:
    state_dict = {
        "layer.weight": torch.zeros((8, 1), dtype=torch.uint8),
        "layer.weight.absmax": torch.ones(1, dtype=torch.uint8),
        "layer.weight" + NF4_QUANT_STATE_SUFFIX: _quant_state((4, 4)),
        "layer.bias": torch.ones(4),
        "norm.weight": torch.ones(4),
    }

    result = without_quantized_payload(state_dict, {"layer.weight": (4, 4)})

    assert set(result) == {"layer.bias", "norm.weight"}
    assert set(state_dict) == {
        "layer.weight",
        "layer.weight.absmax",
        "layer.weight" + NF4_QUANT_STATE_SUFFIX,
        "layer.bias",
        "norm.weight",
    }


def test_text_encoder_conversion_maps_hugging_face_prefixes_and_sidecars() -> None:
    state_dict = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "model.language_model.layers.0.self_attn.q_proj.weight"
        + NF4_QUANT_STATE_SUFFIX: _quant_state((2, 2)),
        "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        "model.language_model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }

    converted = convert_minimax_text_encoder_state_dict(state_dict)

    assert set(converted) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight" + NF4_QUANT_STATE_SUFFIX,
        "visual.blocks.0.attn.qkv.weight",
    }


def test_vae_probe_preserves_comfyui_detection_keys() -> None:
    weight_key = "decoder.transformer_blocks.0.attn.to_qkv.weight"
    state_dict = {
        weight_key: torch.zeros((8, 1), dtype=torch.uint8),
        weight_key + NF4_QUANT_STATE_SUFFIX: _quant_state((4, 4)),
        "decoder.transformer_blocks.0.scale1": torch.ones(4),
        "encoder.down.5.block.0.conv1.weight": torch.ones(1),
        "pre_block.attn.zero_k_bias": torch.zeros(4),
    }

    probe = without_quantized_payload(
        state_dict, nf4_quantized_shapes(state_dict)
    )

    assert set(probe) == {
        "decoder.transformer_blocks.0.scale1",
        "encoder.down.5.block.0.conv1.weight",
        "pre_block.attn.zero_k_bias",
    }


def test_audio_vae_conversion_folds_weight_norm_and_adds_latent_stats() -> None:
    weight_g = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.bfloat16)
    weight_v = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [5.0, 12.0]],
        ],
        dtype=torch.bfloat16,
    )
    state_dict = {
        "encoder.block.0.weight_g": weight_g,
        "encoder.block.0.weight_v": weight_v,
        "encoder.block.0.bias": torch.ones(2),
    }

    converted = convert_minimax_audio_vae_state_dict(state_dict)

    expected_weight = torch.tensor(
        [
            [[1.203125, 1.6015625], [0.0, 0.0]],
            [[0.0, 0.0], [1.15625, 2.765625]],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(
        converted["encoder.block.0.weight"], expected_weight, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        converted["latents_mean"], torch.tensor(AUDIO_LATENTS_MEAN)
    )
    torch.testing.assert_close(
        converted["latents_std"], torch.tensor(AUDIO_LATENTS_STD)
    )
    assert "encoder.block.0.weight_g" in state_dict
    assert "encoder.block.0.weight_v" in state_dict


def test_audio_vae_conversion_rejects_incomplete_weight_norm_pair() -> None:
    with pytest.raises(ValueError, match="missing_weight_g"):
        convert_minimax_audio_vae_state_dict(
            {"encoder.block.0.weight_v": torch.ones((2, 1, 3))}
        )
