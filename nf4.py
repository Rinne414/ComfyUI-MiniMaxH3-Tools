"""Pure helpers for loading pre-quantized BitsAndBytes NF4 checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import torch

NF4_QUANT_STATE_SUFFIX: Final[str] = ".quant_state.bitsandbytes__nf4"
FP4_QUANT_STATE_SUFFIX: Final[str] = ".quant_state.bitsandbytes__fp4"
QuantizedShape = tuple[int, int]

AUDIO_LATENTS_MEAN: Final[tuple[float, ...]] = (
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797,
)
AUDIO_LATENTS_STD: Final[tuple[float, ...]] = (
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363,
)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(
            "NF4 quant-state tensor must be a one-dimensional uint8 tensor; "
            f"got dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    return bytes(int(item) for item in value.detach().cpu().tolist())


def _parse_quant_state_shape(value: torch.Tensor, key: str) -> QuantizedShape:
    try:
        decoded = json.loads(_tensor_bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"NF4 quant-state '{key}' is not valid JSON: {error}"
        ) from error

    if not isinstance(decoded, dict):
        raise ValueError(f"NF4 quant-state '{key}' must contain a JSON object")
    quant_type = decoded.get("quant_type")
    if quant_type != "nf4":
        raise ValueError(
            f"NF4 quant-state '{key}' declares quant_type={quant_type!r}; expected 'nf4'"
        )
    shape = decoded.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(item, int) and item > 0 for item in shape)
    ):
        raise ValueError(
            f"NF4 quant-state '{key}' must contain a positive two-dimensional shape"
        )
    return (shape[0], shape[1])


def nf4_quantized_shapes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, QuantizedShape]:
    """Return logical matrix shapes encoded in BitsAndBytes NF4 metadata."""
    shapes: dict[str, QuantizedShape] = {}
    for key, value in state_dict.items():
        if key.endswith(FP4_QUANT_STATE_SUFFIX):
            raise ValueError(f"FP4 is not supported by this loader: '{key}'")
        if not key.endswith(NF4_QUANT_STATE_SUFFIX):
            continue
        weight_key = key[: -len(NF4_QUANT_STATE_SUFFIX)]
        if weight_key in shapes:
            raise ValueError(f"duplicate NF4 quant-state for '{weight_key}'")
        if weight_key not in state_dict:
            raise ValueError(f"NF4 quant-state '{key}' has no packed weight tensor")
        shapes[weight_key] = _parse_quant_state_shape(value, key)

    if not shapes:
        raise ValueError("state dict contains no BitsAndBytes NF4 quant-state tensors")
    return shapes


def make_shape_probe(
    state_dict: Mapping[str, torch.Tensor],
    shapes: Mapping[str, QuantizedShape],
) -> dict[str, torch.Tensor]:
    """Create a metadata-only state dict whose quantized weights have logical shapes."""
    probe = dict(state_dict)
    for key, shape in shapes.items():
        probe[key] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    return probe


def without_quantized_payload(
    state_dict: Mapping[str, torch.Tensor],
    shapes: Mapping[str, QuantizedShape],
) -> dict[str, torch.Tensor]:
    """Remove packed weights and their sidecars without mutating the input mapping."""
    bases = tuple(shapes)
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key == base or key.startswith(base + ".") for base in bases)
    }


def with_placeholder_weights(
    state_dict: Mapping[str, torch.Tensor],
    shapes: Mapping[str, QuantizedShape],
) -> dict[str, torch.Tensor]:
    """Remove the packed payload, but keep each NF4 weight as a placeholder.

    The placeholder is a float32 view of a single zero with the weight's logical
    shape, so it costs 4 bytes. Under dynamic VRAM, core's lazy Linear fills a
    weight missing from the state dict with a full float32 zeros tensor; with a
    placeholder it takes the view as is.
    """
    probe = without_quantized_payload(state_dict, shapes)
    zero = torch.zeros(())
    for key, shape in shapes.items():
        probe[key] = zero.expand(shape)
    return probe


def convert_minimax_text_encoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the DiffSynth/Hugging Face H3 encoder prefixes to ComfyUI's layout."""
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == "lm_head.weight" or key.startswith("model.language_model.norm."):
            continue
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        elif key.startswith("model.visual."):
            key = "visual." + key[len("model.visual.") :]
        if key in converted:
            raise ValueError(
                f"text encoder prefix conversion produced duplicate key '{key}'"
            )
        converted[key] = value
    return converted


def _fold_weight_norm(
    weight_g: torch.Tensor,
    weight_v: torch.Tensor,
    weight_key: str,
) -> torch.Tensor:
    if weight_v.ndim < 2:
        raise ValueError(
            f"weight-normalized tensor '{weight_key}_v' must have at least two dimensions"
        )
    expected_g_shape = (weight_v.shape[0],) + (1,) * (weight_v.ndim - 1)
    if tuple(weight_g.shape) != expected_g_shape:
        raise ValueError(
            f"weight-normalized tensor '{weight_key}_g' has shape {tuple(weight_g.shape)}; "
            f"expected {expected_g_shape} for '{weight_key}_v'"
        )
    dimensions = tuple(range(1, weight_v.ndim))
    norm = torch.linalg.vector_norm(
        weight_v.to(dtype=torch.float32), dim=dimensions, keepdim=True
    )
    if torch.any(norm == 0):
        raise ValueError(
            f"weight-normalized tensor '{weight_key}_v' has a zero-norm channel"
        )
    folded = weight_g.to(dtype=torch.float32) * weight_v.to(dtype=torch.float32) / norm
    return folded.to(dtype=weight_v.dtype)


def convert_minimax_audio_vae_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fold DiffSynth weight norm and add its published audio latent statistics."""
    weight_g_suffix = ".weight_g"
    weight_v_suffix = ".weight_v"
    weight_g_bases = {
        key[: -len(weight_g_suffix)]
        for key in state_dict
        if key.endswith(weight_g_suffix)
    }
    weight_v_bases = {
        key[: -len(weight_v_suffix)]
        for key in state_dict
        if key.endswith(weight_v_suffix)
    }
    if weight_g_bases != weight_v_bases:
        missing_g = sorted(weight_v_bases - weight_g_bases)
        missing_v = sorted(weight_g_bases - weight_v_bases)
        raise ValueError(
            "audio VAE weight-normalization pairs are incomplete: "
            f"missing_weight_g={missing_g}, missing_weight_v={missing_v}"
        )

    converted = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(weight_g_suffix) and not key.endswith(weight_v_suffix)
    }
    for base in sorted(weight_g_bases):
        weight_key = base + ".weight"
        if weight_key in converted:
            raise ValueError(
                f"audio VAE contains both folded and weight-normalized '{weight_key}'"
            )
        converted[weight_key] = _fold_weight_norm(
            state_dict[base + weight_g_suffix],
            state_dict[base + weight_v_suffix],
            weight_key,
        )

    stats = {
        "latents_mean": torch.tensor(AUDIO_LATENTS_MEAN, dtype=torch.float32),
        "latents_std": torch.tensor(AUDIO_LATENTS_STD, dtype=torch.float32),
    }
    for key, value in stats.items():
        existing = converted.get(key)
        if existing is None:
            converted[key] = value
        elif tuple(existing.shape) != tuple(value.shape):
            raise ValueError(
                f"audio VAE '{key}' has shape {tuple(existing.shape)}; "
                f"expected {tuple(value.shape)}"
            )
    return converted
