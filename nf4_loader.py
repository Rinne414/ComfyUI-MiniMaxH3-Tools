"""ComfyUI loaders for DiffSynth-style MiniMax-H3 BitsAndBytes NF4 weights."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from comfy_api.latest import io

if TYPE_CHECKING:
    from bitsandbytes.functional import QuantState

if __package__:
    from .nf4 import (
        NF4_QUANT_STATE_SUFFIX,
        QuantizedShape,
        convert_minimax_audio_vae_state_dict,
        convert_minimax_text_encoder_state_dict,
        make_shape_probe,
        nf4_quantized_shapes,
        without_quantized_payload,
    )
else:
    from nf4 import (
        NF4_QUANT_STATE_SUFFIX,
        QuantizedShape,
        convert_minimax_audio_vae_state_dict,
        convert_minimax_text_encoder_state_dict,
        make_shape_probe,
        nf4_quantized_shapes,
        without_quantized_payload,
    )

LOGGER = logging.getLogger(__name__)


def _require_bitsandbytes() -> ModuleType:
    try:
        import bitsandbytes
    except ImportError as error:
        raise RuntimeError(
            "MiniMax H3 NF4 requires BitsAndBytes in ComfyUI's Python environment. "
            'Install it with: python -m pip install "bitsandbytes>=0.50.0"'
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MiniMax H3 NF4 requires an NVIDIA CUDA device; "
            f"torch.cuda.is_available() returned False for torch {torch.__version__}"
        )
    if not hasattr(bitsandbytes.nn, "Params4bit"):
        raise RuntimeError(
            f"BitsAndBytes {bitsandbytes.__version__} has no nn.Params4bit implementation"
        )
    return bitsandbytes


def _copy_quant_state(
    quant_state: QuantState | None,
    device: torch.device,
) -> QuantState | None:
    if quant_state is None:
        return None
    copied = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in quant_state.as_dict(packed=False).items()
    }
    return type(quant_state).from_dict(copied, device)


@cache
def _nf4_parameter_type() -> type[torch.nn.Parameter]:
    bitsandbytes = _require_bitsandbytes()

    class _ComfyNF4Params4bit(bitsandbytes.nn.Params4bit):
        """Preserve NF4 state across ComfyUI's copy-aware device moves."""

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.empty_like and args and isinstance(args[0], cls):
                source = args[0]
                options = {} if kwargs is None else kwargs
                dtype = (
                    source.dtype if options.get("dtype") is None else options["dtype"]
                )
                device = (
                    source.device
                    if options.get("device") is None
                    else options["device"]
                )
                device = torch.device(device)
                return cls(
                    torch.empty(source.shape, dtype=dtype, device=device),
                    requires_grad=source.requires_grad,
                    quant_state=_copy_quant_state(source.quant_state, device),
                    blocksize=source.blocksize,
                    compress_statistics=source.compress_statistics,
                    quant_type=source.quant_type,
                    quant_storage=source.quant_storage,
                    bnb_quantized=source.bnb_quantized,
                    module=source.module,
                )
            return super().__torch_function__(func, types, args, kwargs)

        def detach(self):
            return type(self)(
                self,
                requires_grad=False,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
                module=self.module,
            )

        def to(self, *args, **kwargs):
            copy_requested = bool(kwargs.pop("copy", False))
            device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
            if device is not None and device.type != "meta" and not self.bnb_quantized:
                source = self
                if copy_requested:
                    source = type(self)(
                        torch.nn.Parameter.to(self, copy=True),
                        requires_grad=self.requires_grad,
                        quant_state=_copy_quant_state(self.quant_state, self.device),
                        blocksize=self.blocksize,
                        compress_statistics=self.compress_statistics,
                        quant_type=self.quant_type,
                        quant_storage=self.quant_storage,
                        bnb_quantized=self.bnb_quantized,
                        module=self.module,
                    )
                return source._quantize(device)

            target_device = self.device if device is None else device
            moved = type(self)(
                torch.nn.Parameter.to(
                    self,
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                    copy=copy_requested,
                ),
                requires_grad=self.requires_grad,
                quant_state=_copy_quant_state(self.quant_state, target_device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
                module=self.module,
            )
            if moved.module is not None:
                moved.module.quant_state = moved.quant_state
            return moved

    return _ComfyNF4Params4bit


class _NF4Linear(torch.nn.Module):
    """Linear shell that accepts full-precision or pre-quantized BnB weights."""

    # Core reads the attributes of comfy.ops.CastWeightBiasOp on every layer that
    # has comfy_cast_weights. None means "not a member of a CombinedLinear group";
    # core's CombinedLinear overwrites it on its members, and never combines a layer
    # that is not a MixedPrecisionOp.
    _combined_linear = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = None
        self.bias = None
        self.quant_state = None
        self.is_nf4 = False
        self.comfy_cast_weights = True
        self.weight_function = []
        self.bias_function = []
        self.register_parameter(
            "load_anchor",
            torch.nn.Parameter(
                torch.empty(1, device=device, dtype=dtype), requires_grad=False
            ),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del local_metadata, unexpected_keys
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        quant_state_key = weight_key + NF4_QUANT_STATE_SUFFIX

        if weight_key not in state_dict:
            if strict:
                missing_keys.append(weight_key)
            return

        anchor = self.load_anchor
        if quant_state_key in state_dict:
            parameter_type = _nf4_parameter_type()
            quantized_stats = {
                key[len(weight_key) + 1 :]: value
                for key, value in state_dict.items()
                if key.startswith(weight_key + ".")
            }
            try:
                weight = parameter_type.from_prequantized(
                    data=state_dict[weight_key],
                    quantized_stats=quantized_stats,
                    requires_grad=False,
                    device=anchor.device,
                    module=self,
                )
            except (
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ) as error:
                error_msgs.append(
                    f"failed to restore NF4 weight '{weight_key}' on {anchor.device}: {error}"
                )
                return
            quant_state = getattr(weight, "quant_state", None)
            if quant_state is None:
                error_msgs.append(
                    f"restored NF4 weight '{weight_key}' has no BitsAndBytes quantization state"
                )
                return
            expected_shape = (self.out_features, self.in_features)
            logical_shape = tuple(quant_state.shape)
            if logical_shape != expected_shape:
                error_msgs.append(
                    f"NF4 weight '{weight_key}' has logical shape {logical_shape}; "
                    f"expected {expected_shape}"
                )
                return
            self.weight = weight
            self.quant_state = quant_state
            self.is_nf4 = True
        else:
            weight = state_dict[weight_key]
            expected_shape = (self.out_features, self.in_features)
            if tuple(weight.shape) != expected_shape:
                error_msgs.append(
                    f"full-precision weight '{weight_key}' has shape {tuple(weight.shape)}; "
                    f"expected {expected_shape}"
                )
                return
            self.weight = torch.nn.Parameter(
                weight.to(device=anchor.device, dtype=anchor.dtype), requires_grad=False
            )

        if bias_key in state_dict:
            bias = state_dict[bias_key]
            expected_bias_shape = (self.out_features,)
            if tuple(bias.shape) != expected_bias_shape:
                error_msgs.append(
                    f"bias '{bias_key}' has shape {tuple(bias.shape)}; "
                    f"expected {expected_bias_shape}"
                )
                return
            self.bias = torch.nn.Parameter(
                bias.to(device=anchor.device, dtype=anchor.dtype),
                requires_grad=False,
            )
        elif strict and self.has_bias:
            missing_keys.append(bias_key)
        self.register_parameter("load_anchor", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"NF4 Linear {self.out_features}x{self.in_features} has no loaded weight"
            )
        if self.weight_function or self.bias_function:
            raise RuntimeError(
                "LoRA and weight-function patches are not supported on MiniMax H3 "
                "BitsAndBytes NF4 Linear layers"
            )
        if self.is_nf4:
            return self._forward_nf4(inputs)
        return self._forward_dense(inputs)

    # Weights reach the compute device only through core's cast path, as for
    # core's own layers: it stages them through dynamic VRAM, prefetch, pinned
    # memory or offload streams, and the stored parameters are never replaced.

    def _forward_dense(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy.ops

        weight, bias, offload = comfy.ops.cast_bias_weight(
            self, inputs, offloadable=True
        )
        try:
            return functional.linear(inputs, weight, bias)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload)

    def _forward_nf4(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy.ops

        bitsandbytes = _require_bitsandbytes()
        # Keep the stored dtype: the weight is packed 4-bit codes, not numbers.
        packed, bias, offload = comfy.ops.cast_bias_weight(
            self,
            inputs,
            dtype=self.weight.dtype,
            bias_dtype=inputs.dtype,
            offloadable=True,
        )
        try:
            output = bitsandbytes.matmul_4bit(
                inputs,
                packed.t(),
                bias=bias,
                quant_state=self._quant_state_on(packed.device),
            )
        finally:
            comfy.ops.uncast_bias_weight(self, packed, bias, offload)
        return output.to(dtype=inputs.dtype)

    def _quant_state_on(self, device: torch.device) -> QuantState:
        """The quantization state on `device`; copies it there, never moves it."""
        states = [
            state
            for state in (getattr(self.weight, "quant_state", None), self.quant_state)
            if state is not None
        ]
        if not states:
            raise RuntimeError(
                "loaded NF4 weight has no BitsAndBytes quantization state"
            )
        for state in states:
            if state.absmax.device == device:
                return state
        return type(states[0]).from_dict(states[0].as_dict(packed=False), device)

    def set_weight(
        self,
        weight: torch.Tensor,
        inplace_update: bool,
        seed: int,
        return_weight: bool,
    ) -> torch.Tensor | None:
        del inplace_update, seed
        if return_weight:
            return weight
        parameter_type = _nf4_parameter_type()
        if not isinstance(weight, parameter_type) or weight.quant_state is None:
            raise TypeError(
                "MiniMax H3 NF4 Linear can only accept a BitsAndBytes NF4 parameter"
            )
        self.weight = weight
        self.quant_state = weight.quant_state
        self.is_nf4 = True
        return None


class _NF4AudioCausalAttention(torch.nn.Module):
    """Use the NF4 QKV module instead of ComfyUI's direct weight access."""

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.head_dim = source.head_dim
        self.num_heads = source.num_heads
        self.out_dim = source.out_dim
        self.qkv = source.qkv
        self.q_bias = source.q_bias
        self.v_bias = source.v_bias
        self.register_buffer("zero_k_bias", source.zero_k_bias)
        self.proj = source.proj
        self.train(source.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy.ops

        batch, tokens, channels = inputs.shape
        bias = comfy.ops.cast_to_input(
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)), inputs
        )
        qkv = self.qkv(inputs) + bias
        q, k, v = (
            qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = comfy.ops.scaled_dot_product_attention(q, k, v, is_causal=True)
        pooled = functional.adaptive_avg_pool1d(
            torch.mean(attended, dim=1), self.out_dim
        )
        return self.proj(pooled)


class _NF4DiTAttention(torch.nn.Module):
    """Read DiffSynth's head-interleaved MiniMax H3 QKV layout."""

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.heads = source.heads
        self.head_dim = source.head_dim
        self.qkv_proj = source.qkv_proj
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.out_proj = source.out_proj
        self.train(source.training)

    def forward(
        self,
        inputs: torch.Tensor,
        rope_freqs=None,
        transformer_options=None,
    ) -> torch.Tensor:
        import comfy.model_management
        import comfy.quant_ops
        from comfy.ldm.modules.attention import optimized_attention

        options = {} if transformer_options is None else transformer_options
        tokens = inputs.shape[0]
        qkv = self.qkv_proj(inputs).view(
            tokens,
            self.heads,
            3,
            self.head_dim,
        )
        query = qkv[:, :, 0, :].contiguous()
        key = qkv[:, :, 1, :].contiguous()
        value = qkv[:, :, 2, :].contiguous()

        if rope_freqs is not None:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            query_weight = comfy.model_management.cast_to(
                self.q_norm.weight,
                device=inputs.device,
            )
            key_weight = comfy.model_management.cast_to(
                self.k_norm.weight,
                device=inputs.device,
            )
            rotation_dimension = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                query, key = comfy.quant_ops.ck.rms_rope_split_half(
                    query,
                    key,
                    rope_freqs,
                    query_weight,
                    key_weight,
                    epsilon=self.q_norm.eps,
                    rot_dim=rotation_dimension,
                )
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    query,
                    key,
                    rope_freqs,
                    query_weight,
                    key_weight,
                    epsilon=self.q_norm.eps,
                    rot_dim=rotation_dimension,
                )
            query = query[0]
            key = key[0]
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attended = optimized_attention(
            query,
            key,
            value,
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=options,
        )
        return self.out_proj(attended.squeeze(0))


def _nf4_operations():
    import comfy.ops

    class MiniMaxH3NF4Operations(comfy.ops.manual_cast):
        class Linear(_NF4Linear):
            pass

    return MiniMaxH3NF4Operations


def _replace_nf4_dit_attentions(root: torch.nn.Module) -> int:
    import comfy.ldm.minimax.model

    modules = dict(root.named_modules())
    replaced = 0
    for module_path, module in modules.items():
        if not isinstance(module, comfy.ldm.minimax.model.Attention):
            continue
        if not isinstance(module.qkv_proj, _NF4Linear):
            raise TypeError(
                f"MiniMax H3 attention '{module_path}' does not have an NF4-compatible QKV layer"
            )
        parent_path, separator, child_name = module_path.rpartition(".")
        parent = root if not separator else modules[parent_path]
        setattr(parent, child_name, _NF4DiTAttention(module))
        replaced += 1
    if replaced == 0:
        raise RuntimeError(
            "MiniMax H3 NF4 diffusion model contained no attention module"
        )
    return replaced


def _load_state_dict(
    path: str,
) -> tuple[dict[str, torch.Tensor], Mapping[str, str] | None]:
    import comfy.utils

    loaded, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )
    if not isinstance(loaded, dict):
        raise TypeError(f"NF4 checkpoint '{path}' did not contain a state dict")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                f"NF4 checkpoint '{path}' contains a non-tensor entry at key {key!r}"
            )
        state_dict[key] = value
    return state_dict, metadata


def _folder_path(folder_type: str, filename: str) -> str:
    import folder_paths

    path = folder_paths.get_full_path(folder_type, filename)
    if path is None:
        locations = folder_paths.get_folder_paths(folder_type)
        raise FileNotFoundError(
            f"NF4 file '{filename}' was not found in ComfyUI folder type "
            f"'{folder_type}'; searched {locations}"
        )
    return path


def _normalized_diffusion_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    import comfy.model_detection

    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    if not any(key.startswith(prefix) for key in state_dict):
        return dict(state_dict)
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_nf4_diffusion_model(path: str):
    _require_bitsandbytes()
    import comfy.model_detection
    import comfy.model_management
    import comfy.model_patcher

    raw_state_dict, metadata = _load_state_dict(path)
    state_dict = _normalized_diffusion_state_dict(raw_state_dict)
    shapes = nf4_quantized_shapes(state_dict)
    probe = make_shape_probe(state_dict, shapes)
    model_config = comfy.model_detection.model_config_from_unet(
        probe, "", metadata=metadata
    )
    if (
        model_config is None
        or model_config.unet_config.get("image_model") != "minimax_h3"
    ):
        detected = None if model_config is None else model_config.unet_config
        raise ValueError(
            f"NF4 checkpoint '{path}' is not a detectable MiniMax H3 diffusion model; "
            f"detected config={detected}"
        )

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    model_config.custom_operations = _nf4_operations()
    model_config.set_inference_dtype(torch.bfloat16, None, device=load_device)
    model = model_config.get_model(dict(state_dict), "")
    replaced = _replace_nf4_dit_attentions(model.diffusion_model)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(dict(state_dict), "", assign=patcher.is_dynamic())
    LOGGER.info(
        "Enabled MiniMax H3 NF4 interleaved QKV attention",
        extra={"path": path, "attention_count": replaced},
    )
    patcher.cached_patcher_init = (load_nf4_diffusion_model, (path,))
    return patcher


def load_nf4_text_encoder(path: str):
    _require_bitsandbytes()
    import comfy.sd
    import folder_paths

    raw_state_dict, _ = _load_state_dict(path)
    state_dict = convert_minimax_text_encoder_state_dict(raw_state_dict)
    nf4_quantized_shapes(state_dict)
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options={"custom_operations": _nf4_operations()},
    )
    clip.patcher.cached_patcher_init = (load_nf4_text_encoder_patcher, (path,))
    return clip


def load_nf4_text_encoder_patcher(path: str):
    return load_nf4_text_encoder(path).patcher


def _replace_quantized_linears(
    root: torch.nn.Module,
    shapes: Mapping[str, QuantizedShape],
) -> int:
    modules = dict(root.named_modules())
    replaced = 0
    for weight_key in shapes:
        if not weight_key.endswith(".weight"):
            raise ValueError(f"NF4 tensor '{weight_key}' is not a Linear weight")
        module_path = weight_key[: -len(".weight")]
        module = modules.get(module_path)
        if module is None:
            raise ValueError(
                f"NF4 tensor '{weight_key}' has no matching ComfyUI module"
            )
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                f"NF4 tensor '{weight_key}' maps to {type(module).__name__}, not Linear"
            )
        parent_path, separator, child_name = module_path.rpartition(".")
        parent = root if not separator else modules[parent_path]
        replacement = _NF4Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            module.weight.device,
            module.weight.dtype,
        )
        replacement.train(module.training)
        setattr(parent, child_name, replacement)
        replaced += 1
    return replaced


def _replace_nf4_audio_attentions(root: torch.nn.Module) -> int:
    import comfy.ldm.minimax.audio_vae

    modules = dict(root.named_modules())
    replaced = 0
    for module_path, module in modules.items():
        if not isinstance(module, comfy.ldm.minimax.audio_vae.CausalAttention):
            continue
        if not isinstance(module.qkv, _NF4Linear):
            raise TypeError(
                f"MiniMax H3 audio attention '{module_path}' does not have an NF4 QKV layer"
            )
        parent_path, separator, child_name = module_path.rpartition(".")
        parent = root if not separator else modules[parent_path]
        setattr(parent, child_name, _NF4AudioCausalAttention(module))
        replaced += 1
    if replaced == 0:
        raise RuntimeError(
            "MiniMax H3 NF4 audio VAE contained no causal attention module"
        )
    return replaced


def _load_nf4_vae(
    path: str,
    converted_state_dict: Mapping[str, torch.Tensor],
):
    _require_bitsandbytes()
    import comfy.sd

    shapes = nf4_quantized_shapes(converted_state_dict)
    probe = without_quantized_payload(converted_state_dict, shapes)
    vae = comfy.sd.VAE(sd=probe, metadata=None)
    vae.throw_exception_if_invalid()
    replaced = _replace_quantized_linears(vae.first_stage_model, shapes)
    missing, unexpected = vae.first_stage_model.load_state_dict(
        dict(converted_state_dict), strict=False, assign=vae.patcher.is_dynamic()
    )
    unsupported_missing = [
        key for key in missing if key not in {"latents_mean", "latents_std"}
    ]
    if unsupported_missing or unexpected:
        raise RuntimeError(
            f"NF4 VAE '{path}' did not match ComfyUI's MiniMax H3 VAE: "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    vae.size = None
    vae.patcher.size = 0
    vae.model_size()
    vae.patcher.model_size()
    LOGGER.info(
        "Loaded MiniMax H3 NF4 VAE",
        extra={"path": path, "quantized_linear_count": replaced},
    )
    return vae


def load_nf4_video_vae(path: str):
    state_dict, _ = _load_state_dict(path)
    vae = _load_nf4_vae(path, state_dict)
    vae.patcher.cached_patcher_init = (load_nf4_video_vae_patcher, (path,))
    return vae


def load_nf4_video_vae_patcher(path: str):
    return load_nf4_video_vae(path).patcher


def load_nf4_audio_vae(path: str):
    raw_state_dict, _ = _load_state_dict(path)
    state_dict = convert_minimax_audio_vae_state_dict(raw_state_dict)
    vae = _load_nf4_vae(path, state_dict)
    replaced = _replace_nf4_audio_attentions(vae.first_stage_model)
    LOGGER.info(
        "Enabled MiniMax H3 NF4 audio attention",
        extra={"path": path, "attention_count": replaced},
    )
    vae.patcher.cached_patcher_init = (load_nf4_audio_vae_patcher, (path,))
    return vae


def load_nf4_audio_vae_patcher(path: str):
    return load_nf4_audio_vae(path).patcher


class MiniMaxH3XNF4ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        return io.Schema(
            node_id="MiniMaxH3XNF4ModelLoader",
            is_experimental=True,
            display_name="MiniMax H3 NF4 Model Loader",
            category="model/loaders/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "nf4",
                "bitsandbytes",
                "4bit",
                "diffusion model",
                "unet",
                "loader",
                "模型加载",
                "量化",
            ],
            description=(
                "Load a DiffSynth-format MiniMax H3 NF4 diffusion model with "
                "BitsAndBytes."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip=(
                        "The NF4 FL2VA diffusion model, saved as minimax-h3-fl2va-nf4.safetensors in models/diffusion_models."
                    ),
                )
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        path = _folder_path("diffusion_models", model_name)
        return io.NodeOutput(load_nf4_diffusion_model(path))


class MiniMaxH3XNF4TextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        return io.Schema(
            node_id="MiniMaxH3XNF4TextEncoderLoader",
            is_experimental=True,
            display_name="MiniMax H3 NF4 Text Encoder Loader",
            category="model/loaders/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "nf4",
                "bitsandbytes",
                "4bit",
                "text encoder",
                "qwen3-vl",
                "clip",
                "loader",
                "文本编码器",
                "量化",
            ],
            description=("Load the DiffSynth-format MiniMax H3 Qwen3-VL NF4 encoder."),
            inputs=[
                io.Combo.Input(
                    "encoder_name",
                    options=folder_paths.get_filename_list("text_encoders"),
                    tooltip=(
                        "The NF4 Qwen3-VL text encoder, saved as minimax-h3-text-encoder-nf4.safetensors in models/text_encoders."
                    ),
                )
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, encoder_name: str) -> io.NodeOutput:
        path = _folder_path("text_encoders", encoder_name)
        return io.NodeOutput(load_nf4_text_encoder(path))


class MiniMaxH3XNF4VideoVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        return io.Schema(
            node_id="MiniMaxH3XNF4VideoVAELoader",
            is_experimental=True,
            display_name="MiniMax H3 NF4 Video VAE Loader",
            category="model/loaders/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "nf4",
                "bitsandbytes",
                "4bit",
                "video vae",
                "vae",
                "loader",
                "视频 VAE",
                "量化",
            ],
            description=("Load the DiffSynth-format MiniMax H3 NF4 video VAE."),
            inputs=[
                io.Combo.Input(
                    "vae_name",
                    options=folder_paths.get_filename_list("vae"),
                    tooltip=(
                        "The NF4 video VAE, saved as video_vae_nf4.safetensors in models/vae."
                    ),
                )
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name: str) -> io.NodeOutput:
        path = _folder_path("vae", vae_name)
        return io.NodeOutput(load_nf4_video_vae(path))


class MiniMaxH3XNF4AudioVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        return io.Schema(
            node_id="MiniMaxH3XNF4AudioVAELoader",
            is_experimental=True,
            display_name="MiniMax H3 NF4 Audio VAE Loader",
            category="model/loaders/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "nf4",
                "bitsandbytes",
                "4bit",
                "audio vae",
                "vae",
                "loader",
                "音频 VAE",
                "量化",
            ],
            description=("Load the DiffSynth-format MiniMax H3 NF4 audio VAE."),
            inputs=[
                io.Combo.Input(
                    "vae_name",
                    options=folder_paths.get_filename_list("vae"),
                    tooltip=(
                        "The NF4 audio VAE, saved as audio_vae_nf4.safetensors in models/vae."
                    ),
                )
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name: str) -> io.NodeOutput:
        path = _folder_path("vae", vae_name)
        return io.NodeOutput(load_nf4_audio_vae(path))


def get_nf4_nodes() -> list[type[io.ComfyNode]]:
    return [
        MiniMaxH3XNF4ModelLoader,
        MiniMaxH3XNF4TextEncoderLoader,
        MiniMaxH3XNF4VideoVAELoader,
        MiniMaxH3XNF4AudioVAELoader,
    ]
