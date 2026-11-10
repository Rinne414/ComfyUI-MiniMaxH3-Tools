"""Low-RAM prompt encoding for MiniMax-H3-Image: load the text encoder, encode, release it.

The stock CLIPLoader keeps the 27 GB Qwen3-VL encoder (its pinned host buffers) alive in the output
cache, so on a 32 GB PC the DiT cannot load after it. This node owns the encoder for the duration of
one encode and frees it before returning; with an unchanged prompt ComfyUI's cache skips the node and
the encoder never loads.
"""

import gc
import os
import sys
import time

import psutil

import comfy.model_management as mm
import comfy.model_prefetch
import comfy.sd
import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

if __package__:
    from .conditioning_io import save_conditioning, zero_conditioning
    from .image_cache import cache_key, file_identity, folder_stats, read_cache
    from .image_gguf import (
        MISSING,
        find_gguf_loader,
        gguf_parts,
        is_gguf,
        vram_problem,
    )
    from .image_pinning import pinning_disabled
else:
    from conditioning_io import save_conditioning, zero_conditioning
    from image_cache import cache_key, file_identity, folder_stats, read_cache
    from image_gguf import (
        MISSING,
        find_gguf_loader,
        gguf_parts,
        is_gguf,
        vram_problem,
    )
    from image_pinning import pinning_disabled

# the encode ignores the canvas size; the latent comes from the size node instead
_ENCODE_SIZE = 2048
CACHE_SUBFOLDER = "h3image_encode_cache"
_DESCRIPTION = (
    "Loads the text encoder, encodes the prompt and pictures, then frees the "
    "encoder so the DiT fits in 32 GB RAM. An unchanged prompt is reused "
    "without loading the encoder."
)


def _available_mb() -> int:
    return int(psutil.virtual_memory().available / (1024 * 1024))


def free_other_models() -> None:
    """Drop the pinned RAM of models already loaded (the DiT of a previous run) before the 27 GB encoder
    loads; the DiT stays cached and re-stages from disk when the sampler needs it."""
    mm.unload_all_models()
    collect_released_memory()


def encoder_names() -> list:
    """Text encoders: .safetensors from text_encoders, plus .gguf files when ComfyUI-GGUF registered them."""
    names = set(folder_paths.get_filename_list("text_encoders"))
    if "clip_gguf" in folder_paths.folder_names_and_paths:
        names.update(folder_paths.get_filename_list("clip_gguf"))
    return sorted(names)


def encoder_path(clip_name: str) -> str:
    for folder in ("text_encoders", "clip_gguf"):
        if folder in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(folder, clip_name)
            if path:
                return path
    raise FileNotFoundError(f"text encoder not found: {clip_name}")


def load_text_encoder(clip_name: str):
    path = encoder_path(clip_name)
    if is_gguf(path):
        return load_gguf_encoder(path)
    return comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )


def load_gguf_encoder(path: str):
    """ComfyUI-GGUF's own reader, ops and patcher, but built on the GPU: its loader builds on the CPU and
    copied the whole quantized file into private host RAM (15 GB for Q4_K_M) on this core."""
    loader = find_gguf_loader(sys.modules)
    if loader is None:
        raise ValueError(MISSING)
    gguf = loader()
    parts = gguf_parts(loader, sys.modules)
    if parts is None:  # unknown pack layout: use its loader as it is
        return gguf.load_patcher(
            [path], comfy.sd.CLIPType.MINIMAX, gguf.load_data([path])
        )
    ops, patcher = parts
    device = mm.get_torch_device()
    problem = vram_problem(os.path.getsize(path), mm.get_free_memory(device))
    if problem:
        raise ValueError(problem)
    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=gguf.load_data([path]),
        clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options={
            "custom_operations": ops,
            "initial_device": device,
        },
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    clip.patcher = patcher.clone(clip.patcher)
    return clip


def unload_text_encoder(clip) -> None:
    """Free the encoder's VRAM pages and pinned host RAM now, not when the cache drops it."""
    mm.unload_model_and_clones(clip.patcher)


def collect_released_memory() -> None:
    """Run after the last reference to the encoder is gone: its pin entries form cycles."""
    comfy.model_prefetch.cleanup_prefetch_queues()
    mm.reset_cast_buffers()
    gc.collect()
    mm.cleanup_models()
    mm.soft_empty_cache()


def _schema_inputs() -> list:
    return [
        io.Combo.Input(
            "clip_name",
            options=encoder_names(),
            tooltip=(
                "The MiniMax H3 Qwen3-VL text encoder; recommended: "
                "qwen3vl_32b_minimax_h3_int8_convrot. .gguf files are "
                "experimental (unverified, need the ComfyUI-GGUF pack)."
            ),
        ),
        io.String.Input(
            "prompt",
            multiline=True,
            dynamic_prompts=True,
            tooltip=(
                "A long, detailed English description works best; refer "
                "to pictures as <Picture 1>, ..."
            ),
        ),
        io.Boolean.Input(
            "use_cache",
            default=True,
            tooltip=(
                "Keep encoded prompts in output/h3image_encode_cache so "
                "the encoder never loads again for the same prompt and "
                "pictures, also after a restart."
            ),
        ),
        io.Vae.Input(
            "vae",
            optional=True,
            tooltip=(
                "The MiniMax-H3-Image VAE; needed for editing so the "
                "pictures are also anchored as latents."
            ),
        ),
        io.Autogrow.Input(
            "images",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input(
                    "image",
                    tooltip="<Picture n>; image_1 is the picture to edit.",
                ),
                prefix="image_",
                min=0,
                max=16,
            ),
        ),
    ]


class MiniMaxH3XImageEncodeLowRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XImageEncodeLowRAM",
            display_name="MiniMax H3 Image Encode (Low RAM)",
            category="model/conditioning/minimax",
            search_aliases=[
                "h3", "minimax", "h3 image", "low ram", "text encode",
                "prompt encode", "conditioning", "qwen3-vl", "gguf", "低内存",
                "提示词编码", "图像",
            ],
            description=_DESCRIPTION,
            inputs=_schema_inputs(),
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls, clip_name, prompt, use_cache=True, vae=None, images=None
    ) -> io.NodeOutput:
        pictures = [images[n] for n in sorted(images or {}) if images[n] is not None]
        path = None
        note = "cache off"
        if use_cache:
            key = cache_key(
                prompt,
                file_identity(encoder_path(clip_name)),
                vae_identity(vae),
                pictures,
            )
            path = os.path.join(cache_folder(), f"{key}.safetensors")
            cond, note = read_cache(path)
            if cond is not None:
                return io.NodeOutput(
                    cond, zero_conditioning(cond), f"{note}; {cache_summary()}"
                )
        before = _available_mb()
        start = time.monotonic()
        cond = encode_once(clip_name, prompt, vae, images)
        seconds = time.monotonic() - start
        if path is not None:
            save_conditioning(path, cond, {"prompt": prompt, "text_encoder": clip_name})
        report = (
            "%s; encoded with %s in %.1f s and released it; available RAM %d MB before load, %d MB after release"
            % (note, clip_name, seconds, before, _available_mb())
        )
        if path is not None:
            report += "; " + cache_summary()
        return io.NodeOutput(cond, zero_conditioning(cond), report)


def cache_folder() -> str:
    folder = os.path.join(folder_paths.get_output_directory(), CACHE_SUBFOLDER)
    os.makedirs(folder, exist_ok=True)
    return folder


def cache_summary() -> str:
    count, size = folder_stats(cache_folder())
    return f"cache folder output/{CACHE_SUBFOLDER}: {count} files, {size / (1024 * 1024):.1f} MB (delete files there to clear it)"


def vae_identity(vae) -> str:
    if vae is None:
        return "no vae"
    model = vae.first_stage_model
    return f"{type(model).__name__}|{sum(p.numel() for p in model.parameters())}"


def encode_once(clip_name, prompt, vae, images):
    free_other_models()
    with pinning_disabled(mm):  # no pinned host RAM for the one-shot encoder
        clip = load_text_encoder(clip_name)
        try:
            cond = MiniMaxH3ImageConditioning.execute(
                clip, prompt, _ENCODE_SIZE, _ENCODE_SIZE, vae=vae, images=images
            ).result[0]
        finally:
            unload_text_encoder(clip)
            clip = None
    collect_released_memory()
    return cond
