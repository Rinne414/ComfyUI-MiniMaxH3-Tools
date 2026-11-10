"""Local prompt expander for MiniMax-H3-Image: a short prompt in any language -> the long English caption
the model wants, plus the aspect ratio the expander suggests.

Uses ComfyUI's own TextGenerateV2 with the instructions of the official Comfy MiniMax-H3-Image templates,
the same model they use (Qwen3.5-9B w4a8), and frees the model again before the DiT loads. The instruction
text is not part of this pack: it is read at run time from the templates installed with ComfyUI
(image_expand_instructions.py). Turned off, or when the templates are missing, the prompt passes through
and nothing is loaded.
"""

import os
import time
from pathlib import Path

import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerateV2

if __package__:
    from .image_cache import (
        cache_key,
        file_identity,
        read_text_cache,
        write_text_cache,
    )
    from .image_expand_instructions import (
        InstructionsNotFound,
        find_instructions,
    )
    from .image_expand_parse import parse_reply
    from .image_encode import (
        _available_mb,
        collect_released_memory,
        free_other_models,
        load_text_encoder,
    )
    from .image_picture_refs import connected_pictures
    from .image_pinning import pinning_disabled
else:
    from image_cache import cache_key, file_identity, read_text_cache, write_text_cache
    from image_expand_instructions import InstructionsNotFound, find_instructions
    from image_expand_parse import parse_reply
    from image_encode import (
        _available_mb,
        collect_released_memory,
        free_other_models,
        load_text_encoder,
    )
    from image_picture_refs import connected_pictures
    from image_pinning import pinning_disabled

DEFAULT_MODEL = "qwen3.5_9b_w4a8_gptq.safetensors"
CACHE_SUBFOLDER = "h3image_expand_cache"
# the official edit template shows the expander each picture at about 4 MP, sides multiples of 32
PICTURE_PIXELS = 4 * 1024 * 1024
_DESCRIPTION = (
    "Experimental. Expands a short prompt (any language) into the long "
    "English description MiniMax-H3-Image wants, using the instruction and "
    "model of the official template, then frees the model. Off: the prompt "
    "passes through."
)


def user_workflow_directories() -> list[Path]:
    """Folders where a copy of the official template saved by the user may live."""
    return [Path(folder_paths.get_user_directory()) / "default" / "workflows"]


def sampling(seed: int) -> dict:
    """The official template's sampling settings with a fixed seed."""
    return {
        "sampling_mode": "on",
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "seed": seed,
    }


def scaled_picture(image):
    """[B,H,W,C] -> first picture at about 4 MP, sides multiples of 32 (as the official edit template)."""
    h, w = image.shape[1], image.shape[2]
    scale = (PICTURE_PIXELS / (w * h)) ** 0.5
    nw, nh = (max(32, round(v * scale / 32) * 32) for v in (w, h))
    out = comfy.utils.common_upscale(
        image[:1].movedim(-1, 1), nw, nh, "area", "disabled"
    )
    return out.movedim(1, -1)


def generate(
    model_name: str,
    prompt: str,
    system: str,
    seed: int,
    max_length: int,
    pictures: list,
) -> str:
    """Load the text model without pinned RAM, generate once, free it; returns the reply without thinking."""
    free_other_models()
    images = {
        f"image{i}": scaled_picture(p) for i, p in enumerate(pictures, start=1)
    } or None
    with pinning_disabled(mm):
        clip = load_text_encoder(model_name)
        try:
            text, _thinking = TextGenerateV2.execute(
                clip=clip,
                prompt=prompt,
                max_length=max_length,
                sampling_mode=sampling(seed),
                thinking=True,
                use_default_template=True,
                mtp="off",
                system_prompt=system,
                images=images,
            ).args
        finally:
            mm.unload_model_and_clones(clip.patcher)
            clip = None
    collect_released_memory()
    return text or ""


def _schema_inputs() -> list:
    models = folder_paths.get_filename_list("text_encoders")
    return [
        io.String.Input(
            "prompt",
            multiline=True,
            tooltip="A short idea in any language.",
        ),
        io.Boolean.Input(
            "expand",
            default=True,
            tooltip="Off: the prompt is used as it is and no model loads.",
        ),
        io.Combo.Input(
            "text_model",
            options=models,
            default=DEFAULT_MODEL if DEFAULT_MODEL in models else None,
            tooltip=(
                "Experimental. The official template uses "
                "qwen3.5_9b_w4a8_gptq (7.3 GB)."
            ),
        ),
        io.Int.Input(
            "seed",
            default=0,
            min=0,
            max=0xFFFFFFFF,
            tooltip="Same seed and prompt give the same expansion.",
        ),
        io.Int.Input(
            "max_length",
            default=8192,
            min=512,
            max=16384,
            advanced=True,
            tooltip="Token budget including the model's thinking.",
        ),
        io.Autogrow.Input(
            "images",
            optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input(
                    "image",
                    tooltip="For edits: the pictures, <Picture 1> first.",
                ),
                prefix="image_",
                min=0,
                max=4,
            ),
        ),
    ]


def cached_generate(
    model_name: str,
    prompt: str,
    system: str,
    seed: int,
    max_length: int,
    pictures: list,
) -> tuple[str, bool]:
    """Same model, instruction, prompt, seed and pictures -> the reply saved on disk; the model does not load."""
    path_model = folder_paths.get_full_path_or_raise("text_encoders", model_name)
    key = cache_key(
        f"expand|{seed}|{max_length}|{system}|{prompt}",
        file_identity(path_model),
        "expander",
        pictures,
    )
    folder = os.path.join(folder_paths.get_output_directory(), CACHE_SUBFOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.json")
    reply = read_text_cache(path)
    if reply is not None:
        return reply, True
    reply = generate(model_name, prompt, system, seed, max_length, pictures)
    if parse_reply(reply)[0]:  # only a usable reply is kept
        write_text_cache(path, reply, {"model": model_name, "seed": seed})
    return reply, False


class MiniMaxH3XImagePromptExpand(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XImagePromptExpand",
            display_name="MiniMax H3 Image Prompt Expander",
            category="model/conditioning/minimax",
            search_aliases=[
                "h3", "minimax", "h3 image", "prompt expander",
                "expand prompt", "prompt enhance", "caption", "提示词扩写", "扩写",
                "图像提示词",
            ],
            description=_DESCRIPTION,
            inputs=_schema_inputs(),
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="aspect_ratio"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="raw_reply"),
            ],
        )

    @classmethod
    def execute(
        cls, prompt, expand, text_model, seed, max_length=8192, images=None
    ) -> io.NodeOutput:
        if not expand or not prompt.strip():
            return io.NodeOutput(
                prompt,
                "",
                "expansion off: the prompt is used unchanged, no model loaded",
                "",
            )
        pictures = connected_pictures(images)
        try:
            instructions = find_instructions(
                "edit" if pictures else "t2i",
                extra_directories=user_workflow_directories(),
            )
        except InstructionsNotFound as reason:
            return io.NodeOutput(prompt, "", str(reason), "")
        start = time.monotonic()
        reply, hit = cached_generate(
            text_model,
            instructions.task.replace("{input text}", prompt.strip()),
            instructions.system,
            seed,
            max_length,
            pictures,
        )
        caption, ratio, problem = parse_reply(reply)
        seconds = time.monotonic() - start
        if not caption:
            return io.NodeOutput(
                prompt,
                "",
                f"expansion failed after {seconds:.0f} s ({problem}); your prompt is used unchanged",
                reply,
            )
        report = (
            ("expander cache hit, model not loaded; " if hit else "")
            + f"expanded {len(prompt.strip())} -> {len(caption)} characters in {seconds:.0f} s; suggested ratio "
            f"{ratio or 'none'}; model freed, available RAM {_available_mb()} MB"
        )
        return io.NodeOutput(caption, ratio, report, reply)
