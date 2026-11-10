"""Generate the public MiniMax-H3 example workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

WidgetValue = str | int | float | bool
Slot = tuple[str, str]
NodeSchema = tuple[tuple[Slot, ...], tuple[Slot, ...]]
ConnectionSpec = tuple[str, str, int]
NodeSpec = tuple[
    str,
    str,
    tuple[int, int],
    tuple[int, int],
    tuple[WidgetValue, ...],
    tuple[ConnectionSpec, ...],
    str | None,
]
Link = list[int | str]


class WorkflowInput(TypedDict):
    name: str
    type: str
    link: int


class WorkflowOutput(TypedDict):
    name: str
    type: str
    links: list[int]
    slot_index: int


class WorkflowNodeRequired(TypedDict):
    id: int
    type: str
    pos: list[int]
    size: list[int]
    flags: dict[str, bool]
    order: int
    mode: int
    inputs: list[WorkflowInput]
    outputs: list[WorkflowOutput]
    properties: dict[str, str | list[dict[str, str]]]
    widgets_values: list[WidgetValue]


class WorkflowNode(WorkflowNodeRequired, total=False):
    title: str


class Workflow(TypedDict):
    id: str
    revision: int
    last_node_id: int
    last_link_id: int
    nodes: list[WorkflowNode]
    links: list[Link]
    groups: list[dict[str, str]]
    config: dict[str, str]
    extra: dict[str, dict[str, float | list[int]]]
    version: float


HUGGINGFACE_URL = "https://huggingface.co"

# File name -> (Hugging Face repository, folder inside it and in ComfyUI's models
# directory). ComfyUI shows a download link for a file that is missing. Every entry
# was checked against the repository's file listing.
MODEL_SOURCES: dict[str, tuple[str, str]] = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": (
        "Comfy-Org/MiniMax-H3", "diffusion_models"),
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": (
        "Comfy-Org/MiniMax-H3", "diffusion_models"),
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors": (
        "Comfy-Org/MiniMax-H3", "text_encoders"),
    "minimax_h3_video_vae_fp16.safetensors": ("Comfy-Org/MiniMax-H3", "vae"),
    "minimax_h3_audio_vae_fp32.safetensors": ("Comfy-Org/MiniMax-H3", "vae"),
    "minimax_h3_image_pruned_w4a8.safetensors": (
        "Comfy-Org/MiniMax-H3-Image", "diffusion_models"),
    "minimax_h3_image_vae_fp16.safetensors": ("Comfy-Org/MiniMax-H3-Image", "vae"),
    "qwen3.5_9b_w4a8_gptq.safetensors": ("Comfy-Org/Qwen3.5", "text_encoders"),
}
MODEL_DIRECTORIES: dict[str, str] = {
    name: source[1] for name, source in MODEL_SOURCES.items()
}

# Node type -> index of the widget that holds the model file name.
MODEL_WIDGET_INDEX: dict[str, int] = {
    "UNETLoader": 0,
    "CLIPLoader": 0,
    "VAELoader": 0,
    "MiniMaxH3XImageEncodeLowRAM": 0,
    "MiniMaxH3XImagePromptExpand": 2,
}
LOADER_TYPES = frozenset(MODEL_WIDGET_INDEX)


def model_url(name: str) -> str:
    repository, directory = MODEL_SOURCES[name]
    return "%s/%s/resolve/main/%s/%s" % (HUGGINGFACE_URL, repository, directory, name)


def model_download_properties(
        node_type: str, widgets: tuple[WidgetValue, ...]) -> list[dict[str, str]]:
    """Return the ``properties.models`` entry for a model-loading node, or an empty list."""
    index = MODEL_WIDGET_INDEX.get(node_type)
    if index is None or len(widgets) <= index:
        return []
    name = str(widgets[index])
    if name not in MODEL_SOURCES:
        return []
    return [{
        "name": name,
        "url": model_url(name),
        "directory": MODEL_SOURCES[name][1],
    }]


# Core "Preview as Text" accepts any type on its single input.
ANY_TYPE = "*"

NODE_SCHEMAS: dict[str, NodeSchema] = {
    "MarkdownNote": ((), ()),
    "MiniMaxH3XCanvasSolver": (
        (),
        (
            ("width", "INT"),
            ("height", "INT"),
            ("frames", "INT"),
            ("latent_t", "INT"),
            ("audio_t", "INT"),
            ("video_tokens", "INT"),
            ("est_sit_seconds", "FLOAT"),
            ("est_total_minutes", "FLOAT"),
            ("report", "STRING"),
        ),
    ),
    "MiniMaxH3XPromptLint": (
        (),
        (("prompt", "STRING"), ("warnings", "STRING"), ("is_clean", "BOOLEAN")),
    ),
    "EmptyMiniMaxH3LatentAV": ((), (("LATENT", "LATENT"),)),
    "MiniMaxH3XSplitAV": (
        (("latent", "LATENT"),),
        (("video", "LATENT"), ("audio", "LATENT")),
    ),
    "MiniMaxH3XUpscaleVideoLatent": (
        (("latent", "LATENT"),),
        (("LATENT", "LATENT"),),
    ),
    "MiniMaxH3XJoinAV": (
        (("video", "LATENT"), ("audio", "LATENT")),
        (("LATENT", "LATENT"),),
    ),
    "MiniMaxH3XTemporalTrim": (
        (("latent", "LATENT"),),
        (("LATENT", "LATENT"), ("frames", "INT")),
    ),
    "MiniMaxH3XLatentInfo": (
        (("latent", "LATENT"),),
        (("report", "STRING"), ("frames", "INT"), ("video_tokens", "INT")),
    ),
    "MiniMaxH3XAudioReroll": (
        (("conditioning", "CONDITIONING"), ("av_latent", "LATENT")),
        (
            ("positive", "CONDITIONING"),
            ("latent", "LATENT"),
            ("video_passthrough", "LATENT"),
            ("info", "STRING"),
        ),
    ),
    "UNETLoader": ((), (("MODEL", "MODEL"),)),
    "CLIPLoader": ((), (("CLIP", "CLIP"),)),
    "VAELoader": ((), (("VAE", "VAE"),)),
    "MiniMaxH3ReferenceToVideo": (
        (("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE")),
        (("positive", "CONDITIONING"), ("LATENT", "LATENT")),
    ),
    "MiniMaxH3SigmaShift": (
        (("model", "MODEL"),),
        (("MODEL", "MODEL"),),
    ),
    "BasicGuider": (
        (("model", "MODEL"), ("conditioning", "CONDITIONING")),
        (("GUIDER", "GUIDER"),),
    ),
    "KSamplerSelect": ((), (("SAMPLER", "SAMPLER"),)),
    "BasicScheduler": (
        (("model", "MODEL"),),
        (("SIGMAS", "SIGMAS"),),
    ),
    "RandomNoise": ((), (("NOISE", "NOISE"),)),
    "SamplerCustomAdvanced": (
        (
            ("noise", "NOISE"),
            ("guider", "GUIDER"),
            ("sampler", "SAMPLER"),
            ("sigmas", "SIGMAS"),
            ("latent_image", "LATENT"),
        ),
        (("output", "LATENT"), ("denoised_output", "LATENT")),
    ),
    "VAEDecode": (
        (("samples", "LATENT"), ("vae", "VAE")),
        (("IMAGE", "IMAGE"),),
    ),
    "VAEDecodeAudio": (
        (("samples", "LATENT"), ("vae", "VAE")),
        (("AUDIO", "AUDIO"),),
    ),
    "CreateVideo": (
        (("images", "IMAGE"), ("audio", "AUDIO")),
        (("VIDEO", "VIDEO"),),
    ),
    "PreviewAny": ((("source", ANY_TYPE),), (("STRING", "STRING"),)),
    "SaveVideo": ((("video", "VIDEO"),), ()),
    "SaveAudio": ((("audio", "AUDIO"),), ()),
}

PLANNING_PROMPT = """integrated_multimodal_description: [Shot 1] A locked medium shot of a ceramic artist shaping a bowl at a sunlit workbench.
overall_soundscape: Quiet studio room tone, wet clay turning on the wheel, and soft tool contact.
non_diegetic_music: N/A"""

PLANNING_NOTE = """# Planning and prompt check / 规划与提示词检查

**English:** Use the canvas planner to calculate a legal H3 size and frame count before loading models. Use Prompt Lint to validate the documented H3 prompt structure. This workflow does not load a model. The two Preview as Text nodes show the planner report and the lint result on the canvas; the same text is also recorded with the job in the queue and history.

**简体中文：** 在加载模型前，使用画布规划节点计算合法的 H3 尺寸和帧数；使用提示词检查节点验证 H3 文档规定的提示词结构。本工作流不加载模型。两个“Preview as Text”节点会在画布上显示规划报告和检查结果，同样的文字也会随任务记录在队列和历史中。"""

AV_LATENT_NOTE = """# H3 AV latent tools / H3 音视频潜空间工具

**English:** This chain demonstrates the safe order for H3's paired video and audio latent streams: create, resize video only, split, strictly rejoin, trim both streams together, then inspect the result in the Preview as Text node. It does not decode or save media.

**简体中文：** 此链路演示 H3 视频与音频成对潜空间的安全处理顺序：创建、仅调整视频尺寸、拆分、严格合并、同步裁剪两个流，最后在“Preview as Text”节点中查看检查结果。它不会解码或保存媒体。"""

AUDIO_REROLL_NOTE = """# Experimental audio reroll / 实验性音频重生成

**English:** Stage 1 generates the original AV latent. Stage 2 changes the sound text, uses the finished video as a strong reference, samples new audio, and muxes it with the exact original picture. This is experimental and includes two full H3 sampling runs; audio reroll can cost more than generating from scratch.

**简体中文：** 阶段 1 生成原始音视频潜空间。阶段 2 修改声音文字，将完成的视频作为强参考，采样新音频，并与完全一致的原始画面混流。此流程属于实验功能，包含两次完整 H3 采样；音频重生成的成本可能高于从头生成。"""

REROLL_VISUAL = """integrated_multimodal_description: [Shot 1] Live-action cinematic medium close-up of a lighthouse keeper in a weathered yellow oilskin coat on a wet stone balcony at dusk. The keeper grips a rusted railing, turns toward the lens, and says: <d>[English] The storm is coming. Secure the boats.</d> The keeper closes their mouth and turns back toward the sea."""

REROLL_SOUND_ORIGINAL = """overall_soundscape: Heavy rain on oilskin, deep ocean swells breaking against rock, and gusting wind around the iron railing.
non_diegetic_music: Sparse tense orchestral underscore with a sustained low cello drone and distant solo violin, no percussion."""

REROLL_SOUND_NEW = """overall_soundscape: A dry still evening, distant cathedral bells, gravel shifting underfoot, and a lone gull calling twice far away.
non_diegetic_music: Warm solo acoustic guitar with soft fingerpicking and quiet upright bass, no strings and no swell."""


def _node(
        key: str,
        node_type: str,
        position: tuple[int, int],
        size: tuple[int, int],
        widgets: tuple[WidgetValue, ...],
        connections: tuple[ConnectionSpec, ...],
        title: str | None,
) -> NodeSpec:
    return key, node_type, position, size, widgets, connections, title


def _render_workflow(
        workflow_id: str,
        specs: tuple[NodeSpec, ...],
        scale: float,
        offset: tuple[int, int],
) -> Workflow:
    keys = [spec[0] for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("workflow '%s' contains duplicate node keys" % workflow_id)

    id_by_key = {key: index + 1 for index, key in enumerate(keys)}
    type_by_key = {spec[0]: spec[1] for spec in specs}
    links: list[Link] = []
    input_links: dict[tuple[int, int], int] = {}
    output_links: dict[tuple[int, int], list[int]] = {}

    for target_spec in specs:
        target_key, target_type = target_spec[0], target_spec[1]
        if target_type not in NODE_SCHEMAS:
            raise ValueError("workflow '%s' uses unknown node type '%s'" % (
                workflow_id, target_type))
        connections = {item[0]: (item[1], item[2]) for item in target_spec[5]}
        input_slots = NODE_SCHEMAS[target_type][0]
        input_names = {slot[0] for slot in input_slots}
        unknown_inputs = sorted(set(connections) - input_names)
        if unknown_inputs:
            raise ValueError("node '%s' connects unknown inputs: %s" % (
                target_key, ", ".join(unknown_inputs)))

        for input_index, (input_name, input_type) in enumerate(input_slots):
            if input_name not in connections:
                raise ValueError("node '%s' is missing connection '%s'" % (
                    target_key, input_name))
            source_key, output_index = connections[input_name]
            if source_key not in id_by_key:
                raise ValueError("node '%s' references unknown source '%s'" % (
                    target_key, source_key))
            source_type = type_by_key[source_key]
            source_outputs = NODE_SCHEMAS[source_type][1]
            if output_index < 0 or output_index >= len(source_outputs):
                raise ValueError("node '%s' references invalid output %d on '%s'" % (
                    target_key, output_index, source_key))
            output_type = source_outputs[output_index][1]
            if input_type != ANY_TYPE and output_type != input_type:
                raise ValueError(
                    "type mismatch: %s.%s expects %s but %s[%d] returns %s"
                    % (target_key, input_name, input_type, source_key, output_index, output_type)
                )

            link_id = len(links) + 1
            source_id = id_by_key[source_key]
            target_id = id_by_key[target_key]
            links.append([
                link_id,
                source_id,
                output_index,
                target_id,
                input_index,
                output_type,
            ])
            input_links[(target_id, input_index)] = link_id
            output_links.setdefault((source_id, output_index), []).append(link_id)

    nodes: list[WorkflowNode] = []
    for order, spec in enumerate(specs):
        key, node_type, position, size, widgets, _, title = spec
        node_id = id_by_key[key]
        input_slots, output_slots = NODE_SCHEMAS[node_type]
        node: WorkflowNode = {
            "id": node_id,
            "type": node_type,
            "pos": list(position),
            "size": list(size),
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [
                {
                    "name": name,
                    "type": slot_type,
                    "link": input_links[(node_id, input_index)],
                }
                for input_index, (name, slot_type) in enumerate(input_slots)
            ],
            "outputs": [
                {
                    "name": name,
                    "type": slot_type,
                    "links": output_links.get((node_id, output_index), []),
                    "slot_index": output_index,
                }
                for output_index, (name, slot_type) in enumerate(output_slots)
            ],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": list(widgets),
        }
        models = model_download_properties(node_type, widgets)
        if models:
            node["properties"]["models"] = models
        if title is not None:
            node["title"] = title
        nodes.append(node)

    return {
        "id": workflow_id,
        "revision": 0,
        "last_node_id": len(nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": scale, "offset": list(offset)}},
        "version": 0.4,
    }


def _planning_workflow() -> Workflow:
    specs = (
        _node(
            "note",
            "MarkdownNote",
            (60, 40),
            (1080, 240),
            (PLANNING_NOTE,),
            (),
            "How to use / 使用说明",
        ),
        _node(
            "planner",
            "MiniMaxH3XCanvasSolver",
            (60, 330),
            (480, 330),
            (
                "16:9 (Widescreen)",
                0.4,
                5.0,
                20,
                "none (report tokens only)",
                24.0,
                False,
            ),
            (),
            "Plan exact H3 geometry / 规划精确 H3 画布",
        ),
        _node(
            "lint",
            "MiniMaxH3XPromptLint",
            (600, 330),
            (600, 430),
            (PLANNING_PROMPT, "t2va", 5.0, 0, 0, 0, False),
            (),
            "Validate H3 prompt / 检查 H3 提示词",
        ),
        _node(
            "planner_preview",
            "PreviewAny",
            (60, 700),
            (480, 260),
            (),
            (("source", "planner", 8),),
            "Planner report / 规划报告",
        ),
        _node(
            "lint_preview",
            "PreviewAny",
            (600, 800),
            (600, 160),
            (),
            (("source", "lint", 1),),
            "Lint result / 检查结果",
        ),
    )
    return _render_workflow("minimax-h3-planning-and-prompt-lint", specs, 0.8, (40, 20))


def _av_latent_tools_workflow() -> Workflow:
    specs = (
        _node(
            "note",
            "MarkdownNote",
            (40, 40),
            (1040, 240),
            (AV_LATENT_NOTE,),
            (),
            "How to use / 使用说明",
        ),
        _node(
            "empty",
            "EmptyMiniMaxH3LatentAV",
            (40, 350),
            (400, 120),
            (864, 480, 243),
            (),
            "Create 10-second H3 AV latent / 创建 10 秒 H3 音视频潜空间",
        ),
        _node(
            "upscale",
            "MiniMaxH3XUpscaleVideoLatent",
            (500, 330),
            (390, 180),
            (1216, 672, "bicubic"),
            (("latent", "empty", 0),),
            "Resize video only / 仅调整视频尺寸",
        ),
        _node(
            "split",
            "MiniMaxH3XSplitAV",
            (950, 330),
            (380, 100),
            (),
            (("latent", "upscale", 0),),
            "Split video and audio / 拆分视频和音频流",
        ),
        _node(
            "join",
            "MiniMaxH3XJoinAV",
            (1390, 310),
            (400, 140),
            (True,),
            (("video", "split", 0), ("audio", "split", 1)),
            "Strictly rejoin streams / 严格合并匹配的流",
        ),
        _node(
            "trim",
            "MiniMaxH3XTemporalTrim",
            (1850, 310),
            (410, 130),
            (5.0,),
            (("latent", "join", 0),),
            "Trim both streams to about 5s / 将两个流裁剪到约 5 秒",
        ),
        _node(
            "info",
            "MiniMaxH3XLatentInfo",
            (2320, 310),
            (440, 180),
            (),
            (("latent", "trim", 0),),
            "Inspect AV shape and duration / 检查音视频形状和时长",
        ),
        _node(
            "info_preview",
            "PreviewAny",
            (2320, 540),
            (440, 260),
            (),
            (("source", "info", 0),),
            "AV latent report / 音视频潜空间报告",
        ),
    )
    return _render_workflow("minimax-h3-av-latent-tools", specs, 0.65, (20, 40))


def _audio_reroll_workflow() -> Workflow:
    prompt_original = REROLL_VISUAL + "\n" + REROLL_SOUND_ORIGINAL
    prompt_new = REROLL_VISUAL + "\n" + REROLL_SOUND_NEW
    specs = (
        _node(
            "note",
            "MarkdownNote",
            (40, -320),
            (1040, 280),
            (AUDIO_REROLL_NOTE,),
            (),
            "How to use / 使用说明",
        ),
        _node(
            "unet",
            "UNETLoader",
            (40, 40),
            (520, 100),
            ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default"),
            (),
            "MiniMax-H3 ref2va model / ref2va 模型",
        ),
        _node(
            "clip",
            "CLIPLoader",
            (40, 180),
            (520, 120),
            ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "default"),
            (),
            "MiniMax-H3 text encoder / 文本编码器",
        ),
        _node(
            "video_vae",
            "VAELoader",
            (40, 340),
            (520, 80),
            ("minimax_h3_video_vae_fp16.safetensors",),
            (),
            "Video VAE / 视频 VAE",
        ),
        _node(
            "audio_vae",
            "VAELoader",
            (40, 460),
            (520, 80),
            ("minimax_h3_audio_vae_fp32.safetensors",),
            (),
            "Audio VAE / 音频 VAE",
        ),
        _node(
            "condition_original",
            "MiniMaxH3ReferenceToVideo",
            (640, 40),
            (520, 360),
            (prompt_original, 864, 480, 124, "match"),
            (
                ("clip", "clip", 0),
                ("vae", "video_vae", 0),
                ("audio_vae", "audio_vae", 0),
            ),
            "Stage 1 conditioning / 阶段 1 条件",
        ),
        _node(
            "shift",
            "MiniMaxH3SigmaShift",
            (640, 450),
            (420, 100),
            (12.0, 3.0),
            (("model", "unet", 0),),
            "Shared H3 sigma schedule / 共用 H3 sigma 调度",
        ),
        _node(
            "guider_original",
            "BasicGuider",
            (1220, 40),
            (340, 80),
            (),
            (("model", "shift", 0), ("conditioning", "condition_original", 0)),
            "Stage 1 guider / 阶段 1 引导器",
        ),
        _node(
            "sampler_select",
            "KSamplerSelect",
            (1220, 160),
            (340, 80),
            ("res_multistep",),
            (),
            "Sampler / 采样器",
        ),
        _node(
            "scheduler",
            "BasicScheduler",
            (1220, 280),
            (340, 130),
            ("simple", 20, 1.0),
            (("model", "shift", 0),),
            "Scheduler / 调度器",
        ),
        _node(
            "noise_original",
            "RandomNoise",
            (1220, 450),
            (340, 90),
            (4242, "fixed"),
            (),
            "Stage 1 seed / 阶段 1 种子",
        ),
        _node(
            "sample_original",
            "SamplerCustomAdvanced",
            (1600, 80),
            (420, 180),
            (),
            (
                ("noise", "noise_original", 0),
                ("guider", "guider_original", 0),
                ("sampler", "sampler_select", 0),
                ("sigmas", "scheduler", 0),
                ("latent_image", "condition_original", 1),
            ),
            "Sample original AV / 采样原始音视频",
        ),
        _node(
            "condition_new",
            "MiniMaxH3ReferenceToVideo",
            (640, 650),
            (520, 360),
            (prompt_new, 864, 480, 124, "match"),
            (
                ("clip", "clip", 0),
                ("vae", "video_vae", 0),
                ("audio_vae", "audio_vae", 0),
            ),
            "Stage 2 changed sound / 阶段 2 修改声音",
        ),
        _node(
            "reroll",
            "MiniMaxH3XAudioReroll",
            (1220, 650),
            (460, 190),
            (1.0, True),
            (
                ("conditioning", "condition_new", 0),
                ("av_latent", "sample_original", 0),
            ),
            "Experimental audio reroll / 实验性音频重生成",
        ),
        _node(
            "guider_new",
            "BasicGuider",
            (1730, 650),
            (340, 80),
            (),
            (("model", "shift", 0), ("conditioning", "reroll", 0)),
            "Stage 2 guider / 阶段 2 引导器",
        ),
        _node(
            "noise_new",
            "RandomNoise",
            (1730, 770),
            (340, 90),
            (4242, "fixed"),
            (),
            "Stage 2 seed / 阶段 2 种子",
        ),
        _node(
            "sample_new",
            "SamplerCustomAdvanced",
            (2130, 650),
            (420, 180),
            (),
            (
                ("noise", "noise_new", 0),
                ("guider", "guider_new", 0),
                ("sampler", "sampler_select", 0),
                ("sigmas", "scheduler", 0),
                ("latent_image", "reroll", 1),
            ),
            "Sample new audio / 采样新音频",
        ),
        _node(
            "decode_original_video",
            "VAEDecode",
            (2610, 500),
            (420, 90),
            (),
            (("samples", "reroll", 2), ("vae", "video_vae", 0)),
            "Decode original picture / 解码原始画面",
        ),
        _node(
            "decode_new_audio",
            "VAEDecodeAudio",
            (2610, 650),
            (420, 90),
            (),
            (("samples", "sample_new", 0), ("vae", "audio_vae", 0)),
            "Decode rerolled audio / 解码重生成音频",
        ),
        _node(
            "create_video",
            "CreateVideo",
            (3090, 500),
            (420, 130),
            (24.0, 8),
            (("images", "decode_original_video", 0), ("audio", "decode_new_audio", 0)),
            "Original picture + new audio / 原始画面 + 新音频",
        ),
        _node(
            "save_video",
            "SaveVideo",
            (3570, 420),
            (440, 320),
            ("video/MiniMax_H3_Audio_Reroll", "auto", "auto"),
            (("video", "create_video", 0),),
            "Save rerolled video / 保存重生成视频",
        ),
        _node(
            "save_audio",
            "SaveAudio",
            (3570, 790),
            (440, 120),
            ("audio/MiniMax_H3_Audio_Reroll",),
            (("audio", "decode_new_audio", 0),),
            "Save rerolled audio / 单独保存重生成音频",
        ),
    )
    return _render_workflow("minimax-h3-audio-reroll-experimental", specs, 0.5, (40, 300))


def build_workflows() -> tuple[tuple[str, Workflow], ...]:
    return (
        ("MiniMax-H3_Planning_and_Prompt_Lint.json", _planning_workflow()),
        ("MiniMax-H3_AV_Latent_Tools.json", _av_latent_tools_workflow()),
        ("MiniMax-H3_Audio_Reroll_Experimental.json", _audio_reroll_workflow()),
    )


def write_workflows(
        output_directory: Path,
        workflows: tuple[tuple[str, Workflow], ...],
) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, workflow in workflows:
        output_path = output_directory / filename
        output_path.write_text(
            json.dumps(workflow, ensure_ascii=True, indent=1) + "\n",
            encoding="utf-8",
        )
        written.append(output_path)
    return tuple(written)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    paths = write_workflows(repository_root / "example_workflows", build_workflows())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
