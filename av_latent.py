"""AV latent utilities for MiniMax-H3.

H3 latents are NestedTensor pairs:

    video: [B, 24, latent_t, H//16, W//16]
    audio: [B, 32, 2, audio_t]          # dim 2 is L/R

Every stock latent node (LatentUpscale, LatentBlend, LatentComposite, ...)
assumes a plain tensor and will either raise or silently corrupt the pair.
These nodes let a workflow reach one stream without disturbing the other.

Split emits ordinary {"samples": tensor} dicts so downstream tooling works, and
tags them so Join can tell a video latent from an audio one instead of trusting
connection order. A noise_mask travels with its stream through every node.
"""

import comfy.nested_tensor
import comfy.utils
import torch
from comfy_api.latest import ComfyExtension, io, ui

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID_MOD = 17
FRAME_GRID_REM = 5
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
LATENT_SPATIAL_DIVISOR = 16  # VAE spatial compression
TOKEN_SPATIAL_DIVISOR = 32  # after the DiT's 1x2x2 patchify

STREAM_KEY = "minimax_h3_stream"
MASK_KEY = "noise_mask"


def frames_from_latent_t(latent_t):
    """Inverse of video_latent_t(). Returns None when latent_t is off-grid."""
    if latent_t == 2:
        return 5
    if latent_t < 2 or (latent_t - 2) % 5 != 0:
        return None
    return ((latent_t - 2) // 5) * FRAME_GRID_MOD + FRAME_GRID_REM


def video_latent_t(frame_count):
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // FRAME_GRID_MOD) * 5 + 2


def audio_t_for_frames(frame_count):
    return round(frame_count / FPS * AUDIO_LATENT_FPS)


def unpack_av(latent):
    """(video, audio) from an H3 latent dict. Raises with a useful message."""
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent has no 'samples' key")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError(
            "expected a MiniMax-H3 NestedTensor AV latent, got %s with shape %s. "
            "Plain latents from other models are not AV pairs."
            % (type(samples).__name__, getattr(samples, "shape", "?"))
        )
    parts = samples.unbind()
    if len(parts) != 2:
        raise ValueError(
            "expected a 2-stream NestedTensor, got %d streams" % len(parts)
        )
    return parts[0], parts[1]


def unpack_masks(latent):
    """(video_mask, audio_mask) from an H3 latent dict, None where there is none.

    Core's sampler reads a NestedTensor noise_mask as one mask per stream and a
    plain tensor as the video stream's mask, with the audio fully denoised.
    """
    mask = latent.get(MASK_KEY)
    if mask is None:
        return None, None
    if not isinstance(mask, comfy.nested_tensor.NestedTensor):
        return mask, None
    parts = mask.unbind()
    if len(parts) != 2:
        raise ValueError("expected a 2-stream noise_mask, got %d streams" % len(parts))
    return parts[0], parts[1]


def pack_av(video, audio, video_mask=None, audio_mask=None):
    """An H3 latent dict. With any mask, a stream without one gets ones (fully
    denoised), the same as core's Concat AV Latent node."""
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if video_mask is not None or audio_mask is not None:
        latent[MASK_KEY] = comfy.nested_tensor.NestedTensor(
            (
                torch.ones_like(video) if video_mask is None else video_mask,
                torch.ones_like(audio) if audio_mask is None else audio_mask,
            )
        )
    return latent


def keep_mask(output, source):
    """Carry the source latent's noise_mask, unchanged, into a node's output."""
    if source.get(MASK_KEY) is not None:
        output[MASK_KEY] = source[MASK_KEY]
    return output


def stream_latent(samples, stream, mask):
    latent = {"samples": samples, STREAM_KEY: stream}
    if mask is not None:
        latent[MASK_KEY] = mask
    return latent


def trim_mask(mask, samples, dim, length):
    """The first `length` steps of a stream's mask along `dim`.

    Masks keep their own shape until sampling resizes them to the latent. One
    whose time axis is not the latent's is resized onto the latent grid first,
    as core does when it fits audio, so the cut lands on the same frames.
    """
    if mask is None:
        return None
    if mask.ndim != samples.ndim or mask.shape[dim] != samples.shape[dim]:
        mask = comfy.utils.reshape_mask(mask, samples.shape)
    return mask.narrow(dim, 0, length).contiguous()


class MiniMaxH3XSplitAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XSplitAV",
            display_name="MiniMax H3 Split AV Latent",
            category="model/latent/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "split",
                "av latent",
                "video latent",
                "audio latent",
                "nested tensor",
                "拆分",
                "潜空间",
                "音视频",
            ],
            description=(
                "Split an H3 AV latent into separate video and audio latents."
            ),
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.Latent.Output(display_name="video"),
                io.Latent.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        video_mask, audio_mask = unpack_masks(latent)
        return io.NodeOutput(
            stream_latent(video, "video", video_mask),
            stream_latent(audio, "audio", audio_mask),
        )


class MiniMaxH3XJoinAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XJoinAV",
            display_name="MiniMax H3 Join AV Latent",
            category="model/latent/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "join",
                "merge",
                "combine",
                "av latent",
                "nested tensor",
                "合并",
                "潜空间",
                "音视频",
            ],
            description=(
                "Recombine video and audio latents into an H3 AV latent. "
                "Validates that both streams describe the same frame count."
            ),
            inputs=[
                io.Latent.Input("video"),
                io.Latent.Input("audio"),
                io.Boolean.Input(
                    "strict",
                    default=True,
                    tooltip=(
                        "Reject streams whose latent_t and "
                        "audio_t disagree, or whose latent_t is "
                        "off the 17k+5 grid. Turn off only when "
                        "deliberately building an irregular "
                        "pair."
                    ),
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, video, audio, strict=True) -> io.NodeOutput:
        v = video["samples"]
        a = audio["samples"]

        # Catch a swapped connection before it becomes a confusing shape error.
        for name, latent, want in (
            ("video", video, "video"),
            ("audio", audio, "audio"),
        ):
            tag = latent.get(STREAM_KEY)
            if tag is not None and tag != want:
                raise ValueError(
                    "the '%s' input received a latent tagged as '%s' -- the two "
                    "inputs look swapped" % (name, tag)
                )

        if v.ndim != 5:
            raise ValueError(
                "video latent must be [B,24,T,H,W], got shape %s" % (tuple(v.shape),)
            )
        if a.ndim != 4:
            raise ValueError(
                "audio latent must be [B,32,2,T], got shape %s" % (tuple(a.shape),)
            )

        if strict:
            if v.shape[1] != VIDEO_CHANNELS:
                raise ValueError(
                    "video latent has %d channels, expected %d"
                    % (v.shape[1], VIDEO_CHANNELS)
                )
            if a.shape[1] != AUDIO_CHANNELS:
                raise ValueError(
                    "audio latent has %d channels, expected %d"
                    % (a.shape[1], AUDIO_CHANNELS)
                )
            if a.shape[2] != 2:
                raise ValueError(
                    "audio latent dim 2 must be 2 (L/R), got %d" % a.shape[2]
                )
            if v.shape[0] != a.shape[0]:
                raise ValueError(
                    "batch mismatch: video B=%d, audio B=%d" % (v.shape[0], a.shape[0])
                )

            latent_t, audio_t = v.shape[2], a.shape[3]
            frames = frames_from_latent_t(latent_t)
            if frames is None:
                raise ValueError(
                    "video latent_t=%d is off the model's grid (valid: 2, 7, 12, ... "
                    "i.e. (latent_t-2)%%5==0)" % latent_t
                )
            expected_audio_t = audio_t_for_frames(frames)
            if audio_t != expected_audio_t:
                raise ValueError(
                    "stream mismatch: video latent_t=%d implies %d frames and "
                    "audio_t=%d, but the audio latent has audio_t=%d"
                    % (latent_t, frames, expected_audio_t, audio_t)
                )

        return io.NodeOutput(pack_av(v, a, video.get(MASK_KEY), audio.get(MASK_KEY)))


class MiniMaxH3XLatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XLatentInfo",
            display_name="MiniMax H3 AV Latent Info",
            category="model/latent/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "latent info",
                "inspect latent",
                "latent shape",
                "frames",
                "tokens",
                "潜空间信息",
                "潜空间",
                "音视频",
            ],
            description=(
                "Report both stream shapes, implied duration and token count."
            ),
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.String.Output(display_name="report"),
                io.Int.Output(display_name="frames"),
                io.Int.Output(display_name="video_tokens"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        b, c, latent_t, lh, lw = video.shape
        frames = frames_from_latent_t(latent_t)
        width, height = lw * LATENT_SPATIAL_DIVISOR, lh * LATENT_SPATIAL_DIVISOR
        tokens = (
            latent_t
            * (height // TOKEN_SPATIAL_DIVISOR)
            * (width // TOKEN_SPATIAL_DIVISOR)
        )

        lines = [
            "video   %s  dtype=%s  device=%s"
            % (tuple(video.shape), video.dtype, video.device),
            "audio   %s  dtype=%s  device=%s"
            % (tuple(audio.shape), audio.dtype, audio.device),
            "pixels  %dx%d  (latent %dx%d)" % (width, height, lw, lh),
        ]
        if frames is None:
            lines.append(
                "frames  UNKNOWN -- latent_t=%d is off the 17k+5 grid" % latent_t
            )
        else:
            lines.append("frames  %d  (%.3f s at %d fps)" % (frames, frames / FPS, FPS))
            expected_audio_t = audio_t_for_frames(frames)
            if audio.shape[3] != expected_audio_t:
                lines.append(
                    "MISMATCH audio_t=%d but %d frames implies %d"
                    % (audio.shape[3], frames, expected_audio_t)
                )
        lines.append("tokens  %s video tokens" % format(tokens, ","))
        lines.append("batch   %d" % b)
        if c != VIDEO_CHANNELS:
            lines.append("WARNING video channels=%d, expected %d" % (c, VIDEO_CHANNELS))

        report = "\n".join(lines)
        return io.NodeOutput(
            report,
            frames if frames is not None else 0,
            tokens,
            ui=ui.PreviewText(report),
        )


class MiniMaxH3XUpscaleVideoLatent(io.ComfyNode):
    """Spatial resize of the video stream only. Audio passes through untouched.

    Audio length depends on duration, not resolution, so a resolution change
    must not touch it -- that is what makes two-stage refinement possible.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XUpscaleVideoLatent",
            display_name="MiniMax H3 Upscale Video Latent",
            category="model/latent/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "upscale",
                "resize",
                "video latent",
                "two stage",
                "放大",
                "潜空间",
                "视频",
            ],
            description=(
                "Resize the video stream of an H3 AV latent, leaving audio "
                "alone. Target pixel dimensions snap to multiples of 32."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input(
                    "width",
                    default=1216,
                    min=32,
                    max=16384,
                    step=32,
                    tooltip=("Target pixel width; divided by 16 for the latent grid."),
                ),
                io.Int.Input(
                    "height",
                    default=672,
                    min=32,
                    max=16384,
                    step=32,
                    tooltip=("Target pixel height; divided by 16 for the latent grid."),
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["bilinear", "bicubic", "nearest-exact", "area"],
                    default="bicubic",
                    tooltip="Interpolation used to resize every video frame.",
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, width, height, upscale_method) -> io.NodeOutput:
        video, audio = unpack_av(latent)

        # Snap to 32 px so the DiT's 2x2 patchify divides evenly.
        width = max(32, round(width / 32) * 32)
        height = max(32, round(height / 32) * 32)
        target_w = width // LATENT_SPATIAL_DIVISOR
        target_h = height // LATENT_SPATIAL_DIVISOR

        b, c, t, lh, lw = video.shape
        if (lh, lw) == (target_h, target_w):
            return io.NodeOutput(keep_mask(pack_av(video, audio), latent))

        # common_upscale wants [B,C,H,W]; fold time into batch so every frame is
        # resized identically and no temporal blending occurs.
        flat = video.movedim(2, 1).reshape(b * t, c, lh, lw)
        flat = comfy.utils.common_upscale(
            flat, target_w, target_h, upscale_method, "disabled"
        )
        resized = flat.reshape(b, t, c, target_h, target_w).movedim(1, 2)

        # The mask keeps its own shape; sampling resizes it to the new latent.
        return io.NodeOutput(keep_mask(pack_av(resized.contiguous(), audio), latent))


class MiniMaxH3XTemporalTrim(io.ComfyNode):
    """Trim both streams together to a shorter, still-legal duration."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XTemporalTrim",
            display_name="MiniMax H3 Temporal Trim",
            category="model/latent/minimax",
            search_aliases=[
                "h3",
                "minimax",
                "trim",
                "shorten",
                "cut",
                "duration",
                "av latent",
                "裁剪",
                "时长",
                "缩短",
                "潜空间",
            ],
            description=(
                "Shorten an H3 AV latent, keeping video and audio coherent. "
                "The result is snapped to the model's 17k+5 frame grid."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Float.Input(
                    "keep_seconds",
                    default=5.0,
                    min=0.2,
                    max=20.0,
                    step=0.1,
                    tooltip=(
                        "Duration to keep from the start, aligned to the frame grid."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(), io.Int.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, latent, keep_seconds) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        src_latent_t = video.shape[2]
        src_frames = frames_from_latent_t(src_latent_t)
        if src_frames is None:
            raise ValueError(
                "source latent_t=%d is off the 17k+5 grid; cannot trim safely"
                % src_latent_t
            )

        # Snap UP first, matching what the generation nodes do with a duration --
        # asking for "5 s" should mean the same 124 frames everywhere. Then clamp
        # to what the source actually has, so we never index past the end.
        want = max(FRAME_GRID_REM, round(keep_seconds * FPS))
        while want % FRAME_GRID_MOD != FRAME_GRID_REM:
            want += 1

        if want > src_frames:
            raise ValueError(
                "cannot extend: source holds %d frames (%.3f s) but keep_seconds=%.2f "
                "needs %d frames. Trim only shortens."
                % (src_frames, src_frames / FPS, keep_seconds, want)
            )

        new_latent_t = video_latent_t(want)
        new_audio_t = audio_t_for_frames(want)
        if new_audio_t > audio.shape[3]:
            raise ValueError(
                "audio stream is shorter than the request: has audio_t=%d, needs %d"
                % (audio.shape[3], new_audio_t)
            )

        video_mask, audio_mask = unpack_masks(latent)
        return io.NodeOutput(
            pack_av(
                video[:, :, :new_latent_t].contiguous(),
                audio[:, :, :, :new_audio_t].contiguous(),
                trim_mask(video_mask, video, 2, new_latent_t),
                trim_mask(audio_mask, audio, 3, new_audio_t),
            ),
            want,
        )


class AVLatentExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3XSplitAV,
            MiniMaxH3XJoinAV,
            MiniMaxH3XLatentInfo,
            MiniMaxH3XUpscaleVideoLatent,
            MiniMaxH3XTemporalTrim,
        ]


async def comfy_entrypoint() -> AVLatentExtension:
    return AVLatentExtension()
