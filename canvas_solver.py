"""Geometry and optional runtime planning for MiniMax-H3."""

from __future__ import annotations

import math

from comfy_api.latest import ComfyExtension, io

if __package__:
    from .calibration import (
        NO_PROFILE,
        PROFILES_PATH,
        CalibrationProfile,
        estimate_runtime,
        load_profiles,
        memory_notes,
        profile_options,
        resolve_profile,
    )
else:
    from calibration import (
        NO_PROFILE,
        PROFILES_PATH,
        CalibrationProfile,
        estimate_runtime,
        load_profiles,
        memory_notes,
        profile_options,
        resolve_profile,
    )

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID_MOD = 17
FRAME_GRID_REM = 5
TRAINED_FRAME_MIN = 124
TRAINED_FRAME_MAX = 362
MP_UNIT = 1024 * 1024
UNAVAILABLE_ESTIMATE = -1.0

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "21:9 (Ultrawide)": (21, 9),
    "16:9 (Widescreen)": (16, 9),
    "3:2 (Photo)": (3, 2),
    "4:3 (Standard)": (4, 3),
    "1:1 (Square)": (1, 1),
    "3:4 (Portrait Standard)": (3, 4),
    "2:3 (Portrait Photo)": (2, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
}

CALIBRATION_PROFILES = load_profiles(PROFILES_PATH)


def align_frame_count(frame_count: int) -> int:
    """Snap up to the model's 17k+5 frame grid."""
    aligned = max(FRAME_GRID_REM, int(frame_count))
    while aligned % FRAME_GRID_MOD != FRAME_GRID_REM:
        aligned += 1
    return aligned


def video_latent_t(frame_count: int) -> int:
    if frame_count <= FRAME_GRID_REM:
        return 2
    return ((frame_count - FRAME_GRID_REM) // FRAME_GRID_MOD) * 5 + 2


def _canvas_axes(
        width_ratio: int, height_ratio: int, megapixels: float) -> tuple[int, int]:
    scale = math.sqrt(megapixels * MP_UNIT / (width_ratio * height_ratio))
    width = max(
        CANVAS_MULTIPLE,
        round(width_ratio * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    height = max(
        CANVAS_MULTIPLE,
        round(height_ratio * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return width, height


def solve_canvas(
        aspect_ratio: str, megapixels: float, clamp_to_cap: bool) -> tuple[int, int, bool]:
    """Return width, height and whether the original request exceeded the cap."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            "unknown MiniMax-H3 aspect ratio '%s'; expected one of: %s"
            % (aspect_ratio, ", ".join(ASPECT_RATIOS)))
    if not math.isfinite(megapixels) or megapixels <= 0.0:
        raise ValueError("megapixels must be finite and > 0, got %r" % megapixels)

    width_ratio, height_ratio = ASPECT_RATIOS[aspect_ratio]
    width, height = _canvas_axes(width_ratio, height_ratio, megapixels)
    over_cap = width * height > MAX_PIXELS
    if not over_cap or not clamp_to_cap:
        return width, height, over_cap

    capped_megapixels = megapixels
    for _ in range(40):
        capped_megapixels *= 0.98
        width, height = _canvas_axes(width_ratio, height_ratio, capped_megapixels)
        if width * height <= MAX_PIXELS:
            return width, height, True
    raise RuntimeError(
        "failed to fit %.3f MP at aspect ratio %s inside the %d pixel trained cap"
        % (megapixels, aspect_ratio, MAX_PIXELS))


def _cost_report(
        profile_key: str,
        profile: CalibrationProfile | None,
        video_tokens: int,
        steps: int,
        vram_budget_gib: float,
) -> tuple[float, float, str, list[str]]:
    if profile is None:
        return (
            UNAVAILABLE_ESTIMATE,
            UNAVAILABLE_ESTIMATE,
            "cost        not estimated; exact token count only (select an explicit hardware profile to opt in)",
            [],
        )

    seconds_per_iteration, total_minutes, fit = estimate_runtime(
        profile, video_tokens, steps)
    report = (
        "cost        %.1f s/it -> %.1f min at %d steps\n"
        "profile     %s\n"
        "fit         s/it = %.4g * tokens^%.4f (R2=%.4f, %d measurements)"
        % (
            seconds_per_iteration,
            total_minutes,
            steps,
            profile["label"],
            fit["coefficient"],
            fit["exponent"],
            fit["r_squared"],
            fit["sample_count"],
        )
    )
    warnings = memory_notes(profile, video_tokens, vram_budget_gib)
    if video_tokens > fit["max_tokens"]:
        warnings.append(
            "%s tokens exceeds profile '%s' maximum of %s; the runtime is extrapolated"
            % (format(video_tokens, ","), profile_key, format(fit["max_tokens"], ",")))
    return seconds_per_iteration, total_minutes, report, warnings


class MiniMaxH3XCanvasSolver(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XCanvasSolver",
            display_name="MiniMax H3 Canvas Planner / 画布规划",
            category="model/latent/minimax",
            search_aliases=[
                "h3", "minimax", "canvas", "canvas planner", "resolution",
                "17k+5", "frames", "frame count", "tokens", "画布", "分辨率", "帧数",
                "规划",
            ],
            description=(
                "Resolve the exact canvas, frame grid, latent sizes and token count "
                "before queueing. Runtime estimates are disabled unless an explicit "
                "hardware profile is selected. / 在加入队列前计算精确画布、帧网格、"
                "潜空间尺寸和令牌数；仅在明确选择硬件配置后估算运行时间。"),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=list(ASPECT_RATIOS),
                    default="16:9 (Widescreen)",
                ),
                io.Float.Input(
                    "megapixels",
                    default=0.4,
                    min=0.1,
                    max=1.2,
                    step=0.05,
                    tooltip=(
                        "Target area using ComfyUI's 1024*1024 megapixel unit. / "
                        "使用 ComfyUI 的 1024*1024 百万像素单位设置目标面积。"),
                ),
                io.Float.Input(
                    "seconds",
                    default=5.0,
                    min=0.2,
                    max=20.0,
                    step=0.1,
                    tooltip=(
                        "Duration at 24 fps, snapped up to H3's 17k+5 frame grid. / "
                        "按 24 fps 设置时长，并向上对齐到 H3 的 17k+5 帧网格。"),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=200,
                    tooltip=(
                        "Used only when a hardware cost profile is selected. / "
                        "仅在选择硬件成本配置时使用。"),
                ),
                io.Combo.Input(
                    "cost_profile",
                    options=profile_options(CALIBRATION_PROFILES),
                    default=NO_PROFILE,
                    tooltip=(
                        "No runtime estimate is made by default. Profiles are "
                        "machine-specific and must be selected explicitly. / "
                        "默认不估算运行时间；配置仅适用于特定机器，必须明确选择。"),
                ),
                io.Float.Input(
                    "vram_budget_gib",
                    default=24.0,
                    min=4.0,
                    max=192.0,
                    step=0.5,
                    tooltip=(
                        "Compared only with thresholds from the selected profile. / "
                        "仅与所选配置中的阈值比较。"),
                ),
                io.Boolean.Input(
                    "clamp_to_trained_cap",
                    default=False,
                    tooltip=(
                        "Off reports the canvas ComfyUI will actually create. On "
                        "shrinks over-cap requests to H3's documented trained area. / "
                        "关闭时报告 ComfyUI 实际创建的画布；开启时将超限请求缩小到 "
                        "H3 文档中的训练面积。"),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="frames"),
                io.Int.Output(display_name="latent_t"),
                io.Int.Output(display_name="audio_t"),
                io.Int.Output(display_name="video_tokens"),
                io.Float.Output(display_name="est_sit_seconds"),
                io.Float.Output(display_name="est_total_minutes"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
            cls,
            aspect_ratio,
            megapixels,
            seconds,
            steps,
            cost_profile,
            vram_budget_gib,
            clamp_to_trained_cap,
    ) -> io.NodeOutput:
        width, height, over_cap = solve_canvas(
            aspect_ratio, megapixels, clamp_to_trained_cap)
        requested_frames = max(FRAME_GRID_REM, round(seconds * FPS))
        frames = align_frame_count(requested_frames)
        latent_t = video_latent_t(frames)
        audio_t = round(frames / FPS * AUDIO_LATENT_FPS)
        video_tokens = latent_t * (height // 32) * (width // 32)

        profile = resolve_profile(CALIBRATION_PROFILES, cost_profile)
        sit, total_minutes, cost_line, cost_warnings = _cost_report(
            cost_profile, profile, video_tokens, steps, vram_budget_gib)

        actual_megapixels = width * height / MP_UNIT
        lines = [
            "canvas      %dx%d (%.2f MP, %s)" % (
                width, height, actual_megapixels, aspect_ratio),
            "duration    %d frames @ %d fps = %.3f s" % (
                frames, FPS, frames / FPS),
            "latents     video_t=%d audio_t=%d" % (latent_t, audio_t),
            "tokens      %s video tokens" % format(video_tokens, ","),
            cost_line,
        ]

        warnings: list[str] = []
        if over_cap and clamp_to_trained_cap:
            warnings.append(
                "%.2f MP exceeded H3's %.2f MP trained cap; canvas was reduced to %.2f MP"
                % (megapixels, MAX_PIXELS / MP_UNIT, actual_megapixels))
        elif over_cap:
            warnings.append(
                "canvas is %.2f MP, above H3's %.2f MP trained cap. ComfyUI will render it, but the geometry is outside the documented training area"
                % (actual_megapixels, MAX_PIXELS / MP_UNIT))
        if frames != requested_frames:
            warnings.append(
                "%.1f seconds maps to %d frames and was snapped up to %d frames (%.3f seconds)"
                % (seconds, requested_frames, frames, frames / FPS))
        if frames < TRAINED_FRAME_MIN:
            warnings.append(
                "%d frames is below the documented trained range of %d-%d"
                % (frames, TRAINED_FRAME_MIN, TRAINED_FRAME_MAX))
        elif frames > TRAINED_FRAME_MAX:
            warnings.append(
                "%d frames exceeds the documented trained range of %d-%d"
                % (frames, TRAINED_FRAME_MIN, TRAINED_FRAME_MAX))
        if min(width, height) > BASE_SHORT_EDGE:
            warnings.append(
                "short edge %d px exceeds H3's documented %d px native short edge"
                % (min(width, height), BASE_SHORT_EDGE))
        warnings.extend(cost_warnings)

        report = "\n".join(lines)
        if warnings:
            report += "\n\n" + "\n".join("WARNING  " + warning for warning in warnings)
        return io.NodeOutput(
            width,
            height,
            frames,
            latent_t,
            audio_t,
            video_tokens,
            sit,
            total_minutes,
            report,
        )


class CanvasSolverExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3XCanvasSolver]


async def comfy_entrypoint() -> CanvasSolverExtension:
    return CanvasSolverExtension()
