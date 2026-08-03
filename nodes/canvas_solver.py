"""Canvas / duration / cost solver for MiniMax-H3.

The core generation nodes accept any 32-multiple width and height. Nothing
validates against the model's trained canvas (768 short edge, 768*1344 area
cap) and frame counts snap silently onto the 17k+5 grid, so a workflow can
queue an untrained geometry and only find out 30 minutes later.

This node answers "what will this job actually be, and what will it cost"
before anything is queued.

The cost model is a power-law fit over measurements from one RTX 3090; see
CALIBRATION below for how to replace it with your own card's data.
"""

import math

from comfy_api.latest import ComfyExtension, io

# Model constants -- mirrored from comfy_extras/nodes_minimax_h3.py L25-30.
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344          # 1.032 MP, the trained area cap
FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID_MOD = 17
FRAME_GRID_REM = 5
TRAINED_FRAME_MIN = 124          # ~5 s
TRAINED_FRAME_MAX = 362          # ~15 s

# ResolutionSelector (comfy_extras/nodes_resolution.py) defines a megapixel as
# 1024*1024 and rounds each axis to the nearest multiple. Reproduced exactly so
# this solver predicts the geometry a workflow using that node will really get.
MP_UNIT = 1024 * 1024

ASPECT_RATIOS = {
    "21:9 (Ultrawide)": (21, 9),
    "16:9 (Widescreen)": (16, 9),
    "3:2 (Photo)": (3, 2),
    "4:3 (Standard)": (4, 3),
    "1:1 (Square)": (1, 1),
    "3:4 (Portrait Standard)": (3, 4),
    "2:3 (Portrait Photo)": (2, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
}

# --- CALIBRATION -------------------------------------------------------------
# Measured on RTX 3090 24 GB / 32 GB RAM / torch 2.8.0+cu128 / xformers,
# fl2va_pruned_int8_convrot + qwen3vl_32b int8_convrot, res_multistep+simple.
# (video_tokens, seconds_per_iteration)
CALIBRATION = [
    (14985, 25.02),
    (23199, 41.48),
    (29526, 58.67),
    (38184, 90.16),
    (29160, 64.13),
    (45144, 116.50),
    (57456, 167.24),
]
# Fixed overhead outside the sampling loop (weight load, text encode, VAE
# decode). Measured 90-180 s depending on resolution; 130 s is a mid estimate.
OVERHEAD_SECONDS = 130.0

# Peak VRAM plateaus once dynamic offload engages -- it does not keep climbing
# with resolution. Above this token count the cost is time, not memory.
VRAM_PLATEAU_MIB = 23400
VRAM_PLATEAU_TOKENS = 23000
# Past roughly this many tokens on a 24 GB card, peak VRAM reaches the card
# limit and s/it departs the fitted curve upward (offload thrashing). Measured:
# 74304 tokens ran at ~321 s/it where the fit predicts ~225.
THRASH_TOKENS_24GIB = 70000


def _fit_power_law(samples):
    """Least-squares fit of s/it = a * tokens^b in log space.

    A power law rather than a line because this H3 release is full-attention
    only: the attention term grows quadratically in sequence length while the
    FFN term grows linearly, so the exponent lands between 1 and 2.
    """
    n = len(samples)
    sx = sum(math.log(t) for t, _ in samples)
    sy = sum(math.log(s) for _, s in samples)
    sxx = sum(math.log(t) ** 2 for t, _ in samples)
    sxy = sum(math.log(t) * math.log(s) for t, s in samples)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = math.exp((sy - b * sx) / n)
    return a, b


def _r_squared(samples, a, b):
    mean = sum(math.log(s) for _, s in samples) / len(samples)
    ss_res = sum((math.log(s) - math.log(a * t**b)) ** 2 for t, s in samples)
    ss_tot = sum((math.log(s) - mean) ** 2 for _, s in samples)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


FIT_A, FIT_B = _fit_power_law(CALIBRATION)
FIT_R2 = _r_squared(CALIBRATION, FIT_A, FIT_B)
CALIBRATION_MAX_TOKENS = max(t for t, _ in CALIBRATION)


def align_frame_count(n):
    """Snap up to the model's 17k+5 grid. Mirrors the core node's helper."""
    n = max(FRAME_GRID_REM, int(n))
    while n % FRAME_GRID_MOD != FRAME_GRID_REM:
        n += 1
    return n


def video_latent_t(frame_count):
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // FRAME_GRID_MOD) * 5 + 2


def solve_canvas(aspect_ratio, megapixels, clamp_to_cap=False):
    """(width, height, over_cap) for the given ratio and megapixel target.

    Follows ResolutionSelector's algorithm exactly, because that is the node
    workflows actually use and this solver's value depends on predicting real
    behaviour rather than ideal behaviour.

    By default the result is NOT clamped to H3's area cap, even when it exceeds
    it. That is deliberate: `adapt_canvas()` in the core node is only applied to
    reference videos, so the generation path really does accept an over-cap
    canvas. Measured example: 1.0 MP at 16:9 yields 1376x768 = 1,056,768 px,
    over the 1,032,192 cap, and H3 renders it anyway. Reporting the honest
    geometry plus a warning beats silently substituting a different one.

    Pass clamp_to_cap=True to get the largest on-grid canvas within the cap.
    """
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]

    def axes(mp):
        scale = math.sqrt(mp * MP_UNIT / (w_ratio * h_ratio))
        w = max(CANVAS_MULTIPLE,
                round(w_ratio * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        h = max(CANVAS_MULTIPLE,
                round(h_ratio * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        return w, h

    width, height = axes(megapixels)
    over_cap = width * height > MAX_PIXELS
    if not over_cap or not clamp_to_cap:
        return width, height, over_cap

    capped = megapixels
    for _ in range(40):                     # converges in a few passes
        capped *= 0.98
        width, height = axes(capped)
        if width * height <= MAX_PIXELS:
            break
    return width, height, True


def estimate_sit(video_tokens):
    return FIT_A * video_tokens**FIT_B


class MiniMaxH3XCanvasSolver(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XCanvasSolver",
            display_name="MiniMax H3 Canvas & Cost Solver",
            category="model/latent/minimax",
            description=(
                "Resolve a legal H3 canvas and frame count, then estimate token "
                "count, s/it and total runtime before queueing. Cost model is a "
                "power-law fit over RTX 3090 measurements."),
            inputs=[
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS),
                               default="16:9 (Widescreen)"),
                io.Float.Input("megapixels", default=0.4, min=0.1, max=1.2, step=0.05,
                               tooltip="Target area. H3's trained cap is 768*1344 = 1.03 MP; above that the canvas is clamped."),
                io.Float.Input("seconds", default=5.0, min=0.2, max=20.0, step=0.1,
                               tooltip="Duration at 24 fps. Snapped up to the model's 17k+5 frame grid."),
                io.Int.Input("steps", default=20, min=1, max=200,
                             tooltip="Sampling steps. s/it is step-count independent, so runtime scales linearly."),
                io.Float.Input("vram_budget_gib", default=24.0, min=4.0, max=192.0, step=0.5,
                               tooltip="Used only to warn when the job is near this card's offload-thrashing region."),
                io.Boolean.Input("clamp_to_trained_cap", default=False,
                                 tooltip="H3's generation path does NOT enforce its own 768*1344 area cap, so ResolutionSelector-style geometry can exceed it (1.0 MP at 16:9 gives 1376x768, over the cap, and renders anyway). Off = report the real geometry and warn. On = shrink to the largest canvas within the cap."),
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
    def execute(cls, aspect_ratio, megapixels, seconds, steps, vram_budget_gib,
                clamp_to_trained_cap=False) -> io.NodeOutput:
        width, height, over_cap = solve_canvas(aspect_ratio, megapixels,
                                               clamp_to_cap=clamp_to_trained_cap)
        frames = align_frame_count(round(seconds * FPS))
        latent_t = video_latent_t(frames)
        audio_t = round(frames / FPS * AUDIO_LATENT_FPS)

        # //32 not //16: the VAE compresses 16x spatially, then a 1x2x2 patchify
        # halves each spatial axis again before tokens reach the transformer.
        video_tokens = latent_t * (height // 32) * (width // 32)

        sit = estimate_sit(video_tokens)
        total_min = (OVERHEAD_SECONDS + steps * sit) / 60.0

        actual_mp = width * height / MP_UNIT
        lines = [
            "canvas      %dx%d  (%.2f MP, %s)" % (
                width, height, actual_mp, aspect_ratio),
            "duration    %d frames @ %d fps = %.3f s" % (frames, FPS, frames / FPS),
            "latents     video_t=%d  audio_t=%d" % (latent_t, audio_t),
            "tokens      %s video tokens" % format(video_tokens, ","),
            "estimate    %.1f s/it  ->  %.1f min at %d steps" % (sit, total_min, steps),
        ]

        warnings = []
        if over_cap and clamp_to_trained_cap:
            warnings.append(
                "megapixels %.2f exceeded H3's %.2f MP trained cap; canvas shrunk to %.2f MP"
                % (megapixels, MAX_PIXELS / MP_UNIT, actual_mp))
        elif over_cap:
            warnings.append(
                "canvas is %.2f MP, over H3's %.2f MP trained cap. The generation "
                "path does not enforce the cap so this WILL render, but the geometry "
                "is outside training. Enable clamp_to_trained_cap to stay inside."
                % (actual_mp, MAX_PIXELS / MP_UNIT))
        requested_frames = max(FRAME_GRID_REM, round(seconds * FPS))
        if frames != requested_frames:
            warnings.append(
                "%.1f s = %d frames is off-grid; snapped up to %d frames (%.3f s)"
                % (seconds, requested_frames, frames, frames / FPS))
        if frames < TRAINED_FRAME_MIN:
            warnings.append(
                "%d frames is below the trained range (%d-%d); quality is unvalidated"
                % (frames, TRAINED_FRAME_MIN, TRAINED_FRAME_MAX))
        elif frames > TRAINED_FRAME_MAX:
            warnings.append(
                "%d frames exceeds the trained range (%d-%d); MiniMax has not validated this"
                % (frames, TRAINED_FRAME_MIN, TRAINED_FRAME_MAX))
        if min(width, height) > BASE_SHORT_EDGE:
            warnings.append(
                "short edge %d px is above the native %d; H3 was trained at %d"
                % (min(width, height), BASE_SHORT_EDGE, BASE_SHORT_EDGE))

        if video_tokens > CALIBRATION_MAX_TOKENS:
            warnings.append(
                "%s tokens is beyond the calibrated range (max %s measured); the "
                "s/it figure is EXTRAPOLATED and will read low"
                % (format(video_tokens, ","), format(CALIBRATION_MAX_TOKENS, ",")))
        if vram_budget_gib <= 24.5 and video_tokens >= THRASH_TOKENS_24GIB:
            warnings.append(
                "at ~%s tokens a 24 GB card reaches its VRAM limit and s/it departs "
                "the curve upward (measured ~321 s/it at 74,304 where the fit says "
                "~%.0f). Expect far worse than the estimate above."
                % (format(video_tokens, ","), estimate_sit(74304)))
        elif video_tokens >= VRAM_PLATEAU_TOKENS:
            lines.append(
                "vram        expect ~%s MiB peak (plateau -- offload absorbs growth "
                "past this point, so resolution costs time, not memory)"
                % format(VRAM_PLATEAU_MIB, ","))

        report = "\n".join(lines)
        if warnings:
            report += "\n\n" + "\n".join("WARNING  " + w for w in warnings)
        report += ("\n\ncost model   s/it = %.4g * tokens^%.4f  (R2=%.4f over %d "
                   "RTX 3090 points)" % (FIT_A, FIT_B, FIT_R2, len(CALIBRATION)))

        return io.NodeOutput(width, height, frames, latent_t, audio_t,
                             video_tokens, sit, total_min, report)


class CanvasSolverExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3XCanvasSolver]


async def comfy_entrypoint() -> CanvasSolverExtension:
    return CanvasSolverExtension()
