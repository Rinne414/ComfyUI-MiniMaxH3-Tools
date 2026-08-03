"""ComfyUI-MiniMax-H3-Toolkit

Nodes that fill gaps in ComfyUI's core MiniMax-H3 support:

  canvas_solver  -- resolve a legal canvas/duration and estimate cost first
  av_latent      -- reach one stream of an AV NestedTensor pair
  prompt_lint    -- deterministic checks against MiniMax's format rules
  ref_cache      -- stop re-encoding ref2va references every run
  audio_reroll   -- new soundtrack over an existing picture
"""

from comfy_api.latest import ComfyExtension

# Relative when ComfyUI imports the folder as a package; absolute when the
# tests put this directory on sys.path and import the modules directly.
try:
    from .audio_reroll import MiniMaxH3XAudioReroll
    from .av_latent import (
        MiniMaxH3XJoinAV, MiniMaxH3XLatentInfo, MiniMaxH3XSplitAV,
        MiniMaxH3XTemporalTrim, MiniMaxH3XUpscaleVideoLatent,
    )
    from .canvas_solver import MiniMaxH3XCanvasSolver
    from .prompt_lint import MiniMaxH3XPromptLint
    from .ref_cache import MiniMaxH3XRefEncodeCache
except ImportError:  # pragma: no cover - direct-import fallback
    from audio_reroll import MiniMaxH3XAudioReroll
    from av_latent import (
        MiniMaxH3XJoinAV, MiniMaxH3XLatentInfo, MiniMaxH3XSplitAV,
        MiniMaxH3XTemporalTrim, MiniMaxH3XUpscaleVideoLatent,
    )
    from canvas_solver import MiniMaxH3XCanvasSolver
    from prompt_lint import MiniMaxH3XPromptLint
    from ref_cache import MiniMaxH3XRefEncodeCache

NODES = [
    MiniMaxH3XCanvasSolver,
    MiniMaxH3XSplitAV,
    MiniMaxH3XJoinAV,
    MiniMaxH3XLatentInfo,
    MiniMaxH3XUpscaleVideoLatent,
    MiniMaxH3XTemporalTrim,
    MiniMaxH3XPromptLint,
    MiniMaxH3XRefEncodeCache,
    MiniMaxH3XAudioReroll,
]


class MiniMaxH3ToolkitExtension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint() -> MiniMaxH3ToolkitExtension:
    return MiniMaxH3ToolkitExtension()
