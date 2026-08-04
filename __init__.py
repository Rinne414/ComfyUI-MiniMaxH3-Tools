"""ComfyUI-MiniMaxH3-Tools extension entrypoint."""

from comfy_api.latest import ComfyExtension

if __package__:
    from .audio_reroll import MiniMaxH3XAudioReroll
    from .av_latent import (
        MiniMaxH3XJoinAV,
        MiniMaxH3XLatentInfo,
        MiniMaxH3XSplitAV,
        MiniMaxH3XTemporalTrim,
        MiniMaxH3XUpscaleVideoLatent,
    )
    from .canvas_solver import MiniMaxH3XCanvasSolver
    from .nf4_loader import get_nf4_nodes
    from .prompt_lint import MiniMaxH3XPromptLint
else:
    from audio_reroll import MiniMaxH3XAudioReroll
    from av_latent import (
        MiniMaxH3XJoinAV,
        MiniMaxH3XLatentInfo,
        MiniMaxH3XSplitAV,
        MiniMaxH3XTemporalTrim,
        MiniMaxH3XUpscaleVideoLatent,
    )
    from canvas_solver import MiniMaxH3XCanvasSolver
    from nf4_loader import get_nf4_nodes
    from prompt_lint import MiniMaxH3XPromptLint

NODES = (
    MiniMaxH3XCanvasSolver,
    MiniMaxH3XSplitAV,
    MiniMaxH3XJoinAV,
    MiniMaxH3XLatentInfo,
    MiniMaxH3XUpscaleVideoLatent,
    MiniMaxH3XTemporalTrim,
    MiniMaxH3XPromptLint,
    MiniMaxH3XAudioReroll,
    *get_nf4_nodes(),
)


class MiniMaxH3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return list(NODES)


async def comfy_entrypoint() -> MiniMaxH3ToolsExtension:
    return MiniMaxH3ToolsExtension()
