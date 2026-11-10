"""The MiniMax-H3-Image nodes, loaded only when this ComfyUI can run them.

They need ComfyUI 0.45.0 or later (native MiniMax-H3-Image support, PR #17655). On an older ComfyUI the
imports below fail; the video and latent nodes of this pack must keep loading, so that is logged and
no image node is registered.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

PREFIX = "MiniMax-H3 Tools"


def get_image_nodes() -> list:
    try:
        if __package__:
            from .image_conditioning import (
                MiniMaxH3XLoadConditioning,
                MiniMaxH3XSaveConditioning,
            )
            from .image_encode import MiniMaxH3XImageEncodeLowRAM
            from .image_expand import MiniMaxH3XImagePromptExpand
            from .image_size import MiniMaxH3XImageSize
        else:
            from image_conditioning import (
                MiniMaxH3XLoadConditioning,
                MiniMaxH3XSaveConditioning,
            )
            from image_encode import MiniMaxH3XImageEncodeLowRAM
            from image_expand import MiniMaxH3XImagePromptExpand
            from image_size import MiniMaxH3XImageSize
    except ImportError as error:
        LOGGER.warning(
            "%s: H3-Image nodes unavailable (this ComfyUI lacks native MiniMax-H3-Image "
            "support; update to 0.45.0 or later): %s",
            PREFIX,
            error,
        )
        return []
    return [
        MiniMaxH3XImagePromptExpand,
        MiniMaxH3XImageEncodeLowRAM,
        MiniMaxH3XImageSize,
        MiniMaxH3XSaveConditioning,
        MiniMaxH3XLoadConditioning,
    ]
