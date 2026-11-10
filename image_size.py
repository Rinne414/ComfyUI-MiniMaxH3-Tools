"""MiniMax-H3-Image canvas presets: aspect ratio x megapixels -> width, height and the empty latent.

Kept apart from the encoder so changing the size never re-runs the 27 GB text encoder.
"""

import torch

import comfy.model_management
from comfy_api.latest import io

if __package__:
    from .image_sizes import (
        ASPECT_RATIOS,
        MEGAPIXELS,
        canvas_size,
        latent_shape,
        resolve_ratio,
    )
else:
    from image_sizes import (
        ASPECT_RATIOS,
        MEGAPIXELS,
        canvas_size,
        latent_shape,
        resolve_ratio,
    )


class MiniMaxH3XImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XImageSize",
            display_name="MiniMax H3 Image Size",
            category="model/latent/minimax",
            search_aliases=[
                "h3", "minimax", "h3 image", "image size", "canvas",
                "aspect ratio", "megapixels", "empty latent", "图像尺寸", "画布",
                "长宽比",
            ],
            description=(
                "Canvas for MiniMax-H3-Image: 16 MP uses the model card's "
                "table, other areas the same formula; sides are multiples of "
                "32."
            ),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio", options=list(ASPECT_RATIOS), default="1:1"
                ),
                io.Combo.Input(
                    "megapixels",
                    options=list(MEGAPIXELS),
                    default="16 MP (best, model card)",
                    tooltip=(
                        "About 4-20 MP works; 16 MP looks best. Larger "
                        "canvases take longer."
                    ),
                ),
                io.String.Input(
                    "suggested_ratio",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "The prompt expander's aspect ratio; it replaces "
                        "the menu. A connected picture wins over it."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "For edits: connect the picture and the closest "
                        "aspect ratio is used instead of the menu."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="aspect_ratio"),
            ],
        )

    @classmethod
    def execute(
        cls, aspect_ratio, megapixels, suggested_ratio="", image=None
    ) -> io.NodeOutput:
        # IMAGE is [batch, height, width, channels]
        picture = (image.shape[2], image.shape[1]) if image is not None else None
        aspect_ratio, _source = resolve_ratio(aspect_ratio, suggested_ratio, picture)
        width, height = canvas_size(aspect_ratio, MEGAPIXELS[megapixels])
        latent = torch.zeros(
            latent_shape(width, height),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent}, width, height, aspect_ratio)
