"""MiniMax-H3-Image canvas sizes (pure math, no ComfyUI imports).

The model card's 16 MP table is the official recipe from examples/context_ir.py:
side = round(sqrt(area * ratio) / 32) * 32 with area = 4096 * 4096. The same formula gives the
other areas. Sides are multiples of 32 because the VAE downsamples 16x and the DiT patchifies 2x2.
"""

import math

MULTIPLE = 32
MP_UNIT = 1024 * 1024
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "5:4": (5, 4),
    "4:5": (4, 5),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "9:21": (9, 21),
}
# label -> megapixels in 1024*1024 units; the model card's working range is about 4-20 MP, best near 16
MEGAPIXELS = {
    "4 MP (fast)": 4,
    "8 MP": 8,
    "16 MP (best, model card)": 16,
    "20 MP (upper limit)": 20,
}
OFFICIAL_16MP = {
    "1:1": (4096, 4096),
    "4:3": (4736, 3552),
    "3:4": (3552, 4736),
    "3:2": (5024, 3360),
    "2:3": (3360, 5024),
    "16:9": (5472, 3072),
    "9:16": (3072, 5472),
    "21:9": (6272, 2688),
}


def canvas_size(aspect_ratio: str, megapixels: float) -> tuple[int, int]:
    """(width, height) near megapixels at the ratio, both sides multiples of 32."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            "unknown aspect ratio '%s'; expected one of: %s"
            % (aspect_ratio, ", ".join(ASPECT_RATIOS))
        )
    if megapixels <= 0:
        raise ValueError("megapixels must be positive, got %s" % megapixels)
    x, y = ASPECT_RATIOS[aspect_ratio]
    area = megapixels * MP_UNIT
    width = max(MULTIPLE, round(math.sqrt(area * x / y) / MULTIPLE) * MULTIPLE)
    height = max(MULTIPLE, round(math.sqrt(area * y / x) / MULTIPLE) * MULTIPLE)
    return width, height


def closest_ratio(width: int, height: int) -> str:
    """The preset aspect ratio closest to a picture's (compared on a log scale, so 2:1 and 1:2 are symmetric)."""
    if width <= 0 or height <= 0:
        raise ValueError("picture size must be positive, got %dx%d" % (width, height))
    target = math.log(width / height)
    return min(
        ASPECT_RATIOS,
        key=lambda r: abs(math.log(ASPECT_RATIOS[r][0] / ASPECT_RATIOS[r][1]) - target),
    )


def resolve_ratio(
    menu: str, suggested: str = "", picture: tuple[int, int] | None = None
) -> tuple[str, str]:
    """Which aspect ratio to use and why. Precedence: a connected picture (edits keep its shape), then a
    suggested 'W:H' (from the prompt expander, snapped to the closest preset), then the menu."""
    if picture is not None:
        return closest_ratio(*picture), "picture"
    parts = (suggested or "").split(":")
    if len(parts) == 2 and all(p.strip().isdigit() and int(p) > 0 for p in parts):
        return closest_ratio(int(parts[0]), int(parts[1])), "suggested"
    return menu, "menu"


def latent_shape(width: int, height: int) -> list[int]:
    """Empty MiniMax-H3-Image latent: [batch 1, 24 channels, 1 frame, height / 16, width / 16]."""
    if width % MULTIPLE or height % MULTIPLE:
        raise ValueError(
            "width and height must be multiples of %d, got %dx%d"
            % (MULTIPLE, width, height)
        )
    return [1, 24, 1, height // 16, width // 16]
