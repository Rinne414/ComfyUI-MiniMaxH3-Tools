"""Optional GGUF text encoders through the installed ComfyUI-GGUF pack (city96), without copying its code.

ComfyUI imports each custom node folder as a module keyed by its full folder path (dots replaced), so the
GGUF pack's `nodes` module is found by the folder name at the end of that key. When it is not there, a
plain sentence says what to install.
"""

GGUF_PACK_NAMES = ("ComfyUI-GGUF", "comfyui-gguf", "ComfyUI-GGUF-main")
MISSING = (
    "This text encoder is a .gguf file. Loading it needs the ComfyUI-GGUF "
    "node pack (city96/ComfyUI-GGUF); install it, or pick the .safetensors "
    "encoder."
)


def is_gguf(name: str) -> bool:
    return name.lower().endswith(".gguf")


def pack_folder(module_key: str) -> str:
    """Folder name of a '<path>.nodes' module key, lower case, for either path separator."""
    base = module_key[: -len(".nodes")]
    return base.replace("\\", "/").rsplit("/", 1)[-1].lower()


def find_gguf_loader(modules: dict):
    """The CLIPLoaderGGUF class of an installed ComfyUI-GGUF pack, or None."""
    wanted = {name.lower() for name in GGUF_PACK_NAMES}
    for key, mod in list(modules.items()):
        if not key.endswith(".nodes") or pack_folder(key) not in wanted:
            continue
        cls = getattr(mod, "CLIPLoaderGGUF", None)
        if (
            cls is not None
            and hasattr(cls, "load_patcher")
            and hasattr(cls, "load_data")
        ):
            return cls
    return None


def gguf_parts(loader, modules: dict):
    """(GGMLOps, GGUFModelPatcher) from the module that defines the loader, or None if it lacks them."""
    mod = modules.get(getattr(loader, "__module__", ""))
    ops = getattr(mod, "GGMLOps", None)
    patcher = getattr(mod, "GGUFModelPatcher", None)
    if ops is None or patcher is None or not hasattr(patcher, "clone"):
        return None
    return ops, patcher


VRAM_MARGIN = (
    3 * 1024**3
)  # activations of the encode and the CUDA context need room next to the weights


def vram_problem(
    file_bytes: int, free_vram_bytes: int, margin: int = VRAM_MARGIN
) -> str:
    """'' when the quantized file fits in free VRAM with margin, else one plain sentence with the numbers.
    The GGUF encoder is built on the GPU, so it must fit there whole."""
    if file_bytes + margin <= free_vram_bytes:
        return ""
    gib = 1024**3
    return (
        f"This GGUF text encoder needs about {file_bytes / gib:.1f} GB of free VRAM plus {margin / gib:.0f} GB room, "
        f"but only {free_vram_bytes / gib:.1f} GB is free; close other GPU programs or pick a smaller quant."
    )
