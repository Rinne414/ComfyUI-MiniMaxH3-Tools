"""Disk cache for encoded prompts (torch + safetensors only, no ComfyUI imports).

Key = sha256 over a format version, the prompt text, the text encoder file identity, the VAE identity
and the exact pixels of every picture, so a hit is safe across restarts and between workflows.
Files are never deleted automatically; the report tells the user how big the folder is.
"""

import hashlib
import json
import os

import torch

if __package__:
    from .conditioning_io import load_conditioning
else:
    from conditioning_io import load_conditioning

FORMAT_VERSION = "h3image-encode-cache/1"


def file_identity(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}|{st.st_size}|{st.st_mtime_ns}"


def cache_key(
    prompt: str, encoder_identity: str, vae_identity: str, images: list[torch.Tensor]
) -> str:
    h = hashlib.sha256()
    for part in (
        FORMAT_VERSION,
        prompt,
        encoder_identity,
        vae_identity,
        str(len(images)),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    for image in images:
        t = image.detach().to("cpu").contiguous()
        h.update(f"{tuple(t.shape)}|{t.dtype}".encode("utf-8"))
        h.update(
            t.numpy().tobytes()
            if t.dtype != torch.bfloat16
            else t.view(torch.int16).numpy().tobytes()
        )
    return h.hexdigest()


def folder_stats(folder: str) -> tuple[int, int]:
    """(file count, total bytes) of the .safetensors files in folder."""
    if not os.path.isdir(folder):
        return 0, 0
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".safetensors")
    ]
    return len(files), sum(os.path.getsize(f) for f in files)


def read_cache(path: str):
    """-> (conditioning or None, what happened)."""
    if not os.path.exists(path):
        return None, "cache miss"
    try:
        cond, _info = load_conditioning(path)
        return cond, "cache hit, text encoder not loaded"
    except Exception as exc:  # unreadable file: re-encode and overwrite it
        return (
            None,
            f"cache file unreadable ({type(exc).__name__}), re-encoded and overwritten",
        )


def read_text_cache(path: str) -> str | None:
    """Cached expander reply, or None when missing or unreadable (then the caller regenerates and overwrites)."""
    try:
        with open(path, encoding="utf-8") as f:
            reply = json.load(f).get("reply")
        return reply if isinstance(reply, str) else None
    except (OSError, ValueError, AttributeError):
        return None


def write_text_cache(path: str, reply: str, info: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"reply": reply, **info}, f, ensure_ascii=False)
    os.replace(tmp, path)
