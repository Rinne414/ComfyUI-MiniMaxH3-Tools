"""Conditioning <-> .safetensors, no pickle.

A ComfyUI CONDITIONING value is a list of [tensor, dict] pairs whose dicts may hold more tensors,
numbers, strings, None and nested lists/dicts (MiniMax H3 Image puts token tags and reference
latents there). Every tensor is stored as a safetensors entry; the rest of the structure is JSON
in the file metadata, where each tensor is replaced by {"__tensor__": "<key>"} and every dict
is wrapped as {"__dict__": {...}} so stored keys can never collide with the markers.

This module imports only torch and safetensors, so it can be tested without ComfyUI.
"""

import json
import os
import re
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "h3image-conditioning/1"


def _flatten(value: Any, tensors: dict[str, torch.Tensor], seen: dict[int, str]) -> Any:
    if isinstance(value, torch.Tensor):
        key = seen.get(id(value))
        if key is None:
            key = f"t{len(tensors)}"
            # clone: safetensors refuses entries that share storage (views of one tensor)
            tensors[key] = value.detach().to("cpu").clone().contiguous()
            seen[id(value)] = key
        return {"__tensor__": key}
    if isinstance(value, dict):
        return {
            "__dict__": {str(k): _flatten(v, tensors, seen) for k, v in value.items()}
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_flatten(v, tensors, seen) for v in value]}
    if isinstance(value, list):
        return [_flatten(v, tensors, seen) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"a conditioning file cannot store a {type(value).__name__}")


def _unflatten(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, list):
        return [_unflatten(v, tensors) for v in value]
    if isinstance(value, dict):
        if "__tensor__" in value:
            return tensors[value["__tensor__"]]
        if "__tuple__" in value:
            return tuple(_unflatten(v, tensors) for v in value["__tuple__"])
        return {k: _unflatten(v, tensors) for k, v in value["__dict__"].items()}
    return value


def save_conditioning(
    path: str, conditioning: list, info: dict[str, str] | None = None
) -> None:
    """Write conditioning to path atomically; info is free-form text metadata (prompt, encoder)."""
    tensors: dict[str, torch.Tensor] = {}
    structure = _flatten(list(conditioning), tensors, {})
    metadata = {str(k): str(v) for k, v in (info or {}).items()}
    metadata.update({"format": FORMAT, "structure": json.dumps(structure)})
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, path)


def load_conditioning(path: str) -> tuple[list, dict[str, str]]:
    """Read a file written by save_conditioning; returns (conditioning, metadata). Tensors are copied out
    of the file mapping so the file can be deleted or overwritten while ComfyUI runs."""
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(
                f"{os.path.basename(path)} is not a conditioning file (format {metadata.get('format')!r})"
            )
        tensors = {k: f.get_tensor(k).clone() for k in f.keys()}
    conditioning = _unflatten(json.loads(metadata["structure"]), tensors)
    info = {k: v for k, v in metadata.items() if k not in ("format", "structure")}
    return conditioning, info


_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def clean_name(name: str) -> str:
    name = _NAME.sub("_", name.strip()).strip("._")
    if name.endswith(".safetensors"):
        name = name[: -len(".safetensors")]
    return name or "conditioning"


def free_path(folder: str, name: str) -> str:
    """name.safetensors, or name_2, name_3, ... when it is taken."""
    path = os.path.join(folder, f"{name}.safetensors")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name}_{n}.safetensors")
        n += 1
    return path


def zero_conditioning(conditioning: list) -> list:
    """The model runs without CFG; KSampler still wants a negative, so pass zeros."""
    out = []
    for cond, extra in conditioning:
        extra = dict(extra)
        if isinstance(extra.get("pooled_output"), torch.Tensor):
            extra["pooled_output"] = torch.zeros_like(extra["pooled_output"])
        out.append([torch.zeros_like(cond), extra])
    return out
