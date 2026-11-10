"""One log line per optional capability, written when the pack is imported."""

from __future__ import annotations

import importlib.metadata
import logging
import re

LOGGER = logging.getLogger(__name__)

PREFIX = "MiniMax-H3 Tools"
NF4_MINIMUM_VERSION = (0, 50, 0)


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version.split("+")[0])[:3])


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def nf4_status() -> str:
    """Describe whether the NF4 loaders can run on this machine.

    The loader nodes are always registered; this only reports what they will do
    when run. BitsAndBytes is read from package metadata, not imported, so
    startup stays fast and never touches the GPU.
    """
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return "NF4 loaders unavailable (bitsandbytes not installed)"
    if not _cuda_available():
        return "NF4 loaders unavailable (no NVIDIA CUDA device found)"
    if _version_tuple(version) < NF4_MINIMUM_VERSION:
        needed = ".".join(str(part) for part in NF4_MINIMUM_VERSION)
        return "NF4 loaders available (bitsandbytes %s; %s or newer is tested)" % (
            version,
            needed,
        )
    return "NF4 loaders available (bitsandbytes %s)" % version


def log_startup_status() -> None:
    LOGGER.info("%s: %s", PREFIX, nf4_status())
