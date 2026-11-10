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
    """Report what was checked for the NF4 loaders, not a promise that they work.

    The loader nodes are always registered and check again when they run.
    BitsAndBytes is read from package metadata, not imported, so startup stays
    fast and never touches the GPU.
    """
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return "NF4 loaders: bitsandbytes not installed"
    found = "bitsandbytes %s found" % version if version else "bitsandbytes found"
    if version and _version_tuple(version) < NF4_MINIMUM_VERSION:
        needed = ".".join(str(part) for part in NF4_MINIMUM_VERSION)
        found += " (%s or newer is tested)" % needed
    cuda = "CUDA device found" if _cuda_available() else "no CUDA device found"
    return "NF4 loaders: %s, %s" % (found, cuda)


def log_startup_status() -> None:
    """Log the status line. A failing probe is logged at debug level and ignored."""
    try:
        LOGGER.info("%s: %s", PREFIX, nf4_status())
    except Exception:
        LOGGER.debug("%s: could not determine NF4 status", PREFIX, exc_info=True)
