"""Import the pack the way ComfyUI does and check that all twelve nodes register."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

EXPECTED_NODES = frozenset(
    {
        "MiniMaxH3XCanvasSolver",
        "MiniMaxH3XSplitAV",
        "MiniMaxH3XJoinAV",
        "MiniMaxH3XLatentInfo",
        "MiniMaxH3XUpscaleVideoLatent",
        "MiniMaxH3XTemporalTrim",
        "MiniMaxH3XPromptLint",
        "MiniMaxH3XAudioReroll",
        "MiniMaxH3XNF4ModelLoader",
        "MiniMaxH3XNF4TextEncoderLoader",
        "MiniMaxH3XNF4VideoVAELoader",
        "MiniMaxH3XNF4AudioVAELoader",
    }
)


def load_extension(root: Path):
    spec = importlib.util.spec_from_file_location(
        "comfyui_minimaxh3_tools",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create the extension import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_extension(Path(__file__).resolve().parents[1])
    extension = asyncio.run(module.comfy_entrypoint())
    names = {node.__name__ for node in asyncio.run(extension.get_node_list())}
    if names != EXPECTED_NODES:
        raise RuntimeError(
            "extension registration mismatch: expected %r, got %r"
            % (sorted(EXPECTED_NODES), sorted(names))
        )
    print("registered nodes:", ", ".join(sorted(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
