"""Turn ComfyUI's pinned host memory off for one block of code (no ComfyUI imports, testable).

The dynamic-VRAM loader pins the weights that do not fit in VRAM, and its budget lets available RAM
fall to about 2 GB. For a 27 GB text encoder that is used for a single encode, pinning buys little and
can freeze a 32 GB PC. comfy/pinned_memory.py reads comfy.model_management.DISABLE_PINNED_MEMORY on
every call (pin_memory and register), so flipping it for the duration of load + encode is enough.
The previous value is always restored, also when the block raises.
"""

import contextlib


@contextlib.contextmanager
def pinning_disabled(model_management):
    previous = model_management.DISABLE_PINNED_MEMORY
    model_management.DISABLE_PINNED_MEMORY = True
    try:
        yield
    finally:
        model_management.DISABLE_PINNED_MEMORY = previous
