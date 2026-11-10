"""Make the whole suite collectable on a machine without a GPU (the CPU-only CI).

Importing ``comfy.model_management`` raises "No CUDA GPUs are available" unless ComfyUI is told to run
on the CPU. Core's own tests set ``comfy.cli_args.args.cpu`` before that import; doing it here, before
any test module is collected, covers every test that touches ComfyUI, including the ones that skip
themselves with ``pytest.importorskip`` (which only catches ImportError).
"""

from __future__ import annotations

try:
    import torch
    from comfy.cli_args import args
except ImportError:  # ComfyUI is not on the path: only the pure-logic tests can run
    pass
else:
    if not torch.cuda.is_available():
        args.cpu = True
