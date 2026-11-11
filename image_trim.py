"""Keep a memory-mapped model file from filling RAM while it is copied to the GPU (Windows).

ComfyUI-GGUF maps the .gguf file; each page read on the way to the GPU stays in this process's working
set, so a 15 GB file takes 15 GB of available RAM until the load ends. The pages are clean copies of
the file, so handing them back costs nothing: this trims the working set whenever it has grown by
`step_mb` since the last trim. Other systems count such pages as available already, so nothing runs there.
"""

import contextlib
import sys
import threading

TRIM_STEP_MB = 1024
POLL_SECONDS = 0.2


def _windows_api():
    """(read working set MB, trim) for this process, or None when not on Windows."""
    if sys.platform != "win32":
        return None
    import ctypes

    import psutil

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    process = psutil.Process()

    def working_set_mb() -> int:
        return process.memory_info().rss // (1024 * 1024)

    def trim() -> None:
        kernel32.K32EmptyWorkingSet(kernel32.GetCurrentProcess())

    return working_set_mb, trim


def trim_loop(working_set_mb, trim, stop, step_mb: int = TRIM_STEP_MB, poll: float = POLL_SECONDS) -> int:
    """Trim each time the working set is `step_mb` above its level after the last trim; return the count."""
    floor = working_set_mb()
    trims = 0
    while not stop.wait(poll):
        if working_set_mb() - floor >= step_mb:
            trim()
            trims += 1
            floor = working_set_mb()
    return trims


@contextlib.contextmanager
def mapped_pages_trimmed(enabled: bool = True, api=_windows_api):
    """Run the block with a background trimmer; does nothing when disabled or not on Windows."""
    parts = api() if enabled else None
    if parts is None:
        yield
        return
    stop = threading.Event()
    worker = threading.Thread(target=trim_loop, args=(*parts, stop), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=5)
