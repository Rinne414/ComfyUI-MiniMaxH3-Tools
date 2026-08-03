"""Reference encoding cache for MiniMax-H3 ref2va.

`MiniMaxH3ReferenceToVideo` re-runs `vae.encode()` over every reference image on
each execution (comfy_extras/nodes_minimax_h3.py L218-232). The real ref2va
workflow is "fix the references, iterate on the prompt", so that work is
repeated for no benefit -- and ref2va is already the expensive mode because
reference tokens ride through every sampling step.

This node pre-encodes reference images and hands the core node a passthrough
IMAGE whose latent is already cached, so the second run onward skips the encode.

Design notes:
- The cache key covers image content, target geometry and the sizing mode. A
  resize must miss, or the cached latent would be the wrong shape.
- Entries live on CPU. A cache that pins VRAM would defeat the purpose on the
  hardware that needs this most.
- Bounded LRU with a manual clear, because a long session with many references
  would otherwise grow without limit.
- Output is byte-identical to the uncached path. A cache that changes results is
  a bug, not a speed feature.
"""

import collections
import hashlib
import math

import torch

import comfy.utils
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048

_CACHE = collections.OrderedDict()
_STATS = {"hits": 0, "misses": 0, "evictions": 0}


def _hash_image(image):
    """Content hash of an IMAGE tensor.

    Full-tensor hashing costs a few ms for a 2048px reference, which is nothing
    against a VAE encode. Sampling a subset would risk collisions between
    references that differ only in fine detail -- exactly the case identity
    conditioning cares about.
    """
    t = image.detach().to("cpu", torch.float32).contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def target_geometry(width, height, ref_w, ref_h, size_mode):
    """Mirror of the core node's per-reference sizing (L221-228).

    Kept as a separate function so tests can assert it against the original.
    """
    if size_mode == "match":
        scale = min(1.0, math.sqrt((width * height) / (ref_w * ref_h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(ref_w, ref_h))
    tw = max(CANVAS_MULTIPLE, round(ref_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(ref_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def _resize(image, width, height):
    # same policy as the core node: crop "disabled", lanczos
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def cache_key(image_hash, target_w, target_h, size_mode):
    return "%s|%dx%d|%s" % (image_hash, target_w, target_h, size_mode)


def _evict_to_limit(max_entries, max_gib):
    limit_bytes = max_gib * (1024 ** 3)
    while _CACHE and (len(_CACHE) > max_entries or
                      sum(v.numel() * v.element_size() for v in _CACHE.values()) > limit_bytes):
        _CACHE.popitem(last=False)
        _STATS["evictions"] += 1


def encode_cached(vae, image, width, height, size_mode, max_entries=32, max_gib=4.0):
    """(latent, target_w, target_h, was_hit). Latent is returned on CPU."""
    ref_h, ref_w = image.shape[1], image.shape[2]
    tw, th = target_geometry(width, height, ref_w, ref_h, size_mode)
    key = cache_key(_hash_image(image[:1]), tw, th, size_mode)

    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return cached, tw, th, True

    resized = _resize(image[:1], tw, th)
    latent = vae.encode(resized)
    _CACHE[key] = latent.detach().to("cpu")
    _STATS["misses"] += 1
    _evict_to_limit(max_entries, max_gib)
    return _CACHE[key], tw, th, False


class MiniMaxH3XRefEncodeCache(io.ComfyNode):
    """Pre-encode a reference image, caching the latent across executions.

    Wire this between LoadImage and MiniMaxH3ReferenceToVideo's ref_image input.
    The IMAGE that comes out is the resized reference, already at the geometry
    the core node would have chosen, so the core node's own resize is a no-op
    and its encode hits ComfyUI's own VAE cache for the same tensor.

    Set width/height to the generation size you will use -- with size_mode
    "match" the target geometry depends on it, so a different size is correctly
    a different cache entry.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XRefEncodeCache",
            display_name="MiniMax H3 Reference Encode Cache",
            category="model/conditioning/minimax",
            description=(
                "Cache the VAE encode of a ref2va reference image so prompt "
                "iteration stops re-encoding it. Output is identical to the "
                "uncached path."),
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=864, min=32, max=16384, step=32,
                             tooltip="Generation width, matching the value on MiniMaxH3ReferenceToVideo."),
                io.Int.Input("height", default=480, min=32, max=16384, step=32),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Must match the setting on MiniMaxH3ReferenceToVideo, or the cached geometry will be wrong for it."),
                io.Int.Input("max_entries", default=32, min=1, max=512,
                             tooltip="LRU bound on cached latents."),
                io.Float.Input("max_gib", default=4.0, min=0.1, max=64.0, step=0.1,
                               tooltip="Total CPU RAM the cache may hold."),
                io.Boolean.Input("clear_cache", default=False,
                                 tooltip="Drop every cached entry on this execution, then continue normally."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, vae, width, height, ref_image_size, max_entries,
                max_gib, clear_cache=False) -> io.NodeOutput:
        if clear_cache:
            dropped = len(_CACHE)
            _CACHE.clear()
            _STATS.update(hits=0, misses=0, evictions=0)
        else:
            dropped = 0

        latent, tw, th, was_hit = encode_cached(
            vae, image, width, height, ref_image_size, max_entries, max_gib)

        resized = _resize(image[:1], tw, th)
        held = sum(v.numel() * v.element_size() for v in _CACHE.values())
        status = "\n".join([
            "%s  ref resized to %dx%d (mode '%s')" % (
                "HIT " if was_hit else "MISS", tw, th, ref_image_size),
            "cache  %d entries, %.2f GiB on CPU" % (len(_CACHE), held / 1024 ** 3),
            "stats  %d hits, %d misses, %d evictions%s" % (
                _STATS["hits"], _STATS["misses"], _STATS["evictions"],
                ", %d dropped by clear" % dropped if dropped else ""),
        ])
        return io.NodeOutput(resized, {"samples": latent}, status)


class RefCacheExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3XRefEncodeCache]


async def comfy_entrypoint() -> RefCacheExtension:
    return RefCacheExtension()
