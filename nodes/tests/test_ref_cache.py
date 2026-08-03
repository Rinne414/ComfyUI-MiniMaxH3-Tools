"""Tests for the reference encode cache.

Uses a fake VAE that counts encodes, so cache behaviour is measured rather than
assumed. The key property under test is that caching never changes the result.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ref_cache  # noqa: E402
from ref_cache import (  # noqa: E402
    REF_IMAGE_SHORT_EDGE, cache_key, encode_cached, target_geometry,
)


class CountingVAE:
    """Deterministic stand-in: latent is a function of the input, so identical
    inputs must yield identical latents and any drift shows up immediately."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, pixels):
        self.encode_calls += 1
        b, h, w, c = pixels.shape
        # [B,24,1,H/16,W/16], seeded by the pixel content
        base = pixels.mean() * 1000.0
        out = torch.full((b, 24, 1, h // 16, w // 16), float(base))
        return out


@pytest.fixture(autouse=True)
def _clear_cache():
    ref_cache._CACHE.clear()
    ref_cache._STATS.update(hits=0, misses=0, evictions=0)
    yield
    ref_cache._CACHE.clear()


def make_image(height=480, width=864, seed=0, fill=None):
    if fill is not None:
        return torch.full((1, height, width, 3), float(fill))
    torch.manual_seed(seed)
    return torch.rand(1, height, width, 3)


def test_second_call_is_a_hit_and_skips_encode():
    vae = CountingVAE()
    img = make_image()
    first, _, _, hit1 = encode_cached(vae, img, 864, 480, "match")
    second, _, _, hit2 = encode_cached(vae, img, 864, 480, "match")

    assert (hit1, hit2) == (False, True)
    assert vae.encode_calls == 1, "the cached call must not re-encode"
    assert torch.equal(first, second)


def test_cached_result_is_identical_to_uncached():
    img = make_image(seed=3)

    vae_cold = CountingVAE()
    cold, _, _, _ = encode_cached(vae_cold, img, 864, 480, "match")
    cold = cold.clone()

    ref_cache._CACHE.clear()
    vae_warm = CountingVAE()
    encode_cached(vae_warm, img, 864, 480, "match")
    warm, _, _, hit = encode_cached(vae_warm, img, 864, 480, "match")

    assert hit
    assert torch.equal(cold, warm), "caching must not change the latent"


def test_different_content_misses():
    vae = CountingVAE()
    encode_cached(vae, make_image(seed=1), 864, 480, "match")
    encode_cached(vae, make_image(seed=2), 864, 480, "match")
    assert vae.encode_calls == 2


def test_size_mode_change_misses():
    vae = CountingVAE()
    # a large reference so 'match' and 'max' resolve to different geometry
    img = make_image(height=2400, width=3200, seed=5)
    _, tw_match, th_match, _ = encode_cached(vae, img, 864, 480, "match")
    _, tw_max, th_max, hit = encode_cached(vae, img, 864, 480, "max")

    assert (tw_match, th_match) != (tw_max, th_max)
    assert not hit, "changing ref_image_size must invalidate the entry"
    assert vae.encode_calls == 2


def test_generation_size_change_misses_in_match_mode():
    vae = CountingVAE()
    img = make_image(height=1200, width=1600, seed=6)
    encode_cached(vae, img, 864, 480, "match")
    _, _, _, hit = encode_cached(vae, img, 1216, 672, "match")
    assert not hit, "'match' geometry depends on generation size"
    assert vae.encode_calls == 2


def test_generation_size_change_hits_in_max_mode():
    # 'max' geometry ignores the generation size, so the entry is reusable.
    vae = CountingVAE()
    img = make_image(height=2400, width=3200, seed=7)
    encode_cached(vae, img, 864, 480, "max")
    _, _, _, hit = encode_cached(vae, img, 1216, 672, "max")
    assert hit
    assert vae.encode_calls == 1


def test_cached_latents_stay_on_cpu():
    vae = CountingVAE()
    encode_cached(vae, make_image(), 864, 480, "match")
    for value in ref_cache._CACHE.values():
        assert value.device.type == "cpu"


def test_entry_bound_is_enforced():
    vae = CountingVAE()
    for i in range(6):
        encode_cached(vae, make_image(seed=i), 864, 480, "match", max_entries=3)
    assert len(ref_cache._CACHE) <= 3
    assert ref_cache._STATS["evictions"] >= 3


def test_size_bound_is_enforced():
    vae = CountingVAE()
    for i in range(8):
        encode_cached(vae, make_image(seed=100 + i), 864, 480, "match",
                      max_entries=999, max_gib=0.0005)
    held = sum(v.numel() * v.element_size() for v in ref_cache._CACHE.values())
    assert held <= 0.0005 * 1024 ** 3 or len(ref_cache._CACHE) <= 1


def test_lru_keeps_the_recently_used_entry():
    vae = CountingVAE()
    a, b, c = (make_image(seed=s) for s in (11, 12, 13))
    encode_cached(vae, a, 864, 480, "match", max_entries=2)
    encode_cached(vae, b, 864, 480, "match", max_entries=2)
    encode_cached(vae, a, 864, 480, "match", max_entries=2)   # refresh a
    encode_cached(vae, c, 864, 480, "match", max_entries=2)   # should evict b

    _, _, _, a_hit = encode_cached(vae, a, 864, 480, "match", max_entries=2)
    assert a_hit, "the recently used entry was evicted"


def test_target_geometry_matches_core_node_match_mode():
    # replicates comfy_extras/nodes_minimax_h3.py L222-228 for size_mode 'match'
    import math
    for (gw, gh, rw, rh) in ((864, 480, 1920, 1080), (1216, 672, 800, 600),
                             (864, 480, 400, 300)):
        scale = min(1.0, math.sqrt((gw * gh) / (rw * rh)))
        want_w = max(32, round(rw * scale / 32) * 32)
        want_h = max(32, round(rh * scale / 32) * 32)
        assert target_geometry(gw, gh, rw, rh, "match") == (want_w, want_h)


def test_target_geometry_matches_core_node_max_mode():
    for (rw, rh) in ((4000, 3000), (1000, 800), (300, 200)):
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(rw, rh))
        want_w = max(32, round(rw * scale / 32) * 32)
        want_h = max(32, round(rh * scale / 32) * 32)
        assert target_geometry(864, 480, rw, rh, "max") == (want_w, want_h)


def test_geometry_never_upscales():
    # both modes clamp scale at 1.0
    tw, th = target_geometry(3840, 2160, 320, 240, "match")
    assert tw <= 320 + 32 and th <= 240 + 32
    tw, th = target_geometry(864, 480, 320, 240, "max")
    assert tw <= 320 + 32 and th <= 240 + 32


def test_geometry_is_on_the_32_grid():
    for (rw, rh) in ((1920, 1080), (777, 333), (100, 100), (4000, 2250)):
        for mode in ("match", "max"):
            tw, th = target_geometry(864, 480, rw, rh, mode)
            assert tw % 32 == 0 and th % 32 == 0
            assert tw >= 32 and th >= 32


def test_cache_key_includes_all_three_dimensions():
    a = cache_key("abc", 864, 480, "match")
    assert a != cache_key("abd", 864, 480, "match")
    assert a != cache_key("abc", 1216, 480, "match")
    assert a != cache_key("abc", 864, 672, "match")
    assert a != cache_key("abc", 864, 480, "max")


def test_visually_identical_images_share_an_entry():
    vae = CountingVAE()
    encode_cached(vae, make_image(fill=0.5), 864, 480, "match")
    _, _, _, hit = encode_cached(vae, make_image(fill=0.5), 864, 480, "match")
    assert hit
    assert vae.encode_calls == 1


def test_near_identical_images_do_not_collide():
    vae = CountingVAE()
    img_a = make_image(fill=0.5)
    img_b = img_a.clone()
    img_b[0, 0, 0, 0] = 0.51          # one pixel differs
    encode_cached(vae, img_a, 864, 480, "match")
    _, _, _, hit = encode_cached(vae, img_b, 864, 480, "match")
    assert not hit, "a one-pixel change must not hit the cache"
    assert vae.encode_calls == 2


def test_input_image_is_not_mutated():
    vae = CountingVAE()
    img = make_image(seed=21)
    before = img.clone()
    encode_cached(vae, img, 864, 480, "match")
    assert torch.equal(before, img)
