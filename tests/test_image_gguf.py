"""Optional ComfyUI-GGUF lookup: found, missing, or an incompatible version (no ComfyUI needed)."""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_gguf as gguf_support  # noqa: E402


class _Loader:
    def load_data(self, paths):
        return paths

    def load_patcher(self, paths, clip_type, data):
        return data


def test_is_gguf():
    assert gguf_support.is_gguf("qwen3vl-32B-MiniMax-H3-Q4_K_M.GGUF")
    assert not gguf_support.is_gguf("qwen3vl_32b_minimax_h3_int8_convrot.safetensors")


def test_finds_the_installed_pack_under_any_known_folder_name():
    for name in gguf_support.GGUF_PACK_NAMES:
        mods = {f"{name}.nodes": types.SimpleNamespace(CLIPLoaderGGUF=_Loader)}
        assert gguf_support.find_gguf_loader(mods) is _Loader


def test_finds_the_pack_keyed_by_its_full_path_like_comfyui_does():
    key = "C:\\apps\\ComfyUI\\custom_nodes\\ComfyUI-GGUF.nodes"
    mods = {
        key: types.SimpleNamespace(CLIPLoaderGGUF=_Loader),
        "other.nodes": types.SimpleNamespace(),
    }
    assert gguf_support.find_gguf_loader(mods) is _Loader
    other = {"C:/x/Not-GGUF.nodes": types.SimpleNamespace(CLIPLoaderGGUF=_Loader)}
    assert gguf_support.find_gguf_loader(other) is None


def test_missing_or_too_old_pack_gives_none_and_a_plain_message():
    assert gguf_support.find_gguf_loader({}) is None
    old = types.SimpleNamespace(
        CLIPLoaderGGUF=type("Old", (), {"load_clip": lambda self: None})
    )
    assert gguf_support.find_gguf_loader({"ComfyUI-GGUF.nodes": old}) is None
    assert "ComfyUI-GGUF" in gguf_support.MISSING


def test_gguf_parts_come_from_the_loader_module_or_none():
    patcher = type("P", (), {"clone": staticmethod(lambda p: p)})
    _Loader.__module__ = "fake_gguf.nodes"
    mods = {
        "fake_gguf.nodes": types.SimpleNamespace(
            GGMLOps=object, GGUFModelPatcher=patcher
        )
    }
    assert gguf_support.gguf_parts(_Loader, mods) == (object, patcher)
    assert (
        gguf_support.gguf_parts(
            _Loader, {"fake_gguf.nodes": types.SimpleNamespace(GGMLOps=object)}
        )
        is None
    )
    assert gguf_support.gguf_parts(_Loader, {}) is None


def test_vram_check_passes_with_room_and_explains_without():
    gib = 1024**3
    assert gguf_support.vram_problem(9 * gib, 20 * gib) == ""
    msg = gguf_support.vram_problem(15 * gib, 16 * gib)
    assert "15.0 GB" in msg and "16.0 GB" in msg
    assert gguf_support.vram_problem(8 * gib, 11 * gib) == ""
    assert gguf_support.vram_problem(8 * gib, 11 * gib - 1) != ""
