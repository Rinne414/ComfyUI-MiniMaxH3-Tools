"""Tests for the AV latent toolkit.

Runs against real NestedTensor pairs, so it needs torch and ComfyUI's comfy
package on the path. Invoke from the ComfyUI root:

    python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Tools/tests -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from av_latent import (  # noqa: E402
    MiniMaxH3XJoinAV, MiniMaxH3XLatentInfo, MiniMaxH3XSplitAV,
    MiniMaxH3XTemporalTrim, MiniMaxH3XUpscaleVideoLatent,
    audio_t_for_frames, frames_from_latent_t, pack_av, unpack_av,
    video_latent_t,
)


def make_latent(frames=124, width=864, height=480, batch=1, seed=0):
    torch.manual_seed(seed)
    lt = video_latent_t(frames)
    at = audio_t_for_frames(frames)
    video = torch.randn(batch, 24, lt, height // 16, width // 16)
    audio = torch.randn(batch, 32, 2, at)
    return pack_av(video, audio)


def test_frames_latent_t_roundtrip():
    for frames in (5, 22, 39, 124, 175, 243, 362):
        assert frames_from_latent_t(video_latent_t(frames)) == frames


def test_frames_from_latent_t_rejects_off_grid():
    # valid latent_t values are 2, 7, 12, 17, ... i.e. (latent_t-2) % 5 == 0
    for bad in (3, 4, 5, 6, 8, 9, 10, 11, 13, 46, 48):
        assert frames_from_latent_t(bad) is None, "%d should be off-grid" % bad


def test_frames_from_latent_t_accepts_on_grid():
    for good in (2, 7, 12, 37, 72, 107):
        assert frames_from_latent_t(good) is not None, "%d should be on-grid" % good


def test_split_join_is_bit_identical():
    src = make_latent()
    v0, a0 = unpack_av(src)

    video, audio = MiniMaxH3XSplitAV.execute(src).result
    rejoined = MiniMaxH3XJoinAV.execute(video, audio).result[0]
    v1, a1 = unpack_av(rejoined)

    assert torch.equal(v0, v1)
    assert torch.equal(a0, a1)


def test_split_tags_streams():
    video, audio = MiniMaxH3XSplitAV.execute(make_latent()).result
    assert video["minimax_h3_stream"] == "video"
    assert audio["minimax_h3_stream"] == "audio"


def test_join_rejects_swapped_inputs():
    video, audio = MiniMaxH3XSplitAV.execute(make_latent()).result
    with pytest.raises(ValueError, match="swapped"):
        MiniMaxH3XJoinAV.execute(audio, video)


def test_join_rejects_mismatched_streams():
    video, _ = MiniMaxH3XSplitAV.execute(make_latent(frames=124)).result
    _, audio = MiniMaxH3XSplitAV.execute(make_latent(frames=243)).result
    with pytest.raises(ValueError, match="stream mismatch"):
        MiniMaxH3XJoinAV.execute(video, audio)


def test_join_error_names_both_shapes():
    video, _ = MiniMaxH3XSplitAV.execute(make_latent(frames=124)).result
    _, audio = MiniMaxH3XSplitAV.execute(make_latent(frames=243)).result
    with pytest.raises(ValueError) as err:
        MiniMaxH3XJoinAV.execute(video, audio)
    msg = str(err.value)
    assert "latent_t=37" in msg and "audio_t=405" in msg


def test_join_rejects_off_grid_latent_t():
    video = {"samples": torch.randn(1, 24, 8, 30, 54), "minimax_h3_stream": "video"}
    audio = {"samples": torch.randn(1, 32, 2, 207), "minimax_h3_stream": "audio"}
    with pytest.raises(ValueError, match="off the model's grid"):
        MiniMaxH3XJoinAV.execute(video, audio)


def test_join_rejects_wrong_channel_counts():
    video = {"samples": torch.randn(1, 16, 37, 30, 54), "minimax_h3_stream": "video"}
    audio = {"samples": torch.randn(1, 32, 2, 207), "minimax_h3_stream": "audio"}
    with pytest.raises(ValueError, match="24"):
        MiniMaxH3XJoinAV.execute(video, audio)


def test_join_rejects_mono_audio():
    video = {"samples": torch.randn(1, 24, 37, 30, 54), "minimax_h3_stream": "video"}
    audio = {"samples": torch.randn(1, 32, 1, 207), "minimax_h3_stream": "audio"}
    with pytest.raises(ValueError, match=r"dim 2 must be 2"):
        MiniMaxH3XJoinAV.execute(video, audio)


def test_unpack_rejects_plain_latent():
    with pytest.raises(ValueError, match="NestedTensor"):
        unpack_av({"samples": torch.randn(1, 16, 8, 64, 64)})


def test_upscale_touches_video_only():
    src = make_latent(frames=124, width=864, height=480)
    _, audio_before = unpack_av(src)

    out = MiniMaxH3XUpscaleVideoLatent.execute(src, 1216, 672, "bicubic").result[0]
    video_after, audio_after = unpack_av(out)

    assert video_after.shape == (1, 24, 37, 672 // 16, 1216 // 16)
    assert torch.equal(audio_before, audio_after), "audio must pass through untouched"


def test_upscale_snaps_to_32():
    src = make_latent()
    out = MiniMaxH3XUpscaleVideoLatent.execute(src, 1200, 700, "bilinear").result[0]
    video, _ = unpack_av(out)
    # 1200 -> 1216, 700 -> 704
    assert video.shape[4] * 16 % 32 == 0
    assert video.shape[3] * 16 % 32 == 0
    assert video.shape[4] == 1216 // 16
    assert video.shape[3] == 704 // 16


def test_upscale_is_identity_at_same_size():
    src = make_latent(width=864, height=480)
    v0, a0 = unpack_av(src)
    out = MiniMaxH3XUpscaleVideoLatent.execute(src, 864, 480, "bicubic").result[0]
    v1, a1 = unpack_av(out)
    assert torch.equal(v0, v1)
    assert torch.equal(a0, a1)


def test_upscale_preserves_temporal_length():
    src = make_latent(frames=243)
    out = MiniMaxH3XUpscaleVideoLatent.execute(src, 1216, 672, "bicubic").result[0]
    video, _ = unpack_av(out)
    assert video.shape[2] == video_latent_t(243)


def test_upscale_does_not_blend_frames():
    """Folding time into batch must keep frames independent."""
    lt = video_latent_t(124)
    video = torch.zeros(1, 24, lt, 30, 54)
    video[:, :, 3] = 1.0                      # one bright frame
    audio = torch.randn(1, 32, 2, audio_t_for_frames(124))
    out = MiniMaxH3XUpscaleVideoLatent.execute(
        pack_av(video, audio), 1216, 672, "nearest-exact").result[0]
    got, _ = unpack_av(out)
    assert got[:, :, 3].min() == 1.0
    for t in (2, 4):
        assert got[:, :, t].max() == 0.0, "frame %d picked up signal from its neighbour" % t


def test_trim_moves_both_streams():
    src = make_latent(frames=243)
    out, frames = MiniMaxH3XTemporalTrim.execute(src, 5.0).result
    video, audio = unpack_av(out)
    assert frames == 124
    assert video.shape[2] == video_latent_t(124)
    assert audio.shape[3] == audio_t_for_frames(124)


def test_trim_snaps_to_the_frame_grid():
    src = make_latent(frames=243)
    # keep_seconds is snapped UP, matching how the generation nodes read a
    # duration, then clamped to what the source holds.
    for secs, expected in ((5.0, 124), (4.5, 124), (6.0, 158), (7.0, 175)):
        _, frames = MiniMaxH3XTemporalTrim.execute(src, secs).result
        assert frames == expected, "%.1f s gave %d, expected %d" % (secs, frames, expected)
        assert frames % 17 == 5
        assert frames <= 243


def test_trim_result_is_a_valid_pair():
    src = make_latent(frames=362)
    out, frames = MiniMaxH3XTemporalTrim.execute(src, 7.0).result
    video, audio = unpack_av(out)
    # Join in strict mode is the real validator -- it must accept the result.
    MiniMaxH3XJoinAV.execute({"samples": video, "minimax_h3_stream": "video"},
                             {"samples": audio, "minimax_h3_stream": "audio"})


def test_trim_cannot_extend():
    src = make_latent(frames=124)
    with pytest.raises(ValueError, match="cannot extend"):
        MiniMaxH3XTemporalTrim.execute(src, 12.0)


def test_trim_preserves_content_prefix():
    src = make_latent(frames=243, seed=7)
    v0, a0 = unpack_av(src)
    out, _ = MiniMaxH3XTemporalTrim.execute(src, 5.0).result
    v1, a1 = unpack_av(out)
    assert torch.equal(v0[:, :, :v1.shape[2]], v1)
    assert torch.equal(a0[:, :, :, :a1.shape[3]], a1)


def test_latent_info_reports_correct_geometry():
    src = make_latent(frames=243, width=1216, height=672)
    report, frames, tokens = MiniMaxH3XLatentInfo.execute(src).result
    assert frames == 243
    assert tokens == 72 * (672 // 32) * (1216 // 32)
    assert tokens == 57456          # matches the measured calibration row
    assert "1216x672" in report
    assert "MISMATCH" not in report


def test_latent_info_flags_mismatch():
    video = torch.randn(1, 24, 37, 30, 54)
    audio = torch.randn(1, 32, 2, 405)          # wrong: 37 implies 207
    report, _, _ = MiniMaxH3XLatentInfo.execute(pack_av(video, audio)).result
    assert "MISMATCH" in report


def test_batch_is_preserved_through_split_join():
    src = make_latent(batch=2)
    video, audio = MiniMaxH3XSplitAV.execute(src).result
    out = MiniMaxH3XJoinAV.execute(video, audio).result[0]
    v, a = unpack_av(out)
    assert v.shape[0] == 2 and a.shape[0] == 2


def test_no_input_mutation():
    src = make_latent(frames=243)
    v0, a0 = unpack_av(src)
    v0_clone, a0_clone = v0.clone(), a0.clone()

    MiniMaxH3XSplitAV.execute(src)
    MiniMaxH3XUpscaleVideoLatent.execute(src, 1216, 672, "bicubic")
    MiniMaxH3XTemporalTrim.execute(src, 5.0)
    MiniMaxH3XLatentInfo.execute(src)

    v1, a1 = unpack_av(src)
    assert torch.equal(v0_clone, v1), "video stream was mutated in place"
    assert torch.equal(a0_clone, a1), "audio stream was mutated in place"
