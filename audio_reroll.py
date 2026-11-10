"""Audio-only reroll for MiniMax-H3.

H3 samples video and audio in one loop, so changing the score today means
regenerating the picture too -- 9+ minutes at 0.4 MP. Wan and LTX cannot do
better because their audio is added afterwards by a separate model; H3
structurally can, because the two streams are separate tensors in one pack.

The approach here does NOT try to freeze half of a jointly-denoised pack
mid-sampling. Instead it uses machinery the DiT already has:

`model_base.MiniMaxH3.extra_conds` forwards `minimax_refs` to the DiT, and
`comfy/ldm/minimax/model.py:473-495` gives every reference block a noise
augmentation factor. At `visual_cond_noise_aug = 1.0` the reference enters
completely un-noised, and `_forward` pins its timestep near 1.0
(`model.py:534-540`), i.e. "already clean, treat as given".

So: take the finished video latent, hand it back as a `kind: "video"` reference
at aug 1.0, and sample a fresh AV pack. The video stream is regenerated, but
conditioned on itself at full strength -- which is the closest thing to a freeze
that the architecture exposes without patching core.

MEASURED RESULT: see docs/AUDIO_REROLL_FINDINGS.md. This is honest about what
it does and does not achieve -- the video is *strongly* preserved, not
bit-identical, because it is still resampled. If you need a bit-identical
picture, mux the new audio over the old video downstream instead; the
`video_passthrough` output exists for exactly that.
"""

import comfy.nested_tensor
import node_helpers
from comfy_api.latest import ComfyExtension, io

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID_MOD = 17
FRAME_GRID_REM = 5


def _frames_from_latent_t(latent_t):
    if latent_t == 2:
        return 5
    if latent_t < 2 or (latent_t - 2) % 5 != 0:
        return None
    return ((latent_t - 2) // 5) * FRAME_GRID_MOD + FRAME_GRID_REM


def _unpack(latent):
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError(
            "expected a MiniMax-H3 AV latent (NestedTensor pair), got %s. Feed "
            "this node the LATENT output of SamplerCustomAdvanced from an H3 run."
            % type(samples).__name__)
    parts = samples.unbind()
    if len(parts) != 2:
        raise ValueError("expected 2 streams, got %d" % len(parts))
    return parts[0], parts[1]


class MiniMaxH3XAudioReroll(io.ComfyNode):
    """Re-sample audio against a finished video latent used as a clean reference.

    Wire `conditioning` from the same MiniMaxH3ImageToVideo / ReferenceToVideo
    node that produced the original run, but with the audio text fields changed.
    The video latent goes in as `av_latent`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XAudioReroll",
            is_experimental=True,
            display_name=(
                "MiniMax H3 Audio Reroll (Experimental) / 音频重生成（实验性）"),
            category="model/conditioning/minimax",
            search_aliases=[
                "h3", "minimax", "audio reroll", "new soundtrack",
                "regenerate audio", "resample audio", "sound", "音频重生成",
                "重新生成音频", "音频", "配乐",
            ],
            description=(
                "Experimental: condition a fresh H3 sampling run on an already-generated video "
                "latent, so a new soundtrack can be produced without re-authoring "
                "the picture. The video is strongly preserved but resampled, not "
                "frozen -- use video_passthrough downstream for a bit-identical mux. / "
                "以已生成的视频潜空间作为参考重新采样音频；画面会被强力保留但仍会"
                "重采样。如需逐位一致的画面，请在下游使用 video_passthrough 混流。"),
            inputs=[
                io.Conditioning.Input("conditioning",
                    tooltip=(
                        "Conditioning from an H3 conditioning node, with the "
                        "soundscape/music text changed. / 来自 H3 条件节点的条件，"
                        "其中环境音或音乐文字已经修改。")),
                io.Latent.Input("av_latent",
                    tooltip=(
                        "The AV latent from the run whose picture you want to keep. / "
                        "需要保留画面的原始运行所产生的音视频潜空间。")),
                io.Float.Input("video_cond_strength", default=1.0, min=0.5, max=1.0, step=0.005,
                    tooltip=(
                        "Noise augmentation on the video reference. 1.0 feeds it "
                        "perfectly clean for maximum preservation; lower values let "
                        "the picture drift. The DiT keyframe default is 0.999. / "
                        "视频参考的噪声增强值。1.0 表示完全干净并最大程度保留画面；"
                        "较低值会允许画面变化。DiT 的关键帧默认值为 0.999。")),
                io.Boolean.Input("reset_audio_latent", default=True,
                    tooltip=(
                        "Start the audio stream from zeros so the sampler writes a "
                        "genuinely new track. Off reuses the old audio latent and "
                        "biases the result toward the original. / 从零开始音频流，让"
                        "采样器生成全新音轨；关闭时复用旧音频潜空间，结果会偏向原音轨。")),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="video_passthrough"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, av_latent, video_cond_strength=1.0,
                reset_audio_latent=True) -> io.NodeOutput:
        video, audio = _unpack(av_latent)
        latent_t, lh, lw = video.shape[2], video.shape[3], video.shape[4]
        frames = _frames_from_latent_t(latent_t)
        if frames is None:
            raise ValueError(
                "video latent_t=%d is off the model's grid; this latent did not "
                "come from a normal H3 run" % latent_t)

        expected_audio_t = round(frames / FPS * AUDIO_LATENT_FPS)
        if audio.shape[3] != expected_audio_t:
            raise ValueError(
                "stream mismatch: latent_t=%d implies %d frames and audio_t=%d, "
                "but the audio stream has audio_t=%d"
                % (latent_t, frames, expected_audio_t, audio.shape[3]))

        # A whole-clip video reference. kind "video" (not "image") because only
        # that branch of PackedLayout accepts a multi-frame latent_t; see
        # comfy/ldm/minimax/model.py:358-375.
        ref_block = {
            "kind": "video",
            "latent_t": latent_t,
            "latent_h": lh,
            "latent_w": lw,
            "ref_audio_t": 0,
            "latent": video,
            "audio_latent": None,
        }

        cond = node_helpers.conditioning_set_values(conditioning, {
            "minimax_refs": [ref_block],
            "minimax_visual_cond_noise_aug": float(video_cond_strength),
        })

        fresh_audio = audio.new_zeros(audio.shape) if reset_audio_latent else audio.clone()
        out_latent = {"samples": comfy.nested_tensor.NestedTensor(
            (video.new_zeros(video.shape), fresh_audio))}

        info = "\n".join([
            "video ref   %dx%d px, %d frames (%.3f s), latent_t=%d" % (
                lw * 16, lh * 16, frames, frames / FPS, latent_t),
            "cond aug    %.3f  (1.0 = fed perfectly clean)" % video_cond_strength,
            "audio       %s, audio_t=%d" % (
                "reset to zeros" if reset_audio_latent else "seeded from the original",
                fresh_audio.shape[3]),
            "note        the picture is resampled under strong self-conditioning,",
            "            not frozen. Use video_passthrough + a downstream mux if",
            "            you need the original frames byte-for-byte.",
        ])

        return io.NodeOutput(cond, out_latent, {"samples": video}, info)


class AudioRerollExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3XAudioReroll]


async def comfy_entrypoint() -> AudioRerollExtension:
    return AudioRerollExtension()
