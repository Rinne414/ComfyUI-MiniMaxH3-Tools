# Audio Reroll — measured findings

What happens when you feed a finished MiniMax-H3 video latent back as a clean
reference and re-sample with different audio text.

**Rig:** RTX 3090 24 GB · 32 GB RAM · ComfyUI v0.30.0 · torch 2.8.0+cu128 ·
xformers · `minimax_h3_ref2va_pruned_int8_convrot` +
`qwen3vl_32b_minimax_h3_int8_convrot`

**Test:** 864×480, 124 frames (5.167 s), 20 steps, `res_multistep`/`simple`,
seed 4242, `shift_video` 12.0 / `shift_audio` 3.0. Identical visual text and
dialogue in both stages; `overall_soundscape` and `non_diegetic_music` replaced
wholesale in stage 2 (storm/orchestral → still evening/acoustic guitar). The
clip contains `<d>[English] The storm is coming. Secure the boats.</d>` so
lip-sync is observable.

## How it works

H3 does not expose a way to freeze half of a jointly-denoised pack mid-sampling.
It does expose reference conditioning:

- `model_base.MiniMaxH3.extra_conds` forwards `minimax_refs` to the DiT
  (`comfy/model_base.py:2095-2099`)
- every reference block gets a noise-augmentation factor, and at
  `visual_cond_noise_aug = 1.0` the reference enters completely un-noised
  (`comfy/ldm/minimax/model.py:473-482`)
- `_forward` then pins that segment's timestep at `max(t_v, vis_aug)`
  (`model.py:534-540`) — effectively "already clean, treat as given"

So `MiniMaxH3XAudioReroll` hands the finished video latent back as a whole-clip
`kind: "video"` reference at aug 1.0. Only the `"video"` branch of
`PackedLayout` accepts a multi-frame `latent_t` (`model.py:358-375`); the
`"image"` branch used by keyframes is single-frame only.

## Measured result

It works. The audio changes, the picture is strongly preserved, and it is
slower than generating from scratch.

| Measurement | Value |
|---|---|
| Video PSNR, reroll vs original | **43.60 dB average** (y 42.24, u 49.20, v 49.55; min 42.68, max 43.82) |
| Audio SHA-256 | different — `6876d34f13ae…` vs `983c79b465ac…` |
| Audio level, original | mean −27.7 dB, max −10.4 dB |
| Audio level, reroll | mean −28.8 dB, max −10.5 dB |
| Sampling speed, original | **26 s/it** |
| Sampling speed, reroll | **63 s/it** |
| Total wall clock, both stages | 31.7 min |

**PSNR 43.6 dB means the picture is very close but not identical.** For
reference, a lossy re-encode of the same frames typically lands in the 40s, so
the reroll's visual drift is on the order of a compression pass rather than a
regeneration. The `min` of 42.68 across the clip shows no frame collapsed.

The audio genuinely changed rather than being reproduced: different hash, and a
1.1 dB shift in mean level consistent with swapping a storm-and-orchestra bed
for a still-evening-and-guitar one. Peak level barely moved, which is expected —
the dialogue is the loudest element in both and its text was unchanged.

Both stages ran at the same seed (4242), so the visual difference is attributable
to the conditioning path, not to noise.

## The cost surprise

Reference tokens ride through **every** sampling step. Feeding a 124-frame
864×480 clip back as a reference adds 14,985 reference tokens on top of the
14,985 being generated, so the packed sequence roughly doubles.

Measured: **26 s/it for the original run, 63 s/it for the reroll** — 2.4×.

That is the second headline finding, and it is not what you would hope for:

> Rerolling the audio costs more than generating the whole clip again.

The gap is worse than 2× because attention is quadratic in sequence length and
this H3 release is full-attention only (sparse attention is announced but
unreleased). Doubling the sequence more than doubles the attention term.

## When this is still worth using

The node is useful when *identity* matters more than time:

- you want the same performance and framing with different scoring, and a fresh
  generation at a new seed would change the picture entirely
- you are iterating on `overall_soundscape` / `non_diegetic_music` wording and
  need the visual held roughly constant to judge the audio
- the dialogue and lip-sync are right and you do not want to re-roll that lottery

It is **not** the right tool if you simply want a cheaper audio variation. For
that, generate at a fixed seed and accept the picture you get, or mux
externally.

## The bit-identical alternative

If you need the original frames exactly, do not re-sample at all: take the
`video_passthrough` output, decode it, and mux the new audio over it downstream.
That skips the DiT entirely, so it costs one VAE decode and one audio decode —
seconds, not minutes.

The trade-off is that the new audio was generated without seeing the video, so
lip-sync to existing dialogue will not be preserved. Use it for ambience and
score, not for speech.

## Limitations of this test

- **One clip, one seed.** A single measurement of visual preservation is not a
  distribution. Different content may hold together better or worse.
- **`video_cond_strength` swept only at 1.0.** Lower values should let the
  picture drift further; that curve is unmeasured.
- **No A/B on `reset_audio_latent`.** The test used `True` (audio starts from
  zeros). Seeding from the original audio latent would bias toward the original
  soundtrack, which may be desirable for subtle variations — unmeasured.
- **Lip-sync judged by eye**, not by any automatic metric.
- Only the `ref2va` checkpoint was tested. `fl2va` has no `minimax_refs` path in
  its conditioning node, so the reroll node requires `ref2va` weights.
