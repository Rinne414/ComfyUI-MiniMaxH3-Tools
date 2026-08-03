# MiniMax-H3 Audio+Video — Advanced Workflow (measured on RTX 3090)

MiniMax-H3 generates video **and synchronized 32 kHz stereo audio in a single
pass** — dialogue with lip-sync, ambient sound, and score all come out of the
same sampling loop. This workflow is the graph I used to measure the
calibration numbers below, so every figure here came from this exact file.

## What this fixes versus the official template

| | Official template | This one |
|---|---|---|
| Text encoder | `nvfp4_awq` | `int8_convrot` |
| Prompt | one free-text box | H3-Context-IR three-field composer |
| Sigma shift | not exposed | `MiniMaxH3SigmaShift` wired up |
| Audio | muxed into the mp4 only | also saved as a separate FLAC |
| Keyframes | separate template | first/last frame pre-wired, muted |

**The text encoder swap is the important one.** The official template ships
the `nvfp4` encoder, which only has native kernels on Blackwell (RTX 50
series). On a 3090 the ComfyUI log reports:

```
Native ops: int8_tensorwise, convrot_w4a4 , emulated ops: float8_e4m3fn, ..., nvfp4
```

`nvfp4` runs emulated. Use `int8_convrot` on any 30/40-series card.

## Required models

Put these in the usual folders — all from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors   19.5 GB
models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors       25.3 GB
models/vae/minimax_h3_video_vae_fp16.safetensors                            4.9 GB
models/vae/minimax_h3_audio_vae_fp32.safetensors                            0.6 GB
```

For reference-to-video (multi-image / video / voice-clone conditioning), swap
the diffusion model for `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
and use the `MiniMax H3 Reference to Video` node instead.

## Launch flag you cannot skip

```
python main.py --disable-smart-memory
```

Text encoder 25.3 GB + DiT 20.0 GB = 45 GB against 32 GB of system RAM.
Without this flag ComfyUI tries to keep both cached and thrashes.

<!-- CALIBRATION_PLACEHOLDER -->

## Measured calibration — 24576 MiB VRAM

| MP | Resolution | Frames | Video tokens | s/it | Peak VRAM | Headroom | 20-step est. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.4 | 864x480 | 124 | 14,985 | 25.0 | 21,617 MiB | 2,959 MiB | 9.2 min |
| 0.6 | 1056x608 | 124 | 23,199 | 41.5 | 23,368 MiB | 1,208 MiB | 15.3 min |
| 0.8 | 1216x672 | 124 | 29,526 | 58.7 | 23,296 MiB | 1,280 MiB | 21.2 min |
| 1.0 | 1376x768 | 124 | 38,184 | 90.2 | 23,334 MiB | 1,242 MiB | 33.3 min |
| 0.4 | 864x480 | 243 | 29,160 | 64.1 | 23,334 MiB | 1,242 MiB | 23.5 min |
| 0.6 | 1056x608 | 243 | 45,144 | 116.5 | 23,373 MiB | 1,203 MiB | 40.9 min |
| 0.8 | 1216x672 | 243 | 57,456 | 167.2 | 23,613 MiB | 963 MiB | 59.0 min |
| 1.0 | — | 10s | — | — | — | — | **aborted** |

## Cost scaling

| From | To | Tokens | s/it |
|---|---|---:|---:|
| 0.4 MP / 5s | 0.6 MP / 5s | +55% | +66% |
| 0.6 MP / 5s | 0.8 MP / 5s | +27% | +41% |
| 0.8 MP / 5s | 1.0 MP / 5s | +29% | +54% |
| 0.4 MP / 10s | 0.6 MP / 10s | +55% | +82% |
| 0.6 MP / 10s | 0.8 MP / 10s | +27% | +44% |
| 0.4 MP / 5s | 0.4 MP / 10s | +95% | +156% |
| 0.6 MP / 5s | 0.6 MP / 10s | +95% | +181% |
| 0.8 MP / 5s | 0.8 MP / 10s | +95% | +185% |

## Per-token cost

| MP | Duration | ms per 1k tokens per step |
|---:|---:|---:|
| 0.4 | 5s | 1669.7 |
| 0.6 | 5s | 1788.0 |
| 0.4 | 10s | 2199.2 |
| 0.8 | 5s | 1987.1 |
| 1.0 | 5s | 2361.2 |
| 0.6 | 10s | 2580.6 |
| 0.8 | 10s | 2910.7 |

## Measured calibration — 24576 MiB VRAM

| MP | Resolution | Frames | Video tokens | s/it | Peak VRAM | Headroom | 20-step est. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.4 | 864x480 | 124 | 14,985 | 25.0 | 21,617 MiB | 2,959 MiB | 9.2 min |
| 0.6 | 1056x608 | 124 | 23,199 | 41.5 | 23,368 MiB | 1,208 MiB | 15.3 min |
| 0.8 | 1216x672 | 124 | 29,526 | 58.7 | 23,296 MiB | 1,280 MiB | 21.2 min |
| 1.0 | 1376x768 | 124 | 38,184 | 90.2 | 23,334 MiB | 1,242 MiB | 33.3 min |
| 0.4 | 864x480 | 243 | 29,160 | 64.1 | 23,334 MiB | 1,242 MiB | 23.5 min |
| 0.6 | 1056x608 | 243 | 45,144 | 116.5 | 23,373 MiB | 1,203 MiB | 40.9 min |
| 0.8 | 1216x672 | 243 | 57,456 | 167.2 | 23,613 MiB | 963 MiB | 59.0 min |
| 1.0 | — | 10s | — | — | — | — | **aborted** |

## Cost scaling

| From | To | Tokens | s/it |
|---|---|---:|---:|
| 0.4 MP / 5s | 0.6 MP / 5s | +55% | +66% |
| 0.6 MP / 5s | 0.8 MP / 5s | +27% | +41% |
| 0.8 MP / 5s | 1.0 MP / 5s | +29% | +54% |
| 0.4 MP / 10s | 0.6 MP / 10s | +55% | +82% |
| 0.6 MP / 10s | 0.8 MP / 10s | +27% | +44% |
| 0.4 MP / 5s | 0.4 MP / 10s | +95% | +156% |
| 0.6 MP / 5s | 0.6 MP / 10s | +95% | +181% |
| 0.8 MP / 5s | 0.8 MP / 10s | +95% | +185% |

## Per-token cost

| MP | Duration | ms per 1k tokens per step |
|---:|---:|---:|
| 0.4 | 5s | 1669.7 |
| 0.6 | 5s | 1788.0 |
| 0.4 | 10s | 2199.2 |
| 0.8 | 5s | 1987.1 |
| 1.0 | 5s | 2361.2 |
| 0.6 | 10s | 2580.6 |
| 0.8 | 10s | 2910.7 |

## Measured calibration — 24576 MiB VRAM

| MP | Resolution | Frames | Video tokens | s/it | Peak VRAM | Headroom | 20-step est. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.4 | 864x480 | 124 | 14,985 | 25.0 | 21,617 MiB | 2,959 MiB | 9.2 min |
| 0.6 | 1056x608 | 124 | 23,199 | 41.5 | 23,368 MiB | 1,208 MiB | 15.3 min |
| 0.8 | 1216x672 | 124 | 29,526 | 58.7 | 23,296 MiB | 1,280 MiB | 21.2 min |
| 1.0 | 1376x768 | 124 | 38,184 | 90.2 | 23,334 MiB | 1,242 MiB | 33.3 min |
| 0.4 | 864x480 | 243 | 29,160 | 64.1 | 23,334 MiB | 1,242 MiB | 23.5 min |
| 0.6 | 1056x608 | 243 | 45,144 | 116.5 | 23,373 MiB | 1,203 MiB | 40.9 min |
| 0.8 | 1216x672 | 243 | 57,456 | 167.2 | 23,613 MiB | 963 MiB | 59.0 min |

## Cost scaling

| From | To | Tokens | s/it |
|---|---|---:|---:|
| 0.4 MP / 5s | 0.6 MP / 5s | +55% | +66% |
| 0.6 MP / 5s | 0.8 MP / 5s | +27% | +41% |
| 0.8 MP / 5s | 1.0 MP / 5s | +29% | +54% |
| 0.4 MP / 10s | 0.6 MP / 10s | +55% | +82% |
| 0.6 MP / 10s | 0.8 MP / 10s | +27% | +44% |
| 0.4 MP / 5s | 0.4 MP / 10s | +95% | +156% |
| 0.6 MP / 5s | 0.6 MP / 10s | +95% | +181% |
| 0.8 MP / 5s | 0.8 MP / 10s | +95% | +185% |

## Per-token cost

| MP | Duration | ms per 1k tokens per step |
|---:|---:|---:|
| 0.4 | 5s | 1669.7 |
| 0.6 | 5s | 1788.0 |
| 0.4 | 10s | 2199.2 |
| 0.8 | 5s | 1987.1 |
| 1.0 | 5s | 2361.2 |
| 0.6 | 10s | 2580.6 |
| 0.8 | 10s | 2910.7 |

## The prompt format

H3 was trained on output from **H3-Context-IR**, a preprocessing system MiniMax
calls "critical to the quality of the final output" — and did not open-source.
It emits three labelled fields, which the three text boxes in this workflow
assemble:

```
integrated_multimodal_description: [Shot 1] <style, framing, subject, action, dialogue>
overall_soundscape: <ambient + physical action sound, 1-4 sentences>
non_diegetic_music: <score the characters cannot hear, or N/A>
```

Keep the field names — they are what the model saw in training.

**Dialogue** goes inside `<d>[English] Secure the boats.</d>` and drives
lip-sync. 11 languages: Arabic, Chinese, English, French, German, Italian,
Japanese, Korean, Portuguese, Russian, Spanish.

**Multiple shots** need a strictly increasing cut time:
`[Shot 2] At 00:03.500, the camera cuts to ...`

MiniMax's full rule set (camera-motion vocabulary, `(S1)`/`(S2)` speaker IDs,
`<scenetrans>` for dialogue crossing a cut, per-task instruction headers) is in
their [prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
If you want an LLM to do that rewriting for you, `rockerBOO/boo-textgen` and
`MaxKruse/ComfyUI-PromptEnhancer` both already implement it.

## Hard constraints

| Constraint | Value |
|---|---|
| Frame count | snaps up to `n % 17 == 5` → 124 (≈5 s), 243 (≈10 s), 362 (≈15 s) |
| Trained range | ~124–362 frames; beyond that is untested by MiniMax |
| Canvas | multiples of 32; short edge 768; area cap ≈1.03 MP |
| Frame rate | 24 fps fixed |
| Guidance | CFG-distilled weights → `BasicGuider`, **never** `CFGGuider` |
| Sampler | `res_multistep` + `simple`, 20 steps |

The duration box takes **seconds** and a Math Expression node snaps it onto the
frame grid, so you never have to remember the 17k+5 sequence.

## Two things that do NOT help

**SageAttention.** Measured 25.03 s/it versus xformers' 25.07 s/it at
0.4 MP — a 0.2 % difference, i.e. noise. It also perturbs the sampling
trajectory, so the same seed produces a different clip. Stay on xformers.

**Chasing VRAM headroom above 0.6 MP.** Peak VRAM plateaus at ~23.3 GB and
stops climbing, because ComfyUI's dynamic offload absorbs the growth. Above
0.6 MP you are buying resolution with *time*, not memory.

## Workflow tips

- Both `Load Image` nodes ship muted (mode 2) for text-to-video. Ctrl+M on
  `first_frame` for image-to-video; unmute both to interpolate between them.
- First frame is **stretched** to the canvas; last frame is **cover-cropped**.
  Match your first frame's aspect ratio to the output or it will distort.
- Iterate at 0.4 MP to lock composition, timing, and audio, then re-run the
  same seed at higher resolution. The cost curve is superlinear — do not pay it
  while you are still editing the prompt.

Full measured data, the benchmark harness, and the API-format graph:
<!-- REPO_URL_PLACEHOLDER -->
