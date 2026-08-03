# MiniMax-H3 on a single consumer GPU — measured calibration

Every number here was measured on one machine. Nothing is extrapolated from
vendor claims, and the negative results are reported alongside the positive
ones because they are the expensive part to discover.

**Test rig:** RTX 3090 (24 GB, Ampere) · 32 GB system RAM · Windows 11 ·
ComfyUI v0.30.0 · torch 2.8.0+cu128 · xformers 0.0.32.post2

**Model:** [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)
`fl2va_pruned_int8_convrot` (19.5 GB) + `qwen3vl_32b_minimax_h3_int8_convrot`
(25.3 GB) + video/audio VAEs. 33B omni-transformer producing video **and
synchronized 32 kHz stereo audio** in a single pass.

---

## Why this exists

ComfyUI's own launch post says:

> On top of that, the weights ship with an accurate and efficient int8 convrot
> quantization, and custom kernels reduce the peak VRAM use during inference.
> Combining this with our dynamic VRAM offloading enables a next-generation 2K
> video model to run locally on a GPU like the RTX 3060.

That is true and it is the right headline — but it carries no numbers. At what
resolution? How many steps? How long? The tables below are an attempt to answer
exactly that for one card, and to make it cheap for other people to answer it
for theirs.

The most useful thing measured here is the *shape* of that dynamic offloading:
peak VRAM stops climbing at ~23.3 GB regardless of resolution, and once peak
touches ~24 GB the seconds-per-step roughly doubles. "Fits in VRAM" and
"finishes in reasonable time" are two different questions, and above 0.6 MP
only the second one still matters.

---

## TL;DR for 24 GB owners

- **1.0 MP (1376×768) at 5 s fits.** Peak 23,334 MiB of 24,576. It is not
  comfortable, but it does not OOM.
- **VRAM saturates at ~23.3 GB from 0.6 MP upward and stops climbing.**
  ComfyUI's dynamic offload absorbs the growth, so raising resolution past
  0.6 MP costs *time*, not memory. Sizing by "will it fit" is the wrong
  question above 0.6 MP; size by "how long will I wait".
- **`--disable-smart-memory` is mandatory**, not advisory. Text encoder
  (25.3 GB) + DiT (20.0 GB) = 45 GB against 32 GB of RAM.
- **SageAttention does not help.** Measured 25.03 vs 25.07 s/it against
  xformers — a 0.2 % difference, i.e. noise. It also changes the sampling
  trajectory, so it breaks seed reproducibility. Stay on xformers.
- **`nvfp4` weights are a trap on Ampere.** The official ComfyUI template
  ships `qwen3vl_32b_minimax_h3_nvfp4_awq`; on a 3090 the startup log reports
  `emulated ops: ... nvfp4`. Use `int8_convrot`, which shows up under
  `Native ops`.

---

<!-- CALIBRATION_TABLES -->

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

---

## Reading the tables

**Cost is superlinear in tokens.** H3's initial open-source release is
full-attention only (sparse attention is announced but unreleased), so the
attention term grows with the square of sequence length while the FFN term
grows linearly. The measured `ms per 1k tokens per step` column rises with
token count — that rise *is* the quadratic term becoming visible.

**Time tokens cost more than space tokens.** At near-identical token counts,
a longer clip is slower than a higher-resolution one. Compare 0.8 MP / 5 s
against 0.4 MP / 10 s in the table: ~1 % fewer tokens, but measurably slower.
The temporal axis carries MM-RoPE's `t` dimension plus the audio stream's
separately-shifted schedule.

**Practical consequence:** if you need both a long clip and high resolution,
generate at low resolution first to lock composition, timing, and audio, then
re-run the same seed at higher resolution. Do not pay the superlinear cost
while you are still iterating on the prompt.

---

## Reproducing

```bash
# 1. Start ComfyUI with the memory flag
python main.py --port 8188 --disable-smart-memory

# 2. Run the sweep (8 steps per point: s/it and peak VRAM are step-count
#    independent, so this measures the same thing at ~1/3 the cost)
python h3_bench.py h3_advanced_api.json <comfyui-log-path> 8

# 3. Render the tables
python h3_report.py h3_calibration.csv
```

`h3_bench.py` writes `h3_calibration.csv` incrementally, so an OOM or a crash
mid-sweep still leaves every completed point on disk. It skips any point that
is strictly harder than one that already failed.

**Adding your GPU:** run the two commands above and open a PR with your
`h3_calibration.csv` plus your rig line. The interesting gaps are 12 GB and
16 GB cards (where the INT4 builds matter), and 5090-class hardware (where
`nvfp4` should stop being emulated and the cu130 kernels below become live).

### What this data does NOT cover

- **1.0 MP / 10 s (74.3k tokens) is incomplete.** It did *not* OOM — sampling
  was progressing at ~321 s/it with peak VRAM at 24,151 of 24,576 MiB when the
  run was interrupted at step 4/8 because another GPU workload started. The
  partial reading is in the CSV marked `aborted`, and it is worth reporting
  for one reason: at 29 % more tokens than 0.8 MP / 10 s, s/it roughly doubled.
  That break in the curve is offload thrashing, not compute — past ~24 GB of
  peak the card is shuttling weights rather than working. Treat 0.8 MP / 10 s
  as the practical ceiling on 24 GB and 1.0 MP / 10 s as "technically runs,
  don't bother".
- **Ref2VA.** Every completed number above is T2VA on the `fl2va` checkpoint.
  Reference conditioning adds image/video/audio tokens that ride through
  *every* sampling step, so `ref2va` will be strictly more expensive —
  ComfyUI's own tooltip warns that `ref_image_size: max` (2048px short edge)
  "can be several times slower". Unmeasured here.
- **Step counts other than 20.** s/it and peak VRAM are step-count
  independent, so the `20-step est.` column extrapolates cleanly; anything
  about quality-versus-steps is not measured.
- **The 15 s ceiling.** 362 frames is the top of MiniMax's trained range. At
  0.4 MP that extrapolates to roughly 44k tokens, i.e. between the measured
  0.6 MP / 10 s and 0.8 MP / 10 s rows. Not run.
- **Quality.** This is a cost table, not an evaluation. Whether 0.6 MP looks
  meaningfully better than 0.4 MP is a judgement call nobody else can make
  for your material.

---

## Known-unverified optimization

ComfyUI ships fused int8 CUDA kernels via `comfy_kitchen`. On torch cu128 they
self-disable:

```
WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.
Found comfy_kitchen backend cuda:   {'available': True, 'disabled': True, ...}
Found comfy_kitchen backend triton: {'available': True, 'disabled': True, ...}
Found comfy_kitchen backend eager:  {'available': True, 'disabled': False, ...}
```

Only `eager` survives, so every int8 matmul dequantizes on the fly. Since
SageAttention measurably did *not* move the needle, this dequant path — not
attention — is the remaining suspect for the bottleneck.

**This is untested here.** Upgrading to a cu130 torch means rebuilding
sageattention and xformers, which risks a working environment. If you have a
cu130 setup, measuring `1.0 MP / 5 s` before and after would settle it, and
the result is worth a PR either way.

---

## Hard constraints worth knowing before you queue a job

Derived from `comfy_extras/nodes_minimax_h3.py` and MiniMax's own prompt guide.

| Constraint | Value |
|---|---|
| Frame count | Snaps up to `n % 17 == 5` → 5, 22, 39, ... 124 (≈5 s), 243 (≈10 s), 362 (≈15 s) |
| Trained range | ~124–362 frames; outside is untested by MiniMax |
| Canvas | Multiples of 32 per axis; short edge 768; area cap 768×1344 ≈ 1.03 MP |
| Frame rate | 24 fps fixed |
| Audio | 32 kHz stereo |
| Guidance | Weights are **CFG-distilled** → use `BasicGuider`, never `CFGGuider` |
| Sampler | `res_multistep` + `simple`, 20 steps (MiniMax's own default) |
| Sigma shift | `shift_video` 12.0 / `shift_audio` 3.0; both go to the DiT, which derives the audio schedule from the video one |

### Prompt format

H3 was trained on the output of **H3-Context-IR**, a preprocessing system
MiniMax explicitly calls "critical to the quality of the final output" — and
did not open-source. It emits three labelled fields:

```
integrated_multimodal_description: [Shot 1] <style, framing, subject, action, dialogue>
overall_soundscape: <ambient and physical action sound, 1-4 sentences>
non_diegetic_music: <score the characters cannot hear, or N/A>
```

Keep the field names. Dialogue goes in `<d>[English] ...</d>` and drives
lip-sync (11 languages supported). Later shots need a strictly increasing cut
time: `[Shot 2] At 00:03.500, the camera cuts to ...`.

MiniMax's full rule set — camera-motion vocabulary, `(S1)`/`(S2)` speaker IDs,
`<scenetrans>`, `<cutoff>`, per-task instruction headers — is in
[`VIDEO_PROMPT_WRITING_GUIDE_base_en.md`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

If you want that rewriting automated by a local LLM, use
[`rockerBOO/boo-textgen`](https://github.com/rockerBOO/boo-textgen) or
[`MaxKruse/ComfyUI-PromptEnhancer`](https://github.com/MaxKruse/ComfyUI-PromptEnhancer);
both already implement it well and this repo deliberately does not duplicate them.

---

## What's in here

| File | Purpose |
|---|---|
| `h3_calibration.csv` | Raw measurements, one row per point |
| `h3_bench.py` | The sweep harness (resumable, OOM-aware) |
| `h3_report.py` | CSV → the Markdown tables above |
| `workflows/MiniMax-H3_AV_Advanced.json` | ComfyUI workflow: three-field prompt composer, correct int8 encoder, exposed sigma shift, separate audio track |
| `h3_advanced_api.json` | Same graph in API format, for scripted runs |
| `h3_run.py` | Single-shot runner: `h3_run.py <graph> [megapixels] [seconds]` |

The workflow is the graph that produced every number in the tables.

## License

MIT for the scripts and workflow. The MiniMax-H3 weights are under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) — check it before commercial use.
