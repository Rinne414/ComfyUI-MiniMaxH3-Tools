# MiniMax-H3 ComfyUI — task briefs for an agent team

Each task below is self-contained: hand one brief to one agent. Every API
detail was verified against the installed source on 2026-08-04 (ComfyUI
v0.30.0, commit `14b05228`) — file paths and line numbers are real, not guessed.

## Shared context — every agent must read this first

**What H3 is.** A 33B single-stream omni-transformer that generates video **and
synchronized 32 kHz stereo audio in one sampling loop**. Not video-then-dub.
This is the structural fact that makes several tasks below impossible on
Wan/LTX but possible here.

**Core files (read before writing any code):**

| Path | What's in it |
|---|---|
| `comfy_extras/nodes_minimax_h3.py` | 286 LOC, all 4 existing H3 nodes. The reference for style and for the V3 `io.ComfyNode` API. |
| `comfy/ldm/minimax/model.py` | 552 LOC DiT. `self.blocks = nn.ModuleList([...])` at L448. `forward()` at L498, `_forward` wrapped via `comfy.patcher_extension.WrapperExecutor`. |
| `comfy/ldm/minimax/vae.py` | 563 LOC video VAE. **Already has tiling** — `tile_size=256`, `tile_overlap_min=64`, `tiling=True`, plus temporal chunking (`clip_length=17`, `tokens_chunk_size`, `token_overlap`) at L335-356. |
| `comfy/ldm/minimax/audio_vae.py` | Audio VAE, 32 kHz stereo, 40 Hz latent rate. |
| `comfy/nested_tensor.py` | `NestedTensor` class. Methods: `unbind()`, `to()`, `chunk()`, `size()`, `shape`, `ndim`, `device`, arithmetic ops. |
| `comfy/latent_formats.py:605` | `MiniMaxH3AV(MiniMaxH3Video)`, `latent_channels = 32` (max across both streams). |
| `comfy/text_encoders/minimax.py` | Tokenizer. Documents the exact prompt presentation per task type. |
| `comfy/supported_models.py:959` | `MiniMaxH3` config: `shift: 12.0`, `memory_usage_factor: 0.114`. |

**Latent shape — memorize this:**

```python
{"samples": NestedTensor((video, audio))}
#   video: [B, 24, latent_t, H//16, W//16]
#   audio: [B, 32, 2, audio_t]          # dim 2 == L/R channels
# latent_t  = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
# audio_t   = round(frames / 24 * 40)
```

**Hard model constraints** (from `nodes_minimax_h3.py` L25-46):

```python
CANVAS_MULTIPLE = 32; BASE_SHORT_EDGE = 768; MAX_PIXELS = 768 * 1344  # 1.032 MP
FPS = 24; AUDIO_LATENT_FPS = 40; REF_IMAGE_SHORT_EDGE = 2048
frame count must satisfy  n % 17 == 5   # 5, 22, 39, ... 124, 243, 362
trained range ~124-362 frames (5-15 s)
```

Weights are **CFG-distilled** → `BasicGuider`, never `CFGGuider`.
Default sampling: `res_multistep` + `simple`, 20 steps, `shift_video=12.0`,
`shift_audio=3.0`.

**A real defect to know about:** `adapt_canvas()` (L49, the 768-short-edge +
area-cap solver) is **only called for reference videos**. The generation path
(`MiniMaxH3ImageToVideo`, `EmptyMiniMaxH3LatentAV`) passes `width`/`height`
straight through with no validation beyond the 32-step UI constraint. Frame
alignment via `align_frame_count()` happens silently. Task 1 exists because of
this.

**Measured baseline** (RTX 3090 24 GB, 32 GB RAM, torch 2.8.0+cu128, xformers,
T2VA, 20-step estimates) — full data in `h3_calibration.csv`:

| MP | Resolution | Frames | Video tokens | s/it | Peak VRAM |
|---:|---|---:|---:|---:|---:|
| 0.4 | 864×480 | 124 | 14,985 | 25.0 | 21,617 MiB |
| 0.6 | 1056×608 | 124 | 23,199 | 41.5 | 23,368 MiB |
| 0.8 | 1216×672 | 124 | 29,526 | 58.7 | 23,296 MiB |
| 1.0 | 1376×768 | 124 | 38,184 | 90.2 | 23,334 MiB |
| 0.4 | 864×480 | 243 | 29,160 | 64.1 | 23,334 MiB |
| 0.6 | 1056×608 | 243 | 45,144 | 116.5 | 23,373 MiB |
| 0.8 | 1216×672 | 243 | 57,456 | 167.2 | 23,613 MiB |
| 1.0 | 1376×768 | 243 | 74,304 | ~321 (aborted at step 4/8) | 24,151 MiB |

Two facts that follow: peak VRAM **plateaus at ~23.3 GB** (dynamic offload
absorbs growth, so above 0.6 MP you buy resolution with time, not memory); and
once peak touches ~24 GB, s/it roughly doubles (offload thrashing).

**Known-negative results — do not re-litigate:**
- SageAttention: 25.03 vs 25.07 s/it against xformers. Noise. Also perturbs the
  sampling trajectory, breaking seed reproducibility.
- `nvfp4` weights on Ampere: log says `emulated ops: ... nvfp4`. Use
  `int8_convrot`.
- torch cu128 disables `comfy_kitchen`'s cuda+triton backends; only `eager`
  survives, so int8 dequantizes on the fly. Untested whether cu130 helps.

**Ecosystem — already taken, do NOT rebuild:**
`rockerBOO/boo-textgen` (LLM prompt rewriting, has `_minimax_h3_mode()` and
per-task instruction headers), `MaxKruse/ComfyUI-PromptEnhancer` (R2V system
prompt), `Anil-matcha/minimax-h3-comfyui` (API wrapper),
`kitsune123150/minimax-h3-hybrid-cond` (merged ref2va+fl2va conditioning),
INT4/INT8/GGUF/NVFP4 repacks on HF, several prompt libraries.

**Node authoring rules:**
- Use the V3 API: `from comfy_api.latest import ComfyExtension, io`, subclass
  `io.ComfyNode`, implement `define_schema()` + `execute()`. Copy the structure
  from `comfy_extras/nodes_minimax_h3.py` verbatim.
- `io.Schema` fields: `node_id`, `display_name`, `category`, `inputs`,
  `outputs`, `description`, `search_aliases`, `hidden`, `is_input_list`.
- Prefix every `node_id` to avoid collisions, e.g. `MiniMaxH3XCanvasSolver`.
- Ship as `custom_nodes/<pack>/` with `__init__.py` exposing
  `comfy_entrypoint()` returning a `ComfyExtension`.
- **Never mutate an input tensor or latent dict in place.** Clone, then modify.
- Match the terse comment style of the core H3 file: explain *why*, not *what*.

**Verification bar.** No task is done until it runs against a live ComfyUI. Use
the harness in this repo: start ComfyUI with `--disable-smart-memory`, then
`python h3_run.py <api_graph.json> [megapixels] [seconds]`. Report actual
numbers, not expectations. If you could not test something, say so explicitly.

---

# Tier A — pure logic, no GPU required

## Task 1 — `MiniMaxH3CanvasSolver`

**Problem.** The generation nodes accept any 32-multiple width/height. Ask for
1920×1080 and they take it, then produce garbage on an untrained canvas or OOM.
Frame counts snap silently, so "7 seconds" becomes some unstated frame count.
Users currently cannot know the cost of a job before queueing it.

**Build.** One node, pure arithmetic.

Inputs: `aspect_ratio` (combo: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`),
`megapixels` (float 0.1–1.1, default 0.4), `seconds` (float 1–15, default 5.0),
`vram_budget_gib` (float, default 24.0).

Outputs: `width` (INT), `height` (INT), `frames` (INT), `latent_t` (INT),
`audio_t` (INT), `video_tokens` (INT), `est_sit_seconds` (FLOAT),
`est_total_minutes` (FLOAT), `report` (STRING).

Rules — reuse the core helpers rather than reimplementing:
- Canvas: port `adapt_canvas()` logic (`nodes_minimax_h3.py` L49-60) —
  768 short edge, `MAX_PIXELS` cap, per-axis round to 32. Scale by
  `megapixels` relative to the 1.032 MP native area.
- Frames: `align_frame_count(round(seconds * 24))`, min 5.
- `latent_t = video_latent_t(frames)`; `audio_t = round(frames/24 * 40)`.
- `video_tokens = latent_t * (height // 32) * (width // 32)` — note **32**, not
  16: the VAE gives 16× spatial compression, then a `1×2×2` patchify doubles it
  (see `nodes_minimax_h3.py` docstring and the model card's H3-VisualVAE
  section).
- `est_sit_seconds`: fit a curve to the measured table above (tokens → s/it) and
  interpolate. A quadratic or power fit is appropriate — cost is superlinear
  because this release is full-attention only. **State the fit and its R² in a
  comment**, and make it a module-level constant that others can replace with
  their own card's data.
- `report`: human-readable multi-line string. Must warn when frames fall
  outside 124–362, when the requested megapixels exceeded the cap and got
  clamped, and when `est` peak is close to `vram_budget_gib`.

**Validate.** Feed the 8 rows of `h3_calibration.csv` in and confirm the solver
reproduces the measured width/height/frames/token counts exactly. Any mismatch
is a bug in the solver, not in the data. Include this as a pytest.

**Why it's worth doing.** No equivalent exists anywhere. ComfyUI's own launch
post says H3 "runs locally on a GPU like the RTX 3060" without a single number;
this turns the measured curve into something a user consults before waiting
30 minutes for a mistake.

---

## Task 2 — AV latent toolkit

**Problem.** H3 latents are `NestedTensor` pairs. **Every stock latent node
silently fails or corrupts them** — `LatentUpscale`, `LatentBlend`,
`LatentComposite`, `LatentFromBatch` all assume a plain tensor. There is
currently no way to touch one stream without the other.

**Build.** A small node set. This is infrastructure for Tasks 3 and 8, so get
the shapes right before anything else.

1. `MiniMaxH3SplitAV` — LATENT → (`video` LATENT, `audio` LATENT). Wrap each
   stream as an ordinary `{"samples": tensor}` so downstream tools work.
2. `MiniMaxH3JoinAV` — (video LATENT, audio LATENT) → LATENT. Validate that
   `latent_t` and `audio_t` are consistent with a single frame count; raise a
   clear `ValueError` naming both shapes if not.
3. `MiniMaxH3LatentInfo` — LATENT → STRING report: both shapes, implied frames,
   duration, token count, dtype, device. Debug aid; make it thorough.
4. `MiniMaxH3UpscaleVideoLatent` — video-stream-only spatial resize
   (bilinear/bicubic/nearest-exact), snapping to multiples of 2 in latent space
   (= 32 px). Audio passes through untouched. Needed by Task 8.
5. `MiniMaxH3TemporalTrim` — trim/extend along `latent_t` **and** the matching
   `audio_t` together so the pair stays coherent. Reject any result violating
   `latent_t = ((frames-5)//17)*5 + 2`.

Read `comfy/nested_tensor.py` first — `unbind()` gives you the tuple. Construct
with `comfy.nested_tensor.NestedTensor((video, audio))`.

**Validate.** Round-trip test: `Split` → `Join` must reproduce a
bit-identical latent (`torch.equal` on both streams). Then run a real 0.4 MP /
5 s generation through Split→Join before the sampler and confirm the output
video still decodes correctly.

---

## Task 3 — Audio-only reroll  ★ highest novelty

**Problem.** Because H3 samples video and audio jointly, changing the score or
the ambience today means regenerating the whole clip — 9+ minutes at 0.4 MP.
**Wan and LTX cannot do this at all** (their audio is bolted on afterwards by
MMAudio). H3 structurally can, and nobody has built it.

**Build.** Freeze the video stream, re-noise and re-sample only the audio
stream.

Approach — investigate before committing to one:
- Split the pair (Task 2), re-noise only the audio latent to a partial sigma,
  then sample with the video stream pinned at its final value each step.
- The clean hook is a `WrapperExecutor` on the DiT forward (see
  `comfy/ldm/minimax/model.py:498-501` — `comfy.patcher_extension` is already
  wired in) or a custom guider that overwrites the video half of the pack after
  every step. Prefer whichever needs no core patch.

Critical details:
- `shift_audio` is passed **only** via `transformer_options`
  (`nodes_minimax_h3.py` L320-322) and the DiT derives the audio schedule by
  inverting the video one. Read `model.py` to see exactly how before you touch
  the sigma schedule, or the two streams will desynchronize.
- Lip-sync is the acid test. If the clip has `<d>...</d>` dialogue, a correct
  implementation keeps mouth movement matched to the new speech. Test with
  dialogue, not just ambience.

**Deliver.** `MiniMaxH3AudioReroll` node + a demo workflow + at least three
sample outputs from one fixed video with different `overall_soundscape` /
`non_diegetic_music` text. Report measured time versus a full regeneration.

**Honesty requirement.** This may not work — the joint latent might not permit
clean stream isolation. A well-documented negative result ("tried X and Y,
here's the artifact, here's why") is a legitimate deliverable and more useful
than a half-working node. Do not ship something that visibly degrades the video.

---

## Task 4 — Reference encoding cache

**Problem.** `MiniMaxH3ReferenceToVideo.execute()` re-runs `vae.encode()` on
every reference image, video and audio clip on **every** execution
(`nodes_minimax_h3.py` L218-274). The real ref2va workflow is "fix the
references, iterate on the prompt", so this is pure waste — and ref tokens ride
through every sampling step, making ref2va the expensive mode already.

**Build.** A caching wrapper around the ref-encoding path. Cache key must
include the image content hash, target size, and `ref_image_size` mode — a
resize invalidates the entry. Bound the cache (LRU, configurable entry count
and/or GiB) and expose a manual clear. Never hold GPU memory: keep cached
latents on CPU and move them at use time.

Cleanest shape is probably a node that pre-encodes refs and outputs an opaque
cached bundle, rather than monkey-patching the core node. Decide and justify.

**Validate.** Measure cold versus warm wall-clock for a ref2va run with 4
images at `ref_image_size: max`, and assert the produced conditioning is
identical (`torch.equal`) either way. Identical output is non-negotiable — a
cache that changes results is a bug.

---

## Task 5 — Prompt linter (deterministic, not an LLM)

**Problem.** H3's prompt format has strict rules that fail **silently** — a
malformed prompt does not error, it just produces worse output. LLM rewriting
is already well covered by `boo-textgen` and `ComfyUI-PromptEnhancer`; what
neither does is *verify*.

**Build.** `MiniMaxH3PromptLint`: STRING in → (`prompt` STRING passthrough,
`warnings` STRING, `is_valid` BOOLEAN).

Checks, all derived from MiniMax's own
[base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [ref guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
— read both in full first:

- All three field names present, correctly spelled, in order:
  `integrated_multimodal_description:`, `overall_soundscape:`,
  `non_diegetic_music:`.
- `[Shot 1]` carries **no** timestamp; `[Shot N]` for N≥2 must have a strictly
  increasing `At MM:SS.mmm,` that falls inside the clip duration (accept
  duration as an optional input).
- `<d>[Lang] ...</d>` well-formed and closed; `Lang` in the 11 stable
  languages (Arabic, Chinese, English, French, German, Italian, Japanese,
  Korean, Portuguese, Russian, Spanish) — warn, don't fail, on others.
- Speaker IDs `(S1)`, `(S2)`, `(S1,S2)` used consistently; flag an ID that
  appears only once, and dialogue with no speaker attribution.
- Diegetic/non-diegetic confusion: radio, phone, TV, live performance
  mentioned inside `non_diegetic_music` is an error per the guide — those
  belong in the multimodal description.
- `overall_soundscape` restating dialogue or lyrics (the guide forbids it).
- Ref tags `<Picture i>` / `<Video k>` / `<Audio j>`: accept optional counts of
  connected references and flag any tag that exceeds them, plus gaps in
  numbering.
- Voiceover: the guide requires the exact phrase `says in an off-screen
  voiceover` and a following statement that the lips stay closed. Check both.

**Validate.** Build a fixture set: the 4 worked examples from the base guide
plus the 3 from the ref guide must all pass clean. Then a corpus of deliberately
broken variants, one rule violated each, must each produce exactly the expected
warning. pytest, table-driven.

---

# Tier B — needs a GPU, no training

## Task 6 — Blockswap for H3  ★ highest impact, highest risk

**Problem.** The DiT is 19.5 GB (pruned int8) and the text encoder 25.3 GB.
12 GB and 16 GB cards cannot hold the DiT. Wan has
`ComfyUI-wanBlockswap`; H3 has nothing.

**Build.** Per-block CPU↔GPU streaming during the forward pass. The container
is confirmed: `self.blocks = nn.ModuleList([DiTBlock(...) for _ in
range(num_layers)])` at `comfy/ldm/minimax/model.py:448`.

- Hook via `comfy.patcher_extension.WrapperExecutor` — already used at
  `model.py:498-501`, so no core patch should be needed.
- Expose `blocks_to_swap` (INT) and `use_non_blocking` (BOOLEAN). Overlap the
  H2D copy of block *n+1* with compute of block *n* using a second CUDA stream;
  the log shows ComfyUI already does `Using async weight offloading with 2
  streams`, so investigate whether that machinery can be reused instead of
  duplicated.
- Note `memory_usage_factor = 0.114` in `supported_models.py:959` — ComfyUI's
  own estimator. Check whether it needs adjusting when blocks are swapped.
- Roughly 13B of the 33B parameters live in AdaLN branches, and the model card
  states AdaLN modulation outputs can be **precomputed and cached** for
  inference-only use. Look at `AdalnProj` (`model.py:195`) and
  `adaln_t_table` (L439) — there may be a large win here independent of
  swapping.

**Validate.** Same seed with `blocks_to_swap=0` and `blocks_to_swap=N` must
produce a near-identical clip (report PSNR against the unswapped reference;
small numeric drift is acceptable, visible change is not). Report the VRAM/speed
tradeoff curve across several N. If you have access to a 12 GB or 16 GB card,
the headline result is the lowest N that fits.

**Risk.** Hardest task here. Read all 552 lines of `model.py` before writing
anything.

---

## Task 7 — Expose the VAE's existing tiling controls

**Correction to an earlier assumption:** the H3 video VAE **already implements
tiling** — `tile_size=256`, `tile_overlap_min=64`, `tiling=True`, plus temporal
chunking via `clip_length=17` / `tokens_chunk_size` / `token_overlap`
(`comfy/ldm/minimax/vae.py:335-356`). Nothing needs writing from scratch. The
problem is that **none of it is reachable from a workflow** — the values are
constructor defaults.

**Build.** A node that overrides them on a loaded VAE (clone, patch, return —
do not mutate the shared VAE object). Expose `tile_size`, `tile_overlap_min`,
`tiling`, and the temporal chunk size.

**Then measure**, because this is the actual deliverable: peak VRAM occurs
during decode of long high-resolution clips (23,613 MiB at 0.8 MP / 243
frames). Sweep the tiling parameters and report the VRAM/time/seam-quality
tradeoff. Check specifically for visible seams at aggressive settings — the
temporal path uses causal convolutions, so temporal seams are the thing to
watch.

**Possible outcome:** the defaults may already be near-optimal, making this a
"documented, no change needed" result. That is fine and worth publishing.

---

## Task 8 — Two-stage latent refine workflow

**Problem.** Cost is superlinear in tokens: +55 % tokens costs +66 % time at
5 s and +82 % at 10 s. Iterating on a prompt at 0.8 MP is a waste.

**Build.** A workflow (not a node — compose from Tasks 1, 2, and core nodes):
generate at 0.4 MP → `MiniMaxH3UpscaleVideoLatent` → partial re-noise at the
same seed → resample at 0.8 MP with reduced denoise.

Open question the agent must answer with measurements, not assumptions: does
the audio stream need re-sampling at all in stage 2? Resolution does not change
`audio_t`, so it may be reusable verbatim — which would be a significant free
saving. Test both and report.

Also determine the denoise value that preserves stage-1 composition and timing
while actually adding detail. Sweep it; publish the curve.

**Deliver.** Working workflow JSON, measured total time versus a single-pass
0.8 MP run, and a side-by-side comparison. If two-stage is slower or worse than
single-pass, **say so** — that is a valid and useful finding.

---

## Task 9 — Sigma shift sweep

**Problem.** `shift_video=12.0` / `shift_audio=3.0` are the shipped defaults
(`supported_models.py:961` sets `shift: 12.0`). Nobody has published what
happens when you change them, and the audio shift is especially interesting
because the DiT derives the audio schedule from the video one.

**Build.** No new node needed — `MiniMaxH3SigmaShift` already exposes both.
This is a data task: sweep `shift_video` ∈ {6, 9, 12, 15, 20} × `shift_audio` ∈
{1.5, 3, 6} at a fixed seed and prompt, 0.4 MP / 5 s (~9 min each — budget the
GPU time, ~2.5 h for the full 15-cell grid).

Prompt must contain `<d>...</d>` dialogue so lip-sync is measurable. Report:
motion coherence, audio/video sync drift, artifacts. Extend
`h3_calibration.csv` or add a sibling CSV, and reuse `h3_bench.py`'s structure.

**Deliver.** A table plus sample clips. Honest subjective assessment is
acceptable here — label it as subjective.

---

## Task 10 — Steps versus quality

**Problem.** 20 steps is MiniMax's default. Is 12 enough? Nobody has published
a comparison, and this is the single most common question about any new model.

**Build.** Fixed seed and prompt, steps ∈ {8, 12, 16, 20, 28, 40} at 0.4 MP /
5 s. s/it is step-count independent (verified), so cost is exactly
`steps × 25.0 s` + ~2 min overhead — the full sweep is roughly 1 h of GPU time.

Judge video **and** audio separately; they may converge at different rates,
which would itself be a notable finding. Include the dialogue lip-sync case.

**Deliver.** Comparison grid, a recommended step count with reasoning, and the
clips so readers can judge for themselves. Label subjective calls as subjective.

---

# Tier C — flagged, not recommended yet

## Task 11 — GGUF quantization support

`ComfyUI-GGUF` is installed but does not recognize H3. This is the other route
to 12 GB cards. Technically feasible; needs ~50 GB of disk headroom and hours of
conversion time. Third-party GGUF repacks already exist on HF
(`Abiray/MiniMax-H3-GGUF`, `realrebelai/MiniMax-H3_GGUFs`,
`liewge/MinimaxH3-GGUF-Quant`) — **check whether they already load** before
building anything. Verification-first task.

## Task 12 — Step-distillation LoRA — NOT feasible on one consumer GPU

Wan has lightx2v 4-step, LTX has a distilled LoRA, H3 has none. This would be
the single biggest speed win. But it needs a training dataset and days of
multi-GPU time. **Do not assign this to an agent on a 24 GB card.** Listed only
so nobody rediscovers it as an easy idea.

## Task 13 — Long-form chaining past 362 frames

H3's fl2va mode makes "last frame of clip N → first frame of clip N+1" a natural
fit. But colour/identity drift across joins and audio seam continuity are open
research problems, and 362 frames is the edge of the trained range. Genuine
research, not an implementation task.

---

# Suggested order

1. **Task 1** — no GPU, immediately useful, validates against existing data.
2. **Task 2** — infrastructure that Tasks 3 and 8 depend on.
3. **Task 5** — no GPU, parallelizable with 1 and 2.
4. **Task 3** — the one genuinely H3-only capability. Highest novelty.
5. **Task 4**, **Task 7** — contained, measurable.
6. **Task 9**, **Task 10** — GPU-time-bound data collection; can run overnight.
7. **Task 8** — needs Task 2 landed first.
8. **Task 6** — highest impact, attempt only once the rest is stable.

Tasks 1, 2, 5 are independent and can run in parallel from the start.
