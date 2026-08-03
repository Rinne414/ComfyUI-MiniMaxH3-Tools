# ComfyUI-MiniMax-H3-Toolkit

Nine nodes filling gaps in ComfyUI's core MiniMax-H3 support. Built against
ComfyUI v0.30.0 and verified on an RTX 3090.

MiniMax-H3 is a 33B omni-transformer that generates video **and synchronized
32 kHz stereo audio in a single sampling loop**. Core ComfyUI ships four nodes
for it; these nine cover things those four leave out.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> comfyui-minimax-h3-toolkit
```

No extra dependencies. Restart ComfyUI and search for `MiniMax H3` in the node
menu.

## Nodes

### Canvas & Cost Solver

`MiniMax H3 Canvas & Cost Solver` — resolve a legal canvas and frame count, and
estimate the cost, *before* queueing.

The core generation nodes accept any 32-multiple width and height. Nothing
validates against H3's trained canvas, and frame counts snap silently onto the
`17k+5` grid, so it is easy to queue an untrained geometry and discover it 30
minutes later.

Outputs `width`, `height`, `frames`, `latent_t`, `audio_t`, `video_tokens`,
`est_sit_seconds`, `est_total_minutes`, and a `report` string that warns about:

- an area over H3's 768×1344 cap
- a duration that got snapped to the frame grid
- frame counts outside the trained 124–362 range
- token counts past the calibrated measurement range (the estimate is then an
  extrapolation and reads low)
- proximity to the offload-thrashing region on a 24 GB card

The cost model is `s/it = 2.428e-05 × tokens^1.4343` (R²=0.9942 over 7 RTX 3090
measurements). The exponent sitting between 1 and 2 is what you would expect:
this H3 release is full-attention only, so the attention term grows
quadratically in sequence length while the FFN term grows linearly.

Replace `CALIBRATION` in `canvas_solver.py` with your own card's measurements to
retune it.

**A real quirk this surfaced:** `adapt_canvas()` in the core node — the function
that enforces the 768-short-edge rule and the area cap — is only called for
*reference videos*. The generation path never calls it. So 1.0 MP at 16:9 gives
1376×768 = 1,056,768 px, over the 1,032,192 cap, and H3 renders it anyway. The
solver reports that honestly rather than silently substituting a legal canvas;
set `clamp_to_trained_cap` if you want it to shrink instead.

### AV latent toolkit

H3 latents are `NestedTensor` pairs:

```
video: [B, 24, latent_t, H//16, W//16]
audio: [B, 32, 2, audio_t]          # dim 2 is L/R
```

**Every stock latent node** — `LatentUpscale`, `LatentBlend`,
`LatentComposite`, `LatentFromBatch` — assumes a plain tensor and will raise or
silently corrupt the pair. These five nodes let you reach one stream:

| Node | What it does |
|---|---|
| `Split AV Latent` | → separate video / audio latents, tagged so Join can catch a swap |
| `Join AV Latent` | ← recombine, validating that both streams describe the same frame count |
| `AV Latent Info` | shapes, implied duration, token count, mismatch detection |
| `Upscale Video Latent` | spatial resize of video only; audio passes through untouched |
| `Temporal Trim` | shorten both streams together, staying on the frame grid |

`Split → Join` is bit-identical (`torch.equal` on both streams, asserted in the
tests). `Upscale Video Latent` folds time into batch so frames cannot blend into
each other — there is a test that puts signal in one frame and checks its
neighbours stay empty.

Note that `SaveLatent` / `LoadLatent` **cannot** round-trip an H3 latent: they
call `samples["samples"].contiguous()`, which `NestedTensor` does not implement.
Keep AV latents in-graph.

### Prompt Lint

`MiniMax H3 Prompt Lint` — deterministic checks against MiniMax's documented
format. No LLM, no rewriting; the prompt passes through unchanged.

H3's prompt rules fail *silently*: a malformed prompt does not raise, it just
produces worse video. Checks include:

- all required fields present, correctly spelled, in order (3 for t2va/fl2va,
  6 for ref2va)
- `[Shot 1]` has no timestamp; later shots have strictly increasing
  `At MM:SS.mmm,` times inside the clip duration
- shot numbering sequential from 1
- `<d>[Lang] ...</d>` balanced, tagged, and within the 11 stably-supported
  languages
- `(S1)`/`(S2)` speaker IDs numbered consecutively; dialogue attributed
- voiceover uses the exact required phrase *and* the following closed-lips
  statement
- `overall_soundscape` within 1–4 sentences and not restating dialogue
- `non_diegetic_music` within 1–3 sentences and free of diegetic sources
  (radio, phone, TV — things characters can hear)
- ref2va: `<Picture N>`/`<Video N>`/`<Audio N>` numbered without gaps and within
  the connected reference count, `<Subject N>` defined before use, summary
  task-type prefix valid, retention markers from the fixed vocabulary, and no
  `(Sx)` in `retention_analysis` (the guide forbids it)

**Verification bar:** all four worked examples from the base guide and the
complete example from the ref guide pass clean. A linter that flags the
reference prompts from the guide it implements is worse than no linter, so those
are the first tests in the file.

For LLM-based *rewriting*, use
[`rockerBOO/boo-textgen`](https://github.com/rockerBOO/boo-textgen) or
[`MaxKruse/ComfyUI-PromptEnhancer`](https://github.com/MaxKruse/ComfyUI-PromptEnhancer).
This node deliberately does not duplicate them — it verifies, they generate.

### Reference Encode Cache

`MiniMax H3 Reference Encode Cache` — stop re-encoding ref2va references on
every run.

`MiniMaxH3ReferenceToVideo` re-runs `vae.encode()` over every reference image
each execution. The real ref2va workflow is "fix the references, iterate on the
prompt", so that work is repeated for nothing — and ref2va is already the
expensive mode, because reference tokens ride through every sampling step.

- Cache key covers image content hash **and** target geometry **and**
  `ref_image_size` mode. A resize correctly misses.
- Entries live on **CPU**. A cache pinning VRAM would defeat the purpose.
- Bounded LRU with an entry count and a GiB ceiling, plus a manual clear.
- Output is identical to the uncached path — asserted, not assumed.

`ref_image_size: max` ignores generation size, so entries survive a resolution
change; `match` does not, and correctly re-encodes.

### Audio Reroll

`MiniMax H3 Audio Reroll` — a new soundtrack over an existing picture.

This is the one capability here that **Wan and LTX cannot have**: their audio is
added afterwards by a separate model, so there is no joint latent to
re-sample. H3 keeps video and audio as separate tensors in one pack.

It does not try to freeze half of a jointly-denoised pack mid-sampling. Instead
it uses machinery the DiT already exposes: `extra_conds` forwards
`minimax_refs` to the DiT, and every reference block carries a noise
augmentation factor (`comfy/ldm/minimax/model.py:473-495`). At
`visual_cond_noise_aug = 1.0` the reference enters completely un-noised and
`_forward` pins its timestep near 1.0 — "already clean, treat as given". So the
finished video latent goes back in as a whole-clip `kind: "video"` reference at
full strength.

**Measured, not hoped for.** On a 5.167 s / 864×480 / 20-step clip at a fixed
seed, with the soundscape and score text replaced wholesale:

| | Result |
|---|---|
| Video PSNR vs original | **43.60 dB** (min 42.68 — no frame collapsed) |
| Audio | genuinely different (different SHA-256, mean level −27.7 → −28.8 dB) |
| Speed | **26 s/it original → 63 s/it reroll** |

The picture is very close but **not bit-identical**, because it is still
resampled — PSNR in the low 40s is roughly the drift of a lossy re-encode, not a
regeneration.

And the reroll is **2.4× slower than generating the clip from scratch**.
Reference tokens ride through every sampling step, so handing a 124-frame clip
back as a reference roughly doubles the packed sequence, and attention is
quadratic in that length. Use this node when you need the *same performance and
framing* with different scoring; do not use it as a cheap way to get audio
variations.

If you need the original frames exactly, take the `video_passthrough` output and
mux the new audio over the decoded original downstream — that skips the DiT
entirely. The trade-off is that audio generated this way never saw the video, so
lip-sync to existing dialogue is not preserved.

Full method and limitations: [`docs/AUDIO_REROLL_FINDINGS.md`](docs/AUDIO_REROLL_FINDINGS.md).

Requires the `ref2va` checkpoint — `fl2va` has no `minimax_refs` path.

## Tests

```bash
cd ComfyUI
python -m pytest custom_nodes/comfyui-minimax-h3-toolkit/tests -q
```

92 tests. The canvas solver tests replay all 8 rows of measured calibration data
and demand exact agreement on geometry and token counts — that data is the
ground truth, and a mismatch means the solver is wrong.

## Model facts these nodes encode

| Constraint | Value |
|---|---|
| Frame count | must satisfy `n % 17 == 5` → 124 (≈5 s), 243 (≈10 s), 362 (≈15 s) |
| Trained range | ~124–362 frames |
| Canvas | multiples of 32; short edge 768; area cap 768×1344 ≈ 1.03 MP |
| Frame rate | 24 fps fixed |
| Audio | 32 kHz stereo, 40 Hz latent rate |
| `video_tokens` | `latent_t × (H//32) × (W//32)` — **32**, not 16: the VAE compresses 16× spatially, then a `1×2×2` patchify halves each spatial axis again |
| Guidance | CFG-distilled → `BasicGuider`, never `CFGGuider` |
| Sigma shift | `shift_video` 12.0 / `shift_audio` 3.0; the DiT derives the audio schedule from the video one |

On a 24 GB card, use `int8_convrot` weights, not `nvfp4` — the latter runs
emulated on pre-Blackwell hardware (`emulated ops: ... nvfp4` in the startup
log). Launch with `--disable-smart-memory`: text encoder 25.3 GB + DiT 20.0 GB
against 32 GB of system RAM.

## License

MIT. The MiniMax-H3 weights are under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
