# Verification brief — audit another agent's MiniMax-H3 deliverable

You are the reviewer. Another agent completed one of the tasks in
`AGENT_TASKS.md`. Your job is to find out whether their claims are true.

You are not here to be agreeable. A review that finds nothing is only useful if
you actually tried to break the thing. Assume the implementer was competent and
well-intentioned but ran out of time, tested on the happy path, and rounded
their conclusions in their own favour — because that is what usually happens.

## Ground rule: verify, do not re-read

Reading code and agreeing with it is not verification. If a claim can be
checked by executing something, execute it. Specifically:

- A number in a report → recompute it yourself from the raw data.
- "Tests pass" → run them and paste the output.
- "The node works" → submit a graph to a live ComfyUI and inspect the artifact.
- "Identical output" → `torch.equal`, or hash the files. Not eyeballing.
- "Faster" → time both paths yourself, same seed, same machine state.

If you cannot execute something (no GPU time, missing weights, would take
hours), say so explicitly and mark that claim **UNVERIFIED**. Never present a
plausible inference as a confirmed result. That failure mode is exactly what
this review exists to catch.

## Read these first

| Path | Why |
|---|---|
| `AGENT_TASKS.md` | The shared-context block and the specific task's brief. This is the spec you are auditing against. |
| `README.md` | The measured baseline and the documented negative results. |
| `h3_calibration.csv` | Raw data. Ground truth for anything numeric. |
| `comfy_extras/nodes_minimax_h3.py` | 286 LOC. The 4 core H3 nodes — the style and API reference. |
| `comfy/nested_tensor.py` | Latents are `NestedTensor` pairs. Most latent bugs live here. |

Then read the implementer's diff in full before forming an opinion.

## The invariants — check every one that the deliverable touches

These come from the installed source, verified 2026-08-04. An implementation
that violates any of them is wrong regardless of how good its output looks.

```python
# comfy_extras/nodes_minimax_h3.py L25-46
CANVAS_MULTIPLE = 32; BASE_SHORT_EDGE = 768; MAX_PIXELS = 768 * 1344  # 1.032 MP
FPS = 24; AUDIO_LATENT_FPS = 40; REF_IMAGE_SHORT_EDGE = 2048

frames % 17 == 5                                  # 5, 22, ... 124, 243, 362
latent_t     = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
audio_t      = round(frames / FPS * AUDIO_LATENT_FPS)
video_tokens = latent_t * (height // 32) * (width // 32)   # 32, not 16
video latent : [B, 24, latent_t, H // 16, W // 16]
audio latent : [B, 32, 2, audio_t]                # dim 2 == L/R
```

The `// 32` is the classic mistake: the VAE compresses 16× spatially, then a
`1×2×2` patchify halves each spatial axis again. Anyone who used `// 16` will be
off by 4× on every token count.

Also check:
- Weights are **CFG-distilled** → `BasicGuider`. A `CFGGuider` anywhere is a bug.
- `shift_audio` reaches the DiT **only** through `transformer_options`
  (`nodes_minimax_h3.py` L320-322), and the audio schedule is derived by
  inverting the video one. Any change to the sigma schedule must account for
  this or the streams desynchronize.
- Text encoder must be `int8_convrot`, not `nvfp4_awq`, on pre-Blackwell cards.
- No input tensor or latent dict mutated in place.
- V3 node API: `io.ComfyNode` + `define_schema()` + `execute()`, prefixed
  `node_id`.

## Arithmetic audit — do this even for non-numeric tasks

Every number in the deliverable's report and docs gets recomputed. Write a
throwaway script, do not do it in your head.

This is not hypothetical. During preparation of the source material, one
hand-entered CSV row claimed `latent_t=47, video_tokens=76516` for
1376×768 / 243 frames. The formulas give **72** and **74304** — a 3 % token
error and a wrong `latent_t`, which then propagated into a "+32 %" claim in the
README that should have read "+29 %". It survived several human readings because
it looked plausible. The other 7 rows, which were machine-generated, were all
correct.

So: distrust any number a human typed, and specifically check whether derived
percentages were recomputed after their inputs changed.

## Per-task checks

Apply the section matching the task you are reviewing.

**Task 1 (CanvasSolver).** Feed all 8 CSV rows through the solver. It must
reproduce width, height, frames, `latent_t`, and `video_tokens` **exactly** —
any mismatch is a solver bug, not a data bug. Then probe the edges: 1.1 MP
(must clamp to the area cap), 1 s (must floor to 5 frames), 15 s (362), 16 s
(outside trained range → must warn), 21:9 and 9:16 (both axes still multiples
of 32). Check the s/it fit: does it interpolate the measured points closely, and
what does it do when extrapolated past 74k tokens — does it silently return a
confident wrong number, or does it flag that it is extrapolating?

**Task 2 (AV latent toolkit).** `Split` → `Join` must be bit-identical:
`torch.equal` on both streams, not `allclose`. Then run a real 0.4 MP / 5 s
generation with Split→Join spliced in before the sampler and confirm the video
still decodes. Try to break `Join` with mismatched `latent_t`/`audio_t` — it
must raise a clear error naming both shapes, not produce a corrupt latent.
Verify `TemporalTrim` cannot produce a frame count violating `% 17 == 5`, and
that it moves **both** streams together.

**Task 3 (audio reroll).** The video stream must come out **unchanged** — hash
the decoded frames against the source generation, or assert `torch.equal` on the
video latent. "Looks the same" is not acceptable; a subtly drifting video is the
expected failure mode here. Confirm the audio actually changed (not silence, not
identical: check `mean_volume`/`max_volume` via ffmpeg, and compare hashes).
Test with `<d>...</d>` dialogue and judge lip-sync, because ambience-only
testing hides the hard case. Measure time versus full regeneration. **If the
implementer reports this did not work, that is a legitimate outcome** — verify
their negative result is properly documented and reproducible rather than
pushing them to ship something broken.

**Task 4 (ref cache).** Cold and warm runs must produce `torch.equal`
conditioning. A cache that changes results is a critical bug, not a
performance note. Verify the key includes content hash **and** target size
**and** `ref_image_size` mode — swap only the mode and confirm a miss. Check
cached latents live on CPU, not VRAM. Confirm the cache is bounded and that the
bound is actually enforced.

**Task 5 (prompt linter).** The 4 worked examples in the base guide and the 3
in the ref guide must all pass clean — a linter that flags MiniMax's own
reference prompts is worse than no linter. Then confirm each deliberately
broken fixture triggers exactly its expected warning and no others. Probe for
false positives: a legitimate `[Shot 3] At 00:12.000` after `[Shot 2] At
00:04.500` must pass; a diegetic radio described in the multimodal description
(correct placement) must not be flagged.

**Task 6 (blockswap).** Same seed, `blocks_to_swap=0` versus `=N`: report PSNR
against the unswapped reference. Numeric drift is fine, visible change is not.
Verify VRAM actually dropped — `nvidia-smi` sampled during the run, not a
theoretical calculation. Check no core file was patched (the
`WrapperExecutor` hook at `model.py:498-501` should suffice). If they claim a
12 GB or 16 GB card works, that claim needs a real run on that hardware or it is
UNVERIFIED.

**Task 7 (VAE tiling).** Confirm they did not reimplement tiling that already
exists at `vae.py:335-356` — the task was to expose and measure it. Check the
VAE object is cloned, not mutated (a mutated VAE leaks settings into every later
workflow in the session). Inspect decoded output for seams at aggressive
settings, paying attention to temporal seams — the temporal path uses causal
convolutions. "No change needed, defaults are near-optimal" is a valid
conclusion if the measurements support it.

**Task 8 (two-stage refine).** Total two-stage time must be compared against a
real single-pass 0.8 MP run on the same machine, not against the estimate in the
calibration table. Verify the audio-reuse question was answered by measurement
rather than assumption. Look at whether stage 2 actually adds detail or just
resamples — a denoise value set too high will have thrown away stage 1's
composition. If two-stage lost, confirm that is stated plainly.

**Tasks 9 and 10 (sweeps).** These are data tasks, so audit the method: fixed
seed across all cells, one variable at a time, dialogue present where lip-sync
is being judged. Recompute every percentage in their tables. Confirm subjective
judgements are labelled subjective. Check the cell count matches the reported
GPU time — a 15-cell grid at ~9 min cannot have finished in an hour.

## Output format

```
VERDICT: PASS | PASS WITH ISSUES | FAIL

WHAT I EXECUTED
  <commands run, tests run, graphs submitted — with actual output>

CLAIMS CHECKED
  <claim> -> CONFIRMED | WRONG (expected X, got Y) | UNVERIFIED (why)

FINDINGS
  [CRITICAL] wrong output / silent corruption / data loss / invariant violated
  [HIGH]     bug on a realistic path, or a number that is wrong
  [MEDIUM]   maintainability, missing edge case, thin test coverage
  [LOW]      style, naming, docs

  For each: file:line, why it is wrong, and the concrete input that triggers it.

WHAT I COULD NOT CHECK
  <explicitly list it — do not quietly omit>
```

`FAIL` on any CRITICAL. `PASS WITH ISSUES` when only HIGH and below remain.

Two more rules on the writeup:

**Do not invent findings to look thorough.** If the work is genuinely good,
say so and list what you executed to establish that. Padding a review with
style nitpicks to hit a quota wastes everyone's time and buries the real
findings.

**Separate "wrong" from "different from how I would do it."** Only the first is
a finding. If the implementer chose a different but valid approach, that is
their call to make.
