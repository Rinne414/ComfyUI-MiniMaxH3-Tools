# ComfyUI-MiniMaxH3-Tools

Eight focused ComfyUI nodes for gaps in the native MiniMax-H3 workflow.

MiniMax-H3 generates video and synchronized stereo audio in one sampling pass.
ComfyUI provides the model-specific conditioning and sigma-shift nodes; this extension
adds the planning, validation, and AV-latent operations that normal ComfyUI
nodes cannot safely perform on H3 data.

## Why this exists

| Native workflow gap | What goes wrong | Tool provided here |
|---|---|---|
| H3 uses a paired video/audio `NestedTensor` latent | Stock latent nodes expect one tensor and can fail or damage one stream | Five AV latent nodes that split, validate, resize, trim, and rejoin the pair |
| Frame counts silently snap to H3's `17k+5` grid | The queued duration differs from the requested duration | Canvas Planner reports the exact frame count, latent sizes, and token count first |
| The generation path accepts canvases outside the documented trained area | A long render can complete with unvalidated geometry | Canvas Planner warns or explicitly clamps to the trained area |
| H3 prompt structure fails silently | Misspelled fields, bad shot timing, or malformed dialogue reduce quality without an error | Deterministic Prompt Lint passes clean prompts through and reports specific violations |
| Core H3 has no "reuse this performance with different sound" workflow | Changing sound normally regenerates the entire clip | Experimental Audio Reroll reuses the finished video as a clean reference and produces a new AV sample |

This extension does not replace ComfyUI's native MiniMax-H3 nodes, download model
weights, rewrite prompts with an LLM, or claim that one machine's benchmark
predicts another machine's runtime.

## Install

MiniMax-H3 support requires ComfyUI `0.30.0` or later.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools.git
```

No extra Python dependencies are required. Restart ComfyUI and search for
`MiniMax H3`.

The model weights are separate. Use the files and directory layout documented
by [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Nodes

### MiniMax H3 Canvas Planner

Plans the exact geometry before a render is queued.

Outputs:

- `width` and `height` on ComfyUI's 32-pixel grid
- `frames` snapped up to H3's `17k+5` temporal grid
- `latent_t` and `audio_t`
- exact `video_tokens`
- optional runtime estimates and a detailed report

Runtime estimation is disabled by default. The node returns `-1.0` for both
estimate outputs until a hardware profile is explicitly selected. This is
deliberate: GPU model alone is not enough to predict H3 speed. Quantization,
attention backend, PyTorch/CUDA build, system RAM, and offloading behavior also
matter.

`calibration_profiles.json` contains one clearly labelled profile from the
original development machine. Selecting it means accepting that exact stack as
an approximation. Add another profile by following the same schema, then
restart ComfyUI so the combo options are rebuilt. Malformed profiles fail at
startup with the exact field and reason instead of silently disabling estimates.

The original RTX 3090 measurements and reproducibility tools are preserved as
machine-specific research in [research/README.md](research/README.md). They are
not the default behavior of the node.

### MiniMax H3 Split AV Latent

Splits an H3 `NestedTensor` into ordinary video and audio latent dictionaries.
Each output is tagged with its stream type so Join can detect reversed inputs.

### MiniMax H3 Join AV Latent

Recombines video and audio streams. Strict mode validates channel layout,
temporal grid, and the implied frame count before constructing an H3 latent.

### MiniMax H3 AV Latent Info

Reports both tensor shapes, duration, token count, dtype, device, and stream
mismatches. This is intended for inspecting unfamiliar workflows before
sampling.

### MiniMax H3 Upscale Video Latent

Resizes only the spatial video stream and leaves audio unchanged. Time is folded
into the batch dimension during interpolation so adjacent frames cannot blend.

### MiniMax H3 Temporal Trim

Shortens video and audio together. The result remains on H3's temporal grid and
cannot silently produce mismatched stream durations.

`SaveLatent` and `LoadLatent` currently cannot round-trip an H3 AV latent because
the stock node calls `.contiguous()` on the paired `NestedTensor`. Keep these
latents in the graph.

### MiniMax H3 Prompt Lint

Checks MiniMax's documented prompt format without rewriting the prompt.

It validates the required base or ref2va fields, shot numbering and timestamps,
dialogue language tags, speaker IDs, voiceover wording, soundscape/music rules,
and reference tag numbering. The node returns the original prompt, a readable
report, and an `is_clean` boolean. `raise_on_warning` can make validation fail
the workflow explicitly.

This complements prompt-generation nodes; it does not duplicate them.

### MiniMax H3 Audio Reroll (Experimental)

Uses a finished H3 video latent as a clean whole-clip reference while sampling
a new AV latent from changed soundscape or music text.

The limitation is structural and important:

- the picture is strongly conditioned, not frozen or bit-identical
- the full video reference adds tokens to every sampling step and may be slower
  than generating from scratch
- the implementation requires the `ref2va` checkpoint path
- validation so far covers one clip and one seed, not every prompt or machine

Use it when preserving performance and framing matters more than speed. Use the
`video_passthrough` output with the newly decoded audio when the final mux must
keep the original frames exactly.

The measured experiment, method, and untested variables are documented in
[docs/AUDIO_REROLL_FINDINGS.md](docs/AUDIO_REROLL_FINDINGS.md). Those numbers
describe that experiment only.

## Example workflows

The files under `workflows/` can be dragged directly into ComfyUI:

| Workflow | Purpose | Model required |
|---|---|---|
| `MiniMax-H3_Planning_and_Prompt_Lint.json` | Plan exact geometry and validate a prompt before loading weights | No |
| `MiniMax-H3_AV_Latent_Tools.json` | Exercise split, video-only resize, strict join, temporal trim, and latent inspection | No |
| `MiniMax-H3_Audio_Reroll_Experimental.json` | Generate an original AV latent, reroll with changed sound text, then mux the original picture with new audio | Yes, `ref2va` plus the text encoder and both VAEs |
| `MiniMax-H3_AV_Advanced.json` | Original advanced T2VA/FL2VA workflow used for the archived hardware study | Yes |

The three tool-focused examples are generated from
`scripts/generate_example_workflows.py`. Their tests verify every link and
ensure the committed JSON cannot drift from the generator.

## Model constraints

| Constraint | Value |
|---|---|
| Frame grid | `frame_count % 17 == 5` |
| Documented trained duration | about 124-362 frames, roughly 5-15 seconds at 24 fps |
| Canvas grid | multiples of 32 |
| Documented trained area | `768 * 1344` pixels |
| Video frame rate | 24 fps |
| Audio latent rate | 40 Hz |
| Video token count | `latent_t * (height // 32) * (width // 32)` |
| Guidance | CFG-distilled weights use `BasicGuider` |

## Tests

Run the tests inside a ComfyUI environment so its `comfy` and `comfy_api`
packages are available:

```bash
cd ComfyUI
python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Tools/tests -q
```

The tests cover exact canvas/frame calculations, strict calibration profile
validation, bit-identical Split/Join round trips, stream mismatch failures,
spatial and temporal isolation, and MiniMax's published prompt examples.

## Repository layout

| Path | Purpose |
|---|---|
| `*.py` | Installable ComfyUI extension and eight nodes |
| `tests/` | Behavior tests |
| `workflows/` | Loadable planning, AV-latent, reroll, and advanced generation workflows |
| `docs/` | Node-specific experimental findings |
| `research/` | Single-machine calibration data and reproduction scripts |

## License

The extension code and workflows are MIT licensed. MiniMax-H3 weights use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
