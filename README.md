# ComfyUI-MiniMaxH3-Tools

English | [简体中文](README_zh.md)

Tools for running MiniMax-H3 in ComfyUI. They let MiniMax-H3-Image (text to image
and picture editing) run on a 24 GB GPU with 32 GB of RAM, and they add what
ComfyUI lacks for MiniMax-H3 video with audio: planning the canvas and frame
count, checking prompts, working on the paired video/audio latent, rerolling only
the sound, and loading DiffSynth-Studio's NF4 weights.

**Install:** search `MiniMax-H3 Tools` in ComfyUI Manager (or run
`comfy node install minimax-h3-tools`) and restart ComfyUI. The video tools need
ComfyUI `0.30.0` or later and the image nodes need `0.45.0` or later; see
[Install](#install).

**Start here:** open ComfyUI's Templates browser and pick an example from this
pack's entry (the browser names it after the install folder: `minimax-h3-tools`
for a registry install, `ComfyUI-MiniMaxH3-Tools` for a git clone). For images,
open `MiniMax-H3-Image_T2I_Low_RAM`; it needs the model files listed in
[H3-Image](#h3-image-low-ram); the examples carry download links for them. To try the
video tools first, open `MiniMax-H3_Planning_and_Prompt_Lint`, which needs no
model files. Every example is also a plain `.json` file in
[`example_workflows/`](example_workflows/) that you can drag into the editor.

**Runs locally.** The nodes make no network requests and send nothing online.
Model files are downloaded by you, or by ComfyUI when you click the download
link it shows for a missing model in an example workflow.

Node text is available in English, Simplified Chinese and Traditional Chinese and
follows ComfyUI's language setting; other languages show English.

## Why this exists

| Native workflow gap | What goes wrong | Tool provided here |
|---|---|---|
| H3 uses a paired video/audio `NestedTensor` latent | Stock latent nodes expect one tensor and can fail or damage one stream | Five AV latent nodes that split, validate, resize, trim, and rejoin the pair |
| Frame counts silently snap to H3's `17k+5` grid | The queued duration differs from the requested duration | Canvas Planner reports the exact frame count, latent sizes, and token count first |
| The generation path accepts canvases outside the documented trained area | A long render can complete with unvalidated geometry | Canvas Planner warns or explicitly clamps to the trained area |
| H3 prompt structure fails silently | Misspelled fields, bad shot timing, or malformed dialogue reduce quality without an error | Deterministic Prompt Lint passes clean prompts through and reports specific violations |
| Core H3 has no "reuse this performance with different sound" workflow | Changing sound normally regenerates the entire clip | Experimental Audio Reroll reuses the finished video as a clean reference and produces a new AV sample |
| DiffSynth H3 NF4 files store Linear weights as flattened BitsAndBytes tensors | Native ComfyUI cannot infer the H3 architecture or restore the quantization state | Four experimental NF4 loaders restore the logical shapes and load the DiT, Qwen3-VL encoder, and both VAEs |

This extension does not replace ComfyUI's native MiniMax-H3 nodes, download model
weights, rewrite prompts with an LLM, or claim that one machine's benchmark
predicts another machine's runtime.

## H3-Image (low RAM)

MiniMax-H3-Image has a 33B-parameter DiT. On a 24 GB GPU with 32 GB of RAM the
stock nodes cannot run it: the 27 GB Qwen3-VL text encoder stays in RAM after
encoding, so the DiT has no room to load next to it. These nodes load the encoder,
encode, and free it before the DiT loads, so one click runs text to image or
editing on that PC.

| Problem with the stock nodes on 32 GB RAM | What this pack does |
|---|---|
| The text encoder stays in RAM after encoding | **Image Encode (Low RAM)** loads it, encodes, and frees it, with pinned host memory off while it runs |
| Changing the size re-runs the text encoder | **Image Size** makes the latent separately, and an unchanged prompt is reused from ComfyUI's cache or from `output/h3image_encode_cache` |
| The model card's 16 MP sizes are only in a table | **Image Size** has them built in, and 4 / 8 / 20 MP from the same formula |
| A short idea has to be written out as a long English description | **Prompt Expander** (experimental) does it locally with the model and instruction of the official template |

**Requires** ComfyUI with native MiniMax-H3-Image support: `0.45.0` or later, the
first release that includes core pull request #17655 (commit `ede4e5fe8`). On an
older ComfyUI the rest of this pack still loads and the log says the image nodes
are unavailable.

**Models** (from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image),
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and
[Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5)). The example
workflows carry these download links:

| Folder | File | Size |
|---|---|---:|
| `models/diffusion_models` | `minimax_h3_image_pruned_w4a8.safetensors` (default) or `minimax_h3_image_pruned_int8_convrot.safetensors` | 12.5 GB / 21 GB |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27 GB |
| `models/vae` | `minimax_h3_image_vae_fp16.safetensors` | 1.2 GB |
| `models/text_encoders` | `qwen3.5_9b_w4a8_gptq.safetensors`, only for the Prompt Expander | 7.3 GB |

**Examples:**

| Workflow | Purpose |
|---|---|
| `MiniMax-H3-Image_T2I_Low_RAM.json` | Text to image. The Prompt Expander is on |
| `MiniMax-H3-Image_Edit_Low_RAM.json` | Edit one picture. The Prompt Expander is off, so faces, hair and text stay as you wrote them |
| `MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json` | Edit with a second reference picture (`<Picture 1>` and `<Picture 2>`) |

**Measured** on one machine (RTX 3090 24 GB, 32 GB RAM, w4a8 model, 6 steps, cfg 1):

- 4096x4096: 46 s, about 18 GB of VRAM
- 21 MP: 62 s
- 4 MP: about 20 s
- text encoding of a new prompt: about 7.5 s; an unchanged prompt is not encoded again

Treat these as one tested configuration, not a promise for other hardware.

**Prompt Expander (experimental).** It turns a short idea in any language into the
long English description the model wants. Its instruction text belongs to Comfy-Org
and is not part of this pack: the node reads it at run time from the official
MiniMax-H3-Image template installed with ComfyUI (the `comfyui-workflow-templates`
package that ComfyUI `0.45.0` installs). If that template cannot be found, the
node says so in its report and passes your prompt through unchanged; nothing loads.
Expansion can change faces, hair and text on a picture, so it is off in the edit
examples.

**GGUF text encoders (experimental, unverified).** With ComfyUI-GGUF `6ea2651`
the stock load path copied a 14.6 GB `Q4_K_M` file into host RAM on a 32 GB PC and
was stopped. Image Encode (Low RAM) loads `.gguf` files straight to the GPU
instead, but that path is unverified. The INT8 `.safetensors` encoder is
recommended.

## Install

MiniMax-H3 video support requires ComfyUI `0.30.0` or later. The H3-Image nodes
require `0.45.0` or later.

Install `MiniMax-H3 Tools` from ComfyUI Manager, or use Comfy CLI:

```bash
comfy node install minimax-h3-tools
```

For a manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools.git
```

The eight core tools and the five H3-Image nodes require no extra Python
dependencies. The four NF4 loaders require BitsAndBytes in ComfyUI's Python environment:

```bash
python -m pip install "bitsandbytes>=0.50.0"
```

Restart ComfyUI and search for `MiniMax H3`. On startup the log states what it found for the
NF4 loaders, for example
`MiniMax-H3 Tools: NF4 loaders: bitsandbytes 0.50.0 found, CUDA device found`.
The loaders check again when they run.

Model weights are separate. For the standard INT8 path, use the files documented
by [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). For NF4,
use the DiffSynth files below.

### Experimental NF4 loaders

MiniMax publishes the base H3 model. DiffSynth-Studio publishes the pre-quantized
BitsAndBytes NF4 weights and its own inference implementation. This repository
is an independent compatibility layer that adapts those weights to ComfyUI; it
is not official MiniMax, DiffSynth-Studio, or ComfyUI support.

Download the four FL2VA files from
[ModelScope](https://modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) or
[Hugging Face](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4):

| NF4 node | Replaces | ComfyUI model folder | DiffSynth filename | Size |
|---|---|---|---|---:|
| `MiniMax H3 NF4 Model Loader` | `UNETLoader` | `models/diffusion_models` | `minimax-h3-fl2va-nf4.safetensors` | 17.16 GB |
| `MiniMax H3 NF4 Text Encoder Loader` | `CLIPLoader` | `models/text_encoders` | `minimax-h3-text-encoder-nf4.safetensors` | 15.32 GB |
| `MiniMax H3 NF4 Video VAE Loader` | video `VAELoader` | `models/vae` | `video_vae_nf4.safetensors` | 1.61 GB |
| `MiniMax H3 NF4 Audio VAE Loader` | audio `VAELoader` | `models/vae` | `audio_vae_nf4.safetensors` | 0.28 GB |

Keep the rest of the native H3 workflow unchanged. Select the downloaded files
in the four NF4 loaders, then connect their `MODEL`, `CLIP`, and `VAE` outputs
where the standard loaders were connected.

#### NF4 or INT8? (measured on an older core)

The table below was measured in August 2026 on ComfyUI 0.30.0. It has not been
repeated on the current core, so it says nothing about speed or quality there;
see [Known limits](#known-limits).

NF4 stores quantized Linear weights in 4 bits, so it reduces download and disk
footprint. It still dequantizes weights for computation, so smaller does not
automatically mean faster or better output. The standard comparison below uses
Comfy-Org's optimized `int8_convrot` DiT and text encoder with its FP16/FP32
VAEs; it is not a comparison against generic INT8.

| Tested stack | Four model files | RTX 3090, 864x480, 124 frames, 20 steps |
|---|---:|---:|
| DiffSynth NF4 | 34.4 GB | 15:34 |
| Comfy-Org `int8_convrot` plus standard VAEs | 53.9 GB | 11:47 |

On this machine NF4 used about 36% less disk space but took about 32% longer.
Treat these timings as one tested configuration, not a hardware-independent
benchmark.

#### Validation

The loaders restore the serialized `bitsandbytes__nf4` quantization state,
logical Linear shapes, and DiffSynth's head-interleaved DiT QKV layout. They
fail explicitly on FP4, missing quantization sidecars, non-H3 models,
unavailable CUDA, or incomplete files.

On ComfyUI 0.30.0 (August 2026), validation covered real BitsAndBytes pre-quantized round trips, ComfyUI's model
patcher and offload lifecycle, and CUDA matmul on an RTX 3090 with BitsAndBytes
0.50.0. The complete 34.4 GB NF4 stack generated a prompt-aligned 864x480,
124-frame video in a 20-step ComfyUI workflow, with 24 fps H.264 video,
synchronized 32 kHz stereo AAC, and a separate 5.175-second stereo FLAC.
Component-isolation runs also verified the NF4 Qwen3-VL encoder and both VAEs.

#### Verified on the current core

Checked on real models with an RTX 3090 and 32 GB of RAM, ComfyUI at commit
`ede4e5fe8`:

- NF4 text encoder: one encode took 7.0 s, with about 2.1 GB of RAM resident at
  the peak and 14.6 GB of VRAM at the peak. Before the fixes in 0.2.0 the same
  encode left 13.3 GB resident.
- A LoRA on an NF4 model stops the run at the sampler with a plain message (the
  dynamic VRAM path).
- The NF4 diffusion model loads and reaches the start of sampling without the
  earlier `_combined_linear` `AttributeError`.
- `tests/test_nf4_cuda.py` passes 5 of 5 on the GPU.
- In the ComfyUI frontend the experimental badge shows on the canvas, the
  Templates browser lists this pack's templates, and the missing-model dialog
  offers Download buttons for `MiniMax-H3_AV_Advanced`.

#### Known limits

Not verified on the current core:

- NF4 diffusion model: on 32 GB of RAM its resident memory rose from 2.2 GB to
  10.6 GB within about 4 s of the start of sampling, and the test's safety limit
  stopped the run. No step completed, so speed and peak RAM are unknown. The
  cause was found (staged weights were copied into pinned host RAM) and fixed in
  code with tests. Real-model confirmation: pending.
- NF4 video VAE: resident memory rose to 10 GB while loading, and the test was
  stopped before any decode. Decoding with it is unverified. The cause was found
  (core allocated float32 zeros for every NF4 layer) and fixed in code with
  tests. Real-model confirmation: pending.
- The LoRA message on the legacy (non-dynamic) patcher path is unverified.

Because of this, the NF4 diffusion model and NF4 video VAE loaders stay
experimental, and the table above is not a current benchmark.

Also:

- the available NF4 Ref2VA file is not claimed as supported
- LoRA patches on NF4 Linear layers are not supported
- the tested runtime is NVIDIA CUDA; other GPUs and BitsAndBytes versions are
  unverified
- `.incomplete` and `.aria2log` download artifacts are not model files

## Nodes

All 17 nodes:

| Node | What it does |
|---|---|
| MiniMax H3 Canvas Planner | Plans canvas, frame grid, latent sizes and token count before you queue |
| MiniMax H3 Prompt Lint | Checks a video prompt against MiniMax's documented format |
| MiniMax H3 Split AV Latent | Splits the paired video/audio latent into two latents |
| MiniMax H3 Join AV Latent | Rejoins video and audio latents, validating them in strict mode |
| MiniMax H3 AV Latent Info | Reports shapes, duration and token count of an AV latent |
| MiniMax H3 Upscale Video Latent | Resizes only the video stream by interpolation |
| MiniMax H3 Temporal Trim | Shortens video and audio together on the frame grid |
| MiniMax H3 Audio Reroll (Experimental) | Samples new audio against a finished video latent |
| MiniMax H3 NF4 Model Loader (Experimental) | Loads DiffSynth's NF4 diffusion model |
| MiniMax H3 NF4 Text Encoder Loader (Experimental) | Loads DiffSynth's NF4 Qwen3-VL encoder |
| MiniMax H3 NF4 Video VAE Loader (Experimental) | Loads DiffSynth's NF4 video VAE |
| MiniMax H3 NF4 Audio VAE Loader (Experimental) | Loads DiffSynth's NF4 audio VAE |
| MiniMax H3 Image Encode (Low RAM) | Encodes a prompt and pictures, then frees the text encoder. Its report notes when the prompt's `<Picture n>` references and the connected pictures disagree |
| MiniMax H3 Image Prompt Expander | Experimental. Expands a short idea into a long English description |
| MiniMax H3 Image Size | Canvas presets (aspect ratio and megapixels) and the empty image latent |
| MiniMax H3 Save Conditioning | Saves an encoded prompt to `output/conditioning` |
| MiniMax H3 Load Conditioning | Loads a prompt saved by Save Conditioning |

The sections below describe the video and audio nodes. The H3-Image nodes are described in [H3-Image (low RAM)](#h3-image-low-ram).

Canvas Planner, Prompt Lint and AV Latent Info also attach their text report to
the job, so it is recorded with the run in the queue and history. To read it on
the canvas, connect the report output to a core `Preview as Text` node; the
Planning and AV Latent examples already do.

### MiniMax H3 Canvas Planner

Plans the exact geometry before a render is queued.

Outputs:

- `width` and `height` on ComfyUI's 32-pixel grid
- `frames` snapped up to H3's `17k+5` temporal grid
- `latent_t` and `audio_t`
- exact `video_tokens`
- optional runtime estimates and a detailed `report`

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

The files under `example_workflows/` appear in ComfyUI's Templates browser, in
this pack's entry (named after the install folder, `minimax-h3-tools` or
`ComfyUI-MiniMaxH3-Tools`), and can also be dragged directly into the editor. The
loader nodes in them carry download links for the standard Comfy-Org INT8 files
(`Comfy-Org/MiniMax-H3` on Hugging Face), so ComfyUI can offer to download a model
that is missing (checked in the missing-model dialog for `MiniMax-H3_AV_Advanced`). Start with the first workflow; it needs no model files.

| Workflow | Purpose | Model required |
|---|---|---|
| `MiniMax-H3_Planning_and_Prompt_Lint.json` | Plan exact geometry and validate a prompt before loading weights | No |
| `MiniMax-H3_AV_Latent_Tools.json` | Exercise split, video-only resize, strict join, temporal trim, and latent inspection | No |
| `MiniMax-H3_Audio_Reroll_Experimental.json` | Generate an original AV latent, reroll with changed sound text, then mux the original picture with new audio | Yes, `ref2va` plus the text encoder and both VAEs |
| `MiniMax-H3_AV_Advanced.json` | Original advanced T2VA/FL2VA workflow used for the archived hardware study | Yes |
| `MiniMax-H3-Image_T2I_Low_RAM.json` | Text to image with the low-RAM encoder and the Prompt Expander | Yes, the H3-Image models |
| `MiniMax-H3-Image_Edit_Low_RAM.json` | Edit one picture | Yes, the H3-Image models |
| `MiniMax-H3-Image_Edit_Two_Pictures_Low_RAM.json` | Edit one picture with a second reference picture | Yes, the H3-Image models |

The three tool-focused video examples are generated from
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
spatial and temporal isolation, MiniMax's published prompt examples, NF4
quant-state parsing, head-interleaved QKV attention, and optional real
BitsAndBytes CUDA round trips.

The CUDA tests (`tests/test_nf4_cuda.py`) need the pack folder on `PYTHONPATH` as
well as the ComfyUI folder:

```bash
cd ComfyUI
PYTHONPATH=.:custom_nodes/ComfyUI-MiniMaxH3-Tools python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Tools/tests/test_nf4_cuda.py -q
```

## Repository layout

| Path | Purpose |
|---|---|
| `*.py` | Installable ComfyUI extension and seventeen nodes |
| `tests/` | Behavior tests |
| `locales/` | English, Simplified Chinese and Traditional Chinese node text |
| `example_workflows/` | Loadable planning, AV-latent, reroll, and advanced generation workflows |
| `docs/` | Node-specific experimental findings |
| `research/` | Single-machine calibration data and reproduction scripts |

## License

The extension code and workflows are MIT licensed. MiniMax-H3 weights use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
