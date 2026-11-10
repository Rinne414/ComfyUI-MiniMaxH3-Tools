# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-11

MiniMax-H3-Image now runs on a 24 GB GPU with 32 GB of RAM: five new nodes and
three example workflows cover text to image and picture editing, and the NF4
loaders and AV latent nodes got fixes. The pack also gained English, Simplified
Chinese and Traditional Chinese node text, search aliases, Preview as Text in the
examples and a Chinese README.

### Added

- MiniMax-H3-Image (low RAM), merged from the former H3-Image tools. They need
  ComfyUI 0.45.0 or later; on an older ComfyUI the rest of the pack still loads
  and the log says the image nodes are unavailable.
  - `MiniMax H3 Image Encode (Low RAM)` loads the 27 GB text encoder, encodes the
    prompt and up to 16 pictures, and frees the encoder before the DiT loads. An
    unchanged prompt is read from `output/h3image_encode_cache` and the encoder
    does not load. `.gguf` encoders are experimental and unverified. Its report
    adds one line when the prompt's `<Picture n>` references and the connected
    pictures disagree, for example "prompt mentions Picture 3 but only 2 pictures
    are connected"; it never blocks the run.
  - `MiniMax H3 Image Size` offers the model card's aspect ratios at 4, 8, 16 and
    20 MP, follows a connected picture's shape, and makes the empty latent.
  - `MiniMax H3 Image Prompt Expander` (experimental) turns a short idea into the
    long English description the model wants. Its instruction text is not part
    of this pack: it is read at run time from the official template that comes
    with ComfyUI, and without that template the prompt passes through unchanged.
  - `MiniMax H3 Save Conditioning` and `MiniMax H3 Load Conditioning` keep an
    encoded prompt in `output/conditioning` (safetensors, no pickle).
  - Example workflows for text to image, editing one picture, and editing with a
    second reference picture. The expander's seed is fixed in them so its cache
    can hit, and its report names the instruction file it used.
- Node search finds every node by English and Simplified Chinese terms such as
  `h3`, `canvas`, `17k+5`, `nf4`, `prompt lint`, `画布` and `提示词检查`.
- The four NF4 loaders and Audio Reroll carry ComfyUI's experimental badge.
- Every NF4 loader input says which file it expects and which models folder it
  belongs in. Prompt Lint and the AV latent nodes gained tooltips for the inputs
  that had none.
- Canvas Planner, Prompt Lint and AV Latent Info attach their report to the job,
  so it is recorded with the run in the queue and history. Connect it to a core
  `Preview as Text` node to read it on the canvas. The string outputs are
  unchanged.
- The Planning and AV Latent example workflows include `Preview as Text` nodes,
  so pressing Run no longer fails with "Prompt has no outputs".
- English, Simplified Chinese and Traditional Chinese node text through ComfyUI's
  own `locales/` files (`locales/en`, `locales/zh` and `locales/zh-TW`, each with
  a `nodeDefs.json`).
- The loader nodes in the example workflows carry model download links, so
  ComfyUI offers to download a missing file.
- A startup log line states whether BitsAndBytes and a CUDA device were found for the
  NF4 loaders. A failure of that check can never stop the pack from loading.
- `README_zh.md`, a Simplified Chinese README.
- Bug report and feature request templates, a changelog, and a weekly CI run
  against both the pinned ComfyUI commit and ComfyUI master.
- The publish workflow skips a version that is already on the registry, and confirms that the Comfy Registry lists the published
  version as the latest and active one.

### Changed

- Node display names, descriptions and tooltips in the Python schemas are
  English only. The Chinese text moved to `locales/zh/nodeDefs.json`.
  Node ids, input names, output names and their order are unchanged, so existing
  workflows load as before.
- The example workflows moved from `workflows/` to `example_workflows/`, which
  is the folder ComfyUI's template browser reads.
- The README opens with what the pack is, how to install it and which example to
  open first.
- The registry description is one plain sentence, and the package declares
  `Operating System :: OS Independent` and a documentation link.
- CI tests against ComfyUI 0.45.0 (the first release with MiniMax-H3-Image
  support) and, weekly, against ComfyUI master.
- The publish workflow pins `Comfy-Org/publish-node-action` to the commit of the
  `v1` branch that published 0.1.0, instead of the moving `v1` branch.
- `.github/` is excluded from the registry package.

### Fixed

- The NF4 diffusion model no longer stops with
  `AttributeError: '_NF4Linear' object has no attribute '_combined_linear'` when
  ComfyUI's dynamic VRAM prefetches the next block.
- The NF4 loaders no longer write a fresh copy of every weight back into system
  RAM on each run. Weights now reach the GPU through ComfyUI's own loading path,
  as core models' weights do. Measured on the NF4 text encoder: 13.3 GB resident
  before, about 2.1 GB after.
- The NF4 video VAE loader no longer makes ComfyUI allocate about 9 GB of float32
  zeros in system RAM while loading. Under dynamic VRAM, core filled every NF4
  weight missing from the state dict with a full-size float32 tensor (36 blocks
  x 67,108,864 values x 4 bytes = 9,216 MiB). Each NF4 weight is now passed as
  a 4-byte placeholder with its logical shape. Measured: load, encode and decode
  of 5 frames at 256x256 took 4.0 s with a peak of 2.0 GB resident RAM, against
  10 GB before.
- The NF4 diffusion model loader now gives ComfyUI's dynamic VRAM the same
  fast-disk storage policy that core's own loaders pass. Before, every staged
  weight was first copied into pinned host RAM, which grew host memory on the
  first sampling step until core's pin budget stopped it. Measured (5 frames, 2
  steps, 256x256): both steps ran with a peak of 2.6 GB resident RAM, against
  10.6 GB within 4 s before.
- Torch Compile, hook and multi-GPU nodes can rebuild an NF4 model or text
  encoder instead of failing with
  `unexpected keyword argument 'disable_dynamic'`.
- A LoRA on an NF4 model now stops the run with a message that says what to do.
  Under dynamic VRAM it used to be ignored without a word. Verified on the
  dynamic VRAM path only.
- The NF4 video VAE no longer fails with `--fast fp16_accumulation` (checked by
  tests only; see Known limits).
- Split AV Latent, Join AV Latent, Temporal Trim and Upscale Video Latent keep a
  latent's noise mask instead of dropping it. Split gives each stream its own
  mask, Join puts them back together, and Temporal Trim cuts the mask to the
  kept duration.

### Verified on real models

RTX 3090, 32 GB of RAM, ComfyUI at commit `ede4e5fe8`, this pack at `c77db71`:

- NF4 text encoder: one encode took 7.0 s, with about 2.1 GB of RAM resident at
  the peak and 14.6 GB of VRAM at the peak.
- NF4 diffusion model (5 frames, 2 steps, 256x256): the log shows
  `fast_disk=True`; both steps ran (2.53 it/s after initialisation); peak
  resident RAM 2.6 GB, minimum available RAM 13.2 GB.
- NF4 video VAE (load, encode and decode of 5 frames at 256x256): 4.0 s, peak
  resident RAM 2.0 GB, minimum available RAM 14.1 GB, decode correct (28.3 dB
  PSNR against the input).
- A LoRA on an NF4 model stops at the sampler with a plain message (dynamic VRAM
  path).
- `tests/test_nf4_cuda.py` passes 5 of 5 on the GPU. It needs the pack folder on
  `PYTHONPATH`.
- In the ComfyUI frontend the experimental badge shows on the canvas, the
  Templates browser lists the pack's templates, and the missing-model dialog
  offers Download buttons for `MiniMax-H3_AV_Advanced`.
- H3-Image at 4 MP, w4a8 model, total time: 78 s on the first run with the Prompt
  Expander on (expansion 56 s, encode 6.4 s, then sampling); 16 s with expansion
  and encode cached in a fresh process; 11 s with the model already loaded; about
  20 s with the expander off and a new prompt. Editing: 28 s with one picture, 36
  s with two. Peak resident RAM 5.0 to 5.4 GB, minimum available RAM about 10 GB.

### Known limits

- Not measured: the NF4 diffusion model at a real video size and length (only
  256x256, 5 frames, 2 steps were run).
- Not verified: `--fast fp16_accumulation` together with the NF4 video VAE, and
  the LoRA message on the legacy (non-dynamic) patcher path.
- The NF4 diffusion model and video VAE loaders stay experimental. The NF4 versus
  INT8 table in the README was measured on ComfyUI 0.30.0 in August 2026 and is
  labelled as such.

## [0.1.0] - 2026-08-04

First public release on the Comfy Registry.

### Added

- Canvas Planner: exact canvas, `17k+5` frame grid, latent sizes and token count
  before a render is queued. Runtime estimates stay off until a hardware profile
  is chosen.
- AV latent nodes for H3's paired video/audio latent: Split, Join (strict mode
  validates channels, grid and frame count), Latent Info, Upscale Video Latent
  and Temporal Trim.
- Prompt Lint: deterministic checks of MiniMax's documented prompt format, with
  an optional hard failure.
- Audio Reroll (experimental): new audio against a finished video latent used as
  a clean reference.
- Four experimental NF4 loaders for DiffSynth-Studio's BitsAndBytes weights:
  diffusion model, Qwen3-VL text encoder, video VAE and audio VAE.
- Example workflows for planning, AV latent tools, audio reroll and an advanced
  generation graph.
- Machine-specific calibration profile and the research scripts behind it.
- A pytest suite covering geometry, latent round trips, prompt rules, NF4
  quantization state, workflows and node metadata.
