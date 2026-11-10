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
    does not load. `.gguf` encoders are experimental and unverified.
  - `MiniMax H3 Image Size` offers the model card's aspect ratios at 4, 8, 16 and
    20 MP, follows a connected picture's shape, and makes the empty latent.
  - `MiniMax H3 Image Prompt Expander` (experimental) turns a short idea into the
    long English description the model wants. Its instruction text is not part
    of this pack: it is read at run time from the official template that comes
    with ComfyUI, and without that template the prompt passes through unchanged.
  - `MiniMax H3 Save Conditioning` and `MiniMax H3 Load Conditioning` keep an
    encoded prompt in `output/conditioning` (safetensors, no pickle).
  - Example workflows for text to image, editing one picture, and editing with a
    second reference picture.
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
  as core models' weights do.
- Torch Compile, hook and multi-GPU nodes can rebuild an NF4 model or text
  encoder instead of failing with
  `unexpected keyword argument 'disable_dynamic'`.
- A LoRA on an NF4 model now stops the run with a message that says what to do.
  Under dynamic VRAM it used to be ignored without a word.
- The NF4 video VAE works with `--fast fp16_accumulation`.
- Split AV Latent, Join AV Latent, Temporal Trim and Upscale Video Latent keep a
  latent's noise mask instead of dropping it. Split gives each stream its own
  mask, Join puts them back together, and Temporal Trim cuts the mask to the
  kept duration.

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
