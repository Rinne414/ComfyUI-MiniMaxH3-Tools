# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Node search finds every node by English and Simplified Chinese terms such as
  `h3`, `canvas`, `17k+5`, `nf4`, `prompt lint`, `画布` and `提示词检查`.
- The four NF4 loaders and Audio Reroll carry ComfyUI's experimental badge.
- Every NF4 loader input says which file it expects and which models folder it
  belongs in. Prompt Lint and the AV latent nodes gained tooltips for the inputs
  that had none.
- Canvas Planner, Prompt Lint and AV Latent Info show their report on the node,
  so no display node is needed. The string outputs are unchanged.
- English and Simplified Chinese node text through ComfyUI's own `locales/`
  files (`locales/en/nodeDefs.json`, `locales/zh/nodeDefs.json`).
- The four loader nodes in the example workflows carry model download links, so
  ComfyUI offers to download a missing file.
- A startup log line states whether the NF4 loaders can run on this machine.
- `README_zh.md`, a Simplified Chinese README.
- Bug report and feature request templates, a changelog, and a weekly CI run
  against both the pinned ComfyUI commit and ComfyUI master.
- The publish workflow confirms that the Comfy Registry lists the published
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
- The publish workflow pins `Comfy-Org/publish-node-action` to release `1.0.1` (commit SHA)
  instead of the moving `v1` branch, by commit SHA.
- `.github/` is excluded from the registry package.

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
