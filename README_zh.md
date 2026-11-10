# ComfyUI-MiniMaxH3-Tools

[English](README.md) | 简体中文

MiniMax-H3 在一次采样中同时生成视频和同步的立体声音频。这个插件补上了 ComfyUI 里缺少的周边工具：加入队列前规划画布和帧数、检查提示词、直接处理成对的视频/音频潜空间、只重新生成声音，以及加载 DiffSynth-Studio 的 NF4 权重。

**安装：**在 ComfyUI Manager 中搜索 `MiniMax-H3 Tools`（或运行 `comfy node install minimax-h3-tools`），然后重启 ComfyUI。需要 ComfyUI `0.30.0` 或更高版本。四个 NF4 加载节点还需要 BitsAndBytes，见[安装](#安装)。

**从这里开始：**打开 ComfyUI 的“模板”浏览器，在本插件的条目中选择 `MiniMax-H3_Planning_and_Prompt_Lint`（浏览器以安装文件夹命名该条目：通过注册表安装是 `minimax-h3-tools`，用 git clone 安装是 `ComfyUI-MiniMaxH3-Tools`），然后运行。它不需要任何模型文件。所有示例也都是 [`example_workflows/`](example_workflows/) 里的普通 `.json` 文件，可以直接拖进编辑器。

**完全在本地运行。**这些节点不发起任何网络请求，也不会向外发送任何内容。模型文件由你自己下载，或者在示例工作流里缺少模型时，点击 ComfyUI 显示的下载链接，由 ComfyUI 下载。

节点文字提供英文和简体中文，跟随 ComfyUI 的语言设置；其他语言显示英文。

## 为什么需要它

| 原生工作流的缺口 | 会出什么问题 | 这里提供的工具 |
|---|---|---|
| H3 使用成对的视频/音频 `NestedTensor` 潜空间 | 内置的潜空间节点只接受单个张量，可能报错或损坏其中一个流 | 五个音视频潜空间节点，用于拆分、验证、缩放、裁剪和重新合并这一对流 |
| 帧数会悄悄对齐到 H3 的 `17k+5` 网格 | 排队的时长与要求的时长不同 | 画布规划先报告精确的帧数、潜空间尺寸和令牌数 |
| 生成路径接受超出文档训练面积的画布 | 长时间渲染完成后，画面尺寸却没有经过验证 | 画布规划发出警告，或明确地限制到训练面积 |
| H3 提示词的结构错误不会报错 | 字段拼错、切镜时间不对或对白格式错误会在没有任何提示的情况下降低质量 | 确定性的提示词检查：干净的提示词原样通过，问题则逐条报告 |
| 核心 H3 没有“保留表演、更换声音”的工作流 | 更换声音通常要重新生成整个片段 | 实验性的音频重生成把已完成的视频当作干净参考，产生新的音视频样本 |
| DiffSynth 的 H3 NF4 文件把 Linear 权重存成展平的 BitsAndBytes 张量 | 原生 ComfyUI 无法推断 H3 架构，也无法还原量化状态 | 四个实验性 NF4 加载节点还原逻辑形状，并加载 DiT、Qwen3-VL 编码器和两个 VAE |

这个插件不会替代 ComfyUI 原生的 MiniMax-H3 节点，不会下载模型权重，不会用 LLM 改写提示词，也不会声称某一台机器的基准测试能预测另一台机器的运行时间。

## 安装

MiniMax-H3 支持需要 ComfyUI `0.30.0` 或更高版本。

从 ComfyUI Manager 安装 `MiniMax-H3 Tools`，或使用 Comfy CLI：

```bash
comfy node install minimax-h3-tools
```

手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Rinne414/ComfyUI-MiniMaxH3-Tools.git
```

八个核心工具不需要额外的 Python 依赖。四个 NF4 加载节点需要在 ComfyUI 的 Python 环境中安装 BitsAndBytes：

```bash
python -m pip install "bitsandbytes>=0.50.0"
```

重启 ComfyUI 后搜索 `MiniMax H3`。启动时日志会写出为 NF4 加载节点检查到的内容，例如 `MiniMax-H3 Tools: NF4 loaders: bitsandbytes 0.50.0 found, CUDA device found`。加载节点运行时会再次检查。

模型权重需要另行下载。标准 INT8 路径请使用 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 中说明的文件。NF4 请使用下面的 DiffSynth 文件。

### 实验性 NF4 加载节点

MiniMax 发布基础 H3 模型。DiffSynth-Studio 发布预先量化的 BitsAndBytes NF4 权重和它自己的推理实现。本仓库是独立的兼容层，把这些权重适配到 ComfyUI；它不是 MiniMax、DiffSynth-Studio 或 ComfyUI 的官方支持。

从 [ModelScope](https://modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) 或 [Hugging Face](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4) 下载四个 FL2VA 文件：

| NF4 节点 | 替代 | ComfyUI 模型文件夹 | DiffSynth 文件名 | 大小 |
|---|---|---|---|---:|
| `MiniMax H3 NF4 Model Loader` | `UNETLoader` | `models/diffusion_models` | `minimax-h3-fl2va-nf4.safetensors` | 17.16 GB |
| `MiniMax H3 NF4 Text Encoder Loader` | `CLIPLoader` | `models/text_encoders` | `minimax-h3-text-encoder-nf4.safetensors` | 15.32 GB |
| `MiniMax H3 NF4 Video VAE Loader` | 视频 `VAELoader` | `models/vae` | `video_vae_nf4.safetensors` | 1.61 GB |
| `MiniMax H3 NF4 Audio VAE Loader` | 音频 `VAELoader` | `models/vae` | `audio_vae_nf4.safetensors` | 0.28 GB |

原生 H3 工作流的其余部分保持不变。在四个 NF4 加载节点中选择下载好的文件，然后把它们的 `MODEL`、`CLIP` 和 `VAE` 输出接到原来标准加载节点所连接的位置。

#### NF4 还是 INT8？

NF4 用 4 位存储量化后的 Linear 权重，因此能减少下载量和磁盘占用。但计算时仍需把权重反量化，所以体积更小并不自动意味着更快或质量更好。下面的标准对比使用 Comfy-Org 优化过的 `int8_convrot` DiT 和文本编码器，并搭配它的 FP16/FP32 VAE；这不是与通用 INT8 的对比。

| 测试组合 | 四个模型文件 | RTX 3090，864x480，124 帧，20 步 |
|---|---:|---:|
| DiffSynth NF4 | 34.4 GB | 15:34 |
| Comfy-Org `int8_convrot` 加标准 VAE | 53.9 GB | 11:47 |

在这台机器上，NF4 少占约 36% 的磁盘空间，但耗时多约 32%。这些耗时只代表一种测试配置，不是与硬件无关的基准。

#### 验证与限制

加载节点会还原序列化的 `bitsandbytes__nf4` 量化状态、Linear 的逻辑形状，以及 DiffSynth 按注意力头交错排列的 DiT QKV 布局。遇到 FP4、缺少量化附带数据、非 H3 模型、CUDA 不可用或文件不完整时，会明确报错。

验证内容包括：真实的 BitsAndBytes 预量化往返、ComfyUI 的模型 patcher 与卸载生命周期，以及在 RTX 3090、BitsAndBytes 0.50.0 上的 CUDA 矩阵乘法。完整的 34.4 GB NF4 组合在 20 步的 ComfyUI 工作流中生成了与提示词相符的 864x480、124 帧视频，输出为 24 fps H.264 视频、同步的 32 kHz 立体声 AAC，以及单独的 5.175 秒立体声 FLAC。独立组件测试也验证了 NF4 Qwen3-VL 编码器和两个 VAE。

当前限制：

- 只有 NF4 FL2VA 扩散模型完成了端到端验证；现有的 NF4 Ref2VA 文件尚未声明为受支持
- 不支持对 NF4 Linear 层应用 LoRA 补丁
- 已测试的运行环境是 NVIDIA CUDA；其他 GPU 和 BitsAndBytes 版本未经验证
- `.incomplete` 和 `.aria2log` 是下载中间文件，不是模型文件

## 节点

画布规划、提示词检查和音视频潜空间信息还会把文字报告附在任务上，因此报告会随这次运行记录在队列和历史中。若要在画布上查看，请把报告输出连接到核心的 `Preview as Text` 节点；规划和音视频潜空间两个示例已经这样连接。

节点在 ComfyUI 中文界面下显示为中文名称，例如“MiniMax H3 画布规划”。下面的标题使用英文名称，便于对照 ComfyUI 的英文界面。

### MiniMax H3 Canvas Planner（画布规划）

在渲染加入队列前规划精确的几何尺寸。

输出：

- 位于 ComfyUI 32 像素网格上的 `width` 和 `height`
- 向上对齐到 H3 `17k+5` 时间网格的 `frames`
- `latent_t` 和 `audio_t`
- 精确的 `video_tokens`
- 可选的运行时间估算和详细的 `report`

运行时间估算默认关闭。在明确选择硬件配置之前，节点的两个估算输出都返回 `-1.0`。这是有意为之：仅凭 GPU 型号不足以预测 H3 的速度，量化方式、注意力后端、PyTorch/CUDA 版本、系统内存和卸载行为也都有影响。

`calibration_profiles.json` 包含一个来自原开发机器、带有清晰标注的配置。选择它就意味着接受把那套软硬件组合当作近似。按相同的结构添加其他配置，然后重启 ComfyUI 以重建下拉选项。格式错误的配置会在启动时报出具体的字段和原因，而不是悄悄地让估算失效。

最初的 RTX 3090 测量数据和复现工具作为针对单台机器的研究保存在 [research/README.md](research/README.md)。它们不是该节点的默认行为。

### MiniMax H3 Split AV Latent（拆分音视频潜空间）

把 H3 的 `NestedTensor` 拆分为普通的视频和音频潜空间字典。每个输出都带有流类型标记，使 Join 能发现接反的输入。

### MiniMax H3 Join AV Latent（合并音视频潜空间）

重新合并视频流和音频流。严格模式会在构造 H3 潜空间之前验证通道布局、时间网格和推算出的帧数。

### MiniMax H3 AV Latent Info（音视频潜空间信息）

报告两个张量的形状、时长、令牌数、数据类型、设备以及流不匹配的情况，用于在采样前检查陌生的工作流。

### MiniMax H3 Upscale Video Latent（放大视频潜空间）

只调整视频流的空间尺寸，音频保持不变。插值时把时间维折叠进批次维，因此相邻帧不会互相混合。

### MiniMax H3 Temporal Trim（时长裁剪）

同时缩短视频和音频。结果仍然落在 H3 的时间网格上，不会悄悄产生时长不一致的两个流。

`SaveLatent` 和 `LoadLatent` 目前无法往返保存 H3 音视频潜空间，因为内置节点会对成对的 `NestedTensor` 调用 `.contiguous()`。请让这些潜空间留在节点图里传递。

### MiniMax H3 Prompt Lint（提示词检查）

按 MiniMax 文档规定的格式检查提示词，不会改写提示词。

它会验证必需的基础字段或 ref2va 字段、镜头编号和时间戳、对白语言标签、说话人 ID、旁白措辞、环境音/音乐规则以及参考标签编号。节点返回原始提示词、易读的报告和一个 `is_clean` 布尔值。`raise_on_warning` 可以让检查结果直接使工作流失败。

它是对提示词生成节点的补充，而不是重复。

### MiniMax H3 Audio Reroll（音频重生成，实验性）

把已完成的 H3 视频潜空间当作干净的整段参考，并根据修改后的环境音或音乐文字采样出新的音视频潜空间。

它的限制来自结构，而且很重要：

- 画面受到强力约束，但并不是冻结的，也不会逐位一致
- 完整的视频参考会让每个采样步骤多出令牌，可能比从头生成更慢
- 实现要求使用 `ref2va` 检查点
- 目前的验证只覆盖一个片段和一个种子，并未涵盖所有提示词或机器

当保留表演和构图比速度更重要时使用它。若最终混流必须保留原始帧，请把 `video_passthrough` 输出与新解码的音频一起使用。

实验、方法和未测试的变量记录在 [docs/AUDIO_REROLL_FINDINGS.md](docs/AUDIO_REROLL_FINDINGS.md)（英文）。这些数字只描述那一次实验。

## 示例工作流

`example_workflows/` 下的文件会出现在 ComfyUI 的“模板”浏览器中，位于本插件的条目里（以安装文件夹命名：`minimax-h3-tools` 或 `ComfyUI-MiniMaxH3-Tools`），也可以直接拖进编辑器。其中的加载节点带有 Comfy-Org 标准 INT8 文件（Hugging Face 上的 `Comfy-Org/MiniMax-H3`）的下载链接，缺少模型时 ComfyUI 会提示下载。请先从第一个工作流开始，它不需要任何模型文件。

| 工作流 | 用途 | 是否需要模型 |
|---|---|---|
| `MiniMax-H3_Planning_and_Prompt_Lint.json` | 在加载权重之前规划精确的几何尺寸并检查提示词 | 否 |
| `MiniMax-H3_AV_Latent_Tools.json` | 演示拆分、仅视频缩放、严格合并、时长裁剪和潜空间检查 | 否 |
| `MiniMax-H3_Audio_Reroll_Experimental.json` | 生成原始音视频潜空间，用修改后的声音文字重生成，再把原始画面与新音频混流 | 是，需要 `ref2va` 以及文本编码器和两个 VAE |
| `MiniMax-H3_AV_Advanced.json` | 用于存档硬件研究的原始高级 T2VA/FL2VA 工作流 | 是 |

三个以工具为主的示例由 `scripts/generate_example_workflows.py` 生成。测试会验证每一条连线，并确保已提交的 JSON 不会与生成器脱节。

## 模型约束

| 约束 | 数值 |
|---|---|
| 帧网格 | `frame_count % 17 == 5` |
| 文档记载的训练时长 | 约 124-362 帧，在 24 fps 下大约 5-15 秒 |
| 画布网格 | 32 的倍数 |
| 文档记载的训练面积 | `768 * 1344` 像素 |
| 视频帧率 | 24 fps |
| 音频潜空间速率 | 40 Hz |
| 视频令牌数 | `latent_t * (height // 32) * (width // 32)` |
| 引导 | CFG 蒸馏权重使用 `BasicGuider` |

## 测试

请在 ComfyUI 环境中运行测试，这样才能使用它的 `comfy` 和 `comfy_api` 包：

```bash
cd ComfyUI
python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Tools/tests -q
```

测试覆盖精确的画布/帧数计算、严格的校准配置验证、逐位一致的 Split/Join 往返、流不匹配时的失败、空间与时间的隔离、MiniMax 公布的提示词示例、NF4 量化状态解析、按注意力头交错的 QKV 注意力，以及可选的真实 BitsAndBytes CUDA 往返。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `*.py` | 可安装的 ComfyUI 扩展和十二个节点 |
| `tests/` | 行为测试 |
| `locales/` | 英文和简体中文的节点文字 |
| `example_workflows/` | 可直接加载的规划、音视频潜空间、重生成和高级生成工作流 |
| `docs/` | 针对具体节点的实验性发现 |
| `research/` | 单台机器的校准数据和复现脚本 |

## 许可证

扩展代码和工作流使用 MIT 许可证。MiniMax-H3 权重使用 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。
