"""Generate the MiniMax-H3 T2VA/FL2VA workflow in both ComfyUI UI and API formats.

Both formats are emitted from one graph definition so the API run that validates
semantics also validates the graph the user loads in the browser.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- slot tables
# (input_name, type) / (output_name, type) per node class, in schema order.
SLOTS = {
    "UNETLoader":            ([], [("MODEL", "MODEL")]),
    "CLIPLoader":            ([], [("CLIP", "CLIP")]),
    "VAELoader":             ([], [("VAE", "VAE")]),
    "PrimitiveStringMultiline": ([], [("STRING", "STRING")]),
    "PrimitiveFloat":        ([], [("FLOAT", "FLOAT")]),
    "StringConcatenate":     ([("string_a", "STRING"), ("string_b", "STRING")],
                              [("STRING", "STRING")]),
    "ResolutionSelector":    ([], [("width", "INT"), ("height", "INT")]),
    "ComfyMathExpression":   ([("values.a", "FLOAT,INT,BOOLEAN")],
                              [("FLOAT", "FLOAT"), ("INT", "INT"), ("BOOL", "BOOLEAN")]),
    "LoadImage":             ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "MiniMaxH3ImageToVideo": ([("clip", "CLIP"), ("vae", "VAE"),
                               ("first_frame", "IMAGE"), ("last_frame", "IMAGE"),
                               ("prompt", "STRING"), ("width", "INT"),
                               ("height", "INT"), ("length", "INT")],
                              [("positive", "CONDITIONING"), ("LATENT", "LATENT")]),
    "MiniMaxH3SigmaShift":   ([("model", "MODEL")], [("MODEL", "MODEL")]),
    "BasicGuider":           ([("model", "MODEL"), ("conditioning", "CONDITIONING")],
                              [("GUIDER", "GUIDER")]),
    "KSamplerSelect":        ([], [("SAMPLER", "SAMPLER")]),
    "BasicScheduler":        ([("model", "MODEL")], [("SIGMAS", "SIGMAS")]),
    "RandomNoise":           ([], [("NOISE", "NOISE")]),
    "SamplerCustomAdvanced": ([("noise", "NOISE"), ("guider", "GUIDER"),
                               ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                               ("latent_image", "LATENT")],
                              [("output", "LATENT"), ("denoised_output", "LATENT")]),
    "VAEDecode":             ([("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")]),
    "VAEDecodeAudio":        ([("samples", "LATENT"), ("vae", "VAE")], [("AUDIO", "AUDIO")]),
    "CreateVideo":           ([("images", "IMAGE"), ("audio", "AUDIO")], [("VIDEO", "VIDEO")]),
    "SaveVideo":             ([("video", "VIDEO")], []),
    "SaveAudio":             ([("audio", "AUDIO")], []),
    "MarkdownNote":          ([], []),
}

# widget names per class, in order — needed so link-fed widgets carry {"widget": ...}
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "PrimitiveStringMultiline": ["value"],
    "PrimitiveFloat": ["value"],
    "StringConcatenate": ["string_a", "string_b", "delimiter"],
    "ResolutionSelector": ["aspect_ratio", "megapixels", "multiple"],
    "ComfyMathExpression": ["expression"],
    "LoadImage": ["image"],
    "MiniMaxH3ImageToVideo": ["prompt", "width", "height", "length"],
    "MiniMaxH3SigmaShift": ["shift_video", "shift_audio"],
    "KSamplerSelect": ["sampler_name"],
    "BasicScheduler": ["scheduler", "steps", "denoise"],
    "RandomNoise": ["noise_seed"],
    "CreateVideo": ["fps", "bit_depth"],
    "SaveVideo": ["filename_prefix", "format", "codec"],
    "SaveAudio": ["filename_prefix"],
    "MarkdownNote": ["text"],
}

VISUAL = """integrated_multimodal_description: [Shot 1] Cinematic medium close-up, shallow depth of field, slow push-in. A middle-aged lighthouse keeper in a weathered yellow oilskin coat stands on a wet stone balcony at dusk, both hands gripping a rusted iron railing. Sea spray drifts across the frame and rain beads on the coat. The keeper turns to face the lens and speaks. <d>[English] The storm is coming. Secure the boats.</d> [Shot 2] At 00:03.500, cut to a low-angle wide shot of the lighthouse against a bruised purple sky, its lamp sweeping a hard beam through the rain as waves detonate against the rocks below."""

SOUNDSCAPE = """overall_soundscape: Heavy rain drumming on oilskin fabric, deep rhythmic ocean swells breaking against rock far below, gusting wind whistling past the iron railing, and a low metallic groan from the lighthouse structure. On the cut, the perspective opens out: the rain thins into ambient hiss and the wave impacts become broader and more distant."""

MUSIC = """non_diegetic_music: Sparse, tense orchestral underscore - a sustained low cello drone beneath a distant solo violin line, slow build, no percussion, swelling slightly on the cut to the wide shot."""

NOTE_MAIN = """# MiniMax-H3 音影同步生成 — 進階工作流

H3 一次同時生成**影像 + 32kHz 立體聲音訊**（對白／環境音／配樂），不是先出片再配音。

## 為什麼不用官方模板

| 項目 | 官方模板 | 這份 |
|---|---|---|
| Text encoder | `nvfp4_awq` | `int8_convrot` |
| Prompt | 單一自由文字框 | H3-Context-IR 三段式結構 |
| Sigma shift | 沒有暴露 | `MiniMaxH3SigmaShift` 可調 |
| 音訊 | 只有混流進 mp4 | 另存獨立音軌 |

**nvfp4 是 Blackwell（RTX 50 系）專用。** 在 30/40 系上 ComfyUI 會回報
`emulated ops: ... nvfp4` 走模擬路徑；`int8_convrot` 才是 `Native ops`。

## 三段式 Prompt（最重要）

MiniMax 官方 README 明講 H3-Context-IR「對最終品質至關重要」，但那套前處理系統
沒有開源。這裡用三個欄位重建它的輸出格式 —— **開頭的 key 名稱請保留**：

- `integrated_multimodal_description:` — 鏡頭語言、主體外觀、動作、時間點
- `overall_soundscape:` — 畫面內發生的聲音（環境、動作音）
- `non_diegetic_music:` — 配樂（畫面外的音樂）

### 對白語法
`<d>[English] Secure the boats.</d>` — 會生成對應口型與人聲。
支援 11 種語言：Arabic / Chinese / English / French / German / Italian /
Japanese / Korean / Portuguese / Russian / Spanish。

### 多鏡頭
用 `[Shot 1]` `[Shot 2]` 分段，並以 `At 00:03.500,` 標時間點。"""

NOTE_LIMITS = """## 硬性限制（踩到會被靜默修正）

### 幀數必須落在 17k+5 網格
節點內部會自動往上進位到 `n % 17 == 5`：
`5, 22, 39, 56, 73, 90, 107, 124, 141, ... 362`

| 秒數 | 幀數 |
|---|---|
| 5s | 124 |
| 7s | 175 |
| 10s | 243 |
| 15s | 362 |

**訓練範圍約 124–362 幀（5–15 秒）**，超出未經驗證。
上面的 Math Expression 已自動換算，改「秒數」即可。

### 畫布規則
- 長寬都必須是 **32 的倍數**
- 短邊原生 768，面積上限 768×1344 ≈ 1.03 MP
- 超過上限會被等比縮回

### 取樣
- 這是 **CFG-distilled** 權重 → 用 `BasicGuider`（無 CFG），**不要**接 CFGGuider
- `res_multistep` + `simple`，20 步是官方預設
- `shift_video` 12.0 / `shift_audio` 3.0 是模型預設值；
  兩者會被一起送進 DiT，動到 video 那個就等於改動整條排程"""

NOTE_PERF = """## RTX 3090 24GB 實測

本機實跑數據（832×480 / 124 幀 / 20 步）：

| 項目 | 數值 |
|---|---|
| 耗時 | **9.4 分鐘** |
| VRAM 峰值 | 20.2 / 24 GB |
| 系統 RAM 剩餘 | 1.6 / 31.8 GB |
| 產出 | h264 832×480@24fps + AAC 32kHz 立體聲 5.167s |

### 解析度階梯
`megapixels` 調這個值，`multiple_of` 保持 32：

| MP | 16:9 輸出 | 狀態 |
|---|---|---|
| 0.4 | 832×480 | **已實測通過** |
| 0.6 | 1024×576 | 未測，VRAM 應仍可 |
| 1.0 | 1344×768 | 模型原生畫布，24GB 可能不足 |

### 記憶體注意
Text encoder 25.3 GB + DiT 20.0 GB = 45 GB，遠超 32 GB 系統記憶體。
啟動請加 `--disable-smart-memory`，讓 ComfyUI 在兩者之間確實釋放。
跑之前把瀏覽器以外的大型程式關掉。

### 可選的提速
啟動日誌若出現
`WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations`
代表 `comfy_kitchen` 的 cuda / triton 後端被停用，int8 走 eager 反量化。
升級到 cu130 版 torch 可啟用融合 kernel，但需連帶重裝
sageattention / xformers，風險自負。"""

NOTE_MODES = """## 三種模式怎麼切

### T2VA — 純文字生成（目前狀態）
兩個 Load Image 節點保持 **靜音（Ctrl+M）**。

### FL2VA — 首/尾幀生成
- 只解開 `first_frame` → 首幀延伸
- 只解開 `last_frame` → 倒推到該幀
- 兩個都解開 → 首尾幀之間補間

注意首幀是**直接拉伸**到畫布（不裁切），尾幀是**等比裁切**。
所以首幀圖的長寬比最好先對齊你設定的輸出比例。

### Ref2VA — 多模態參考（需換模型）
把 `UNETLoader` 換成 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`，
並改用 `MiniMaxH3ReferenceToVideo` 節點。它吃：
- 圖片 ≤ 9 張 → prompt 裡用 `<Picture 1>` 指涉
- 影片 ≤ 3 段（2–15 秒）→ `<Video 1>`
- 音訊 ≤ 3 段 → `<Audio 1>`（可做音色克隆）
- 全部檔案總數 ≤ 12

這是 H3 最強的模式：角色一致性、動作參考、聲音克隆都在這裡。
`ref_image_size` 選 `max` 身份保真度最高，但參考 token 每步都要算，會慢好幾倍。"""


def build():
    """Return (nodes, links, api) — nodes/links in UI format, api in API format."""
    N = []          # node dicts (UI)
    L = []          # link records
    api = {}
    state = {"nid": 0, "lid": 0}

    def node(cls, pos, widgets=None, conns=None, size=None, mode=0, title=None):
        """conns: {input_name: (src_node_dict, output_index)}"""
        state["nid"] += 1
        nid = state["nid"]
        widgets = list(widgets or [])
        conns = conns or {}
        ins, outs = SLOTS[cls]
        wnames = WIDGETS.get(cls, [])

        ui_inputs = []
        api_inputs = {}
        for name, typ in ins:
            entry = {"name": name, "type": typ, "link": None}
            if name in wnames:
                entry["widget"] = {"name": name}
            if name in conns:
                src, slot = conns[name]
                state["lid"] += 1
                lid = state["lid"]
                entry["link"] = lid
                L.append([lid, src["id"], slot, nid, len(ui_inputs), typ])
                src["outputs"][slot]["links"].append(lid)
                # A muted upstream node (mode 2) contributes nothing at runtime;
                # the API graph has no "mute", so the optional input is dropped.
                if src["mode"] == 0:
                    api_inputs[name] = [str(src["id"]), slot]
            ui_inputs.append(entry)

        for i, wn in enumerate(wnames):
            if wn in conns or i >= len(widgets):
                continue
            api_inputs[wn] = widgets[i]

        d = {
            "id": nid, "type": cls, "pos": list(pos),
            "size": list(size or [300, 100]), "flags": {}, "order": nid - 1,
            "mode": mode,
            "inputs": ui_inputs,
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": cls},
            "widgets_values": widgets,
        }
        if title:
            d["title"] = title
        N.append(d)
        if cls != "MarkdownNote" and mode == 0:
            api[str(nid)] = {"class_type": cls, "inputs": api_inputs}
        return d

    # ---- notes
    node("MarkdownNote", (-2260, 40), [NOTE_MAIN], size=[620, 900])
    node("MarkdownNote", (-2260, 980), [NOTE_LIMITS], size=[620, 780])
    node("MarkdownNote", (-1600, 980), [NOTE_PERF], size=[600, 780])
    node("MarkdownNote", (-980, 980), [NOTE_MODES], size=[600, 780])

    # ---- loaders
    unet = node("UNETLoader", (-1600, 40),
                ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"],
                size=[600, 90], title="DiT — fl2va (T2VA / 首尾幀)")
    clip = node("CLIPLoader", (-1600, 180),
                ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "default"],
                size=[600, 110], title="H3-Encoder — Qwen3-VL-32B @ layer50 (int8, Ampere 原生)")
    vvae = node("VAELoader", (-1600, 340), ["minimax_h3_video_vae_fp16.safetensors"],
                size=[600, 60], title="Visual VAE (f16t4d24)")
    avae = node("VAELoader", (-1600, 440), ["minimax_h3_audio_vae_fp32.safetensors"],
                size=[600, 60], title="Audio VAE (32kHz 立體聲)")

    # ---- H3-Context-IR prompt composer
    p_vis = node("PrimitiveStringMultiline", (-1600, 560), [VISUAL],
                 size=[600, 260], title="① 畫面 — integrated_multimodal_description")
    p_snd = node("PrimitiveStringMultiline", (-980, 560), [SOUNDSCAPE],
                 size=[600, 180], title="② 環境音 — overall_soundscape")
    p_mus = node("PrimitiveStringMultiline", (-980, 770), [MUSIC],
                 size=[600, 160], title="③ 配樂 — non_diegetic_music")

    cat1 = node("StringConcatenate", (-330, 560), ["", "", "\n"],
                {"string_a": (p_vis, 0), "string_b": (p_snd, 0)}, size=[280, 100])
    cat2 = node("StringConcatenate", (-330, 700), ["", "", "\n"],
                {"string_a": (cat1, 0), "string_b": (p_mus, 0)}, size=[280, 100],
                title="組成 H3-Context-IR")

    # ---- geometry / duration
    res = node("ResolutionSelector", (-330, 40), ["16:9 (Widescreen)", 0.4, 32],
               size=[280, 110], title="解析度 (0.4MP=832x480 已實測)")
    secs = node("PrimitiveFloat", (-330, 200), [5.0], size=[280, 60], title="秒數")
    frames = node("ComfyMathExpression", (-330, 300),
                  ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"],
                  {"values.a": (secs, 0)}, size=[280, 130], title="秒 → 幀 (17k+5 網格)")

    # ---- optional keyframes (muted = T2VA)
    ff = node("LoadImage", (-330, 860), ["example.png", "image"], size=[280, 320],
              mode=2, title="首幀 (靜音中 — Ctrl+M 啟用)")
    lf = node("LoadImage", (-330, 1210), ["example.png", "image"], size=[280, 320],
              mode=2, title="尾幀 (靜音中 — Ctrl+M 啟用)")

    # ---- conditioning + latent
    i2v = node("MiniMaxH3ImageToVideo", (30, 40), ["", 832, 480, 124],
               {"clip": (clip, 0), "vae": (vvae, 0), "prompt": (cat2, 0),
                "width": (res, 0), "height": (res, 1), "length": (frames, 1),
                "first_frame": (ff, 0), "last_frame": (lf, 0)},
               size=[360, 260])

    shift = node("MiniMaxH3SigmaShift", (30, 340), [12.0, 3.0],
                 {"model": (unet, 0)}, size=[360, 90],
                 title="Sigma Shift — video/audio 排程")

    guider = node("BasicGuider", (440, 40), [],
                  {"model": (shift, 0), "conditioning": (i2v, 0)}, size=[280, 60])
    sampler = node("KSamplerSelect", (440, 140), ["res_multistep"], size=[280, 60])
    sched = node("BasicScheduler", (440, 240), ["simple", 20, 1.0],
                 {"model": (shift, 0)}, size=[280, 110])
    noise = node("RandomNoise", (440, 390), [0, "randomize"], size=[280, 80])

    samp = node("SamplerCustomAdvanced", (770, 40), [],
                {"noise": (noise, 0), "guider": (guider, 0), "sampler": (sampler, 0),
                 "sigmas": (sched, 0), "latent_image": (i2v, 1)}, size=[280, 140])

    # ---- decode + save
    dv = node("VAEDecode", (1090, 40), [], {"samples": (samp, 0), "vae": (vvae, 0)},
              size=[240, 60])
    da = node("VAEDecodeAudio", (1090, 140), [], {"samples": (samp, 0), "vae": (avae, 0)},
              size=[240, 60])
    cv = node("CreateVideo", (1090, 240), [24, 8],
              {"images": (dv, 0), "audio": (da, 0)}, size=[240, 110])
    node("SaveVideo", (1370, 40), ["video/MiniMax_H3", "auto", "auto"],
         {"video": (cv, 0)}, size=[400, 380])
    node("SaveAudio", (1370, 460), ["audio/MiniMax_H3"],
         {"audio": (da, 0)}, size=[400, 120], title="獨立音軌 (flac) — 方便重剪")

    return N, L, api, state


def main():
    N, L, api, state = build()
    ui = {
        "id": "minimax-h3-av-advanced",
        "revision": 0,
        "last_node_id": state["nid"],
        "last_link_id": state["lid"],
        "nodes": N,
        "links": L,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [2400, 60]}},
        "version": 0.4,
    }
    out_dir = os.environ.get(
        "H3_WORKFLOW_DIR", os.path.join(REPOSITORY_ROOT, "example_workflows"))
    os.makedirs(out_dir, exist_ok=True)
    ui_path = os.path.join(out_dir, "MiniMax-H3_AV_Advanced.json")
    with open(ui_path, "w", encoding="utf-8") as f:
        json.dump(ui, f, ensure_ascii=False, indent=1)
    api_path = os.environ.get("H3_API_JSON",
                              os.path.join(HERE, "h3_advanced_api.json"))
    with open(api_path, "w", encoding="utf-8") as f:
        json.dump(api, f, ensure_ascii=False, indent=1)
    print("UI  ->", ui_path)
    print("API ->", api_path)
    print("nodes=%d links=%d api_nodes=%d" % (len(N), len(L), len(api)))


if __name__ == "__main__":
    main()
