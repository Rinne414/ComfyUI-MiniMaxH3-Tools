"""Deterministic prompt linter for MiniMax-H3.

H3's prompt format has strict rules that fail *silently*: a malformed prompt
does not raise, it just produces worse video. LLM-based rewriting is already
well covered by rockerBOO/boo-textgen and MaxKruse/ComfyUI-PromptEnhancer --
this module deliberately does no rewriting and only verifies.

Every rule cites the guide it comes from:
  BASE = docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md   (T2VA / I2VA / FL2VA / L2VA)
  REF  = docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md    (full-reference mode)

Both live in the MiniMaxAI/MiniMax-H3 repo.
"""

import re

from comfy_api.latest import ComfyExtension, io

FPS = 24

# BASE 4.4 / REF 5.4 -- the 11 languages with stable dialogue support.
STABLE_LANGUAGES = {
    "arabic", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
}

BASE_FIELDS = ["integrated_multimodal_description", "overall_soundscape",
               "non_diegetic_music"]
REF_FIELDS = ["subject_definitions", "summary", "retention_analysis",
              "detailed_description", "overall_soundscape", "non_diegetic_music"]

# REF 4.1 / 4.2 -- fixed English values, not free text.
VISUAL_MARKERS = {"fully_preserved", "partially_preserved", "attribute_transfer",
                  "weak_reference"}
AUDIO_MARKERS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

# REF 3 -- task types allowed in the summary prefix.
TASK_TYPES = {"keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference"}

# BASE 4.7 -- things characters can hear are diegetic and must not appear in
# non_diegetic_music.
DIEGETIC_WORDS = ["radio", "phone", "television", " tv ", "live performance",
                  "jukebox", "loudspeaker", "car stereo", "record player"]

DIALOGUE_RE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
DIALOGUE_OPEN_RE = re.compile(r"<d>")
DIALOGUE_CLOSE_RE = re.compile(r"</d>")
LANG_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]")
SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]")
SPEAKER_RE = re.compile(r"\(S(\d+(?:\s*,\s*S?\d+)*)\)")
TIMESTAMP_RE = re.compile(r"At\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?")
PICTURE_RE = re.compile(r"<Picture\s+(\d+)>")
VIDEO_TAG_RE = re.compile(r"<Video\s+(\d+)>")
AUDIO_TAG_RE = re.compile(r"<Audio\s+(\d+)>")
SUBJECT_RE = re.compile(r"<Subject\s+(\d+)>")


def _sections(text, fields):
    """Split on 'field:' labels at line starts; returns {field: body}."""
    out = {}
    positions = []
    for f in fields:
        m = re.search(r"^[ \t]*%s\s*:" % re.escape(f), text, re.MULTILINE)
        if m:
            positions.append((m.start(), m.end(), f))
    positions.sort()
    for i, (_, end, f) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[f] = text[end:stop].strip()
    return out, [f for _, _, f in positions]


def _check_fields(text, mode, warn):
    fields = REF_FIELDS if mode == "ref2va" else BASE_FIELDS
    sections, order = _sections(text, fields)

    for f in fields:
        if f not in sections:
            warn("missing required field '%s:' (%s mode expects %s)"
                 % (f, mode, ", ".join(fields)))

    present = [f for f in fields if f in sections]
    if present != [f for f in order if f in fields]:
        warn("fields are out of order: found %s, expected %s"
             % (" -> ".join(order), " -> ".join(fields)))

    for f, body in sections.items():
        if not body or body.strip() in (":", "-"):
            warn("field '%s:' is empty" % f)
    return sections


def _check_shots(body, duration_seconds, warn):
    """BASE 4.2 -- Shot 1 has no timestamp; later shots need increasing times."""
    shots = [(m.start(), int(m.group(1))) for m in SHOT_RE.finditer(body)]
    if not shots:
        warn("no [Shot N] marker found in the description")
        return

    numbers = [n for _, n in shots]
    if numbers[0] != 1:
        warn("first shot is [Shot %d]; the opening shot must be [Shot 1]" % numbers[0])
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        warn("shot numbers are %s; they must run sequentially from 1 with no gaps "
             "or repeats" % numbers)

    last_ms = None
    for i, (pos, num) in enumerate(shots):
        stop = shots[i + 1][0] if i + 1 < len(shots) else len(body)
        window = body[pos:stop]
        ts = TIMESTAMP_RE.search(window)

        if num == 1:
            if ts and ts.start() < 60:
                warn("[Shot 1] carries a timestamp (%s); the opening shot must not "
                     "have one" % ts.group(0))
            continue

        if ts is None:
            warn("[Shot %d] has no 'At MM:SS.mmm,' cut time; every shot after the "
                 "first needs one" % num)
            continue

        mins, secs = int(ts.group(1)), int(ts.group(2))
        frac = ts.group(3) or "0"
        ms = mins * 60000 + secs * 1000 + int(frac.ljust(3, "0"))
        if ts.group(3) is None:
            warn("[Shot %d] cut time '%s' omits milliseconds; the format is "
                 "MM:SS.mmm" % (num, ts.group(0)))
        if last_ms is not None and ms <= last_ms:
            warn("[Shot %d] cut time %s is not after the previous shot's; cut times "
                 "must strictly increase" % (num, ts.group(0)))
        if duration_seconds > 0 and ms >= duration_seconds * 1000:
            warn("[Shot %d] cut time %s falls at or past the %.3f s clip duration"
                 % (num, ts.group(0), duration_seconds))
        last_ms = ms


def _check_dialogue(text, warn):
    """BASE 4.4 -- <d>[Lang] ...</d>, language tag required and recognized."""
    opens = len(DIALOGUE_OPEN_RE.findall(text))
    closes = len(DIALOGUE_CLOSE_RE.findall(text))
    if opens != closes:
        warn("unbalanced dialogue tags: %d <d> and %d </d>" % (opens, closes))

    for body in DIALOGUE_RE.findall(text):
        lang = LANG_TAG_RE.match(body)
        if lang is None:
            snippet = body.strip()[:40]
            warn("dialogue '<d>%s...' has no [Language] tag" % snippet)
            continue
        name = lang.group(1).strip().lower()
        if name not in STABLE_LANGUAGES:
            warn("dialogue language '[%s]' is outside the 11 with stable support "
                 "(%s); it may still work but is unvalidated"
                 % (lang.group(1).strip(), ", ".join(sorted(STABLE_LANGUAGES))))
        spoken = body[lang.end():].strip()
        if not spoken:
            warn("dialogue tagged [%s] has no spoken content" % lang.group(1).strip())


def _speaker_ids(text):
    ids = set()
    for m in SPEAKER_RE.finditer(text):
        for part in m.group(1).split(","):
            part = part.strip().lstrip("S")
            if part.isdigit():
                ids.add(int(part))
    return ids


def _check_speakers(body, warn):
    """BASE 4.4 -- stable (S1)/(S2) IDs, and dialogue should be attributed."""
    ids = _speaker_ids(body)
    if ids:
        expected = set(range(1, max(ids) + 1))
        missing = sorted(expected - ids)
        if missing:
            warn("speaker IDs jump: found %s but %s never appear; number speakers "
                 "consecutively from S1"
                 % (sorted(ids), ", ".join("S%d" % i for i in missing)))

    # A <d> block with no (Sx) anywhere before it in the same shot is unattributed.
    for m in DIALOGUE_RE.finditer(body):
        preceding = body[:m.start()]
        shot_start = max((s.start() for s in SHOT_RE.finditer(preceding)), default=0)
        if not SPEAKER_RE.search(preceding[shot_start:]):
            warn("dialogue '<d>%s...' has no (Sx) speaker ID earlier in its shot"
                 % m.group(1).strip()[:35])


def _check_voiceover(body, warn):
    """BASE 4.4 -- exact phrase, plus a closed-lips statement right after."""
    for m in re.finditer(r"(?i)\bvoice[\s-]?over\b|\boff[\s-]screen voice\b", body):
        window = body[max(0, m.start() - 80): m.start() + 400]
        if "says in an off-screen voiceover" not in window:
            warn("voiceover near '%s' does not use the required exact phrase "
                 "'says in an off-screen voiceover'" % m.group(0))
        after = body[m.end(): m.end() + 400]
        if not re.search(r"(?i)lips? (remain|stay|are)|mouth (remains|stays|is) closed",
                         after):
            warn("voiceover near '%s' is not followed by a statement that the "
                 "on-screen character's lips remain closed" % m.group(0))


def _check_soundscape(body, description, warn):
    """BASE 4.6 -- 1-4 sentences, no dialogue restated, N/A only for silence."""
    if body.strip() == "N/A":
        return
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s]
    if len(sentences) > 4:
        warn("overall_soundscape has %d sentences; the guide asks for 1-4"
             % len(sentences))
    if DIALOGUE_RE.search(body) or "<d>" in body:
        warn("overall_soundscape contains a <d> dialogue block; dialogue belongs "
             "only in the description")
    for spoken in DIALOGUE_RE.findall(description):
        words = re.sub(r"^\s*\[[^\]]+\]", "", spoken).strip()
        if len(words) > 12 and words.lower() in body.lower():
            warn("overall_soundscape restates dialogue ('%s...'); the guide "
                 "forbids repeating it" % words[:30])


def _check_music(body, warn):
    """BASE 4.7 -- audience-only score; diegetic sources belong elsewhere."""
    if body.strip() == "N/A":
        return
    low = body.lower()
    for word in DIEGETIC_WORDS:
        if word.strip() in low:
            warn("non_diegetic_music mentions '%s', which characters can hear. "
                 "Diegetic sound belongs in the description, not here."
                 % word.strip())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s]
    if len(sentences) > 3:
        warn("non_diegetic_music has %d sentences; the guide asks for 1-3"
             % len(sentences))
    if DIALOGUE_RE.search(body):
        warn("non_diegetic_music contains a <d> dialogue block")


def _check_ref_tags(text, sections, counts, warn):
    """REF 2 -- tags must be defined, numbered without gaps, within supply."""
    limits = {"Picture": counts.get("images", 0), "Video": counts.get("videos", 0),
              "Audio": counts.get("audios", 0)}
    patterns = {"Picture": PICTURE_RE, "Video": VIDEO_TAG_RE, "Audio": AUDIO_TAG_RE}

    for kind, rx in patterns.items():
        used = sorted({int(m.group(1)) for m in rx.finditer(text)})
        if not used:
            continue
        if used[0] != 1 or used != list(range(1, used[-1] + 1)):
            warn("<%s N> numbering has gaps: found %s; number consecutively from 1"
                 % (kind, used))
        limit = limits[kind]
        if limit and used[-1] > limit:
            warn("<%s %d> is referenced but only %d %s input(s) are connected"
                 % (kind, used[-1], limit, kind.lower()))

    defs = sections.get("subject_definitions", "")
    if defs:
        defined = {int(m.group(1)) for m in SUBJECT_RE.finditer(defs)}
        for m in SUBJECT_RE.finditer(text):
            n = int(m.group(1))
            if n not in defined:
                warn("<Subject %d> is used but never defined in subject_definitions" % n)
                break


def _check_ref_sections(sections, warn):
    """REF 3 / 4 -- task-type prefix, fixed markers, no (Sx) in retention."""
    summary = sections.get("summary", "")
    if summary:
        m = re.match(r"\s*\[([^\]]+)\]", summary)
        if m is None:
            warn("summary does not begin with a [task type] prefix (e.g. "
                 "[reference generation + audio reference])")
        else:
            for part in m.group(1).split("+"):
                part = part.strip()
                if part and part not in TASK_TYPES:
                    warn("summary task type '%s' is not one of: %s"
                         % (part, ", ".join(sorted(TASK_TYPES))))

    retention = sections.get("retention_analysis", "")
    if retention:
        if SPEAKER_RE.search(retention):
            warn("retention_analysis contains a (Sx) speaker ID; the guide states "
                 "they must not appear in this section")
        for line in retention.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            marker = line.split(":", 1)[1].strip().split()
            if not marker:
                continue
            token = marker[0].rstrip("-").strip()
            is_audio = line.lstrip().startswith("<Audio")
            allowed = AUDIO_MARKERS if is_audio else VISUAL_MARKERS
            if token not in allowed:
                warn("retention line '%s...' uses marker '%s'; %s entries must use "
                     "one of: %s" % (line[:34], token,
                                     "audio" if is_audio else "visual",
                                     ", ".join(sorted(allowed))))


def lint(text, mode="t2va", duration_seconds=0.0, ref_images=0, ref_videos=0,
         ref_audios=0):
    """Returns a list of warning strings. Empty means clean."""
    warnings = []

    def warn(msg):
        warnings.append(msg)

    if not text or not text.strip():
        return ["prompt is empty"]

    sections = _check_fields(text, mode, warn)
    main_field = "detailed_description" if mode == "ref2va" else BASE_FIELDS[0]
    description = sections.get(main_field, "")

    if description:
        _check_shots(description, duration_seconds, warn)
        _check_speakers(description, warn)
        _check_voiceover(description, warn)

    _check_dialogue(text, warn)

    if "overall_soundscape" in sections:
        _check_soundscape(sections["overall_soundscape"], description, warn)
    if "non_diegetic_music" in sections:
        _check_music(sections["non_diegetic_music"], warn)

    if mode == "ref2va":
        _check_ref_tags(text, sections, {"images": ref_images, "videos": ref_videos,
                                         "audios": ref_audios}, warn)
        _check_ref_sections(sections, warn)
    else:
        stray = (PICTURE_RE.search(text) or VIDEO_TAG_RE.search(text)
                 or SUBJECT_RE.search(text))
        if stray and mode == "t2va":
            warn("prompt uses reference tag '%s' but mode is t2va; those tags only "
                 "carry meaning in ref2va" % stray.group(0))

    return warnings


class MiniMaxH3XPromptLint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XPromptLint",
            display_name="MiniMax H3 Prompt Lint / 提示词检查",
            category="model/conditioning/minimax",
            description=(
                "Check an H3 prompt against MiniMax's documented format rules. "
                "Deterministic -- no LLM, no rewriting. Passes the prompt through "
                "unchanged so it can sit inline in a workflow. / 按 MiniMax 文档规则"
                "检查 H3 提示词；不使用 LLM、不改写，并原样传递提示词。"),
            inputs=[
                io.String.Input("prompt", multiline=True),
                io.Combo.Input("mode", options=["t2va", "fl2va", "ref2va"], default="t2va",
                               tooltip=(
                                   "t2va/fl2va expect the three base fields; ref2va "
                                   "expects the six full-reference fields. / t2va/fl2va "
                                   "需要三个基础字段；ref2va 需要六个完整参考字段。")),
                io.Float.Input("duration_seconds", default=0.0, min=0.0, max=60.0, step=0.1,
                               tooltip=(
                                   "Clip duration, used to flag cut times past the end. "
                                   "0 disables that check. / 片段时长，用于标记超出结尾的"
                                   "切镜时间；0 表示禁用此检查。")),
                io.Int.Input("ref_images", default=0, min=0, max=9,
                             tooltip=(
                                 "How many reference images are connected, so "
                                 "<Picture N> beyond that can be flagged. / 已连接的参考"
                                 "图片数量，用于标记超出范围的 <Picture N>。")),
                io.Int.Input("ref_videos", default=0, min=0, max=3),
                io.Int.Input("ref_audios", default=0, min=0, max=3),
                io.Boolean.Input("raise_on_warning", default=False,
                                 tooltip=(
                                     "Fail the workflow instead of passing the prompt "
                                     "through. Off by default so a lint warning never "
                                     "blocks a render. / 发现警告时让工作流失败，而不是"
                                     "继续传递提示词；默认关闭，避免检查警告阻止渲染。")),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="warnings"),
                io.Boolean.Output(display_name="is_clean"),
            ],
        )

    @classmethod
    def execute(cls, prompt, mode, duration_seconds, ref_images, ref_videos,
                ref_audios, raise_on_warning=False) -> io.NodeOutput:
        warnings = lint(prompt, mode=mode, duration_seconds=duration_seconds,
                        ref_images=ref_images, ref_videos=ref_videos,
                        ref_audios=ref_audios)
        if warnings and raise_on_warning:
            raise ValueError("H3 prompt lint failed:\n" +
                             "\n".join("  - " + w for w in warnings))
        text = ("clean -- no issues found" if not warnings else
                "\n".join("%d. %s" % (i + 1, w) for i, w in enumerate(warnings)))
        return io.NodeOutput(prompt, text, not warnings)


class PromptLintExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3XPromptLint]


async def comfy_entrypoint() -> PromptLintExtension:
    return PromptLintExtension()
