"""Read the prompt expander's reply (pure, no ComfyUI imports).

The official expander instruction asks for one JSON object {"expanded_caption": ..., "aspect_ratio": "W:H"}
(the edit instruction calls the second key "target_aspect_ratio"). Models sometimes wrap it in code
fences or add text around it, so the first decodable object with a non-empty caption wins. Nothing is
guessed: an unreadable reply gives an empty caption and an empty ratio plus the reason.
"""

import json
import math
import re

_RATIO = re.compile(r"^\s*(\d{1,5})\s*[:x/]\s*(\d{1,5})\s*$")


def normalize_ratio(text: str) -> str:
    """'16:9', '1920x1080', '4/3' -> reduced 'W:H'; anything else -> ''."""
    m = _RATIO.match(text or "")
    if not m:
        return ""
    w, h = int(m.group(1)), int(m.group(2))
    if w == 0 or h == 0:
        return ""
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def parse_reply(text: str) -> tuple[str, str, str]:
    """-> (caption, ratio, problem). problem is '' when a caption was found."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        caption = obj.get("expanded_caption")
        if isinstance(caption, str) and caption.strip():
            ratio = obj.get("aspect_ratio") or obj.get("target_aspect_ratio") or ""
            return caption.strip(), normalize_ratio(str(ratio)), ""
    if not (text or "").strip():
        return (
            "",
            "",
            "the model returned no text (it may have run out of tokens while thinking)",
        )
    return "", "", "the reply has no JSON object with an expanded_caption"
