"""Check that the prompt's picture references match the pictures that are connected.

The official MiniMax-H3-Image templates and ComfyUI's own node text write the references as
``<Picture 1>``, ``<Picture 2>``, ... (angle brackets, one space, capital P). This check also reads
case and spacing variants inside the brackets (``<picture 2>``, ``<Picture2>``, ``< Picture  2 >``).
A bare ``Picture 2`` without brackets is not counted: in ordinary prose it is not a reference.

The check only describes; it never blocks a run. No ComfyUI imports.
"""

from __future__ import annotations

import re

PICTURE_TAG = re.compile(r"<\s*picture\s*(\d+)\s*>", re.IGNORECASE)


def referenced_pictures(prompt: str) -> list[int]:
    """The distinct picture numbers a prompt refers to, in ascending order."""
    return sorted({int(number) for number in PICTURE_TAG.findall(prompt or "")})


def _pictures(count: int) -> str:
    return "1 picture is" if count == 1 else "%d pictures are" % count


def _names(numbers: list[int]) -> str:
    return " and ".join(
        ", ".join("Picture %d" % number for number in numbers).rsplit(", ", 1)
    )


def picture_count_note(prompt: str, connected: int) -> str:
    """One plain sentence when the references and the connected pictures disagree, else ''.

    A single connected picture that the prompt does not name is fine: ``image_1`` is the picture to
    edit, so a short instruction like "make the sky red" needs no tag.
    """
    referenced = referenced_pictures(prompt)
    missing = [number for number in referenced if number > connected]
    if missing:
        have = (
            "no pictures are connected"
            if connected == 0
            else "only %s connected" % _pictures(connected)
        )
        return "prompt mentions %s but %s" % (_names(missing), have)
    unnamed = [number for number in range(1, connected + 1) if number not in referenced]
    if not unnamed or (connected == 1 and not referenced):
        return ""
    plural = "pictures" if connected != 1 else "picture"
    return "%d %s connected but the prompt never mentions %s" % (
        connected,
        plural,
        _names(unnamed),
    )


def append_note(report: str, prompt: str, connected: int) -> str:
    """The report with the picture-count sentence added when there is one."""
    note = picture_count_note(prompt, connected)
    return "%s; %s" % (report, note) if note else report


def connected_pictures(images: dict | None) -> list:
    """The connected pictures in order (image_1, image_2, ..., numerically), skipping empty slots."""
    names = sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1]))
    return [images[name] for name in names if images[name] is not None]
