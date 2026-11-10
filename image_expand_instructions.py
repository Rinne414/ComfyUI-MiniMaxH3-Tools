"""Find the prompt expander's instructions in the user's own ComfyUI installation.

The expander uses the instruction text of the official MiniMax-H3-Image workflow templates. That text
belongs to Comfy-Org and is not part of this pack: it is read at run time from the template files that
ComfyUI's template package (comfyui-workflow-templates-json) installs, or from a copy of the template
saved in the user's workflows folder. When no usable template is found the caller passes the user's
prompt through unchanged and shows ``InstructionsNotFound.args[0]``.

No ComfyUI imports, so this can be tested with a small fake template.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

TEMPLATE_FILES = {
    "t2i": "image_minimax_h3_image_t2i.json",
    "edit": "image_minimax_h3_image_edit.json",
}
PACKAGE_NAME = "comfyui_workflow_templates_json"
SYSTEM_TITLE = "system_prompt"
TASK_TITLE = "task_prompt"
TASK_PLACEHOLDER = "{input text}"
STRING_NODE_TYPE = "PrimitiveStringMultiline"

NOT_FOUND = (
    "The official MiniMax-H3-Image template that holds the expander's instructions was not found on "
    "this PC, so your prompt is used unchanged. Update ComfyUI to 0.45.0 or later and run "
    "'python -m pip install -r requirements.txt' in the ComfyUI folder (it installs the template "
    "package), then restart ComfyUI."
)
UNRECOGNISED = (
    "The MiniMax-H3-Image template found at {path} has a layout this node does not recognise, "
    "so your prompt is used unchanged. Update this node pack."
)


class Instructions(NamedTuple):
    system: str
    task: str
    source: str


class InstructionsNotFound(Exception):
    """Raised with a plain-language reason when no usable template exists."""


def package_directories() -> list[Path]:
    """The ``templates`` folders of every installed comfyui_workflow_templates_json."""
    found: list[Path] = []
    spec = importlib.util.find_spec(PACKAGE_NAME)
    if spec is not None:
        for location in spec.submodule_search_locations or []:
            found.append(Path(location) / "templates")
    for entry in sys.path:
        candidate = Path(entry or ".") / PACKAGE_NAME / "templates"
        if candidate not in found:
            found.append(candidate)
    return [folder for folder in found if folder.is_dir()]


def _string_nodes(template: dict) -> Iterable[dict]:
    yield from template.get("nodes", [])
    for subgraph in template.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _node_text(node: dict) -> str:
    values = node.get("widgets_values") or [""]
    return values[0] if isinstance(values[0], str) else ""


def extract(template: object) -> tuple[str, str] | None:
    """(system prompt, task prompt) of a template, or None when it has no usable pair."""
    if not isinstance(template, dict):
        return None
    system = task = ""
    for node in _string_nodes(template):
        if node.get("type") != STRING_NODE_TYPE:
            continue
        title = str(node.get("title") or "")
        if SYSTEM_TITLE in title:
            system = system or _node_text(node)
        elif TASK_TITLE in title:
            task = task or _node_text(node)
    if system.strip() and TASK_PLACEHOLDER in task:
        return system, task
    return None


@functools.lru_cache(maxsize=8)
def _read_template(path: str, modified_ns: int) -> dict | None:
    del modified_ns  # part of the cache key: an updated file is read again
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _candidate_files(
    kind: str, directories: Iterable[Path], extra: Iterable[Path]
) -> list[Path]:
    name = TEMPLATE_FILES[kind]
    files = [folder / name for folder in directories]
    for folder in extra:
        if folder.is_dir():
            # a copy saved from the Templates browser may be renamed; the kind is in the file name
            files.extend(sorted(folder.rglob(f"*minimax_h3_image*{kind}*.json")))
    return [path for path in files if path.is_file()]


def find_instructions(
    kind: str,
    directories: Iterable[Path] | None = None,
    extra_directories: Iterable[Path] = (),
) -> Instructions:
    """Instructions for ``kind`` ('t2i' or 'edit'); raises InstructionsNotFound with a plain reason."""
    if kind not in TEMPLATE_FILES:
        raise ValueError("kind must be 't2i' or 'edit', got %r" % kind)
    folders = package_directories() if directories is None else list(directories)
    unrecognised = ""
    for path in _candidate_files(kind, folders, extra_directories):
        template = _read_template(str(path), os.stat(path).st_mtime_ns)
        pair = extract(template)
        if pair is not None:
            return Instructions(pair[0], pair[1], str(path))
        unrecognised = unrecognised or UNRECOGNISED.format(path=path.name)
    raise InstructionsNotFound(unrecognised or NOT_FOUND)
