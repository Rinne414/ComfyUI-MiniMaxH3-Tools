"""The two READMEs must link each other, resolve their links and agree on numbers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
READMES = ("README.md", "README_zh.md")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{2,4} (.+)$", re.MULTILINE)
TABLE_ROW = re.compile(r"^\|.*\|$", re.MULTILINE)
NUMBER = re.compile(r"\d+(?:[.:]\d+)*")


def _text(name: str) -> str:
    return (REPOSITORY_ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", READMES)
def test_relative_links_and_images_resolve(name: str) -> None:
    text = _text(name)
    for target in LINK.findall(text) + IMAGE.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (REPOSITORY_ROOT / target.split("#")[0]).exists(), (name, target)


def test_the_readmes_link_each_other_at_the_top() -> None:
    assert "(README_zh.md)" in _text("README.md").split("\n\n")[1]
    assert "(README.md)" in _text("README_zh.md").split("\n\n")[1]


@pytest.mark.parametrize("name", READMES)
def test_readme_has_no_image_placeholders(name: str) -> None:
    text = _text(name)
    assert not IMAGE.findall(text)
    assert "TODO" not in text


def test_the_readmes_have_the_same_section_count() -> None:
    assert len(HEADING.findall(_text("README.md"))) == len(HEADING.findall(_text("README_zh.md")))


def test_benchmark_numbers_are_identical_in_both_languages() -> None:
    def comparison_numbers(name: str) -> list[list[str]]:
        rows = [
            row
            for row in TABLE_ROW.findall(_text(name))
            if "DiffSynth NF4 |" in row or "int8_convrot" in row and "GB" in row
        ]
        return [NUMBER.findall(row.replace("int8_convrot", "")) for row in rows]

    english, chinese = comparison_numbers("README.md"), comparison_numbers("README_zh.md")
    assert english and english == chinese


def test_the_readme_names_the_example_workflow_folder() -> None:
    for name in READMES:
        assert "example_workflows/" in _text(name)
        assert "`workflows/`" not in _text(name)
