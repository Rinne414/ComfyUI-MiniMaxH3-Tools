"""The startup log line must say what was checked and must never break the import."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import startup_status  # noqa: E402


def _fake_version(version: str | None):
    def lookup(name: str) -> str | None:
        assert name == "bitsandbytes"
        if version == "missing":
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    return lookup


def test_reports_missing_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("missing"))
    assert startup_status.nf4_status() == "NF4 loaders: bitsandbytes not installed"


def test_reports_what_was_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.0"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    assert startup_status.nf4_status() == (
        "NF4 loaders: bitsandbytes 0.50.0 found, CUDA device found"
    )


def test_reports_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.0"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: False)
    assert startup_status.nf4_status() == (
        "NF4 loaders: bitsandbytes 0.50.0 found, no CUDA device found"
    )


def test_flags_a_bitsandbytes_older_than_the_tested_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.45.2"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    status = startup_status.nf4_status()
    assert "bitsandbytes 0.45.2 found (0.50.0 or newer is tested)" in status


def test_accepts_local_version_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.1+cu130"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    assert "bitsandbytes 0.50.1+cu130 found, CUDA device found" in startup_status.nf4_status()


def test_survives_a_package_without_a_version_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version(None))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    assert startup_status.nf4_status() == (
        "NF4 loaders: bitsandbytes found, CUDA device found"
    )


def test_logs_one_line_through_the_logging_module(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("missing"))
    with caplog.at_level(logging.INFO, logger=startup_status.LOGGER.name):
        startup_status.log_startup_status()
    assert [record.getMessage() for record in caplog.records] == [
        "MiniMax-H3 Tools: NF4 loaders: bitsandbytes not installed"
    ]


def test_a_failing_probe_never_raises_and_is_logged_at_debug(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(_: str) -> str:
        raise RuntimeError("corrupt metadata")

    monkeypatch.setattr(importlib.metadata, "version", explode)
    with caplog.at_level(logging.DEBUG, logger=startup_status.LOGGER.name):
        startup_status.log_startup_status()
    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
    assert caplog.records[0].exc_info is not None


def test_importing_the_pack_survives_a_failing_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.smoke_registration import load_extension

    def explode(_: str) -> str:
        raise RuntimeError("corrupt metadata")

    monkeypatch.setattr(importlib.metadata, "version", explode)
    module = load_extension(Path(__file__).resolve().parents[1])
    assert len(module.NODES) == 12


def test_importing_the_pack_logs_the_status(caplog: pytest.LogCaptureFixture) -> None:
    from scripts.smoke_registration import load_extension

    with caplog.at_level(logging.INFO):
        load_extension(Path(__file__).resolve().parents[1])
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("MiniMax-H3 Tools: NF4 loaders: ") for m in messages)
