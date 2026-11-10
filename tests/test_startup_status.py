"""The startup log line must say whether the NF4 loaders can run."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import startup_status  # noqa: E402


def _fake_version(version: str | None):
    def lookup(name: str) -> str:
        assert name == "bitsandbytes"
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    return lookup


def test_reports_missing_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version(None))
    assert startup_status.nf4_status() == (
        "NF4 loaders unavailable (bitsandbytes not installed)"
    )


def test_reports_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.0"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: False)
    assert "no NVIDIA CUDA device" in startup_status.nf4_status()
    assert startup_status.nf4_status().startswith("NF4 loaders unavailable")


def test_reports_available_with_the_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.0"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    assert startup_status.nf4_status() == "NF4 loaders available (bitsandbytes 0.50.0)"


def test_flags_a_bitsandbytes_older_than_the_tested_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.45.2"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    status = startup_status.nf4_status()
    assert status.startswith("NF4 loaders available")
    assert "0.45.2" in status and "0.50.0 or newer" in status


def test_accepts_local_version_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version("0.50.1+cu130"))
    monkeypatch.setattr(startup_status, "_cuda_available", lambda: True)
    assert startup_status.nf4_status() == "NF4 loaders available (bitsandbytes 0.50.1+cu130)"


def test_logs_one_line_through_the_logging_module(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", _fake_version(None))
    with caplog.at_level(logging.INFO, logger=startup_status.LOGGER.name):
        startup_status.log_startup_status()
    assert [record.getMessage() for record in caplog.records] == [
        "MiniMax-H3 Tools: NF4 loaders unavailable (bitsandbytes not installed)"
    ]


def test_importing_the_pack_logs_the_status(caplog: pytest.LogCaptureFixture) -> None:
    from pathlib import Path

    from scripts.smoke_registration import load_extension

    with caplog.at_level(logging.INFO):
        load_extension(Path(__file__).resolve().parents[1])
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("MiniMax-H3 Tools: NF4 loaders ") for m in messages)
