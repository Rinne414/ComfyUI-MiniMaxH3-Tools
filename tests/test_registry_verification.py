"""The publish workflow's registry check must pass only for the latest active version."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_registry_release import (
    ACTIVE_STATUS,
    describe_problem,
    read_pack,
    wait_for_release,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _entry(version: str, status: str = ACTIVE_STATUS) -> dict:
    return {"latest_version": {"version": version, "status": status}}


def test_read_pack_matches_the_registry_id_and_version() -> None:
    node_id, version = read_pack(REPOSITORY_ROOT / "pyproject.toml")
    assert node_id == "minimax-h3-tools"
    assert version.count(".") == 2


def test_the_latest_active_version_passes() -> None:
    assert describe_problem(_entry("0.2.0"), "0.2.0") is None


def test_an_older_latest_version_fails() -> None:
    assert "0.1.0" in (describe_problem(_entry("0.1.0"), "0.2.0") or "")


def test_a_pending_version_fails() -> None:
    problem = describe_problem(_entry("0.2.0", "NodeVersionStatusPending"), "0.2.0")
    assert problem and "NodeVersionStatusPending" in problem


def test_an_entry_without_versions_fails() -> None:
    assert describe_problem({}, "0.2.0")


def test_polling_waits_until_the_new_version_appears() -> None:
    answers = iter([_entry("0.1.0"), _entry("0.1.0"), _entry("0.2.0")])
    sleeps: list[float] = []
    result = wait_for_release(
        "minimax-h3-tools",
        "0.2.0",
        fetch=lambda _: next(answers),
        sleep=sleeps.append,
        timeout_seconds=100,
        poll_seconds=10,
    )
    assert result is None
    assert sleeps == [10, 10]


def test_polling_gives_up_with_the_last_problem() -> None:
    sleeps: list[float] = []
    result = wait_for_release(
        "minimax-h3-tools",
        "0.2.0",
        fetch=lambda _: _entry("0.1.0"),
        sleep=sleeps.append,
        timeout_seconds=30,
        poll_seconds=10,
    )
    assert result and "0.1.0" in result
    assert len(sleeps) == 3


def test_polling_survives_network_errors_and_reports_them() -> None:
    def broken(_: str) -> dict:
        raise OSError("offline")

    result = wait_for_release(
        "minimax-h3-tools",
        "0.2.0",
        fetch=broken,
        sleep=lambda _: None,
        timeout_seconds=20,
        poll_seconds=10,
    )
    assert result and "offline" in result
