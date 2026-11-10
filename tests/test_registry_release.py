"""The publish workflow's registry check must pass only for the latest active version."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.registry_release import (
    ACTIVE_STATUS,
    decide_publish,
    describe_problem,
    run_decide,
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


def _workflow(name: str) -> dict:
    import yaml

    return yaml.safe_load((REPOSITORY_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _python_versions(job: dict) -> list[str]:
    return [
        str(step["with"]["python-version"])
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python")
    ]


def test_registry_check_runs_in_its_own_job_on_python_3_11() -> None:
    jobs = _workflow("publish.yml")["jobs"]
    publish, verify = jobs["publish-node"], jobs["verify-release"]
    assert "publish-node" in verify["needs"]
    assert all("registry_release" not in str(step) for step in publish["steps"])
    assert any("registry_release.py" in str(step.get("run", "")) for step in verify["steps"])
    assert _python_versions(verify) == ["3.11"]
    assert "secrets." not in str(verify)


def test_the_script_only_needs_the_python_the_verify_job_sets() -> None:
    """tomllib arrived in 3.11, so the verify job and CI must not use anything older."""
    (version,) = _python_versions(_workflow("publish.yml")["jobs"]["verify-release"])
    ci_versions = _python_versions(_workflow("ci.yml")["jobs"]["verify"])
    assert ci_versions == [version]
    assert tuple(int(part) for part in version.split(".")) >= (3, 11)


def test_a_version_already_on_the_registry_is_not_published() -> None:
    payload = [{"version": "0.1.0", "status": ACTIVE_STATUS}]
    should_publish, reason = decide_publish("0.1.0", payload)
    assert should_publish is False
    assert "already on the registry" in reason


def test_a_new_version_is_published() -> None:
    payload = [{"version": "0.1.0"}, {"version": "0.1.1"}]
    assert decide_publish("0.2.0", payload)[0] is True


def test_a_node_the_registry_does_not_know_yet_is_published() -> None:
    assert decide_publish("0.1.0", [])[0] is True


@pytest.mark.parametrize("payload", [None, {}, {"message": "error"}, "oops", [1, 2]])
def test_an_unexpected_answer_is_an_error_not_a_reason_to_publish(payload: object) -> None:
    with pytest.raises(ValueError):
        decide_publish("0.1.0", payload)


def test_decide_writes_the_job_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        "scripts.registry_release.fetch_versions", lambda _: [{"version": "0.1.0"}]
    )
    assert run_decide("minimax-h3-tools", "0.1.0") == 0
    assert output.read_text(encoding="utf-8") == "publish=false\n"
    monkeypatch.setattr("scripts.registry_release.fetch_versions", lambda _: [])
    assert run_decide("minimax-h3-tools", "0.1.0") == 0
    assert output.read_text(encoding="utf-8") == "publish=false\npublish=true\n"


def test_decide_fails_loudly_and_publishes_nothing_on_an_api_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def broken(_: str) -> list:
        raise OSError("HTTP Error 503")

    monkeypatch.setattr("scripts.registry_release.fetch_versions", broken)
    assert run_decide("minimax-h3-tools", "0.1.0") == 1
    assert "::error::" in capsys.readouterr().out
    assert not output.exists()


def test_publish_and_verify_jobs_wait_for_the_version_check() -> None:
    jobs = _workflow("publish.yml")["jobs"]
    assert jobs["publish-node"]["needs"] == "check-version"
    assert "publish == 'true'" in jobs["publish-node"]["if"]
    assert "check-version" in jobs["verify-release"]["needs"]
    assert "publish == 'true'" in jobs["verify-release"]["if"]
    assert "secrets." not in str(jobs["check-version"])
