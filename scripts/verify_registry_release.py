"""Fail unless the Comfy Registry lists this pack's version as the latest and active."""

from __future__ import annotations

import json
import sys
import time
import tomllib
import urllib.request
from collections.abc import Callable
from pathlib import Path

REGISTRY_URL = "https://api.comfy.org/nodes/%s"
ACTIVE_STATUS = "NodeVersionStatusActive"
POLL_SECONDS = 20
TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 30


def read_pack(pyproject: Path) -> tuple[str, str]:
    """Return ``(registry_id, version)`` from pyproject.toml."""
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["name"], project["version"]


def fetch_registry_entry(node_id: str) -> dict:
    request = urllib.request.Request(
        REGISTRY_URL % node_id, headers={"User-Agent": "registry-release-check"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def describe_problem(entry: dict, version: str) -> str | None:
    """Return why the entry is not the wanted release, or None when it is."""
    latest = entry.get("latest_version") or {}
    if latest.get("version") != version:
        return "latest version is %r, wanted %r" % (latest.get("version"), version)
    if latest.get("status") != ACTIVE_STATUS:
        return "version %s has status %r, wanted %r" % (
            version,
            latest.get("status"),
            ACTIVE_STATUS,
        )
    return None


def wait_for_release(
    node_id: str,
    version: str,
    fetch: Callable[[str], dict] = fetch_registry_entry,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = TIMEOUT_SECONDS,
    poll_seconds: float = POLL_SECONDS,
) -> str | None:
    """Poll the registry; return None on success or the last problem on timeout."""
    attempts = max(1, int(timeout_seconds // poll_seconds) + 1)
    problem: str | None = "registry was not reachable"
    for attempt in range(attempts):
        try:
            problem = describe_problem(fetch(node_id), version)
        except OSError as error:
            problem = "registry request failed: %s" % error
        if problem is None:
            return None
        if attempt + 1 < attempts:
            sleep(poll_seconds)
    return problem


def main() -> int:
    node_id, version = read_pack(Path(__file__).resolve().parents[1] / "pyproject.toml")
    problem = wait_for_release(node_id, version)
    if problem is not None:
        print("Registry check failed for %s %s: %s" % (node_id, version, problem))
        return 1
    print("Registry lists %s %s as the latest active version." % (node_id, version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
