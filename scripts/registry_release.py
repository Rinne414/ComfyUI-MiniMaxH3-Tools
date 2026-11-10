"""Release checks against the Comfy Registry.

``decide``  says whether the version in pyproject.toml still has to be published.
``verify``  fails unless the registry lists that version as the latest and active one.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
import urllib.request
from collections.abc import Callable
from pathlib import Path

REGISTRY_URL = "https://api.comfy.org/nodes/%s"
VERSIONS_URL = "https://api.comfy.org/nodes/%s/versions"
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


def fetch_versions(node_id: str) -> list:
    """Return the registry's version list; an unknown node gives an empty list."""
    request = urllib.request.Request(
        VERSIONS_URL % node_id, headers={"User-Agent": "registry-release-check"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def decide_publish(version: str, payload: object) -> tuple[bool, str]:
    """Return ``(should_publish, reason)`` for a version and the registry's list.

    Anything that is not a list of version objects is an error, never a reason
    to publish: a broken answer must not look like "version not found".
    """
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("unexpected registry answer: %.200r" % (payload,))
    if any(item.get("version") == version for item in payload):
        return False, "version %s is already on the registry; skipping publish" % version
    return True, "version %s is not on the registry yet; publishing" % version


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


def run_decide(node_id: str, version: str) -> int:
    try:
        should_publish, reason = decide_publish(version, fetch_versions(node_id))
    except (OSError, ValueError) as error:
        print("::error::Cannot tell whether %s is published: %s" % (version, error))
        return 1
    print("::notice::%s" % reason)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write("publish=%s\n" % str(should_publish).lower())
    return 0


def main(argv: list[str] | None = None) -> int:
    command = (argv if argv is not None else sys.argv[1:]) or ["verify"]
    if command not in (["decide"], ["verify"]):
        print("usage: registry_release.py [decide|verify]")
        return 2
    node_id, version = read_pack(Path(__file__).resolve().parents[1] / "pyproject.toml")
    if command == ["decide"]:
        return run_decide(node_id, version)
    problem = wait_for_release(node_id, version)
    if problem is not None:
        print("Registry check failed for %s %s: %s" % (node_id, version, problem))
        return 1
    print("Registry lists %s %s as the latest active version." % (node_id, version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
