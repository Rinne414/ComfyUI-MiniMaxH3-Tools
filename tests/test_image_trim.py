"""The working-set trimmer fires on growth, re-bases after each trim, and is inert off Windows."""

from __future__ import annotations

import threading

from image_trim import mapped_pages_trimmed, trim_loop


class _Stop:
    """Stands in for threading.Event: lets the loop run a fixed number of polls."""

    def __init__(self, polls: int) -> None:
        self.left = polls

    def wait(self, _timeout: float) -> bool:
        self.left -= 1
        return self.left < 0


def test_trims_once_per_step_of_growth() -> None:
    state = {"ws": 1000, "trims": 0}

    def working_set_mb() -> int:
        state["ws"] += 600
        return state["ws"]

    def trim() -> None:
        state["trims"] += 1
        state["ws"] = 1000

    count = trim_loop(working_set_mb, trim, _Stop(6), step_mb=1024)

    assert count == state["trims"]
    assert count >= 2


def test_no_trim_while_the_working_set_is_flat() -> None:
    trims = []

    assert trim_loop(lambda: 5000, lambda: trims.append(1), _Stop(5), step_mb=1024) == 0
    assert trims == []


def test_block_runs_without_a_thread_when_disabled_or_unsupported() -> None:
    before = threading.active_count()

    with mapped_pages_trimmed(enabled=False):
        assert threading.active_count() == before
    with mapped_pages_trimmed(api=lambda: None):
        assert threading.active_count() == before


def test_block_starts_and_stops_the_trimmer() -> None:
    before = threading.active_count()

    with mapped_pages_trimmed(api=lambda: (lambda: 0, lambda: None)):
        assert threading.active_count() == before + 1

    assert threading.active_count() == before
