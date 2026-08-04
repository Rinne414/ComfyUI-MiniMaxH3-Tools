"""Tests for explicit, machine-specific runtime profiles."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import (  # noqa: E402
    PROFILES_PATH,
    CalibrationProfileError,
    estimate_runtime,
    fit_power_law,
    load_profiles,
    memory_notes,
    parse_profiles,
)


def profile_document():
    return {
        "schema_version": 1,
        "profiles": {
            "test-rig": {
                "label": "Test GPU, exact test stack",
                "vram_total_gib": 24.0,
                "overhead_seconds": 60.0,
                "measurements": [
                    {"video_tokens": 10000, "seconds_per_iteration": 10.0},
                    {"video_tokens": 20000, "seconds_per_iteration": 20.0},
                ],
                "vram_plateau_mib": 23000,
                "vram_plateau_tokens": 15000,
                "thrash_tokens": 30000,
                "ignored_vendor_field": "allowed",
            }
        },
        "ignored_root_field": True,
    }


def test_shipped_profiles_are_valid_and_opt_in():
    profiles = load_profiles(PROFILES_PATH)
    assert set(profiles) == {"rtx-3090-int8-convrot-xformers"}
    assert "Single RTX 3090" in profiles["rtx-3090-int8-convrot-xformers"]["label"]


def test_external_profile_is_validated_and_extra_fields_are_ignored(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profile_document()), encoding="utf-8")
    profile = load_profiles(path)["test-rig"]
    assert profile["measurements"][1]["video_tokens"] == 20000
    assert "ignored_vendor_field" not in profile


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda document: document.pop("schema_version"), "schema_version"),
        (lambda document: document.update(schema_version=2), "unsupported"),
        (
            lambda document: document["profiles"]["test-rig"].pop("label"),
            "label",
        ),
        (
            lambda document: document["profiles"]["test-rig"].update(
                measurements=[{"video_tokens": 10000, "seconds_per_iteration": 10.0}]
            ),
            "at least two",
        ),
    ],
)
def test_invalid_profiles_fail_with_specific_errors(mutate, expected):
    document = profile_document()
    mutate(document)
    with pytest.raises(CalibrationProfileError, match=expected):
        parse_profiles(document)


def test_invalid_json_reports_location(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(CalibrationProfileError, match=r"line 1 column"):
        load_profiles(path)


def test_power_law_and_total_runtime_are_derived_from_selected_profile():
    profile = parse_profiles(profile_document())["test-rig"]
    fit = fit_power_law(profile)
    seconds_per_iteration, total_minutes, returned_fit = estimate_runtime(
        profile, 40000, 20)
    assert fit["exponent"] == pytest.approx(1.0)
    assert returned_fit == fit
    assert seconds_per_iteration == pytest.approx(40.0)
    assert total_minutes == pytest.approx((60.0 + 20 * 40.0) / 60.0)


def test_profile_memory_thresholds_require_matching_vram_budget():
    profile = parse_profiles(profile_document())["test-rig"]
    assert "plateau" in memory_notes(profile, 20000, 24.0)[0]
    assert "thrashing" in memory_notes(profile, 30000, 24.0)[0]
    assert "not applied" in memory_notes(profile, 30000, 16.0)[0]
