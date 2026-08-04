"""Optional, explicitly selected runtime profiles for MiniMax-H3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict, cast

PROFILES_PATH = Path(__file__).with_name("calibration_profiles.json")
NO_PROFILE = "none (report tokens only)"


class CalibrationMeasurement(TypedDict):
    video_tokens: int
    seconds_per_iteration: float


class CalibrationProfile(TypedDict):
    label: str
    vram_total_gib: float
    overhead_seconds: float
    measurements: tuple[CalibrationMeasurement, ...]
    vram_plateau_mib: int | None
    vram_plateau_tokens: int | None
    thrash_tokens: int | None


class CostFit(TypedDict):
    coefficient: float
    exponent: float
    r_squared: float
    max_tokens: int
    sample_count: int


class CalibrationProfileError(ValueError):
    """Raised when a calibration profile is missing or malformed."""


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CalibrationProfileError(
            "%s must be a JSON object, got %s" % (location, type(value).__name__))
    return cast(dict[str, object], value)


def _required(mapping: dict[str, object], key: str, location: str) -> object:
    if key not in mapping:
        raise CalibrationProfileError("%s is missing required field '%s'" % (location, key))
    return mapping[key]


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationProfileError("%s must be a non-empty string" % location)
    return value.strip()


def _number(value: object, location: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationProfileError("%s must be a number" % location)
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise CalibrationProfileError(
            "%s must be finite and >= %s, got %r" % (location, minimum, value))
    return number


def _integer(value: object, location: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationProfileError("%s must be an integer" % location)
    if value < minimum:
        raise CalibrationProfileError(
            "%s must be >= %d, got %d" % (location, minimum, value))
    return value


def _optional_integer(
        mapping: dict[str, object], key: str, location: str) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _integer(mapping[key], "%s.%s" % (location, key), 1)


def _measurement(value: object, location: str) -> CalibrationMeasurement:
    raw = _mapping(value, location)
    return {
        "video_tokens": _integer(
            _required(raw, "video_tokens", location),
            "%s.video_tokens" % location,
            1,
        ),
        "seconds_per_iteration": _number(
            _required(raw, "seconds_per_iteration", location),
            "%s.seconds_per_iteration" % location,
            0.000001,
        ),
    }


def _profile(value: object, location: str) -> CalibrationProfile:
    raw = _mapping(value, location)
    measurements_value = _required(raw, "measurements", location)
    if not isinstance(measurements_value, list):
        raise CalibrationProfileError("%s.measurements must be a JSON array" % location)
    measurements = tuple(
        _measurement(item, "%s.measurements[%d]" % (location, index))
        for index, item in enumerate(measurements_value)
    )
    if len(measurements) < 2:
        raise CalibrationProfileError(
            "%s.measurements needs at least two positive samples" % location)
    if len({item["video_tokens"] for item in measurements}) < 2:
        raise CalibrationProfileError(
            "%s.measurements needs at least two distinct video_tokens values" % location)

    return {
        "label": _string(_required(raw, "label", location), "%s.label" % location),
        "vram_total_gib": _number(
            _required(raw, "vram_total_gib", location),
            "%s.vram_total_gib" % location,
            0.1,
        ),
        "overhead_seconds": _number(
            _required(raw, "overhead_seconds", location),
            "%s.overhead_seconds" % location,
            0.0,
        ),
        "measurements": measurements,
        "vram_plateau_mib": _optional_integer(raw, "vram_plateau_mib", location),
        "vram_plateau_tokens": _optional_integer(raw, "vram_plateau_tokens", location),
        "thrash_tokens": _optional_integer(raw, "thrash_tokens", location),
    }


def parse_profiles(value: object) -> dict[str, CalibrationProfile]:
    root = _mapping(value, "calibration profile document")
    version = _integer(
        _required(root, "schema_version", "calibration profile document"),
        "calibration profile document.schema_version",
        1,
    )
    if version != 1:
        raise CalibrationProfileError(
            "unsupported calibration profile schema_version %d; expected 1" % version)

    profiles_value = _required(root, "profiles", "calibration profile document")
    profiles_raw = _mapping(profiles_value, "calibration profile document.profiles")
    profiles: dict[str, CalibrationProfile] = {}
    for key, raw_profile in profiles_raw.items():
        profile_key = _string(key, "calibration profile key")
        profiles[profile_key] = _profile(raw_profile, "profiles.%s" % profile_key)
    return profiles


def load_profiles(path: Path) -> dict[str, CalibrationProfile]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except json.JSONDecodeError as error:
        raise CalibrationProfileError(
            "invalid JSON in %s at line %d column %d: %s"
            % (path, error.lineno, error.colno, error.msg)) from error
    except OSError as error:
        raise CalibrationProfileError(
            "cannot read calibration profiles from %s: %s" % (path, error)) from error
    return parse_profiles(raw)


def profile_options(profiles: dict[str, CalibrationProfile]) -> list[str]:
    return [NO_PROFILE, *sorted(profiles)]


def resolve_profile(
        profiles: dict[str, CalibrationProfile], choice: str) -> CalibrationProfile | None:
    if choice == NO_PROFILE:
        return None
    if choice not in profiles:
        raise CalibrationProfileError(
            "calibration profile '%s' does not exist in %s; restart ComfyUI after editing the file"
            % (choice, PROFILES_PATH))
    return profiles[choice]


def fit_power_law(profile: CalibrationProfile) -> CostFit:
    samples = profile["measurements"]
    count = len(samples)
    xs = [math.log(item["video_tokens"]) for item in samples]
    ys = [math.log(item["seconds_per_iteration"]) for item in samples]
    sum_x = sum(xs)
    sum_y = sum(ys)
    denominator = count * sum(value * value for value in xs) - sum_x * sum_x
    if abs(denominator) < 1e-12:
        raise CalibrationProfileError(
            "profile '%s' cannot be fitted because its token counts have no variance"
            % profile["label"])

    exponent = (
        count * sum(x * y for x, y in zip(xs, ys, strict=True)) - sum_x * sum_y
    ) / denominator
    coefficient = math.exp((sum_y - exponent * sum_x) / count)
    mean_y = sum_y / count
    residual = sum(
        (y - math.log(coefficient * math.exp(x) ** exponent)) ** 2
        for x, y in zip(xs, ys, strict=True)
    )
    total = sum((y - mean_y) ** 2 for y in ys)
    return {
        "coefficient": coefficient,
        "exponent": exponent,
        "r_squared": 1.0 - residual / total if total > 0.0 else 0.0,
        "max_tokens": max(item["video_tokens"] for item in samples),
        "sample_count": count,
    }


def estimate_runtime(
        profile: CalibrationProfile, video_tokens: int, steps: int) -> tuple[float, float, CostFit]:
    fit = fit_power_law(profile)
    seconds_per_iteration = (
        fit["coefficient"] * video_tokens ** fit["exponent"]
    )
    total_minutes = (
        profile["overhead_seconds"] + steps * seconds_per_iteration
    ) / 60.0
    return seconds_per_iteration, total_minutes, fit


def memory_notes(
        profile: CalibrationProfile, video_tokens: int, vram_budget_gib: float) -> list[str]:
    if abs(profile["vram_total_gib"] - vram_budget_gib) > 1.0:
        return [
            "selected cost profile was measured with %.1f GiB VRAM, but the workflow budget is %.1f GiB; profile-specific memory thresholds are not applied"
            % (profile["vram_total_gib"], vram_budget_gib)
        ]

    thrash_tokens = profile["thrash_tokens"]
    if thrash_tokens is not None and video_tokens >= thrash_tokens:
        return [
            "this profile measured offload thrashing at about %s tokens; expect runtime to exceed the fitted estimate"
            % format(thrash_tokens, ",")
        ]

    plateau_mib = profile["vram_plateau_mib"]
    plateau_tokens = profile["vram_plateau_tokens"]
    if (
        plateau_mib is not None
        and plateau_tokens is not None
        and video_tokens >= plateau_tokens
    ):
        return [
            "this profile measured a VRAM plateau near %s MiB after about %s tokens"
            % (format(plateau_mib, ","), format(plateau_tokens, ","))
        ]
    return []
