"""Power analysis: how many players a test needs, decided before it starts."""

from __future__ import annotations

import math

from scipy.stats import norm


def _validate_rate(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def sample_size_two_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Players needed per group to detect an absolute change in a rate.

    Normal approximation, two-sided test. Example: baseline 0.19, MDE 0.01
    (one percentage point).
    """
    _validate_rate("baseline_rate", baseline_rate)
    treatment_rate = baseline_rate + minimum_detectable_effect
    _validate_rate("baseline_rate + minimum_detectable_effect", treatment_rate)
    if minimum_detectable_effect == 0:
        raise ValueError("minimum_detectable_effect must be non-zero")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    pooled = (baseline_rate + treatment_rate) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * math.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate))
    ) ** 2
    return math.ceil(numerator / minimum_detectable_effect**2)


def minimum_detectable_effect(
    baseline_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest absolute change we can detect with this many players per group.

    Assumes both groups have the baseline variance, which is close enough for planning.
    """
    _validate_rate("baseline_rate", baseline_rate)
    if sample_size_per_group <= 0:
        raise ValueError("sample_size_per_group must be positive")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    standard_error = math.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size_per_group)
    return (z_alpha + z_power) * standard_error


def mde_by_test(tests, alpha: float = 0.05, power: float = 0.80):
    """Smallest detectable change per test, as a share of its click rate.

    tests needs columns ctr, impressions and arms. Assumes readers were split
    equally and compares two versions (with more arms each pair has fewer readers).
    """
    per_arm = tests["impressions"] / tests["arms"]
    absolute = [
        minimum_detectable_effect(rate, int(n), alpha, power) if 0 < rate < 1 else float("nan")
        for rate, n in zip(tests["ctr"], per_arm, strict=True)
    ]
    return tests.assign(mde_absolute=absolute, mde_relative=lambda t: t["mde_absolute"] / t["ctr"])
