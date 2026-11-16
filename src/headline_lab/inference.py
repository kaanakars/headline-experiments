"""Effect size and its uncertainty."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class EffectEstimate:
    """Treatment minus control, with a confidence interval."""

    control_value: float
    treatment_value: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float | None  # None when the method only gives an interval

    @property
    def relative_lift(self) -> float:
        """Difference as a share of control, e.g. -0.04 = 4% worse."""
        if self.control_value == 0:
            return float("nan")
        return self.difference / self.control_value


def two_proportion_ztest(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Compare two rates, e.g. 7-day retention.

    The p-value uses the pooled standard error, because it assumes the rates
    are equal. The CI uses the unpooled one, because it describes the gap we saw.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("group sizes must be positive")
    if not (0 <= successes_control <= n_control and 0 <= successes_treatment <= n_treatment):
        raise ValueError("successes must be between 0 and the group size")

    rate_control = successes_control / n_control
    rate_treatment = successes_treatment / n_treatment
    difference = rate_treatment - rate_control

    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    # both groups at 0% or 100%: nothing to compare
    p_value = 1.0 if pooled_se == 0 else float(2 * norm.sf(abs(difference) / pooled_se))

    unpooled_se = math.sqrt(
        rate_control * (1 - rate_control) / n_control
        + rate_treatment * (1 - rate_treatment) / n_treatment
    )
    z_crit = norm.ppf(1 - alpha / 2)
    return EffectEstimate(
        control_value=rate_control,
        treatment_value=rate_treatment,
        difference=difference,
        ci_low=difference - z_crit * unpooled_se,
        ci_high=difference + z_crit * unpooled_se,
        p_value=p_value,
    )


def bootstrap_difference(
    control: np.ndarray,
    treatment: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> EffectEstimate:
    """Percentile bootstrap CI for any statistic (mean, median, ...).

    For skewed metrics like game rounds, where a few heavy players make
    normal-theory intervals unreliable.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("both groups need at least one value")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        control_sample = rng.choice(control, size=control.size, replace=True)
        treatment_sample = rng.choice(treatment, size=treatment.size, replace=True)
        diffs[i] = statistic(treatment_sample) - statistic(control_sample)

    control_value = float(statistic(control))
    treatment_value = float(statistic(treatment))
    ci_low, ci_high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return EffectEstimate(
        control_value=control_value,
        treatment_value=treatment_value,
        difference=treatment_value - control_value,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=None,
    )
