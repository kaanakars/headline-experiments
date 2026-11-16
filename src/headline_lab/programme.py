"""Questions about the whole programme of tests, not one test.

With thousands of tests, some "winners" are luck. These tools estimate how many.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

TEST = "clickability_test_id"


def pvalues_by_test(packages: pd.DataFrame) -> pd.Series:
    """Chi-square p-value per test: do the versions' click rates differ at all?"""
    pvalues = {}
    for test_id, group in packages.groupby(TEST):
        table = np.column_stack([group["clicks"], group["impressions"] - group["clicks"]])
        if (table.sum(axis=0) == 0).any():
            continue  # no clicks (or no non-clicks) at all: nothing to test
        pvalues[test_id] = chi2_contingency(table)[1]
    return pd.Series(pvalues, name="p_value")


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """Which tests stay significant when only q of the 'discoveries' may be false.

    Sort the p-values; the k-th smallest passes if p <= q * k / m. Everything up
    to the largest passing k counts as a discovery.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    passes = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, dtype=bool)
    if passes.any():
        keep[order[: np.flatnonzero(passes).max() + 1]] = True
    return keep


def share_with_no_effect(p_values, lam: float = 0.5) -> float:
    """Storey's estimate of the share of tests where nothing really differed.

    With no real effect, p-values are spread evenly from 0 to 1, so the share
    above lam, scaled up by 1 / (1 - lam), estimates the 'null' tests.
    """
    p = np.asarray(p_values, dtype=float)
    return float(min(1.0, (p > lam).mean() / (1 - lam)))


@dataclass(frozen=True)
class WinnersCurse:
    raw_lift: float  # average lift of the best version over its test's average
    shrunk_lift: float  # the same after removing the expected luck
    true_spread: float  # estimated real spread of version effects (relative)

    @property
    def overstatement(self) -> float:
        """How much bigger the raw lift is than the believable one, e.g. 0.5 = 50%."""
        return self.raw_lift / self.shrunk_lift - 1 if self.shrunk_lift > 0 else float("inf")


def winners_curse(packages: pd.DataFrame) -> WinnersCurse:
    """Empirical-Bayes estimate of how much the winning versions are overrated.

    Each version's lift over its test's average is real effect + noise. The
    real spread is what's left after removing the noise we expect from the
    sample sizes. Each lift is then shrunk toward 0, more when it is noisy.
    """
    test_ctr = packages.groupby(TEST)["clicks"].transform("sum") / packages.groupby(TEST)[
        "impressions"
    ].transform("sum")
    usable = test_ctr > 0
    df = packages[usable]
    rate = test_ctr[usable]
    lift = df["ctr"] / rate - 1
    se2 = (rate * (1 - rate) / df["impressions"]) / rate**2
    true_spread = max(float(lift.var() - se2.mean()), 0.0)
    shrunk = lift * true_spread / (true_spread + se2)

    best = df.groupby(TEST)["ctr"].idxmax()
    return WinnersCurse(
        raw_lift=float(lift.loc[best].mean()),
        shrunk_lift=float(shrunk.loc[best].mean()),
        true_spread=true_spread,
    )
