"""Checks to run before reading any result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chisquare


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    p_value: float
    has_mismatch: bool


def sample_ratio_mismatch(counts, shares=None, threshold: float = 0.001) -> SRMResult:
    """Chi-square test: does the split of readers match the one we asked for?

    Works for any number of arms (equal split unless shares are given). A
    mismatch means assignment or logging broke, so the test can't be trusted.
    The threshold is strict (0.001) because this check runs on every test, and
    0.05 would give a false alarm in 1 test out of 20.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size < 2 or (counts < 0).any() or counts.sum() == 0:
        raise ValueError("need at least two non-negative group sizes")
    shares = np.full(counts.size, 1 / counts.size) if shares is None else np.asarray(shares)
    if not np.isclose(shares.sum(), 1):
        raise ValueError("shares must add up to 1")

    _, p_value = chisquare(counts, f_exp=counts.sum() * shares)
    return SRMResult(tuple(int(c) for c in counts), float(p_value), bool(p_value < threshold))


def srm_by_test(packages: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """SRM p-value for every test. Upworthy aimed for an equal split between versions."""
    rows = [
        {
            "clickability_test_id": test_id,
            "p_value": sample_ratio_mismatch(group["impressions"], threshold=threshold).p_value,
        }
        for test_id, group in packages.groupby("clickability_test_id")
    ]
    out = pd.DataFrame(rows)
    out["has_mismatch"] = out["p_value"] < threshold
    return out
