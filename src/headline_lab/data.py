"""Load the Upworthy Research Archive packages (one row per headline version)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REQUIRED = ["clickability_test_id", "headline", "impressions", "clicks", "created_at"]
TEXT = ["headline", "excerpt", "lede", "slug"]


def find_file(raw_dir: str | Path, split: str) -> Path:
    files = sorted(Path(raw_dir).glob(f"*{split}*packages*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No '{split}' packages CSV in {raw_dir}. See data/README.md for the download."
        )
    return files[-1]


def load_packages(
    raw_dir: str | Path = RAW_DIR, split: str = "exploratory", confirm: bool = False
) -> pd.DataFrame:
    """Read one split and clean it.

    The confirmatory split is locked: read it only after docs/PREREGISTRATION.md
    is committed, so the plan can't be shaped by its results.
    """
    if split == "confirmatory" and not confirm:
        raise PermissionError("Commit docs/PREREGISTRATION.md first, then call with confirm=True.")
    raw = pd.read_csv(find_file(raw_dir, split), low_memory=False)
    return clean_packages(raw)


def clean_packages(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")]).copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in TEXT:
        df[col] = df[col].fillna("").astype(str) if col in df else ""

    # the archive flags tests from Jun 2013 to Jan 2014 as badly randomised
    flag = next((c for c in ["problem", "randomization_imbalance_risk"] if c in df), None)
    if flag:
        df = df[df[flag] != 1]

    df = df[(df["impressions"] > 0) & (df["clicks"] >= 0) & (df["clicks"] <= df["impressions"])]
    # exact copies of a row add nothing and would double count readers
    df = df.drop_duplicates(
        subset=["clickability_test_id", "headline", "eyecatcher_id"]
        if "eyecatcher_id" in df
        else ["clickability_test_id", "headline"]
    )
    # a test needs at least two versions to compare
    arms = df.groupby("clickability_test_id")["headline"].transform("size")
    df = df[arms >= 2].copy()
    df["ctr"] = df["clicks"] / df["impressions"]
    return df.reset_index(drop=True)


def summarise_tests(packages: pd.DataFrame) -> pd.DataFrame:
    """One row per test: arms, readers, clicks and click rate."""
    grouped = packages.groupby("clickability_test_id")
    out = grouped.agg(
        arms=("headline", "size"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        created_at=("created_at", "min"),
    )
    out["ctr"] = out["clicks"] / out["impressions"]
    return out.reset_index()
