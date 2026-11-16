"""Which wording raises clicks? Compared only between versions of the same story.

Each test is one story shown with different headlines to random readers. By
comparing versions within a test, the story, its topic, the day and the
audience are held fixed, so a click difference comes from the wording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

TEST = "clickability_test_id"


def demean(df: pd.DataFrame, cols: list[str], weights: pd.Series) -> pd.DataFrame:
    """Subtract each test's reader-weighted mean, so only within-test differences remain."""
    weighted = df[cols].mul(weights, axis=0).groupby(df[TEST]).transform("sum")
    means = weighted.div(weights.groupby(df[TEST]).transform("sum"), axis=0)
    return df[cols] - means


def within_test_effects(packages: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Change in click rate per unit of each feature, all features in one model.

    Weighted least squares on test-demeaned data (a fixed-effects model),
    weighted by readers, with standard errors clustered by test because
    versions of the same story are not independent.
    """
    weights = packages["impressions"].astype(float)
    data = demean(packages, ["ctr", *features], weights)
    # features that never vary within a test carry no information here
    usable = [f for f in features if data[f].abs().sum() > 0]
    fit = sm.WLS(data["ctr"], data[usable], weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": packages[TEST].astype("category").cat.codes}
    )
    base = np.average(packages["ctr"], weights=weights)
    ci = fit.conf_int()
    out = pd.DataFrame(
        {
            "feature": usable,
            "effect": fit.params.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "p_value": fit.pvalues.values,
        }
    )
    # as a share of the average click rate: "+3%" reads better than "+0.0005"
    for col in ["effect", "ci_low", "ci_high"]:
        out[f"{col}_relative"] = out[col] / base
    out.attrs["n_tests"] = packages[TEST].nunique()
    out.attrs["base_ctr"] = base
    return out


def effects_by_topic(
    packages: pd.DataFrame, features: list[str], topics: pd.Series
) -> pd.DataFrame:
    """The same model fitted separately for each topic."""
    topic = packages[TEST].map(topics)
    frames = [
        within_test_effects(packages[topic == name], features).assign(topic=name)
        for name in sorted(topic.dropna().unique())
    ]
    return pd.concat(frames, ignore_index=True)


def topic_difference(
    packages: pd.DataFrame, feature: str, topics: pd.Series, topic: str = "politics"
) -> pd.Series:
    """Does a feature work differently for one topic? Tests feature x topic in one model.

    The topic itself drops out (it's the same for every version of a story),
    but its interaction with wording doesn't.
    """
    in_topic = (packages[TEST].map(topics) == topic).astype(float)
    data = packages.assign(interaction=packages[feature] * in_topic)
    table = within_test_effects(data, [feature, "interaction"]).set_index("feature")
    return table.loc["interaction"]
