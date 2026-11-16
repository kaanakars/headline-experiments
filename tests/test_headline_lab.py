import numpy as np
import pandas as pd
import pytest

from headline_lab.checks import sample_ratio_mismatch, srm_by_test
from headline_lab.data import clean_packages, load_packages, summarise_tests
from headline_lab.effects import topic_difference, within_test_effects
from headline_lab.features import add_features, headline_features
from headline_lab.inference import two_proportion_ztest
from headline_lab.labels import validate
from headline_lab.power import mde_by_test, sample_size_two_proportions
from headline_lab.programme import (
    benjamini_hochberg,
    pvalues_by_test,
    share_with_no_effect,
    winners_curse,
)
from headline_lab.topics import tag_tests, tag_text

NEG_EFFECT = 0.10  # +10% clicks per negative word, everywhere
POLITICS_EXTRA = 0.10  # another +10% per negative word in political stories


def simulate(n_tests=600, seed=0, neg_effect=NEG_EFFECT, extra=POLITICS_EXTRA):
    """A programme of headline tests where the true effects are known."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tests):
        politics = t % 3 == 0
        base = rng.uniform(0.01, 0.03)
        for _ in range(rng.integers(2, 5)):
            neg = int(rng.integers(0, 3))
            rate = base * (1 + (neg_effect + extra * politics) * neg)
            n = int(rng.integers(3000, 6000))
            rows.append(
                {
                    "clickability_test_id": f"t{t}",
                    "headline": "x",
                    "created_at": "2014-01-01",
                    "impressions": n,
                    "clicks": rng.binomial(n, rate),
                    "negative_words": neg,
                    "topic": "politics" if politics else "other",
                }
            )
    df = pd.DataFrame(rows)
    df["ctr"] = df["clicks"] / df["impressions"]
    return df


def test_within_test_model_recovers_effect():
    df = simulate(extra=0)
    table = within_test_effects(df, ["negative_words"]).set_index("feature")
    est = table.loc["negative_words"]
    assert est["effect_relative"] == pytest.approx(NEG_EFFECT, abs=0.03)
    assert est["ci_low_relative"] > 0


def test_topic_difference_found():
    df = simulate()
    topics = df.groupby("clickability_test_id")["topic"].first()
    diff = topic_difference(df, "negative_words", topics)
    assert diff["ci_low"] > 0


def test_tagger():
    assert tag_text("Senate Republicans block the vote") == "politics"
    assert tag_text("Walmart CEO earns 1,000 times a worker's wage") == "business"
    assert tag_text("A puppy learns to swim") == "other"


def test_topic_is_the_same_for_every_version_of_a_story():
    df = pd.DataFrame(
        {
            "clickability_test_id": ["a", "a", "b"],
            "headline": ["Watch this", "Congress just did something", "Cute dog"],
            "excerpt": ["", "", ""],
            "slug": ["", "", ""],
        }
    )
    assert tag_tests(df).to_dict() == {"a": "politics", "b": "other"}


def test_headline_features():
    f = headline_features("This shameful lie will make you angry. Why do 3 senators agree?")
    assert f["negative_words"] >= 2
    assert f["outrage_words"] == 2
    assert f["question"] == 1 and f["has_number"] == 1 and f["you"] == 1
    assert f["curiosity_gap"] == 1


def test_add_features_keeps_rows():
    df = pd.DataFrame({"headline": ["Good news", "Bad news?"]})
    assert len(add_features(df)) == 2


def test_srm_any_number_of_arms():
    assert sample_ratio_mismatch([5000, 5050, 4980]).has_mismatch is False
    assert sample_ratio_mismatch([5000, 6000, 5000]).has_mismatch is True
    df = pd.DataFrame(
        {"clickability_test_id": ["a", "a", "b", "b"], "impressions": [1000, 1010, 1000, 1500]}
    )
    assert srm_by_test(df).set_index("clickability_test_id")["has_mismatch"].to_dict() == {
        "a": False,
        "b": True,
    }


def test_benjamini_hochberg():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # textbook example: only the two smallest survive at q = 0.05
    assert benjamini_hochberg(p).tolist() == [True, True] + [False] * 8


def test_share_with_no_effect():
    rng = np.random.default_rng(0)
    assert share_with_no_effect(rng.uniform(size=5000)) == pytest.approx(1, abs=0.05)
    assert share_with_no_effect(rng.uniform(0, 0.01, size=5000)) == 0


def test_winners_curse_when_nothing_is_real():
    # no real differences: every 'win' is luck, so the believable lift is about 0
    curse = winners_curse(simulate(neg_effect=0, extra=0))
    assert curse.raw_lift > 0.03
    assert curse.shrunk_lift < curse.raw_lift / 3


def test_pvalues_one_per_test():
    df = simulate(n_tests=20)
    assert len(pvalues_by_test(df)) == 20


def test_clean_drops_problem_rows_and_single_versions():
    raw = pd.DataFrame(
        {
            "clickability_test_id": ["a", "a", "b", "b", "c"],
            "headline": ["h1", "h2", "h1", "h2", "only one"],
            "created_at": ["2014-01-01"] * 5,
            "impressions": [100, 100, 100, 100, 100],
            "clicks": [1, 2, 1, 2, 3],
            "problem": [0, 0, 1, 1, 0],
        }
    )
    clean = clean_packages(raw)
    assert set(clean["clickability_test_id"]) == {"a"}
    assert summarise_tests(clean)["arms"].tolist() == [2]


def test_confirmatory_split_is_locked(tmp_path):
    with pytest.raises(PermissionError):
        load_packages(tmp_path, split="confirmatory")


def test_power():
    assert 3830 <= sample_size_two_proportions(0.10, 0.02) <= 3850
    tests = pd.DataFrame({"ctr": [0.02, 0.02], "impressions": [4000, 40000], "arms": [2, 2]})
    mde = mde_by_test(tests)["mde_relative"]
    assert mde[1] < mde[0]


def test_ztest():
    result = two_proportion_ztest(1000, 10_000, 1200, 10_000)
    assert result.p_value < 0.001 and result.ci_low > 0


def test_validate_labels():
    labels = pd.DataFrame(
        {"clickability_test_id": ["a", "b", "c"], "label": ["politics", "other", ""]}
    )
    predicted = pd.Series({"a": "politics", "b": "politics", "c": "other"})
    table = validate(labels, predicted).set_index("topic")
    assert table.loc["politics", "precision"] == 0.5
    assert table.loc["politics", "recall"] == 1.0
