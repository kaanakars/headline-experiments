"""Measurable things about a headline's wording."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

import pandas as pd

WORD_RE = re.compile(r"[a-z']+")
NUMBER_RE = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
YOU_RE = re.compile(r"\b(you|your|you're|yourself)\b", re.I)
# "This man...", "Here's why...": the headline holds back what the story is about
CURIOSITY_RE = re.compile(
    r"^(this|these|here'?s|what happens|you won'?t believe|watch)\b|\bwhat happen(s|ed) next\b",
    re.I,
)

FEATURES = {
    "negative_words": "Negative words (sentiment lexicon)",
    "positive_words": "Positive words (sentiment lexicon)",
    "outrage_words": "Outrage and conflict words",
    "question": "Is a question",
    "has_number": "Contains a number",
    "you": "Speaks to the reader (you / your)",
    "curiosity_gap": "Holds back the point (This..., Here's...)",
    "word_count": "Number of words",
}


@lru_cache(maxsize=1)
def sentiment_lexicon() -> dict[str, float]:
    """VADER word scores, from -4 (very negative) to +4 (very positive)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer().lexicon


@lru_cache(maxsize=1)
def outrage_words() -> frozenset[str]:
    text = resources.files("headline_lab").joinpath("lexicons/outrage.txt").read_text()
    return frozenset(w.strip() for w in text.splitlines() if w.strip() and not w.startswith("#"))


def headline_features(headline: str) -> dict[str, float]:
    words = WORD_RE.findall(headline.lower())
    lexicon, outrage = sentiment_lexicon(), outrage_words()
    return {
        "negative_words": sum(lexicon.get(w, 0) < 0 for w in words),
        "positive_words": sum(lexicon.get(w, 0) > 0 for w in words),
        "outrage_words": sum(w in outrage for w in words),
        "question": float(headline.strip().endswith("?")),
        "has_number": float(bool(NUMBER_RE.search(headline))),
        "you": float(bool(YOU_RE.search(headline))),
        "curiosity_gap": float(bool(CURIOSITY_RE.search(headline.strip()))),
        "word_count": len(words),
    }


def add_features(packages: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame([headline_features(h) for h in packages["headline"]])
    return pd.concat([packages.reset_index(drop=True), features], axis=1)
