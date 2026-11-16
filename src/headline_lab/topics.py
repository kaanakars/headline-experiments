"""Tag each story as politics, business or other, from its text.

A keyword list, not a model: easy to read, easy to check, and checked against
hand labels (see labels.py). A story is one test, so every version of the
same story gets the same topic.
"""

from __future__ import annotations

import re

import pandas as pd

POLITICS = [
    r"obama",
    r"congress\w*",
    r"senat\w*",
    r"republican\w*",
    r"democrat\w*",
    r"gop",
    r"elect\w*",
    r"vot(e|es|ed|er|ers|ing)",
    r"campaign\w*",
    r"politic\w*",
    r"president\w*",
    r"supreme court",
    r"governor\w*",
    r"legislat\w*",
    r"lawmaker\w*",
    r"fox news",
    r"obamacare",
    r"tea party",
    r"midterm\w*",
    r"white house",
    r"filibuster",
    r"partisan",
    r"conservative\w*",
    r"liberal\w*",
    r"romney",
    r"boehner",
    r"mcconnell",
    r"ballot\w*",
]
BUSINESS = [
    r"ceo\w*",
    r"compan(y|ies)",
    r"corporat\w*",
    r"wall street",
    r"econom\w*",
    r"wage\w*",
    r"minimum wage",
    r"tax(es)?",
    r"billionaire\w*",
    r"millionaire\w*",
    r"profit\w*",
    r"bank\w*",
    r"walmart",
    r"mcdonald'?s",
    r"amazon",
    r"income",
    r"inequality",
    r"salar(y|ies)",
    r"unemploy\w*",
    r"jobs",
    r"debt",
    r"stock\w*",
    r"invest\w*",
    r"business\w*",
    r"industr(y|ies)",
    r"workers?",
    r"paycheck\w*",
    r"1 ?percent",
]


def _pattern(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", flags=re.IGNORECASE)


POLITICS_RE, BUSINESS_RE = _pattern(POLITICS), _pattern(BUSINESS)


def tag_text(text: str) -> str:
    """'politics', 'business' or 'other'. Politics wins a tie: most money-and-power
    stories on Upworthy were about policy."""
    n_politics = len(POLITICS_RE.findall(text))
    n_business = len(BUSINESS_RE.findall(text))
    if n_politics == 0 and n_business == 0:
        return "other"
    return "politics" if n_politics >= n_business else "business"


def tag_tests(packages: pd.DataFrame) -> pd.Series:
    """Topic per test id, using all text from every version of the story."""
    text_cols = [c for c in ["headline", "excerpt", "lede", "slug"] if c in packages]
    # slugs use hyphens between words
    text = packages[text_cols].astype(str).agg(" ".join, axis=1).str.replace("-", " ")
    story = text.groupby(packages["clickability_test_id"]).agg(" ".join)
    return story.map(tag_text).rename("topic")
