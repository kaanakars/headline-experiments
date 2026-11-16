"""Hand labels to check the topic tagger.

1. headline-lab-label-sample writes data/labels/to_label.csv (200 random stories).
2. Fill the 'label' column with politics, business or other, by reading the story.
3. validate() compares your labels with the tagger.
The tagger's guess isn't in the file, so it can't sway your labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from headline_lab.data import load_packages

OUT = Path("data/labels/to_label.csv")


def sample_for_labelling(packages: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    stories = packages.groupby("clickability_test_id").first().reset_index()
    sample = stories.sample(n=min(n, len(stories)), random_state=seed)
    out = sample[["clickability_test_id", "headline", "excerpt", "slug"]].copy()
    out["label"] = ""
    return out


def validate(labels: pd.DataFrame, predicted: pd.Series) -> pd.DataFrame:
    """Precision and recall per topic.

    precision: of the stories the tagger called X, the share that really are X.
    recall: of the stories that really are X, the share the tagger found.
    """
    labelled = labels[labels["label"].str.strip() != ""].copy()
    labelled["predicted"] = labelled["clickability_test_id"].map(predicted)
    rows = []
    for topic in ["politics", "business", "other"]:
        said = labelled["predicted"] == topic
        truly = labelled["label"].str.strip().str.lower() == topic
        rows.append(
            {
                "topic": topic,
                "labelled": int(truly.sum()),
                "precision": (said & truly).sum() / max(said.sum(), 1),
                "recall": (said & truly).sum() / max(truly.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"{OUT} already exists. Delete it first to draw a new sample.")
    sample_for_labelling(load_packages()).to_csv(OUT, index=False)
    print(f"Saved {OUT}. Fill the 'label' column with politics, business or other.")


if __name__ == "__main__":
    main()
