# What Sells Political and Business News? 32,487 Real Headline A/B Tests

![CI](https://github.com/akars7/headline-experiments/actions/workflows/ci.yml/badge.svg)

## Question
**Which headline wording makes people click, and does negativity or outrage sell political and business news more than other stories?**

From 2013 to 2015, the news site Upworthy tested 2 to 8 headlines for almost every story it published. Each reader was randomly shown one version, and the site counted who clicked. The archive of those tests, released for research in 2021, is one of the few public records of a real company's experimentation programme.

That makes this an A/B testing project at the level a company actually works at: not "did B beat A?", but **how many of our wins were real, and what kinds of changes actually work?**

## Design
| | |
|---|---|
| Unit | One headline version (a "package"), with its readers and clicks |
| Comparison | Only between versions of the **same story**, so topic, day and audience are held fixed and a click difference comes from the wording |
| Exploratory split | 4,873 tests, used to find patterns |
| Confirmatory split | 22,743 tests, **locked in code** until [the pre-registration](docs/PREREGISTRATION.md) is committed, then used once to check the patterns |

## What's in the toolkit
| Module | Answers |
|---|---|
| `data.py` | Loads and cleans the archive, drops badly randomised tests, locks the confirmatory split |
| `checks.py` | Were readers split evenly? Sample ratio mismatch for any number of versions |
| `topics.py` | Politics, business or other, from a keyword list checked against hand labels (`labels.py`) |
| `features.py` | Negative, positive and outrage words, questions, numbers, "you", curiosity gaps, length |
| `effects.py` | Which wording raises clicks? A within-test (fixed-effects) model, with clustered standard errors, overall and by topic |
| `programme.py` | How many wins were luck? False discovery rate, the share of tests with no real effect, and the winner's curse |
| `power.py` | How big did a test need to be, and how many were too small? |

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
pytest -q          # simulation tests recover known effects
```
Then download the data (see [data/README.md](data/README.md)) and run `notebooks/00_data_checks.ipynb` and `01_exploratory.ipynb`.

## Roadmap
- [x] Toolkit with tests: SRM, within-test effects, FDR, winner's curse, power
- [ ] Data checks and topic tagger validated on 200 hand-labelled stories
- [ ] Exploratory analysis and a committed pre-registration
- [ ] Confirmatory analysis: which patterns replicate?
- [ ] Headline playbook: a one-page memo for a newsroom or club comms team
- [ ] Extension: how left- and right-leaning outlets word the same events (NELA-GT news data)

## Limitations
- One outlet, left-leaning, 2013–2015. Results describe Upworthy's readers, not all news readers.
- Topics come from keywords, so some stories will be mislabelled; the tagger's precision and recall are reported.
- Headline images were tested too, but the image files aren't in the archive.

## Talking points
- Why compare only within a test, and what the test fixed effects remove
- Why thousands of tests need a false discovery rate, and what the winner's curse does to "winning" headlines
- Why the confirmatory split is locked in code until the plan is committed

## Data
Matias, J.N., Munger, K., Le Quere, M.A. & Ebersole, C. *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media.* Scientific Data 8, 195 (2021).
