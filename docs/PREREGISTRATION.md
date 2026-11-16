# Pre-registration

**Status: DRAFT.** Fill in after notebook 01, then commit **before** the
confirmatory split is read. The commit date is the proof: notebook 02 must be
newer than this file's final commit.

## Data
- Upworthy Research Archive. Exploratory split (4,873 tests) used to find
  patterns; confirmatory split (22,743 tests) used once, to check them.
- Same cleaning in both: drop tests flagged as badly randomised, tests with one
  version, and tests that fail the split check (SRM, p < 0.001).

## Model (fixed)
`headline_lab.effects.within_test_effects`: click rate on headline features,
compared only within the same test (test fixed effects), weighted by readers,
standard errors clustered by test. Effects reported relative to the average
click rate, with 95% CIs.

Topics from `headline_lab.topics`, keyword list frozen at commit
`<fill in the commit hash>`. Tagger precision and recall on hand labels:
`<fill in>`.

## Hypotheses
Fill in from notebook 01. One line each: what, direction, and size expected.

| | Hypothesis | Test | Confirmed if |
|---|---|---|---|
| H1 | | | 95% CI excludes 0 in the predicted direction |
| H2 | | | |
| H3 | | | |

## Multiple testing
With 3 hypotheses, each is tested at alpha = 0.05 / 3 (Bonferroni), so the
chance of any false confirmation stays at 5%.

## What will be reported either way
All hypotheses, confirmed or not, with their effects and CIs. Failed
replications are results too.

## Changes after registration
None.
