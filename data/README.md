# Data

Not committed (except `labels/`). Source: **The Upworthy Research Archive**
(Matias et al., *Scientific Data*, 2021), https://osf.io/jd64p/

1. Open https://osf.io/jd64p/ → **Files**.
2. Download the **exploratory packages** CSV and the **confirmatory packages**
   CSV (file names contain `exploratory-packages` and `confirmatory-packages`).
3. Save both in `data/raw/`. The loader finds them by name.

The confirmatory file is locked in code (`load_packages(split="confirmatory")`
raises an error) until the pre-registration is committed.

Columns used: `clickability_test_id`, `headline`, `excerpt`, `lede`, `slug`,
`impressions`, `clicks`, `created_at`, and the randomisation flag. See
https://upworthy.natematias.com/about-the-archive for all columns.
