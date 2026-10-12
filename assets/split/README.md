# Frozen train/test split

`split.csv` has one row per problem:

| Column | Meaning |
|---|---|
| `sample_id` | The problem, as in `sample_<id>.json` |
| `part` | `dev` (80%, to train and tune) or `test` (20%, scored once, at the end) |
| `fold` | The cross-validation fold of a dev problem, 0 to 4; -1 in test |
| `group` | The problem's near-duplicate group, from `near_duplicate_groups()` on the full dataset. The ids are labels: problems with the same id are versions of one problem, and share their part and fold |

`notebooks/03_train_test_split.ipynb` explains the method and writes this file once, with `codeforces_tags.splitting.make_split()`, `config.SPLIT_SEED` and `freeze_split()`. Re-running the notebook checks the file against a fresh split instead of overwriting it. In short:
- near-duplicate groups stay whole;
- the groups with a focus tag and those without are split separately, each by iterative stratification:
  - the focus stratum is balanced on the 8 focus tags, the pairs of them shared by at least 20 problems (`config.SPLIT_MIN_PAIR_PROBLEMS`) and the release year;
  - the no-focus stratum is balanced on the release year.
- fold 0 is the test set; the same rule splits dev into the 5 folds.

Read it with `codeforces_tags.splitting`:
- `load_split()` returns `part`, `fold` and `group`, indexed by `sample_id`. It fails if a column is missing (a file from before `group` existed: delete it and re-run notebook 03) or if a group spans two parts or two folds (a hand-edited file).
- `select_part(load_dataset(), load_split(), "dev")` returns the dev problems, in the dataset's order, with their `fold` and `group` columns; `"test"` returns the test problems.

Take the near-duplicate groups from this file wherever problems must be grouped, as in the group bootstrap. Calling `near_duplicate_groups()` on a subset refits its TF-IDF, which can link pairs nobody judged.
