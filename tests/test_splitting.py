"""Tests for the dev/test split and the cross-validation folds."""

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS, SPLIT_MIN_PAIR_PROBLEMS, SPLIT_TEST_FOLD
from codeforces_tags.data.labels import frequent_pairs, multi_hot
from codeforces_tags.splitting import (
    freeze_split,
    load_split,
    make_split,
    release_years,
    select_part,
    stratified_group_folds,
    two_stage_folds,
)


@pytest.fixture(name="problems")
def problems_fixture():
    """Return 100 problems: 51 with a focus tag, 49 without, over five years, some in pairs."""
    ids = range(100)
    tags = [["math", "greedy"] if k % 2 == 0 else ["implementation"] for k in ids]
    tags[1] = ["graphs"]  # pair (0, 1): both with a focus tag
    tags[3] = ["greedy"]  # pair (2, 3): problem 2 has math, 3 has no focus tag (mixed)
    created = pd.to_datetime([f"{2016 + k % 5}-06-01" for k in ids], utc=True)
    return pd.DataFrame(
        {"tags": tags, "prob_desc_created_at": created},
        index=pd.Index(ids, name="sample_id"),
    )


@pytest.fixture(name="groups")
def groups_fixture(problems):
    """Pair problems 0-1, 2-3 and 4-5; every other problem is its own group."""
    group = pd.Series(problems.index, index=problems.index, name="group")
    group[[1, 3, 5]] = [0, 2, 4]
    return group


def test_release_years_reads_the_release_date(problems):
    assert release_years(problems).head(3).tolist() == [2016, 2017, 2018]


def test_stratified_group_folds_keeps_groups_whole_and_balances_labels():
    labels = pd.DataFrame({"a": [1, 1] + [1] * 8 + [0] * 10, "b": [0, 0] + [0] * 8 + [1] * 10})
    groups = pd.Series([0, 0] + list(range(1, 19)))

    folds = stratified_group_folds(labels, groups, n_folds=2, seed=0)

    assert folds[0] == folds[1]  # the pair stays together
    assert labels["b"].groupby(folds).sum().tolist() == [5, 5]  # b is split evenly


def test_two_stage_folds_keeps_groups_whole(problems, groups):
    folds = two_stage_folds(problems, groups, n_folds=5, seed=0)

    assert folds.groupby(groups).nunique().max() == 1
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_two_stage_folds_balances_each_stratum(problems, groups):
    folds = two_stage_folds(problems, groups, n_folds=5, seed=0)

    no_focus = problems["tags"].map(lambda tags: "math" not in tags and "graphs" not in tags)
    per_fold = no_focus.groupby(folds).sum()
    assert per_fold.max() - per_fold.min() <= 2  # about 10 no-focus problems per fold


def test_two_stage_folds_depends_on_the_seed_only(problems, groups):
    first = two_stage_folds(problems, groups, n_folds=5, seed=1)

    assert first.equals(two_stage_folds(problems, groups, n_folds=5, seed=1))
    assert not first.equals(two_stage_folds(problems, groups, n_folds=5, seed=2))


def test_make_split_takes_one_fold_for_test_and_splits_dev_again(problems, groups):
    split = make_split(problems, groups, seed=0)

    test = split["part"] == "test"
    assert 15 <= test.sum() <= 25  # about one problem in five
    assert (split.loc[test, "fold"] == -1).all()
    assert sorted(split.loc[~test, "fold"].unique()) == [0, 1, 2, 3, 4]
    assert split.groupby(groups)["part"].nunique().max() == 1
    assert split.groupby(groups)["fold"].nunique().max() == 1


def test_stratified_group_folds_counts_a_label_any_member_has():
    # Group 0 is items 0 and 1, and only item 1 has the rare label; group 1 (item 2) has it too.
    labels = pd.DataFrame({"rare": [0, 1, 1] + [0] * 7, "common": [1, 0, 0] + [1] * 7})
    groups = pd.Series([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])

    for seed in range(10):
        folds = stratified_group_folds(labels, groups, n_folds=2, seed=seed)
        assert folds[1] != folds[2]  # the two groups with the rare label are spread


def test_stratified_group_folds_accepts_a_single_label():
    labels = pd.DataFrame({"year_2020": [1, 0] * 5})

    folds = stratified_group_folds(labels, pd.Series(range(10)), n_folds=2, seed=0)

    assert sorted(labels["year_2020"].groupby(folds).sum()) == [2, 3]  # 5 positives, spread


def test_stratified_group_folds_needs_a_group_per_fold():
    labels = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})

    with pytest.raises(ValueError, match="3 groups cannot fill 5 folds"):
        stratified_group_folds(labels, pd.Series([0, 1, 2]), n_folds=5, seed=0)


def test_two_stage_folds_stratifies_each_stratum_on_its_own_columns(problems, groups):
    folds = two_stage_folds(problems, groups, n_folds=5, seed=0)

    focus = multi_hot(problems["tags"], FOCUS_TAGS)
    years = pd.get_dummies(release_years(problems), prefix="year", dtype=int)
    in_focus = focus.any(axis=1).groupby(groups).transform("any")
    pairs = frequent_pairs(focus, SPLIT_MIN_PAIR_PROBLEMS)  # none in this fixture
    focus_labels = pd.concat([focus, pairs, years], axis=1)[in_focus]
    assert folds[in_focus].equals(stratified_group_folds(focus_labels, groups[in_focus], 5, 0))
    other = stratified_group_folds(years[~in_focus], groups[~in_focus], 5, 0)  # year only
    assert folds[~in_focus].equals(other)


def test_two_stage_folds_also_stratifies_the_focus_stratum_on_frequent_tag_pairs(problems, groups):
    tags = [["math", "graphs"] if k % 4 == 0 else row for k, row in enumerate(problems["tags"])]
    paired = problems.assign(tags=tags)  # 25 problems have both math and graphs

    folds = two_stage_folds(paired, groups, n_folds=5, seed=0)

    focus = multi_hot(paired["tags"], FOCUS_TAGS)
    pairs = frequent_pairs(focus, SPLIT_MIN_PAIR_PROBLEMS)
    years = pd.get_dummies(release_years(paired), prefix="year", dtype=int)
    in_focus = focus.any(axis=1).groupby(groups).transform("any")
    labels = pd.concat([focus, pairs, years], axis=1)[in_focus]
    assert pairs.columns.tolist() == ["math & graphs"]
    assert folds[in_focus].equals(stratified_group_folds(labels, groups[in_focus], 5, 0))


def test_two_stage_folds_skips_an_empty_stratum(problems, groups):
    all_focus = problems.assign(tags=[["math"]] * len(problems))

    folds = two_stage_folds(all_focus, groups, n_folds=5, seed=0)

    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_make_split_uses_one_seed_for_the_test_fold_and_the_dev_folds(problems, groups):
    split = make_split(problems, groups, seed=3)

    is_test = two_stage_folds(problems, groups, seed=3) == SPLIT_TEST_FOLD
    assert (split["part"] == "test").equals(is_test.rename("part"))
    dev = ~is_test
    assert split.loc[dev, "fold"].equals(two_stage_folds(problems[dev], groups[dev], seed=3))


def test_make_split_stores_the_near_duplicate_group_as_int64(problems, groups):
    split = make_split(problems, groups.astype("int32"), seed=0)  # connected_components' dtype

    assert split.columns.tolist() == ["part", "fold", "group"]
    assert split["group"].dtype == "int64"
    assert split["group"].tolist() == groups.tolist()


def test_freeze_split_saves_a_split_that_load_split_reads_back(problems, groups, tmp_path):
    split = make_split(problems, groups, seed=0)
    path = tmp_path / "assets" / "split.csv"  # the folder is created

    assert freeze_split(split, path) == "saved 100 problems to split.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == "sample_id,part,fold,group"
    assert load_split(path).equals(split)


def test_freeze_split_checks_an_existing_file_instead_of_overwriting_it(problems, groups, tmp_path):
    split = make_split(problems, groups, seed=0)
    path = tmp_path / "split.csv"
    freeze_split(split, path)

    assert freeze_split(split, path) == "split.csv matches the split (100 problems)"


def test_freeze_split_raises_when_the_split_differs_from_the_file(problems, groups, tmp_path):
    path = tmp_path / "split.csv"
    freeze_split(make_split(problems, groups, seed=0), path)
    frozen = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="make_split no longer gives the frozen split in"):
        freeze_split(make_split(problems, groups, seed=1), path)
    assert path.read_text(encoding="utf-8") == frozen  # the frozen file is kept


def test_load_split_names_a_missing_column(tmp_path):
    path = tmp_path / "split.csv"
    pd.DataFrame({"sample_id": [10], "part": ["dev"], "fold": [0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no 'group' column.*03_train_test_split"):
        load_split(path)


def test_load_split_rejects_a_hand_edited_file_that_splits_a_group(problems, groups, tmp_path):
    path = tmp_path / "split.csv"
    freeze_split(make_split(problems, groups, seed=0), path)
    edited = pd.read_csv(path, index_col="sample_id")
    other_part = {"dev": "test", "test": "dev"}[edited.loc[0, "part"]]
    edited.loc[1, "part"] = other_part  # problem 1 is in problem 0's group
    edited.to_csv(path)

    with pytest.raises(ValueError, match="group 0 is split across parts"):
        load_split(path)


def test_load_split_rejects_a_group_split_across_dev_folds(tmp_path):
    path = tmp_path / "split.csv"
    rows = {"sample_id": [10, 11, 12], "part": ["dev"] * 3, "fold": [1, 2, 0], "group": [7, 7, 8]}
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="group 7 is split across folds"):
        load_split(path)


def test_select_part_keeps_the_problems_order_and_appends_fold_and_group(problems, groups):
    split = make_split(problems, groups, seed=0)
    reversed_problems = problems.iloc[::-1]

    dev = select_part(reversed_problems, split.sample(frac=1, random_state=0), "dev")

    in_dev = split.loc[reversed_problems.index, "part"] == "dev"
    assert dev.index.equals(reversed_problems.index[in_dev])
    assert dev.columns.tolist() == ["tags", "prob_desc_created_at", "fold", "group"]
    assert dev[["fold", "group"]].equals(split.loc[dev.index, ["fold", "group"]])


def test_select_part_splits_the_problems_into_dev_and_test(problems, groups):
    split = make_split(problems, groups, seed=0)

    dev, test = select_part(problems, split, "dev"), select_part(problems, split, "test")

    assert len(dev) + len(test) == len(problems)
    assert (test["fold"] == -1).all()
    assert sorted(dev["fold"].unique()) == [0, 1, 2, 3, 4]


def test_select_part_rejects_an_unknown_part(problems, groups):
    split = make_split(problems, groups, seed=0)

    with pytest.raises(ValueError, match="Unknown part 'train'"):
        select_part(problems, split, "train")


def test_select_part_counts_the_problems_missing_on_each_side(problems, groups):
    split = make_split(problems, groups, seed=0)  # problems 0 to 99

    with pytest.raises(ValueError, match="3 problems are not in the split, 2 split rows"):
        select_part(problems.iloc[2:], split.iloc[:97], "dev")
