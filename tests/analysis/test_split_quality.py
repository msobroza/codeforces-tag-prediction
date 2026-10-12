"""Tests for the split quality measures and the random baselines."""

import pandas as pd
import pytest

from codeforces_tags.analysis.split_quality import (
    BETTER,
    MEASURE_LABELS,
    SPLIT_METHODS,
    compare_split_methods,
    leaked_pairs,
    random_group_test,
    random_problem_test,
    share_deviation,
    split_quality,
    stratified_group_test,
)
from codeforces_tags.config import SPLIT_MIN_PAIR_PROBLEMS
from codeforces_tags.splitting import two_stage_folds


def test_leaked_pairs_counts_pairs_across_dev_and_test():
    groups = pd.Series([0, 0, 1, 1, 2, 2, 2])
    is_test = pd.Series([True, False, True, True, True, False, False])

    assert leaked_pairs(groups, is_test) == 1 + 0 + 2


def test_share_deviation_is_relative_to_the_overall_share():
    assert share_deviation(0.22, 0.20) == pytest.approx(10)
    assert share_deviation(0.18, 0.20) == pytest.approx(10)
    assert share_deviation(pd.Series([0.5, 0.1]), pd.Series([0.5, 0.2])).tolist() == [0, 50]


def test_split_quality_measures_the_test_set(problems, groups):
    is_test = pd.Series(
        [True, False, False, False, True, True, False, False, False, False], index=problems.index
    )

    quality = split_quality(problems, groups, is_test)

    graphs = 100 * abs(1 / 3 - 0.1) / 0.1
    math = 100 * abs(1 / 3 - 0.4) / 0.4
    assert quality["leaked duplicate pairs"] == 1
    assert quality["max tag share deviation (%)"] == pytest.approx(graphs)
    assert quality["mean tag share deviation (%)"] == pytest.approx((math + graphs) / 2)
    assert quality["min test problems per tag"] == 1
    assert quality["no-focus share deviation (%)"] == pytest.approx(100 * abs(1 / 3 - 0.5) / 0.5)
    assert quality["year mix distance (%)"] == pytest.approx(100 * abs(2 / 3 - 0.5))
    assert quality["test share (%)"] == pytest.approx(30)
    assert pd.isna(quality["max tag-pair share deviation (%)"])  # no pair of tags is frequent


def test_split_quality_measures_the_frequent_tag_pairs():
    n = SPLIT_MIN_PAIR_PROBLEMS  # problems per pair: just frequent enough
    rare_pair = [["graphs", "trees"]]  # in test, but too rare to be measured
    tags = [["math", "graphs"]] * n + [["math", "trees"]] * n + rare_pair + [["greedy"]] * (n - 1)
    index = pd.Index(range(3 * n), name="sample_id")
    created = pd.Series(pd.Timestamp("2020-01-01", tz="UTC"), index=index)
    problems = pd.DataFrame({"tags": tags, "prob_desc_created_at": created}, index=index)
    in_test = [*range(6), *range(n, n + 3), *range(2 * n, 2 * n + 6)]  # 6, 3 and 6 of each kind
    is_test = pd.Series(index.isin(in_test), index=index)

    quality = split_quality(problems, pd.Series(index, index=index), is_test)

    graphs = 100 * abs(6 / 15 - 1 / 3) / (1 / 3)  # math & graphs: 20%
    trees = 100 * abs(3 / 15 - 1 / 3) / (1 / 3)  # math & trees: 40%
    assert quality["max tag-pair share deviation (%)"] == pytest.approx(trees)
    assert quality["mean tag-pair share deviation (%)"] == pytest.approx((graphs + trees) / 2)


def test_every_measure_has_a_better_direction_and_a_label(problems, groups):
    is_test = pd.Series([True] + [False] * 9, index=problems.index)

    measures = split_quality(problems, groups, is_test).index.tolist()

    assert measures == list(BETTER)
    assert MEASURE_LABELS["leaked duplicate pairs"] == "leaked duplicate pairs ↓"
    assert MEASURE_LABELS["min test problems per tag"] == "min test problems per tag ↑"
    assert MEASURE_LABELS["test share (%)"] == "test share (%) ≈ 20"


def test_random_problem_test_picks_one_problem_in_five(problems, groups):
    is_test = random_problem_test(problems, groups, seed=0)

    assert is_test.sum() == 2
    assert is_test.equals(random_problem_test(problems, groups, seed=0))


def test_random_group_test_keeps_groups_whole_and_takes_one_group_in_five():
    groups = pd.Series([k // 2 for k in range(50)] + list(range(25, 75)))  # 25 pairs, 50 alone
    problems = pd.DataFrame(index=groups.index)

    for seed in range(5):
        is_test = random_group_test(problems, groups, seed)
        assert is_test.groupby(groups).nunique().max() == 1
        assert is_test.groupby(groups).first().sum() == 15  # 75 groups / 5


def test_stratified_group_test_is_the_first_two_stage_fold(many_problems, many_groups):
    expected = two_stage_folds(many_problems, many_groups, seed=1) == 0

    assert stratified_group_test(many_problems, many_groups, seed=1).equals(expected)


def test_compare_split_methods_measures_each_method_on_each_seed(many_problems, many_groups):
    results = compare_split_methods(many_problems, many_groups, seeds=[0, 1])

    assert results.index.tolist() == [(name, seed) for name in SPLIT_METHODS for seed in (0, 1)]


def test_compare_split_methods_accepts_seeds_that_can_be_read_once(many_problems, many_groups):
    results = compare_split_methods(many_problems, many_groups, seeds=iter([0, 1]))

    assert len(results) == 2 * len(SPLIT_METHODS)
    assert results.columns.tolist() == list(BETTER)
