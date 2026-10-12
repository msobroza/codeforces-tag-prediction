"""Tests for the multi-label encoding."""

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.data.labels import focus_labels, frequent_pairs, multi_hot


def test_multi_hot_uses_vocabulary_order_and_ignores_unknown_tags():
    tags = pd.Series([["math", "dp"], [], ["games"]], index=[7, 8, 9])

    indicators = multi_hot(tags, vocabulary=["games", "math"])

    assert indicators.index.tolist() == [7, 8, 9]
    assert indicators.columns.tolist() == ["games", "math"]
    assert indicators.to_numpy().tolist() == [[0, 1], [0, 0], [1, 0]]


def test_multi_hot_defaults_to_sorted_observed_vocabulary():
    indicators = multi_hot(pd.Series([["math", "dp"], ["games"]]))

    assert indicators.columns.tolist() == ["dp", "games", "math"]


def test_multi_hot_accepts_sets_as_well_as_lists():
    indicators = multi_hot(pd.Series([{"sys", "math"}, set()]), vocabulary=["math"])

    assert indicators["math"].tolist() == [1, 0]


def test_frequent_pairs_marks_the_rows_with_both_labels_of_each_frequent_pair():
    labels = pd.DataFrame(
        {"math": [1, 1, 1, 0], "graphs": [1, 1, 0, 1], "trees": [0, 1, 0, 1]}, index=[5, 6, 7, 8]
    )

    pairs = frequent_pairs(labels, min_count=2)  # math & trees is on one row only

    assert pairs.columns.tolist() == ["math & graphs", "graphs & trees"]
    assert pairs.index.tolist() == [5, 6, 7, 8]
    assert pairs.to_numpy().tolist() == [[1, 0], [1, 1], [0, 0], [0, 1]]


def test_frequent_pairs_has_no_column_when_no_pair_is_frequent():
    labels = pd.DataFrame({"math": [1, 0], "graphs": [0, 1]})

    assert frequent_pairs(labels, min_count=1).shape == (2, 0)


def test_focus_labels_has_one_column_per_focus_tag_in_config_order():
    problems = pd.DataFrame(
        {"tags": [["games", "dp"], [], ["probabilities", "math"]]},
        index=pd.Index([4, 5, 6], name="sample_id"),
    )

    labels = focus_labels(problems)

    assert labels.columns.tolist() == list(FOCUS_TAGS)
    assert labels.index.equals(problems.index)
    assert labels.sum(axis=1).tolist() == [1, 0, 2]  # dp is not a focus tag
    assert labels.loc[6, ["math", "probabilities"]].tolist() == [1, 1]
