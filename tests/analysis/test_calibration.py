"""Tests for the reliability of a method's probabilities."""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.analysis.calibration import (
    RELIABILITY_COLUMNS,
    expected_calibration_error,
    reliability_table,
)
from codeforces_tags.config import CALIBRATION_BINS


@pytest.fixture(name="scores")
def scores_fixture():
    """Return the probabilities of two tags on eight problems.

    Math's are exact in binary, and three of them (0.25, 0.5 and 0.75) sit on an edge of four
    bins. Graphs' fall in the first and the third bin only.
    """
    return pd.DataFrame(
        {
            "math": [0.0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0],
            "graphs": [0.1, 0.2, 0.1, 0.2, 0.6, 0.7, 0.6, 0.7],
        },
        index=pd.Index(range(10, 18), name="sample_id"),
    )


@pytest.fixture(name="truth")
def truth_fixture():
    """Return the same problems' true tags."""
    return pd.DataFrame(
        {"math": [0, 0, 1, 0, 1, 1, 1, 0], "graphs": [0, 0, 0, 1, 1, 1, 0, 1]},
        index=pd.Index(range(10, 18), name="sample_id"),
    )


def test_reliability_table_bins_each_tags_probabilities_in_equal_widths(scores, truth):
    table = reliability_table(scores, truth, bins=4)

    # Four bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75) and [0.75, 1]. A bin includes its lower
    # edge, so math's 0.25 is in bin 1; math's 1.0 joins 0.75 and 0.875 in the last bin; graphs
    # has nothing in bins 1 and 3, which are left out.
    expected = pd.DataFrame(
        {
            "mean_probability": [0.0625, 0.25, 0.5625, 0.875, 0.15, 0.65],
            "positive_rate": [0.0, 1.0, 0.5, 2 / 3, 0.25, 0.75],
            "problems": [2, 1, 2, 3, 4, 4],
        },
        index=pd.MultiIndex.from_tuples(
            [("math", 0), ("math", 1), ("math", 2), ("math", 3), ("graphs", 0), ("graphs", 2)],
            names=["tag", "bin"],
        ),
    )
    pd.testing.assert_frame_equal(table, expected)
    assert list(table.columns) == list(RELIABILITY_COLUMNS)


def test_reliability_table_cuts_the_range_into_the_projects_ten_bins_by_default(scores, truth):
    table = reliability_table(scores, truth)

    assert CALIBRATION_BINS == 10
    assert table.loc["math"].index.tolist() == [0, 1, 2, 5, 6, 7, 8, 9]
    assert table.loc["math", "problems"].tolist() == [1] * 8


def test_a_probability_of_1_is_in_the_last_bin_and_empty_bins_are_left_out():
    # Three failed answers score exactly 0 and share bin 0; 0.95 and 1.0 share bin 9.
    scores = pd.DataFrame({"math": [0.0, 0.0, 0.0, 0.95, 1.0]})
    truth = pd.DataFrame({"math": [0, 1, 0, 1, 1]})

    table = reliability_table(scores, truth)

    assert table.index.tolist() == [("math", 0), ("math", 9)]
    assert table["problems"].tolist() == [3, 2]
    assert table["mean_probability"].tolist() == pytest.approx([0.0, 0.975])
    assert table["positive_rate"].tolist() == pytest.approx([1 / 3, 1.0])


def test_every_problem_falls_in_one_bin_of_each_tag():
    rng = np.random.default_rng(0)
    index = pd.Index(range(300), name="sample_id")
    scores = pd.DataFrame(rng.random((300, 3)).round(1), index, ["math", "graphs", "games"])
    truth = pd.DataFrame((rng.random((300, 3)) < 0.3).astype(int), index, scores.columns)

    table = reliability_table(scores, truth)

    # The bins of a tag add up to its problems, its positives and the sum of its probabilities.
    sums = table[["mean_probability", "positive_rate"]].mul(table["problems"], axis=0)
    per_tag = sums.groupby("tag", sort=False).sum()
    assert table.groupby("tag", sort=False)["problems"].sum().tolist() == [300] * 3
    assert per_tag["positive_rate"].round().tolist() == truth.sum().tolist()
    assert per_tag["mean_probability"].tolist() == pytest.approx(scores.sum().tolist())


@pytest.mark.parametrize(
    ("change", "bins", "message"),
    [
        pytest.param(
            lambda scores, truth: (scores.iloc[1:], truth),
            4,
            "same problems and tags",
            id="fewer problems",
        ),
        pytest.param(
            lambda scores, truth: (scores[["graphs", "math"]], truth),
            4,
            "same problems and tags",
            id="tags in another order",
        ),
        pytest.param(
            lambda scores, truth: (scores + 0.01, truth),
            4,
            "from 0 to 1",
            id="a score above 1",
        ),
        pytest.param(
            lambda scores, truth: (scores - 0.01, truth),
            4,
            "from 0 to 1",
            id="a negative score",
        ),
        pytest.param(
            lambda scores, truth: (scores.where(scores != 0.25), truth),
            4,
            "none missing",
            id="a missing score",
        ),
        pytest.param(
            lambda scores, truth: (scores, truth * 2),
            4,
            "only 0 and 1",
            id="a true tag of 2",
        ),
        pytest.param(
            lambda scores, truth: (scores, truth.where(truth != 1)),
            4,
            "only 0 and 1",
            id="a missing true tag",
        ),
        pytest.param(lambda scores, truth: (scores, truth), 0, "at least 1", id="no bin"),
        pytest.param(
            lambda scores, truth: (scores.iloc[:0], truth.iloc[:0]),
            4,
            "no problem or no tag",
            id="no problem",
        ),
    ],
)
def test_reliability_table_refuses_misaligned_frames_scores_outside_0_1_nan_truth_not_0_1_no_bin(
    change, bins, message, scores, truth
):
    scores, truth = change(scores, truth)

    with pytest.raises(ValueError, match=message):
        reliability_table(scores, truth, bins)
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(scores, truth, bins)


def test_expected_calibration_error_weights_each_bins_gap_by_its_problems(scores, truth):
    error = expected_calibration_error(scores, truth, bins=4)

    # Math's bins have gaps 0.0625, 0.75, 0.0625 and 5/24 over 2, 1, 2 and 3 of its 8 problems:
    # (2 * 0.0625 + 1 * 0.75 + 2 * 0.0625 + 3 * 5 / 24) / 8. Graphs' two bins of 4 problems
    # both have a gap of 0.1.
    assert error.tolist() == pytest.approx([0.203125, 0.1])
    assert error.index.tolist() == ["math", "graphs"]
    assert error.index.name == "tag"
    assert error.name == "ece"


def test_a_perfectly_calibrated_score_has_no_calibration_error():
    # A quarter of the problems scored 0.25 have the tag, and three quarters of those scored 0.75.
    scores = pd.DataFrame({"math": [0.25] * 4 + [0.75] * 4})
    truth = pd.DataFrame({"math": [1, 0, 0, 0, 1, 1, 1, 0]})

    table = reliability_table(scores, truth, bins=2)

    assert table["mean_probability"].tolist() == table["positive_rate"].tolist()
    assert expected_calibration_error(scores, truth, bins=2).tolist() == pytest.approx([0.0])
