"""Tests for the tables of notebook 11: F1 curves, their peaks and Platt-mapped probabilities."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit  # pylint: disable=no-name-in-module

from codeforces_tags.analysis.calibrators import calibrated_probabilities
from codeforces_tags.analysis.threshold_rules import (
    PEAK_COLUMNS,
    curve_peaks,
    f1_curve,
    fold_curves,
    grid_cuts,
    platt_probabilities,
)
from codeforces_tags.config import PLATT_CLIP
from codeforces_tags.model.platt_lipton import platt_fit
from codeforces_tags.model.thresholds import best_threshold


@pytest.fixture(name="grid_scores")
def grid_scores_fixture():
    """Return 60 problems' scores of two tags on Jev's 0.01 grid, their labels and four folds.

    Each label is drawn with the probability its score says, so the scores are roughly calibrated
    and every fold holds positives and negatives of both tags. Problem 0's math score is a failed
    answer's 0.0.
    """
    rng = np.random.default_rng(0)
    index = pd.Index(range(100, 160), name="sample_id")
    values = rng.integers(1, 100, size=(60, 2)) / 100
    values[0, 0] = 0.0
    scores = pd.DataFrame(values, index=index, columns=["math", "graphs"])
    labels = pd.DataFrame(
        (rng.random((60, 2)) < values).astype(int), index=index, columns=["math", "graphs"]
    )
    folds = pd.Series(np.repeat([0, 1, 2, 3], 15), index=index, name="fold")
    return scores, labels, folds


def test_grid_cuts_lie_halfway_between_the_values_of_jevs_grid():
    cuts = grid_cuts()

    assert len(cuts) == 100
    assert (cuts[0], cuts[12], cuts[-1]) == (0.005, 0.125, 0.995)
    assert grid_cuts(0.25).tolist() == [0.125, 0.375, 0.625, 0.875]


@pytest.mark.parametrize("grid", [0, -0.1, 0.3, 1.5])
def test_grid_cuts_refuse_a_step_that_does_not_divide_one(grid):
    with pytest.raises(ValueError, match="divides 1"):
        grid_cuts(grid)


def test_f1_curve_predicts_every_score_at_or_above_each_cut():
    scores = pd.Series([0.1, 0.2, 0.2, 0.6, 0.9], name="math")
    labels = pd.Series([0, 1, 0, 1, 1])

    curve = f1_curve(scores, labels, grid=0.25)

    # 3 positives. At 0.125, 4 predicted with 3 right: 6 / 7; at 0.375, 2 and 2: 4 / 5; at 0.625
    # and 0.875, 0.9 alone: 2 / 4.
    assert curve.index.tolist() == [0.125, 0.375, 0.625, 0.875]
    assert curve.index.name == "threshold"
    assert curve.name == "math"
    assert curve.tolist() == pytest.approx([6 / 7, 4 / 5, 1 / 2, 1 / 2])


def test_the_highest_point_of_f1_curve_is_the_f1_of_best_threshold(grid_scores):
    scores, labels, _ = grid_scores
    for tag in scores.columns:
        values, truth = scores[tag].to_numpy(), labels[tag].to_numpy()
        cut = best_threshold(values, truth)
        predicted = values >= cut
        f1 = 2 * (predicted & (truth == 1)).sum() / (predicted.sum() + truth.sum())

        assert f1_curve(scores[tag], labels[tag]).max() == pytest.approx(f1, abs=1e-12)


def test_f1_curve_is_zero_for_a_tag_without_positive_even_where_nothing_is_predicted():
    curve = f1_curve(pd.Series([0.1, 0.2]), pd.Series([0, 0]), grid=0.25)

    assert curve.tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (pd.Series([0, 1], index=[5, 6]), "index of scores"),
        (pd.Series([0, 2]), "only 0 and 1"),
    ],
)
def test_f1_curve_refuses_misaligned_or_non_binary_labels(labels, message):
    with pytest.raises(ValueError, match=message):
        f1_curve(pd.Series([0.1, 0.2]), labels)


def test_fold_curves_read_each_folds_curve_on_the_other_folds(grid_scores):
    scores, labels, folds = grid_scores

    curves = fold_curves(scores["math"], labels["math"], folds)

    assert curves.columns.tolist() == [0, 1, 2, 3]
    assert curves.columns.name == "fold"
    others = (folds != 2).to_numpy()
    expected = f1_curve(scores["math"][others], labels["math"][others])
    pd.testing.assert_series_equal(curves[2], expected, check_names=False)


def test_fold_curves_refuse_folds_not_indexed_like_the_scores(grid_scores):
    scores, labels, folds = grid_scores

    with pytest.raises(ValueError, match="one fold per problem"):
        fold_curves(scores["math"], labels["math"], folds.iloc[::-1])


def test_fold_curves_refuse_a_problem_without_a_fold(grid_scores):
    scores, labels, folds = grid_scores
    # A missing fold would be a column of its own: the curve of every problem.
    missing = folds.astype(float).mask(folds.index < 102)

    with pytest.raises(ValueError, match="^2 problems have no fold"):
        fold_curves(scores["math"], labels["math"], missing)


def test_fold_curves_refuse_labels_not_indexed_like_the_scores(grid_scores):
    scores, labels, folds = grid_scores

    with pytest.raises(ValueError, match="index of scores"):
        fold_curves(scores["math"], labels["math"].reset_index(drop=True), folds)


def test_curve_peaks_groups_the_cuts_near_the_maximum_into_runs():
    curve = pd.Series(
        [0.5, 0.76, 0.755, 0.6, 0.62, 0.758, 0.758, 0.4],
        index=pd.Index([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], name="threshold"),
    )

    peaks = curve_peaks(curve, within=0.01)

    # Two runs within 0.01 of 0.76: 0.2-0.3, highest at 0.2, and 0.6-0.7, tied, so the highest
    # cut, 0.7, as best_threshold breaks a tie.
    assert peaks.columns.tolist() == list(PEAK_COLUMNS)
    assert peaks.index.name == "peak"
    assert peaks[["start", "end", "threshold"]].to_numpy().tolist() == [
        [0.2, 0.3, 0.2],
        [0.6, 0.7, 0.7],
    ]
    assert peaks["f1"].tolist() == [0.76, 0.758]
    assert peaks["below_max"].tolist() == pytest.approx([0.0, 0.002])


def test_curve_peaks_with_no_margin_keeps_the_maximum_alone():
    curve = pd.Series([0.2, 0.6, 0.5], index=[0.125, 0.375, 0.625])

    peaks = curve_peaks(curve, within=0)

    assert peaks[["start", "end", "threshold", "f1", "below_max"]].to_numpy().tolist() == [
        [0.375, 0.375, 0.375, 0.6, 0.0]
    ]


@pytest.mark.parametrize(
    ("curve", "within", "message"),
    [
        (pd.Series([], dtype=float), 0.01, "at least one F1"),
        (pd.Series([0.5, math.nan]), 0.01, "no NaN"),
        (pd.Series([0.5]), -0.01, "0 or more"),
    ],
)
def test_curve_peaks_refuses_an_empty_curve_a_nan_or_a_negative_margin(curve, within, message):
    with pytest.raises(ValueError, match=message):
        curve_peaks(curve, within=within)


def test_platt_probabilities_map_each_fold_with_the_rules_platt_fit_on_the_other_folds(
    grid_scores,
):
    scores, labels, folds = grid_scores

    mapped = platt_probabilities(scores, labels, folds)

    assert mapped.index.equals(scores.index)
    assert mapped.columns.equals(scores.columns)
    held_out = (folds == 0).to_numpy()
    for tag in scores.columns:
        slope, intercept = platt_fit(
            scores.loc[~held_out, tag].to_numpy(), labels.loc[~held_out, tag].to_numpy()
        )
        clipped = np.clip(scores.loc[held_out, tag].to_numpy(), PLATT_CLIP, 1 - PLATT_CLIP)
        expected = expit(slope * logit(clipped) + intercept)
        np.testing.assert_allclose(mapped.loc[held_out, tag], expected, rtol=0, atol=1e-6)
    # The failed answer's 0.0 is clipped before its logit: its probability is a number.
    assert 0 < mapped.iloc[0, 0] < 1


def test_platt_probabilities_are_the_platt_logit_calibrators_to_the_last_bit(grid_scores):
    scores, labels, folds = grid_scores

    mapped = platt_probabilities(scores, labels, folds)

    compared = calibrated_probabilities(scores, labels, folds)["platt_logit"]
    pd.testing.assert_frame_equal(mapped, compared, check_exact=True)


def test_platt_probabilities_refuse_a_margin_as_the_platt_feature_does(grid_scores):
    scores, labels, folds = grid_scores
    # From -2 to 2: a margin, not a probability.
    margins = scores.assign(math=4 * scores["math"] - 2)

    with pytest.raises(ValueError, match=r"^A Platt map reads scores in \[0, 1\]"):
        platt_probabilities(margins, labels, folds)


def test_platt_probabilities_refuse_a_tag_with_one_class_outside_a_fold(grid_scores):
    scores, labels, folds = grid_scores
    labels = labels.assign(graphs=0)

    with pytest.raises(ValueError, match="one class"):
        platt_probabilities(scores, labels, folds)
