"""Tests for the Platt + Lipton threshold rule and its cross-fitted estimator."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logit  # pylint: disable=no-name-in-module
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted

from codeforces_tags.config import (
    METHODS_ASSETS_DIR,
    PLATT_CLIP,
    PLATT_SOLVER,
    PLATT_TOL,
    SCORE_GRID,
)
from codeforces_tags.evaluation.calibration import platt_calibrator
from codeforces_tags.methods import MODEL_FOLDER, load_method
from codeforces_tags.model.platt_lipton import (
    RULE_SETTINGS,
    PlattLiptonThresholds,
    apply_fold_thresholds,
    best_f1,
    check_unit_interval,
    cross_fitted_lipton_thresholds,
    cross_fitted_rows,
    lipton_threshold,
    lipton_thresholds,
    platt_feature,
    platt_fit,
    rule_calibrator,
    snap_to_grid,
)
from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    best_threshold,
    cross_fitted_thresholds,
    per_tag_thresholds,
)

# Jev's grid of probabilities, 0.01 to 0.99.
GRID = np.round(np.arange(1, 100) / 100, 2)


def calibrated_case():
    """Return 900 problems in nine groups of 100: probability p, exactly 100 p positives.

    The probabilities are calibrated, so their Platt map on the logit is the identity (a = 1,
    b = 0). Predicting the groups from 0.4 up gives the best F1, 2 * 390 / (600 + 450) = 52/70;
    half of it is 0.3714..., whose snapped threshold is 0.375.
    """
    probabilities = np.round(np.arange(1, 10) / 10, 2)
    scores = np.repeat(probabilities, 100)
    labels = np.concatenate([np.arange(100) < round(100 * p) for p in probabilities])
    return scores, labels


def random_grid_case(seed, size=300):
    """Return scores on Jev's grid and labels drawn with those probabilities, squared."""
    rng = np.random.default_rng(seed)
    scores = rng.choice(GRID, size=size)
    labels = rng.random(size) < scores**2
    labels[:2] = (True, False)  # both classes, whatever the draw
    return scores, labels


def rare_positive_case():
    """Return 2,000 scores on Jev's grid and their labels, of which only 55 are positive.

    Each label is drawn with probability ``expit(logit(score) / 2 - 4)``. On so rare a class and
    a feature as wide as the logit, unstandardised, lbfgs at scikit-learn's default tolerance
    stops visibly short of the maximum of the likelihood.
    """
    rng = np.random.default_rng(1)
    scores = rng.integers(1, 100, size=2000) / 100
    labels = (rng.random(2000) < expit(logit(scores) / 2 - 4)).astype(int)
    return scores, labels


def maximum_likelihood(feature, labels):
    """Return the slope and intercept of ``expit(a x + b)`` that maximise the likelihood.

    Found by scipy's trust-region Newton method on the negative log-likelihood, its gradient and
    its Hessian, to a gradient below 1e-9: a fit independent of scikit-learn's solvers.
    """
    design = np.column_stack([feature, np.ones_like(feature)])

    def negative_log_likelihood(weights):
        logits = design @ weights
        return np.sum(np.logaddexp(0, logits) - labels * logits)

    def gradient(weights):
        return design.T @ (expit(design @ weights) - labels)

    def hessian(weights):
        probabilities = expit(design @ weights)
        return design.T @ (design * (probabilities * (1 - probabilities))[:, np.newaxis])

    found = minimize(
        negative_log_likelihood,
        np.zeros(2),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-9},
    )
    assert found.success
    return found.x


def brute_force_best_f1(scores, labels):
    """Return the best F1 of predicting the scores at or above one of them."""
    best = 0.0
    for candidate in np.unique(scores):
        predicted = scores >= candidate
        best = max(best, 2 * np.sum(predicted & labels) / (np.sum(predicted) + np.sum(labels)))
    return best


def test_the_rule_settings_are_json_values_of_the_rule():
    assert RULE_SETTINGS == {
        "name": "platt-lipton",
        "feature": "logit",
        "clip": 1e-4,
        "C": "inf",
        "solver": "newton-cholesky",
        "tol": 1e-12,
        "grid": 0.01,
    }
    # Built from the project-wide settings that notebook 11's helpers read too.
    recorded = [RULE_SETTINGS[key] for key in ("clip", "solver", "tol", "grid")]
    assert recorded == [PLATT_CLIP, PLATT_SOLVER, PLATT_TOL, SCORE_GRID]
    # MLflow flattens a recipe's mappings by their dots: a dotted key would be one value.
    assert not [key for key in RULE_SETTINGS if "." in key]


def test_rule_calibrator_is_a_fresh_unfitted_unpenalised_map_fitted_to_convergence():
    calibrator = rule_calibrator()

    assert isinstance(calibrator, LogisticRegression)
    assert calibrator is not rule_calibrator()
    with pytest.raises(NotFittedError):
        check_is_fitted(calibrator)
    params = calibrator.get_params()
    assert np.isposinf(params["C"])
    assert (params["solver"], params["tol"], params["fit_intercept"]) == (
        PLATT_SOLVER,
        PLATT_TOL,
        True,
    )


def test_platt_feature_is_the_logit_of_the_scores_clipped_away_from_0_and_1():
    scores = pd.Series([0.0, 1e-5, 0.25, 0.5, 0.99, 1.0])

    feature = platt_feature(scores)

    assert isinstance(feature, np.ndarray)
    clipped = [PLATT_CLIP, PLATT_CLIP, 0.25, 0.5, 0.99, 1 - PLATT_CLIP]
    np.testing.assert_allclose(feature, logit(clipped))
    # A failed answer's 0.0 and a certain 1.0 get finite logits, symmetric around 0.5's 0.
    assert np.isfinite(feature).all()
    assert feature[0] == pytest.approx(-feature[-1])
    assert feature[3] == 0.0


@pytest.mark.parametrize(
    ("scores", "outside"),
    [([0.0, 0.5, 1.0, 1.2], 1), ([-0.1, 0.3, 1.5, np.nan], 3), ([np.nan, 0.7], 1)],
)
def test_platt_feature_refuses_scores_outside_0_and_1_counting_a_nan(scores, outside):
    message = rf"scores in \[0, 1\]: {outside} of {len(scores)} scores lie outside it, or are NaN"

    with pytest.raises(ValueError, match=message):
        platt_feature(np.array(scores))


def test_check_unit_interval_keeps_0_and_1_and_names_the_reader_of_scores_outside():
    check_unit_interval(np.array([[0.0, 1.0], [0.5, 0.25]]), "A Platt map")

    message = r"^Beta calibration reads scores in \[0, 1\]: 2 of 4 scores lie outside it"
    with pytest.raises(ValueError, match=message):
        check_unit_interval(np.array([[0.0, -1.0], [np.inf, 0.25]]), "Beta calibration")


def test_platt_fit_on_calibrated_probabilities_is_the_identity():
    scores, labels = calibrated_case()

    slope, intercept = platt_fit(scores, labels)

    assert slope == pytest.approx(1.0, abs=1e-4)
    assert intercept == pytest.approx(0.0, abs=1e-4)


def test_platt_fit_is_the_maximum_likelihood_map_where_lbfgs_stops_short():
    scores, labels = rare_positive_case()
    feature = platt_feature(scores)[:, np.newaxis]
    exact = maximum_likelihood(feature[:, 0], labels)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # converged: no ConvergenceWarning, no LinAlgWarning
        slope, intercept = platt_fit(scores, labels)

    np.testing.assert_allclose([slope, intercept], exact, rtol=0, atol=1e-7)
    # The case is built for it: lbfgs at its default tolerance, evaluation.calibration's Platt
    # map, stops more than 1e-4 away from the rule's converged map, enough to move a threshold.
    early, converged = (
        model.fit(feature, labels) for model in (platt_calibrator(), rule_calibrator())
    )
    gaps = [
        early.coef_[0, 0] - converged.coef_[0, 0],
        early.intercept_[0] - converged.intercept_[0],
    ]
    assert np.abs(gaps).max() > 1e-4


def test_platt_fit_converges_without_a_warning_on_scores_that_separate_the_classes():
    scores, labels = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), np.array([0, 0, 0, 1, 1, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slope, _ = platt_fit(scores, labels)

    assert slope > 0


def test_platt_fit_reads_the_logit_of_the_scores():
    scores, labels = calibrated_case()
    squared = expit(2 * logit(scores))  # the logit of each score, doubled

    slope, intercept = platt_fit(squared, labels)

    assert slope == pytest.approx(0.5, abs=1e-4)
    assert intercept == pytest.approx(0.0, abs=1e-4)


def test_platt_fit_clips_a_failed_answers_zero_to_a_finite_logit():
    scores, labels = calibrated_case()
    failed = np.append(scores, [0.0, 0.0])

    slope, intercept = platt_fit(failed, np.append(labels, [False, False]))

    assert math.isfinite(slope) and slope > 0
    assert math.isfinite(intercept)
    assert logit(PLATT_CLIP) == pytest.approx(-9.21, abs=1e-2)


def test_platt_fit_refuses_a_slope_at_or_below_zero():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    with pytest.raises(ValueError, match=r"slope is -0\.47\d*, not above 0: it would reverse"):
        platt_fit(scores, np.array([1, 0, 1, 0, 1, 0]))
    with pytest.raises(ValueError, match="not above 0"):
        lipton_threshold(scores, np.array([1, 1, 0, 1, 0, 0]))


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_platt_fit_refuses_labels_of_one_class(labels):
    with pytest.raises(ValueError, match="positive and negative problems"):
        platt_fit(np.array([0.2, 0.5, 0.8]), np.array(labels))


def test_platt_fit_refuses_scores_and_labels_of_different_lengths():
    with pytest.raises(ValueError, match="3 scores but 2 labels"):
        platt_fit(np.array([0.2, 0.5, 0.8]), np.array([0, 1]))


@pytest.mark.parametrize(
    ("threshold", "snapped"),
    [
        (0.752, 0.755),
        (0.6531, 0.655),
        (0.70, 0.695),  # a grid value itself is predicted
        (0.7000001, 0.705),  # just above one is not
        (0.0, 0.005),
        (-1.0, 0.005),
        (1e-6, 0.005),
        (0.99, 0.985),
        (0.999, 0.995),
        (1.0, 0.995),
        (math.inf, math.inf),
    ],
)
def test_snap_to_grid_takes_the_midpoint_below_the_smallest_grid_value_at_or_above(
    threshold, snapped
):
    assert snap_to_grid(threshold) == snapped


@pytest.mark.parametrize(
    ("threshold", "snapped"),
    [
        (0.07000000000000002, 0.065),  # 0.07 and the float noise of expit and logit
        (0.07 + 0.5e-11, 0.065),  # half the slack, 0.5e-9 grid steps, above 0.07
        (0.07 + 2e-11, 0.075),  # twice the slack above 0.07: a real gap
    ],
)
def test_snap_to_grid_counts_a_threshold_within_1e_9_grid_steps_above_a_grid_value_as_it(
    threshold, snapped
):
    # The threshold itself lies above 0.07: it would not predict a score of 0.07.
    assert threshold > 0.07
    assert snap_to_grid(threshold) == snapped
    # Inside the window, it counts as 0.07, which its snapped threshold predicts; beyond, not.
    assert (snapped <= 0.07) is (threshold - 0.07 <= 1e-9 * SCORE_GRID)
    # 0.07 / 0.01 is 7.000000000000001 in floats: a grid value itself needs the slack too.
    assert 0.07 / SCORE_GRID > 7
    assert snap_to_grid(0.07) == 0.065


@pytest.mark.parametrize("seed", range(20))
def test_snap_to_grid_predicts_on_grid_scores_what_the_threshold_predicts(seed):
    thresholds = np.random.default_rng(seed).random(50)

    for threshold in thresholds:
        snapped = snap_to_grid(float(threshold))
        assert np.array_equal(snapped <= GRID, threshold <= GRID)
        assert snapped == round(snapped, 3)


def test_a_snapped_threshold_never_predicts_a_failed_answer():
    for threshold in (-1.0, 0.0, 1e-9, 0.004, 0.01, 0.5):
        assert snap_to_grid(threshold) > 0.0


def test_best_f1_is_the_f1_at_the_best_threshold():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 0, 1, 0])

    found = best_f1(scores, labels)

    # 2 positives. From 0.9 down: F1 2/3, then 2/4, 4/5 and 4/6: the best cut keeps 0.3.
    assert best_threshold(scores, labels) == 0.25
    assert found == pytest.approx(4 / 5)
    assert found == pytest.approx(max(f1_score(labels, scores >= cut) for cut in scores))
    assert isinstance(found, float)


@pytest.mark.parametrize("seed", range(10))
def test_best_f1_is_the_highest_f1_of_any_cut(seed):
    scores, labels = random_grid_case(seed)

    assert best_f1(scores, labels) == pytest.approx(brute_force_best_f1(scores, labels))


def test_best_f1_is_0_without_a_positive():
    scores, labels = np.array([0.9, 0.5, 0.1]), np.array([0, 0, 0])

    assert best_threshold(scores, labels) == math.inf
    assert best_f1(scores, labels) == 0.0
    assert max(f1_score(labels, scores >= cut, zero_division=0) for cut in scores) == 0.0


def test_best_f1_refuses_scores_and_labels_of_different_lengths():
    with pytest.raises(ValueError, match="3 scores but 2 truth values"):
        best_f1(np.array([0.2, 0.5, 0.8]), np.array([0, 1]))


def test_lipton_threshold_on_calibrated_probabilities_is_half_the_best_f1_snapped():
    scores, labels = calibrated_case()

    threshold = lipton_threshold(scores, labels)

    best = brute_force_best_f1(scores, labels)
    assert best == pytest.approx(780 / 1050)
    assert threshold == snap_to_grid(best / 2) == 0.375
    assert np.array_equal(scores >= threshold, scores >= 0.4)


def test_lipton_threshold_reads_the_calibrated_probability_not_the_raw_score():
    scores, labels = calibrated_case()
    squared = expit(2 * logit(scores))

    threshold = lipton_threshold(squared, labels)

    # The Platt map undoes the doubled logit: the cut is half the best F1 on its scale.
    assert threshold == snap_to_grid(float(expit(2 * logit(780 / 1050 / 2)))) == 0.255
    assert np.array_equal(squared >= threshold, scores >= 0.375)


@pytest.mark.parametrize("seed", range(50))
def test_lipton_threshold_predicts_where_the_platt_probability_reaches_half_the_best_f1(seed):
    scores, labels = random_grid_case(seed)

    threshold = lipton_threshold(scores, labels)

    slope, intercept = platt_fit(scores, labels)
    calibrated = expit(slope * logit(scores) + intercept)
    half_best = brute_force_best_f1(scores, labels) / 2
    assert np.array_equal(scores >= threshold, calibrated >= half_best)


def test_lipton_threshold_is_inf_without_a_positive():
    assert lipton_threshold(np.array([0.2, 0.9, 0.5]), np.array([0, 0, 0])) == math.inf


def test_lipton_threshold_refuses_a_tag_of_every_problem_and_lengths_that_differ():
    with pytest.raises(ValueError, match="positive and negative problems"):
        lipton_threshold(np.array([0.2, 0.9, 0.5]), np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="3 scores but 2 truth values"):
        lipton_threshold(np.array([0.2, 0.9, 0.5]), np.array([1, 0]))


@pytest.fixture(name="sample_ids")
def sample_ids_fixture():
    """Return the sample ids of 200 dev problems."""
    return pd.Index(range(1000, 1200), name="sample_id")


@pytest.fixture(name="scores")
def scores_fixture(sample_ids):
    """Return probabilities on Jev's grid for three tags."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.choice(GRID, size=(len(sample_ids), 3)),
        index=sample_ids,
        columns=pd.Index(["math", "games", "trees"], name="label"),
    )


@pytest.fixture(name="labels")
def labels_fixture(scores):
    """Return 0/1 labels drawn with the squared probabilities; no problem has trees."""
    rng = np.random.default_rng(8)
    labels = (rng.random(scores.shape) < scores.to_numpy() ** 2).astype(int)
    return pd.DataFrame(labels, index=scores.index, columns=scores.columns).assign(trees=0)


@pytest.fixture(name="folds")
def folds_fixture(sample_ids):
    """Assign the problems to the five dev folds 4, 3, 2, 1, 0, 4, 3, ..."""
    return pd.Series(4 - np.arange(len(sample_ids)) % 5, index=sample_ids, name="fold")


def test_lipton_thresholds_choose_each_tag_on_its_own_column(scores, labels):
    thresholds = lipton_thresholds(scores, labels)

    assert thresholds.name == "threshold"
    assert thresholds.index.equals(scores.columns)
    for tag in ("math", "games"):
        assert thresholds[tag] == lipton_threshold(scores[tag], labels[tag])
    assert thresholds["trees"] == math.inf


@pytest.mark.parametrize("change", ["index", "columns"])
def test_lipton_thresholds_refuse_labels_of_other_problems_or_tags(scores, labels, change):
    other = labels.iloc[::-1] if change == "index" else labels[list(reversed(labels.columns))]

    with pytest.raises(ValueError, match="the same problems and tags, in order"):
        lipton_thresholds(scores, other)


def test_cross_fitted_rows_give_each_fold_the_choice_on_the_other_folds_only(scores, labels, folds):
    seen = []

    def column_means(fold_scores, fold_labels):
        """Record the folds of the rows chosen on, and return each tag's mean score."""
        assert fold_labels.index.equals(fold_scores.index)
        seen.append(set(folds[fold_scores.index]))
        return fold_scores.mean()

    rows = cross_fitted_rows(scores, labels, folds, column_means)

    assert seen == [set(range(5)) - {fold} for fold in range(5)]
    assert rows.index.tolist() == [0, 1, 2, 3, 4]
    assert rows.index.name == "fold"
    assert rows.columns.equals(scores.columns)
    for fold in range(5):
        expected = scores[folds != fold].mean()
        pd.testing.assert_series_equal(rows.loc[fold], expected, check_names=False)


def test_cross_fitted_rows_of_the_f1_argmax_are_the_cross_fitted_thresholds(scores, labels, folds):
    rows = cross_fitted_rows(scores, labels, folds, per_tag_thresholds)

    pd.testing.assert_frame_equal(rows, cross_fitted_thresholds(scores, labels, folds))


@pytest.mark.parametrize("name", ["labels", "folds"])
def test_cross_fitted_rows_refuse_labels_or_folds_of_other_problems(scores, labels, folds, name):
    aligned = {"labels": labels, "folds": folds}
    aligned[name] = aligned[name].iloc[::-1]

    with pytest.raises(ValueError, match=f"{name} must have the index of scores"):
        cross_fitted_rows(scores, aligned["labels"], aligned["folds"], per_tag_thresholds)


def test_cross_fitted_lipton_thresholds_choose_each_fold_on_the_four_other_folds(
    scores, labels, folds
):
    rows = cross_fitted_lipton_thresholds(scores, labels, folds)

    assert rows.index.tolist() == [0, 1, 2, 3, 4]
    assert rows.index.name == "fold"
    assert rows.columns.equals(scores.columns)
    for fold in range(5):
        others = folds != fold
        expected = lipton_thresholds(scores[others], labels[others])
        pd.testing.assert_series_equal(rows.loc[fold], expected, check_names=False)


def test_cross_fitted_lipton_thresholds_ignore_the_labels_of_the_fold_they_apply_to(
    scores, labels, folds
):
    flipped = labels.assign(math=labels["math"].mask(folds == 2, 1 - labels["math"]))

    before = cross_fitted_lipton_thresholds(scores, labels, folds)
    after = cross_fitted_lipton_thresholds(scores, flipped, folds)

    pd.testing.assert_series_equal(after.loc[2], before.loc[2])
    assert not after.drop(index=2).equals(before.drop(index=2))


def test_cross_fitted_lipton_thresholds_refuse_folds_of_other_problems(scores, labels, folds):
    with pytest.raises(ValueError, match="folds must have the index of scores"):
        cross_fitted_lipton_thresholds(scores, labels, folds.iloc[::-1])


def test_apply_fold_thresholds_predicts_each_fold_with_its_own_row():
    index = pd.Index([10, 11, 12, 13], name="sample_id")
    scores = pd.DataFrame(
        {"math": [0.2, 0.6, 0.6, 0.2], "games": [0.9, 0.4, 0.3, 0.8]}, index=index
    )
    folds = pd.Series([0, 1, 0, 1], index=index, name="fold")
    fold_thresholds = pd.DataFrame(
        {"math": [0.5, 0.1], "games": [0.95, 0.35]}, index=pd.Index([0, 1], name="fold")
    )

    predictions = apply_fold_thresholds(scores, fold_thresholds, folds)

    # Fold 0 (problems 10 and 12) cuts at 0.5 and 0.95; fold 1 (11 and 13) at 0.1 and 0.35.
    expected = pd.DataFrame({"math": [0, 1, 1, 1], "games": [0, 1, 0, 1]}, index=index)
    pd.testing.assert_frame_equal(predictions, expected)


def test_apply_fold_thresholds_refuses_folds_of_other_problems_or_without_a_row(scores, folds):
    rows = pd.DataFrame(0.5, index=pd.Index([0, 1, 2, 3], name="fold"), columns=scores.columns)

    with pytest.raises(ValueError, match="folds must have the index of scores"):
        apply_fold_thresholds(scores, rows, folds.iloc[::-1])
    with pytest.raises(ValueError, match=r"The folds \[4\] have no row of thresholds"):
        apply_fold_thresholds(scores, rows, folds)


class ProbabilityColumns(BaseEstimator):
    """A scorer that learns nothing: its rows are already one probability per tag."""

    def fit(self, X, y):  # pylint: disable=unused-argument
        """Set ``classes_``, one per tag, as the score-estimator contract asks."""
        self.classes_ = np.arange(np.shape(y)[1])
        return self

    def decision_function(self, X):
        """Return the rows as they are."""
        return X.to_numpy(dtype=float)

    def predict(self, X):
        """Predict the tags whose score is at least 0."""
        return (self.decision_function(X) >= 0).astype(int)


def test_the_probability_columns_follow_the_score_estimator_contract(scores, labels):
    scorer = ProbabilityColumns().fit(scores, labels.to_numpy())

    assert scorer.classes_.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(scorer.decision_function(scores), scores.to_numpy())
    np.testing.assert_array_equal(scorer.predict(scores), np.ones(scores.shape, dtype=int))


def test_platt_lipton_thresholds_cross_fit_the_rule_on_out_of_fold_scores(scores, labels, folds):
    wrapper = PlattLiptonThresholds(ProbabilityColumns()).fit(scores, labels, groups=folds)

    rows = cross_fitted_lipton_thresholds(scores, labels, folds)
    expected = pd.concat(
        scores[folds == fold].ge(rows.loc[fold], axis="columns").astype(int) for fold in rows.index
    ).loc[scores.index]
    np.testing.assert_array_equal(wrapper.oof_scores_, scores.to_numpy())
    np.testing.assert_array_equal(wrapper.fold_thresholds_, rows.to_numpy())
    np.testing.assert_array_equal(wrapper.oof_predictions_, expected.to_numpy())
    np.testing.assert_array_equal(wrapper.thresholds_, lipton_thresholds(scores, labels).to_numpy())
    assert wrapper.oof_predictions_.dtype == np.int64
    assert wrapper.fold_ids_.tolist() == [0, 1, 2, 3, 4]
    assert wrapper.tags_ == ("math", "games", "trees")
    np.testing.assert_array_equal(
        wrapper.predict(scores), (scores.to_numpy() >= wrapper.thresholds_).astype(int)
    )


def test_platt_lipton_thresholds_choose_another_threshold_than_the_f1_argmax(scores, labels, folds):
    rule = PlattLiptonThresholds(ProbabilityColumns()).fit(scores, labels, groups=folds)
    argmax = CrossFittedTagThresholds(ProbabilityColumns()).fit(scores, labels, groups=folds)

    # The parent still chooses the F1 argmax: the subclass changed nothing of it.
    np.testing.assert_array_equal(argmax.thresholds_, per_tag_thresholds(scores, labels).to_numpy())
    assert not np.array_equal(rule.thresholds_, argmax.thresholds_)
    np.testing.assert_array_equal(rule.oof_scores_, argmax.oof_scores_)


def test_platt_lipton_thresholds_keep_the_parents_parameters_and_clone(scores, labels, folds):
    wrapper = PlattLiptonThresholds(ProbabilityColumns(), refit=False)

    copy = clone(wrapper)

    assert isinstance(copy, PlattLiptonThresholds)
    assert isinstance(copy, CrossFittedTagThresholds)
    assert copy.get_params(deep=False) == {"estimator": copy.estimator, "refit": False}
    copy.fit(scores, labels, groups=folds)
    assert not hasattr(copy, "estimator_")
    assert PlattLiptonThresholds.__init__ is CrossFittedTagThresholds.__init__
    assert PlattLiptonThresholds.fit is not CrossFittedTagThresholds.fit


def test_the_rule_refuses_margins_instead_of_clipping_them(scores, labels, folds):
    margins = 4 * scores - 2  # a linear model's margins: order kept, no longer probabilities

    with pytest.raises(ValueError, match=r"scores in \[0, 1\]"):
        platt_fit(margins["math"].to_numpy(), labels["math"].to_numpy())
    with pytest.raises(ValueError, match=r"scores in \[0, 1\]"):
        PlattLiptonThresholds(ProbabilityColumns()).fit(margins, labels, groups=folds)


def test_the_frozen_tfidf_model_still_loads_its_cross_fitted_tag_thresholds():
    # TF-IDF's model.skops names CrossFittedTagThresholds by its import path: the subclass lives
    # in a module of its own and leaves that class, and its frozen thresholds, as they were.
    method = load_method(METHODS_ASSETS_DIR / "tfidf" / MODEL_FOLDER)

    assert method.estimator.__class__ is CrossFittedTagThresholds
    assert np.isfinite(method.estimator.thresholds_).all()
