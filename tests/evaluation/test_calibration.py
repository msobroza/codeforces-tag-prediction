"""Tests for the calibration of out-of-fold scores."""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit  # pylint: disable=no-name-in-module  # a ufunc, built in C
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from codeforces_tags.config import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_METHODS,
    FOCUS_TAGS,
)
from codeforces_tags.evaluation.bootstrap import percentile_interval, resample_weights
from codeforces_tags.evaluation.calibration import (
    CALIBRATION_COLUMNS,
    LIPTON_COLUMNS,
    METRICS,
    POOLED,
    calibration_table,
    check_probabilities,
    cross_fitted_probabilities,
    isotonic_calibrator,
    lipton_table,
    mapped_predictions,
    minmax_probabilities,
    platt_calibrator,
    platt_maps,
    probabilities_by_method,
    sigmoid_probabilities,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.model.thresholds import (
    apply_thresholds,
    cross_fitted_predictions,
    per_tag_thresholds,
)


def platt_by_hand(train, truth, held_out):
    """Fit a Platt map on one tag's scores, and map the held-out ones."""
    fitted = platt_calibrator().fit(train.to_frame(), truth)
    return fitted.predict_proba(held_out.to_frame())[:, 1]


def isotonic_by_hand(train, truth, held_out):
    """Fit an isotonic map on one tag's scores, and map the held-out ones."""
    return isotonic_calibrator().fit(train.to_numpy(), truth).predict(held_out.to_numpy())


def test_platt_calibrator_is_an_unregularised_logistic_regression_that_fits_separable_scores():
    calibrator = platt_calibrator()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrator.fit(pd.DataFrame({"math": [-2.0, -1.0, 1.0, 2.0]}), [0, 0, 1, 1])

    assert isinstance(calibrator, LogisticRegression)
    assert np.isposinf(calibrator.C)
    assert calibrator.coef_[0, 0] > 0


def test_isotonic_calibrator_clips_held_out_scores_outside_the_range_it_was_fitted_on():
    # IsotonicRegression's default, out_of_bounds="nan", silently maps a held-out score outside the
    # range of the other folds' scores to NaN.
    scores, truth = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])
    held_out = np.array([-5.0, 1.5, 10.0])

    default = IsotonicRegression().fit(scores, truth).predict(held_out)
    clipped = isotonic_calibrator().fit(scores, truth).predict(held_out)

    assert np.isnan(default[[0, 2]]).all()
    assert clipped.tolist() == [0.0, 0.5, 1.0]


def test_sigmoid_probabilities_are_the_logistic_function_of_the_logits(calibration_data):
    logits = calibration_data.logits

    probabilities = sigmoid_probabilities(logits)

    pd.testing.assert_frame_equal(probabilities, 1 / (1 + np.exp(-logits)), rtol=1e-12)


def test_minmax_probabilities_scale_each_fold_on_the_other_folds_and_clip_to_0_1():
    index = pd.Index([11, 12, 13, 14, 15, 16], name="sample_id")
    scores = pd.DataFrame(
        {"math": [0.0, 10.0, 2.0, 4.0, -1.0, 5.0], "trees": [1.0, 3.0, 0.0, 2.0, 2.0, 4.0]},
        index=index,
    )
    folds = pd.Series([0, 0, 1, 1, 2, 2], index=index, name="fold")

    probabilities = minmax_probabilities(scores, folds)

    # Fold 0 of math: min -1 and max 5 on folds 1 and 2, so 0 -> 1/6 and 10 -> 11/6, clipped.
    expected = pd.DataFrame(
        {"math": [1 / 6, 1, 3 / 11, 5 / 11, 0, 0.5], "trees": [0.25, 0.75, 0, 1 / 3, 2 / 3, 1]},
        index=index,
    )
    pd.testing.assert_frame_equal(probabilities, expected, rtol=1e-12)


def test_minmax_probabilities_refuse_a_tag_constant_on_the_other_folds_of_a_fold():
    scores = pd.DataFrame({"math": [0.0, 1.0, 2.0, 3.0], "games": [5.0, 9.0, 1.0, 1.0]})

    with pytest.raises(ValueError, match=r"Outside fold 0, the tags \['games'\] have one score"):
        minmax_probabilities(scores, pd.Series([0, 0, 1, 1]))


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        (pd.Series([0, 1, -1, 1]), "negative fold"),
        (pd.Series([0, 0, 0, 0]), "at least two folds"),
        (pd.Series([0, 1, 0, 1], index=[3, 2, 1, 0]), "folds must have the same index"),
        # A problem never held out: its row was left as whatever the memory held.
        (pd.Series([np.nan, 0, 1, 1]), "1 problems have no fold"),
    ],
    ids=["test problems", "one fold", "misaligned folds", "a missing fold"],
)
def test_minmax_probabilities_refuse_folds_that_are_not_two_dev_folds_or_more(folds, message):
    scores = pd.DataFrame({"math": [0.0, 1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match=message):
        minmax_probabilities(scores, folds)


@pytest.mark.parametrize(
    ("calibrator", "by_hand"),
    [(platt_calibrator, platt_by_hand), (isotonic_calibrator, isotonic_by_hand)],
    ids=["platt", "isotonic"],
)
def test_cross_fitted_probabilities_map_each_fold_with_a_calibrator_fitted_on_the_other_folds(
    calibration_data, calibrator, by_hand
):
    scores, labels, folds = (
        calibration_data.svm_scores,
        calibration_data.labels,
        calibration_data.folds,
    )

    probabilities = cross_fitted_probabilities(scores, labels, folds, calibrator())

    expected = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    for fold in sorted(folds.unique()):
        held_out = folds == fold
        for tag in scores.columns:
            expected.loc[held_out, tag] = by_hand(
                scores.loc[~held_out, tag], labels.loc[~held_out, tag], scores.loc[held_out, tag]
            )
    pd.testing.assert_frame_equal(probabilities, expected, check_exact=True)


@pytest.mark.parametrize("label", [0, 1], ids=["no positive", "no negative"])
def test_cross_fitted_probabilities_name_the_tag_and_fold_with_one_class_outside_it(
    calibration_data, label
):
    labels = calibration_data.labels.copy()
    labels.loc[calibration_data.folds != 1, "games"] = label

    for calibrator in (platt_calibrator(), isotonic_calibrator()):
        with pytest.raises(ValueError, match=r"Outside fold 1, the tags \['games'\] have one"):
            cross_fitted_probabilities(
                calibration_data.svm_scores, labels, calibration_data.folds, calibrator
            )


def test_cross_fitted_probabilities_refuse_labels_of_other_tags_or_problems_and_test_problems(
    calibration_data,
):
    scores, labels, folds = (
        calibration_data.svm_scores,
        calibration_data.labels,
        calibration_data.folds,
    )

    with pytest.raises(ValueError, match="labels must have the columns of scores"):
        cross_fitted_probabilities(scores, labels[sorted(FOCUS_TAGS)], folds, platt_calibrator())
    with pytest.raises(ValueError, match="labels must have the same index"):
        cross_fitted_probabilities(scores, labels.iloc[::-1], folds, platt_calibrator())
    with pytest.raises(ValueError, match="negative fold"):
        cross_fitted_probabilities(scores, labels, folds.replace(2, -1), platt_calibrator())


def test_probabilities_by_method_give_the_four_probabilities_in_the_settings_order(
    calibration_data,
):
    data = calibration_data

    probabilities = probabilities_by_method(data.svm_scores, data.logits, data.labels, data.folds)

    assert list(probabilities) == list(CALIBRATION_METHODS)
    expected = {
        "logistic_regression": sigmoid_probabilities(data.logits),
        "svm_minmax": minmax_probabilities(data.svm_scores, data.folds),
        "svm_platt": cross_fitted_probabilities(
            data.svm_scores, data.labels, data.folds, platt_calibrator()
        ),
        "svm_isotonic": cross_fitted_probabilities(
            data.svm_scores, data.labels, data.folds, isotonic_calibrator()
        ),
    }
    for method, frame in expected.items():
        pd.testing.assert_frame_equal(probabilities[method], frame, check_exact=True)
    assert check_probabilities(probabilities, data.labels) is None


@pytest.mark.parametrize("scores", ["svm_scores", "logits"])
@pytest.mark.parametrize(
    "change",
    [lambda frame: frame[sorted(FOCUS_TAGS)], lambda frame: frame.iloc[::-1]],
    ids=["tags sorted by name", "problems reversed"],
)
def test_probabilities_by_method_refuse_scores_not_aligned_with_the_labels(
    calibration_data, scores, change
):
    # notebook 04's first look is fitted on every tag, which multi_hot sorts by name: its focus
    # logits are selected by name, logits[labels.columns], never by position.
    inputs = {"svm_scores": calibration_data.svm_scores, "logits": calibration_data.logits}
    inputs[scores] = change(inputs[scores])

    with pytest.raises(ValueError, match=f"{scores} must have the index and columns of labels"):
        probabilities_by_method(
            **inputs, labels=calibration_data.labels, folds=calibration_data.folds
        )


@pytest.fixture(name="spread")
def spread_fixture(calibration_data):
    """Return probabilities from exactly 0 to exactly 1, with the axes of the labels."""
    labels = calibration_data.labels
    values = np.linspace(0, 1, labels.size).reshape(labels.shape)
    return pd.DataFrame(values, index=labels.index, columns=labels.columns)


def test_check_probabilities_accept_probabilities_of_every_problem_and_tag(
    calibration_data, spread
):
    labels = calibration_data.labels

    assert check_probabilities({"spread": spread, "labels": labels.astype(float)}, labels) is None


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda frame: frame.iloc[::-1], "must have the index and columns of labels"),
        (lambda frame: frame.iloc[:, :-1], "must have the index and columns of labels"),
        (lambda frame: frame.replace(0.0, np.nan), "hold NaN"),
        (lambda frame: frame - 0.1, r"lie outside \[0, 1\]"),
        (lambda frame: frame * 1.5, r"lie outside \[0, 1\]"),
    ],
    ids=["problems reversed", "a tag missing", "NaN", "below 0", "above 1"],
)
def test_check_probabilities_name_the_method_whose_frame_is_not_probabilities_of_the_labels(
    calibration_data, spread, change, problem
):
    probabilities = {"spread": spread, "broken": change(spread)}

    with pytest.raises(ValueError, match=f"The probabilities of 'broken' {problem}"):
        check_probabilities(probabilities, calibration_data.labels)


def test_check_probabilities_refuse_no_method(calibration_data):
    with pytest.raises(ValueError, match="No method"):
        check_probabilities({}, calibration_data.labels)


def problems(n_problems):
    """Return the index of n problems."""
    return pd.Index(np.arange(n_problems) + 100, name="sample_id")


def resampled_brier_and_skill(weights, predicted, labels):
    """Compute each resample's Brier score and Brier skill per tag, by hand."""
    drawn = weights.sum(axis=1, keepdims=True)
    prevalence = weights @ labels.to_numpy() / drawn
    brier = weights @ ((predicted - labels) ** 2).to_numpy() / drawn
    return {"brier": brier, "brier_skill": 1 - brier / (prevalence * (1 - prevalence))}


def test_calibration_table_has_a_row_per_method_and_tag_then_all_pairs_and_each_bound(
    calibration_data, probabilities
):
    backwards = {method: probabilities[method] for method in reversed(probabilities)}

    table = calibration_table(
        backwards, calibration_data.labels, calibration_data.groups, resamples=20
    )

    assert (POOLED, METRICS) == ("all", ("brier", "log_loss", "brier_skill"))
    assert CALIBRATION_COLUMNS == (
        "brier",
        "brier_low",
        "brier_high",
        "log_loss",
        "log_loss_low",
        "log_loss_high",
        "brier_skill",
        "brier_skill_low",
        "brier_skill_high",
    )
    assert table.index.names == ["method", "tag"]
    assert table.index.tolist() == [
        (method, tag) for method in reversed(CALIBRATION_METHODS) for tag in [*FOCUS_TAGS, "all"]
    ]
    assert table.columns.tolist() == list(CALIBRATION_COLUMNS)
    assert (table.dtypes == "float64").all()
    assert table.notna().all(axis=None)


def test_calibration_table_estimates_are_scikit_learns_brier_score_and_log_loss(
    calibration_data, probabilities
):
    labels = calibration_data.labels
    variance = labels.mean() * (1 - labels.mean())

    table = calibration_table(probabilities, labels, calibration_data.groups, resamples=20)

    for method, frame in probabilities.items():
        rows = table.loc[method]
        tags = rows.loc[labels.columns]
        brier = [brier_score_loss(labels[tag], frame[tag]) for tag in labels.columns]
        losses = [log_loss(labels[tag], frame[tag]) for tag in labels.columns]
        np.testing.assert_allclose(tags["brier"], brier, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tags["log_loss"], losses, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tags["brier_skill"], 1 - brier / variance, rtol=0, atol=1e-12)
        truth, predicted = labels.to_numpy().ravel(), frame.to_numpy().ravel()
        assert rows.at[POOLED, "brier"] == pytest.approx(brier_score_loss(truth, predicted))
        assert rows.at[POOLED, "log_loss"] == pytest.approx(log_loss(truth, predicted))
        assert rows.at[POOLED, "brier_skill"] == pytest.approx(1 - sum(brier) / variance.sum())


@pytest.mark.parametrize(
    "bootstrap",
    [{}, {"resamples": 50, "confidence": 0.8, "seed": 3}],
    ids=["the bootstrap settings", "given"],
)
def test_calibration_table_bounds_are_percentiles_over_the_resamples_evaluate_draws(
    calibration_data, probabilities, bootstrap
):
    labels, groups = calibration_data.labels, calibration_data.groups
    settings = {
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "seed": BOOTSTRAP_SEED,
    } | bootstrap

    table = calibration_table(probabilities, labels, groups, **bootstrap)

    # evaluate(..., groups=groups, resamples=r, seed=s) draws resample_weights(groups, r, s).
    weights = resample_weights(groups, settings["resamples"], settings["seed"])
    # Groups of unequal sizes: each resample's means divide by the problems it drew, not by n.
    assert len(set(weights.sum(axis=1))) > 1
    for method, frame in probabilities.items():
        rows = table.loc[method].loc[labels.columns]
        for metric, values in resampled_brier_and_skill(weights, frame, labels).items():
            low, high = percentile_interval(values, settings["confidence"])
            np.testing.assert_allclose(rows[f"{metric}_low"], low, rtol=1e-12)
            np.testing.assert_allclose(rows[f"{metric}_high"], high, rtol=1e-12)


def test_calibration_table_draws_whole_near_duplicate_groups():
    # Each pair holds a positive and a negative problem. Drawn whole, every resample has a
    # prevalence of 1/2, where the constant 1/2 has a skill of exactly 0.
    index = problems(40)
    labels = pd.DataFrame({"math": np.arange(40) % 2}, index=index)
    half = {"half": pd.DataFrame({"math": 0.5}, index=index)}
    pairs = pd.Series(np.arange(40) // 2, index=index)

    by_pair = calibration_table(half, labels, pairs, resamples=200)
    by_problem = calibration_table(
        half, labels, pd.Series(np.arange(40), index=index), resamples=200
    )

    assert by_pair.loc[("half", "math"), ["brier_skill_low", "brier_skill_high"]].tolist() == [0, 0]
    assert by_problem.at[("half", "math"), "brier_skill_low"] < 0


def test_calibration_table_recomputes_the_prevalence_of_each_resample(calibration_data):
    # Forecasting the prevalence p0 has no skill: in a resample of prevalence p, its skill is
    # -(p - p0)^2 / (p (1 - p)), never above 0. Against p0 (1 - p0), it would be above 0 in many.
    labels, groups = calibration_data.labels, calibration_data.groups
    prevalence = labels.mean()
    forecast = labels * 0 + prevalence

    table = calibration_table({"prevalence": forecast}, labels, groups, resamples=200)

    skill = table.loc["prevalence"]
    np.testing.assert_allclose(skill["brier_skill"], 0, rtol=0, atol=1e-12)
    assert (skill["brier_skill_high"] <= 1e-12).all()
    assert (skill["brier_skill_low"] < 0).all()
    weights = resample_weights(groups, 200, BOOTSTRAP_SEED)
    brier = resampled_brier_and_skill(weights, forecast, labels)["brier"]
    against_p0 = 1 - brier / (prevalence * (1 - prevalence)).to_numpy()
    assert (percentile_interval(against_p0, BOOTSTRAP_CONFIDENCE)[1] > 0).all()


def test_calibration_table_log_loss_stays_finite_on_probabilities_of_exactly_0_and_1():
    # A wrong 0 or 1 costs -log(eps) = 36.04, as scikit-learn's log_loss clips it: not inf, and
    # a right one costs about 0, not NaN from 0 * log(0).
    index = problems(4)
    labels = pd.DataFrame({"math": [1, 0, 1, 0]}, index=index)
    exact = pd.DataFrame({"math": [1.0, 0.0, 0.0, 1.0]}, index=index)

    table = calibration_table(
        {"exact": exact}, labels, pd.Series([0, 1, 2, 3], index=index), resamples=20
    )

    log_losses = table.loc["exact", ["log_loss", "log_loss_low", "log_loss_high"]]
    assert np.isfinite(log_losses).all(axis=None)
    assert log_losses.at["math", "log_loss"] == pytest.approx(log_loss(labels, exact), abs=1e-12)
    assert log_losses.at["math", "log_loss"] == pytest.approx(-np.log(sys.float_info.epsilon) / 2)


def test_calibration_table_gives_nan_skill_bounds_when_a_resample_draws_one_class_of_a_tag():
    index = problems(40)
    labels = pd.DataFrame(
        {
            "math": np.arange(40) % 2,  # both classes in every resample
            "games": (np.arange(40) == 0).astype(int),  # one positive, missed by some resamples
            "trees": 0,  # no positive at all
            "geometry": 1,  # no negative at all
        },
        index=index,
    )
    # trees and geometry are predicted exactly: their Brier score is 0 too, so 0 / 0.
    predicted = labels.astype(float).assign(math=0.5, games=0.1)
    groups = pd.Series(np.arange(40), index=index)
    assert (resample_weights(groups, 200, BOOTSTRAP_SEED)[:, 0] == 0).any()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning from a division by 0
        table = calibration_table({"m": predicted}, labels, groups, resamples=200)

    skill = table.loc["m", ["brier_skill", "brier_skill_low", "brier_skill_high"]]
    assert np.isfinite(skill.loc[["math", POOLED]]).all(axis=None)
    assert np.isfinite(skill.at["games", "brier_skill"])
    assert skill.loc["games", ["brier_skill_low", "brier_skill_high"]].isna().all()
    assert skill.loc[["trees", "geometry"]].isna().all(axis=None)
    assert np.isfinite(table.drop(columns=skill.columns)).all(axis=None)
    assert not np.isinf(table.to_numpy()).any()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, g: ({}, g), "No method"),
        (
            lambda p, g: (p | {"svm_minmax": p["svm_minmax"] * 2}, g),
            r"The probabilities of 'svm_minmax' lie outside \[0, 1\]",
        ),
        (lambda p, g: (p, g.iloc[::-1]), "groups must have the index of labels"),
        (lambda p, g: (p, g.astype(float).where(g != 5)), "Every row needs a group"),
    ],
    ids=["no method", "not probabilities", "misaligned groups", "a missing group"],
)
def test_calibration_table_refuses_probabilities_or_groups_not_of_the_labels(
    calibration_data, probabilities, change, message
):
    broken, groups = change(probabilities, calibration_data.groups)

    with pytest.raises(ValueError, match=message):
        calibration_table(broken, calibration_data.labels, groups, resamples=20)


@pytest.fixture(name="cross_fitted")
def cross_fitted_fixture(calibration_data):
    """Return the SVM scores' fold thresholds (best_threshold midpoints) and predictions."""
    data = calibration_data
    fold_thresholds, predictions = cross_fitted_predictions(
        data.svm_scores, data.labels, data.folds
    )
    return fold_thresholds, predictions


def proba(fitted, values):
    """Map values through a fitted Platt map, as scikit-learn does: column 1 of predict_proba."""
    name = fitted.feature_names_in_[0]
    return fitted.predict_proba(pd.DataFrame({name: np.asarray(values, dtype=float)}))[:, 1]


def test_platt_maps_fit_one_unregularised_increasing_map_per_tag_on_all_scores(calibration_data):
    scores, labels = calibration_data.svm_scores, calibration_data.labels

    maps = platt_maps(scores, labels)

    assert list(maps) == list(FOCUS_TAGS)
    for tag, fitted in maps.items():
        by_hand = platt_calibrator().fit(scores[[tag]], labels[tag])
        assert np.isposinf(fitted.C)
        assert fitted.coef_[0, 0] == by_hand.coef_[0, 0] > 0
        assert fitted.intercept_[0] == by_hand.intercept_[0]


def test_platt_maps_refuse_a_map_that_does_not_increase(calibration_data):
    # A slope at or below 0 reverses the order of the scores: mapped thresholds would then flip
    # predictions.
    labels = calibration_data.labels.copy()
    labels["games"] = 1 - labels["games"]

    with pytest.raises(ValueError, match=r"The Platt maps of the tags \['games'\] do not increase"):
        platt_maps(calibration_data.svm_scores, labels)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda labels: labels.iloc[::-1], "labels must have the index and columns of scores"),
        (lambda labels: labels[sorted(FOCUS_TAGS)], "labels must have the index and columns"),
        (lambda labels: labels.assign(trees=0), r"tags \['trees'\] have one class"),
    ],
    ids=["problems reversed", "tags sorted by name", "a tag with no positive"],
)
def test_platt_maps_refuse_labels_not_of_the_scores_or_with_one_class(
    calibration_data, change, message
):
    with pytest.raises(ValueError, match=message):
        platt_maps(calibration_data.svm_scores, change(calibration_data.labels))


def test_mapped_predictions_equal_the_cross_fitted_predictions(calibration_data, cross_fitted):
    # The fold thresholds are best_threshold's midpoints, away from every score: each Platt map
    # increases, so a score reaches its threshold exactly when its probability reaches the
    # threshold's.
    data = calibration_data
    fold_thresholds, predictions = cross_fitted
    maps = platt_maps(data.svm_scores, data.labels)

    mapped = mapped_predictions(data.svm_scores, fold_thresholds, data.folds, maps)

    pd.testing.assert_frame_equal(mapped, predictions, check_exact=True)
    assert mapped.dtypes.eq("int64").all()
    for tag, fitted in maps.items():
        threshold = proba(fitted, fold_thresholds.loc[data.folds, tag])
        by_hand = (proba(fitted, data.svm_scores[tag]) >= threshold).astype(np.int64)
        np.testing.assert_array_equal(mapped[tag], by_hand)


def test_mapped_predictions_keep_an_infinite_threshold_infinite(calibration_data, cross_fitted):
    # best_threshold gives inf to a fold whose other folds hold no positive: nothing is predicted.
    # Mapped naively, inf becomes expit(inf) = 1.0, which a saturated score reaches.
    data = calibration_data
    fold_thresholds, predictions = cross_fitted
    maps = platt_maps(data.svm_scores, data.labels)
    scores = data.svm_scores.copy()
    saturated = scores.index[data.folds == 0][0]
    scores.loc[saturated, "games"] = 1e3
    fold_thresholds.loc[0, "games"] = np.inf
    assert proba(maps["games"], [1e3])[0] == expit(np.inf) == 1.0

    mapped = mapped_predictions(scores, fold_thresholds, data.folds, maps)

    assert mapped.loc[data.folds == 0, "games"].eq(0).all()
    expected = predictions.copy()
    expected.loc[data.folds == 0, "games"] = 0
    pd.testing.assert_frame_equal(mapped, expected, check_exact=True)


def test_mapped_predictions_only_add_a_prediction_where_rounding_ties_a_score_with_its_threshold(
    calibration_data, cross_fitted
):
    # In float64, expit(a s + b) is only non-decreasing: near probability 1, a score 1 ulp below
    # its threshold maps to the threshold's probability, so it gains a prediction. A score at or
    # above its threshold always keeps its own.
    data = calibration_data
    fold_thresholds, _ = cross_fitted
    maps = platt_maps(data.svm_scores, data.labels)
    slope, intercept = maps["math"].coef_[0, 0], maps["math"].intercept_[0]
    threshold = (np.log(1e9) - intercept) / slope  # a probability of 1 - 1e-9
    below = np.nextafter(threshold, -np.inf)
    assert below < threshold
    assert proba(maps["math"], [below])[0] == proba(maps["math"], [threshold])[0]
    scores = data.svm_scores.copy()
    tied = scores.index[data.folds == 1][0]
    scores.loc[tied, "math"] = below
    fold_thresholds.loc[1, "math"] = threshold
    by_score = pd.concat(
        apply_thresholds(scores[data.folds == fold], fold_thresholds.loc[fold])
        for fold in fold_thresholds.index
    ).loc[scores.index]

    mapped = mapped_predictions(scores, fold_thresholds, data.folds, maps)

    gained = (mapped - by_score).stack()
    assert gained[gained != 0].to_dict() == {(tied, "math"): 1}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda scores, thresholds, folds, maps: (scores, thresholds, folds.iloc[::-1], maps),
            "folds must have the index of scores",
        ),
        (
            lambda scores, thresholds, folds, maps: (scores, thresholds.drop(index=2), folds, maps),
            r"The folds \[2\] have no row in fold_thresholds",
        ),
        (
            lambda scores, thresholds, folds, maps: (
                scores,
                thresholds,
                folds.where(folds != 2, -1),
                maps,
            ),
            r"The folds \[-1\] have no row in fold_thresholds",
        ),
        (
            lambda scores, thresholds, folds, maps: (
                scores,
                thresholds,
                folds,
                {tag: maps[tag] for tag in sorted(maps)},
            ),
            "maps must hold the tags",
        ),
        (
            lambda scores, thresholds, folds, maps: (
                scores,
                thresholds.drop(columns="games"),
                folds,
                maps,
            ),
            "fold_thresholds must hold the tags",
        ),
    ],
    ids=["misaligned folds", "a fold missing", "test problems", "maps sorted", "a tag missing"],
)
def test_mapped_predictions_refuse_folds_thresholds_or_maps_not_of_the_scores(
    calibration_data, cross_fitted, change, message
):
    data = calibration_data
    maps = platt_maps(data.svm_scores, data.labels)
    arguments = change(data.svm_scores, cross_fitted[0], data.folds, maps)

    with pytest.raises(ValueError, match=message):
        mapped_predictions(*arguments)


@pytest.fixture(name="final")
def final_fixture(calibration_data):
    """Return the final thresholds of the SVM scores, and the F1 they reach on those scores."""
    scores, labels = calibration_data.svm_scores, calibration_data.labels
    thresholds = per_tag_thresholds(scores, labels)
    tag_f1 = evaluate(labels, apply_thresholds(scores, thresholds)).tag_f1
    return thresholds, tag_f1.rename_axis("metric")


def test_lipton_table_sets_each_mapped_threshold_beside_half_its_f1(calibration_data, final):
    thresholds, tag_f1 = final
    maps = platt_maps(calibration_data.svm_scores, calibration_data.labels)

    table = lipton_table(thresholds, maps, tag_f1)

    assert LIPTON_COLUMNS == ("threshold", "probability_threshold", "f1", "half_f1")
    assert table.columns.tolist() == list(LIPTON_COLUMNS)
    assert table.index.equals(pd.Index(FOCUS_TAGS, name="tag"))
    np.testing.assert_array_equal(table["threshold"], thresholds)
    np.testing.assert_array_equal(table["f1"], tag_f1)
    np.testing.assert_array_equal(table["half_f1"], tag_f1 / 2)
    for tag, fitted in maps.items():
        assert table.at[tag, "probability_threshold"] == proba(fitted, [thresholds[tag]])[0]


def test_lipton_table_keeps_an_infinite_threshold_infinite(calibration_data, final):
    thresholds, tag_f1 = final
    thresholds["games"] = np.inf
    maps = platt_maps(calibration_data.svm_scores, calibration_data.labels)

    table = lipton_table(thresholds, maps, tag_f1)

    assert np.isposinf(table.at["games", "probability_threshold"])
    assert np.isfinite(table["probability_threshold"].drop("games")).all()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda thresholds, maps, f1: (thresholds[::-1], maps, f1), "maps must hold the tags"),
        (
            lambda thresholds, maps, f1: (thresholds, maps, f1.drop("games")),
            "tag_f1 must hold the tags",
        ),
        (
            lambda thresholds, maps, f1: (thresholds, {"math": maps["math"]}, f1),
            "maps must hold the tags",
        ),
    ],
    ids=["thresholds reversed", "an F1 missing", "a map missing"],
)
def test_lipton_table_refuses_tags_that_differ_or_come_in_another_order(
    calibration_data, final, change, message
):
    maps = platt_maps(calibration_data.svm_scores, calibration_data.labels)

    with pytest.raises(ValueError, match=message):
        lipton_table(*change(final[0], maps, final[1]))
