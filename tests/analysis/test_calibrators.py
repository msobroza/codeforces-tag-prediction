"""Tests for notebook 11's comparison of calibrators: the maps, the rules and the rankings."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit  # pylint: disable=no-name-in-module
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from codeforces_tags.analysis.calibrators import (
    CALIBRATORS,
    DECISION_RULES,
    RANKING_COLUMNS,
    RULE_KEY,
    BetaCalibration,
    beta_features,
    calibrated_probabilities,
    make_calibrator,
    ranking_table,
    rule_predictions,
)
from codeforces_tags.config import PLATT_CLIP
from codeforces_tags.evaluation.calibration import cross_fitted_probabilities
from codeforces_tags.model.platt_lipton import platt_feature, platt_fit, rule_calibrator
from codeforces_tags.model.thresholds import best_threshold, cross_fitted_predictions


@pytest.fixture(name="dev", scope="module")
def dev_fixture():
    """Return 600 dev problems in five folds of 120: two tags' scores, their labels, the folds.

    Every fold holds the same 120 scores of each tag, in another order, so a cut chosen on four
    folds falls between the same scores of the fifth. The scores lie on Jev's 0.01 grid, with a
    failed answer's 0.0 in every fold, and they are too high: each label is drawn with
    probability expit(logit(score) - 1). Every fold holds positives and negatives of both tags.
    """
    rng = np.random.default_rng(2017)
    shared = rng.integers(1, 100, size=(120, 2)) / 100
    shared[0] = 0.0
    values = np.concatenate([rng.permuted(shared, axis=0) for _ in range(5)])
    index = pd.Index(range(7000, 7600), name="sample_id")
    tags = pd.Index(["math", "graphs"], name="label")
    truth = rng.random(values.shape) < expit(logit(values) - 1)
    scores = pd.DataFrame(values, index=index, columns=tags)
    labels = pd.DataFrame(truth.astype(int), index=index, columns=tags)
    folds = pd.Series(np.repeat(np.arange(5), 120), index=index, name="fold")
    assert labels.groupby(folds).nunique().eq(2).all().all()
    return scores, labels, folds


@pytest.fixture(name="probabilities", scope="module")
def probabilities_fixture(dev):
    """Return the fixture's probabilities of every calibrator."""
    scores, labels, folds = dev
    return calibrated_probabilities(scores, labels, folds)


def last_step(calibrator):
    """Return the model at the end of a calibrator's pipeline."""
    return calibrator.steps[-1][1]


# A fine grid of scores from 0 to 1, where a map that decreases anywhere shows it.
FINE_GRID = np.linspace(0, 1, 10_001)[:, np.newaxis]
TENTHS = np.round(np.arange(1, 10) / 10, 2)
# Positives rise with the score up to 0.9 and fall a little above.
FALLING_AT_THE_TOP = {**{q: q for q in TENTHS}, 0.95: 0.7, 0.99: 0.5}
# Positives fall with the score below 0.1 and rise above.
FALLING_AT_THE_BOTTOM = {0.01: 0.5, 0.02: 0.3, 0.05: 0.1, **{q: q for q in TENTHS}}
# Positives fall with the score everywhere: 270 of the 900 problems hold the tag.
FALLING = {q: 0.6 * (1 - q) for q in TENTHS}


def grouped_case(shares):
    """Return 100 problems per score, round(100 * share) of them positive: scores and labels."""
    scores = np.repeat(list(shares), 100)[:, np.newaxis]
    labels = np.concatenate([np.arange(100) < round(100 * share) for share in shares.values()])
    return scores, labels.astype(int)


def free_beta(scores, labels):
    """Return beta calibration without its constraint: a logistic regression on beta_features."""
    return rule_calibrator().fit(beta_features(scores), labels)


def grid_draws(seed, truth_logit):
    """Return 20,000 probabilities on Jev's grid and labels drawn from a map of their logits."""
    rng = np.random.default_rng(seed)
    probabilities = rng.integers(1, 100, size=20_000) / 100
    labels = (rng.random(20_000) < expit(truth_logit(logit(probabilities)))).astype(int)
    return probabilities[:, np.newaxis], labels


def test_beta_features_are_the_logs_of_the_clipped_score_and_of_its_complement():
    values = np.array([0.0, 0.5, 1.0])
    low, high = PLATT_CLIP, 1 - PLATT_CLIP
    expected = [
        [np.log(low), -np.log(1 - low)],
        [np.log(0.5), -np.log(0.5)],
        [np.log(high), -np.log(1 - high)],
    ]

    for scores in (values, pd.DataFrame({"math": values})):
        features = beta_features(scores)

        assert features.shape == (3, 2)
        np.testing.assert_allclose(features, expected)
        assert np.isfinite(features).all()


def test_beta_features_refuse_more_than_one_score_per_problem():
    with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, 1\), not \(3, 2\)"):
        beta_features(np.zeros((3, 2)))


@pytest.mark.parametrize(
    ("scores", "outside"),
    [([0.0, 0.5, 1.0, 1.2], 1), ([-0.1, 0.3, 1.5, np.nan], 3), ([np.nan, 0.7], 1)],
)
def test_beta_refuses_scores_outside_0_and_1_as_the_platt_feature_does(dev, scores, outside):
    # Regression: the clip before the logs would turn a scorer's margins into features silently.
    values = np.array(scores)[:, np.newaxis]
    labels = [0, 1] * (len(scores) // 2)
    fitted = make_calibrator("beta").fit(dev[0][["math"]], dev[1]["math"])
    reads = rf"reads scores in \[0, 1\]: {outside} of {len(scores)} scores lie outside it"
    refusals = [
        lambda: beta_features(values),
        lambda: make_calibrator("beta").fit(values, labels),
        lambda: fitted.predict_proba(values),
        lambda: fitted.predict(values),
    ]

    for refused in refusals:
        with pytest.raises(ValueError, match=f"^Beta calibration {reads}, or are NaN"):
            refused()
    with pytest.raises(ValueError, match=f"^A Platt map {reads}, or are NaN"):
        platt_feature(values)


def test_make_calibrator_returns_a_fresh_unfitted_map_per_call():
    assert make_calibrator("isotonic") is not make_calibrator("isotonic")
    assert last_step(make_calibrator("temperature")).fit_intercept is False
    assert last_step(make_calibrator("platt_logit")).fit_intercept is True
    for name in CALIBRATORS[1:]:
        with pytest.raises(NotFittedError):
            check_is_fitted(make_calibrator(name))


@pytest.mark.parametrize("name", ["none", "sigmoid"])
def test_make_calibrator_refuses_a_name_that_is_no_map_naming_every_calibrator(name):
    with pytest.raises(ValueError, match=f"'{name}' is no map") as refused:
        make_calibrator(name)

    for calibrator in CALIBRATORS:
        assert f"'{calibrator}'" in str(refused.value)


def test_every_logistic_map_is_the_rules_converged_fit_and_platt_logit_is_its_map(dev):
    scores, labels, _ = dev
    converged = rule_calibrator().get_params()
    without_intercept = {**converged, "fit_intercept": False}

    assert last_step(make_calibrator("platt_logit")).get_params() == converged
    assert make_calibrator("platt_raw").get_params() == converged
    assert last_step(make_calibrator("temperature")).get_params() == without_intercept
    # Exactly the map of the Platt + Lipton rule: the same slope and intercept, to the last bit.
    fitted = last_step(make_calibrator("platt_logit").fit(scores[["math"]], labels["math"]))
    rule_map = platt_fit(scores["math"].to_numpy(), labels["math"].to_numpy())
    assert (fitted.coef_[0, 0], fitted.intercept_[0]) == rule_map


def test_platt_logit_maps_calibrated_probabilities_onto_themselves():
    probabilities, labels = grid_draws(3, truth_logit=lambda feature: feature)
    grid = np.linspace(0.05, 0.95, 19)

    calibrator = make_calibrator("platt_logit").fit(probabilities, labels)

    mapped = calibrator.predict_proba(grid[:, np.newaxis])[:, 1]
    np.testing.assert_allclose(mapped, grid, atol=0.03)


def test_temperature_fits_one_slope_on_the_logit_and_no_shift():
    probabilities, labels = grid_draws(4, truth_logit=lambda feature: feature / 2)

    model = last_step(make_calibrator("temperature").fit(probabilities, labels))

    assert model.coef_[0, 0] == pytest.approx(0.5, abs=0.1)
    assert model.intercept_.tolist() == [0.0]


@pytest.mark.parametrize(
    ("name", "feature"), [("platt_logit", platt_feature), ("platt_raw", np.asarray)]
)
def test_the_platt_maps_are_a_sigmoid_of_the_logit_or_of_the_raw_score(dev, name, feature):
    scores, labels, _ = dev

    calibrator = make_calibrator(name).fit(scores[["math"]], labels["math"])

    model = last_step(calibrator) if name == "platt_logit" else calibrator
    by_hand = expit(model.coef_[0, 0] * feature(scores["math"]) + model.intercept_[0])
    np.testing.assert_allclose(calibrator.predict_proba(scores[["math"]])[:, 1], by_hand)


def test_beta_is_a_sigmoid_of_the_log_of_the_score_and_of_its_complement(dev):
    scores, labels, _ = dev

    calibrator = make_calibrator("beta").fit(scores[["math"]], labels["math"])

    assert calibrator.coef_.shape == (1, 2)
    (a, b), c = calibrator.coef_[0], calibrator.intercept_[0]
    q = np.clip(scores["math"].to_numpy(), PLATT_CLIP, 1 - PLATT_CLIP)
    by_hand = expit(a * np.log(q) - b * np.log(1 - q) + c)
    np.testing.assert_allclose(calibrator.predict_proba(scores[["math"]])[:, 1], by_hand)
    assert calibrator.predict_proba(scores[["math"]]).sum(axis=1) == pytest.approx(1.0)
    predicted = calibrator.predict(scores[["math"]])
    np.testing.assert_array_equal(predicted, (by_hand > 0.5).astype(int))
    assert 0 < predicted.sum() < len(predicted)


def test_beta_is_the_free_fit_when_both_slopes_are_positive(dev):
    scores, labels, _ = dev
    free = free_beta(scores[["math"]], labels["math"])

    calibrator = make_calibrator("beta").fit(scores[["math"]], labels["math"])

    assert (free.coef_ > 0).all()
    np.testing.assert_array_equal(calibrator.coef_, free.coef_)
    np.testing.assert_allclose(
        calibrator.predict_proba(FINE_GRID),
        free.predict_proba(beta_features(FINE_GRID)),
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("shares", "kept", "dropped"), [(FALLING_AT_THE_TOP, 0, 1), (FALLING_AT_THE_BOTTOM, 1, 0)]
)
def test_beta_refits_without_the_feature_of_a_negative_slope_so_it_never_decreases(
    shares, kept, dropped
):
    scores, labels = grouped_case(shares)
    free = free_beta(scores, labels)
    free_map = free.predict_proba(beta_features(FINE_GRID))[:, 1]

    calibrator = make_calibrator("beta").fit(scores, labels)

    # The case is built for it: the free fit's slope is negative, and its map decreases somewhere.
    assert free.coef_[0, dropped] < 0
    assert (np.diff(free_map) < 0).any()
    mapped = calibrator.predict_proba(FINE_GRID)[:, 1]
    assert (np.diff(mapped) >= 0).all()
    assert calibrator.coef_[0, dropped] == 0
    alone = rule_calibrator().fit(beta_features(scores)[:, [kept]], labels)
    assert alone.coef_[0, 0] > 0
    np.testing.assert_array_equal(calibrator.coef_[0, kept], alone.coef_[0, 0])
    np.testing.assert_array_equal(calibrator.intercept_, alone.intercept_)
    assert np.abs(mapped - free_map).max() > 0.01


def test_beta_falls_back_on_the_prevalence_when_one_feature_still_gives_a_negative_slope():
    scores, labels = grouped_case(FALLING)

    calibrator = make_calibrator("beta").fit(scores, labels)

    assert (free_beta(scores, labels).coef_ < 0).all()
    assert calibrator.coef_.tolist() == [[0.0, 0.0]]
    np.testing.assert_allclose(calibrator.predict_proba(FINE_GRID)[:, 1], 0.3)


def test_beta_is_a_classifier_that_clones_unfitted(dev):
    scores, labels, _ = dev
    fitted = make_calibrator("beta").fit(scores[["math"]], labels["math"])

    copy = clone(fitted)

    assert isinstance(copy, BetaCalibration)
    assert is_classifier(copy)
    assert copy.get_params() == {}
    assert fitted.classes_.tolist() == [0, 1]
    for method in (copy.predict_proba, copy.predict):
        with pytest.raises(NotFittedError):
            method(scores[["math"]])


@pytest.mark.parametrize("labels", [[1, 1, 1, 1], [0, 1, 2, 1]])
def test_beta_refuses_labels_of_other_than_two_classes(labels):
    with pytest.raises(ValueError, match=r"two classes, not \[") as refused:
        BetaCalibration().fit(np.array([0.1, 0.4, 0.6, 0.9]), labels)

    assert str(sorted(set(labels))) in str(refused.value)


def test_beta_maps_each_fold_by_its_fit_on_the_other_folds_in_cross_fitted_probabilities(dev):
    scores, labels, folds = dev

    mapped = cross_fitted_probabilities(scores, labels, folds, make_calibrator("beta"))

    for fold in range(5):
        own = (folds == fold).to_numpy()
        for tag in scores.columns:
            fitted = BetaCalibration().fit(scores.loc[~own, [tag]], labels.loc[~own, tag])
            expected = fitted.predict_proba(scores.loc[own, [tag]])[:, 1]
            np.testing.assert_allclose(mapped.loc[own, tag], expected)


def test_calibrated_probabilities_give_each_calibrator_in_order(dev, probabilities):
    scores, _, _ = dev

    assert list(probabilities) == list(CALIBRATORS)
    pd.testing.assert_frame_equal(probabilities["none"], scores)
    for frame in probabilities.values():
        assert frame.index.equals(scores.index)
        assert frame.columns.equals(scores.columns)
        assert ((frame >= 0) & (frame <= 1)).all().all()
    # The calibrators move the too-high scores down: their mean comes nearer the labels' mean.
    assert (probabilities["platt_logit"].mean() < scores.mean()).all()


def test_the_scores_of_none_are_a_copy(dev):
    scores, labels, folds = dev

    copy = calibrated_probabilities(scores, labels, folds)["none"]
    copy.loc[:, "math"] = -1.0

    assert (scores["math"] >= 0).all()


def test_calibrated_probabilities_map_each_fold_by_the_other_folds_only(dev, probabilities):
    scores, labels, folds = dev
    in_fold = (folds == 0).to_numpy()
    flipped = labels.copy()
    flipped[in_fold] = 1 - labels[in_fold]

    after = calibrated_probabilities(scores, flipped, folds)

    for name in CALIBRATORS:
        pd.testing.assert_frame_equal(after[name][in_fold], probabilities[name][in_fold])
    for name in CALIBRATORS[1:]:
        assert not after[name][~in_fold].equals(probabilities[name][~in_fold])


def test_calibrated_probabilities_refuse_a_margin_as_the_platt_feature_does(dev):
    scores, labels, folds = dev
    # From -2 to 2: a margin, not a probability.
    margins = scores.assign(math=4 * scores["math"] - 2)

    with pytest.raises(ValueError, match=r"^A Platt map reads scores in \[0, 1\]"):
        calibrated_probabilities(margins, labels, folds)


def test_calibrated_probabilities_refuse_folds_of_other_problems(dev):
    scores, labels, folds = dev

    with pytest.raises(ValueError, match="folds must have the same index as problems"):
        calibrated_probabilities(scores, labels, folds.iloc[::-1])


def test_rule_predictions_give_every_rule_of_every_calibrator_as_0_1(dev, probabilities):
    _, labels, folds = dev

    predictions = rule_predictions(probabilities, labels, folds)

    keys = [RULE_KEY.format(calibrator=c, rule=r) for c in CALIBRATORS for r in DECISION_RULES]
    assert list(predictions) == keys
    assert keys[:3] == ["none / at_0.5", "none / at_half_best_f1", "none / best_f1"]
    for frame in predictions.values():
        assert frame.index.equals(labels.index)
        assert frame.columns.equals(labels.columns)
        assert (frame.dtypes == "int64").all()
        assert frame.isin((0, 1)).all().all()


def half_best_f1_cuts(scores, labels, folds):
    """Predict each fold where its score reaches half the best F1 that the other folds reach."""
    expected = pd.DataFrame(0, index=scores.index, columns=scores.columns)
    for fold in folds.unique():
        own, others = folds == fold, folds != fold
        for tag in scores.columns:
            fit_scores, fit_labels = scores.loc[others, tag], labels.loc[others, tag]
            cut = best_threshold(fit_scores, fit_labels)
            half = f1_score(fit_labels, fit_scores >= cut) / 2
            expected.loc[own, tag] = (scores.loc[own, tag] >= half).astype(int)
    return expected


def test_the_rules_cut_at_05_at_half_the_best_f1_and_at_the_f1_argmax(dev):
    scores, labels, folds = dev

    predictions = rule_predictions({"none": scores}, labels, folds)

    at_05 = predictions["none / at_0.5"]
    pd.testing.assert_frame_equal(at_05, (scores >= 0.5).astype("int64"))
    half_best = predictions["none / at_half_best_f1"]
    pd.testing.assert_frame_equal(half_best, half_best_f1_cuts(scores, labels, folds))
    argmax = predictions["none / best_f1"]
    pd.testing.assert_frame_equal(argmax, cross_fitted_predictions(scores, labels, folds)[1])
    assert not half_best.equals(argmax)


def test_no_rule_predicts_a_tag_that_no_problem_has(dev):
    scores, labels, folds = dev

    predictions = rule_predictions({"none": scores}, labels.assign(graphs=0), folds)

    assert predictions["none / at_half_best_f1"]["graphs"].eq(0).all()
    assert predictions["none / best_f1"]["graphs"].eq(0).all()
    assert predictions["none / at_0.5"]["graphs"].eq(1).any()


def test_an_increasing_map_moves_the_cut_at_05_but_not_the_f1_argmax(dev):
    scores, labels, folds = dev
    mapped = expit(2 * logit(np.clip(scores, PLATT_CLIP, 1 - PLATT_CLIP)) + 1)

    predictions = rule_predictions({"none": scores, "mapped": mapped}, labels, folds)

    pd.testing.assert_frame_equal(predictions["mapped / best_f1"], predictions["none / best_f1"])
    assert not predictions["mapped / at_0.5"].equals(predictions["none / at_0.5"])


def test_rule_predictions_refuse_labels_of_other_tags(dev):
    scores, labels, folds = dev

    with pytest.raises(ValueError, match="same columns"):
        rule_predictions({"none": scores}, labels[["graphs", "math"]], folds)


def test_ranking_table_averages_each_tags_auc_and_average_precision(dev, probabilities):
    _, labels, _ = dev
    compared = {"isotonic": probabilities["isotonic"], "reversed": 1 - probabilities["none"]}

    table = ranking_table(compared, labels)

    assert table.index.tolist() == ["isotonic", "reversed"]
    assert table.index.name == "calibrator"
    assert table.columns.tolist() == list(RANKING_COLUMNS)
    for name, frame in compared.items():
        aucs = [roc_auc_score(labels[tag], frame[tag]) for tag in labels.columns]
        precisions = [average_precision_score(labels[tag], frame[tag]) for tag in labels.columns]
        assert table.loc[name, "macro_auc"] == pytest.approx(np.mean(aucs))
        assert table.loc[name, "macro_ap"] == pytest.approx(np.mean(precisions))
    assert table.loc["reversed", "macro_auc"] < 0.5 < table.loc["isotonic", "macro_auc"]


def test_ranking_table_refuses_a_tag_of_one_class_naming_it(dev):
    scores, labels, _ = dev

    with pytest.raises(ValueError, match=r"\['graphs'\] have one class"):
        ranking_table({"none": scores}, labels.assign(graphs=0))


def test_ranking_table_refuses_probabilities_of_other_problems(dev):
    scores, labels, _ = dev

    with pytest.raises(ValueError, match="must have the index and columns of labels"):
        ranking_table({"none": scores.iloc[::-1]}, labels)
