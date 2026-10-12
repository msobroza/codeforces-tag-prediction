"""Tests for choosing and applying per-tag thresholds."""

import math

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import estimator_checks

from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    apply_thresholds,
    best_threshold,
    check_dev_folds,
    cross_fitted_predictions,
    cross_fitted_thresholds,
    per_tag_thresholds,
)


def oracle_threshold(scores, truth):
    """Brute force: try every distinct score, keep the highest one with the best F1."""
    best, best_f1 = None, -1.0
    for candidate in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= candidate
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:  # strictly better only: ties keep the higher threshold
            best, best_f1 = candidate, f1
    return best


def random_case(seed):
    """Return random scores, often tied (rounded to 0-3 decimals), and truth with a positive."""
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 40))
    scores = np.round(rng.normal(size=size), int(rng.integers(0, 4)))
    truth = rng.random(size) < rng.random()
    truth[rng.integers(size)] = True
    return scores, truth


@pytest.mark.parametrize("seed", range(200))
def test_best_threshold_predicts_on_its_scores_what_a_brute_force_f1_oracle_predicts(seed):
    scores, truth = random_case(seed)

    threshold = best_threshold(scores, truth)

    assert np.array_equal(scores >= threshold, scores >= oracle_threshold(scores, truth))


def test_best_threshold_returns_the_midpoint_below_the_best_score():
    scores = np.array([0.9, 0.7, 0.4, 0.1])

    assert best_threshold(scores, np.array([1, 1, 0, 0])) == (0.7 + 0.4) / 2


def test_best_threshold_keeps_the_highest_of_tied_thresholds():
    scores = np.array([4.0, 3.0, 2.0, 1.0])

    # Predicting from 4.0 (TP 1, FN 1) and from 1.0 (TP 2, FP 2) both give F1 = 2/3.
    assert best_threshold(scores, np.array([1, 0, 0, 1])) == 3.5


def test_best_threshold_takes_a_run_of_equal_scores_whole():
    scores = np.array([0.8, 0.5, 0.5, 0.5, 0.5, 0.2])

    # Predicting from 0.5 also takes the three negatives tied with the positive: F1 4/7 < 2/3.
    assert best_threshold(scores, np.array([1, 1, 0, 0, 0, 0])) == (0.8 + 0.5) / 2


def test_best_threshold_is_the_lowest_score_when_every_problem_should_be_predicted():
    assert best_threshold(np.array([0.2, 0.9, 0.5]), np.array([1, 1, 1])) == 0.2


def test_best_threshold_is_the_shared_score_when_all_scores_are_equal():
    assert best_threshold(np.full(4, 0.3), np.array([1, 0, 0, 0])) == 0.3


def test_best_threshold_predicts_nothing_without_a_positive():
    assert best_threshold(np.array([0.2, 0.8]), np.array([0, 0])) == math.inf
    assert best_threshold(np.array([]), np.array([])) == math.inf


def test_best_threshold_rejects_scores_and_truth_of_different_lengths():
    with pytest.raises(ValueError, match="3 scores but 2 truth values"):
        best_threshold(np.array([0.1, 0.2, 0.3]), np.array([1, 0]))


@pytest.fixture(name="sample_ids")
def sample_ids_fixture():
    """Return the sample ids of 30 problems."""
    return pd.Index(range(100, 130), name="sample_id")


@pytest.fixture(name="scores")
def scores_fixture(sample_ids):
    """Return random scores for two tags."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(30, 2)),
        index=sample_ids,
        columns=pd.Index(["math", "games"], name="label"),
    )


@pytest.fixture(name="truth")
def truth_fixture(scores):
    """Return 0/1 labels that agree with the scores more often than not."""
    rng = np.random.default_rng(1)
    noisy = scores + rng.normal(size=scores.shape)
    return (noisy > 0.5).astype(int)


@pytest.fixture(name="folds")
def folds_fixture(sample_ids):
    """Assign the problems to folds 2, 0, 1, 2, 0, 1, ..."""
    return pd.Series([2, 0, 1] * 10, index=sample_ids, name="fold")


def test_per_tag_thresholds_choose_each_tag_on_its_own_column(scores, truth):
    thresholds = per_tag_thresholds(scores, truth)

    assert thresholds.index.equals(scores.columns)
    assert thresholds["math"] == best_threshold(scores["math"].to_numpy(), truth["math"].to_numpy())
    assert thresholds["games"] == best_threshold(
        scores["games"].to_numpy(), truth["games"].to_numpy()
    )


def test_per_tag_thresholds_reject_truth_on_other_problems(scores, truth):
    with pytest.raises(ValueError, match="same index"):
        per_tag_thresholds(scores, truth.iloc[::-1])


def test_per_tag_thresholds_reject_truth_with_other_tags(scores, truth):
    with pytest.raises(ValueError, match="same columns"):
        per_tag_thresholds(scores, truth[["games", "math"]])


def test_cross_fitted_thresholds_have_one_row_per_sorted_fold_and_one_column_per_tag(
    scores, truth, folds
):
    thresholds = cross_fitted_thresholds(scores, truth, folds)

    assert thresholds.index.tolist() == [0, 1, 2]
    assert thresholds.index.name == "fold"
    assert thresholds.columns.equals(scores.columns)


def test_cross_fitted_thresholds_choose_each_fold_on_the_other_folds(scores, truth, folds):
    thresholds = cross_fitted_thresholds(scores, truth, folds)

    for fold in (0, 1, 2):
        others = folds != fold
        expected = per_tag_thresholds(scores[others], truth[others])
        assert np.array_equal(thresholds.loc[fold].to_numpy(), expected.to_numpy())


def test_cross_fitted_thresholds_ignore_the_truth_of_the_fold_they_apply_to(scores, truth, folds):
    flipped = truth.mask(folds == 1, 1 - truth, axis=0)

    before = cross_fitted_thresholds(scores, truth, folds)
    after = cross_fitted_thresholds(scores, flipped, folds)

    pd.testing.assert_series_equal(after.loc[1], before.loc[1])
    assert not after.loc[[0, 2]].equals(before.loc[[0, 2]])  # the other folds did use them


def test_cross_fitted_thresholds_reject_folds_of_other_problems(scores, truth, folds):
    with pytest.raises(ValueError, match="folds"):
        cross_fitted_thresholds(scores, truth, folds.iloc[1:])


def test_cross_fitted_thresholds_reject_misaligned_truth(scores, truth, folds):
    with pytest.raises(ValueError, match="same columns"):
        cross_fitted_thresholds(scores, truth[["games"]], folds)


def test_apply_thresholds_predicts_scores_at_or_above_each_tag_threshold():
    scores = pd.DataFrame({"math": [0.1, 0.5, 0.9], "games": [3.0, -1.0, 2.0]}, index=[7, 8, 9])
    thresholds = pd.Series({"math": 0.5, "games": math.inf})

    predicted = apply_thresholds(scores, thresholds)

    expected = pd.DataFrame({"math": [0, 1, 1], "games": [0, 0, 0]}, index=[7, 8, 9])
    pd.testing.assert_frame_equal(predicted, expected)


def test_apply_thresholds_reproduces_the_chosen_predictions_on_the_training_scores(scores, truth):
    predicted = apply_thresholds(scores, per_tag_thresholds(scores, truth))

    for tag in scores.columns:
        column, labels = scores[tag].to_numpy(), truth[tag].to_numpy() == 1
        assert np.array_equal(predicted[tag] == 1, column >= oracle_threshold(column, labels))


def test_apply_thresholds_rejects_thresholds_for_other_tags():
    scores = pd.DataFrame({"math": [0.1], "games": [0.2]})

    with pytest.raises(ValueError, match="do not match the score columns"):
        apply_thresholds(scores, pd.Series({"games": 0.0, "math": 0.0}))


def test_check_dev_folds_accepts_labels_and_folds_aligned_on_two_or_more_folds(
    sample_ids, truth, folds
):
    assert check_dev_folds(sample_ids, truth, folds) is None


def test_check_dev_folds_rejects_labels_of_other_problems(sample_ids, truth, folds):
    with pytest.raises(ValueError, match="labels must have the same index"):
        check_dev_folds(sample_ids, truth.iloc[::-1], folds)


def test_check_dev_folds_rejects_folds_of_other_problems(sample_ids, truth, folds):
    with pytest.raises(ValueError, match="folds must have the same index"):
        check_dev_folds(sample_ids, truth, folds.iloc[1:])


def test_check_dev_folds_refuses_the_test_fold(sample_ids, truth, folds):
    # A dev frame holding test rows passed by mistake: LeaveOneGroupOut would take the test
    # set's -1 for one more fold.
    with_test = folds.mask(folds == 2, -1)

    with pytest.raises(ValueError, match="10 problems have a negative fold"):
        check_dev_folds(sample_ids, truth, with_test)


def test_check_dev_folds_refuses_a_problem_without_a_fold(sample_ids, truth, folds):
    # A fold column reindexed onto other problems: NaN is not negative, nunique skips it, and
    # no fold of folds.unique() ever holds its problem out.
    without = folds.astype(float).mask(folds.index.isin([100, 104]))

    with pytest.raises(ValueError, match="2 problems have no fold"):
        check_dev_folds(sample_ids, truth, without)


def test_check_dev_folds_refuses_a_problem_that_appears_twice(sample_ids, truth, folds):
    # A bad concat of dev frames: every out-of-fold result would gain a row per extra copy.
    repeated = sample_ids[[0, 0, 1, 2, 2, *range(3, 30)]]

    with pytest.raises(ValueError, match=r"\[100, 102\] appear more than once"):
        check_dev_folds(repeated, truth.reindex(repeated), folds.reindex(repeated))


def test_check_dev_folds_needs_two_folds(sample_ids, truth, folds):
    with pytest.raises(ValueError, match="at least two folds"):
        check_dev_folds(sample_ids, truth, folds * 0)


def test_cross_fitted_predictions_predict_each_fold_with_its_own_thresholds(scores, truth, folds):
    fold_thresholds, predictions = cross_fitted_predictions(scores, truth, folds)

    pd.testing.assert_frame_equal(fold_thresholds, cross_fitted_thresholds(scores, truth, folds))
    expected = pd.concat(
        apply_thresholds(scores[folds == fold], fold_thresholds.loc[fold]) for fold in (0, 1, 2)
    ).loc[scores.index]
    pd.testing.assert_frame_equal(predictions, expected)


def test_cross_fitted_predictions_refuse_the_test_fold(scores, truth, folds):
    with pytest.raises(ValueError, match="negative fold"):
        cross_fitted_predictions(scores, truth, folds.mask(folds == 0, -1))


class LinearScorer(BaseEstimator):
    """Least squares of the 0/1 labels on numeric columns, in numpy alone; it cannot predict."""

    def __init__(self, columns=("x0", "x1"), set_classes=True):
        """Score from ``columns``; ``set_classes=False`` breaks the contract (no ``classes_``)."""
        self.columns = columns
        self.set_classes = set_classes

    def fit(self, X, y):
        """Fit one linear score per tag, checking what the score-estimator contract passes."""
        # The raw rows as given, with the fold and group columns that select_part appends,
        # and the labels as an int64 matrix.
        assert {"fold", "group"} <= set(X.columns)
        assert isinstance(y, np.ndarray) and y.dtype == np.int64
        self.coef_ = np.linalg.lstsq(self._design(X), y, rcond=None)[0]
        if self.set_classes:
            self.classes_ = np.arange(y.shape[1])
        return self

    def decision_function(self, X):
        """Return one score per problem and tag, 0 at the halfway label."""
        return self._design(X) @ self.coef_ - 0.5

    def _design(self, X):
        values = X[list(self.columns)].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(values)), values])


class ContractScorer(LinearScorer):
    """A scorer that follows the whole score-estimator contract of ``CrossFittedTagThresholds``."""

    def predict(self, X):
        """Predict the tags whose score is at least 0."""
        return (self.decision_function(X) >= 0).astype(int)


@pytest.fixture(name="rows")
def rows_fixture(scores, folds):
    """Return raw rows to score: two numeric columns, then the fold and group of each problem."""
    rows = pd.DataFrame(scores.to_numpy(), index=scores.index, columns=["x0", "x1"])
    return rows.assign(fold=folds, group=range(len(rows)))


def test_the_contract_scorer_predicts_where_its_score_is_at_least_zero(rows, truth):
    scorer = ContractScorer().fit(rows, truth.to_numpy())

    np.testing.assert_array_equal(scorer.predict(rows), scorer.decision_function(rows) >= 0)


def test_cross_fitted_tag_thresholds_equal_the_hand_written_fold_loop(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer()).fit(rows, truth, groups=folds)

    scores = pd.DataFrame(np.nan, index=rows.index, columns=truth.columns)
    for fold in (0, 1, 2):
        held_out = folds == fold
        scorer = ContractScorer().fit(rows[~held_out], truth[~held_out].to_numpy())
        scores[held_out] = scorer.decision_function(rows[held_out])
    fold_thresholds, predictions = cross_fitted_predictions(scores, truth, folds)
    np.testing.assert_array_equal(model.oof_scores_, scores.to_numpy())
    np.testing.assert_array_equal(model.fold_thresholds_, fold_thresholds.to_numpy())
    np.testing.assert_array_equal(model.oof_predictions_, predictions.to_numpy())
    np.testing.assert_array_equal(model.thresholds_, per_tag_thresholds(scores, truth).to_numpy())


def test_cross_fitted_tag_thresholds_keep_plain_types_that_skops_can_save(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer()).fit(rows, truth, groups=folds)

    assert model.tags_ == ("math", "games")
    assert {type(tag) for tag in model.tags_} == {str}
    np.testing.assert_array_equal(model.oof_index_, np.arange(100, 130))
    assert model.oof_index_.dtype == np.int64
    np.testing.assert_array_equal(model.fold_ids_, [0, 1, 2])
    assert model.oof_predictions_.dtype == np.int64
    for name in ("oof_scores_", "oof_predictions_", "fold_thresholds_", "thresholds_"):
        assert isinstance(getattr(model, name), np.ndarray)


def test_cross_fitted_tag_thresholds_refit_the_scorer_on_every_problem(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer()).fit(rows, truth, groups=folds)

    expected = ContractScorer().fit(rows, truth.to_numpy()).decision_function(rows)
    np.testing.assert_array_equal(model.decision_function(rows), expected)


def test_cross_fitted_tag_thresholds_predict_scores_at_or_above_the_thresholds(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer()).fit(rows, truth, groups=folds)
    first = model.decision_function(rows.iloc[:1])[0]

    model.thresholds_ = first  # the first problem's scores sit exactly on the thresholds
    np.testing.assert_array_equal(model.predict(rows.iloc[:1]), [[1, 1]])
    model.thresholds_ = np.nextafter(first, np.inf)
    np.testing.assert_array_equal(model.predict(rows.iloc[:1]), [[0, 0]])


def test_cross_fitted_tag_thresholds_need_a_scorer_that_can_predict(rows, truth, folds):
    with pytest.raises(ValueError, match="implementing 'fit' and 'predict'"):
        CrossFittedTagThresholds(LinearScorer()).fit(rows, truth, groups=folds)


def test_cross_fitted_tag_thresholds_need_a_scorer_with_classes(rows, truth, folds):
    with pytest.raises(AttributeError, match="classes_"):
        CrossFittedTagThresholds(ContractScorer(set_classes=False)).fit(rows, truth, groups=folds)


def test_cross_fitted_tag_thresholds_without_refit_keep_out_of_fold_results_only(
    rows, truth, folds
):
    model = CrossFittedTagThresholds(ContractScorer(), refit=False).fit(rows, truth, groups=folds)

    assert model.oof_scores_.shape == (30, 2)
    assert not hasattr(model, "estimator_")
    with pytest.raises(NotFittedError, match="refit"):
        model.predict(rows)
    with pytest.raises(NotFittedError, match="refit"):
        model.decision_function(rows)


def test_cross_fitted_tag_thresholds_require_the_folds_as_groups(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer())

    with pytest.raises(TypeError, match="groups"):
        model.fit(rows, truth)  # pylint: disable=missing-kwoa
    with pytest.raises(ValueError, match="groups must be"):
        model.fit(rows, truth, groups=None)
    with pytest.raises(ValueError, match="groups must be"):
        model.fit(rows, truth, groups=folds.to_numpy())


def test_cross_fitted_tag_thresholds_require_frames(rows, truth, folds):
    model = CrossFittedTagThresholds(ContractScorer())

    with pytest.raises(TypeError, match="DataFrame"):
        model.fit(rows, truth.to_numpy(), groups=folds)
    with pytest.raises(TypeError, match="DataFrame"):
        model.fit(rows.to_numpy(), truth, groups=folds)


def test_cross_fitted_tag_thresholds_refuse_labels_that_are_not_0_1(rows, truth, folds):
    with pytest.raises(ValueError, match="0/1"):
        CrossFittedTagThresholds(ContractScorer()).fit(rows, truth * 2, groups=folds)


def test_cross_fitted_tag_thresholds_refuse_a_tag_of_every_problem(rows, truth, folds):
    # cross_val_predict label-encodes each column: a column of ones would reach the fold fits
    # as a column of zeros.
    always = truth.assign(games=1)

    with pytest.raises(ValueError, match=r"\['games'\] .* every problem"):
        CrossFittedTagThresholds(ContractScorer()).fit(rows, always, groups=folds)


def test_cross_fitted_tag_thresholds_refuse_the_test_fold(rows, truth, folds):
    with pytest.raises(ValueError, match="negative fold"):
        CrossFittedTagThresholds(ContractScorer()).fit(
            rows, truth, groups=folds.mask(folds == 1, -1)
        )


def test_cross_fitted_tag_thresholds_handles_a_tag_absent_from_a_fold(rows, truth, folds):
    # games: positives in fold 1 only, so folds 0 and 2 choose its thresholds on no positive;
    # math: no positive at all.
    labels = truth.assign(math=0, games=(folds == 1).astype(int) * truth["games"])

    model = CrossFittedTagThresholds(ContractScorer()).fit(rows, labels, groups=folds)

    games, absent = model.tags_.index("games"), model.tags_.index("math")
    assert model.fold_thresholds_[1, games] == math.inf  # chosen on folds 0 and 2
    assert np.isfinite(model.fold_thresholds_[[0, 2], games]).all()
    assert model.oof_predictions_[(folds == 1).to_numpy(), games].sum() == 0
    assert math.isfinite(model.thresholds_[games])
    assert model.fold_thresholds_[:, absent].tolist() == [math.inf] * 3
    assert model.thresholds_[absent] == math.inf
    assert model.predict(rows)[:, absent].sum() == 0


def test_cross_fitted_tag_thresholds_clone_with_nested_parameters():
    model = CrossFittedTagThresholds(OneVsRestClassifier(LogisticRegression(C=0.5)), refit=False)

    copy = clone(model).set_params(estimator__estimator__C=2.0)

    assert model.get_params()["estimator__estimator__C"] == 0.5
    assert copy.get_params()["estimator__estimator__C"] == 2.0
    assert copy.get_params()["refit"] is False


# scikit-learn's checks that neither fit nor pickle: its fitting checks call fit(X, y)
# without groups, and its pickle and memmap checks write pickle files.
SKLEARN_CHECKS = (
    estimator_checks.check_estimator_cloneable,
    estimator_checks.check_estimator_repr,
    estimator_checks.check_no_attributes_set_in_init,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_parameters_default_constructible,
    estimator_checks.check_do_not_raise_errors_in_init_or_set_params,
    estimator_checks.check_mixin_order,
    estimator_checks.check_estimators_unfitted,
)


@pytest.mark.parametrize("check", SKLEARN_CHECKS, ids=lambda check: check.__name__)
def test_cross_fitted_tag_thresholds_pass_sklearn_checks_that_do_not_fit(check):
    check(
        "CrossFittedTagThresholds",
        CrossFittedTagThresholds(OneVsRestClassifier(LogisticRegression())),
    )


def test_cross_fitted_tag_thresholds_round_trip_through_skops(rows, truth, folds):
    model = CrossFittedTagThresholds(OneVsRestClassifier(LogisticRegression()))
    data = sio.dumps(model.fit(rows, truth, groups=folds))

    # The only type skops does not trust by default: the class itself, by its import path.
    trusted = ["codeforces_tags.model.thresholds.CrossFittedTagThresholds"]
    assert sio.get_untrusted_types(data=data) == trusted
    loaded = sio.loads(data, trusted=trusted)
    np.testing.assert_array_equal(loaded.predict(rows), model.predict(rows))
    assert loaded.tags_ == model.tags_
    for name in ("thresholds_", "fold_thresholds_", "oof_scores_", "oof_index_", "fold_ids_"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
