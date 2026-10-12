"""Choose and apply the per-tag thresholds that turn scores into predicted tags.

A tag is predicted when its score reaches the tag's threshold, chosen to maximise the tag's F1
on out-of-fold scores. Cross-fitted thresholds score each fold with thresholds chosen on the
other folds, so the F1 measured on a fold is not fitted to that fold's labels
(docs/adr/0001-model-selection-protocol.md). ``CrossFittedTagThresholds`` runs the whole
protocol inside one scikit-learn estimator, around any scorer of tags.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.utils.validation import check_is_fitted


def best_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    """Choose the threshold that maximises one tag's F1 on given scores.

    Each distinct score is a candidate that predicts the tag for every score at or above it.
    Among the candidates with the highest F1, the highest one wins. The threshold returned
    sits halfway between it and the next lower distinct score: it predicts the same on these
    scores, and leaves a margin on both sides for new ones.

    Args:
        scores: One score per problem; a higher score means the tag is more likely.
        truth: 0/1 or boolean true label per problem, aligned on ``scores``.

    Returns:
        The threshold: the winning candidate itself when it is the lowest score (every
        problem predicted), and ``math.inf`` (nothing predicted) when no problem has the tag.

    Raises:
        ValueError: If ``scores`` and ``truth`` differ in length.
    """
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if len(scores) != len(truth):
        raise ValueError(f"{len(scores)} scores but {len(truth)} truth values.")
    positives = int(truth.sum())
    if positives == 0:
        return math.inf
    order = np.argsort(-scores)
    descending = scores[order]
    true_positives = np.cumsum(truth[order])
    # A candidate predicts a whole run of equal scores: keep the last position of each run.
    ends = np.flatnonzero(np.append(descending[1:] != descending[:-1], True))
    candidates = descending[ends]
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (predicted + positives)
    f1 = 2 * true_positives[ends] / (ends + 1 + positives)
    best = int(np.argmax(f1))  # the first maximum, so the highest of tied candidates
    if best == len(candidates) - 1:
        return float(candidates[best])
    return float((candidates[best] + candidates[best + 1]) / 2)


def per_tag_thresholds(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """Choose each tag's threshold on the scores and true labels of the same problems.

    Args:
        scores: One column of scores per tag, one row per problem.
        truth: 0/1 true labels, with the index and columns of ``scores``.

    Returns:
        The ``best_threshold`` of each tag, indexed by the columns of ``scores``.

    Raises:
        ValueError: If ``scores`` and ``truth`` differ in index or columns.
    """
    _check_aligned(scores, truth)
    thresholds = [
        best_threshold(scores[tag].to_numpy(), truth[tag].to_numpy()) for tag in scores.columns
    ]
    return pd.Series(thresholds, index=scores.columns, name="threshold", dtype=float)


def cross_fitted_thresholds(
    scores: pd.DataFrame, truth: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Choose each fold's per-tag thresholds on the problems of the other folds.

    Applied to fold k, the thresholds of row k were chosen without fold k's labels.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per problem.
        truth: 0/1 true labels, with the index and columns of ``scores``.
        folds: Fold id per problem, with the index of ``scores``.

    Returns:
        One row of ``per_tag_thresholds`` per fold, indexed by the sorted fold ids (index
        named like ``folds``), with the columns of ``scores``.

    Raises:
        ValueError: If ``scores`` and ``truth`` differ in index or columns, or if ``folds``
            is not indexed like ``scores``.
    """
    _check_aligned(scores, truth)
    if not folds.index.equals(scores.index):
        raise ValueError("folds must have the index of scores: one fold per problem, in order.")
    fold_ids = np.sort(folds.unique())
    rows = []
    for fold in fold_ids:
        others = (folds != fold).to_numpy()
        rows.append(per_tag_thresholds(scores[others], truth[others]))
    return pd.DataFrame(rows, index=pd.Index(fold_ids, name=folds.name), columns=scores.columns)


def apply_thresholds(scores: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """Predict each tag whose score reaches the tag's threshold.

    Args:
        scores: One column of scores per tag, one row per problem.
        thresholds: Threshold per tag, indexed by the columns of ``scores``, in order.

    Returns:
        0/1 predictions, with the index and columns of ``scores``.

    Raises:
        ValueError: If the index of ``thresholds`` is not the columns of ``scores``.
    """
    if not thresholds.index.equals(scores.columns):
        raise ValueError(
            f"Thresholds for {list(thresholds.index)} do not match the score columns "
            f"{list(scores.columns)}."
        )
    return scores.ge(thresholds, axis="columns").astype(int)


def check_dev_folds(index: pd.Index, labels: pd.DataFrame, folds: pd.Series) -> None:
    """Fail unless labels and folds align on the problems, over two or more dev folds.

    Every out-of-fold entry point calls it, so a dev frame that holds test problems by mistake
    is refused: ``LeaveOneGroupOut`` would otherwise take the test set's -1 for one more fold.

    Args:
        index: The problems' index (sample ids), in order.
        labels: 0/1 labels, one row per problem.
        folds: Dev fold per problem.

    Raises:
        ValueError: If a problem appears more than once in ``index``, if ``labels`` or
            ``folds`` is not indexed like ``index``, if a fold is missing (no fold would hold
            its problem out) or negative (the test set's -1), or if there are fewer than two
            folds.
    """
    if not index.is_unique:
        repeated = index[index.duplicated()].unique().tolist()
        raise ValueError(
            f"The problems {repeated} appear more than once: pass one row per problem, or the "
            "out-of-fold results would gain a row per extra copy."
        )
    for name, aligned in (("labels", labels), ("folds", folds)):
        if not aligned.index.equals(index):
            raise ValueError(
                f"{name} must have the same index as problems: the same problems, in order."
            )
    missing = int(folds.isna().sum())
    if missing:
        raise ValueError(
            f"{missing} problems have no fold: no fold would hold them out. Take the folds of the "
            "split's dev problems, select_part(problems, split, 'dev')."
        )
    negative = int((folds < 0).sum())
    if negative:
        raise ValueError(
            f"{negative} problems have a negative fold (the test set's -1): the test set is never "
            "cross-fitted. Pass the dev set, select_part(problems, split, 'dev')."
        )
    n_folds = folds.nunique()
    if n_folds < 2:
        raise ValueError(f"Cross-fitting needs at least two folds, not {n_folds}.")


def cross_fitted_predictions(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each fold with the thresholds chosen on the other folds' scores and labels.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per dev problem.
        labels: 0/1 true labels, with the index and columns of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.

    Returns:
        ``(fold_thresholds, predictions)``: the ``cross_fitted_thresholds`` (one row per sorted
        fold id), and the 0/1 predictions of each fold with its own row, with the index and
        columns of ``scores``.

    Raises:
        ValueError: As ``check_dev_folds`` and ``cross_fitted_thresholds`` do.
    """
    check_dev_folds(scores.index, labels, folds)
    fold_thresholds = cross_fitted_thresholds(scores, labels, folds)
    predictions = pd.concat(
        apply_thresholds(scores[folds == fold], fold_thresholds.loc[fold])
        for fold in fold_thresholds.index
    ).loc[scores.index]
    return fold_thresholds, predictions


_NO_REFIT = (
    "This %(name)s has no refit estimator: it is not fitted, or was fitted with refit=False, "
    "which keeps the out-of-fold results only."
)


class CrossFittedTagThresholds(  # pylint: disable=too-many-instance-attributes
    MetaEstimatorMixin, ClassifierMixin, BaseEstimator
):
    """Per-tag thresholds chosen out of fold, around a multi-label scorer (ADR 0001).

    ``fit`` scores each fold with a clone of ``estimator`` fitted on the other folds, predicts
    each fold with thresholds chosen on the other folds' scores, chooses the final thresholds
    on every out-of-fold score, then refits ``estimator`` on every problem. ``estimator``
    follows the score-estimator contract: a ``BaseEstimator`` whose ``fit(X, y)`` takes the
    rows as given and an (n, n_tags) int64 array and sets ``classes_ = np.arange(n_tags)``,
    whose ``decision_function(X)`` returns an (n, n_tags) float array, and whose
    ``predict(X)`` returns ``(decision_function(X) >= 0).astype(int)`` (``cross_val_predict``
    requires it; this class never calls it). Its import path is frozen: saved method folders
    refer to it by that path.

    Attributes:
        oof_scores_: Out-of-fold scores, (n, n_tags) floats, in the order of the rows fitted.
        oof_predictions_: 0/1 out-of-fold predictions, (n, n_tags) ints: each fold predicted
            with its own row of ``fold_thresholds_``.
        fold_thresholds_: (n_folds, n_tags) floats: row k chosen on the other folds' scores;
            ``inf`` where those folds have no positive of the tag.
        thresholds_: (n_tags,) floats, chosen on every out-of-fold score: the thresholds of
            ``predict``.
        tags_: The label columns, a tuple of str.
        oof_index_: The sample ids of the rows fitted, an int64 array.
        fold_ids_: The sorted fold ids, an array.
        estimator_: ``estimator`` refit on every problem; only when ``refit`` is true.
    """

    def __init__(self, estimator, *, refit: bool = True):
        """Wrap a scorer.

        Args:
            estimator: The unfitted scorer of tags, cloned for every fit.
            refit: Whether ``fit`` also refits ``estimator`` on every problem, which ``predict``
                and ``decision_function`` need. ``False`` keeps the out-of-fold results only
                (e.g. in a tuning objective).
        """
        self.estimator = estimator
        self.refit = refit

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, *, groups: pd.Series
    ) -> CrossFittedTagThresholds:
        """Score the dev set out of fold, choose the thresholds, then refit.

        Args:
            X: The dev problems' rows, indexed by ``sample_id``, as the scorer takes them.
            y: 0/1 labels, one column per tag, with the index of ``X``.
            groups: Dev fold per problem, with the index of ``X`` (e.g. ``dev["fold"]``): the
                out-of-fold scores and the cross-fitted thresholds share these folds.

        Returns:
            The fitted estimator.

        Raises:
            TypeError: If ``X`` or ``y`` is not a DataFrame.
            ValueError: If ``groups`` is not a Series, if ``y`` is not 0/1 or has a tag of
                every problem, or as ``check_dev_folds`` does.
        """
        _check_fit_input(X, y, groups)
        scores = cross_val_predict(
            clone(self.estimator),
            X,
            y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            method="decision_function",
        )
        oof_scores = pd.DataFrame(scores, index=X.index, columns=y.columns)
        fold_thresholds, predictions = cross_fitted_predictions(oof_scores, y, groups)
        self.oof_scores_ = scores
        self.oof_predictions_ = predictions.to_numpy(dtype=np.int64)
        self.fold_thresholds_ = fold_thresholds.to_numpy(dtype=float)
        self.thresholds_ = per_tag_thresholds(oof_scores, y).to_numpy(dtype=float)
        self.tags_ = tuple(str(tag) for tag in y.columns)
        self.oof_index_ = X.index.to_numpy(dtype=np.int64)
        self.fold_ids_ = fold_thresholds.index.to_numpy()
        if self.refit:
            # The fold fits get y as an int64 array from cross_val_predict: so does the refit.
            self.estimator_ = clone(self.estimator).fit(X, y.to_numpy(dtype=np.int64))
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Score problems with the refit scorer.

        Args:
            X: Rows as the scorer takes them.

        Returns:
            One score per problem and tag, ``estimator_.decision_function(X)``.

        Raises:
            NotFittedError: If the estimator was not fitted, or was fitted with ``refit=False``.
        """
        check_is_fitted(self, "estimator_", msg=_NO_REFIT)
        return self.estimator_.decision_function(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict each tag whose score reaches its threshold (``>=``).

        Args:
            X: Rows as the scorer takes them.

        Returns:
            0/1 ints, one row per problem, one column per tag of ``tags_``.

        Raises:
            NotFittedError: If the estimator was not fitted, or was fitted with ``refit=False``.
        """
        return (self.decision_function(X) >= self.thresholds_).astype(int)


def _check_fit_input(X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series) -> None:
    """Fail unless the rows, the 0/1 labels and the folds can be cross-fitted."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("X and y must be DataFrames indexed by sample_id.")
    if not isinstance(groups, pd.Series):
        raise ValueError(
            "groups must be the dev fold of each problem, a Series indexed like X "
            f"(e.g. dev['fold']), not {type(groups).__name__}."
        )
    if not y.isin((0, 1)).all().all():
        raise ValueError("y must hold 0/1 labels.")
    always = [str(tag) for tag in y.columns if y[tag].eq(1).all()]
    if always:
        raise ValueError(
            f"The tags {always} are positive for every problem: cross_val_predict would "
            "encode them as 0 in the fold fits. Drop them."
        )
    check_dev_folds(X.index, y, groups)


def _check_aligned(scores: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Fail unless ``scores`` and ``truth`` have the same index and columns, in order."""
    if not scores.index.equals(truth.index):
        raise ValueError("scores and truth must have the same index: the same problems, in order.")
    if not scores.columns.equals(truth.columns):
        raise ValueError(
            f"scores and truth must have the same columns, in order: {list(scores.columns)} "
            f"against {list(truth.columns)}."
        )
