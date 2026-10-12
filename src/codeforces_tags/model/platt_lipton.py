"""A threshold rule that reads calibrated probabilities: Platt scaling, then half the best F1.

The F1-argmax threshold of ``model.thresholds`` sits where a tag's F1 curve peaks. When the curve
has two almost equal peaks, which folds choose the threshold decides which peak wins, and the
threshold jumps between them. This threshold rule reads only the height of the curve. A Platt
map, fitted on the logit of the scores, turns them into calibrated probabilities; on calibrated
probabilities the F1-optimal threshold is half the best F1 (Lipton et al., 2014). The rule
predicts a tag where the calibrated probability reaches half the best F1 on the same scores,
maps that cut back to the score scale and snaps it to Jev's 0.01 grid (docs/adr/0011). The
Platt map is the maximum-likelihood fit, ``rule_calibrator()``, solved to convergence, so that
a threshold does not depend on when a solver stopped. The clip before the logit, the grid and
the solver's settings are the project-wide ``config.PLATT_CLIP``, ``config.SCORE_GRID``,
``config.PLATT_SOLVER`` and ``config.PLATT_TOL``, which notebook 11's helpers read too;
``RULE_SETTINGS`` records them. Those helpers call this module's functions rather than copies of
them: ``analysis.calibrators`` calls ``platt_feature``, ``rule_calibrator`` (every logistic map
it compares), ``check_unit_interval`` (the [0, 1] check of ``platt_feature``, before beta
calibration's own clip), ``best_f1``, ``cross_fitted_rows`` and ``apply_fold_thresholds``, and
``analysis.threshold_rules`` maps through its ``platt_logit``, the rule's own map.

``PlattLiptonThresholds`` runs the rule inside the cross-fitting of ``CrossFittedTagThresholds``.
The frozen class is never edited, since the tag model's skops file names it by its import path.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import expit, logit  # pylint: disable=no-name-in-module  # ufuncs, built in C
from sklearn.linear_model import LogisticRegression

from codeforces_tags.config import PLATT_CLIP, PLATT_SOLVER, PLATT_TOL, SCORE_GRID
from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    apply_thresholds,
    best_threshold,
)

# The threshold rule as JSON values, built from its project-wide settings: in a calibrated twin's
# recipe and in its jev.json hash.
RULE_SETTINGS: dict[str, object] = {
    "name": "platt-lipton",
    "feature": "logit",
    "clip": PLATT_CLIP,
    "C": "inf",
    "solver": PLATT_SOLVER,
    "tol": PLATT_TOL,
    "grid": SCORE_GRID,
}
# How far above a grid value, in grid steps, a mapped-back threshold still counts as that value:
# the float noise of expit and logit, and of the division by the grid (0.07 / 0.01 is
# 7.000000000000001), never a real gap between two grid values. A threshold below a grid value
# needs no slack: the ceiling lifts it to that value.
_SNAP_SLACK = 1e-9


def check_unit_interval(scores: np.ndarray, reader: str) -> None:
    """Fail unless every score lies in [0, 1], before a map of probabilities clips them.

    ``platt_feature`` and beta calibration's features (``analysis.calibrators.beta_features``)
    clip the scores to ``[PLATT_CLIP, 1 - PLATT_CLIP]``: clipped, a scorer's margins or
    log-odds would turn silently into features that mean nothing, so both call this first.

    Args:
        scores: The scores, of any shape.
        reader: What reads them, which the message names, e.g. ``"A Platt map"``.

    Raises:
        ValueError: If a score lies outside [0, 1] or is NaN, naming how many do.
    """
    values = np.asarray(scores, dtype=float)
    outside = int(np.count_nonzero(~((values >= 0) & (values <= 1))))
    if outside:
        raise ValueError(
            f"{reader} reads scores in [0, 1]: {outside} of {values.size} scores lie outside "
            "it, or are NaN. A probability is in [0, 1]; a margin or a log-odds is not."
        )


def rule_calibrator() -> LogisticRegression:
    """Return the rule's Platt map, unfitted: a logistic regression without penalty, to convergence.

    It is ``evaluation.calibration.platt_calibrator()`` (``C=np.inf``) fitted to the maximum of
    the likelihood: ``config.PLATT_SOLVER`` at ``config.PLATT_TOL``, so that a threshold does not
    depend on when the solver stopped. ``platt_calibrator()`` stays as it is, lbfgs at
    scikit-learn's default tolerance: it serves the calibration of ``tfidf`` (``tracking
    calibrate``, notebook 04).

    Returns:
        ``LogisticRegression(C=np.inf, solver=PLATT_SOLVER, tol=PLATT_TOL)``, a new object on
        every call: its probability of the tag is column 1 of ``predict_proba``.
    """
    return LogisticRegression(C=np.inf, solver=PLATT_SOLVER, tol=PLATT_TOL)


def platt_feature(scores: np.ndarray) -> np.ndarray:
    """Return the feature a Platt map reads: the logit of the scores, clipped away from 0 and 1.

    Args:
        scores: Scores in [0, 1] (Jev's probabilities; a failed answer scores 0.0).

    Returns:
        ``logit(clip(scores, PLATT_CLIP, 1 - PLATT_CLIP))``, a float array: finite for every
        score, so a failed answer's 0.0 stays a usable point.

    Raises:
        ValueError: As ``check_unit_interval`` does, before any clip: a score outside [0, 1] or
            NaN, naming how many.
    """
    values = np.asarray(scores, dtype=float)
    check_unit_interval(values, "A Platt map")
    return logit(np.clip(values, PLATT_CLIP, 1 - PLATT_CLIP))


def platt_fit(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a Platt map on the logit of one tag's scores.

    The feature is ``platt_feature(scores)``, and the map ``expit(a * feature + b)`` is
    ``rule_calibrator()``: the maximum-likelihood fit, without a penalty, to convergence.

    Args:
        scores: One score in [0, 1] per problem (Jev's probabilities).
        labels: 0/1 or boolean true label per problem, aligned on ``scores``.

    Returns:
        ``(a, b)``: the map's slope, above 0, and its intercept.

    Raises:
        ValueError: As ``platt_feature`` does (a score outside [0, 1]), if ``scores`` and
            ``labels`` differ in length, if the labels hold one class only, or if the slope is
            not above 0: such a map would reverse the order of the scores, so no threshold on
            the scores would match it.
    """
    feature = platt_feature(scores)
    labels = np.asarray(labels, dtype=bool).astype(int)
    if len(feature) != len(labels):
        raise ValueError(f"{len(feature)} scores but {len(labels)} labels.")
    if np.unique(labels).size < 2:
        raise ValueError(
            "A Platt map needs positive and negative problems: these labels hold one class."
        )
    fitted = rule_calibrator().fit(feature[:, np.newaxis], labels)
    slope, intercept = float(fitted.coef_[0, 0]), float(fitted.intercept_[0])
    if slope <= 0:
        raise ValueError(
            f"The Platt map's slope is {slope:.6g}, not above 0: it would reverse the order of "
            "the scores, so no threshold on the scores matches it."
        )
    return slope, intercept


def snap_to_grid(threshold: float, grid: float = SCORE_GRID) -> float:
    """Move a threshold to the midpoint below the smallest grid value at or above it.

    A threshold up to 1e-9 grid steps above a grid value counts as that value, the float noise
    of expit and logit (e.g. 0.07000000000000002 snaps to 0.065, so 0.07 is predicted, though
    ``0.07 >= 0.07000000000000002`` is false). Outside those windows, on scores of the grid, the
    snapped threshold predicts exactly what ``threshold`` predicts. It is a short decimal that a
    frozen file stores exactly, clipped to ``[grid / 2, 1 - grid / 2]``, so a failed answer's
    0.0 is never predicted.

    Args:
        threshold: A threshold on the score scale, or ``math.inf`` (nothing predicted).
        grid: The step of the scores' grid; its midpoints must have at most three decimals
            (``SCORE_GRID``, 0.01, gives 0.005, 0.015, ..., 0.995).

    Returns:
        ``ceil(threshold / grid - 1e-9) * grid - grid / 2``, clipped to
        ``[grid / 2, 1 - grid / 2]`` and rounded to three decimals; ``math.inf`` unchanged.
    """
    if math.isinf(threshold):
        return threshold
    snapped = math.ceil(threshold / grid - _SNAP_SLACK) * grid - grid / 2
    return round(min(max(snapped, grid / 2), 1 - grid / 2), 3)


def best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return one tag's F1 at the threshold ``model.thresholds.best_threshold`` chooses.

    Args:
        scores: One score per problem; a higher score means the tag is more likely.
        labels: 0/1 or boolean true label per problem, aligned on ``scores``.

    Returns:
        ``2 TP / (predicted + positives)`` when every score at or above ``best_threshold`` is
        predicted: the highest F1 of any cut of these scores. 0.0 when no problem has the tag,
        since no cut finds a positive that does not exist.

    Raises:
        ValueError: If ``scores`` and ``labels`` differ in length.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    threshold = best_threshold(scores, labels)
    if math.isinf(threshold):
        return 0.0
    predicted = scores >= threshold
    return float(2 * np.sum(predicted & labels) / (np.sum(predicted) + np.sum(labels)))


def lipton_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose one tag's threshold by the Platt + Lipton threshold rule.

    The best F1 is ``best_f1`` of these scores. The Platt map of ``platt_fit`` predicts the tag
    where the calibrated probability reaches half of it; that cut, mapped back to the score
    scale, is ``expit((logit(best_f1 / 2) - b) / a)``, then ``snap_to_grid``.

    Args:
        scores: One score in [0, 1] per problem (Jev's probabilities).
        labels: 0/1 or boolean true label per problem, aligned on ``scores``.

    Returns:
        The snapped threshold, or ``math.inf`` (nothing predicted) when no problem has the tag.

    Raises:
        ValueError: If ``scores`` and ``labels`` differ in length, or as ``platt_fit`` does
            (a score outside [0, 1], every problem has the tag, or the slope is not above 0).
    """
    labels = np.asarray(labels, dtype=bool)
    half = best_f1(scores, labels) / 2
    if not labels.any():
        return math.inf
    slope, intercept = platt_fit(scores, labels)
    cut = expit((logit(half) - intercept) / slope)
    return snap_to_grid(float(cut))


def lipton_thresholds(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Choose each tag's threshold by the Platt + Lipton threshold rule.

    Args:
        scores: One column of scores per tag, one row per problem.
        labels: 0/1 true labels, with the index and columns of ``scores``.

    Returns:
        The ``lipton_threshold`` of each tag, indexed by the columns of ``scores``, named
        ``threshold``.

    Raises:
        ValueError: If ``labels`` does not have the index and columns of ``scores``, in order,
            or as ``lipton_threshold`` does.
    """
    if not (labels.index.equals(scores.index) and labels.columns.equals(scores.columns)):
        raise ValueError(
            "labels must have the index and columns of scores: the same problems and tags, in "
            "order."
        )
    found = [lipton_threshold(scores[tag].to_numpy(), labels[tag].to_numpy()) for tag in scores]
    return pd.Series(found, index=scores.columns, name="threshold", dtype=float)


def cross_fitted_rows(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
    choose: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """Run a per-tag choice once per fold, on the problems of the other folds.

    Applied to fold k, row k was chosen without fold k's labels, as the rows of
    ``model.thresholds.cross_fitted_thresholds`` are (whose choice is ``per_tag_thresholds``).

    Args:
        scores: Out-of-fold scores, one column per tag, one row per problem.
        labels: 0/1 true labels, with the index of ``scores``.
        folds: Fold id per problem, with the index of ``scores``.
        choose: Takes the scores and labels of some problems and returns one value per tag,
            indexed by the columns of ``scores`` (e.g. ``lipton_thresholds``).

    Returns:
        One row of ``choose`` per fold, indexed by the sorted fold ids (index named like
        ``folds``), with the columns of ``scores``.

    Raises:
        ValueError: If ``labels`` or ``folds`` is not indexed like ``scores``, or as ``choose``
            does.
    """
    for name, aligned in (("labels", labels), ("folds", folds)):
        if not aligned.index.equals(scores.index):
            raise ValueError(
                f"{name} must have the index of scores: one row per problem, in order."
            )
    fold_ids = np.sort(folds.unique())
    kept = [(folds != fold).to_numpy() for fold in fold_ids]
    rows = [choose(scores[others], labels[others]) for others in kept]
    return pd.DataFrame(rows, index=pd.Index(fold_ids, name=folds.name), columns=scores.columns)


def apply_fold_thresholds(
    scores: pd.DataFrame, fold_thresholds: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Predict each fold's problems with that fold's own row of thresholds.

    With rows chosen on the other folds (``cross_fitted_rows``), every prediction is made by
    thresholds chosen without the problem's own label.

    Args:
        scores: Scores, one column per tag, one row per problem.
        fold_thresholds: One row of thresholds per fold, indexed by fold id, with the columns of
            ``scores``.
        folds: Fold id per problem, with the index of ``scores``.

    Returns:
        0/1 predictions, with the index and columns of ``scores``.

    Raises:
        ValueError: If ``folds`` is not indexed like ``scores``, if a fold of ``folds`` has no
            row in ``fold_thresholds``, or as ``apply_thresholds`` does.
    """
    if not folds.index.equals(scores.index):
        raise ValueError("folds must have the index of scores: one fold per problem, in order.")
    fold_ids = np.sort(folds.unique())
    missing = pd.Index(fold_ids).difference(fold_thresholds.index).tolist()
    if missing:
        raise ValueError(f"The folds {missing} have no row of thresholds.")
    held_out = [
        apply_thresholds(scores[(folds == fold).to_numpy()], fold_thresholds.loc[fold])
        for fold in fold_ids
    ]
    return pd.concat(held_out).loc[scores.index]


def cross_fitted_lipton_thresholds(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Choose each fold's thresholds by the threshold rule, on the problems of the other folds.

    Applied to fold k, the thresholds of row k were chosen without fold k's labels, as those of
    ``model.thresholds.cross_fitted_thresholds``.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per problem.
        labels: 0/1 true labels, with the index and columns of ``scores``.
        folds: Fold id per problem, with the index of ``scores``.

    Returns:
        One row of ``lipton_thresholds`` per fold, indexed by the sorted fold ids (index named
        like ``folds``), with the columns of ``scores``.

    Raises:
        ValueError: As ``cross_fitted_rows`` does (``labels`` or ``folds`` not indexed like
            ``scores``), or as ``lipton_thresholds``.
    """
    return cross_fitted_rows(scores, labels, folds, lipton_thresholds)


class PlattLiptonThresholds(CrossFittedTagThresholds):
    """``CrossFittedTagThresholds`` whose thresholds follow the Platt + Lipton threshold rule.

    ``fit`` scores the dev set out of fold and refits the scorer as its parent does, then
    replaces the threshold choice only: row k of ``fold_thresholds_`` is ``lipton_thresholds``
    on the out-of-fold scores of the other folds, ``oof_predictions_`` predicts each fold with
    its own row, and ``thresholds_`` is ``lipton_thresholds`` on every out-of-fold score. Its
    scorer follows the parent's score-estimator contract and gives scores in [0, 1] (e.g.
    ``JevProbabilities``). The parent is never edited: its import path is frozen.
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *, groups: pd.Series) -> PlattLiptonThresholds:
        """Score the dev set out of fold, choose the thresholds by the rule, then refit.

        Args:
            X: The dev problems' rows, indexed by ``sample_id``, as the scorer takes them.
            y: 0/1 labels, one column per tag, with the index of ``X``.
            groups: Dev fold per problem, with the index of ``X``.

        Returns:
            The fitted estimator.

        Raises:
            TypeError: As ``CrossFittedTagThresholds.fit``.
            ValueError: As ``CrossFittedTagThresholds.fit``, or as ``lipton_threshold`` (a
                score outside [0, 1], a tag of every problem of the other folds, or a slope not
                above 0).
        """
        super().fit(X, y, groups=groups)
        scores = pd.DataFrame(self.oof_scores_, index=X.index, columns=y.columns)
        fold_thresholds = cross_fitted_lipton_thresholds(scores, y, groups)
        predictions = apply_fold_thresholds(scores, fold_thresholds, groups)
        self.oof_predictions_ = predictions.to_numpy(dtype=np.int64)
        self.fold_thresholds_ = fold_thresholds.to_numpy(dtype=float)
        self.thresholds_ = lipton_thresholds(scores, y).to_numpy(dtype=float)
        return self
