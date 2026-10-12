"""Read out-of-fold scores as probabilities: calibrate them, out of fold.

A probability is calibrated when, among the problems given a probability p of a tag, a share p
holds the tag. A logistic regression's logits give probabilities through the logistic function; a
linear SVM's margins do not, so they are mapped to [0, 1] by a calibrator. The four
probabilities compared (``config.CALIBRATION_METHODS``) are built from out-of-fold dev scores
only, and each map is fitted on the folds that do not hold the problems it maps, as the
thresholds are chosen (docs/adr/0001-model-selection-protocol.md). None of them changes a
prediction. ``calibration_table`` scores them by Brier score, log loss and Brier skill, with
group bootstrap intervals. ``platt_maps``, ``mapped_predictions`` and ``lipton_table`` map the
cross-fitted thresholds to probabilities instead: the maps increase, so F1 does not move, except
where float rounding ties a score with its threshold, which gains a prediction (the caller counts
the cells that differ); and each threshold is set beside half the F1 it reaches (Lipton et al.,
2014). This module draws nothing: ``analysis.plotting`` draws the reliability diagrams.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit  # pylint: disable=no-name-in-module  # a ufunc, built in C
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from codeforces_tags.config import BOOTSTRAP_CONFIDENCE, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from codeforces_tags.evaluation.bootstrap import percentile_interval, resample_weights
from codeforces_tags.model.thresholds import check_dev_folds

POOLED = "all"  # the tag of each method's row over all problem-tag pairs
METRICS = ("brier", "log_loss", "brier_skill")
# Each metric, then the bounds of its interval: the columns of calibration_table.
CALIBRATION_COLUMNS = tuple(
    f"{metric}{bound}" for metric in METRICS for bound in ("", "_low", "_high")
)
LIPTON_COLUMNS = ("threshold", "probability_threshold", "f1", "half_f1")


def platt_calibrator() -> LogisticRegression:
    """Return an unfitted Platt scaling: a sigmoid of one score, fitted without a penalty.

    ``C=np.inf`` turns the penalty off (``penalty=None`` is deprecated since scikit-learn 1.8),
    so the map is ``expit(a * score + b)`` fitted by likelihood, with lbfgs at scikit-learn's
    default tolerance: close to the maximum-likelihood map, not exactly it (the threshold rule
    fits ``model.platt_lipton.rule_calibrator`` to convergence). It fits without a warning even
    on scores that separate the classes.

    Returns:
        ``LogisticRegression(C=np.inf)``, a classifier: its probability of the tag is column 1
        of ``predict_proba``.
    """
    return LogisticRegression(C=np.inf)


def isotonic_calibrator() -> IsotonicRegression:
    """Return an unfitted isotonic calibration: a non-decreasing step map of one score.

    Returns:
        ``IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)``, a regressor: its
        ``predict`` is the probability. ``"clip"`` maps a held-out score outside the range of
        the scores it was fitted on to the map's end value, where the default ``"nan"`` would
        silently give NaN.
    """
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)


def sigmoid_probabilities(logits: pd.DataFrame) -> pd.DataFrame:
    """Turn logits into probabilities with the logistic function, as a logistic regression does.

    Args:
        logits: Log-odds, one column per tag, one row per problem (e.g. a logistic regression's
            out-of-fold ``decision_function``).

    Returns:
        ``expit(logits)``, with the index and columns of ``logits``.
    """
    return pd.DataFrame(
        expit(logits.to_numpy(dtype=float)), index=logits.index, columns=logits.columns
    )


def minmax_probabilities(scores: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Scale each fold's scores to [0, 1] with the minimum and maximum of the other folds.

    Fold k's score s of a tag becomes ``(s - min) / (max - min)``, with the tag's min and max
    over the other folds' scores, clipped to [0, 1]: the naive reading of a margin as a
    probability, with nothing fitted to fold k.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per dev problem.
        folds: Dev fold per problem, with the index of ``scores``.

    Returns:
        Probabilities in [0, 1], float64, with the index and columns of ``scores``.

    Raises:
        ValueError: As ``check_dev_folds`` does (misaligned folds, a missing or negative fold,
            fewer than two folds), or if a tag has one score on all the folds but one, which
            leaves nothing to scale by.
    """
    # Min-max scaling reads no label: the scores stand in for the labels' index.
    check_dev_folds(scores.index, scores, folds)
    values = scores.to_numpy(dtype=float)
    probabilities = np.full_like(values, np.nan)  # a row no fold held out stays NaN
    for fold in np.sort(folds.unique()):
        held_out = (folds == fold).to_numpy()
        low, high = values[~held_out].min(axis=0), values[~held_out].max(axis=0)
        constant = [str(tag) for tag in scores.columns[high == low]]
        if constant:
            raise ValueError(
                f"Outside fold {fold}, the tags {constant} have one score: min-max scaling on "
                "the other folds would divide by zero."
            )
        probabilities[held_out] = np.clip((values[held_out] - low) / (high - low), 0, 1)
    return pd.DataFrame(probabilities, index=scores.index, columns=scores.columns)


def cross_fitted_probabilities(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series, calibrator: BaseEstimator
) -> pd.DataFrame:
    """Calibrate each tag's out-of-fold scores: fold k with a calibrator fitted on the other folds.

    Per tag, ``cross_val_predict`` fits a clone of ``calibrator`` on the other folds' scores and
    labels of the tag (``LeaveOneGroupOut`` over ``folds``), and maps fold k's scores. A
    classifier (``platt_calibrator``) reads the tag's scores as a one-column DataFrame and gives
    column 1 of ``predict_proba``; any other estimator (``isotonic_calibrator``, which has no
    ``predict_proba``) reads them as a 1-D array and gives ``predict``.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.
        calibrator: The unfitted map of one score to a probability, cloned for every fit.

    Returns:
        Probabilities, float64, with the index and columns of ``scores``.

    Raises:
        ValueError: As ``check_dev_folds`` does, if ``labels`` has other columns than
            ``scores``, or, before anything is fitted, if the folds other than one hold only
            positives or only negatives of a tag: the error names the tags and that fold.
    """
    check_dev_folds(scores.index, labels, folds)
    if not labels.columns.equals(scores.columns):
        raise ValueError(
            f"labels must have the columns of scores, in order: {list(labels.columns)} against "
            f"{list(scores.columns)}."
        )
    _check_both_classes_outside_each_fold(labels, folds)
    classifier = is_classifier(calibrator)
    columns = []
    for tag in scores.columns:
        X = scores[[tag]] if classifier else scores[tag].to_numpy()
        predicted = cross_val_predict(
            clone(calibrator),
            X,
            labels[tag],
            cv=LeaveOneGroupOut(),
            groups=folds,
            method="predict_proba" if classifier else "predict",
        )
        columns.append(predicted[:, 1] if classifier else predicted)
    return pd.DataFrame(
        np.column_stack(columns).astype(float), index=scores.index, columns=scores.columns
    )


def _check_both_classes_outside_each_fold(labels: pd.DataFrame, folds: pd.Series) -> None:
    """Fail unless the folds other than each one hold positives and negatives of every tag."""
    for fold in np.sort(folds.unique()):
        others = labels[(folds != fold).to_numpy()]
        one_class = [str(tag) for tag in labels.columns[others.nunique() < 2]]
        if one_class:
            raise ValueError(
                f"Outside fold {fold}, the tags {one_class} have one class: a calibrator fitted "
                "on the other folds needs positive and negative problems of each tag."
            )


def probabilities_by_method(
    svm_scores: pd.DataFrame, logits: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> dict[str, pd.DataFrame]:
    """Build the four probabilities of ``config.CALIBRATION_METHODS``, out of fold.

    Args:
        svm_scores: The tag model's out-of-fold SVM margins, one column per tag.
        logits: The first look's out-of-fold logits of the same tags, selected by name.
        labels: 0/1 labels of the dev problems, one column per tag.
        folds: Dev fold per problem, with the index of ``labels``.

    Returns:
        By method id, in the order of ``CALIBRATION_METHODS``: ``logistic_regression``, the
        sigmoid of the logits; ``svm_minmax``, the margins min-max scaled on the other folds;
        ``svm_platt`` and ``svm_isotonic``, the margins mapped by a Platt or an isotonic
        calibrator fitted on the other folds. Each frame has the index and columns of
        ``labels``.

    Raises:
        ValueError: If ``svm_scores`` or ``logits`` does not have the index and columns of
            ``labels``, in order, or as ``minmax_probabilities`` and
            ``cross_fitted_probabilities`` do.
    """
    for name, scores in (("svm_scores", svm_scores), ("logits", logits)):
        if not (scores.index.equals(labels.index) and scores.columns.equals(labels.columns)):
            raise ValueError(
                f"{name} must have the index and columns of labels: the same problems and tags, "
                "in order (select a model's tags by name: scores[labels.columns])."
            )
    return {
        "logistic_regression": sigmoid_probabilities(logits),
        "svm_minmax": minmax_probabilities(svm_scores, folds),
        "svm_platt": cross_fitted_probabilities(svm_scores, labels, folds, platt_calibrator()),
        "svm_isotonic": cross_fitted_probabilities(
            svm_scores, labels, folds, isotonic_calibrator()
        ),
    }


def check_probabilities(probabilities: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> None:
    """Fail unless each method gives a probability in [0, 1] of every problem and tag of labels.

    Args:
        probabilities: Probabilities by method, as ``probabilities_by_method`` returns them.
        labels: 0/1 labels, one column per tag, one row per problem.

    Raises:
        ValueError: Naming the method, if a method's probabilities do not have the index and
            columns of ``labels``, in order, hold NaN, or lie outside [0, 1]; or if there is no
            method.
    """
    if not probabilities:
        raise ValueError("No method to check: pass the probabilities of at least one method.")
    for method, frame in probabilities.items():
        problem = _probability_problem(frame, labels)
        if problem:
            raise ValueError(f"The probabilities of {method!r} {problem}.")


def _probability_problem(frame: pd.DataFrame, labels: pd.DataFrame) -> str | None:
    """Say what keeps a frame from being probabilities of the labels, or return None."""
    if not (frame.index.equals(labels.index) and frame.columns.equals(labels.columns)):
        return "must have the index and columns of labels: the same problems and tags, in order"
    values = frame.to_numpy(dtype=float)
    if np.isnan(values).any():
        return "hold NaN"
    if ((values < 0) | (values > 1)).any():
        return "lie outside [0, 1]"
    return None


def calibration_table(  # pylint: disable=too-many-arguments
    probabilities: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    groups: pd.Series,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Score each method's probabilities per tag and over all pairs, with bootstrap intervals.

    Per tag: the Brier score (the mean of ``(p - y)^2``), the log loss (the mean of
    ``-(y log q + (1 - y) log(1 - q))``, with ``q`` the probability clipped to ``[eps, 1 - eps]``
    as scikit-learn's ``log_loss`` clips it, so an exact 0 or 1 costs at most 36.04) and the
    Brier skill ``1 - brier / (prevalence (1 - prevalence))``: the share of the Brier score of a
    constant forecast of the prevalence that the method saves. Over all problem-tag pairs
    (``POOLED``): the mean Brier score and log loss, and the skill
    ``1 - sum of brier / sum of prevalence (1 - prevalence)`` over the tags. Every resample of
    the group bootstrap recomputes them all, the prevalence included, and every method is
    scored on the same resamples, those ``evaluate`` draws for the same arguments. A skill is
    NaN, never infinite, where a tag has one class: in ``labels`` for the estimate, in a
    resample for that resample, and so in the bounds of the tag's interval.

    Args:
        probabilities: Probabilities by method, as ``probabilities_by_method`` returns them,
            in the order to report.
        labels: 0/1 labels, one column per tag, one row per problem.
        groups: Near-duplicate group per problem, with the index of ``labels``.
        resamples: Group bootstrap resamples.
        confidence: Share of the resamples each interval covers.
        seed: Seed of the resamples.

    Returns:
        One row per method and tag (index ``method``, ``tag``), each method's tags in the order
        of ``labels``' columns, then ``POOLED``; the columns ``CALIBRATION_COLUMNS``: each
        metric of ``METRICS``, then the low and high bounds of its percentile interval.

    Raises:
        ValueError: As ``check_probabilities`` does, if ``groups`` is not indexed like
            ``labels``, or if a problem has no group.
    """
    check_probabilities(probabilities, labels)
    if not groups.index.equals(labels.index):
        raise ValueError("groups must have the index of labels: one group per problem, in order.")
    truth = labels.to_numpy(dtype=float)
    weights = resample_weights(groups, resamples, seed).astype(float)
    tags = pd.Index([*labels.columns, POOLED], name="tag")
    tables = {
        method: _method_table(frame.to_numpy(dtype=float), truth, weights, confidence, tags)
        for method, frame in probabilities.items()
    }
    return pd.concat(tables, names=["method", "tag"])


def _method_table(
    predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray, confidence: float, tags: pd.Index
) -> pd.DataFrame:
    """One method's rows of ``calibration_table``: each metric and its interval, per tag."""
    losses = _pair_losses(predicted, truth)
    estimates = _calibration_scores(losses, truth, np.ones((1, len(truth))))[0]
    low, high = percentile_interval(_calibration_scores(losses, truth, weights), confidence)
    # (metric, estimate or bound, tag): the rows of the reshape follow CALIBRATION_COLUMNS.
    values = np.stack([estimates, low, high], axis=1).reshape(len(CALIBRATION_COLUMNS), -1)
    return pd.DataFrame(values.T, index=tags, columns=list(CALIBRATION_COLUMNS))


def _pair_losses(predicted: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The squared error and the log loss of each problem-tag pair, finite on exact 0 and 1."""
    eps = sys.float_info.epsilon  # np.finfo(float).eps, where scikit-learn's log_loss clips
    clipped = np.clip(predicted, eps, 1 - eps)
    log_losses = -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))
    return (predicted - truth) ** 2, log_losses


def _calibration_scores(
    losses: tuple[np.ndarray, np.ndarray], truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Each metric of ``METRICS`` per tag, then over all pairs: (len(weights), 3, tags + 1)."""
    squared, log_losses = losses
    drawn = weights.sum(axis=1, keepdims=True)
    brier, log_loss = weights @ squared / drawn, weights @ log_losses / drawn
    prevalence = weights @ truth / drawn
    variance = prevalence * (1 - prevalence)
    # Skill over all pairs: 1 - sum of brier / sum of variance, a ratio of sums over the tags.
    brier_sums = np.column_stack([brier, brier.sum(axis=1)])
    variance_sums = np.column_stack([variance, variance.sum(axis=1)])
    # NaN where a tag has one class: no division by 0, so no RuntimeWarning and no infinity.
    ratio = np.divide(
        brier_sums, variance_sums, out=np.full(brier_sums.shape, np.nan), where=variance_sums > 0
    )
    return np.stack(
        [
            np.column_stack([brier, brier.mean(axis=1)]),
            np.column_stack([log_loss, log_loss.mean(axis=1)]),
            1 - ratio,
        ],
        axis=1,
    )


def platt_maps(scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, LogisticRegression]:
    """Fit one Platt map per tag on all its out-of-fold scores, and check that each increases.

    Unlike the per-fold calibrators of ``cross_fitted_probabilities``, these maps never give a
    probability to compare by Brier score: they map the scores and the cross-fitted thresholds
    together (``mapped_predictions``, ``lipton_table``), and a map that increases keeps their
    order.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.

    Returns:
        By tag, in the order of ``scores``' columns: ``platt_calibrator()`` fitted on the tag's
        scores (a one-column DataFrame) and labels.

    Raises:
        ValueError: If ``labels`` does not have the index and columns of ``scores``, in order;
            if a tag has one class; or, naming the tags, if a map's slope is not above 0: it
            would reverse the order of the scores, and so change predictions.
    """
    if not (labels.index.equals(scores.index) and labels.columns.equals(scores.columns)):
        raise ValueError(
            "labels must have the index and columns of scores: the same problems and tags, in "
            "order."
        )
    one_class = [str(tag) for tag in labels.columns[labels.nunique() < 2]]
    if one_class:
        raise ValueError(
            f"The tags {one_class} have one class: a Platt map needs positive and negative "
            "problems of each tag."
        )
    maps = {tag: platt_calibrator().fit(scores[[tag]], labels[tag]) for tag in scores.columns}
    flat = [str(tag) for tag, fitted in maps.items() if not fitted.coef_[0, 0] > 0]
    if flat:
        raise ValueError(
            f"The Platt maps of the tags {flat} do not increase (slope at or below 0): they "
            "would reverse the order of the scores, and change the predictions of mapped "
            "thresholds."
        )
    return maps


def mapped_predictions(
    scores: pd.DataFrame,
    fold_thresholds: pd.DataFrame,
    folds: pd.Series,
    maps: Mapping[str, LogisticRegression],
) -> pd.DataFrame:
    """Predict each fold with its cross-fitted thresholds, scores and thresholds Platt-mapped.

    Problem i of fold k gets tag t when ``expit(a * score + b) >= expit(a * threshold + b)``,
    with ``score`` its score, ``threshold`` fold k's threshold of t and ``(a, b)`` the slope and
    intercept of t's map. This closed form gives ``predict_proba(...)[:, 1]`` bit for bit, but
    keeps an infinite threshold infinite: ``best_threshold``'s inf predicts nothing, where
    ``expit(inf) = 1.0`` would predict a score whose probability rounds to 1. With maps that
    increase (``platt_maps``), this is in exact arithmetic the prediction of the scores
    themselves. In float64 ``expit`` is only non-decreasing: a score that maps to
    its threshold's probability gains a prediction, and none is ever lost, so the caller
    counts the cells that differ.

    Args:
        scores: Out-of-fold scores, one column per tag, one row per dev problem.
        fold_thresholds: Each fold's per-tag thresholds, one row per fold id (e.g.
            ``OutOfFold.fold_thresholds``), with the columns of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.
        maps: Platt map per tag, as ``platt_maps`` returns them, in the order of ``scores``'
            columns.

    Returns:
        0/1 int64 predictions, with the index and columns of ``scores``.

    Raises:
        ValueError: If ``maps`` or ``fold_thresholds`` does not hold the tags of ``scores``
            in order, if ``folds`` is not indexed like ``scores``, or if a fold has no row in
            ``fold_thresholds``.
    """
    _check_same_tags(scores.columns, maps=maps, fold_thresholds=fold_thresholds.columns)
    if not folds.index.equals(scores.index):
        raise ValueError("folds must have the index of scores: one fold per problem, in order.")
    missing = folds[~folds.isin(fold_thresholds.index)].unique().tolist()
    if missing:
        raise ValueError(
            f"The folds {missing} have no row in fold_thresholds: each fold is predicted with "
            "the thresholds chosen on the other folds."
        )
    rows = fold_thresholds.index.get_indexer(folds.to_numpy())
    thresholds = _platt_map(fold_thresholds.to_numpy(dtype=float), maps)[rows]
    predicted = _platt_map(scores.to_numpy(dtype=float), maps) >= thresholds
    return pd.DataFrame(predicted.astype(np.int64), index=scores.index, columns=scores.columns)


def lipton_table(
    thresholds: pd.Series, maps: Mapping[str, LogisticRegression], tag_f1: pd.Series
) -> pd.DataFrame:
    """Set each tag's threshold, mapped to a probability, beside half the F1 it reaches.

    For calibrated probabilities, the threshold that maximises F1 is half that F1 (Lipton, Elkan
    and Narayanaswamy, 2014): mapped through its tag's Platt map, a final threshold should then
    lie near F1 / 2, far below 0.5 for a tag whose F1 is low.

    Args:
        thresholds: Final per-tag thresholds, on the scale of the scores (e.g.
            ``OutOfFold.thresholds``).
        maps: Platt map per tag, as ``platt_maps`` returns them.
        tag_f1: Per-tag F1 that ``thresholds`` reach on the scores they were chosen on. Its
            index may have another name than ``thresholds``', not other tags.

    Returns:
        One row per tag (index ``tag``), with the columns ``LIPTON_COLUMNS``: ``threshold``,
        ``probability_threshold`` (mapped as ``mapped_predictions`` maps it: an infinite
        threshold stays infinite), ``f1`` and ``half_f1``.

    Raises:
        ValueError: If ``maps`` or ``tag_f1`` does not hold the tags of ``thresholds``, in order.
    """
    _check_same_tags(thresholds.index, maps=maps, tag_f1=tag_f1.index)
    values = thresholds.to_numpy(dtype=float)
    f1 = tag_f1.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "threshold": values,
            "probability_threshold": _platt_map(values[np.newaxis], maps)[0],
            "f1": f1,
            "half_f1": f1 / 2,
        },
        index=pd.Index(thresholds.index, name="tag"),
    )


def _check_same_tags(tags: pd.Index, **others: Iterable[str]) -> None:
    """Fail unless each of ``others`` holds ``tags``, in the same order."""
    for name, other in others.items():
        if list(other) != list(tags):
            raise ValueError(
                f"{name} must hold the tags {list(tags)}, in this order, not {list(other)}."
            )


def _platt_map(values: np.ndarray, maps: Mapping[str, LogisticRegression]) -> np.ndarray:
    """Map column t of ``values`` through the t-th map, ``expit(a x + b)``, keeping infinities."""
    slopes = np.array([fitted.coef_[0, 0] for fitted in maps.values()])
    intercepts = np.array([fitted.intercept_[0] for fitted in maps.values()])
    return np.where(np.isinf(values), values, expit(values * slopes + intercepts))
