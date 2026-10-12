"""Notebook 11's comparison of calibrators: which map of Jev's probabilities, read by which rule.

A calibrator is a non-decreasing map of a score to a probability, fitted on labelled problems: it
changes the numbers, not their order. A rule that reads only the order, such as the F1-argmax
threshold of ``model.thresholds``, predicts the same tags when one increasing map, whatever it is,
is applied to every problem. That is exact when the held-out scores are among the fitting scores, as
they nearly always are on Jev's 0.01 grid: out of fold, ``best_threshold`` cuts at the midpoint
between two fitting scores, and a nonlinear map can move a held-out score that lies strictly between
them to the other side of the cut. A rule that reads the number itself, a cut at 0.5 or at half the
best F1 (Lipton et al., 2014), needs probabilities that mean what they say. Here each fold gets its
own map, fitted on the other folds, so the order across folds can move a little: a few problems near
a cut change sides, and the rankings shift. These helpers map a Jev method's dev scores by five
calibrators and the raw scores (``CALIBRATORS``): Platt scaling (Platt, 1999) on the logit, the map
of the Platt + Lipton rule; a logistic regression on the raw probability, close to what
scikit-learn's ``CalibratedClassifierCV(method="sigmoid")`` fits (scikit-learn also smooths the
0/1 targets, as Platt did); temperature scaling (Guo et al., 2017), one slope on the logit and no
shift; beta calibration (Kull, Silva Filho and Flach, 2017), whose two slopes are kept at or above
0, as the paper constrains them, so that its map never decreases; and isotonic regression
(Zadrozny and Elkan, 2002; Niculescu-Mizil and Caruana, 2005). Each map's probabilities are then
cut by three rules (``DECISION_RULES``) and ranked by ROC AUC and average precision;
``evaluation.calibration.calibration_table`` gives their Brier score and log loss.

The comparison is descriptive: the threshold rule was fixed before it (docs/adr/0011), and
nothing is chosen from it. It reads dev only (ADR 0001), sends no request and draws nothing. It
calls the rule's code rather than copies of it: every logistic regression it fits (both Platt
maps, temperature scaling without its intercept, and each fit of beta calibration) is the
rule's converged ``platt_lipton.rule_calibrator()``, so ``platt_logit`` is exactly the rule's
map and the maps compared are all maximum-likelihood fits. Platt scaling on the logit and
temperature scaling read ``platt_lipton.platt_feature`` (the [0, 1] check, the clip and the
logit); beta calibration's features run the same check, ``platt_lipton.check_unit_interval``,
and clip by the same ``config.PLATT_CLIP``, then take their own logs of the score and of its
complement. The out-of-fold maps are ``evaluation.calibration.cross_fitted_probabilities``, the
fold loop ``cross_fitted_rows`` and ``apply_fold_thresholds``, and the F1 at the best cut
``platt_lipton.best_f1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, logit  # pylint: disable=no-name-in-module  # ufuncs, built in C
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from codeforces_tags.config import PLATT_CLIP
from codeforces_tags.evaluation.calibration import (
    check_probabilities,
    cross_fitted_probabilities,
    isotonic_calibrator,
)
from codeforces_tags.model.platt_lipton import (
    apply_fold_thresholds,
    best_f1,
    check_unit_interval,
    cross_fitted_rows,
    platt_feature,
    rule_calibrator,
)
from codeforces_tags.model.thresholds import apply_thresholds, cross_fitted_predictions

# The raw scores ("none"), then each map of a score to a probability, in the order to report.
CALIBRATORS = ("none", "platt_logit", "platt_raw", "temperature", "beta", "isotonic")
# The rules that turn one calibrator's probabilities into 0/1 tags, in the order to report.
DECISION_RULES = ("at_0.5", "at_half_best_f1", "best_f1")
# The columns of ranking_table: the mean over the tags of ROC AUC and of average precision.
RANKING_COLUMNS = ("macro_auc", "macro_ap")
# The key of one calibrator's predictions by one rule, in rule_predictions.
RULE_KEY = "{calibrator} / {rule}"


def beta_features(scores: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Return the two features of beta calibration: the log of the score and of its complement.

    A logistic regression on them is ``expit(a log q - b log(1 - q) + c)``, the three-parameter
    beta calibration map (Kull, Silva Filho and Flach, 2017), which ``BetaCalibration`` fits
    with ``a`` and ``b`` at or above 0.

    Args:
        scores: One score in [0, 1] per problem, of shape ``(n,)`` or ``(n, 1)`` (an array or a
            one-column DataFrame).

    Returns:
        The ``(n, 2)`` float array ``[log q, -log(1 - q)]``, with ``q`` the score clipped to
        ``[PLATT_CLIP, 1 - PLATT_CLIP]``, so a failed answer's 0.0 gets finite features.

    Raises:
        ValueError: If ``scores`` holds more than one score per problem, or, before the clip, as
            ``platt_lipton.check_unit_interval`` does: a score outside [0, 1] or NaN, refused
            as ``platt_lipton.platt_feature`` refuses it.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            f"beta_features reads one score per problem: shape (n,) or (n, 1), not {values.shape}."
        )
    check_unit_interval(values, "Beta calibration")
    clipped = np.clip(values, PLATT_CLIP, 1 - PLATT_CLIP)
    return np.column_stack([np.log(clipped), -np.log(1 - clipped)])


def _platt_logit() -> BaseEstimator:
    """Platt scaling on the clipped logit: the Platt + Lipton rule's map (docs/adr/0011)."""
    return make_pipeline(FunctionTransformer(platt_feature), rule_calibrator())


def _temperature() -> BaseEstimator:
    """Temperature scaling for one tag: one slope (1 / T) on the logit, and no intercept."""
    return make_pipeline(
        FunctionTransformer(platt_feature), rule_calibrator().set_params(fit_intercept=False)
    )


class BetaCalibration(ClassifierMixin, BaseEstimator):
    """Beta calibration of one tag's scores, with both slopes kept at or above 0.

    The map is ``expit(a log q - b log(1 - q) + c)``, a logistic regression on
    ``beta_features`` (Kull, Silva Filho and Flach, 2017). Free, a negative ``a`` or ``b`` makes
    it decrease near 0 or near 1; with ``a, b >= 0``, as the paper constrains them, it never
    decreases. So a free fit whose slope on log q is negative is refitted on -log(1 - q) alone
    (``a = 0``); one whose slope on -log(1 - q) is negative, on log q alone (``b = 0``); and a
    refit whose one slope is still negative gives way to the constant prevalence of the labels
    (``a = b = 0``). Each fit is ``platt_lipton.rule_calibrator()``, to convergence. As a
    classifier, it reads the scores as ``evaluation.calibration.cross_fitted_probabilities``
    passes them, a one-column DataFrame.

    Attributes:
        classes_: The two labels, sorted: the probability is that of the second.
        coef_: ``[[a, b]]``, a ``(1, 2)`` array, 0 where a feature was dropped.
        intercept_: ``[c]``, a ``(1,)`` array.
    """

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BetaCalibration:
        """Fit the map on one score and one label per problem.

        Args:
            X: One score in [0, 1] per problem, as ``beta_features`` reads it.
            y: One label per problem, of two classes.

        Returns:
            The fitted calibration.

        Raises:
            ValueError: If ``y`` does not hold exactly two classes, naming those it holds, or as
                ``beta_features`` does.
        """
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if self.classes_.size != 2:
            raise ValueError(
                "Beta calibration maps one tag's scores: its labels hold two classes, not "
                f"{self.classes_.tolist()}."
            )
        positive = (labels == self.classes_[1]).astype(int)
        self.coef_, self.intercept_ = _nonnegative_slopes(beta_features(X), positive)
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Map each score to the probability of each class.

        Args:
            X: One score in [0, 1] per problem, as ``beta_features`` reads it.

        Returns:
            The ``(n, 2)`` array of the probabilities of ``classes_``: column 1 is the mapped
            probability of the tag.

        Raises:
            NotFittedError: If the calibration was not fitted.
            ValueError: As ``beta_features`` does.
        """
        tag = expit(self._logits(X))
        return np.column_stack([1 - tag, tag])

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict the class of each score: the second where its probability is above 0.5.

        ``cross_val_predict`` asks every estimator for it, though only ``predict_proba`` is read.

        Args:
            X: One score in [0, 1] per problem, as ``beta_features`` reads it.

        Returns:
            One label of ``classes_`` per problem.

        Raises:
            NotFittedError: If the calibration was not fitted.
            ValueError: As ``beta_features`` does.
        """
        above = self._logits(X) > 0
        return self.classes_[above.astype(int)]

    def _logits(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """The log-odds of the tag, ``a log q - b log(1 - q) + c``, one per problem."""
        check_is_fitted(self)
        return beta_features(X) @ self.coef_[0] + self.intercept_[0]


def _nonnegative_slopes(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit beta calibration's ``coef_`` and ``intercept_`` with both slopes at or above 0."""
    free = rule_calibrator().fit(features, labels)
    if (free.coef_ >= 0).all():
        return free.coef_, free.intercept_
    # A negative slope on log q (column 0) keeps -log(1 - q) alone; one on -log(1 - q), log q.
    kept = 1 if free.coef_[0, 0] < 0 else 0
    alone = rule_calibrator().fit(features[:, [kept]], labels)
    slopes = np.zeros((1, 2))
    if alone.coef_[0, 0] >= 0:
        slopes[0, kept] = alone.coef_[0, 0]
        return slopes, alone.intercept_
    return slopes, np.array([logit(labels.mean())])


# Each map of CALIBRATORS, by name: a function that returns it unfitted.
_MAPS: dict[str, Callable[[], BaseEstimator]] = {
    "platt_logit": _platt_logit,
    "platt_raw": rule_calibrator,
    "temperature": _temperature,
    "beta": BetaCalibration,
    "isotonic": isotonic_calibrator,
}


def make_calibrator(name: str) -> BaseEstimator:
    """Return one calibrator of ``CALIBRATORS``, unfitted, as a new object on every call.

    Args:
        name: ``"platt_logit"`` (Platt scaling on the clipped logit), ``"platt_raw"`` (a
            logistic regression on the raw probability, close to what
            ``CalibratedClassifierCV(method="sigmoid")`` fits: scikit-learn also smooths the
            targets, as Platt did), ``"temperature"`` (one slope on the clipped logit, no shift),
            ``"beta"`` (``BetaCalibration``: a logistic regression on ``beta_features`` whose
            two slopes are kept at or above 0, so that the map never decreases) or
            ``"isotonic"``.

    Returns:
        The map, which ``evaluation.calibration.cross_fitted_probabilities`` clones and fits:
        a classifier (the four logistic maps) or ``isotonic_calibrator()``.

    Raises:
        ValueError: If ``name`` is ``"none"``, which maps nothing (``calibrated_probabilities``
            returns the scores themselves), or is not a calibrator; the message names every
            calibrator of ``CALIBRATORS``.
    """
    if name not in _MAPS:
        raise ValueError(
            f"{name!r} is no map: the calibrators are {', '.join(map(repr, CALIBRATORS))}, and "
            "'none' keeps the scores as they are."
        )
    return _MAPS[name]()


def calibrated_probabilities(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> dict[str, pd.DataFrame]:
    """Map the dev scores by every calibrator, each fold by maps fitted on the other folds.

    Args:
        scores: Out-of-fold probabilities from 0 to 1 (a Jev method's dev scores), one column per
            tag, one row per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.

    Returns:
        By calibrator, in the order of ``CALIBRATORS``: ``"none"``, a float copy of the scores;
        every other one, ``cross_fitted_probabilities`` of its ``make_calibrator``. Each frame has
        the index and columns of ``scores``.

    Raises:
        ValueError: As ``cross_fitted_probabilities`` does: misaligned labels or folds, a missing
            or negative fold, fewer than two folds, or a tag with one class outside a fold; or
            if a score lies outside [0, 1] or is NaN, as ``platt_lipton.platt_feature`` refuses
            it.
    """
    probabilities = {"none": scores.astype(float)}
    for name in CALIBRATORS[1:]:
        probabilities[name] = cross_fitted_probabilities(
            scores, labels, folds, make_calibrator(name)
        )
    return probabilities


def _half_best_f1s(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Half of each tag's best F1 on these problems, or inf (nothing predicted) without a positive.

    Half of a best F1 of 0.0 would cut at 0 and predict every problem: a tag that no problem has
    is predicted for none instead, as ``best_threshold`` does.
    """
    halves = [
        best_f1(scores[tag].to_numpy(), labels[tag].to_numpy()) / 2
        if labels[tag].any()
        else math.inf
        for tag in scores.columns
    ]
    return pd.Series(halves, index=scores.columns, name="threshold", dtype=float)


def _rules(
    probabilities: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> dict[str, pd.DataFrame]:
    """One calibrator's 0/1 predictions by each rule of ``DECISION_RULES``, in its order."""
    # cross_fitted_predictions checks the labels and folds first, for the three rules.
    _, best = cross_fitted_predictions(probabilities, labels, folds)
    halves = cross_fitted_rows(probabilities, labels, folds, _half_best_f1s)
    by_rule = {
        "at_0.5": apply_thresholds(probabilities, pd.Series(0.5, index=probabilities.columns)),
        "at_half_best_f1": apply_fold_thresholds(probabilities, halves, folds),
        "best_f1": best,
    }
    return {rule: by_rule[rule].astype(np.int64) for rule in DECISION_RULES}


def rule_predictions(
    probabilities: Mapping[str, pd.DataFrame], labels: pd.DataFrame, folds: pd.Series
) -> dict[str, pd.DataFrame]:
    """Turn each calibrator's probabilities into 0/1 tags by each rule of ``DECISION_RULES``.

    ``"at_0.5"`` predicts a tag where its probability reaches 0.5. ``"at_half_best_f1"`` cuts
    fold k at half the best F1 that the other folds' probabilities reach (Lipton et al., 2014),
    and a tag that no problem of the other folds has at inf. ``"best_f1"`` cuts fold k at the
    F1-argmax threshold of the other folds, ``cross_fitted_predictions``.

    Args:
        probabilities: Probabilities by calibrator, as ``calibrated_probabilities`` returns
            them, in the order to report.
        labels: 0/1 labels of the dev problems, one column per tag.
        folds: Dev fold per problem, with the index of ``labels``.

    Returns:
        For each calibrator, in order, then each rule, in order, the key
        ``RULE_KEY.format(calibrator=..., rule=...)`` and the 0/1 int64 predictions, with the
        index and columns of ``labels``.

    Raises:
        ValueError: As ``cross_fitted_predictions`` does: probabilities not indexed like
            ``labels``, other tags, or folds that ``check_dev_folds`` refuses.
    """
    predictions = {}
    for calibrator, frame in probabilities.items():
        for rule, predicted in _rules(frame, labels, folds).items():
            predictions[RULE_KEY.format(calibrator=calibrator, rule=rule)] = predicted
    return predictions


def ranking_table(probabilities: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Rank each calibrator's probabilities: the mean over the tags of ROC AUC and AP.

    Both read only the order of the probabilities: one increasing map applied to every problem
    keeps them, and a non-decreasing one (isotonic regression) can lower them by the ties it
    makes. Out-of-fold maps differ by fold, so they can move them a little either way.

    Args:
        probabilities: Probabilities by calibrator, as ``calibrated_probabilities`` returns
            them, in the order to report.
        labels: 0/1 labels, one column per tag, one row per problem.

    Returns:
        One row per calibrator, in order (index ``calibrator``), with the columns
        ``RANKING_COLUMNS``: the mean over the tags of ``roc_auc_score`` and of
        ``average_precision_score``.

    Raises:
        ValueError: As ``evaluation.calibration.check_probabilities`` does, or, before any
            score, naming the tags of ``labels`` that have one class: neither score is defined
            there.
    """
    check_probabilities(probabilities, labels)
    one_class = [str(tag) for tag in labels.columns if labels[tag].nunique() < 2]
    if one_class:
        raise ValueError(
            f"The tags {one_class} have one class: ROC AUC and average precision need positive "
            "and negative problems of each tag."
        )
    rows = {name: _macro_ranking(frame, labels) for name, frame in probabilities.items()}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(RANKING_COLUMNS))
    return table.rename_axis("calibrator")


def _macro_ranking(frame: pd.DataFrame, labels: pd.DataFrame) -> list[float]:
    """The mean over the tags of ROC AUC, then of average precision: a row of ``ranking_table``."""
    return [
        float(np.mean([metric(labels[tag], frame[tag]) for tag in labels.columns]))
        for metric in (roc_auc_score, average_precision_score)
    ]
