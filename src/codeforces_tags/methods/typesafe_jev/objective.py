"""The search's objective: the balanced Brier score of Jev's probabilities (docs/adr/0010).

For focus tag t, with prevalence pi_t in the learning folds, a problem's error on t is weighted
so that the tag's positives and negatives count equally: w_t(1) = 1 / (2 pi_t) and
w_t(0) = 1 / (2 (1 - pi_t)). A problem's score is one minus the mean over the 8 focus tags of
w_t(y) (p - y)^2, the number GEPA maximises; over problems whose prevalences are pi, the mean of
one minus the scores is the macro balanced Brier score: per tag, half the mean squared error on
its positives plus half on its negatives, averaged over the tags (lower is better). A failed Jev
answer scores 0.0 on every tag (``answers.probabilities``), so it costs w_t(1) on each of its
problem's tags and nothing on the others.
"""

from __future__ import annotations

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS


def tag_weights(labels: pd.DataFrame) -> pd.DataFrame:
    """Weigh each focus tag's positives and negatives so that they count equally.

    Args:
        labels: The 0/1 focus labels of the problems the prevalences come from (the learning
            folds), one column per focus tag, in ``FOCUS_TAGS`` order.

    Returns:
        One row per focus tag, in ``FOCUS_TAGS`` order, and two float columns: ``1``, the weight
        1 / (2 pi_t) of a positive, and ``0``, the weight 1 / (2 (1 - pi_t)) of a negative,
        where pi_t is the tag's mean label.

    Raises:
        ValueError: If the columns are not the focus tags, a label is not 0 or 1, or a tag has
            no positive or no negative (its weight would be infinite); the message names the
            tags.
    """
    _check_labels(labels)
    prevalence = labels.mean().astype("float64")
    degenerate = [tag for tag in FOCUS_TAGS if not 0.0 < prevalence[tag] < 1.0]
    if degenerate:
        raise ValueError(f"tags with no positive or no negative have no weight: {degenerate}")
    weights = pd.DataFrame(
        {0: 1.0 / (2.0 * (1.0 - prevalence)), 1: 1.0 / (2.0 * prevalence)},
        index=pd.Index(FOCUS_TAGS, name="tag"),
    )
    return weights


def weighted_errors(
    probabilities: pd.DataFrame, labels: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    """Compute each problem's weighted squared error on each focus tag, w_t(y) (p - y)^2.

    Args:
        probabilities: Jev's probability of each focus tag, one row per problem.
        labels: The problems' 0/1 focus labels, with the index of ``probabilities`` (same
            values, same order).
        weights: ``tag_weights`` of the problems the prevalences come from.

    Returns:
        Float errors with the index and the columns (``FOCUS_TAGS``) of ``probabilities``.

    Raises:
        ValueError: If a frame's columns are not the focus tags, a label is not 0 or 1, the
            two indexes differ, or ``weights`` is not one row per focus tag with columns 0 and 1.
    """
    _check_pair(probabilities, labels)
    _check_weights(weights)
    labels = labels.astype("float64")
    weight = labels * weights[1] + (1.0 - labels) * weights[0]
    return weight * (probabilities.astype("float64") - labels) ** 2


def problem_scores(
    probabilities: pd.DataFrame, labels: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    """Score each problem as GEPA sees it: one minus its mean weighted error over the tags.

    A confident error on a rare tag can make a score negative; GEPA only sums and averages
    scores.

    Args:
        probabilities: Jev's probability of each focus tag, one row per problem.
        labels: The problems' 0/1 focus labels, with the index of ``probabilities``.
        weights: ``tag_weights`` of the problems the prevalences come from.

    Returns:
        One float score per problem, with the index of ``probabilities``, named ``"score"``.

    Raises:
        ValueError: As ``weighted_errors``.
    """
    errors = weighted_errors(probabilities, labels, weights)
    return (1.0 - errors.mean(axis=1)).rename("score")


def balanced_brier(probabilities: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Compute each focus tag's balanced Brier score on a set of problems.

    Per tag: half the mean of (1 - p)^2 over its positives plus half the mean of p^2 over its
    negatives; lower is better, and a constant 0.5 scores 0.25.

    Args:
        probabilities: Jev's probability of each focus tag, one row per problem.
        labels: The problems' 0/1 focus labels, with the index of ``probabilities``.

    Returns:
        One float per focus tag, indexed by ``FOCUS_TAGS``, named ``"balanced_brier"``; NaN for
        a tag with no positive or no negative among the problems.

    Raises:
        ValueError: As ``weighted_errors``, for the two frames.
    """
    _check_pair(probabilities, labels)
    squared = (probabilities.astype("float64") - labels.astype("float64")) ** 2
    positive = labels == 1
    on_positives = squared.where(positive).mean()
    on_negatives = squared.where(~positive).mean()
    scores = 0.5 * on_positives + 0.5 * on_negatives
    return pd.Series(
        scores.to_numpy(), index=pd.Index(FOCUS_TAGS, name="tag"), name="balanced_brier"
    )


def macro_balanced_brier(probabilities: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Average the focus tags' balanced Brier scores, leaving out the tags it is NaN for.

    Args:
        probabilities: Jev's probability of each focus tag, one row per problem.
        labels: The problems' 0/1 focus labels, with the index of ``probabilities``.

    Returns:
        The mean of ``balanced_brier`` over the tags that have a positive and a negative; NaN
        when no tag has both.

    Raises:
        ValueError: As ``balanced_brier``.
    """
    return float(balanced_brier(probabilities, labels).mean())


def _check_labels(labels: pd.DataFrame) -> None:
    _check_columns(labels, "labels")
    if not labels.isin([0, 1]).all().all():
        raise ValueError("labels must be 0 or 1")


def _check_columns(frame: pd.DataFrame, what: str) -> None:
    if list(frame.columns) != list(FOCUS_TAGS):
        raise ValueError(f"the columns of {what} must be the focus tags, in FOCUS_TAGS order")


def _check_pair(probabilities: pd.DataFrame, labels: pd.DataFrame) -> None:
    _check_columns(probabilities, "probabilities")
    _check_labels(labels)
    if not probabilities.index.equals(labels.index):
        raise ValueError("probabilities and labels must have the same index, in the same order")


def _check_weights(weights: pd.DataFrame) -> None:
    if list(weights.index) != list(FOCUS_TAGS) or list(weights.columns) != [0, 1]:
        raise ValueError("weights must be tag_weights: one row per focus tag, columns 0 and 1")
