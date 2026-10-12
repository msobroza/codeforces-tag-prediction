"""Calibration: do a method's probabilities mean what they say?

A method that answers a probability per focus tag (the TypeSafe Jev methods) can be read by its
probabilities as well as by the tags they become. A reliability table groups the problems of a
tag by predicted probability and compares, in each group, the mean probability with the share of
problems that have the tag: a calibrated score has the two equal. Read on dev (ADR 0001), from
the out-of-fold scores of a frozen method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codeforces_tags.config import CALIBRATION_BINS

# The columns of reliability_table: a bin's mean predicted probability, the share of its problems
# that have the tag, and how many problems it holds.
RELIABILITY_COLUMNS = ("mean_probability", "positive_rate", "problems")


def reliability_table(
    scores: pd.DataFrame, truth: pd.DataFrame, bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    """Compare each tag's probabilities with how often the tag is true, in equal-width bins.

    The bins cut the range from 0 to 1 into ``bins`` parts of equal width. Bin ``k`` holds the
    problems whose probability ``p`` has ``k <= p * bins < k + 1``, so a bin includes its lower
    edge, and a probability of 1 is in the last bin. A failed answer scores exactly 0 on every
    tag, so failed answers share bin 0 with the lowest scores.

    Args:
        scores: Probabilities from 0 to 1, one column per tag, one row per problem.
        truth: 0/1 true tags, with the index and columns of ``scores``.
        bins: Number of equal-width bins.

    Returns:
        One row per tag and bin that holds a problem (index ``tag``, ``bin``; the tags in the
        order of ``scores``, the bins from the lowest probabilities to the highest, a bin without
        problem left out): ``mean_probability``, ``positive_rate`` (the share of the bin's
        problems that have the tag) and ``problems``.

    Raises:
        ValueError: If ``scores`` holds no problem or no tag, if ``scores`` and ``truth`` are not
            on the same problems and tags, in the same order, if a score is not a number from 0
            to 1, if ``truth`` is not 0/1, or if ``bins`` is below 1.
    """
    _check_inputs(scores, truth, bins)
    probabilities = scores.to_numpy(float)
    positives = truth.to_numpy(float)
    bin_numbers = np.minimum(np.floor(probabilities * bins), bins - 1).astype(np.int64)
    tables = {}
    for place, tag in enumerate(scores.columns):
        rows = pd.DataFrame(
            {
                "bin": bin_numbers[:, place],
                "probability": probabilities[:, place],
                "positive": positives[:, place],
            }
        )
        tables[tag] = rows.groupby("bin").agg(
            mean_probability=("probability", "mean"),
            positive_rate=("positive", "mean"),
            problems=("probability", "size"),
        )
    return pd.concat(tables, names=["tag", "bin"])


def expected_calibration_error(
    scores: pd.DataFrame, truth: pd.DataFrame, bins: int = CALIBRATION_BINS
) -> pd.Series:
    """Average, per tag, how far the probabilities are from how often the tag is true.

    The gap of a bin is the distance between its share of problems with the tag and its mean
    probability; the error of a tag is the sum of its bins' gaps, each weighted by the share of
    the tag's problems that fall in the bin. A calibrated score has an error of 0.

    Args:
        scores: Probabilities from 0 to 1, one column per tag, one row per problem.
        truth: 0/1 true tags, with the index and columns of ``scores``.
        bins: Number of equal-width bins, as in ``reliability_table``.

    Returns:
        The expected calibration error of each tag, named ``ece`` and indexed by ``tag``, in the
        order of the columns of ``scores``.

    Raises:
        ValueError: As ``reliability_table``.
    """
    table = reliability_table(scores, truth, bins)
    gap = (table["positive_rate"] - table["mean_probability"]).abs()
    weighted = gap * table["problems"] / len(scores)
    return weighted.groupby(level="tag", sort=False).sum().rename("ece")


def _check_inputs(scores: pd.DataFrame, truth: pd.DataFrame, bins: int) -> None:
    """Refuse scores and truth that cannot be binned and compared."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, not {bins}.")
    if scores.empty:
        raise ValueError("scores hold no problem or no tag.")
    if not (scores.index.equals(truth.index) and scores.columns.equals(truth.columns)):
        raise ValueError("scores and truth must have the same problems and tags, in order.")
    probabilities = scores.to_numpy(float)
    if not ((probabilities >= 0) & (probabilities <= 1)).all():  # a NaN fails both comparisons
        raise ValueError("scores must be probabilities from 0 to 1, with none missing.")
    if not truth.isin((0, 1)).all().all():
        raise ValueError("truth must hold only 0 and 1.")
