"""Group bootstrap: resample near-duplicate groups, so both versions of a problem come together.

Drawing problems one by one would treat the easy and hard versions of a problem as two
independent problems, and give intervals that are too narrow. A resample is a row of weights,
how many times each problem was drawn, so any count over problems becomes a weighted sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_weights(groups: pd.Series, resamples: int, seed: int) -> np.ndarray:
    """Draw near-duplicate groups with replacement, and count how often each row is drawn.

    Each resample draws as many groups as there are distinct groups, uniformly with
    replacement, so the number of draws of each group follows a multinomial distribution. A
    row's weight is the number of draws of its group: the rows of a group are always drawn
    together.

    Args:
        groups: Near-duplicate group per row.
        resamples: Number of resamples.
        seed: Seed of ``np.random.default_rng``.

    Returns:
        An int64 array of shape ``(resamples, len(groups))``; with no row, each resample draws
        nothing.

    Raises:
        ValueError: If a row has no group.
    """
    if groups.isna().any():
        raise ValueError("Every row needs a group: groups holds missing values.")
    if groups.empty:
        return np.zeros((resamples, 0), dtype=np.int64)
    codes, uniques = pd.factorize(groups)
    n_groups = len(uniques)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=resamples)
    return draws[:, codes]


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the percentile interval of bootstrap values.

    Args:
        values: One row per resample, e.g. one column per metric.
        confidence: Share of the resamples the interval covers, e.g. 0.95.

    Returns:
        The lower and upper bounds: the ``(1 - confidence) / 2`` and ``(1 + confidence) / 2``
        quantiles along the first axis.
    """
    low, high = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
    return low, high
