"""Encode multi-label targets as indicator matrices."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS


def multi_hot(labels: pd.Series, vocabulary: Sequence[str] | None = None) -> pd.DataFrame:
    """Build a 0/1 indicator matrix from label collections.

    Args:
        labels: One collection (list or set) per row, e.g. tags or imported modules.
        vocabulary: Columns to build, in order. Labels outside it are ignored. Defaults to
            every observed label, sorted alphabetically.

    Returns:
        One column per label and one row per entry of ``labels``, with the same index.
    """
    if vocabulary is None:
        vocabulary = sorted({label for row_labels in labels for label in row_labels})
    rows = [[int(label in set(row_labels)) for label in vocabulary] for row_labels in labels]
    return pd.DataFrame(rows, index=labels.index, columns=pd.Index(list(vocabulary), name="label"))


def focus_labels(problems: pd.DataFrame) -> pd.DataFrame:
    """Build the 0/1 matrix of the focus tags, the labels the model predicts.

    Args:
        problems: Problems with ``tags``, raw (``read_samples()``) or preprocessed: the labels
            are the same, since preprocessing only drops repeated tags.

    Returns:
        One column per focus tag, in ``FOCUS_TAGS`` order, with the index of ``problems``.
    """
    return multi_hot(problems["tags"], FOCUS_TAGS)


def frequent_pairs(labels: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Mark the rows that carry both labels of each frequent pair of labels.

    Args:
        labels: 0/1 indicator columns, e.g. from ``multi_hot``.
        min_count: Rows a pair must share to count as frequent.

    Returns:
        One 0/1 column per frequent pair, named ``"a & b"`` in the order of the columns of
        ``labels``, with the index of ``labels``.
    """
    pairs = {
        f"{first} & {second}": labels[first] & labels[second]
        for first, second in combinations(labels.columns, 2)
    }
    frame = pd.DataFrame(pairs, index=labels.index, dtype=int)
    return frame.loc[:, frame.sum() >= min_count]
