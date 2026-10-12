"""Find the problems a tag model gets wrong, to read them in notebook 04.

The notebook reads out-of-fold predictions only: looking at the test set's errors would turn
it into a second dev set.
"""

from __future__ import annotations

import pandas as pd

from codeforces_tags.config import ERROR_EXCERPT_CHARS

# Each kind of error on a tag: the (predicted, true) values of the tag.
ERROR_KINDS = {"false positive": (1, 0), "false negative": (0, 1)}


def misclassified(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    problems: pd.DataFrame,
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    tag: str,
    kind: str,
    limit: int = 5,
) -> pd.DataFrame:
    """List the problems that a tag model gets wrong on one tag, in one way.

    Args:
        problems: Preprocessed problems, with ``tags`` and ``statement``.
        truth: 0/1 focus-tag matrix, one row per problem.
        predicted: 0/1 predictions, with the index and columns of ``truth``.
        tag: The focus tag to inspect.
        kind: ``"false positive"`` (tag predicted, not true) or ``"false negative"`` (tag
            true, not predicted).
        limit: Most problems returned.

    Returns:
        At most ``limit`` problems, in index order, with columns ``tags`` (all their tags) and
        ``statement`` (its first ``ERROR_EXCERPT_CHARS`` characters).

    Raises:
        ValueError: If ``kind`` is not a key of ``ERROR_KINDS``.
    """
    if kind not in ERROR_KINDS:
        known = ", ".join(map(repr, ERROR_KINDS))
        raise ValueError(f"kind must be one of {known}, not {kind!r}")
    predicted_value, true_value = ERROR_KINDS[kind]
    wrong = (predicted[tag] == predicted_value) & (truth[tag] == true_value)
    chosen = problems.loc[wrong.index[wrong]].sort_index().head(limit)
    return pd.DataFrame(
        {"tags": chosen["tags"], "statement": chosen["statement"].str[:ERROR_EXCERPT_CHARS]}
    )
