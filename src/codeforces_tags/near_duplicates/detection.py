"""Find near-duplicate problem statements, so that splits can keep them together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from codeforces_tags.config import (
    NEAR_DUPLICATE_CANDIDATE_THRESHOLD,
    NEAR_DUPLICATE_FIELD,
    NEAR_DUPLICATE_VERDICTS,
)


def statement_vectors(texts: pd.Series) -> spmatrix:
    """Embed statements as L2-normalised TF-IDF vectors, so a dot product is a cosine.

    Args:
        texts: One statement per row.

    Returns:
        A sparse matrix with one row per text (word unigrams and bigrams, sublinear tf).
    """
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)).fit_transform(texts)


def similar_pairs(texts: pd.Series, min_similarity: float, block_size: int = 1000) -> pd.DataFrame:
    """List every pair of texts whose TF-IDF cosine similarity reaches a minimum.

    Args:
        texts: One document per row.
        min_similarity: Minimum cosine similarity, between 0 and 1.
        block_size: Rows compared at a time; lower it to use less memory.

    Returns:
        Columns ``first`` and ``second`` (index labels of ``texts``, each pair once) and
        ``similarity``, most similar first.
    """
    first, second, similarity = _similar_positions(
        statement_vectors(texts), min_similarity, block_size
    )
    labels = texts.index.to_numpy()
    pairs = pd.DataFrame(
        {"first": labels[first], "second": labels[second], "similarity": similarity}
    )
    return pairs.sort_values("similarity", ascending=False, ignore_index=True)


def near_duplicate_groups(
    problems: pd.DataFrame,
    *,
    verdicts: pd.DataFrame | None = None,
    threshold: float = NEAR_DUPLICATE_CANDIDATE_THRESHOLD,
    block_size: int = 1000,
) -> pd.Series:
    """Group problems that are the same task, so that a split keeps each group together.

    Codeforces publishes easy and hard versions of a problem with near-identical statements.
    A random split would put one version in train and the other in test. Detection compares
    the ``NEAR_DUPLICATE_FIELD`` column in two stages:

    1. Candidates: pairs whose TF-IDF cosine similarity reaches ``threshold``, set low so
       that no duplicate is missed. Only candidates can be linked: stage 1 sets the recall.
    2. Verdicts: the judges' verdicts filter the candidates. A candidate judged not a
       duplicate is dropped; one judged a duplicate, or never judged, stays linked. A wrong
       link only keeps two problems on the same side, while a missed duplicate leaks a near
       copy of a test problem into training.

    A group is a chain of links. To group by similarity alone, use
    ``groups_from_pairs(texts.index, similar_pairs(texts, threshold))``.

    Args:
        problems: The dataset, indexed by ``sample_id`` as ``load_dataset()`` returns it (the
            verdicts are keyed by sample id). Group the full dataset, then restrict the groups
            to a subset: on a subset the TF-IDF is refit, so pairs just below ``threshold``
            that nobody judged can become linked (extra grouping, never a leak).
        verdicts: Judged pairs, as returned by ``load_verdicts``. Defaults to the frozen audit.
        threshold: Minimum cosine similarity, between 0 and 1, for a candidate pair.
        block_size: Rows compared at a time; lower it to use less memory.

    Returns:
        A group id per problem, with the same index as ``problems``.

    Raises:
        ValueError: If ``problems`` is not indexed by unique integer sample ids.
    """
    _check_sample_ids(problems.index)
    if verdicts is None:
        verdicts = load_verdicts()
    texts = problems[NEAR_DUPLICATE_FIELD]
    candidates = similar_pairs(texts, threshold, block_size)
    return groups_from_pairs(texts.index, filter_by_verdicts(candidates, verdicts))


def load_verdicts(path: Path = NEAR_DUPLICATE_VERDICTS) -> pd.DataFrame:
    """Read a judge's verdicts on candidate duplicate pairs.

    Args:
        path: CSV with columns ``sample_id_a``, ``sample_id_b`` and ``final``
            (``duplicate`` or ``not_duplicate``), like the audit's ``judgments.csv``.

    Returns:
        Columns ``first`` and ``second`` (sample ids) and ``is_duplicate``.

    Raises:
        ValueError: If a verdict is neither ``duplicate`` nor ``not_duplicate``.
    """
    judged = pd.read_csv(path, usecols=["sample_id_a", "sample_id_b", "final"])
    unknown = set(judged["final"]) - {"duplicate", "not_duplicate"}
    if unknown:
        raise ValueError(f"Unknown verdicts in {path}: {sorted(map(str, unknown))}")
    return pd.DataFrame(
        {
            "first": judged["sample_id_a"],
            "second": judged["sample_id_b"],
            "is_duplicate": judged["final"] == "duplicate",
        }
    )


def filter_by_verdicts(candidates: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Drop the candidate pairs that a judge rejected.

    Args:
        candidates: Columns ``first`` and ``second``, e.g. from ``similar_pairs``.
        verdicts: Columns ``first``, ``second`` and ``is_duplicate``; the order of the two
            items in a pair does not matter. Verdicts on pairs that are not candidates are
            ignored.

    Returns:
        The candidates, without those judged not duplicates.
    """
    rejected = verdicts[~verdicts["is_duplicate"].astype(bool)]
    return candidates[~_pair_keys(candidates).isin(_pair_keys(rejected))]


def groups_from_pairs(index: pd.Index, pairs: pd.DataFrame) -> pd.Series:
    """Turn linked pairs into groups: a group is a chain of links.

    Use it with the pairs of ``similar_pairs`` (similarity alone) or of
    ``filter_by_verdicts`` (candidates filtered by a judge).

    Args:
        index: Labels of all items; items in no pair get a group of their own.
        pairs: Columns ``first`` and ``second`` holding labels from ``index``.

    Returns:
        A group id per label of ``index``.

    Raises:
        ValueError: If ``index`` has repeated labels, which would make links ambiguous.
    """
    if not index.is_unique:
        raise ValueError("Labels must be unique to tell which item a pair links.")
    position = pd.Series(np.arange(len(index)), index=index)
    first = position.loc[pairs["first"]].to_numpy()
    second = position.loc[pairs["second"]].to_numpy()
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(len(index), len(index)))
    _, groups = connected_components(graph, directed=False)
    return pd.Series(groups, index=index, name="group")


def _check_sample_ids(index: pd.Index) -> None:
    """Fail unless ``index`` holds unique integer sample ids, the keys of the verdicts."""
    if index.name != "sample_id" or not pd.api.types.is_integer_dtype(index) or not index.is_unique:
        raise ValueError(
            "Index the problems by unique integer sample ids named 'sample_id', as load_dataset() "
            "does: verdicts are keyed by sample id, so another index would apply them to the "
            f"wrong rows. Got name={index.name!r}, dtype={index.dtype}, unique={index.is_unique}."
        )


def _pair_keys(pairs: pd.DataFrame) -> pd.MultiIndex:
    """Key each pair by its two labels in sorted order, so that (a, b) and (b, a) match."""
    first, second = pairs["first"].to_numpy(), pairs["second"].to_numpy()
    swap = first > second
    return pd.MultiIndex.from_arrays([np.where(swap, second, first), np.where(swap, first, second)])


def _similar_positions(
    vectors: spmatrix, min_similarity: float, block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row positions (i < j) and cosine of every pair at or above ``min_similarity``."""
    firsts, seconds, values = [], [], []
    for start in range(0, vectors.shape[0], block_size):
        similarity = (vectors[start : start + block_size] @ vectors.T).toarray()
        rows, cols = np.nonzero(similarity >= min_similarity - 1e-9)  # tolerate float rounding
        upper = rows + start < cols  # each pair once, no self-pairs
        firsts.append(rows[upper] + start)
        seconds.append(cols[upper])
        values.append(similarity[rows[upper], cols[upper]])
    return np.concatenate(firsts), np.concatenate(seconds), np.concatenate(values)
