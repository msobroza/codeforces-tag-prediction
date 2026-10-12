"""Find candidate duplicate pairs, sort them into judged tiers, and write what the judges read.

The audit's candidates are the pairs flagged by any of six signals, which are wider than the
detector's stage-1 candidates (TF-IDF similarity at or above the threshold, here tier A): the
audit also looks for duplicates that the detector would miss.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from codeforces_tags.config import (
    AUDIT_CHAR_NGRAMS,
    AUDIT_CHAR_THRESHOLD,
    AUDIT_CODE_THRESHOLD,
    AUDIT_CODE_TOKENS,
    AUDIT_CONTROL_SAMPLE,
    AUDIT_SAME_ROUND_WORD_THRESHOLD,
    AUDIT_SEEDS,
    AUDIT_SIMILARITY_DECIMALS,
    AUDIT_TEXT_LIMITS,
    AUDIT_VERSION_PATTERN,
    AUDIT_WEAK_SIGNAL_SAMPLE,
    NEAR_DUPLICATE_CANDIDATE_THRESHOLD,
    NEAR_DUPLICATE_FIELD,
)
from codeforces_tags.near_duplicates.detection import similar_pairs, statement_vectors

# Signals, in the column order of the pair tables: similarities, then boolean flags.
SIMILARITIES = ("word", "char", "code")
FLAGS = ("same_contest", "same_sample_in", "same_sample_out", "both_version")
# Tier labels, as written in the judged pairs ("A: word>=0.4", ...).
_CUT = NEAR_DUPLICATE_CANDIDATE_THRESHOLD
TIERS = (
    f"A: word>={_CUT}",
    f"B: word<{_CUT}, strong signal",
    f"C: word<{_CUT}, weak signal only (sample)",
    "D: same contest, no signal (random control)",
)


def cosine_matrix(texts: pd.Series, **vectorizer_args: Any) -> np.ndarray:
    """Compute the pairwise cosine similarity of TF-IDF vectors.

    Args:
        texts: One document per row.
        **vectorizer_args: Passed to ``TfidfVectorizer``, which always uses sublinear tf.

    Returns:
        A dense square matrix with a zero diagonal.
    """
    vectors = TfidfVectorizer(sublinear_tf=True, **vectorizer_args).fit_transform(texts)
    return _dense_cosine(vectors)


def same_examples(examples: pd.Series) -> np.ndarray:
    """Flag pairs of problems with identical, non-empty sample I/O.

    Args:
        examples: Sample inputs or outputs, one list per problem (missing when absent).

    Returns:
        A square boolean matrix.
    """
    text = examples.map(lambda x: json.dumps(x) if isinstance(x, list) else "").to_numpy()
    present = (text != "[]") & (text != "")
    return (text[:, None] == text[None, :]) & present[:, None]


def signal_matrices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute every duplicate signal for every pair of problems.

    Args:
        df: The preprocessed dataset.

    Returns:
        One square matrix per signal, rows and columns in the order of ``df``: the cosine
        similarities ``word`` (the detector's), ``char`` and ``code``, then the boolean flags
        ``same_contest``, ``same_sample_in``, ``same_sample_out`` and ``both_version``.
    """
    description = df[NEAR_DUPLICATE_FIELD]
    timestamps = df["prob_desc_created_at"].to_numpy()
    version = description.str.contains(AUDIT_VERSION_PATTERN, case=False).to_numpy()
    return {
        "word": _dense_cosine(statement_vectors(description)),
        "char": cosine_matrix(description, analyzer="char_wb", ngram_range=AUDIT_CHAR_NGRAMS),
        "code": cosine_matrix(df["source_code"], token_pattern=AUDIT_CODE_TOKENS),
        "same_contest": timestamps[:, None] == timestamps[None, :],
        "same_sample_in": same_examples(df["prob_desc_sample_inputs"]),
        "same_sample_out": same_examples(df["prob_desc_sample_outputs"]),
        "both_version": version[:, None] & version[None, :],
    }


def pair_rows(
    ids: np.ndarray, signals: dict[str, np.ndarray], positions: Iterable[tuple[int, int]]
) -> pd.DataFrame:
    """Describe pairs of problems by their signals.

    Args:
        ids: Sample id of each row of the signal matrices.
        signals: Matrices from ``signal_matrices``.
        positions: ``(i, j)`` row positions of the pairs.

    Returns:
        One row per pair: sample ids ``a`` and ``b``, then each signal (similarities rounded).
    """
    rows = []
    for i, j in positions:
        row = {"a": int(ids[i]), "b": int(ids[j])}
        row |= {
            name: round(float(signals[name][i, j]), AUDIT_SIMILARITY_DECIMALS)
            for name in SIMILARITIES
        }
        row |= {name: bool(signals[name][i, j]) for name in FLAGS}
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_pairs(ids: np.ndarray, signals: dict[str, np.ndarray]) -> pd.DataFrame:
    """List the pairs flagged by at least one duplicate signal.

    Args:
        ids: Sample id of each row of the signal matrices.
        signals: Matrices from ``signal_matrices``.

    Returns:
        One row per flagged pair, as built by ``pair_rows``.
    """
    word, same_contest = signals["word"], signals["same_contest"]
    flagged = (
        (word >= NEAR_DUPLICATE_CANDIDATE_THRESHOLD)
        | (signals["char"] >= AUDIT_CHAR_THRESHOLD)
        | signals["same_sample_in"]
        | (same_contest & (word >= AUDIT_SAME_ROUND_WORD_THRESHOLD))
        | (same_contest & signals["both_version"])
        | (signals["code"] >= AUDIT_CODE_THRESHOLD)
    )
    return pair_rows(ids, signals, zip(*np.nonzero(np.triu(flagged, 1)), strict=True))


def same_contest_control(
    df: pd.DataFrame, candidates: pd.DataFrame, size: int, seed: int
) -> list[tuple[int, int]]:
    """Draw random pairs from the same round among the pairs no signal flagged.

    Args:
        df: The preprocessed dataset.
        candidates: Flagged pairs, from ``candidate_pairs``.
        size: Number of pairs to draw.
        seed: Seed of the numpy generator.

    Returns:
        ``(i, j)`` row positions of the drawn pairs.
    """
    timestamps = df["prob_desc_created_at"].to_numpy()
    ids = df.index.to_numpy()
    flagged = set(zip(candidates.a, candidates.b, strict=True))
    same = [
        (i, j)
        for t in np.unique(timestamps)
        for idx in [np.nonzero(timestamps == t)[0]]
        for x, i in enumerate(idx)
        for j in idx[x + 1 :]
    ]
    unflagged = [(i, j) for i, j in same if (ids[i], ids[j]) not in flagged]
    picks = np.random.default_rng(seed).choice(len(unflagged), size, replace=False)
    return [unflagged[k] for k in picks]


def judging_tiers(
    df: pd.DataFrame, candidates: pd.DataFrame, signals: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Pick the judged pairs: tiers A and B in full, samples of tier C and of the control D.

    Args:
        df: The preprocessed dataset.
        candidates: Flagged pairs, from ``candidate_pairs``.
        signals: Matrices from ``signal_matrices``, used to describe the control pairs.

    Returns:
        The judged pairs in random order, with a ``tier`` and a ``pair_id``.
    """
    strong = (
        candidates.both_version
        | candidates.same_contest
        | (candidates.char >= AUDIT_CHAR_THRESHOLD)
    )
    below = candidates.word < NEAR_DUPLICATE_CANDIDATE_THRESHOLD
    weak = candidates[below & ~strong]
    control = same_contest_control(
        df, candidates, AUDIT_CONTROL_SAMPLE, AUDIT_SEEDS["control_sample"]
    )
    pairs = pd.concat(
        [
            candidates[~below].assign(tier=TIERS[0]),
            candidates[below & strong].assign(tier=TIERS[1]),
            weak.sample(
                min(AUDIT_WEAK_SIGNAL_SAMPLE, len(weak)),
                random_state=AUDIT_SEEDS["weak_signal_sample"],
            ).assign(tier=TIERS[2]),
            pair_rows(df.index.to_numpy(), signals, control).assign(tier=TIERS[3]),
        ],
        ignore_index=True,
    )
    return _number_in_random_order(pairs)


def stage_1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """List the detector's candidate pairs: the only pairs judged when the audit scales.

    The pairs come from the detector's blockwise similarity, so memory grows with a block of
    rows, not with the square of the dataset as in ``signal_matrices``.

    Args:
        df: The preprocessed dataset.

    Returns:
        The pairs whose TF-IDF similarity reaches the candidate threshold, in random order,
        with sample ids ``a`` and ``b``, the similarity ``word``, a ``tier`` and a ``pair_id``.
    """
    candidates = similar_pairs(df[NEAR_DUPLICATE_FIELD], NEAR_DUPLICATE_CANDIDATE_THRESHOLD)
    pairs = pd.DataFrame(
        {
            "a": candidates["first"].astype(int),
            "b": candidates["second"].astype(int),
            "word": candidates["similarity"].round(AUDIT_SIMILARITY_DECIMALS),
            "tier": TIERS[0],
        }
    )
    return _number_in_random_order(pairs)


def pair_items(df: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """Build the text a judge reads for each pair, without ids, scores or tiers.

    Args:
        df: The preprocessed dataset.
        pairs: Judged pairs, from ``judging_tiers``.

    Returns:
        One item per pair, in the order of ``pairs``: its ``pair_id`` and the truncated
        description, input spec and output spec of both sides ``A`` and ``B``.
    """

    def side(sample_id: int) -> dict:
        return {
            field: value[:limit] if isinstance(value, str) else ""
            for field, limit in AUDIT_TEXT_LIMITS.items()
            for value in [df.loc[sample_id, f"prob_desc_{field}"]]
        }

    return [
        {"pair_id": int(row.pair_id), "A": side(row.a), "B": side(row.b)}
        for row in pairs.itertuples()
    ]


def write_batches(items: list[dict], output: Path, name: str, n_batches: int) -> list[Path]:
    """Split items into consecutive batches and write each one as JSON.

    Args:
        items: Pair items, from ``pair_items``.
        output: Folder to write into.
        name: File name prefix; batch ``k`` goes to ``<name>_<k>.json``.
        n_batches: Number of batches.

    Returns:
        The paths written, in batch order.
    """
    paths = []
    for k, positions in enumerate(np.array_split(np.arange(len(items)), n_batches)):
        path = output / f"{name}_{k}.json"
        batch = [items[position] for position in positions]
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(path)
    return paths


def _number_in_random_order(pairs: pd.DataFrame) -> pd.DataFrame:
    """Shuffle pairs, so that batch order says nothing about similarity, then number them."""
    pairs = pairs.sample(frac=1, random_state=AUDIT_SEEDS["shuffle"]).reset_index(drop=True)
    pairs["pair_id"] = np.arange(len(pairs))
    return pairs


def _dense_cosine(vectors: spmatrix) -> np.ndarray:
    """Turn L2-normalised row vectors into a dense cosine matrix with a zero diagonal."""
    similarity = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarity, 0)
    return similarity
