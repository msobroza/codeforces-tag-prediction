"""Choose the pairs each judging round sees, and merge the judges' labels into verdicts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codeforces_tags.config import AUDIT_FORMER_THRESHOLD

# Labels that make a pair a duplicate: the same task, or two versions of one task.
DUPLICATE_LABELS = ("same_problem", "versions")
# Columns of judgments.csv, in order.
PAIR_COLUMNS = {
    "pair_id": "pair_id",
    "a": "sample_id_a",
    "b": "sample_id_b",
    "tier": "tier",
    "word": "word_cosine",
    "char": "char_cosine",
    "code": "code_cosine",
    "same_contest": "same_contest",
    "same_sample_in": "same_sample_in",
    "same_sample_out": "same_sample_out",
    "both_version": "both_version",
}
CONFIRMED_COLUMNS = [
    "sample_id_a",
    "sample_id_b",
    "word_cosine",
    "char_cosine",
    "same_contest",
    "judge1_label",
    "judge2_label",
    "judge1_reason",
    "judge2_reason",
]


def labels_frame(labels: list[dict]) -> pd.DataFrame:
    """Turn a judge's answers into a table.

    Args:
        labels: Answers with ``pair_id``, ``label``, ``confidence`` and ``reason``.

    Returns:
        Columns ``label``, ``confidence`` and ``reason``, indexed by ``pair_id`` in input order.
    """
    columns = ["pair_id", "label", "confidence", "reason"]
    return pd.DataFrame(labels, columns=columns).set_index("pair_id")


def says_duplicate(labels: pd.Series) -> pd.Series:
    """Tell which labels call a pair a duplicate.

    Args:
        labels: One judge's labels; missing labels count as not duplicate.

    Returns:
        A boolean per label.
    """
    return labels.isin(DUPLICATE_LABELS)


def round_1_pairs(
    first: pd.DataFrame, word: pd.Series, threshold: float = AUDIT_FORMER_THRESHOLD
) -> list[int]:
    """Pick the pairs the second judge checks first: the calls that decide the former errors.

    A first-judge call is re-checked when it disagrees with the former one-stage detector (a
    duplicate below ``threshold``, or a non-duplicate at or above it), or when its confidence
    is low.

    Args:
        first: First-judge labels, from ``labels_frame``.
        word: Word cosine similarity per ``pair_id``.
        threshold: Threshold of the former detector.

    Returns:
        The pair ids, in the order of ``first``.
    """
    above = word.reindex(first.index) >= threshold
    chosen = (says_duplicate(first["label"]) != above) | (first["confidence"] == "low")
    return first.index[chosen].tolist()


def round_2_pairs(pair_ids: list[int], round_1: list[int]) -> list[int]:
    """Pick every pair the second judge has not seen yet.

    Args:
        pair_ids: All judged pair ids.
        round_1: Pair ids of round 1.

    Returns:
        The other pair ids, in increasing order.
    """
    return sorted(set(pair_ids) - set(round_1))


def merge_judgments(pairs: pd.DataFrame, raw: dict[str, list]) -> pd.DataFrame:
    """Merge both judges' labels into one verdict per pair.

    A pair is a duplicate when at least one judge says so: a wrong link only keeps two
    problems on the same side of a split, while a wrong cut leaks a near copy into training.

    Args:
        pairs: Judged pairs, from ``candidates.judging_tiers`` or ``candidates.stage_1_pairs``
            (which has the word similarity alone among the signals).
        raw: The judges' answers: ``first`` (first pass), ``second`` (second pass, round 1)
            and ``second_round_2``.

    Returns:
        The table saved as ``judgments.csv``: the pair and its signals, each judge's label,
        confidence and reason, and the ``final`` verdict, most similar pairs first. The round
        of each second label stays in the raw answers.
    """
    second = pd.concat(
        [labels_frame(raw.get("second", [])), labels_frame(raw.get("second_round_2", []))]
    )
    columns = {name: new for name, new in PAIR_COLUMNS.items() if name in pairs}
    judgments = (
        pairs[list(columns)]
        .rename(columns=columns)
        .join(labels_frame(raw["first"]).add_prefix("judge1_"), on="pair_id")
        .join(second.add_prefix("judge2_"), on="pair_id")
    )
    duplicate = says_duplicate(judgments["judge1_label"]) | says_duplicate(
        judgments["judge2_label"]
    )
    judgments["final"] = np.where(duplicate, "duplicate", "not_duplicate")
    return judgments.sort_values("word_cosine", ascending=False)


def confirmed_pairs(judgments: pd.DataFrame) -> pd.DataFrame:
    """Keep the duplicate pairs, with both judges' labels and reasons.

    Args:
        judgments: Merged verdicts, from ``merge_judgments``.

    Returns:
        The table saved as ``confirmed_pairs.csv``.
    """
    columns = [column for column in CONFIRMED_COLUMNS if column in judgments]
    return judgments.loc[judgments["final"] == "duplicate", columns]
