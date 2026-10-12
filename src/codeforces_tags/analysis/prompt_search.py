"""What a search over Jev's words did, for notebook 10: tables built from its frozen record.

Each function takes what the notebook read from committed files (the search record of
``assets/methods/jev-statement-code-gepa/tuning/``, as ``search_record.read_search_record``
returns it, and the frozen scores and predictions of two methods) and returns one table. They
import neither gepa nor mlflow and send no request. Comparisons and error reading use dev only
(ADR 0001): the held-out fold for the honest comparison, all of dev only when labelled in sample.
"""

from __future__ import annotations

import difflib
from collections.abc import Hashable, Iterable, Mapping, Sequence

import pandas as pd

from codeforces_tags.methods.typesafe_jev.reflection import REFUSAL_REASONS
from codeforces_tags.methods.typesafe_jev.search_record import parents_of
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    JevWords,
    length_cap,
    parse_piece,
    piece_length,
    piece_text,
)

# The columns of piece_changes, one row per piece (index ``piece``, in PIECES order).
PIECE_COLUMNS = ("seed_chars", "chosen_chars", "length_cap", "changed", "changed_words", "diff")
# The columns of search_progress, one row per iteration (index ``iteration``); the costs are
# cumulative, in USD.
PROGRESS_COLUMNS = (
    "best_selection_mean",
    "accepted",
    "jev_cost_usd",
    "luna_cost_usd",
    "total_cost_usd",
)
# The columns of lineage, one row per candidate from the seed to the chosen one (index ``index``).
LINEAGE_COLUMNS = ("iteration", "piece", "selection_mean", "gain")
# The columns of probability_shift, one row per tag (index ``tag``).
SHIFT_COLUMNS = ("base_rate", "mean_before", "mean_after")
# The columns of flipped, one row per problem whose 0/1 changed on the tag (index ``sample_id``).
FLIPPED_COLUMNS = ("gold", "before", "after", "now_right")
# What search_progress and lineage read of the frozen record.
_HISTORY_NEEDS = ("accepted", "jev_cost_usd", "luna_cost_usd")
_CANDIDATE_NEEDS = ("parents", "iteration", "piece", "selection_mean", "passed_check")
# How a boolean column of the record may read back from its CSV file.
_FLAGS = {True: True, False: False, "True": True, "False": False}
# The seed candidate: GEPA's index 0, present before the first iteration.
_SEED = 0


def word_diff(before: str, after: str) -> str:
    """Show how one text became another, word by word, as ``git diff --word-diff=plain`` does.

    Both texts are split on whitespace, so a change of spacing or line breaks alone is no
    change; the result joins the words with single spaces.

    Args:
        before: The old text.
        after: The new text.

    Returns:
        The words of both texts in order: a run of words only ``before`` has as ``[-...-]``, a
        run only ``after`` has as ``{+...+}``, a replaced run as both, side by side.
    """
    old, new = before.split(), after.split()
    chunks = [
        _diff_chunk(operation, old[i1:i2], new[j1:j2])
        for operation, i1, i2, j1, j2 in _opcodes(old, new)
    ]
    return " ".join(chunks)


def piece_changes(seed: JevWords, chosen: JevWords) -> pd.DataFrame:
    """Compare each piece of the seed words with the chosen words.

    A piece's length is the one its length cap bounds (``words.piece_length``): the
    convention's characters, or the characters of a tag's compact JSON criteria.

    Args:
        seed: The words the search started from (``words.SEED_WORDS``).
        chosen: The words it chose (``words.read_words`` of ``tuning/params.json``).

    Returns:
        One row per piece, in ``PIECES`` order (index ``piece``): ``seed_chars``,
        ``chosen_chars``, ``length_cap`` (the most a proposal could hold), ``changed`` (whether
        the piece's text differs), ``changed_words`` (the words removed plus the words added)
        and ``diff`` (``word_diff`` of the two pieces' texts).
    """
    rows = [_piece_row(seed, chosen, piece) for piece in PIECES]
    return pd.DataFrame(rows, index=pd.Index(PIECES, name="piece"), columns=list(PIECE_COLUMNS))


def search_progress(history: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Follow the search iteration by iteration: its best candidate so far and what it spent.

    The best candidate is the one the selection rule would choose at that point: the highest
    mean score on the selection fold among the candidates that passed the dataset-text check,
    scored at or before the iteration. The seed is there from the start.

    Args:
        history: The search record's ``history.csv``, one row per iteration (index
            ``iteration``), with at least ``accepted``, ``jev_cost_usd`` and ``luna_cost_usd``
            (each iteration's own cost, in USD).
        candidates: The search record's ``candidates.csv`` (index ``index``, the seed 0), with
            at least ``iteration``, ``selection_mean`` and ``passed_check``.

    Returns:
        One row per iteration, in the order of ``history`` (index ``iteration``):
        ``best_selection_mean``, ``accepted`` (bool), then the costs so far: ``jev_cost_usd``,
        ``luna_cost_usd`` and ``total_cost_usd``.

    Raises:
        ValueError: If a frame lacks a column it needs, if a flag is not a boolean, or if no
            candidate passed the check.
    """
    _require(history, _HISTORY_NEEDS, "history")
    _require(candidates, _CANDIDATE_NEEDS, "candidates")
    passed = candidates[_flags(candidates["passed_check"], "passed_check")]
    if passed.empty:
        raise ValueError("No candidate passed the dataset-text check: there is no best one.")
    known = [
        (passed["iteration"] <= iteration) | (passed.index == _SEED) for iteration in history.index
    ]
    jev = history["jev_cost_usd"].astype(float).cumsum()
    luna = history["luna_cost_usd"].astype(float).cumsum()
    return pd.DataFrame(
        {
            "best_selection_mean": [passed.loc[mask, "selection_mean"].max() for mask in known],
            "accepted": _flags(history["accepted"], "accepted").to_numpy(),
            "jev_cost_usd": jev.to_numpy(),
            "luna_cost_usd": luna.to_numpy(),
            "total_cost_usd": (jev + luna).to_numpy(),
        },
        index=history.index.rename("iteration"),
    )


def lineage(candidates: pd.DataFrame, index: int) -> pd.DataFrame:
    """Follow one candidate back to the seed, through each candidate's first parent.

    Args:
        candidates: The search record's ``candidates.csv`` (index ``index``), with at least
            ``parents`` (the parents' indices, however ``search_record.parents_of`` reads them;
            empty or NaN for the seed), ``iteration``, ``piece`` and ``selection_mean``.
        index: The candidate to follow, e.g. ``params["chosen"]``.

    Returns:
        One row per candidate from the seed to ``index`` (index ``index``): ``iteration``,
        ``piece`` (the piece it changed), ``selection_mean`` and ``gain`` (its selection mean
        minus its parent's; NaN for the seed).

    Raises:
        ValueError: If a frame lacks a column it needs, or if a candidate on the way is missing
            or is its own ancestor.
    """
    _require(candidates, _CANDIDATE_NEEDS, "candidates")
    chain: list[Hashable] = []
    current: Hashable | None = index
    while current is not None:
        if current not in candidates.index:
            raise ValueError(f"Candidate {current} is not in the search record.")
        if current in chain:
            raise ValueError(f"Candidate {current} is its own ancestor.")
        chain.append(current)
        parents = parents_of(candidates.at[current, "parents"])
        current = parents[0] if parents else None
    rows = candidates.loc[chain[::-1], ["iteration", "piece", "selection_mean"]].copy()
    rows["gain"] = rows["selection_mean"].diff()
    return rows.rename_axis("index")


def refusal_counts(history: pd.DataFrame) -> pd.Series:
    """Count the iterations whose proposal was refused, by reason.

    Args:
        history: The search record's ``history.csv``, with ``refusal`` (a reason of
            ``reflection.REFUSAL_REASONS``, or ``""`` or NaN when the proposal was not
            refused).

    Returns:
        The number of iterations per reason, in ``REFUSAL_REASONS`` order, 0 for a reason that
        never occurred (index ``reason``, name ``iterations``).

    Raises:
        ValueError: If ``history`` has no ``refusal`` column, or holds another reason.
    """
    _require(history, ("refusal",), "history")
    reasons = history["refusal"].dropna()
    reasons = reasons[reasons != ""]
    unknown = sorted(set(reasons) - set(REFUSAL_REASONS))
    if unknown:
        raise ValueError(f"Unknown refusal reasons {unknown}; known: {list(REFUSAL_REASONS)}.")
    counts = reasons.value_counts().reindex(list(REFUSAL_REASONS), fill_value=0)
    return counts.astype("int64").rename_axis("reason").rename("iterations")


def probability_shift(
    before: pd.DataFrame, after: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Compare each tag's mean probability under two sets of words with the tag's base rate.

    Args:
        before: One method's probabilities, one row per problem and one column per tag.
        after: Another method's, with the index and columns of ``before``.
        labels: 0/1 true tags, with the index and columns of ``before``.

    Returns:
        One row per tag of ``labels`` (index ``tag``): ``base_rate`` (the share of problems with
        the tag), ``mean_before`` and ``mean_after``.

    Raises:
        ValueError: If the frames are not aligned, or a probability is outside [0, 1].
    """
    _check_aligned(labels, {"before": before, "after": after})
    for name, frame in {"before": before, "after": after}.items():
        if not ((frame >= 0) & (frame <= 1)).all().all():
            raise ValueError(f"{name} must hold probabilities between 0 and 1.")
    table = pd.DataFrame(
        {"base_rate": labels.mean(), "mean_before": before.mean(), "mean_after": after.mean()}
    )
    return table.rename_axis("tag")


def flipped(
    before: pd.DataFrame, after: pd.DataFrame, labels: pd.DataFrame, tag: str
) -> pd.DataFrame:
    """List the problems whose 0/1 prediction on one tag differs between two methods.

    Args:
        before: One method's 0/1 predicted tags, one row per problem.
        after: Another method's, with the index and columns of ``before``.
        labels: 0/1 true tags, with the index and columns of ``before``.
        tag: A column of ``labels``.

    Returns:
        One row per problem whose prediction on ``tag`` changed, in the order of ``labels``
        (index ``sample_id``): ``gold``, ``before``, ``after`` (each 0 or 1) and ``now_right``
        (whether ``after`` equals ``gold``: a wrong prediction made right, else a right one made
        wrong).

    Raises:
        ValueError: If the frames are not aligned, hold a value other than 0 and 1, or lack
            ``tag``.
    """
    _check_aligned(labels, {"before": before, "after": after})
    for name, frame in {"labels": labels, "before": before, "after": after}.items():
        if not frame.isin((0, 1)).all().all():
            raise ValueError(f"{name} must hold 0/1 tags.")
    if tag not in labels.columns:
        raise ValueError(f"{tag!r} is not a tag of labels: {list(labels.columns)}.")
    changed = before[tag] != after[tag]
    table = pd.DataFrame(
        {
            "gold": labels.loc[changed, tag].astype("int64"),
            "before": before.loc[changed, tag].astype("int64"),
            "after": after.loc[changed, tag].astype("int64"),
        }
    )
    table["now_right"] = table["after"] == table["gold"]
    return table.rename_axis("sample_id")


def _opcodes(old: Sequence[str], new: Sequence[str]) -> list[tuple[str, int, int, int, int]]:
    """Return difflib's operations that turn the words ``old`` into ``new``."""
    return difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()


def _diff_chunk(operation: str, removed: Sequence[str], added: Sequence[str]) -> str:
    """Write one operation of a word diff: kept words as they are, others marked."""
    if operation == "equal":
        return " ".join(removed)
    deleted = f"[-{' '.join(removed)}-]" if removed else ""
    inserted = f"{{+{' '.join(added)}+}}" if added else ""
    return deleted + inserted


def _changed_words(before: str, after: str) -> int:
    """Count the words removed plus the words added from one text to the other."""
    old, new = before.split(), after.split()
    return sum(
        (i2 - i1) + (j2 - j1)
        for operation, i1, i2, j1, j2 in _opcodes(old, new)
        if operation != "equal"
    )


def _piece_row(seed: JevWords, chosen: JevWords, piece: str) -> dict[str, object]:
    """Return the row of piece_changes for one piece."""
    before, after = piece_text(seed, piece), piece_text(chosen, piece)
    return {
        "seed_chars": piece_length(piece, parse_piece(piece, before)),
        "chosen_chars": piece_length(piece, parse_piece(piece, after)),
        "length_cap": length_cap(piece),
        "changed": before != after,
        "changed_words": _changed_words(before, after),
        "diff": word_diff(before, after),
    }


def _flags(column: pd.Series, name: str) -> pd.Series:
    """Read a boolean column of the record, whether CSV gave it bools or their names."""
    flags = column.map(_FLAGS)
    if flags.isna().any():
        raise ValueError(f"{name} must hold True or False only.")
    return flags.astype(bool)


def _require(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    """Refuse a frame that lacks one of the columns a table is built from."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} lacks the columns {missing}.")


def _check_aligned(labels: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> None:
    """Refuse a frame that is not on the problems and tags of labels, in the same order."""
    for name, frame in frames.items():
        if not (frame.index.equals(labels.index) and frame.columns.equals(labels.columns)):
            raise ValueError(f"{name} must have the problems and tags of labels, in order.")
