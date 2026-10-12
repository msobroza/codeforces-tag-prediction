"""Score a duplicate detector against judged pairs, and show judged pairs for reading."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import pandas as pd


def detection_curve(
    similarity: pd.Series, is_duplicate: pd.Series, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Score a similarity threshold against judged pairs.

    Args:
        similarity: Similarity of each judged pair.
        is_duplicate: True for the pairs judged to be duplicates, aligned on ``similarity``.
        thresholds: Thresholds to evaluate; a pair is grouped when its similarity reaches one.

    Returns:
        One row per threshold: duplicates kept together, duplicates missed, non-duplicates
        grouped anyway, recall and precision. Recall only counts the duplicates among the
        judged pairs, so it is an upper bound if others hide among unjudged pairs.
    """
    rows = []
    for threshold in thresholds:
        grouped = similarity >= threshold
        kept = int((grouped & is_duplicate).sum())
        missed = int((~grouped & is_duplicate).sum())
        false_groupings = int((grouped & ~is_duplicate).sum())
        rows.append(
            {
                "threshold": threshold,
                "duplicates kept together": kept,
                "duplicates missed": missed,
                "non-duplicates grouped": false_groupings,
                "recall": kept / max(kept + missed, 1),
                "precision": kept / max(kept + false_groupings, 1),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def two_stage_errors(
    similarity: pd.Series,
    step2_duplicate: pd.Series,
    is_duplicate: pd.Series,
    stage1_threshold: float,
    one_stage_threshold: float,
) -> pd.DataFrame:
    """Count the first step's errors and how many a second, filtering step catches.

    The two-stage detector keeps a pair when its similarity reaches ``stage1_threshold`` (a low
    threshold, for recall) and the second step accepts it (e.g. an LLM judge, for precision).
    It is compared with a one-stage detector that only applies ``one_stage_threshold``.

    Args:
        similarity: First-step similarity of each judged pair.
        step2_duplicate: Second-step decision per pair (True = keep as a duplicate).
        is_duplicate: Reference label per pair.
        stage1_threshold: First-step threshold of the two-stage detector.
        one_stage_threshold: Threshold of the one-stage detector it replaces.

    Returns:
        One row per kind of first-step error, with how many errors there are, how many the
        two-stage detector fixes and how many are left; plus a row for the second step's own
        mistakes on the candidates it sees.
    """
    candidate = similarity >= stage1_threshold
    kept = candidate & step2_duplicate
    one_stage = similarity >= one_stage_threshold
    # Each first-step error: (which pairs are wrong, which of them the two-stage output gets right)
    errors = {
        f"false positives of one stage (>= {one_stage_threshold})": (
            one_stage & ~is_duplicate,
            ~kept,
        ),
        f"false negatives of one stage (>= {one_stage_threshold})": (
            ~one_stage & is_duplicate,
            kept,
        ),
        f"false positives of stage 1 (>= {stage1_threshold})": (candidate & ~is_duplicate, ~kept),
        f"false negatives of stage 1 (>= {stage1_threshold})": (~candidate & is_duplicate, kept),
    }
    rows = {name: _error_row(wrong, fixed) for name, (wrong, fixed) in errors.items()}
    # Nothing comes after the second step, so none of its own mistakes get fixed.
    stage2_mistakes = candidate & (step2_duplicate != is_duplicate)
    rows["stage 2 mistakes on stage-1 candidates"] = _error_row(stage2_mistakes, fixed=False)
    return pd.DataFrame.from_dict(rows, orient="index")


def describe_pair(pair: pd.Series, texts: pd.Series, chars: int = 400, width: int = 100) -> str:
    """Show a judged pair for reading: both judges' calls, then the start of each statement.

    Each call shows the judge's label, confidence and reason.

    Args:
        pair: One row of the audit's ``judgments.csv``.
        texts: Statement per ``sample_id``.
        chars: Characters kept from each statement, after removing the ``$$$`` math
            delimiters and collapsing whitespace.
        width: Line width.

    Returns:
        Plain text, one block per pair.
    """
    a, b = int(pair["sample_id_a"]), int(pair["sample_id_b"])
    lines = [f"{a} / {b}: word cosine {pair['word_cosine']:.3f}, verdict {pair['final']}"]
    for judge in ("judge1", "judge2"):
        verdict = f"{pair[f'{judge}_label']}, {pair[f'{judge}_confidence']} confidence"
        call = f"{judge[:5]} {judge[5]} ({verdict}): {pair[f'{judge}_reason']}"
        lines.append(textwrap.fill(call, width, initial_indent="  ", subsequent_indent="    "))
    for sample_id in (a, b):
        text = " ".join(texts[sample_id].replace("$$$", "").split())
        excerpt = text[:chars] + ("…" if len(text) > chars else "")
        block = f"[{sample_id}] {excerpt}"
        lines.append(textwrap.fill(block, width, initial_indent="  ", subsequent_indent="    "))
    return "\n".join(lines)


def _error_row(wrong: pd.Series, fixed: pd.Series | bool) -> dict[str, int]:
    """Count errors, how many are fixed and how many are left."""
    count, caught = int(wrong.sum()), int((wrong & fixed).sum())
    return {"errors": count, "fixed by the two-stage detector": caught, "left": count - caught}


def zero_failure_upper_bound(n_checked: int, confidence: float = 0.95) -> float:
    """Bound the rate of an event that never occurred in a random sample.

    If none of ``n_checked`` randomly sampled items shows the event, its true rate is below
    the returned value with the given confidence (exact binomial bound; about 3 / n at 95%,
    the "rule of three").

    Args:
        n_checked: Number of items checked, all negative.
        confidence: Confidence level of the bound.

    Returns:
        The upper bound on the rate, between 0 and 1.
    """
    return 1 - (1 - confidence) ** (1 / n_checked)
