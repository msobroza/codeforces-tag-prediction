"""The tables of the zero-shot LLM methods' notebook, built from frozen files.

Each function takes what the notebook read from the committed files of its methods (run records,
0/1 predictions, ``per_problem.csv`` rows) and returns one table. They import neither mlflow nor
a provider's SDK. Like ``comparison``, they are generic over methods: any prompted method's
record and calls give the same tables. Comparisons and examples read dev only (ADR 0001).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from codeforces_tags.config import ALL_TAGS, PAPER_TAGS
from codeforces_tags.evaluation.report import all_tag_macro_f1
from codeforces_tags.llm.client import USAGE_KEYS
from codeforces_tags.provenance import RunRecord

# The columns of answer_health: how many answers, how they went, then the tokens they used and
# what they cost.
ANSWER_HEALTH_COLUMNS = (
    "answers",
    "completed",
    "failed",
    "unknown_pieces",
    "empty_answers",
    "served_models",
    *USAGE_KEYS,
    "cost_usd",
)
# The row of tags_per_answer that counts the true labels.
TRUE_LABELS = "true labels"
# The columns of tags_per_answer.
TAGS_PER_ANSWER_COLUMNS = ("mean", "median", "max", "share_empty")
# The columns of all_tag_scores: the headline, the tags it averages, then its two variants.
ALL_TAG_COLUMNS = (
    "all_tag_macro_f1",
    "present_tags",
    "all_tag_macro_f1_zero_absent",
    "paper_tag_macro_f1",
)
# The columns of cost_per_thousand: three totals scaled to 1,000 problems, then a median.
_TOTALS = ("input_tokens", "output_tokens", "total_cost_usd")
COST_COLUMNS = (*_TOTALS, "median_latency_s")
_THOUSAND = 1000


def answer_health(records: Mapping[str, RunRecord]) -> pd.DataFrame:
    """Tabulate how each prompted method's answers went: their statuses, tokens and cost.

    The numbers are the summaries that ``llm score`` froze in each run record, from the answers
    files (``llm.answers.answer_summary``).

    Args:
        records: Each prompted method's run record, by method name, in the order to report.

    Returns:
        One row per method and part (index ``method``, ``part``), with the columns of
        ``ANSWER_HEALTH_COLUMNS``: ``answers``; ``completed`` (those with the status
        ``completed``); ``failed`` (those that score as the empty set); ``unknown_pieces``;
        ``empty_answers`` (completed ones naming no tag); ``served_models`` (a dict: the answers
        each model served); the tokens of ``USAGE_KEYS``; and ``cost_usd``, the total cost.

    Raises:
        ValueError: If a record has no answers, as that of a method that sends no request.
    """
    methods, parts, rows = [], [], []
    for method, record in records.items():
        summaries = record.method_entries.get("answers")
        if summaries is None:
            raise ValueError(
                f"The run record of {method} has no answers: it is not a prompted method's."
            )
        for part, summary in summaries.items():
            methods.append(method)
            parts.append(part)
            rows.append(_health_row(summary))
    index = pd.MultiIndex.from_arrays([methods, parts], names=["method", "part"])
    return pd.DataFrame(rows, index=index, columns=list(ANSWER_HEALTH_COLUMNS))


def tags_per_answer(predictions: Mapping[str, pd.DataFrame], truth: pd.DataFrame) -> pd.DataFrame:
    """Count the tags per problem that each method names, beside the true labels.

    A method that names far more or fewer tags than the problems have is over- or under-tagging,
    which its all-tag score then reflects.

    Args:
        predictions: Each method's 0/1 predicted tags of one part, by method name, in the order
            to report, with the problems of ``truth``.
        truth: 0/1 true tags of the same problems.

    Returns:
        One row per method, then the row ``true labels`` (index ``method``), with the columns of
        ``TAGS_PER_ANSWER_COLUMNS``: the ``mean``, ``median`` and ``max`` number of tags per
        problem, and ``share_empty``, the share of problems with none.

    Raises:
        ValueError: If a method's predictions are not on ``truth``'s problems, in its order.
    """
    _check_same_problems(truth, predictions)
    counts = {method: predicted.sum(axis=1) for method, predicted in predictions.items()}
    counts[TRUE_LABELS] = truth.sum(axis=1)
    rows = [
        {
            "mean": tags.mean(),
            "median": tags.median(),
            "max": tags.max(),
            "share_empty": (tags == 0).mean(),
        }
        for tags in counts.values()
    ]
    return pd.DataFrame(
        rows, index=pd.Index(list(counts), name="method"), columns=list(TAGS_PER_ANSWER_COLUMNS)
    )


def all_tag_scores(truth: pd.DataFrame, predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Score each method as the xCodeEval paper does, over all the tags, not the focus ones.

    The score is a fraction of 1: the paper's are percents, so notebooks print 100 times it
    beside ``settings.PAPER_ALL_TAG_MACRO_F1``.

    Args:
        truth: 0/1 true tags with a column for every tag of ``ALL_TAGS``, one row per problem.
        predictions: Each method's 0/1 predicted tags, with the index and columns of ``truth``,
            by method name, in the order to report.

    Returns:
        One row per method (index ``method``), with the columns of ``ALL_TAG_COLUMNS``:
        ``all_tag_macro_f1``, the macro F1 over the tags with a true positive in these problems
        (the headline); ``present_tags``, how many those are; ``all_tag_macro_f1_zero_absent``,
        the same over all 37 tags, an absent tag's F1 counted as 0; and ``paper_tag_macro_f1``,
        the headline over the paper's 31 tags.

    Raises:
        ValueError: As ``evaluation.report.all_tag_macro_f1``: a method's predictions or
            ``truth`` lack a tag, or are not 0/1 on the same problems.
    """
    rows = []
    for predicted in predictions.values():
        headline, present = all_tag_macro_f1(truth, predicted, ALL_TAGS)
        zero_absent, _ = all_tag_macro_f1(truth, predicted, ALL_TAGS, absent="zero")
        paper, _ = all_tag_macro_f1(truth, predicted, PAPER_TAGS)
        rows.append(
            {
                "all_tag_macro_f1": headline,
                "present_tags": present,
                "all_tag_macro_f1_zero_absent": zero_absent,
                "paper_tag_macro_f1": paper,
            }
        )
    return pd.DataFrame(
        rows, index=pd.Index(list(predictions), name="method"), columns=list(ALL_TAG_COLUMNS)
    )


def cost_per_thousand(per_problem: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Scale each method's recorded calls to 1,000 problems: their tokens, cost and time.

    Args:
        per_problem: Each method's ``per_problem.csv`` (``load_per_problem``), indexed by part
            and ``sample_id``, by method name, in the order to report.

    Returns:
        One row per method and part (index ``method``, ``part``), with the columns of
        ``COST_COLUMNS``: the ``input_tokens``, ``output_tokens`` and ``total_cost_usd`` of
        1,000 problems (the mean over the part's problems, failed answers included, times
        1,000), and the ``median_latency_s`` of one problem's call.
    """
    methods, parts, rows = [], [], []
    for method, calls in per_problem.items():
        for part, part_calls in calls.groupby(level="part"):
            methods.append(method)
            parts.append(part)
            rows.append(_cost_row(part_calls))
    index = pd.MultiIndex.from_arrays([methods, parts], names=["method", "part"])
    return pd.DataFrame(rows, index=index, columns=list(COST_COLUMNS))


def disagreements(
    truth: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, tag: str, *, limit: int = 3
) -> pd.DataFrame:
    """List problems where exactly one of two methods is right about a tag, to read them.

    A method is right about a problem when its 0/1 value equals the true one, as in
    ``comparison.agreement``, which counts them. Read on dev (ADR 0001).

    Args:
        truth: 0/1 true tags, one row per problem.
        a: One method's 0/1 predicted tags, with the problems of ``truth``.
        b: Another method's, on the same problems.
        tag: The tag to read: a column of all three.
        limit: Most problems returned.

    Returns:
        At most ``limit`` problems, the first in the order of ``truth``, indexed like it, with
        the columns ``truth``, ``a`` and ``b``: their values of the tag.

    Raises:
        ValueError: If ``a`` or ``b`` is not on ``truth``'s problems, in its order.
    """
    _check_same_problems(truth, {"a": a, "b": b})
    only_one_right = (a[tag] == truth[tag]) != (b[tag] == truth[tag])
    values = pd.DataFrame({"truth": truth[tag], "a": a[tag], "b": b[tag]})
    return values[only_one_right].head(limit)


def _health_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a part's row of answer_health from an ``answer_summary`` dict."""
    return {
        "answers": summary["answers"],
        "completed": summary["statuses"].get("completed", 0),
        "failed": summary["failed"],
        "unknown_pieces": summary["unknown_pieces"],
        "empty_answers": summary["empty_answers"],
        "served_models": dict(summary["served_models"]),
        **{key: summary["usage"][key] for key in USAGE_KEYS},
        "cost_usd": summary["cost_usd"]["total_cost"],
    }


def _cost_row(calls: pd.DataFrame) -> dict[str, float]:
    """Return a part's row of cost_per_thousand from its recorded calls."""
    per_thousand = calls[list(_TOTALS)].mean() * _THOUSAND
    return {**per_thousand.to_dict(), "median_latency_s": float(calls["latency_s"].median())}


def _check_same_problems(truth: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> None:
    """Refuse a frame whose problems are not truth's, in truth's order."""
    for name, frame in frames.items():
        if not frame.index.equals(truth.index):
            raise ValueError(f"{name} must be on truth's problems, in the same order.")
