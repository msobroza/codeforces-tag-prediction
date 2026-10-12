"""Compare tagging methods on the same problems: their scores, by slice too, and where they agree.

Generic over methods, for any method's notebook. Each method's frozen 0/1 predictions of one
part (``load_predictions(method_dir(name)).loc[part]``, restricted to the focus tags) are
scored against the same truth, with the same group bootstrap resamples, so that their intervals
and paired differences are comparable. Comparisons and error reading use dev only (ADR 0001).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from codeforces_tags.evaluation.report import ALL_PROBLEMS, Evaluation, compare, evaluate

# The columns of method_scores: each metric's F1 and its interval, then, given a baseline, the
# paired difference with it and its interval.
SCORE_COLUMNS = ("f1", "f1_low", "f1_high")
DELTA_COLUMNS = ("delta", "delta_low", "delta_high")
# The columns of slice_differences: the slice's size, then compare's columns on one metric.
SLICE_COLUMNS = ("problems", "baseline", "candidate", *DELTA_COLUMNS)
# The columns of agreement: the problems both methods get right, only one of them, or neither.
AGREEMENT_COLUMNS = ("both_right", "only_a_right", "only_b_right", "both_wrong")
_OVERALL = ("macro_f1", "micro_f1")


def method_scores(
    truth: pd.DataFrame,
    predictions_by_method: Mapping[str, pd.DataFrame],
    groups: pd.Series,
    *,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Score each method's predictions of the same problems, with group bootstrap intervals.

    Every method is scored by ``evaluate`` with the same resamples, and compared with the
    baseline by ``compare``, so the differences are paired.

    Args:
        truth: 0/1 true focus tags, one row per problem.
        predictions_by_method: Each method's 0/1 predicted tags, with the index and columns of
            ``truth``, by method name, in the order to report.
        groups: Near-duplicate group per problem, with the index of ``truth``.
        baseline: A key of ``predictions_by_method``; given, each method's paired difference
            with it is added (0 for the baseline itself).

    Returns:
        One row per method and metric (index ``method``, ``metric``; the metrics are
        ``macro_f1``, ``micro_f1``, then each tag, as in ``compare``): ``f1``, ``f1_low`` and
        ``f1_high``, then, given a baseline, ``delta`` (the method minus the baseline),
        ``delta_low`` and ``delta_high``.

    Raises:
        ValueError: If there is no method, if ``baseline`` is not one of them, or if a method's
            predictions or ``truth`` are not 0/1 tags on the same problems and tags.
    """
    if not predictions_by_method:
        raise ValueError("predictions_by_method is empty: there is no method to score.")
    if baseline is not None and baseline not in predictions_by_method:
        raise ValueError(
            f"The baseline {baseline!r} is not one of the methods {list(predictions_by_method)}."
        )
    _check_aligned(truth, predictions_by_method)
    tables = {}
    for name, predicted in predictions_by_method.items():
        table = _f1_with_intervals(evaluate(truth, predicted, groups=groups))
        if baseline is not None:
            paired = compare(truth, predictions_by_method[baseline], predicted, groups)
            table = table.join(paired[list(DELTA_COLUMNS)])
        tables[name] = table
    return pd.concat(tables, names=["method", "metric"])


def slice_differences(  # pylint: disable=too-many-arguments
    truth: pd.DataFrame,
    predictions_by_method: Mapping[str, pd.DataFrame],
    groups: pd.Series,
    *,
    baseline: str,
    slices: Mapping[str, np.ndarray],
    row: str = "macro_f1",
) -> pd.DataFrame:
    """Compare each method with a baseline on each slice of the problems, paired, on one metric.

    Each slice's problems are scored by ``compare``, so a method and the baseline are scored on
    the same group bootstrap resamples of that slice (e.g. a release era).

    Args:
        truth: 0/1 true focus tags, one row per problem.
        predictions_by_method: Each method's 0/1 predicted tags, with the index and columns of
            ``truth``, by method name, in the order to report; the baseline among them.
        groups: Near-duplicate group per problem, with the index of ``truth``.
        baseline: The key of ``predictions_by_method`` every other method is compared with.
        slices: A boolean mask over the rows of ``truth`` per slice name, in the order to report.
        row: The metric of ``compare`` to report: ``macro_f1``, ``micro_f1`` or a tag.

    Returns:
        One row per slice and method other than the baseline (index ``slice``, ``method``):
        ``problems`` (the slice's size), ``baseline`` and ``candidate`` (the baseline's and the
        method's value of ``row`` on the slice), ``delta`` (the method minus the baseline),
        ``delta_low`` and ``delta_high``.

    Raises:
        ValueError: If ``baseline`` is not one of the methods or is the only one, if a method's
            predictions or ``truth`` are not 0/1 tags on the same problems and tags, if ``row``
            is not a metric of ``compare``, or if there is no slice or a slice is not a boolean
            mask of ``truth``'s length with at least one problem.
    """
    others = _compared_methods(predictions_by_method, baseline)
    _check_aligned(truth, predictions_by_method)
    if row not in (*_OVERALL, *truth.columns):
        raise ValueError(f"row must be one of {[*_OVERALL, *truth.columns]}, not {row!r}.")
    masks = _slice_masks(slices, len(truth))
    reference = predictions_by_method[baseline]
    paired = {
        (name, method): compare(
            truth[mask], reference[mask], predictions_by_method[method][mask], groups[mask]
        ).loc[row, list(SLICE_COLUMNS[1:])]
        for name, mask in masks.items()
        for method in others
    }
    table = pd.DataFrame(
        list(paired.values()), index=pd.MultiIndex.from_tuples(paired, names=["slice", "method"])
    )
    table.insert(0, "problems", [int(masks[name].sum()) for name, _ in paired])
    return table


def agreement(truth: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Count, per tag, the problems two methods both get right, one of them, or neither.

    A method gets a problem right for a tag when its 0/1 value equals the true one, so a true
    negative counts as right. Read on dev, as notebooks read errors (ADR 0001).

    Args:
        truth: 0/1 true tags, one row per problem.
        a: One method's 0/1 predicted tags, with the index and columns of ``truth``.
        b: Another method's, aligned the same way.

    Returns:
        One row per tag of ``truth`` (index ``tag``): ``both_right``, ``only_a_right``,
        ``only_b_right`` and ``both_wrong``; each row sums to the number of problems.

    Raises:
        ValueError: If ``truth``, ``a`` or ``b`` are not 0/1 tags on the same problems and tags.
    """
    _check_aligned(truth, {"a": a, "b": b})
    a_right = a.to_numpy() == truth.to_numpy()
    b_right = b.to_numpy() == truth.to_numpy()
    cells = {
        "both_right": a_right & b_right,
        "only_a_right": a_right & ~b_right,
        "only_b_right": ~a_right & b_right,
        "both_wrong": ~a_right & ~b_right,
    }
    return pd.DataFrame(
        {name: cell.sum(axis=0) for name, cell in cells.items()},
        index=pd.Index(truth.columns, name="tag"),
    )


def _check_aligned(truth: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> None:
    """Refuse a frame (or ``truth``) that is not 0/1 tags on truth's problems and tags."""
    for name, frame in {"truth": truth, **frames}.items():
        same_cells = frame.index.equals(truth.index) and frame.columns.equals(truth.columns)
        if not same_cells or not frame.isin((0, 1)).all().all():
            raise ValueError(
                f"{name} must hold 0/1 tags on truth's problems and tags, in the same order."
            )


def _compared_methods(
    predictions_by_method: Mapping[str, pd.DataFrame], baseline: str
) -> list[str]:
    """Return the methods other than the baseline; refuse an unknown or a lone baseline."""
    if baseline not in predictions_by_method:
        raise ValueError(
            f"The baseline {baseline!r} is not one of the methods {list(predictions_by_method)}."
        )
    others = [name for name in predictions_by_method if name != baseline]
    if not others:
        raise ValueError(f"The baseline {baseline!r} is the only method: there is none to compare.")
    return others


def _slice_masks(slices: Mapping[str, np.ndarray], rows: int) -> dict[str, np.ndarray]:
    """Return each slice's mask as an array; refuse no slice, or a mask that selects nothing."""
    if not slices:
        raise ValueError("slices is empty: there is no slice to compare on.")
    masks = {name: np.asarray(mask) for name, mask in slices.items()}
    for name, mask in masks.items():
        if mask.dtype != bool or mask.shape != (rows,) or not mask.any():
            raise ValueError(
                f"The slice {name!r} must be a boolean mask of the {rows} problems, selecting at "
                "least one."
            )
    return masks


def _f1_with_intervals(evaluation: Evaluation) -> pd.DataFrame:
    """Macro F1, micro F1, then each tag's F1 over all problems, with their intervals."""
    summary = evaluation.summary.loc[ALL_PROBLEMS]
    overall = pd.DataFrame(
        [
            [summary[metric], summary[f"{metric}_low"], summary[f"{metric}_high"]]
            for metric in _OVERALL
        ],
        index=list(_OVERALL),
        columns=list(SCORE_COLUMNS),
    )
    per_tag = evaluation.per_tag.loc[ALL_PROBLEMS, list(SCORE_COLUMNS)]
    return pd.concat([overall, per_tag]).rename_axis("metric")
