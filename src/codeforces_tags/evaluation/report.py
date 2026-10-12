"""Score predicted focus tags: per tag, averaged over tags, by slice, with bootstrap intervals.

Every count is a sum of weights over problems: a weight of one per problem gives the point
estimate, and the group bootstrap's draw counts give one resample per row of weights. A whole
set of resamples is then one matrix product, and every metric and slice of an evaluation, or
both predictions of a comparison, use the same resamples (paired).

Precision, recall and F1 are 0 when undefined, as with scikit-learn's ``zero_division=0``,
except in an empty slice, such as a release era absent from the problems: with nothing to
score, its counts are 0 and every metric and interval is NaN.

``report_text`` shows an evaluation over all problems as text for a terminal, and
``save_report`` saves it as files: its scores over all problems (``report_headline``) as JSON,
and its tables, with any other table given (e.g. the predictions evaluated), as CSV.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from codeforces_tags.config import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    RELEASE_ERA_FIRST_YEAR,
)
from codeforces_tags.evaluation.bootstrap import percentile_interval, resample_weights

NO_FOCUS_TAG = "no focus tag"
NOTHING_PREDICTED = "nothing predicted"
ALL_PROBLEMS, FOCUS_TAGGED = "all problems", "focus-tagged"
EARLY_ERA = f"up to {RELEASE_ERA_FIRST_YEAR - 1}"
LATE_ERA = f"from {RELEASE_ERA_FIRST_YEAR}"
# The columns that count problems, which stay 0 in an empty slice while the others are NaN.
_COUNTS = ("problems", "support", "tp", "fp", "fn", "tn")
# Positions on the last axis of _f1_scores: macro F1, micro F1, then each tag's F1.
_MACRO, _MICRO, _PER_TAG = 0, 1, slice(2, None)
_OVERALL_METRICS = ("macro_f1", "micro_f1")  # the names of _MACRO and _MICRO
# What all_tag_macro_f1 does with a tag no problem has: leave it out, or count its F1 as 0.
ABSENT_TAG_RULES = ("skip", "zero")
# The files save_report writes: the scores as JSON, then the tables of an Evaluation as CSV.
REPORT_FILE = "report.json"
REPORT_TABLES = ("per_tag.csv", "summary.csv", "tag_confusion.csv")
# The columns of the per-tag table that report_text shows, those an evaluation has, in order.
_TEXT_COLUMNS = (
    *("support", "tp", "fp", "fn", "tn"),
    *("precision", "recall", "f1", "f1_low", "f1_high"),
)


@dataclass(frozen=True)
class Evaluation:
    """The scores of one set of predicted tags.

    An empty slice keeps its rows: its counts are 0, and every metric and interval is NaN.

    Attributes:
        per_tag: One row per slice and tag (index ``slice``, ``tag``): ``support``, the
            per-tag confusion matrix (``tp``, ``fp``, ``fn``, ``tn``), ``precision``, ``recall``
            and ``f1``, plus ``f1_low`` and ``f1_high`` when groups were given.
        summary: One row per slice: ``problems``, ``macro_f1``, ``micro_f1`` and
            ``no_focus_nothing_predicted`` (the share of the slice's no-focus problems that got
            no tag, NaN when it has none), plus ``macro_f1_low``, ``macro_f1_high``,
            ``micro_f1_low`` and ``micro_f1_high`` when groups were given.
        tag_confusion: The tag confusion matrix over all problems. Rows are the true focus
            tags and ``NO_FOCUS_TAG``, columns the predicted ones and ``NOTHING_PREDICTED``; a
            cell is the share of the row's problems that got the column's prediction, NaN for a
            tag no problem has. Rows need not sum to 1: a problem can have several tags.
    """

    per_tag: pd.DataFrame
    summary: pd.DataFrame
    tag_confusion: pd.DataFrame

    @property
    def macro_f1(self) -> float:
        """The macro F1 over all problems, the primary metric."""
        return float(self.summary.at[ALL_PROBLEMS, "macro_f1"])

    @property
    def micro_f1(self) -> float:
        """The micro F1 over all problems."""
        return float(self.summary.at[ALL_PROBLEMS, "micro_f1"])

    @property
    def tag_f1(self) -> pd.Series:
        """Each tag's F1 over all problems, indexed by tag."""
        return self.per_tag.loc[ALL_PROBLEMS, "f1"]

    def overall_f1(self) -> pd.Series:
        """Return the macro F1, the micro F1, then each tag's F1, over all problems.

        Returns:
            One float per metric, index ``metric``: ``macro_f1``, ``micro_f1``, then the tags.
        """
        overall = pd.Series({"macro_f1": self.macro_f1, "micro_f1": self.micro_f1})
        return pd.concat([overall, self.tag_f1]).astype(float).rename_axis("metric")


def evaluate(  # pylint: disable=too-many-arguments
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    release_years: pd.Series | None = None,
    groups: pd.Series | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Evaluation:
    """Score predicted tags against the true ones, per tag and per slice of problems.

    The slices are all problems, the focus-tagged ones and, given release years, the two
    release eras, split at ``RELEASE_ERA_FIRST_YEAR``; a problem without a year (NaN) is in
    neither era. Given groups, a group bootstrap adds percentile intervals on per-tag, macro
    and micro F1. A slice without problems, such as an era absent from ``release_years``,
    keeps its rows with every metric and interval NaN.

    Args:
        truth: 0/1 true focus tags, one row per problem.
        predicted: 0/1 predicted tags, with the index and columns of ``truth``.
        release_years: Release year per problem, with the index of ``truth``; NaN if unknown.
        groups: Near-duplicate group per problem, with the index of ``truth``. None gives
            point estimates only.
        resamples: Group bootstrap resamples.
        confidence: Share of the resamples each interval covers.
        seed: Seed of the resamples.

    Returns:
        The per-tag table, the summary per slice and the tag confusion matrix.

    Raises:
        ValueError: If ``predicted``, ``release_years`` or ``groups`` is not aligned on
            ``truth``, or if ``truth`` or ``predicted`` holds a value other than 0 and 1.
    """
    _check_predictions(truth, {"predicted": predicted})
    _check_rows(truth, {"release_years": release_years, "groups": groups})
    slices = _slices(truth, release_years)
    per_tag, summary = _point_estimates(truth, predicted, slices)
    if groups is not None:
        weights = resample_weights(groups, resamples, seed)
        resampled = _f1_scores(_counts(truth, predicted, slices, weights))
        low, high = percentile_interval(resampled, confidence)
        per_tag = per_tag.assign(f1_low=low[:, _PER_TAG].ravel(), f1_high=high[:, _PER_TAG].ravel())
        summary = summary.assign(
            macro_f1_low=low[:, _MACRO],
            macro_f1_high=high[:, _MACRO],
            micro_f1_low=low[:, _MICRO],
            micro_f1_high=high[:, _MICRO],
        )
    empty = summary["problems"] == 0
    return Evaluation(
        _undefined_if_empty(per_tag, empty),
        _undefined_if_empty(summary, empty),
        _tag_confusion(truth, predicted),
    )


def compare(  # pylint: disable=too-many-arguments
    truth: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    groups: pd.Series,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compare two sets of predicted tags for the same problems, over all problems.

    The difference is paired: each resample of the group bootstrap scores both sets of
    predictions on the same problems, so the interval reflects how the two differ rather than
    how hard the drawn problems are.

    Args:
        truth: 0/1 true focus tags, one row per problem.
        baseline: 0/1 predicted tags of the reference, with the index and columns of ``truth``.
        candidate: 0/1 predicted tags compared with it, aligned the same way.
        groups: Near-duplicate group per problem, with the index of ``truth``.
        resamples: Group bootstrap resamples.
        confidence: Share of the resamples the interval covers.
        seed: Seed of the resamples.

    Returns:
        One row per metric (``macro_f1``, ``micro_f1``, then each tag's F1); columns
        ``baseline``, ``candidate``, ``delta`` (candidate minus baseline), ``delta_low`` and
        ``delta_high``.

    Raises:
        ValueError: If an input is not aligned on ``truth``, or if a set of tags holds a value
            other than 0 and 1.
    """
    _check_predictions(truth, {"baseline": baseline, "candidate": candidate})
    _check_rows(truth, {"groups": groups})
    ones = np.ones((1, len(truth)))
    table = pd.DataFrame(
        {
            "baseline": _overall_f1_scores(truth, baseline, ones)[0],
            "candidate": _overall_f1_scores(truth, candidate, ones)[0],
        },
        index=pd.Index([*_OVERALL_METRICS, *truth.columns], name="metric"),
    )
    weights = resample_weights(groups, resamples, seed)
    baseline_scores = _overall_f1_scores(truth, baseline, weights)
    candidate_scores = _overall_f1_scores(truth, candidate, weights)
    low, high = percentile_interval(candidate_scores - baseline_scores, confidence)
    table["delta"] = table["candidate"] - table["baseline"]
    table["delta_low"], table["delta_high"] = low, high
    return table


def all_tag_macro_f1(
    truth: pd.DataFrame, predicted: pd.DataFrame, tags: Sequence[str], *, absent: str = "skip"
) -> tuple[float, int]:
    """Average the F1 of many tags, to compare with the xCodeEval paper (All-tag macro F1).

    The headline is the macro F1 over the tags of ``ALL_TAGS`` that have at least one true
    positive problem in the part scored (37 on dev, 36 on test); its variants count an absent
    tag's F1 as 0, or average over ``PAPER_TAGS``. A tag's F1 is 0 when undefined, as with
    scikit-learn's ``zero_division=0``.

    Args:
        truth: 0/1 true tags, one row per problem, with a column for every tag of ``tags``.
        predicted: 0/1 predicted tags, with the index of ``truth`` and a column for every tag of
            ``tags``.
        tags: The tags to average over, e.g. ``ALL_TAGS`` or ``PAPER_TAGS``; other columns are
            ignored.
        absent: ``"skip"`` leaves out a tag with no true positive problem; ``"zero"`` counts its
            F1 as 0.

    Returns:
        The macro F1, and the number of tags averaged.

    Raises:
        ValueError: If ``absent`` is not one of ``ABSENT_TAG_RULES``, if a frame has no column
            for a tag of ``tags``, if ``predicted`` is not aligned on ``truth`` or holds a value
            other than 0 and 1, or if no tag is left to average.
    """
    if absent not in ABSENT_TAG_RULES:
        raise ValueError(f"absent must be one of {ABSENT_TAG_RULES}, not {absent!r}.")
    columns = list(tags)
    for name, frame in (("truth", truth), ("predicted", predicted)):
        missing = [tag for tag in columns if tag not in frame.columns]
        if missing:
            raise ValueError(f"{name} has no column for the tags {missing}.")
    truth, predicted = truth[columns], predicted[columns]
    _check_predictions(truth, {"predicted": predicted})
    actual, guessed = truth.to_numpy() == 1, predicted.to_numpy() == 1
    tp = (actual & guessed).sum(axis=0)
    fp = (~actual & guessed).sum(axis=0)
    fn = (actual & ~guessed).sum(axis=0)
    per_tag = _f1(tp, fp, fn)
    if absent == "skip":
        per_tag = per_tag[actual.any(axis=0)]
    if not per_tag.size:
        raise ValueError(
            "No tag to average: tags is empty, or none of its tags has a true positive problem."
        )
    return float(per_tag.mean()), int(per_tag.size)


def report_text(evaluation: Evaluation) -> str:
    """Show an evaluation over all problems as text, for a terminal.

    Args:
        evaluation: What ``evaluate`` returned.

    Returns:
        The per-tag table (support, the per-tag confusion matrix, precision, recall, F1 and its
        interval when there is one), the macro and micro F1 with their intervals, then the tag
        confusion matrix; scores with 4 decimals, and a share that is NaN as ``-``.
    """
    per_tag = evaluation.per_tag.loc[ALL_PROBLEMS]
    headline = report_headline(evaluation)
    averages = [
        f"{label} {headline[metric]:.4f}{_interval_text(headline[f'{metric}_interval'])}"
        for metric, label in (("macro_f1", "macro F1"), ("micro_f1", "micro F1"))
    ]
    return "\n".join(
        [
            _table_text(per_tag[[column for column in _TEXT_COLUMNS if column in per_tag]]),
            "",
            *averages,
            "",
            "Tag confusion matrix: the share of each true tag's problems (rows) that got each "
            "predicted tag (columns)",
            _table_text(evaluation.tag_confusion),
        ]
    )


def report_headline(evaluation: Evaluation) -> dict[str, object]:
    """Return an evaluation's scores over all problems, as plain JSON values.

    Args:
        evaluation: What ``evaluate`` returned.

    Returns:
        ``macro_f1`` and ``micro_f1``; ``macro_f1_interval`` and ``micro_f1_interval``, each
        ``[low, high]``, or None without groups; and ``per_tag``: each tag's ``support``,
        ``precision``, ``recall``, ``f1`` and ``f1_interval``.
    """
    summary = evaluation.summary.loc[ALL_PROBLEMS]
    per_tag = {
        str(tag): {
            "support": int(row["support"]),
            **{metric: float(row[metric]) for metric in ("precision", "recall", "f1")},
            "f1_interval": _interval(row, "f1"),
        }
        for tag, row in evaluation.per_tag.loc[ALL_PROBLEMS].iterrows()
    }
    return {
        "macro_f1": evaluation.macro_f1,
        "micro_f1": evaluation.micro_f1,
        "macro_f1_interval": _interval(summary, "macro_f1"),
        "micro_f1_interval": _interval(summary, "micro_f1"),
        "per_tag": per_tag,
    }


def save_report(
    evaluation: Evaluation,
    folder: str | os.PathLike[str],
    details: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame | None] | None = None,
) -> list[Path]:
    """Save an evaluation in a folder: its scores as JSON, and its tables as CSV.

    Args:
        evaluation: What ``evaluate`` returned.
        folder: An existing folder.
        details: What was evaluated (e.g. the method and the problems), the first keys of
            ``REPORT_FILE``.
        tables: Other tables to save as CSV, by file name, e.g. the predictions evaluated; a
            table that is None is left out.

    Returns:
        The paths written: ``REPORT_FILE`` (``details``, then ``report_headline``), then
        ``REPORT_TABLES`` (``per_tag``, ``summary`` and ``tag_confusion``), then those of
        ``tables``.

    Raises:
        ValueError: If a score of the headline is NaN (an evaluation of no problem), which
            strict JSON cannot hold.
    """
    folder = Path(folder)
    report = {**details, **report_headline(evaluation)}
    (folder / REPORT_FILE).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    own = (evaluation.per_tag, evaluation.summary, evaluation.tag_confusion)
    every_table = dict(zip(REPORT_TABLES, own, strict=True)) | dict(tables or {})
    written = [name for name, table in every_table.items() if table is not None]
    for name in written:
        every_table[name].to_csv(folder / name)
    return [folder / name for name in (REPORT_FILE, *written)]


def _check_predictions(truth: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> None:
    for name, frame in {"truth": truth, **predictions}.items():
        if not (frame.index.equals(truth.index) and frame.columns.equals(truth.columns)):
            raise ValueError(f"{name} must have truth's index and columns, in the same order.")
        if not np.isin(frame.to_numpy(), (0, 1)).all():
            raise ValueError(f"{name} must hold only 0 and 1.")


def _check_rows(truth: pd.DataFrame, aligned: dict[str, pd.Series | None]) -> None:
    for name, per_problem in aligned.items():
        if per_problem is not None and not per_problem.index.equals(truth.index):
            raise ValueError(f"{name} must have truth's index, in the same order.")


def _slices(truth: pd.DataFrame, release_years: pd.Series | None) -> pd.DataFrame:
    """One bool column per slice, True for the problems in it."""
    slices = pd.DataFrame({ALL_PROBLEMS: True, FOCUS_TAGGED: truth.any(axis=1)}, index=truth.index)
    if release_years is not None:
        # Both comparisons are False for a missing year, which is then in neither era.
        slices[EARLY_ERA] = release_years < RELEASE_ERA_FIRST_YEAR
        slices[LATE_ERA] = release_years >= RELEASE_ERA_FIRST_YEAR
    return slices


def _counts(
    truth: pd.DataFrame, predicted: pd.DataFrame, slices: pd.DataFrame, weights: np.ndarray
) -> np.ndarray:
    """Sum the problems' weights per outcome, slice and tag.

    The result has shape ``(len(weights), 4, slices, tags)``; axis 1 holds the true positives,
    false positives, false negatives and true negatives, in that order.
    """
    actual, guessed = truth.to_numpy() == 1, predicted.to_numpy() == 1
    outcomes = np.stack(
        [actual & guessed, ~actual & guessed, actual & ~guessed, ~actual & ~guessed], axis=1
    )  # (problems, 4, tags)
    in_slice = outcomes[:, :, None, :] & slices.to_numpy()[:, None, :, None]
    return np.tensordot(weights.astype(float), in_slice.astype(float), axes=1)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide element-wise, with 0 where the denominator is 0."""
    ratios = np.zeros(np.shape(numerator))
    return np.divide(numerator, denominator, out=ratios, where=denominator > 0)


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _f1_scores(counts: np.ndarray) -> np.ndarray:
    """Macro F1, micro F1, then each tag's F1, shape ``(len(weights), slices, 2 + tags)``."""
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    per_tag = _f1(tp, fp, fn)
    micro = _f1(tp.sum(axis=-1), fp.sum(axis=-1), fn.sum(axis=-1))
    macro = per_tag.mean(axis=-1, keepdims=True)
    return np.concatenate([macro, micro[..., None], per_tag], axis=-1)


def _overall_f1_scores(
    truth: pd.DataFrame, predicted: pd.DataFrame, weights: np.ndarray
) -> np.ndarray:
    """Macro F1, micro F1, then each tag's F1 over all problems, one row per row of weights."""
    every_problem = pd.DataFrame({ALL_PROBLEMS: True}, index=truth.index)
    return _f1_scores(_counts(truth, predicted, every_problem, weights))[:, 0]


def _point_estimates(
    truth: pd.DataFrame, predicted: pd.DataFrame, slices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-tag table and the summary per slice, without intervals."""
    counts = _counts(truth, predicted, slices, np.ones((1, len(truth))))
    # Each outcome is (slices, tags), flattened slice by slice like the index below.
    tp, fp, fn, tn = (outcome.ravel() for outcome in counts[0].astype(np.int64))
    per_tag = pd.DataFrame(
        {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        index=pd.MultiIndex.from_product([slices.columns, truth.columns], names=["slice", "tag"]),
    ).assign(precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn), f1=_f1(tp, fp, fn))
    scores = _f1_scores(counts)[0]
    summary = pd.DataFrame(
        {
            "problems": slices.sum(),
            "macro_f1": scores[:, _MACRO],
            "micro_f1": scores[:, _MICRO],
            "no_focus_nothing_predicted": _no_focus_nothing_predicted(truth, predicted, slices),
        }
    ).rename_axis("slice")
    return per_tag, summary


def _undefined_if_empty(table: pd.DataFrame, empty: pd.Series) -> pd.DataFrame:
    """Set every column but the counts to NaN in the rows of an empty slice (index ``slice``)."""
    rows = empty.reindex(table.index.get_level_values("slice")).to_numpy()
    table = table.copy()
    table.loc[rows, table.columns.difference(_COUNTS, sort=False)] = np.nan
    return table


def _no_focus_nothing_predicted(
    truth: pd.DataFrame, predicted: pd.DataFrame, slices: pd.DataFrame
) -> pd.Series:
    """Per slice, the share of its no-focus problems with no tag predicted (NaN if none)."""
    no_focus, got_nothing = ~truth.any(axis=1), ~predicted.any(axis=1)
    return slices[no_focus & got_nothing].sum() / slices[no_focus].sum()


def _tag_confusion(truth: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """The share of each true tag's problems (rows) that got each prediction (columns)."""
    rows = truth.assign(**{NO_FOCUS_TAG: ~truth.any(axis=1)}).astype(int)
    columns = predicted.assign(**{NOTHING_PREDICTED: ~predicted.any(axis=1)}).astype(int)
    shares = (rows.T @ columns).div(rows.sum(), axis=0)
    return shares.rename_axis(index="true", columns="predicted")


def _interval(row: pd.Series, metric: str) -> list[float] | None:
    """The percentile interval of a metric, as ``[low, high]``; None without groups."""
    if f"{metric}_low" not in row:
        return None
    return [float(row[f"{metric}_low"]), float(row[f"{metric}_high"])]


def _interval_text(interval: list[float] | None) -> str:
    return "" if interval is None else f" [{interval[0]:.4f}, {interval[1]:.4f}]"


def _table_text(table: pd.DataFrame) -> str:
    """A table as text, scores with 4 decimals and NaN as ``-``, with no trailing spaces."""
    text = table.to_string(float_format="{:.4f}".format, na_rep="-")
    return "\n".join(line.rstrip() for line in text.splitlines())
