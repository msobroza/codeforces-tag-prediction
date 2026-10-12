"""Why a threshold rule matters, for notebook 11: F1 curves, their peaks, Platt-mapped scores.

A score method turns a tag's scores into 0/1 through its threshold rule. The F1-argmax rule takes
the threshold where the F1 curve of its fitting scores is highest: when that curve has two peaks
of almost equal height far apart, the four dev folds that choose a fold's threshold decide which
peak wins. The Platt + Lipton rule (``model.platt_lipton``) reads the height of the curve instead,
on Platt-mapped probabilities. These functions show both from the frozen scores of a Jev method:
each tag's F1 curve over Jev's grid of probabilities (``f1_curve``, between the cuts of
``grid_cuts``), the curves that choose each fold's threshold (``fold_curves``), the runs of a
curve within a margin of its maximum (``curve_peaks``), and Jev's probabilities mapped out of
fold by Platt maps of their logits (``platt_probabilities``: the ``platt_logit`` calibrator of
``analysis.calibrators``, whose feature and map are the rule's own). They read dev only (ADR
0001), send no request and draw nothing: ``plotting.plot_f1_curve`` draws a curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codeforces_tags.analysis.calibrators import make_calibrator
from codeforces_tags.config import SCORE_GRID
from codeforces_tags.evaluation.calibration import cross_fitted_probabilities

# The columns of curve_peaks, one row per run of cuts near the curve's maximum (index ``peak``):
# its first and last cut, the cut of its highest F1 and that F1, and how far below the maximum.
PEAK_COLUMNS = ("start", "end", "threshold", "f1", "below_max")
# Two F1 values closer than this are equal: each is a ratio of counts, computed in float64.
_TIE = 1e-12


def grid_cuts(grid: float = SCORE_GRID) -> np.ndarray:
    """Return the cuts halfway between consecutive values of a grid of scores from 0 to 1.

    On scores that lie on the grid (Jev's probabilities lie on 0.01, and a failed answer scores
    0.0), a cut at a midpoint predicts what every threshold between the two grid values around it
    predicts, so these cuts give every distinct 0/1 prediction of a tag.

    Args:
        grid: The step of the grid, which divides 1.

    Returns:
        ``(k + 0.5) * grid`` for k from 0 to ``1 / grid - 1``, rounded to 9 decimals, so that
        0.125 is 0.125: the 100 cuts 0.005, 0.015, ..., 0.995 for Jev's grid.

    Raises:
        ValueError: If ``grid`` is not above 0 and at most 1, or does not divide 1.
    """
    if not 0 < grid <= 1 or not np.isclose(round(1 / grid) * grid, 1, rtol=0, atol=1e-9):
        raise ValueError(f"The grid must be a step above 0 that divides 1, not {grid}.")
    return np.round((np.arange(round(1 / grid)) + 0.5) * grid, 9)


def f1_curve(scores: pd.Series, labels: pd.Series, grid: float = SCORE_GRID) -> pd.Series:
    """Compute one tag's F1 when every score at or above a cut is predicted, at each cut.

    Args:
        scores: One tag's scores, one row per problem.
        labels: The tag's 0/1 true labels, with the index of ``scores``.
        grid: The step of the scores' grid (``grid_cuts``).

    Returns:
        The F1 at each cut of ``grid_cuts(grid)`` (index ``threshold``, named like ``scores``):
        ``2 TP / (predicted + positives)``, and 0 where nothing is predicted and no problem has
        the tag.

    Raises:
        ValueError: If ``labels`` does not have the index of ``scores``, or holds another value
            than 0 and 1; or as ``grid_cuts`` does.
    """
    if not labels.index.equals(scores.index):
        raise ValueError("labels must have the index of scores: one label per problem, in order.")
    if not labels.isin((0, 1)).all():
        raise ValueError("labels must hold only 0 and 1.")
    cuts = grid_cuts(grid)
    truth = labels.to_numpy(dtype=bool)
    predicted = scores.to_numpy(dtype=float)[:, np.newaxis] >= cuts
    true_positives = (predicted & truth[:, np.newaxis]).sum(axis=0)
    denominator = predicted.sum(axis=0) + truth.sum()
    f1 = np.divide(2 * true_positives, denominator, out=np.zeros(len(cuts)), where=denominator > 0)
    return pd.Series(f1, index=pd.Index(cuts, name="threshold"), name=scores.name)


def fold_curves(
    scores: pd.Series, labels: pd.Series, folds: pd.Series, grid: float = SCORE_GRID
) -> pd.DataFrame:
    """Compute, for each dev fold, the F1 curve of the other folds, which its threshold is read on.

    Under a cross-fitted rule, fold k's threshold is chosen on the problems of the other folds, so
    the F1-argmax threshold of fold k is the highest point of column k.

    Args:
        scores: One tag's out-of-fold scores, one row per dev problem.
        labels: The tag's 0/1 true labels, with the index of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.
        grid: The step of the scores' grid (``grid_cuts``).

    Returns:
        One column per fold, in sorted order (columns named like ``folds``): ``f1_curve`` of the
        problems of every other fold, indexed by ``threshold``.

    Raises:
        ValueError: If ``folds`` does not have the index of ``scores`` or misses a problem's
            fold, as ``check_dev_folds`` refuses it for ``platt_probabilities``; or as
            ``f1_curve`` does.
    """
    if not folds.index.equals(scores.index):
        raise ValueError("folds must have the index of scores: one fold per problem, in order.")
    missing = int(folds.isna().sum())
    if missing:
        raise ValueError(
            f"{missing} problems have no fold: a missing fold would add a curve of every problem, "
            "held out by no fold. Take the folds of the split's dev problems."
        )
    curves = {}
    for fold in np.sort(folds.unique()):
        # By position, not by index: when labels is indexed differently from the scores,
        # f1_curve says so, where a mask by index would raise pandas' IndexingError.
        others = (folds != fold).to_numpy()
        curves[fold] = f1_curve(scores[others], labels[others], grid)
    return pd.DataFrame(curves).rename_axis(columns=folds.name)


def curve_peaks(curve: pd.Series, within: float = 0.01) -> pd.DataFrame:
    """Find the runs of consecutive cuts whose F1 is within a margin of the curve's maximum.

    Two runs far apart are two peaks of almost equal height: an F1-argmax threshold can fall on
    either, depending on the problems it is chosen on.

    Args:
        curve: F1 by cut, as ``f1_curve`` returns it, in increasing order of the cuts.
        within: The margin below the maximum F1 that a cut may lie and still belong to a peak.

    Returns:
        One row per run, from the lowest cuts to the highest (index ``peak``, from 0), with the
        columns ``PEAK_COLUMNS``: ``start`` and ``end`` (its first and last cut), ``threshold``
        (the cut of its highest F1, the highest such cut on a tie, as ``best_threshold`` breaks
        ties), ``f1`` (that F1) and ``below_max`` (the curve's maximum minus it).

    Raises:
        ValueError: If ``curve`` is empty or holds a NaN, or if ``within`` is negative.
    """
    if curve.empty or curve.isna().any():
        raise ValueError("The curve must hold at least one F1, and no NaN.")
    if within < 0:
        raise ValueError(f"within must be 0 or more, not {within}.")
    best = float(curve.max())
    near = (curve >= best - within - _TIE).to_numpy()
    runs = np.cumsum(near & ~np.concatenate([[False], near[:-1]]))
    rows = []
    for number in range(1, int(runs.max()) + 1):
        members = curve[near & (runs == number)]
        top = float(members.max())
        highest = members.index[members >= top - _TIE][-1]
        rows.append([members.index[0], members.index[-1], highest, top, best - top])
    return pd.DataFrame(
        rows, index=pd.RangeIndex(len(rows), name="peak"), columns=list(PEAK_COLUMNS)
    )


def platt_probabilities(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Map each fold's probabilities through Platt maps of their logits fitted on the other folds.

    The map is the ``platt_logit`` calibrator of ``analysis.calibrators``, so these are its
    probabilities in ``calibrated_probabilities``, to the last bit. Its feature is the Platt +
    Lipton rule's (``platt_lipton.platt_feature``): the logit of the probability clipped to
    ``[PLATT_CLIP, 1 - PLATT_CLIP]``, so a failed answer's 0.0 maps to a finite value; and its
    map is the rule's converged ``platt_lipton.rule_calibrator()``. Each tag's map is fitted on
    the other folds' scores and labels (``evaluation.calibration.cross_fitted_probabilities``),
    so no problem's probability was fitted on its own label: a reliability table of these
    probabilities is out of fold, as the rule's thresholds are.

    Args:
        scores: Probabilities from 0 to 1 (a Jev method's dev scores), one column per tag, one row
            per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.
        folds: Dev fold per problem, with the index of ``scores``.

    Returns:
        The Platt-mapped probabilities, float64, with the index and columns of ``scores``.

    Raises:
        ValueError: As ``cross_fitted_probabilities`` does, before anything is fitted:
            misaligned folds or labels, a missing or negative fold, fewer than two folds, or a
            tag with one class outside a fold; then, when a fit or an out-of-fold mapping first
            reads it, if a score lies outside [0, 1] or is NaN, as ``platt_lipton.platt_feature``
            refuses it (other tags may have been fitted by then).
    """
    return cross_fitted_probabilities(scores, labels, folds, make_calibrator("platt_logit"))
