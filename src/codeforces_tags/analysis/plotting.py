"""Shared plot style and plots, for the notebooks and the figures ``tracking calibrate`` logs.

Needs the ``analysis`` group.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

from codeforces_tags.analysis.eda import values_by_label
from codeforces_tags.config import (
    CALIBRATION_BINS,
    CALIBRATION_METHODS,
    CALIBRATION_STRATEGY,
    FIGURE_DPI,
    FIGURE_SIZE,
    FOCUS_COLOR,
    FOCUS_TAGS,
    LATENCY_BUDGET_S,
    OTHER_COLOR,
)
from codeforces_tags.evaluation.calibration import check_probabilities

# The cells of a per-tag confusion matrix: problems with the tag on the first row, without it on
# the second; predicted on the first column, not predicted on the second.
CONFUSION_CELLS = (("tp", "fn"), ("fp", "tn"))
# One marker per method, in order: the curves of a reliability diagram stay apart in grey scale.
RELIABILITY_MARKERS = ("^", "v", "s", "o")
# The axes of a reliability diagram: the mean prediction of a bin, and its share of positives.
RELIABILITY_LABELS = ("Mean predicted probability", "Fraction of positives")
# Figure sizes, in inches. The diagram over four histograms is scikit-learn's calibration example
# (10 x 10). The grid keeps the notebooks' 10-inch width, as tall as two rows of square diagrams,
# the legend and the figure's labels need.
RELIABILITY_SIZE = (10, 10)
RELIABILITY_GRID_SIZE = (10, 5.6)
# Eight diagrams share the grid: thinner curves and smaller markers than the single diagram's.
RELIABILITY_GRID_LINES = {"markersize": 4, "linewidth": 1.2}
# A search's progress: its best score above, its cumulative costs below, sharing the iterations.
SEARCH_PROGRESS_SIZE = (10, 6)
# The cost columns of prompt_search.search_progress, and the name of each curve.
SEARCH_COSTS = (("jev_cost_usd", "Jev"), ("luna_cost_usd", "Luna"), ("total_cost_usd", "total"))
# An F1 curve's line widths: each fold's curve thin, under the curve of every problem.
F1_CURVE_LINES = {"fold": 0.8, "curve": 2.0}


def setup_style() -> None:
    """Apply the same seaborn and matplotlib theme in every notebook."""
    sns.set_theme(style="whitegrid", context="notebook", palette="tab10")
    plt.rcParams.update(
        {"figure.figsize": FIGURE_SIZE, "figure.dpi": FIGURE_DPI, "axes.titleweight": "bold"}
    )


def focus_palette(tags: Iterable[str]) -> list[str]:
    """Pick one bar color per tag, highlighting the challenge's focus tags.

    Args:
        tags: Tag names, in plotting order.

    Returns:
        ``FOCUS_COLOR`` for focus tags and ``OTHER_COLOR`` for the others.
    """
    return [FOCUS_COLOR if tag in FOCUS_TAGS else OTHER_COLOR for tag in tags]


def boxplot_by_label(
    values: pd.Series, indicators: pd.DataFrame, log: bool = False, none_label: str = "(none)"
) -> Axes:
    """Draw one horizontal box of values per label, sorted by median.

    A row with several labels appears in each of their boxes; rows without any label form a
    grey box named ``none_label``.

    Args:
        values: One value per row of ``indicators``.
        indicators: 0/1 label matrix, e.g. from ``multi_hot``.
        log: Use a log scale for the values.
        none_label: Name of the box for rows without any label.

    Returns:
        The axes, ready for a title and axis labels.
    """
    long_values = values_by_label(values, indicators, none_label=none_label)
    order = long_values.groupby("label")["value"].median().sort_values().index
    palette = {label: OTHER_COLOR if label == none_label else FOCUS_COLOR for label in order}
    _, ax = plt.subplots()
    sns.boxplot(
        data=long_values,
        x="value",
        y="label",
        order=order,
        hue="label",
        palette=palette,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    if log:
        ax.set_xscale("log")
    ax.set(ylabel="")
    return ax


def plot_intervals(
    table: pd.DataFrame, estimate: str, low: str, high: str, ax: Axes | None = None
) -> Axes:
    """Draw one point per row with a horizontal bar over its interval, such as a score's CI.

    The bar is drawn as given, even when it does not contain the point: a bootstrap percentile
    interval need not.

    Args:
        table: One row per point, labelled by the index, in the order to draw from top to
            bottom.
        estimate: Column of the points.
        low: Column of the intervals' lower ends.
        high: Column of the intervals' upper ends.
        ax: Axes to draw on; a new figure when None.

    Returns:
        The axes, with the estimate column as x label.
    """
    if ax is None:
        _, ax = plt.subplots()
    positions = range(len(table))
    ax.hlines(positions, table[low], table[high], color=FOCUS_COLOR)
    ax.plot(table[estimate], positions, "o", color=FOCUS_COLOR)
    ax.set_yticks(positions, labels=table.index.astype(str))
    ax.invert_yaxis()
    ax.set(xlabel=estimate, ylabel="")
    return ax


def plot_tag_confusion(matrix: pd.DataFrame) -> Axes:
    """Draw the tag confusion matrix as a heatmap of shares, annotated in percent.

    Args:
        matrix: Rows the true focus tags plus "no focus tag", columns the predicted focus tags
            plus "nothing predicted"; a cell is the share of the row's problems that received
            the column's prediction, such as ``Evaluation.tag_confusion``.

    Returns:
        The axes, on a colour scale from 0 to 1 whatever the shares.
    """
    _, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax)
    ax.set(xlabel="predicted", ylabel="true")
    return ax


def plot_confusion_grid(per_tag: pd.DataFrame) -> Figure:
    """Draw each tag's per-tag confusion matrix as a small heatmap, in two rows.

    Each heatmap reads TP FN on its first row (the problems with the tag) and FP TN on its
    second (the problems without it), annotated with the counts. A cell's colour is its share
    of the row, so the first row shows the recall, and a rare tag's errors are not washed out
    by its many true negatives.

    Args:
        per_tag: Counts ``tp``, ``fp``, ``fn`` and ``tn``, one row per tag, such as one slice
            of ``Evaluation.per_tag``.

    Returns:
        The figure, with one axes per tag, titled by the tag, in the order of ``per_tag``.
    """
    figure, grid = plt.subplots(2, math.ceil(len(per_tag) / 2), squeeze=False, layout="constrained")
    axes = grid.ravel()
    for ax, (tag, counts) in zip(axes, per_tag.iterrows(), strict=False):
        _confusion_heatmap(counts, ax)
        ax.set_title(tag)
    for unused in axes[len(per_tag) :]:
        unused.remove()
    return figure


def plot_study_history(trials: pd.DataFrame) -> Axes:
    """Draw each complete trial's value and the best value so far, by trial number.

    Args:
        trials: One row per trial, with columns ``number``, ``state`` (Optuna's state name) and
            ``value``. Trials that are not ``COMPLETE`` (failed, or still running) are left out.

    Returns:
        The axes, with the trials in ``OTHER_COLOR`` and the best value so far in
        ``FOCUS_COLOR``.
    """
    complete = trials[trials["state"] == "COMPLETE"].sort_values("number")
    _, ax = plt.subplots()
    ax.scatter(complete["number"], complete["value"], color=OTHER_COLOR, label="trial")
    ax.step(
        complete["number"],
        complete["value"].cummax(),
        where="post",
        color=FOCUS_COLOR,
        label="best so far",
    )
    ax.set(xlabel="trial number", ylabel="value")
    ax.legend()
    return ax


def plot_method_f1(scores: pd.DataFrame) -> Axes:
    """Draw each method's F1 with its interval: one row per metric, the methods side by side.

    Args:
        scores: The table of ``comparison.method_scores``: index ``(method, metric)``, columns
            ``f1``, ``f1_low`` and ``f1_high``.

    Returns:
        The axes: one row per metric, from top to bottom in the order of ``scores``. In each
        row, each method has a point and a horizontal bar over its interval, in its own colour,
        offset from the row's centre in the order of the methods; the legend names the methods.
        The F1 axis runs from 0 to 1.
    """
    methods = list(scores.index.unique("method"))
    metrics = list(scores.index.unique("metric"))
    rows = np.arange(len(metrics))
    step = 0.8 / len(methods)  # the methods of a row share 80% of its height
    colors = sns.color_palette(n_colors=len(methods))
    _, ax = plt.subplots()
    for number, (method, color) in enumerate(zip(methods, colors, strict=True)):
        table = scores.loc[method].reindex(metrics)
        positions = rows + (number - (len(methods) - 1) / 2) * step
        ax.hlines(positions, table["f1_low"], table["f1_high"], color=color)
        ax.plot(table["f1"], positions, "o", color=color, label=method)
    ax.set_yticks(rows, labels=metrics)
    ax.invert_yaxis()
    ax.set(xlabel="F1", ylabel="", xlim=(0, 1))
    ax.legend()
    return ax


def plot_latency(latencies: Mapping[str, pd.Series]) -> Axes:
    """Draw each method's single-problem latencies on a log scale, against the time budget.

    Args:
        latencies: Seconds per problem (``timing.time_single``, or the problem rows of a
            ``latency.csv``), by method name, in the order to draw from top to bottom.

    Returns:
        The axes: one row per method, with a point per problem in ``OTHER_COLOR`` and the
        median as a tick in ``FOCUS_COLOR``, and a dashed line at ``LATENCY_BUDGET_S``, the
        single-problem budget of PROBLEM.md; the seconds axis is logarithmic.
    """
    seconds = list(latencies.values())
    rows = np.arange(len(seconds))
    _, ax = plt.subplots()
    ax.scatter(
        np.concatenate([times.to_numpy(float) for times in seconds]),
        np.repeat(rows, [len(times) for times in seconds]),
        color=OTHER_COLOR,
        label="problem",
    )
    medians = [times.median() for times in seconds]
    ax.scatter(medians, rows, marker="|", s=400, color=FOCUS_COLOR, label="median")
    budget = f"{LATENCY_BUDGET_S:g} s budget"
    ax.axvline(LATENCY_BUDGET_S, color="black", linestyle="--", label=budget)
    ax.set_yticks(rows, labels=list(latencies))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set(xlabel="seconds per problem", ylabel="")
    ax.legend()
    return ax


def plot_reliability(probabilities: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> Figure:
    """Draw one reliability diagram over all problem-tag pairs, and a histogram per method below.

    The diagram has a curve per method: the share of positives in each bin of predicted
    probability against the bin's mean prediction, in ``CALIBRATION_BINS`` bins. The strategy,
    ``CALIBRATION_STRATEGY``, is "quantile": bins of equal counts, so that the high end of a rare
    tag is not left empty. Probabilities that are calibrated follow the dotted diagonal. Below
    it, one histogram per method counts the pairs in each tenth of [0, 1]. Its count axis is
    logarithmic, since most pairs get a probability near 0, and shared, so that the methods'
    counts compare.

    Args:
        probabilities: Probabilities by method, as ``probabilities_by_method`` returns them. A
            method of ``config.CALIBRATION_METHODS`` is named by its name, any other by its id.
        labels: 0/1 labels, with the index and columns of every method's probabilities.

    Returns:
        The figure, 10 by 10 inches: the diagram on the two top rows of a grid of four, then the
        histograms in the order of the methods, two to a row, each in the colour of its curve.

    Raises:
        ValueError: As ``check_probabilities`` does, before anything is drawn.
    """
    check_probabilities(probabilities, labels)
    styles = _method_styles(probabilities)
    figure = plt.figure(figsize=RELIABILITY_SIZE, layout="constrained")
    # The diagram takes two rows; the histograms follow, two to a row.
    grid = figure.add_gridspec(2 + math.ceil(len(probabilities) / 2), 2)
    diagram = figure.add_subplot(grid[:2, :])
    truth = labels.to_numpy().ravel()
    for method, frame in probabilities.items():
        _draw_reliability_curve(truth, frame.to_numpy().ravel(), method, diagram, **styles[method])
    xlabel, ylabel = RELIABILITY_LABELS
    diagram.set(title="Reliability diagram: all problem-tag pairs", xlabel=xlabel, ylabel=ylabel)
    _draw_histograms(figure, grid, probabilities, styles)
    return figure


def plot_reliability_grid(
    probabilities: Mapping[str, pd.DataFrame], labels: pd.DataFrame
) -> Figure:
    """Draw one reliability diagram per tag, in two rows, each with every method's curve.

    A tag's diagram is the one of ``plot_reliability``, over that tag's problems alone, with each
    method in the colour and marker it has there.

    Args:
        probabilities: Probabilities by method, as ``probabilities_by_method`` returns them. A
            method of ``config.CALIBRATION_METHODS`` is named by its name, any other by its id.
        labels: 0/1 labels, with the index and columns of every method's probabilities.

    Returns:
        The figure, 10 by 5.6 inches: one square axes per tag, titled by the tag, in the order of
        the columns of ``labels``; a place the tags leave over stays empty. One legend above the
        diagrams names the methods and the diagonal in one row, and one label beside them names
        each axis.

    Raises:
        ValueError: As ``check_probabilities`` does, before anything is drawn.
    """
    check_probabilities(probabilities, labels)
    styles = _method_styles(probabilities)
    tags = list(labels.columns)
    figure, grid = plt.subplots(
        2,
        math.ceil(len(tags) / 2),
        squeeze=False,
        layout="constrained",
        figsize=RELIABILITY_GRID_SIZE,
    )
    axes = grid.ravel()
    for ax, tag in zip(axes, tags, strict=False):
        _draw_tag_diagram(ax, tag, labels, probabilities, styles)
    for unused in axes[len(tags) :]:
        unused.remove()
    _label_reliability_grid(figure, axes[0])
    return figure


def plot_reliability_table(table: pd.DataFrame) -> Figure:
    """Draw each tag's reliability curve in its own panel, against the diagonal.

    A calibrated score has its points on the diagonal: in every bin, the share of problems with
    the tag equals the mean probability.

    Args:
        table: The table of ``calibration.reliability_table``: index ``tag``, ``bin``, with the
            columns ``mean_probability`` and ``positive_rate``.

    Returns:
        The figure, with one axes per tag, titled by the tag, in the order of ``table`` and in
        two rows; all of them share the x and y axes, from 0 to 1. Each draws the dashed
        diagonal first, then a point per bin joined by a line: the bin's mean probability
        against the share of its problems that have the tag.
    """
    tags = list(table.index.unique("tag"))
    figure, grid = plt.subplots(
        2, math.ceil(len(tags) / 2), squeeze=False, sharex=True, sharey=True, layout="constrained"
    )
    axes = grid.ravel()
    for ax, tag in zip(axes, tags, strict=False):
        bins = table.loc[tag]
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax.plot(bins["mean_probability"], bins["positive_rate"], marker="o", color=FOCUS_COLOR)
        ax.set(title=tag, xlim=(0, 1), ylim=(0, 1))
    for unused in axes[len(tags) :]:
        unused.remove()
    figure.supxlabel("mean probability in the bin")
    figure.supylabel("share of problems with the tag")
    return figure


def plot_search_progress(progress: pd.DataFrame) -> Figure:
    """Draw a search's best selection score and its spending, iteration by iteration.

    Args:
        progress: The table of ``prompt_search.search_progress``: index ``iteration``, columns
            ``best_selection_mean``, ``accepted`` and the cumulative ``jev_cost_usd``,
            ``luna_cost_usd`` and ``total_cost_usd``.

    Returns:
        The figure, 10 by 6 inches, with two axes sharing the iteration axis. The top one draws
        the best mean score on the selection fold so far as a step line in ``FOCUS_COLOR``, and
        each iteration whose child was accepted as a point on it in ``OTHER_COLOR``. The bottom
        one draws the cumulative cost of Jev, of the reflection model and of both, in USD.
    """
    figure, (score, cost) = plt.subplots(
        2, 1, sharex=True, figsize=SEARCH_PROGRESS_SIZE, layout="constrained"
    )
    iterations = progress.index.to_numpy()
    best = progress["best_selection_mean"].to_numpy(float)
    score.step(iterations, best, where="post", color=FOCUS_COLOR, label="best so far")
    accepted = progress["accepted"].to_numpy(bool)
    score.scatter(
        iterations[accepted], best[accepted], color=OTHER_COLOR, zorder=3, label="accepted child"
    )
    score.set(ylabel="mean score, selection fold")
    score.legend(loc="lower right")
    for column, name in SEARCH_COSTS:
        cost.plot(iterations, progress[column].to_numpy(float), label=name)
    cost.set(xlabel="iteration", ylabel="cumulative cost (USD)")
    cost.legend(loc="upper left")
    return figure


def plot_f1_curve(
    curve: pd.Series, thresholds: Mapping[str, float], folds: pd.DataFrame | None = None
) -> Figure:
    """Draw one tag's F1 curve against the threshold, with a dashed line at each named threshold.

    Args:
        curve: F1 by cut, as ``threshold_rules.f1_curve`` returns it, named by its tag.
        thresholds: A threshold by name (e.g. each threshold rule's), in the order to draw. An
            infinite threshold (a tag that no problem has) is left out: it predicts nothing.
        folds: Optional curves of ``threshold_rules.fold_curves``, one column per fold, drawn
            under ``curve``.

    Returns:
        The figure, with one axes titled by the tag: first each fold's curve, thin, in
        ``OTHER_COLOR`` and labelled ``without fold <k>``; then ``curve`` in ``FOCUS_COLOR``,
        labelled ``all problems``; then a vertical dashed line at each finite threshold, each in
        its own colour, labelled ``<name> <threshold to 3 decimals>``. The threshold axis runs
        from 0 to 1, and the legend, at the bottom centre where the curve is low, names every
        line.
    """
    figure, ax = plt.subplots(layout="constrained")
    if folds is not None:
        for fold in folds.columns:
            ax.plot(
                folds.index,
                folds[fold].to_numpy(dtype=float),
                color=OTHER_COLOR,
                linewidth=F1_CURVE_LINES["fold"],
                label=f"without fold {fold}",
            )
    ax.plot(
        curve.index,
        curve.to_numpy(dtype=float),
        color=FOCUS_COLOR,
        linewidth=F1_CURVE_LINES["curve"],
        label="all problems",
    )
    finite = {name: value for name, value in thresholds.items() if np.isfinite(value)}
    colors = sns.color_palette("dark", n_colors=max(len(finite), 1))
    for (name, value), color in zip(finite.items(), colors, strict=False):
        ax.axvline(value, color=color, linestyle="--", label=f"{name} {value:.3f}")
    ax.set(title=str(curve.name), xlabel="threshold", ylabel="F1", xlim=(0, 1))
    ax.legend(loc="lower center")
    return figure


def _confusion_heatmap(counts: pd.Series, ax: Axes) -> None:
    """Draw one tag's counts as a 2x2 heatmap coloured by the share of each row."""
    cells = pd.DataFrame([[counts[name] for name in row] for row in CONFUSION_CELLS])
    shares = cells.div(cells.sum(axis=1), axis=0).fillna(0)  # a row without problems: 0
    labels = [[f"{name.upper()}\n{counts[name]:,.0f}" for name in row] for row in CONFUSION_CELLS]
    sns.heatmap(
        shares,
        annot=labels,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )


def _method_styles(methods: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Give each method, in order, a colour of the notebooks' palette and a marker."""
    ids = list(methods)
    colors = sns.color_palette(n_colors=len(ids))
    markers = itertools.cycle(RELIABILITY_MARKERS)
    return {
        method: {"color": color, "marker": marker}
        for method, color, marker in zip(ids, colors, markers, strict=False)
    }


def _method_name(method: str) -> str:
    """Name a method as the figures do: by its name in the calibration settings, else by its id."""
    return CALIBRATION_METHODS.get(method, method)


def _draw_reliability_curve(
    truth: np.ndarray, probability: np.ndarray, method: str, ax: Axes, **style: Any
) -> None:
    """Draw one method's reliability curve on ax, with the bins of the calibration settings."""
    CalibrationDisplay.from_predictions(
        truth,
        probability,
        n_bins=CALIBRATION_BINS,
        strategy=CALIBRATION_STRATEGY,
        name=_method_name(method),
        ax=ax,
        **style,
    )


def _draw_histograms(
    figure: Figure,
    grid: GridSpec,
    probabilities: Mapping[str, pd.DataFrame],
    styles: dict[str, dict[str, Any]],
) -> None:
    """Draw each method's histogram below the diagram: two to a row, from the grid's third row."""
    histograms: list[Axes] = []
    for number, (method, frame) in enumerate(probabilities.items()):
        first = histograms[0] if histograms else None
        ax = figure.add_subplot(grid[2 + number // 2, number % 2], sharey=first)
        ax.hist(
            frame.to_numpy().ravel(),
            range=(0, 1),
            bins=CALIBRATION_BINS,
            log=True,
            color=styles[method]["color"],
        )
        ax.set(title=_method_name(method), xlabel="Predicted probability", ylabel="Count")
        histograms.append(ax)
    histograms[0].set_ylim(bottom=1)  # one baseline: a bar's length is the log of its count


def _draw_tag_diagram(
    ax: Axes,
    tag: str,
    labels: pd.DataFrame,
    probabilities: Mapping[str, pd.DataFrame],
    styles: dict[str, dict[str, Any]],
) -> None:
    """Draw every method's reliability curve for one tag on ax, small and square, unlabelled."""
    truth = labels[tag].to_numpy()
    for method, frame in probabilities.items():
        style = styles[method] | RELIABILITY_GRID_LINES
        _draw_reliability_curve(truth, frame[tag].to_numpy(), method, ax, **style)
    # CalibrationDisplay.plot gave this axes a legend and two long labels: the figure has them.
    ax.get_legend().remove()
    ax.set(title=tag, xlabel="", ylabel="")
    ax.set_box_aspect(1)


def _label_reliability_grid(figure: Figure, ax: Axes) -> None:
    """Give the figure one legend, of the curves drawn on ax, and one label for each axis."""
    handles, names = ax.get_legend_handles_labels()
    figure.legend(handles, names, loc="outside upper center", ncols=len(names), frameon=False)
    # Constrained layout leaves no room for a label wider than its axes, so a long label under
    # each of four narrow diagrams ran off the figure.
    xlabel, ylabel = RELIABILITY_LABELS
    figure.supxlabel(xlabel)
    figure.supylabel(ylabel)
