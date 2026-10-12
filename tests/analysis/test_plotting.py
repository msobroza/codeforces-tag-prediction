"""Tests for the shared notebook plot style and plots."""

import io
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.transforms import Bbox
from sklearn.calibration import calibration_curve

from codeforces_tags.analysis.calibration import reliability_table
from codeforces_tags.analysis.plotting import (
    boxplot_by_label,
    focus_palette,
    plot_confusion_grid,
    plot_f1_curve,
    plot_intervals,
    plot_latency,
    plot_method_f1,
    plot_reliability,
    plot_reliability_grid,
    plot_reliability_table,
    plot_search_progress,
    plot_study_history,
    plot_tag_confusion,
    setup_style,
)
from codeforces_tags.config import (
    CALIBRATION_BINS,
    CALIBRATION_METHODS,
    CALIBRATION_STRATEGY,
    FOCUS_COLOR,
    FOCUS_TAGS,
    LATENCY_BUDGET_S,
    METHODS_ASSETS_DIR,
    OTHER_COLOR,
)
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.evaluation.timing import load_latency


@pytest.fixture(name="close_figures", autouse=True)
def close_figures_fixture():
    """Close every figure a test opened, even when one of its assertions fails."""
    yield
    plt.close("all")


def test_focus_palette_highlights_only_focus_tags():
    assert focus_palette(["math", "greedy", "trees"]) == [FOCUS_COLOR, OTHER_COLOR, FOCUS_COLOR]


def test_setup_style_sets_the_shared_figure_defaults():
    setup_style()

    assert tuple(plt.rcParams["figure.figsize"]) == (10, 4.5)
    assert plt.rcParams["axes.titleweight"] == "bold"


@pytest.mark.parametrize(("log", "scale"), [(True, "log"), (False, "linear")])
def test_boxplot_by_label_orders_boxes_by_median(log, scale):
    indicators = multi_hot(pd.Series([["a"], ["a"], ["b"], []]), vocabulary=["a", "b"])
    values = pd.Series([10.0, 30.0, 5.0, 50.0])

    ax = boxplot_by_label(values, indicators, log=log, none_label="none")

    box_colors = [to_hex(patch.get_facecolor()) for patch in ax.patches]

    assert [label.get_text() for label in ax.get_yticklabels()] == ["b", "a", "none"]
    assert ax.get_xscale() == scale
    assert box_colors[-1] == OTHER_COLOR != box_colors[0]


def test_plot_intervals_draws_a_point_and_an_interval_per_row_from_top_to_bottom():
    table = pd.DataFrame(
        {"f1": [0.8, 0.5], "low": [0.7, 0.3], "high": [0.9, 0.6]}, index=["math", "graphs"]
    )

    ax = plot_intervals(table, "f1", "low", "high")

    (points,) = ax.lines
    intervals = [segment.tolist() for segment in ax.collections[0].get_segments()]
    assert points.get_xdata().tolist() == [0.8, 0.5]
    assert intervals == [[[0.7, 0.0], [0.9, 0.0]], [[0.3, 1.0], [0.6, 1.0]]]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["math", "graphs"]
    assert ax.yaxis_inverted()  # row 0 at the top


def test_plot_intervals_draws_an_interval_that_excludes_its_estimate_as_given():
    table = pd.DataFrame({"f1": [0.6], "low": [0.65], "high": [0.7]}, index=["games"])

    ax = plot_intervals(table, "f1", "low", "high")

    assert ax.collections[0].get_segments()[0][:, 0].tolist() == [0.65, 0.7]


def test_plot_intervals_draws_on_the_given_axes():
    _, given = plt.subplots()
    table = pd.DataFrame({"delta": [0.02], "low": [-0.01], "high": [0.04]}, index=["code"])

    ax = plot_intervals(table, "delta", "low", "high", ax=given)

    assert ax is given


def test_plot_tag_confusion_annotates_each_share_on_a_zero_to_one_scale():
    matrix = pd.DataFrame(
        [[0.75, 0.0, 0.25], [0.1, 0.5, 0.4], [0.05, 0.0, 0.95]],
        index=["math", "graphs", "no focus tag"],
        columns=["math", "graphs", "nothing predicted"],
    )

    ax = plot_tag_confusion(matrix)

    annotations = [text.get_text() for text in ax.texts]
    assert annotations == ["75%", "0%", "25%", "10%", "50%", "40%", "5%", "0%", "95%"]
    assert ax.collections[0].get_clim() == (0, 1)
    assert [label.get_text() for label in ax.get_yticklabels()] == list(matrix.index)
    assert [label.get_text() for label in ax.get_xticklabels()] == list(matrix.columns)


@pytest.fixture(name="per_tag")
def per_tag_fixture():
    """Return the counts of three tags; probabilities has no positive problem."""
    return pd.DataFrame(
        {"tp": [30, 2, 0], "fp": [10, 1, 0], "fn": [5, 3, 0], "tn": [55, 94, 100]},
        index=pd.Index(["math", "games", "probabilities"], name="tag"),
    )


def test_plot_confusion_grid_draws_tp_fn_fp_tn_coloured_by_row_share(per_tag):
    figure = plot_confusion_grid(per_tag)

    math_ax, _, probabilities_ax = figure.axes
    assert [text.get_text() for text in math_ax.texts] == ["TP\n30", "FN\n5", "FP\n10", "TN\n55"]
    assert math_ax.collections[0].get_array().tolist() == [[30 / 35, 5 / 35], [10 / 65, 55 / 65]]
    assert probabilities_ax.collections[0].get_array().tolist() == [[0, 0], [0, 1]]


def test_plot_confusion_grid_puts_one_titled_heatmap_per_tag_in_two_rows(per_tag):
    figure = plot_confusion_grid(per_tag)

    places = [
        (ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)
        for ax in figure.axes
    ]
    assert [ax.get_title() for ax in figure.axes] == ["math", "games", "probabilities"]
    assert places == [(0, 0), (0, 1), (1, 0)]  # the grid's fourth place is left empty


def test_plot_study_history_shows_complete_trials_and_the_best_value_so_far():
    trials = pd.DataFrame(
        {
            "number": [3, 0, 2, 1, 4],
            "state": ["COMPLETE", "COMPLETE", "FAIL", "COMPLETE", "COMPLETE"],
            "value": [0.6, 0.5, np.nan, 0.4, 0.55],
        }
    )

    ax = plot_study_history(trials)

    (best,) = ax.lines
    assert ax.collections[0].get_offsets().tolist() == [[0, 0.5], [1, 0.4], [3, 0.6], [4, 0.55]]
    assert best.get_xdata().tolist() == [0, 1, 3, 4]
    assert best.get_ydata().tolist() == [0.5, 0.5, 0.6, 0.6]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["trial", "best so far"]


@pytest.fixture(name="scores")
def scores_fixture():
    """Return two methods' F1 and intervals on two metrics, as method_scores gives them."""
    return pd.DataFrame(
        {
            "f1": [0.6, 0.8, 0.5, 0.7],
            "f1_low": [0.5, 0.7, 0.4, 0.6],
            "f1_high": [0.7, 0.9, 0.6, 0.8],
        },
        index=pd.MultiIndex.from_product(
            [["label_prior", "tfidf"], ["macro_f1", "math"]], names=["method", "metric"]
        ),
    )


def test_plot_method_f1_draws_each_methods_points_side_by_side_in_each_metrics_row(scores):
    ax = plot_method_f1(scores)

    prior, tfidf = ax.lines
    assert prior.get_xdata().tolist() == [0.6, 0.8]
    assert prior.get_ydata().tolist() == pytest.approx([-0.2, 0.8])
    assert tfidf.get_ydata().tolist() == pytest.approx([0.2, 1.2])
    assert prior.get_color() != tfidf.get_color()
    assert [label.get_text() for label in ax.get_yticklabels()] == ["macro_f1", "math"]
    assert ax.yaxis_inverted()  # the first metric at the top
    assert ax.get_xlim() == (0, 1)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["label_prior", "tfidf"]


def test_plot_method_f1_draws_each_interval_on_its_methods_row(scores):
    ax = plot_method_f1(scores)

    tfidf_intervals = ax.collections[1].get_segments()
    assert [segment[:, 0].tolist() for segment in tfidf_intervals] == [[0.4, 0.6], [0.6, 0.8]]
    assert [segment[0, 1] for segment in tfidf_intervals] == pytest.approx([0.2, 1.2])


def test_plot_method_f1_centres_a_single_method_on_each_row(scores):
    ax = plot_method_f1(scores.loc[["tfidf"]])

    (tfidf,) = ax.lines
    assert tfidf.get_ydata().tolist() == [0.0, 1.0]


def test_plot_latency_draws_each_problem_and_median_on_a_log_scale_against_the_budget():
    latencies = {"label_prior": pd.Series([0.01, 0.02, 0.03]), "slow": pd.Series([2.0, 12.0])}

    ax = plot_latency(latencies)

    problems, medians = ax.collections
    assert problems.get_offsets().tolist() == [
        [0.01, 0],
        [0.02, 0],
        [0.03, 0],
        [2.0, 1],
        [12.0, 1],
    ]
    assert medians.get_offsets().tolist() == [[0.02, 0], [7.0, 1]]
    (budget,) = ax.lines
    assert list(budget.get_xdata()) == [LATENCY_BUDGET_S] * 2
    assert ax.get_xscale() == "log"
    assert [label.get_text() for label in ax.get_yticklabels()] == ["label_prior", "slow"]
    assert ax.yaxis_inverted()  # the first method at the top
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["problem", "median", "10 s budget"]


def test_plot_latency_draws_the_committed_latency_of_label_prior():
    # Needs no dataset: tracking latency committed label_prior's latency.csv.
    seconds = load_latency(METHODS_ASSETS_DIR / "label_prior").seconds

    ax = plot_latency({"label_prior": seconds})

    problems, medians = ax.collections
    assert len(problems.get_offsets()) == 50
    assert problems.get_offsets()[:, 0].tolist() == seconds.tolist()
    assert medians.get_offsets().tolist() == [[seconds.median(), 0]]
    assert ax.get_xscale() == "log"
    assert [label.get_text() for label in ax.get_yticklabels()] == ["label_prior"]


REFERENCE = "Perfectly calibrated"  # the label CalibrationDisplay gives the dotted diagonal
METHOD_NAMES = list(CALIBRATION_METHODS.values())


def curve_by_hand(truth, probability):
    """Return a reliability curve's x and y as scikit-learn bins them with the settings."""
    prob_true, prob_pred = calibration_curve(
        truth, probability, n_bins=CALIBRATION_BINS, strategy=CALIBRATION_STRATEGY
    )
    return prob_pred.tolist(), prob_true.tolist()


def curve_points(line):
    """Return a drawn curve's x and y as lists."""
    return line.get_xdata().tolist(), line.get_ydata().tolist()


def curve_styles(ax):
    """Return the colour and marker of each method's curve on ax, after the reference line."""
    return [(to_hex(line.get_color()), line.get_marker()) for line in ax.lines[1:]]


def grid_places(figure):
    """Return the row and column in their grid of each of the figure's axes."""
    return [
        (ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)
        for ax in figure.axes
    ]


def test_plot_reliability_draws_one_curve_per_method_over_all_problem_tag_pairs(
    calibration_data, probabilities
):
    labels = calibration_data.labels

    figure = plot_reliability(probabilities, labels)

    diagram = figure.axes[0]
    reference, *curves = diagram.lines
    assert reference.get_label() == REFERENCE
    assert [curve.get_label() for curve in curves] == METHOD_NAMES
    truth = labels.to_numpy().ravel()
    for curve, frame in zip(curves, probabilities.values(), strict=True):
        assert curve_points(curve) == curve_by_hand(truth, frame.to_numpy().ravel())
    assert len(curves[0].get_xdata()) == CALIBRATION_BINS  # the logits' sigmoids never tie
    colours, markers = zip(*curve_styles(diagram), strict=True)
    assert len(set(colours)) == len(set(markers)) == len(curves)
    assert [text.get_text() for text in diagram.get_legend().get_texts()] == [
        REFERENCE,
        *METHOD_NAMES,
    ]


def test_plot_reliability_puts_a_log_histogram_of_each_methods_probabilities_under_the_diagram(
    calibration_data, probabilities
):
    figure = plot_reliability(probabilities, calibration_data.labels)

    diagram, *histograms = figure.axes
    assert figure.get_size_inches().tolist() == [10, 10]
    assert diagram.get_gridspec().get_geometry() == (4, 2)
    assert diagram.get_subplotspec().rowspan == range(2)
    assert diagram.get_subplotspec().colspan == range(2)
    assert grid_places(figure)[1:] == [(2, 0), (2, 1), (3, 0), (3, 1)]
    assert [ax.get_title() for ax in histograms] == METHOD_NAMES
    curves = diagram.lines[1:]
    for ax, frame, curve in zip(histograms, probabilities.values(), curves, strict=True):
        counts, edges = np.histogram(frame.to_numpy().ravel(), bins=CALIBRATION_BINS, range=(0, 1))
        assert [patch.get_height() for patch in ax.patches] == counts.tolist()
        assert [patch.get_x() for patch in ax.patches] == pytest.approx(edges[:-1])
        assert to_hex(ax.patches[0].get_facecolor()) == to_hex(curve.get_color())
        assert ax.get_yscale() == "log"
        assert ax.get_ylim() == histograms[0].get_ylim()  # one count axis for the four methods
    assert histograms[0].get_ylim()[0] == 1  # and one baseline, so a bar's length is log10(count)


def test_plot_reliability_names_a_method_outside_the_settings_by_its_id(
    calibration_data, probabilities
):
    figure = plot_reliability({"my_method": probabilities["svm_platt"]}, calibration_data.labels)

    diagram, histogram = figure.axes
    assert [line.get_label() for line in diagram.lines] == [REFERENCE, "my_method"]
    assert histogram.get_title() == "my_method"
    assert grid_places(figure) == [(0, 0), (2, 0)]  # a lone histogram takes the first place


def test_plot_reliability_grid_puts_one_titled_diagram_per_tag_in_two_rows(
    calibration_data, probabilities
):
    figure = plot_reliability_grid(probabilities, calibration_data.labels)

    assert [ax.get_title() for ax in figure.axes] == list(FOCUS_TAGS)
    assert grid_places(figure) == [(row, column) for row in range(2) for column in range(4)]
    # Square diagrams: the diagonal of a calibrated method is at 45 degrees whatever the size.
    assert all(ax.get_box_aspect() == 1 for ax in figure.axes)
    assert figure.get_size_inches().tolist() == [10, 5.6]


def test_plot_reliability_grid_leaves_the_places_it_does_not_need_empty(
    calibration_data, probabilities
):
    tags = ["math", "trees", "games"]
    probabilities = {method: frame[tags] for method, frame in probabilities.items()}

    figure = plot_reliability_grid(probabilities, calibration_data.labels[tags])

    assert [ax.get_title() for ax in figure.axes] == tags
    assert grid_places(figure) == [(0, 0), (0, 1), (1, 0)]  # the grid's fourth place is empty


def test_plot_reliability_grid_draws_every_methods_curve_of_the_tag_in_each_diagram(
    calibration_data, probabilities
):
    labels = calibration_data.labels

    figure = plot_reliability_grid(probabilities, labels)

    for ax, tag in zip(figure.axes, labels.columns, strict=True):
        reference, *curves = ax.lines
        assert reference.get_label() == REFERENCE
        assert [curve.get_label() for curve in curves] == METHOD_NAMES
        for curve, frame in zip(curves, probabilities.values(), strict=True):
            assert curve_points(curve) == curve_by_hand(labels[tag].to_numpy(), frame[tag])
        assert curve_styles(ax) == curve_styles(figure.axes[0])


def test_the_two_reliability_figures_give_a_method_the_same_colour_and_marker(
    calibration_data, probabilities
):
    labels = calibration_data.labels

    pooled = plot_reliability(probabilities, labels).axes[0]
    by_tag = plot_reliability_grid(probabilities, labels)

    assert curve_styles(by_tag.axes[0]) == curve_styles(pooled)


def test_plot_reliability_grid_has_one_figure_legend_and_none_on_a_diagram(
    calibration_data, probabilities
):
    # CalibrationDisplay.plot adds a legend of its own to every axes it draws on: eight legends of
    # five entries hid the curves.
    figure = plot_reliability_grid(probabilities, calibration_data.labels)

    (legend,) = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [REFERENCE, *METHOD_NAMES]
    assert [ax.get_legend() for ax in figure.axes] == [None] * len(FOCUS_TAGS)


def test_plot_reliability_grid_labels_the_figure_once_and_not_each_diagram(
    calibration_data, probabilities
):
    # CalibrationDisplay.plot labels every axes "Mean predicted probability (Positive class: 1)".
    figure = plot_reliability_grid(probabilities, calibration_data.labels)

    assert figure.get_supxlabel() == "Mean predicted probability"
    assert figure.get_supylabel() == "Fraction of positives"
    assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in figure.axes] == [("", "")] * 8


def test_plot_reliability_grid_keeps_the_legend_and_the_figure_labels_clear_of_the_diagrams(
    calibration_data, probabilities
):
    # Constrained layout ignores how far an x label reaches along its axis: a label under each of
    # four narrow diagrams ran off the figure. A legend under the figure's x label hid it. The
    # notebooks' style has the larger fonts, at which that label overflows.
    with plt.rc_context():
        setup_style()
        figure = plot_reliability_grid(probabilities, calibration_data.labels)
        figure.canvas.draw()
    drawn, (width, height) = figure.get_tightbbox(), figure.get_size_inches()
    assert drawn.x0 >= 0 and drawn.x1 <= width  # nothing, a label or the legend, is cut off
    assert drawn.y0 >= 0 and drawn.y1 <= height
    diagrams = Bbox.union([ax.get_tightbbox() for ax in figure.axes])
    legend = figure.legends[0].get_window_extent()
    entries = figure.legends[0].get_texts()
    assert len({round(entry.get_window_extent().y0) for entry in entries}) == 1  # in one row
    texts = {text.get_text(): text.get_window_extent() for text in figure.texts}
    xlabel, ylabel = texts["Mean predicted probability"], texts["Fraction of positives"]
    assert legend.y0 >= diagrams.y1
    assert xlabel.y1 <= diagrams.y0
    assert ylabel.x1 <= diagrams.x0


def test_plot_reliability_keeps_the_diagram_the_histograms_and_their_labels_clear_of_each_other(
    calibration_data, probabilities
):
    # Without constrained layout, each histogram's title printed over the tick labels of the plot
    # above it, and the first histogram hid the diagram's x label. The notebooks' style has the
    # larger fonts.
    with plt.rc_context():
        setup_style()
        figure = plot_reliability(probabilities, calibration_data.labels)
        figure.canvas.draw()
    drawn, (width, height) = figure.get_tightbbox(), figure.get_size_inches()
    assert drawn.x0 >= 0 and drawn.x1 <= width  # nothing is cut off
    assert drawn.y0 >= 0 and drawn.y1 <= height
    boxes = [ax.get_tightbbox() for ax in figure.axes]  # each axes with its title and labels
    assert not [pair for pair in itertools.combinations(boxes, 2) if pair[0].overlaps(pair[1])]
    diagram, *histograms = figure.axes
    assert (diagram.get_title(), diagram.get_xlabel(), diagram.get_ylabel()) == (
        "Reliability diagram: all problem-tag pairs",
        "Mean predicted probability",
        "Fraction of positives",
    )
    assert [(ax.get_xlabel(), ax.get_ylabel()) for ax in histograms] == [
        ("Predicted probability", "Count")
    ] * len(probabilities)


@pytest.mark.parametrize("plot", [plot_reliability, plot_reliability_grid])
def test_the_reliability_figures_render_as_png(calibration_data, probabilities, plot):
    image = io.BytesIO()

    plot(probabilities, calibration_data.labels).savefig(image, format="png")

    assert image.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize("plot", [plot_reliability, plot_reliability_grid])
def test_the_reliability_figures_refuse_probabilities_outside_zero_and_one_before_drawing(
    calibration_data, probabilities, plot
):
    probabilities["svm_platt"] = probabilities["svm_platt"] * 1.5

    with pytest.raises(ValueError, match=r"The probabilities of 'svm_platt' lie outside \[0, 1\]"):
        plot(probabilities, calibration_data.labels)

    assert not plt.get_fignums()


@pytest.fixture(name="reliability")
def reliability_fixture():
    """Return the reliability table of three tags: two bins for math and graphs, one for games."""
    index = pd.MultiIndex.from_tuples(
        [("math", 0), ("math", 3), ("graphs", 1), ("graphs", 3), ("games", 2)],
        names=["tag", "bin"],
    )
    return pd.DataFrame(
        {
            "mean_probability": [0.1, 0.9, 0.2, 0.8, 0.6],
            "positive_rate": [0.2, 0.7, 0.1, 0.9, 0.5],
            "problems": [10, 5, 8, 7, 3],
        },
        index=index,
    )


def test_plot_reliability_table_draws_one_titled_panel_per_tag_with_the_diagonal_first(reliability):
    figure = plot_reliability_table(reliability)

    assert [ax.get_title() for ax in figure.axes] == ["math", "graphs", "games"]
    diagonal, curve = figure.axes[0].lines
    assert diagonal.get_linestyle() == "--"
    assert list(diagonal.get_xdata()) == list(diagonal.get_ydata()) == [0, 1]
    assert curve.get_marker() == "o"
    assert list(curve.get_xdata()) == [0.1, 0.9]
    assert list(curve.get_ydata()) == [0.2, 0.7]
    _, graphs_curve = figure.axes[1].lines
    assert list(graphs_curve.get_xdata()) == [0.2, 0.8]
    assert list(graphs_curve.get_ydata()) == [0.1, 0.9]
    assert figure.get_supxlabel() == "mean probability in the bin"
    assert figure.get_supylabel() == "share of problems with the tag"


def test_plot_reliability_table_shares_axes_from_0_to_1_between_the_panels(reliability):
    figure = plot_reliability_table(reliability)

    first = figure.axes[0]
    assert all(ax.get_xlim() == ax.get_ylim() == (0, 1) for ax in figure.axes)
    assert all(first.get_shared_x_axes().joined(first, ax) for ax in figure.axes)
    assert all(first.get_shared_y_axes().joined(first, ax) for ax in figure.axes)


def test_plot_reliability_table_removes_the_unused_panels(reliability):
    figure = plot_reliability_table(reliability)

    assert len(figure.axes) == 3  # the fourth place of the 2 by 2 grid is left empty
    assert figure.axes[0].get_gridspec().get_geometry() == (2, 2)


def test_plot_reliability_table_puts_the_eight_focus_tags_in_two_rows_of_four():
    bins = pd.DataFrame(
        {"mean_probability": [0.2, 0.8], "positive_rate": [0.1, 0.9], "problems": [5, 5]},
        index=pd.Index([0, 3], name="bin"),
    )
    table = pd.concat({tag: bins for tag in FOCUS_TAGS}, names=["tag", "bin"])

    figure = plot_reliability_table(table)

    assert [ax.get_title() for ax in figure.axes] == list(FOCUS_TAGS)
    assert figure.axes[0].get_gridspec().get_geometry() == (2, 4)


def test_plot_reliability_table_draws_the_table_that_reliability_table_gives():
    scores = pd.DataFrame({"math": [0.0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]})
    truth = pd.DataFrame({"math": [0, 0, 1, 0, 1, 1, 1, 0]})

    figure = plot_reliability_table(reliability_table(scores, truth, bins=4))

    (math_ax,) = figure.axes
    curve = math_ax.lines[1]
    assert math_ax.get_title() == "math"
    assert list(curve.get_xdata()) == pytest.approx([0.0625, 0.25, 0.5625, 0.875])
    assert list(curve.get_ydata()) == pytest.approx([0.0, 1.0, 0.5, 2 / 3])


@pytest.fixture(name="progress")
def progress_fixture():
    """Return three iterations of a search: the second accepted a child."""
    return pd.DataFrame(
        {
            "best_selection_mean": [0.70, 0.72, 0.72],
            "accepted": [False, True, False],
            "jev_cost_usd": [0.1, 0.3, 0.3],
            "luna_cost_usd": [0.01, 0.02, 0.03],
            "total_cost_usd": [0.11, 0.32, 0.33],
        },
        index=pd.Index([1, 2, 3], name="iteration"),
    )


def test_plot_search_progress_draws_the_best_score_and_marks_the_accepted_children(progress):
    figure = plot_search_progress(progress)

    score, cost = figure.axes
    (best,) = score.get_lines()
    (accepted,) = score.collections

    assert best.get_drawstyle() == "steps-post"
    assert best.get_xdata().tolist() == [1, 2, 3]
    assert best.get_ydata().tolist() == [0.70, 0.72, 0.72]
    assert to_hex(best.get_color()) == FOCUS_COLOR
    assert accepted.get_offsets().tolist() == [[2.0, 0.72]]
    assert to_hex(accepted.get_facecolor()[0]) == OTHER_COLOR
    assert score.get_shared_x_axes().joined(score, cost)


def test_plot_search_progress_draws_the_cumulative_cost_of_jev_luna_and_both(progress):
    figure = plot_search_progress(progress)

    cost = figure.axes[1]

    assert [line.get_label() for line in cost.get_lines()] == ["Jev", "Luna", "total"]
    assert [line.get_ydata().tolist() for line in cost.get_lines()] == [
        [0.1, 0.3, 0.3],
        [0.01, 0.02, 0.03],
        [0.11, 0.32, 0.33],
    ]
    assert cost.get_xlabel() == "iteration"
    assert tuple(figure.get_size_inches()) == (10, 6)


@pytest.fixture(name="f1_curve_data")
def f1_curve_data_fixture():
    """Return a tag's F1 curve on four cuts, and the curves of two folds' other folds."""
    cuts = pd.Index([0.125, 0.375, 0.625, 0.875], name="threshold")
    curve = pd.Series([0.4, 0.7, 0.68, 0.3], index=cuts, name="probabilities")
    folds = pd.DataFrame(
        {0: [0.5, 0.6, 0.69, 0.2], 1: [0.3, 0.72, 0.6, 0.35]},
        index=cuts,
    ).rename_axis(columns="fold")
    return curve, folds


def test_plot_f1_curve_draws_the_curve_then_a_dashed_line_at_each_threshold(f1_curve_data):
    curve, _ = f1_curve_data

    figure = plot_f1_curve(curve, {"R0": 0.375, "R2": 0.625})

    (ax,) = figure.axes
    drawn, first, second = ax.get_lines()
    assert drawn.get_xdata().tolist() == [0.125, 0.375, 0.625, 0.875]
    assert drawn.get_ydata().tolist() == [0.4, 0.7, 0.68, 0.3]
    assert to_hex(drawn.get_color()) == FOCUS_COLOR
    assert [line.get_xdata()[0] for line in (first, second)] == [0.375, 0.625]
    assert {first.get_linestyle(), second.get_linestyle()} == {"--"}
    assert first.get_color() != second.get_color()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "all problems",
        "R0 0.375",
        "R2 0.625",
    ]
    assert ax.get_title() == "probabilities"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("threshold", "F1")
    assert ax.get_xlim() == (0, 1)


def test_plot_f1_curve_draws_each_folds_curve_first_thin_and_grey(f1_curve_data):
    curve, folds = f1_curve_data

    figure = plot_f1_curve(curve, {"R0": 0.375}, folds=folds)

    lines = figure.axes[0].get_lines()
    assert [line.get_label() for line in lines] == [
        "without fold 0",
        "without fold 1",
        "all problems",
        "R0 0.375",
    ]
    assert [line.get_ydata().tolist() for line in lines[:2]] == [
        [0.5, 0.6, 0.69, 0.2],
        [0.3, 0.72, 0.6, 0.35],
    ]
    assert {to_hex(line.get_color()) for line in lines[:2]} == {OTHER_COLOR}
    assert lines[0].get_linewidth() < lines[2].get_linewidth()


def test_plot_f1_curve_leaves_out_an_infinite_threshold(f1_curve_data):
    curve, _ = f1_curve_data

    figure = plot_f1_curve(curve, {"R0": math.inf})

    (ax,) = figure.axes
    assert [line.get_label() for line in ax.get_lines()] == ["all problems"]
