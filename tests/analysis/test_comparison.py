"""Tests for comparing tagging methods on the same problems."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from codeforces_tags.analysis.comparison import (
    AGREEMENT_COLUMNS,
    DELTA_COLUMNS,
    SCORE_COLUMNS,
    SLICE_COLUMNS,
    agreement,
    method_scores,
    slice_differences,
)
from codeforces_tags.analysis.plotting import plot_method_f1
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.report import ALL_PROBLEMS, compare, evaluate
from codeforces_tags.methods.label_prior.method import LabelPrior

TAGS = ["math", "graphs"]


@pytest.fixture(name="truth")
def truth_fixture():
    """Return 100 problems' true tags: a frequent and a rarer tag."""
    values = np.random.default_rng(0).random((100, 2)) < [0.4, 0.2]
    return pd.DataFrame(values.astype(int), pd.Index(range(100), name="sample_id"), TAGS)


@pytest.fixture(name="groups")
def groups_fixture(truth):
    """Pair consecutive problems: 50 near-duplicate groups of two."""
    return pd.Series(truth.index // 2, index=truth.index)


@pytest.fixture(name="predictions")
def predictions_fixture(truth):
    """Return two methods: one predicts every tag, one gets each value wrong 10% of the time."""
    wrong = np.random.default_rng(1).random(truth.shape) < 0.1
    return {
        "every tag": pd.DataFrame(1, index=truth.index, columns=TAGS),
        "noisy": truth.where(~wrong, 1 - truth),
    }


def test_method_scores_gives_each_methods_f1_with_its_interval(truth, predictions, groups):
    scores = method_scores(truth, predictions, groups)

    assert scores.index.names == ["method", "metric"]
    assert list(scores.columns) == list(SCORE_COLUMNS) == ["f1", "f1_low", "f1_high"]
    assert scores.index.tolist() == [
        (method, metric)
        for method in ("every tag", "noisy")
        for metric in ("macro_f1", "micro_f1", "math", "graphs")
    ]


def test_method_scores_are_evaluates_estimates_and_intervals(truth, predictions, groups):
    noisy = method_scores(truth, predictions, groups).loc["noisy"]

    evaluation = evaluate(truth, predictions["noisy"], groups=groups)
    summary = evaluation.summary.loc[ALL_PROBLEMS]
    for metric in ("macro_f1", "micro_f1"):
        expected = [summary[metric], summary[f"{metric}_low"], summary[f"{metric}_high"]]
        assert noisy.loc[metric].tolist() == expected
    per_tag = evaluation.per_tag.loc[ALL_PROBLEMS, ["f1", "f1_low", "f1_high"]]
    assert noisy.loc[TAGS].to_numpy().tolist() == per_tag.to_numpy().tolist()


def test_method_scores_with_a_baseline_adds_the_paired_difference(truth, predictions, groups):
    scores = method_scores(truth, predictions, groups, baseline="every tag")

    assert list(scores.columns) == [*SCORE_COLUMNS, *DELTA_COLUMNS]
    expected = compare(truth, predictions["every tag"], predictions["noisy"], groups)
    delta = scores.loc["noisy", list(DELTA_COLUMNS)]
    assert delta.to_numpy().tolist() == expected[list(DELTA_COLUMNS)].to_numpy().tolist()
    assert (scores.loc["every tag", list(DELTA_COLUMNS)] == 0).all().all()


def test_method_scores_refuses_no_method_or_an_unknown_baseline(truth, predictions, groups):
    with pytest.raises(ValueError, match="no method to score"):
        method_scores(truth, {}, groups)
    with pytest.raises(ValueError, match="baseline 'tfidf' is not one of"):
        method_scores(truth, predictions, groups, baseline="tfidf")


def test_method_scores_names_a_method_not_aligned_on_the_truth(truth, predictions, groups):
    predictions["noisy"] = predictions["noisy"].iloc[1:]

    with pytest.raises(ValueError, match="noisy must hold 0/1 tags on truth's problems"):
        method_scores(truth, predictions, groups)


def test_slice_differences_compares_each_method_with_the_baseline_per_slice(
    truth, predictions, groups
):
    predictions["also noisy"] = predictions["noisy"].iloc[::-1].set_axis(truth.index)
    early = truth.index < 60
    slices = {"early": early, "late": ~early}

    table = slice_differences(truth, predictions, groups, baseline="every tag", slices=slices)

    assert list(table.columns) == list(SLICE_COLUMNS)
    assert table.index.names == ["slice", "method"]
    assert table.index.tolist() == [
        (name, method) for name in ("early", "late") for method in ("noisy", "also noisy")
    ]
    assert table["problems"].tolist() == [60, 60, 40, 40]
    for (name, method), row in table.iterrows():
        mask = slices[name]
        expected = compare(
            truth[mask], predictions["every tag"][mask], predictions[method][mask], groups[mask]
        ).loc["macro_f1", list(SLICE_COLUMNS[1:])]
        assert row[list(SLICE_COLUMNS[1:])].tolist() == expected.tolist()


def test_slice_differences_reports_the_row_asked_for(truth, predictions, groups):
    slices = {"all": np.ones(len(truth), dtype=bool)}

    table = slice_differences(
        truth, predictions, groups, baseline="every tag", slices=slices, row="graphs"
    )

    expected = compare(truth, predictions["every tag"], predictions["noisy"], groups)
    assert table.loc[("all", "noisy"), "candidate"] == expected.at["graphs", "candidate"]
    assert table.loc[("all", "noisy"), "problems"] == len(truth)


def test_slice_differences_refuses_a_baseline_it_cannot_compare_with(truth, predictions, groups):
    slices = {"all": np.ones(len(truth), dtype=bool)}

    with pytest.raises(ValueError, match="baseline 'tfidf' is not one of"):
        slice_differences(truth, predictions, groups, baseline="tfidf", slices=slices)
    with pytest.raises(ValueError, match="is the only method"):
        alone = {"noisy": predictions["noisy"]}
        slice_differences(truth, alone, groups, baseline="noisy", slices=slices)


def test_slice_differences_refuses_misaligned_predictions_an_unknown_row_or_no_slice(
    truth, predictions, groups
):
    slices = {"all": np.ones(len(truth), dtype=bool)}

    with pytest.raises(ValueError, match="slices is empty"):
        slice_differences(truth, predictions, groups, baseline="noisy", slices={})
    with pytest.raises(ValueError, match="row must be one of"):
        slice_differences(truth, predictions, groups, baseline="noisy", slices=slices, row="trees")
    predictions["noisy"] = predictions["noisy"].iloc[1:]
    with pytest.raises(ValueError, match="noisy must hold 0/1 tags on truth's problems"):
        slice_differences(truth, predictions, groups, baseline="every tag", slices=slices)


@pytest.mark.parametrize(
    "mask",
    [np.zeros(100, dtype=bool), np.ones(99, dtype=bool), np.ones(100, dtype=int)],
    ids=["selects nothing", "wrong length", "not boolean"],
)
def test_slice_differences_refuses_a_slice_that_is_not_a_mask_of_problems(
    truth, predictions, groups, mask
):
    with pytest.raises(ValueError, match="slice 'odd' must be a boolean mask of the 100 problems"):
        slice_differences(truth, predictions, groups, baseline="noisy", slices={"odd": mask})


def test_agreement_counts_who_is_right_per_tag():
    truth = pd.DataFrame({"math": [1, 1, 0, 0, 1], "graphs": [0, 1, 0, 1, 0]})
    a = pd.DataFrame({"math": [1, 0, 0, 1, 1], "graphs": [0, 1, 1, 1, 0]})
    b = pd.DataFrame({"math": [1, 1, 1, 0, 0], "graphs": [1, 1, 1, 0, 0]})

    counts = agreement(truth, a, b)

    assert list(counts.columns) == list(AGREEMENT_COLUMNS)
    assert counts.index.name == "tag"
    assert counts.to_dict("index") == {
        "math": {"both_right": 1, "only_a_right": 2, "only_b_right": 2, "both_wrong": 0},
        "graphs": {"both_right": 2, "only_a_right": 2, "only_b_right": 0, "both_wrong": 1},
    }
    assert (counts.sum(axis=1) == len(truth)).all()


def test_agreement_refuses_predictions_not_aligned_on_the_truth(truth, predictions):
    noisy = predictions["noisy"]

    with pytest.raises(ValueError, match="a must hold 0/1 tags on truth's problems and tags"):
        agreement(truth, noisy.iloc[1:], noisy)
    with pytest.raises(ValueError, match="a must hold 0/1 tags on truth's problems and tags"):
        agreement(truth, noisy.iloc[:, ::-1], noisy)  # its tags in another order
    with pytest.raises(ValueError, match="b must hold 0/1 tags"):
        agreement(truth, noisy, noisy.replace(1, 2))
    with pytest.raises(ValueError, match="truth must hold 0/1 tags"):
        agreement(truth.replace(1, 2), noisy, noisy)


def test_method_scores_gives_label_prior_the_f1_of_a_constant_prediction():
    # label_prior predicts every focus tag with a positive for every problem and reads no
    # column, so its F1 on a tag is 2p / (1 + p), p being the tag's share of positives.
    values = np.random.default_rng(2).random((100, len(FOCUS_TAGS))) < 0.3
    index = pd.Index(range(100), name="sample_id")
    truth = pd.DataFrame(values.astype(int), index, list(FOCUS_TAGS))
    raw = pd.DataFrame(index=index)
    predicted = LabelPrior().fit(raw, truth).predict(raw)

    scores = method_scores(truth, {"label_prior": predicted}, pd.Series(index // 2, index=index))

    share = truth.mean()
    f1 = scores.loc["label_prior"].loc[list(FOCUS_TAGS), "f1"]
    assert f1.tolist() == pytest.approx((2 * share / (1 + share)).tolist())
    ax = plot_method_f1(scores)
    (line,) = ax.lines
    assert line.get_label() == "label_prior"
    assert line.get_xdata().tolist() == scores["f1"].tolist()
    plt.close(ax.figure)
