"""Tests for the evaluation report: per-tag and averaged metrics, slices and intervals, files.

``report_text``, ``report_headline`` and ``save_report`` show an evaluation as text and save it
as JSON and CSV files.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_recall_fscore_support

from codeforces_tags.config import RELEASE_ERA_FIRST_YEAR
from codeforces_tags.evaluation.report import (
    ALL_PROBLEMS,
    EARLY_ERA,
    FOCUS_TAGGED,
    LATE_ERA,
    NO_FOCUS_TAG,
    NOTHING_PREDICTED,
    REPORT_FILE,
    REPORT_TABLES,
    all_tag_macro_f1,
    compare,
    evaluate,
    report_headline,
    report_text,
    save_report,
)

TAGS = ["math", "graphs", "trees"]


def _frame(rows):
    return pd.DataFrame(rows, columns=TAGS)


def _width(summary, metric):
    return summary[f"{metric}_high"] - summary[f"{metric}_low"]


@pytest.fixture(name="truth")
def truth_fixture():
    """Return 200 problems' true tags: a frequent, a middling and a rare tag; 40% no-focus."""
    values = np.random.default_rng(0).random((200, 3)) < [0.4, 0.25, 0.08]
    return pd.DataFrame(values.astype(int), pd.Index(range(200), name="sample_id"), TAGS)


@pytest.fixture(name="predicted")
def predicted_fixture(truth):
    """Return predictions that get each value of ``truth`` wrong with probability 0.2."""
    wrong = np.random.default_rng(1).random(truth.shape) < 0.2
    return truth.where(~wrong, 1 - truth)


@pytest.fixture(name="years")
def years_fixture(truth):
    """Return release years over 8 years, half of them before the late era."""
    return pd.Series(RELEASE_ERA_FIRST_YEAR - 4 + truth.index % 8, index=truth.index)


@pytest.fixture(name="groups")
def groups_fixture(truth):
    """Pair consecutive problems: 100 near-duplicate groups of two."""
    return pd.Series(truth.index // 2, index=truth.index)


def test_evaluate_matches_sklearn_per_tag(truth, predicted):
    per_tag = evaluate(truth, predicted).per_tag.loc[ALL_PROBLEMS]

    expected = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    assert np.allclose(per_tag[["precision", "recall", "f1"]].T, expected[:3])
    assert per_tag["support"].tolist() == expected[3].tolist()


def test_evaluate_matches_sklearn_macro_and_micro_f1(truth, predicted):
    summary = evaluate(truth, predicted).summary

    for average in ("macro", "micro"):
        expected = f1_score(truth, predicted, average=average, zero_division=0)
        assert summary.at[ALL_PROBLEMS, f"{average}_f1"] == pytest.approx(expected)


def test_evaluate_scores_each_slice_on_its_own_problems(truth, predicted, years):
    evaluation = evaluate(truth, predicted, release_years=years)

    slices = {
        ALL_PROBLEMS: pd.Series(True, index=truth.index),
        FOCUS_TAGGED: truth.any(axis=1),
        EARLY_ERA: years < RELEASE_ERA_FIRST_YEAR,
        LATE_ERA: years >= RELEASE_ERA_FIRST_YEAR,
    }
    assert evaluation.summary.index.tolist() == list(slices)
    for name, rows in slices.items():
        f1 = f1_score(truth[rows], predicted[rows], average=None, zero_division=0)
        assert np.allclose(evaluation.per_tag.loc[name, "f1"], f1)
        assert evaluation.summary.at[name, "macro_f1"] == pytest.approx(f1.mean())
        assert evaluation.summary.at[name, "problems"] == rows.sum()


def test_evaluate_splits_the_release_eras_at_the_first_year():
    truth = _frame([[1, 0, 0]] * 4)
    years = pd.Series([RELEASE_ERA_FIRST_YEAR - 1] + [RELEASE_ERA_FIRST_YEAR] * 3)

    summary = evaluate(truth, truth, release_years=years).summary

    eras = [f"up to {RELEASE_ERA_FIRST_YEAR - 1}", f"from {RELEASE_ERA_FIRST_YEAR}"]
    assert summary.index.tolist() == [ALL_PROBLEMS, FOCUS_TAGGED, *eras]
    assert summary.loc[eras, "problems"].tolist() == [1, 3]


def test_evaluate_puts_a_problem_without_a_release_year_in_neither_era():
    truth = _frame([[1, 0, 0]] * 3)
    years = pd.Series([RELEASE_ERA_FIRST_YEAR - 1, math.nan, RELEASE_ERA_FIRST_YEAR])

    summary = evaluate(truth, truth, release_years=years).summary

    assert summary.at[ALL_PROBLEMS, "problems"] == 3
    assert summary.loc[[EARLY_ERA, LATE_ERA], "problems"].tolist() == [1, 1]


def test_evaluate_reports_an_empty_slice_as_undefined(truth, predicted, groups):
    years = pd.Series(RELEASE_ERA_FIRST_YEAR, index=truth.index)  # no problem in the early era

    evaluation = evaluate(truth, predicted, release_years=years, groups=groups, resamples=200)

    summary, per_tag = evaluation.summary, evaluation.per_tag.loc[EARLY_ERA]
    assert summary.at[EARLY_ERA, "problems"] == 0
    assert summary.loc[EARLY_ERA].drop("problems").isna().all()  # undefined, not 0
    assert summary.loc[LATE_ERA].notna().all()
    assert (per_tag[["support", "tp", "fp", "fn", "tn"]] == 0).all().all()
    assert per_tag[["precision", "recall", "f1", "f1_low", "f1_high"]].isna().all().all()


def test_evaluate_with_groups_reports_a_set_of_no_problem_as_undefined():
    nothing = pd.DataFrame(np.zeros((0, len(TAGS)), dtype=int), columns=TAGS)

    evaluation = evaluate(nothing, nothing, groups=pd.Series([], dtype=int), resamples=50)

    summary = evaluation.summary.loc[ALL_PROBLEMS]
    assert summary["problems"] == 0
    assert summary.drop("problems").isna().all()
    assert evaluation.per_tag[["f1", "f1_low", "f1_high"]].isna().all().all()


def test_evaluate_counts_each_outcome_per_tag():
    truth = _frame([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    predicted = _frame([[1, 1, 0], [0, 1, 0], [0, 0, 0]])

    per_tag = evaluate(truth, predicted).per_tag.loc[ALL_PROBLEMS]

    assert per_tag[["support", "tp", "fp", "fn", "tn"]].to_dict("index") == {
        "math": {"support": 2, "tp": 1, "fp": 0, "fn": 1, "tn": 1},
        "graphs": {"support": 1, "tp": 1, "fp": 1, "fn": 0, "tn": 1},
        "trees": {"support": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 3},
    }


@pytest.mark.filterwarnings("error")
def test_evaluate_scores_an_undefined_ratio_as_zero():
    truth = _frame([[1, 0, 0], [0, 0, 0]])
    predicted = _frame([[0, 0, 1], [0, 0, 0]])
    nothing = _frame([[0, 0, 0], [0, 0, 0]])

    per_tag = evaluate(truth, predicted).per_tag.loc[ALL_PROBLEMS]
    summary = evaluate(nothing, nothing).summary

    # math is never predicted, graphs never true nor predicted, trees never true.
    assert per_tag[["precision", "recall", "f1"]].to_numpy().tolist() == [[0, 0, 0]] * 3
    assert summary.loc[ALL_PROBLEMS, ["macro_f1", "micro_f1"]].tolist() == [0, 0]


def test_evaluate_measures_how_often_no_focus_problems_get_nothing():
    truth = _frame([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    predicted = _frame([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]])

    summary = evaluate(truth, predicted).summary

    assert summary.at[ALL_PROBLEMS, "no_focus_nothing_predicted"] == 0.75  # 3 of the 4
    assert summary.at[FOCUS_TAGGED, "problems"] == 1  # the no-focus problems are left out
    assert math.isnan(summary.at[FOCUS_TAGGED, "no_focus_nothing_predicted"])


def test_evaluate_tag_confusion_gives_the_share_of_each_row_per_prediction():
    truth = _frame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    predicted = _frame([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])

    matrix = evaluate(truth, predicted).tag_confusion

    # Rows need not sum to 1: problem 2 has two tags, and both were predicted. No problem has
    # trees, so its row is undefined.
    expected = pd.DataFrame(
        [
            [2 / 3, 2 / 3, 0, 0],
            [1 / 2, 1 / 2, 0, 1 / 2],
            [math.nan] * 4,
            [1 / 2, 0, 0, 1 / 2],
        ],
        index=pd.Index([*TAGS, NO_FOCUS_TAG], name="true"),
        columns=pd.Index([*TAGS, NOTHING_PREDICTED], name="predicted"),
    )
    pd.testing.assert_frame_equal(matrix, expected)


def test_evaluate_without_groups_gives_point_estimates_only(truth, predicted):
    evaluation = evaluate(truth, predicted)

    per_tag_columns = ["support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]
    summary_columns = ["problems", "macro_f1", "micro_f1", "no_focus_nothing_predicted"]
    assert evaluation.per_tag.columns.tolist() == per_tag_columns
    assert evaluation.per_tag.index.names == ["slice", "tag"]
    assert evaluation.summary.columns.tolist() == summary_columns
    assert evaluation.summary.index.tolist() == [ALL_PROBLEMS, FOCUS_TAGGED]


def test_evaluation_macro_f1_is_the_macro_f1_over_all_problems(truth, predicted):
    evaluation = evaluate(truth, predicted)

    assert evaluation.macro_f1 == evaluation.summary.at[ALL_PROBLEMS, "macro_f1"]
    assert isinstance(evaluation.macro_f1, float)


def test_evaluation_micro_f1_and_tag_f1_are_over_all_problems(truth, predicted, years):
    evaluation = evaluate(truth, predicted, release_years=years)

    assert evaluation.micro_f1 == evaluation.summary.at[ALL_PROBLEMS, "micro_f1"]
    assert isinstance(evaluation.micro_f1, float)
    pd.testing.assert_series_equal(evaluation.tag_f1, evaluation.per_tag.loc[ALL_PROBLEMS, "f1"])
    assert evaluation.tag_f1.index.tolist() == TAGS


def test_evaluation_overall_f1_lists_the_macro_micro_then_each_tag_f1(truth, predicted, years):
    evaluation = evaluate(truth, predicted, release_years=years)

    overall = evaluation.overall_f1()

    assert overall.index.name == "metric"
    assert overall.index.tolist() == ["macro_f1", "micro_f1", *TAGS]
    assert overall.dtype == float
    expected = [evaluation.macro_f1, evaluation.micro_f1, *evaluation.tag_f1]
    assert overall.tolist() == expected


def test_evaluate_with_groups_adds_intervals_around_the_estimates(truth, predicted, years, groups):
    evaluation = evaluate(truth, predicted, release_years=years, groups=groups, resamples=500)

    per_tag, summary = evaluation.per_tag, evaluation.summary
    assert per_tag.columns.tolist()[-2:] == ["f1_low", "f1_high"]
    intervals = ["macro_f1_low", "macro_f1_high", "micro_f1_low", "micro_f1_high"]
    assert summary.columns.tolist()[-4:] == intervals
    assert ((per_tag["f1_low"] < per_tag["f1"]) & (per_tag["f1"] < per_tag["f1_high"])).all()
    for metric in ("macro_f1", "micro_f1"):
        assert (summary[f"{metric}_low"] < summary[metric]).all()
        assert (summary[metric] < summary[f"{metric}_high"]).all()


def test_evaluate_intervals_depend_on_the_seed_only(truth, predicted, groups):
    first = evaluate(truth, predicted, groups=groups, resamples=200, seed=1)

    again = evaluate(truth, predicted, groups=groups, resamples=200, seed=1)
    other = evaluate(truth, predicted, groups=groups, resamples=200, seed=2)
    pd.testing.assert_frame_equal(first.per_tag, again.per_tag)
    pd.testing.assert_frame_equal(first.summary, again.summary)
    assert not first.summary.equals(other.summary)


def test_evaluate_draws_the_same_resamples_for_every_slice(truth, predicted, groups):
    all_focus = truth.assign(math=1)  # every problem is focus-tagged: the two slices coincide

    summary = evaluate(all_focus, predicted, groups=groups, resamples=200).summary

    same = summary.loc[ALL_PROBLEMS].drop("no_focus_nothing_predicted")
    assert same.equals(summary.loc[FOCUS_TAGGED].drop("no_focus_nothing_predicted"))


def test_evaluate_intervals_do_not_narrow_when_each_problem_is_duplicated(truth, predicted):
    copies = np.repeat(np.arange(len(truth)), 2)  # problem k becomes rows 2k and 2k + 1
    index = pd.RangeIndex(len(copies))
    doubled_truth = truth.iloc[copies].set_axis(index)
    doubled_predicted = predicted.iloc[copies].set_axis(index)
    alone = pd.Series(np.arange(len(truth)), index=truth.index)

    single = evaluate(truth, predicted, groups=alone, resamples=300).summary
    by_group = evaluate(
        doubled_truth, doubled_predicted, groups=pd.Series(copies, index=index), resamples=300
    ).summary
    by_row = evaluate(
        doubled_truth, doubled_predicted, groups=pd.Series(index, index=index), resamples=300
    ).summary

    intervals = ["macro_f1_low", "macro_f1_high", "micro_f1_low", "micro_f1_high"]
    assert np.allclose(by_group[intervals], single[intervals])
    assert (_width(by_row, "macro_f1") < _width(single, "macro_f1")).all()  # a row bootstrap


def test_evaluate_rejects_predictions_not_aligned_on_the_truth(truth, predicted):
    with pytest.raises(ValueError, match="predicted must have truth's index and columns"):
        evaluate(truth, predicted.iloc[::-1])
    with pytest.raises(ValueError, match="predicted must have truth's index and columns"):
        evaluate(truth, predicted[TAGS[::-1]])


def test_evaluate_rejects_years_or_groups_not_aligned_on_the_truth(truth, predicted, years, groups):
    with pytest.raises(ValueError, match="release_years must have truth's index"):
        evaluate(truth, predicted, release_years=years.iloc[1:])
    with pytest.raises(ValueError, match="groups must have truth's index"):
        evaluate(truth, predicted, groups=groups.iloc[::-1])


def test_evaluate_rejects_values_other_than_zero_and_one(truth, predicted):
    with pytest.raises(ValueError, match="predicted must hold only 0 and 1"):
        evaluate(truth, predicted * 0.7)  # scores instead of 0/1 predictions
    with pytest.raises(ValueError, match="truth must hold only 0 and 1"):
        evaluate(truth * 2, predicted)


def test_compare_of_identical_predictions_has_a_zero_delta(truth, predicted, groups):
    table = compare(truth, predicted, predicted, groups, resamples=200)

    assert table.index.tolist() == ["macro_f1", "micro_f1", *TAGS]
    assert (table[["delta", "delta_low", "delta_high"]] == 0).all().all()


def test_compare_reports_both_estimates_and_their_difference(truth, predicted, groups):
    table = compare(truth, predicted, truth, groups, resamples=200)  # the candidate is perfect

    evaluation = evaluate(truth, predicted)
    assert table.at["macro_f1", "baseline"] == pytest.approx(evaluation.macro_f1)
    assert np.allclose(table.loc[TAGS, "baseline"], evaluation.per_tag.loc[ALL_PROBLEMS, "f1"])
    assert table["candidate"].tolist() == [1.0] * 5
    assert np.allclose(table["delta"], table["candidate"] - table["baseline"])
    assert (table["delta_low"] > 0).all()  # a clear improvement on every metric


def test_compare_draws_the_same_resamples_for_both_predictions(truth, predicted, groups):
    wrong = (predicted != truth).any(axis=1).to_numpy().argmax()  # the first problem with an error
    candidate = predicted.copy()
    candidate.iloc[wrong] = truth.iloc[wrong]

    table = compare(truth, predicted, candidate, groups, resamples=500)

    own = evaluate(truth, predicted, groups=groups, resamples=500).summary
    paired_width = table.at["macro_f1", "delta_high"] - table.at["macro_f1", "delta_low"]
    assert table.at["macro_f1", "delta_low"] >= 0  # fixing a prediction never lowers F1
    assert paired_width < _width(own, "macro_f1")[ALL_PROBLEMS] / 5


def test_compare_rejects_predictions_not_aligned_on_the_truth(truth, predicted, groups):
    with pytest.raises(ValueError, match="candidate must have truth's index and columns"):
        compare(truth, predicted, predicted.iloc[1:], groups)
    with pytest.raises(ValueError, match="groups must have truth's index"):
        compare(truth, predicted, predicted, groups.iloc[1:])


@pytest.fixture(name="many_tags")
def many_tags_fixture():
    """Return true and predicted tags of six problems over four tags; fft has no true positive."""
    index = pd.Index(range(6), name="sample_id")
    truth = pd.DataFrame(
        {
            "dp": [1, 1, 0, 0, 1, 0],
            "fft": [0, 0, 0, 0, 0, 0],
            "games": [0, 1, 1, 0, 0, 0],
            "math": [1, 0, 1, 1, 0, 0],
        },
        index=index,
    )
    predicted = pd.DataFrame(
        {
            "dp": [1, 0, 0, 1, 1, 0],
            "fft": [0, 1, 0, 0, 0, 0],
            "games": [0, 1, 0, 0, 0, 0],
            "math": [1, 0, 1, 1, 1, 1],
        },
        index=index,
    )
    return truth, predicted


def test_all_tag_macro_f1_averages_the_tags_with_a_true_positive(many_tags):
    truth, predicted = many_tags

    score, tags = all_tag_macro_f1(truth, predicted, ["dp", "fft", "games", "math"])

    present = ["dp", "games", "math"]
    expected = f1_score(truth[present], predicted[present], average="macro", zero_division=0)
    assert (score, tags) == (pytest.approx(expected), 3)


def test_all_tag_macro_f1_can_count_an_absent_tag_as_zero(many_tags):
    truth, predicted = many_tags

    score, tags = all_tag_macro_f1(truth, predicted, list(truth.columns), absent="zero")

    expected = f1_score(truth, predicted, average="macro", zero_division=0)
    skipped, _ = all_tag_macro_f1(truth, predicted, list(truth.columns))
    assert (score, tags) == (pytest.approx(expected), 4)
    assert score == pytest.approx(skipped * 3 / 4)  # fft's F1 counts as 0


def test_all_tag_macro_f1_reads_only_the_given_tags(many_tags):
    truth, predicted = many_tags

    score, tags = all_tag_macro_f1(truth, predicted, ("math", "games"))

    expected = f1_score(truth[["math", "games"]], predicted[["math", "games"]], average="macro")
    assert (score, tags) == (pytest.approx(expected), 2)


def test_all_tag_macro_f1_refuses_another_rule_for_absent_tags(many_tags):
    with pytest.raises(ValueError, match="absent must be one of"):
        all_tag_macro_f1(*many_tags, ["dp"], absent="drop")


def test_all_tag_macro_f1_refuses_a_tag_without_a_column(many_tags):
    truth, predicted = many_tags

    with pytest.raises(ValueError, match=r"predicted has no column for the tags \['strings'\]"):
        all_tag_macro_f1(truth.assign(strings=0), predicted, ["dp", "strings"])
    with pytest.raises(ValueError, match=r"truth has no column for the tags \['strings'\]"):
        all_tag_macro_f1(truth, predicted.assign(strings=0), ["dp", "strings"])


def test_all_tag_macro_f1_refuses_predictions_not_aligned_on_the_truth(many_tags):
    truth, predicted = many_tags

    with pytest.raises(ValueError, match="predicted must have truth's index and columns"):
        all_tag_macro_f1(truth, predicted.iloc[1:], ["dp"])
    with pytest.raises(ValueError, match="predicted must hold only 0 and 1"):
        all_tag_macro_f1(truth, predicted.replace(1, 2), ["dp"])


def test_all_tag_macro_f1_refuses_to_average_no_tag(many_tags):
    with pytest.raises(ValueError, match="No tag to average"):
        all_tag_macro_f1(*many_tags, ["fft"])
    with pytest.raises(ValueError, match="No tag to average"):
        all_tag_macro_f1(*many_tags, [], absent="zero")


def test_report_text_shows_each_tag_then_the_averages_then_the_tag_confusion_matrix(
    truth, predicted, groups
):
    evaluation = evaluate(truth, predicted, groups=groups)

    lines = report_text(evaluation).splitlines()

    summary = evaluation.summary.loc[ALL_PROBLEMS]
    math_row = evaluation.per_tag.loc[(ALL_PROBLEMS, "math")]
    metrics = ["precision", "recall", "f1", "f1_low", "f1_high"]
    assert lines[0].split() == ["support", "tp", "fp", "fn", "tn", *metrics]
    assert lines[1:3] == ["tag", lines[2]]
    assert lines[2].split() == [
        "math",
        *(str(int(math_row[count])) for count in ["support", "tp", "fp", "fn", "tn"]),
        *(f"{math_row[metric]:.4f}" for metric in metrics),
    ]
    assert lines[5:9] == [
        "",
        f"macro F1 {summary['macro_f1']:.4f} "
        f"[{summary['macro_f1_low']:.4f}, {summary['macro_f1_high']:.4f}]",
        f"micro F1 {summary['micro_f1']:.4f} "
        f"[{summary['micro_f1_low']:.4f}, {summary['micro_f1_high']:.4f}]",
        "",
    ]
    assert lines[9] == (
        "Tag confusion matrix: the share of each true tag's problems (rows) that got each "
        "predicted tag (columns)"
    )
    assert lines[10].split() == ["predicted", *TAGS, "nothing", "predicted"]
    assert [line.split()[0] for line in lines[11:]] == ["true", *TAGS, "no"]


def test_report_text_without_groups_shows_no_interval_and_a_missing_share_as_a_dash(truth):
    tagged = truth[truth.any(axis=1)]  # no problem without a focus tag

    lines = report_text(evaluate(tagged, tagged)).splitlines()

    assert lines[0].split() == ["support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]
    assert lines[6:8] == ["macro F1 1.0000", "micro F1 1.0000"]
    assert lines[-1].split() == ["no", "focus", "tag", *["-"] * (len(TAGS) + 1)]


def test_report_headline_holds_the_scores_over_all_problems_as_plain_json_values(
    truth, predicted, groups
):
    evaluation = evaluate(truth, predicted, groups=groups)

    headline = report_headline(evaluation)

    summary, trees = evaluation.summary.loc[ALL_PROBLEMS], evaluation.per_tag.loc[ALL_PROBLEMS]
    assert list(headline) == [
        "macro_f1",
        "micro_f1",
        "macro_f1_interval",
        "micro_f1_interval",
        "per_tag",
    ]
    assert headline["macro_f1"] == evaluation.macro_f1
    assert headline["micro_f1_interval"] == [summary["micro_f1_low"], summary["micro_f1_high"]]
    assert list(headline["per_tag"]) == TAGS
    assert headline["per_tag"]["trees"] == {
        "support": trees.at["trees", "support"],
        "precision": trees.at["trees", "precision"],
        "recall": trees.at["trees", "recall"],
        "f1": trees.at["trees", "f1"],
        "f1_interval": [trees.at["trees", "f1_low"], trees.at["trees", "f1_high"]],
    }
    assert json.loads(json.dumps(headline)) == headline


def test_report_headline_without_groups_has_no_interval(truth, predicted):
    headline = report_headline(evaluate(truth, predicted))

    assert headline["macro_f1_interval"] is headline["micro_f1_interval"] is None
    assert headline["per_tag"]["math"]["f1_interval"] is None


def test_save_report_writes_the_details_and_headline_as_json_then_the_tables_as_csv(
    truth, predicted, groups, tmp_path
):
    evaluation = evaluate(truth, predicted, groups=groups)
    details = {"method": "tfidf", "part": "test"}
    tables = {"predictions.csv": predicted, "scores.csv": None}

    paths = save_report(evaluation, tmp_path, details, tables)

    names = [REPORT_FILE, *REPORT_TABLES, "predictions.csv"]
    assert paths == [tmp_path / name for name in names]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)
    assert REPORT_TABLES == ("per_tag.csv", "summary.csv", "tag_confusion.csv")
    assert pd.read_csv(tmp_path / "predictions.csv", index_col="sample_id").equals(predicted)
    report = json.loads((tmp_path / REPORT_FILE).read_text(encoding="utf-8"))
    assert report == details | report_headline(evaluation)
    per_tag = pd.read_csv(tmp_path / "per_tag.csv", index_col=["slice", "tag"])
    summary = pd.read_csv(tmp_path / "summary.csv", index_col="slice")
    confusion = pd.read_csv(tmp_path / "tag_confusion.csv", index_col="true")
    pd.testing.assert_frame_equal(per_tag, evaluation.per_tag)
    pd.testing.assert_frame_equal(summary, evaluation.summary)
    pd.testing.assert_frame_equal(confusion, evaluation.tag_confusion, check_names=False)


def test_save_report_refuses_a_score_that_is_not_a_number_rather_than_write_it(truth, tmp_path):
    empty = evaluate(truth.iloc[:0], truth.iloc[:0])  # nothing to score: every score is NaN

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        save_report(empty, tmp_path, {})
