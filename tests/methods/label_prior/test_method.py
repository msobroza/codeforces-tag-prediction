"""Tests for the label prior, the F1 floor of every other method."""

import json
import shutil

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.conformance import check_method_folder, check_tagging_method
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods import ScoreMethod, TaggingMethod, load_method, save_method
from codeforces_tags.methods.label_prior.method import LabelPrior, load
from codeforces_tags.methods.label_prior.settings import NAME, PRIOR_FILE


def make_labels(index, positives):
    """Return 0/1 focus labels for ``index``: a 1 for the problems listed under each tag."""
    labels = pd.DataFrame(0, index=index, columns=list(FOCUS_TAGS))
    for tag, sample_ids in positives.items():
        labels.loc[sample_ids, tag] = 1
    return labels


def test_a_label_prior_is_a_tagging_method_over_the_focus_tags():
    method = LabelPrior()

    assert isinstance(method, TaggingMethod)
    assert not isinstance(method, ScoreMethod)
    assert (method.name, method.tags) == ("label_prior", FOCUS_TAGS) == (NAME, FOCUS_TAGS)


def test_fit_predicts_each_focus_tag_with_a_positive_in_the_fit_data(raw_problems):
    labels = make_labels(raw_problems.index, {"trees": [10], "math": [30, 20]})

    fitted = LabelPrior().fit(raw_problems, labels)

    assert fitted.predicted == ("math", "trees")  # in FOCUS_TAGS order, whatever the labels' order


def test_fit_returns_a_new_method_and_leaves_the_unfitted_one_predicting_nothing(raw_problems):
    unfitted = LabelPrior()

    fitted = unfitted.fit(raw_problems, make_labels(raw_problems.index, {"games": [10]}))

    assert (unfitted.predicted, fitted.predicted) == ((), ("games",))


def test_predict_gives_every_problem_the_same_int_0_1_row(raw_problems):
    predictions = LabelPrior(("math", "trees")).predict(raw_problems)

    assert list(predictions.columns) == list(FOCUS_TAGS)
    assert predictions.index.equals(raw_problems.index)  # out of sample id order: 30, 10, 20
    assert (predictions.dtypes == "int64").all()
    assert predictions.to_numpy().tolist() == [[1, 0, 0, 0, 1, 0, 0, 0]] * 3


def test_predict_gives_an_empty_int_frame_for_no_problem(raw_problems):
    predictions = LabelPrior(("math",)).predict(raw_problems.iloc[:0])

    assert predictions.shape == (0, len(FOCUS_TAGS))
    assert (predictions.dtypes == "int64").all()


def test_predict_reads_no_column_of_the_raw_problems(raw_problems):
    method = LabelPrior(("math",))

    pd.testing.assert_frame_equal(method.predict(raw_problems), method.predict(raw_problems[[]]))


def test_the_predicted_tags_are_focus_tags_in_focus_order_without_repeats():
    assert LabelPrior(("trees", "math", "math")).predicted == ("math", "trees")
    with pytest.raises(ValueError, match=r"\['implementation'\] are not focus tags"):
        LabelPrior(("math", "implementation"))


def test_a_tag_predicted_for_every_problem_scores_f1_2p_over_1_plus_p():
    # The F1-optimal constant (docs/adr/0004): p is the tag's share of positives.
    index = pd.Index(range(10), name="sample_id")
    labels = make_labels(index, {"math": [0, 1], "trees": [0, 1, 2, 3, 4]})
    nothing = pd.DataFrame(index=index)

    predictions = LabelPrior().fit(nothing, labels).predict(nothing)

    f1 = evaluate(labels, predictions).tag_f1
    assert f1["math"] == pytest.approx(2 * 0.2 / (1 + 0.2))
    assert f1["trees"] == pytest.approx(2 * 0.5 / (1 + 0.5))
    assert f1["graphs"] == 0  # no positive in the fit data: never predicted


def test_save_writes_the_predicted_tags_and_load_reads_them(tmp_path):
    LabelPrior(("math", "trees")).save(tmp_path)

    stored = json.loads((tmp_path / PRIOR_FILE).read_text(encoding="utf-8"))
    assert stored == {"predicted": ["math", "trees"]}
    assert load(tmp_path) == LabelPrior(("math", "trees"))


def test_load_refuses_a_file_that_names_another_tag(tmp_path):
    (tmp_path / PRIOR_FILE).write_text('{"predicted": ["dp"]}', encoding="utf-8")

    with pytest.raises(ValueError, match="not focus tags"):
        load(tmp_path)


def test_label_prior_folder_holds_no_absolute_path(tmp_path):
    # A method folder moved to another directory, or machine, must load.
    original = tmp_path / "here" / "model"
    save_method(LabelPrior(("math",)), original)
    moved = tmp_path / "there" / "model"
    moved.parent.mkdir()
    shutil.move(original, moved)

    check_method_folder(moved)
    assert [path.name for path in sorted(moved.iterdir())] == ["label_prior.json", "method.json"]
    for path in moved.iterdir():
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    assert load_method(moved) == LabelPrior(("math",))


def test_label_prior_passes_check_tagging_method(raw_problems, tmp_path):
    # Raw problems as select_part gives them (extra columns, NaN fields).
    check_tagging_method(LabelPrior(("math", "trees")), raw_problems, tmp_path / "model")
