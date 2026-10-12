"""Tests for the search's objective: tag weights, problem scores and the balanced Brier score."""

import math
import re

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev import answers
from codeforces_tags.methods.typesafe_jev.objective import (
    balanced_brier,
    macro_balanced_brier,
    problem_scores,
    tag_weights,
    weighted_errors,
)

# Four problems whose labels give each focus tag a prevalence of 1/4, 1/2 or 3/4, and
# probabilities in quarters, so that every expected number below is an exact fraction.
LABELS = {
    "math": [1, 1, 0, 0],
    "graphs": [1, 0, 0, 0],
    "strings": [0, 1, 1, 1],
    "number theory": [1, 1, 0, 0],
    "trees": [0, 0, 1, 0],
    "geometry": [0, 0, 0, 1],
    "games": [1, 0, 1, 0],
    "probabilities": [0, 1, 0, 1],
}
PROBABILITIES = {
    "math": [0.75, 0.5, 0.25, 0.0],
    "graphs": [0.5, 0.25, 0.0, 0.5],
    "strings": [1.0, 0.75, 0.5, 0.25],
    "number theory": [0.25, 1.0, 0.5, 0.0],
    "trees": [0.0, 0.25, 0.75, 0.5],
    "geometry": [0.5, 0.0, 0.25, 1.0],
    "games": [0.5, 0.5, 0.5, 0.5],
    "probabilities": [0.0, 0.0, 1.0, 0.75],
}
INDEX = pd.Index([11, 12, 13, 14], name="sample_id")
# Each tag's balanced Brier score on the four problems, worked out by hand: e.g. probabilities,
# 1/2 * ((1 - 0)^2 + (1 - 0.75)^2) / 2 + 1/2 * (0^2 + 1^2) / 2 = 17/64 + 16/64.
BRIER = {
    "math": 3 / 32,
    "graphs": 17 / 96,
    "strings": 31 / 48,
    "number theory": 13 / 64,
    "trees": 1 / 12,
    "geometry": 5 / 96,
    "games": 1 / 4,
    "probabilities": 33 / 64,
}
# Their mean over the 8 tags, and each problem's score s_i = 1 - b_i, by hand.
MACRO_BRIER = 97 / 384
SCORES = [107 / 192, 51 / 64, 293 / 384, 335 / 384]


def frame(columns, index=INDEX):
    """Return a frame of the focus tags' columns, in FOCUS_TAGS order."""
    return pd.DataFrame(columns, index=index)[list(FOCUS_TAGS)]


def one_problem(values, sample_id=1):
    """Return one problem's row: ``values`` maps focus tags to numbers, 0 for a tag left out."""
    row = {tag: [values.get(tag, 0)] for tag in FOCUS_TAGS}
    return frame(row, index=pd.Index([sample_id], name="sample_id"))


def hand_weights(positive, negative):
    """Return a weights frame that gives every tag the same two weights."""
    return pd.DataFrame(
        {0: [negative] * len(FOCUS_TAGS), 1: [positive] * len(FOCUS_TAGS)},
        index=pd.Index(FOCUS_TAGS, name="tag"),
    )


def test_tag_weights_count_a_tags_positives_and_negatives_equally():
    weights = tag_weights(frame(LABELS))

    assert list(weights.index) == list(FOCUS_TAGS)
    assert list(weights.columns) == [0, 1]
    # math: prevalence 1/2; graphs: 1/4; strings: 3/4.
    assert weights.loc["math"].tolist() == [1.0, 1.0]
    assert weights.loc["graphs", 1] == 2.0
    assert weights.loc["graphs", 0] == pytest.approx(2 / 3)
    assert weights.loc["strings", 1] == pytest.approx(2 / 3)
    assert weights.loc["strings", 0] == 2.0
    # Each weight times its class share is 1/2: positives and negatives weigh the same.
    prevalence = frame(LABELS).mean()
    assert (weights[1] * prevalence).tolist() == pytest.approx([0.5] * 8)
    assert (weights[0] * (1 - prevalence)).tolist() == pytest.approx([0.5] * 8)


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"games": [0, 0, 0, 0]}, "['games']"),
        ({"strings": [1, 1, 1, 1]}, "['strings']"),
        ({"games": [0, 0, 0, 0], "probabilities": [1, 1, 1, 1]}, "['games', 'probabilities']"),
    ],
)
def test_tag_weights_refuse_a_tag_with_no_positive_or_no_negative(changes, named):
    with pytest.raises(ValueError, match=r"no positive or no negative.*" + re.escape(named)):
        tag_weights(frame(LABELS | changes))


def test_tag_weights_refuse_labels_that_are_not_the_focus_tags_or_not_0_1():
    with pytest.raises(ValueError, match="columns of labels must be the focus tags"):
        tag_weights(frame(LABELS)[list(reversed(FOCUS_TAGS))])
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        tag_weights(frame(LABELS | {"math": [2, 1, 0, 0]}))


def test_a_problem_score_is_one_minus_its_mean_weighted_error():
    labels = one_problem({"math": 1})
    probabilities = one_problem({"math": 0.5, "graphs": 0.5, "probabilities": 0.25})
    weights = hand_weights(positive=2.0, negative=0.5)

    errors = weighted_errors(probabilities, labels, weights)
    scores = problem_scores(probabilities, labels, weights)

    # math: 2 * (0.5 - 1)^2; graphs: 0.5 * 0.5^2; probabilities: 0.5 * 0.25^2; others 0.
    expected = {"math": 0.5, "graphs": 0.125, "probabilities": 0.03125}
    assert errors.iloc[0].to_dict() == {tag: expected.get(tag, 0.0) for tag in FOCUS_TAGS}
    # s = 1 - (0.5 + 0.125 + 0.03125) / 8, exact in binary.
    assert scores.to_dict() == {1: 0.91796875}
    assert scores.name == "score"


def test_a_failed_answer_scores_0_on_every_tag_and_costs_the_weight_of_each_true_tag(jev_line):
    # A failed line (here malformed) scores 0.0 on every tag (answers.probabilities); a
    # confident miss on a rare tag can push a problem's score below 0.
    lines = [jev_line(sample_id=7, status="malformed"), jev_line(sample_id=8)]
    index = pd.Index([7, 8], name="sample_id")
    probabilities = answers.probabilities(lines, index)
    labels = frame({tag: [0, 0] for tag in FOCUS_TAGS} | {"math": [1, 0], "games": [1, 0]}, index)
    weights = hand_weights(positive=6.0, negative=0.5)

    scores = problem_scores(probabilities, labels, weights)

    assert probabilities.loc[7].tolist() == [0.0] * 8
    # Problem 7: 6 * (0 - 1)^2 on math and on games, 0 elsewhere: 1 - 12 / 8.
    assert scores[7] == -0.5
    # Problem 8, all 0.5 with no true tag: 1 - 0.5 * 0.25.
    assert scores[8] == 0.875


def test_scores_and_balanced_brier_of_the_hand_built_problems():
    probabilities, labels = frame(PROBABILITIES), frame(LABELS)

    scores = problem_scores(probabilities, labels, tag_weights(labels))
    brier = balanced_brier(probabilities, labels)

    assert scores.index.equals(INDEX)
    assert scores.tolist() == pytest.approx(SCORES, abs=1e-12)
    assert list(brier.index) == list(FOCUS_TAGS)
    assert brier.name == "balanced_brier"
    assert brier.to_dict() == pytest.approx(BRIER, abs=1e-12)
    assert macro_balanced_brier(probabilities, labels) == pytest.approx(MACRO_BRIER, abs=1e-12)


def test_mean_of_one_minus_the_scores_is_the_macro_balanced_brier_under_the_same_labels():
    # The objective's identity: with weights from the labels scored, the mean
    # over problems of b_i is the mean over tags of the balanced Brier score, and each tag's
    # mean weighted error is that tag's balanced Brier score.
    probabilities, labels = frame(PROBABILITIES), frame(LABELS)
    weights = tag_weights(labels)

    mean_error = float((1 - problem_scores(probabilities, labels, weights)).mean())
    per_tag = weighted_errors(probabilities, labels, weights).mean()

    assert mean_error == pytest.approx(macro_balanced_brier(probabilities, labels), abs=1e-12)
    assert mean_error == pytest.approx(MACRO_BRIER, abs=1e-12)
    assert per_tag.to_dict() == pytest.approx(balanced_brier(probabilities, labels).to_dict())


def test_balanced_brier_is_nan_for_a_tag_with_one_class_and_the_macro_skips_it():
    labels = frame(LABELS | {"games": [0, 0, 0, 0]})

    brier = balanced_brier(frame(PROBABILITIES), labels)

    assert math.isnan(brier["games"])
    assert brier.drop("games").to_dict() == pytest.approx(
        {tag: value for tag, value in BRIER.items() if tag != "games"}, abs=1e-12
    )
    # The 7 other tags' mean: (8 * 97/384 - 1/4) / 7.
    macro = macro_balanced_brier(frame(PROBABILITIES), labels)
    assert macro == pytest.approx(85 / 336, abs=1e-12)


@pytest.mark.parametrize(
    "scorer",
    [
        lambda probabilities, labels: weighted_errors(probabilities, labels, hand_weights(1, 1)),
        lambda probabilities, labels: problem_scores(probabilities, labels, hand_weights(1, 1)),
        balanced_brier,
        macro_balanced_brier,
    ],
)
def test_every_score_refuses_misaligned_problems(scorer):
    probabilities, labels = frame(PROBABILITIES), frame(LABELS)

    # The same problems in another order, and other problems, are both refused.
    with pytest.raises(ValueError, match="same index, in the same order"):
        scorer(probabilities, labels.iloc[::-1])
    with pytest.raises(ValueError, match="same index, in the same order"):
        scorer(probabilities, labels.set_axis([11, 12, 13, 15]))
    with pytest.raises(ValueError, match="columns of probabilities must be the focus tags"):
        scorer(probabilities.drop(columns="games"), labels)


def test_weighted_errors_refuse_weights_that_are_not_tag_weights():
    probabilities, labels = frame(PROBABILITIES), frame(LABELS)
    weights = tag_weights(labels)

    with pytest.raises(ValueError, match="weights must be tag_weights"):
        weighted_errors(probabilities, labels, weights.iloc[::-1])
    with pytest.raises(ValueError, match="weights must be tag_weights"):
        weighted_errors(probabilities, labels, weights[[1, 0]])
