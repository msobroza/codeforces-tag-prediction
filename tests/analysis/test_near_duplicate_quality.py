"""Tests for scoring a duplicate detector and reporting judged pairs."""

import math

import pandas as pd

from codeforces_tags.analysis.near_duplicate_quality import (
    describe_pair,
    detection_curve,
    two_stage_errors,
    zero_failure_upper_bound,
)


def test_detection_curve_counts_kept_missed_and_false_groupings():
    similarity = pd.Series([0.95, 0.75, 0.6, 0.85])
    is_duplicate = pd.Series([True, True, False, False])

    curve = detection_curve(similarity, is_duplicate, thresholds=[0.7, 0.9])

    assert curve.loc[0.7].to_dict() == {
        "duplicates kept together": 2,
        "duplicates missed": 0,
        "non-duplicates grouped": 1,
        "recall": 1.0,
        "precision": 2 / 3,
    }
    assert curve.loc[0.9, "duplicates missed"] == 1
    assert curve.loc[0.9, "precision"] == 1.0


def test_zero_failure_upper_bound_matches_the_rule_of_three():
    assert math.isclose(zero_failure_upper_bound(150), 0.0198, abs_tol=1e-4)
    assert abs(zero_failure_upper_bound(300) - 3 / 300) < 1e-3


def test_two_stage_errors_counts_what_the_filter_fixes_and_what_it_cannot_see():
    similarity = pd.Series([0.95, 0.85, 0.75, 0.6, 0.3])
    is_duplicate = pd.Series([True, False, True, False, True])
    step2_duplicate = pd.Series([True, False, True, True, True])  # stage 2 wrongly keeps pair 3

    table = two_stage_errors(similarity, step2_duplicate, is_duplicate, 0.5, 0.8)

    assert table.loc["false positives of one stage (>= 0.8)"].tolist() == [1, 1, 0]
    assert table.loc["false negatives of one stage (>= 0.8)"].tolist() == [2, 1, 1]
    assert table.loc["false positives of stage 1 (>= 0.5)"].tolist() == [2, 1, 1]
    assert table.loc["false negatives of stage 1 (>= 0.5)"].tolist() == [1, 0, 1]
    assert table.loc["stage 2 mistakes on stage-1 candidates"].tolist() == [1, 0, 1]


def test_describe_pair_shows_both_calls_and_both_statements():
    pair = pd.Series(
        {
            "sample_id_a": 1,
            "sample_id_b": 2,
            "word_cosine": 0.4971,
            "final": "duplicate",
            "judge1_label": "versions",
            "judge1_confidence": "high",
            "judge1_reason": "Same task, larger limits.",
            "judge2_label": "related",
            "judge2_confidence": "low",
            "judge2_reason": "Different counting task.",
        }
    )
    texts = pd.Series({1: "Find $$$n$$$ quickly.", 2: "word " * 30})

    lines = describe_pair(pair, texts, chars=20, width=70).splitlines()

    assert lines[0] == "1 / 2: word cosine 0.497, verdict duplicate"
    assert lines[1] == "  judge 1 (versions, high confidence): Same task, larger limits."
    assert lines[2] == "  judge 2 (related, low confidence): Different counting task."
    assert lines[3] == "  [1] Find n quickly."
    assert lines[4] == "  [2] word word word word …"
