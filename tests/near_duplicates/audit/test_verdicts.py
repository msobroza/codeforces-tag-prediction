"""Tests for the judging rounds and the merged verdicts."""

import pandas as pd

from codeforces_tags.near_duplicates.audit.verdicts import (
    confirmed_pairs,
    labels_frame,
    merge_judgments,
    round_1_pairs,
    round_2_pairs,
    says_duplicate,
)


def answer(pair_id, label, confidence="high"):
    """Build one judge answer."""
    return {"pair_id": pair_id, "label": label, "confidence": confidence, "reason": "r"}


PAIRS = pd.DataFrame(
    {
        "a": [10, 20, 30, 40],
        "b": [11, 21, 31, 41],
        "word": [0.95, 0.5, 0.9, 0.1],
        "char": [0.9, 0.5, 0.8, 0.1],
        "code": [0.2, 0.1, 0.1, 0.0],
        "same_contest": [True, True, False, True],
        "same_sample_in": [False] * 4,
        "same_sample_out": [False] * 4,
        "both_version": [True, False, False, False],
        "tier": ["A", "A", "A", "D"],
        "pair_id": [0, 1, 2, 3],
    }
)


def test_labels_frame_indexes_answers_by_pair():
    frame = labels_frame([answer(3, "different"), answer(1, "versions")])

    assert frame.index.tolist() == [3, 1]
    assert frame.columns.tolist() == ["label", "confidence", "reason"]


def test_says_duplicate_accepts_same_problem_and_versions_only():
    labels = pd.Series(["same_problem", "versions", "related", "different", None])

    assert says_duplicate(labels).tolist() == [True, True, False, False, False]


def test_round_1_pairs_rechecks_calls_against_the_former_threshold_and_low_confidence():
    first = labels_frame(
        [
            answer(0, "versions"),  # duplicate above 0.8: agrees with the former detector
            answer(1, "versions"),  # duplicate below 0.8: a miss of the former detector
            answer(2, "related"),  # non-duplicate above 0.8: a false link of it
            answer(3, "different", confidence="low"),
        ]
    )

    assert round_1_pairs(first, PAIRS.set_index("pair_id")["word"]) == [1, 2, 3]


def test_round_1_pairs_counts_the_threshold_itself_as_above():
    first = labels_frame([answer(0, "versions"), answer(1, "related")])
    word = pd.Series([0.8, 0.8], index=[0, 1])

    assert round_1_pairs(first, word, threshold=0.8) == [1]  # a non-duplicate at 0.8


def test_round_2_pairs_takes_every_pair_not_seen_in_round_1():
    assert round_2_pairs([3, 0, 2, 1], [2]) == [0, 1, 3]


def test_merge_judgments_links_a_pair_when_either_judge_says_duplicate():
    raw = {
        "first": [answer(0, "versions"), answer(1, "versions"), answer(2, "related"),
                  answer(3, "different")],
        "second": [answer(1, "related")],
        "second_round_2": [answer(0, "same_problem"), answer(2, "different"),
                           answer(3, "different")],
    }  # fmt: skip

    judgments = merge_judgments(PAIRS, raw)

    assert judgments["pair_id"].tolist() == [0, 2, 1, 3]  # most similar first
    final = judgments.set_index("pair_id")["final"]
    assert final.tolist() == ["duplicate", "not_duplicate", "duplicate", "not_duplicate"]
    # Pair 1's second label comes from round 1, the others' from round 2.
    second = judgments.set_index("pair_id")["judge2_label"]
    assert second.tolist() == ["same_problem", "different", "related", "different"]
    assert judgments.columns[:5].tolist() == [
        "pair_id", "sample_id_a", "sample_id_b", "tier", "word_cosine",
    ]  # fmt: skip
    assert judgments.columns[-1] == "final"


def test_confirmed_pairs_keeps_duplicates_with_both_labels():
    raw = {"first": [answer(k, "versions" if k == 0 else "different") for k in range(4)]}

    confirmed = confirmed_pairs(merge_judgments(PAIRS, raw))

    assert confirmed[["sample_id_a", "sample_id_b"]].to_numpy().tolist() == [[10, 11]]
    assert "judge2_label" in confirmed.columns
