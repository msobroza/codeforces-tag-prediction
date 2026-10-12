"""Tests for near-duplicate detection."""

import pandas as pd
import pytest

from codeforces_tags.near_duplicates import detection
from codeforces_tags.near_duplicates.detection import (
    filter_by_verdicts,
    groups_from_pairs,
    load_verdicts,
    near_duplicate_groups,
    similar_pairs,
)

NO_VERDICTS = pd.DataFrame({"first": [], "second": [], "is_duplicate": []})

# Two near-identical descriptions (10, 11), and two unrelated ones (12, 13).
PROBLEMS = pd.DataFrame(
    {
        "prob_desc_description": [
            "This is the easy version. Given an array of n integers, find the maximum sum.",
            "This is the hard version. Given an array of n integers, find the maximum sum.",
            "A tree with n vertices is given. Count the leaves of the tree.",
            "Two players take stones from a pile in turn. Who wins if both play optimally?",
        ]
    },
    index=pd.Index([10, 11, 12, 13], name="sample_id"),
)


def verdicts(*rows):
    """Build verdicts from ``(first, second, is_duplicate)`` rows."""
    return pd.DataFrame(rows, columns=["first", "second", "is_duplicate"])


def test_near_duplicate_groups_links_similar_texts_when_nothing_is_judged():
    groups = near_duplicate_groups(PROBLEMS, verdicts=NO_VERDICTS, threshold=0.5, block_size=2)

    assert groups.index.tolist() == [10, 11, 12, 13]
    assert groups[10] == groups[11]
    assert groups.nunique() == 3


def test_near_duplicate_groups_puts_every_problem_in_its_own_group_at_threshold_one():
    assert near_duplicate_groups(PROBLEMS, verdicts=NO_VERDICTS, threshold=1.0).nunique() == 4


def test_near_duplicate_groups_drops_the_candidates_judged_not_duplicates():
    judged = verdicts((11, 10, False))

    groups = near_duplicate_groups(PROBLEMS, verdicts=judged, threshold=0.5)

    assert groups[10] != groups[11]  # similar, but judged different problems


def test_near_duplicate_groups_links_only_candidates_whatever_the_verdicts():
    judged = verdicts((10, 11, True), (13, 12, True))

    groups = near_duplicate_groups(PROBLEMS, verdicts=judged, threshold=0.5)

    assert groups[10] == groups[11]  # a candidate judged duplicate
    assert groups[12] != groups[13]  # judged duplicate, but below the threshold: stage 1 decides


def test_near_duplicate_groups_reads_the_frozen_verdicts_by_default(monkeypatch):
    monkeypatch.setattr(detection, "load_verdicts", lambda: verdicts((10, 11, False)))

    assert near_duplicate_groups(PROBLEMS, threshold=0.5).nunique() == 4


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(4),  # e.g. after reset_index(drop=True)
        pd.Index(["10", "11", "12", "13"], name="sample_id"),
        pd.Index([10, 10, 12, 13], name="sample_id"),
    ],
    ids=["reset index", "string ids", "repeated ids"],
)
def test_near_duplicate_groups_rejects_an_index_that_is_not_sample_ids(index):
    with pytest.raises(ValueError, match="sample ids"):
        near_duplicate_groups(PROBLEMS.set_axis(index), verdicts=NO_VERDICTS)


def test_near_duplicate_groups_takes_its_options_by_keyword_only():
    with pytest.raises(TypeError):
        near_duplicate_groups(PROBLEMS, NO_VERDICTS)  # pylint: disable=too-many-function-args


def test_filter_by_verdicts_drops_only_the_rejected_candidates():
    candidates = pd.DataFrame({"first": [1, 3, 5], "second": [2, 4, 6], "similarity": [0.9] * 3})
    judged = verdicts((2, 1, False), (4, 3, True), (7, 8, True))  # (7, 8) is no candidate

    links = filter_by_verdicts(candidates, judged)

    assert links[["first", "second"]].to_numpy().tolist() == [[3, 4], [5, 6]]


def test_load_verdicts_reads_the_audit():
    judged = load_verdicts()

    assert judged.columns.tolist() == ["first", "second", "is_duplicate"]
    assert len(judged) == 485
    assert judged["is_duplicate"].sum() == 82


def test_load_verdicts_rejects_an_unknown_verdict(tmp_path):
    path = tmp_path / "judgments.csv"
    path.write_text("sample_id_a,sample_id_b,final\n1,2,duplicate\n3,4,maybe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maybe"):
        load_verdicts(path)


def test_similar_pairs_lists_each_pair_once_with_its_similarity():
    texts = pd.Series(
        [
            "Given an array of n integers, find the maximum sum of a subarray.",
            "Given an array of n integers, find the maximum sum of a subarray quickly.",
            "A tree with n vertices is given. Count the leaves of the tree.",
        ],
        index=[7, 8, 9],
    )

    pairs = similar_pairs(texts, min_similarity=0.5)

    assert pairs[["first", "second"]].to_numpy().tolist() == [[7, 8]]
    assert 0.5 <= pairs.loc[0, "similarity"] <= 1.0


def test_similar_pairs_is_empty_when_nothing_is_similar_enough():
    texts = pd.Series(["alpha beta", "gamma delta"])

    assert similar_pairs(texts, min_similarity=0.9).empty


def test_groups_from_pairs_chains_links_and_keeps_unpaired_items_alone():
    index = pd.Index([10, 11, 12, 13, 14])
    pairs = pd.DataFrame({"first": [10, 11], "second": [11, 12]})

    groups = groups_from_pairs(index, pairs)

    assert groups[10] == groups[11] == groups[12]
    assert groups.nunique() == 3
    assert groups.index.tolist() == [10, 11, 12, 13, 14]


def test_groups_from_pairs_rejects_repeated_labels():
    pairs = pd.DataFrame({"first": [10], "second": [11]})

    with pytest.raises(ValueError, match="unique"):
        groups_from_pairs(pd.Index([10, 10, 11]), pairs)
