"""Tests for candidate pairs, judged tiers and batch files."""

import json

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.near_duplicates.audit.candidates import (
    TIERS,
    candidate_pairs,
    judging_tiers,
    pair_items,
    same_contest_control,
    same_examples,
    signal_matrices,
    stage_1_pairs,
    write_batches,
)


def test_same_examples_flags_identical_non_empty_examples_only():
    examples = pd.Series([["1 2"], ["1 2"], [], [], None])

    flags = same_examples(examples)

    assert flags[0, 1] and flags[1, 0]
    assert not flags[2, 3]  # empty lists are not a match
    assert not flags[4, 4]  # nor are missing examples


def test_signal_matrices_are_square_with_an_empty_diagonal(problems):
    signals = signal_matrices(problems)

    assert {matrix.shape for matrix in signals.values()} == {(30, 30)}
    assert np.all(np.diag(signals["word"]) == 0)
    assert signals["both_version"][4, 5]  # 104 and 105 both announce a version


def test_candidate_pairs_flags_one_pair_per_signal(problems, signal_pairs):
    candidates = candidate_pairs(problems.index.to_numpy(), signal_matrices(problems))

    assert [tuple(pair) for pair in candidates[["a", "b"]].to_numpy()] == list(signal_pairs)
    assert candidates.columns.tolist() == [
        "a", "b", "word", "char", "code",
        "same_contest", "same_sample_in", "same_sample_out", "both_version",
    ]  # fmt: skip


def test_same_contest_control_draws_unflagged_pairs_of_one_round(problems, signal_pairs):
    candidates = candidate_pairs(problems.index.to_numpy(), signal_matrices(problems))

    picks = same_contest_control(problems, candidates, size=5, seed=1)

    ids = problems.index.to_numpy()
    rounds = problems["prob_desc_created_at"].to_numpy()
    assert len(set(picks)) == 5
    assert all(rounds[i] == rounds[j] for i, j in picks)
    assert not {(ids[i], ids[j]) for i, j in picks} & set(signal_pairs)
    assert picks == same_contest_control(problems, candidates, size=5, seed=1)


@pytest.mark.usefixtures("small_audit")
def test_judging_tiers_sorts_each_signal_into_its_tier(problems, signal_pairs):
    signals = signal_matrices(problems)
    candidates = candidate_pairs(problems.index.to_numpy(), signals)

    pairs = judging_tiers(problems, candidates, signals)

    tier_of = {(row.a, row.b): row.tier[0] for row in pairs.itertuples()}
    assert {pair: tier_of[pair] for pair in signal_pairs} == signal_pairs
    assert (pairs["tier"] == TIERS[3]).sum() == 5
    assert pairs["pair_id"].tolist() == list(range(len(pairs)))


def test_stage_1_pairs_lists_only_the_detector_candidates(problems):
    pairs = stage_1_pairs(problems)

    assert pairs[["a", "b", "tier", "pair_id"]].to_numpy().tolist() == [[100, 101, TIERS[0], 0]]
    assert pairs.loc[0, "word"] == pytest.approx(0.883, abs=1e-3)


def test_tier_labels_name_the_candidate_threshold():
    assert TIERS[:2] == ("A: word>=0.4", "B: word<0.4, strong signal")


def test_pair_items_hide_ids_and_truncate_texts(problems):
    pairs = pd.DataFrame({"pair_id": [7], "a": [100], "b": [101]})

    [item] = pair_items(problems, pairs)

    assert set(item) == {"pair_id", "A", "B"}
    assert len(item["A"]["input_spec"]) == 700  # truncated
    assert item["B"]["input_spec"] == ""  # missing


def test_write_batches_splits_items_in_order(tmp_path):
    items = [{"pair_id": k} for k in range(5)]

    paths = write_batches(items, tmp_path, "batch", n_batches=2)

    assert [path.name for path in paths] == ["batch_0.json", "batch_1.json"]
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert batches == [items[:3], items[3:]]
