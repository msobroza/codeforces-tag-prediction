"""Tests for the tables of a search over Jev's words (notebook 10), on hand-built records."""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.analysis.prompt_search import (
    FLIPPED_COLUMNS,
    LINEAGE_COLUMNS,
    PIECE_COLUMNS,
    PROGRESS_COLUMNS,
    SHIFT_COLUMNS,
    flipped,
    lineage,
    piece_changes,
    probability_shift,
    refusal_counts,
    search_progress,
    word_diff,
)
from codeforces_tags.methods.typesafe_jev.reflection import REFUSAL_REASONS
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    JevWords,
    length_cap,
    parse_piece,
    piece_length,
    piece_text,
)

TAGS = ["math", "graphs"]
PROBLEMS = pd.Index([10, 11, 12, 13], name="sample_id")


@pytest.fixture(name="history")
def history_fixture():
    """Return four iterations: 1 and 3 accepted, 2 refused for copied text, 4 refused too long."""
    return pd.DataFrame(
        {
            "parent": [0, 1, 1, 2],
            "piece": ["math", "graphs", "graphs", "convention"],
            "parent_sum": [30.0, 31.0, 31.0, 32.0],
            "child_sum": [31.0, np.nan, 32.0, np.nan],
            "accepted": [True, False, True, False],
            "refusal": [np.nan, "copied_text", np.nan, "too_long"],
            "candidate": [1.0, np.nan, 2.0, np.nan],
            "jev_cost_usd": [0.10, 0.0, 0.20, 0.0],
            "luna_cost_usd": [0.01, 0.02, 0.01, 0.03],
            "luna_input_tokens": [100, 110, 120, 130],
            "luna_output_tokens": [10, 11, 12, 13],
        },
        index=pd.Index([1, 2, 3, 4], name="iteration"),
    )


@pytest.fixture(name="candidates")
def candidates_fixture():
    """Return the seed (0), its child 1, and 1's child 2, which failed the dataset-text check."""
    return pd.DataFrame(
        {
            "parents": [np.nan, "0", "1"],
            "iteration": [0, 1, 3],
            "piece": [np.nan, "math", "graphs"],
            "selection_mean": [0.70, 0.72, 0.75],
            "words_sha256": ["a" * 64, "b" * 64, "c" * 64],
            "passed_check": [True, True, False],
            "copied_sequences": [0, 0, 3],
        },
        index=pd.Index([0, 1, 2], name="index"),
    )


def words_with_definition(tag: str, definition: str) -> JevWords:
    """Return the seed words with the definition of a tag's yes replaced."""
    definitions = SEED_WORDS.settings()["tag_definitions"]  # deep-copied plain dicts
    definitions[tag]["true"]["definition"] = definition
    return JevWords(SEED_WORDS.convention, SEED_WORDS.question, definitions)


def test_word_diff_marks_removed_added_and_replaced_runs_of_words():
    assert word_diff("a b c", "a x c") == "a [-b-]{+x+} c"
    assert word_diff("a b", "a b c d") == "a b {+c d+}"
    assert word_diff("a b c", "a c") == "a [-b-] c"


def test_word_diff_ignores_a_change_of_spacing_alone():
    assert word_diff("one  two\nthree", "one two three") == "one two three"
    assert word_diff("", "") == ""


def test_piece_changes_has_one_row_per_piece_and_marks_only_the_changed_one():
    chosen = words_with_definition("graphs", "The problem is about a graph.")

    table = piece_changes(SEED_WORDS, chosen)

    assert list(table.index) == list(PIECES)
    assert table.index.name == "piece"
    assert list(table.columns) == list(PIECE_COLUMNS)
    assert table.index[table["changed"]].tolist() == ["graphs"]
    assert (table.loc[table.index != "graphs", "changed_words"] == 0).all()


def test_piece_changes_measures_the_capped_length_and_diffs_the_piece_text():
    chosen = words_with_definition("graphs", "The problem is about a graph.")
    seed_text, chosen_text = piece_text(SEED_WORDS, "graphs"), piece_text(chosen, "graphs")

    row = piece_changes(SEED_WORDS, chosen).loc["graphs"]

    assert row["seed_chars"] == piece_length("graphs", parse_piece("graphs", seed_text))
    assert row["chosen_chars"] == piece_length("graphs", parse_piece("graphs", chosen_text))
    assert row["chosen_chars"] < row["seed_chars"] <= row["length_cap"] == length_cap("graphs")
    assert row["diff"] == word_diff(seed_text, chosen_text)
    assert "[-" in row["diff"] and "{+" in row["diff"] and row["changed_words"] > 0


def test_piece_changes_of_a_changed_convention():
    chosen = JevWords("Tags name ideas.", SEED_WORDS.question, SEED_WORDS.tag_definitions)

    row = piece_changes(SEED_WORDS, chosen).loc["convention"]

    assert row["changed"]
    assert row["chosen_chars"] == len("Tags name ideas.")
    assert row["seed_chars"] == len(SEED_WORDS.convention)


def test_search_progress_keeps_the_best_passing_candidate_and_sums_the_costs(history, candidates):
    progress = search_progress(history, candidates)

    assert list(progress.columns) == list(PROGRESS_COLUMNS)
    assert progress.index.name == "iteration"
    assert progress.index.tolist() == [1, 2, 3, 4]
    # Candidate 2 (0.75, iteration 3) failed the dataset-text check: candidate 1 stays the best.
    assert progress["best_selection_mean"].tolist() == [0.72, 0.72, 0.72, 0.72]
    assert progress["accepted"].tolist() == [True, False, True, False]
    np.testing.assert_allclose(progress["jev_cost_usd"], [0.10, 0.10, 0.30, 0.30])
    np.testing.assert_allclose(progress["luna_cost_usd"], [0.01, 0.03, 0.04, 0.07])
    np.testing.assert_allclose(progress["total_cost_usd"], [0.11, 0.13, 0.34, 0.37])


def test_search_progress_counts_the_seed_from_the_start(history, candidates):
    # A seed recorded at a later iteration number is still known before the first iteration.
    candidates.loc[0, "iteration"] = 99
    candidates.loc[1, "selection_mean"] = 0.60

    progress = search_progress(history, candidates)

    assert progress["best_selection_mean"].tolist() == [0.70, 0.70, 0.70, 0.70]


def test_search_progress_reads_flags_written_as_names(history, candidates):
    history["accepted"] = history["accepted"].map(str)
    candidates["passed_check"] = ["True", "True", "True"]

    progress = search_progress(history, candidates)

    assert progress["accepted"].dtype == bool
    assert progress["best_selection_mean"].tolist() == [0.72, 0.72, 0.75, 0.75]


def test_search_progress_refuses_a_flag_that_is_not_a_boolean(history, candidates):
    history["accepted"] = ["yes", "no", "yes", "no"]

    with pytest.raises(ValueError, match="accepted must hold True or False"):
        search_progress(history, candidates)


def test_search_progress_refuses_a_missing_column(history, candidates):
    with pytest.raises(ValueError, match=r"history lacks the columns \['jev_cost_usd'\]"):
        search_progress(history.drop(columns="jev_cost_usd"), candidates)


def test_search_progress_refuses_a_record_where_no_candidate_passed(history, candidates):
    candidates["passed_check"] = False

    with pytest.raises(ValueError, match="No candidate passed"):
        search_progress(history, candidates)


@pytest.mark.parametrize(
    "parents",
    [
        [np.nan, "0", "1"],
        [np.nan, 0.0, 1.0],
        [[], [0], [1]],
        ["", "0", "1 0"],
    ],
)
def test_lineage_follows_the_first_parent_back_to_the_seed(candidates, parents):
    candidates["parents"] = pd.Series(parents, index=candidates.index, dtype=object)

    table = lineage(candidates, 2)

    assert table.index.name == "index"
    assert table.index.tolist() == [0, 1, 2]
    assert list(table.columns) == list(LINEAGE_COLUMNS)
    assert table["piece"].tolist()[1:] == ["math", "graphs"]
    np.testing.assert_allclose(table["gain"].to_numpy()[1:], [0.02, 0.03])
    assert np.isnan(table["gain"].iloc[0])


def test_lineage_of_the_seed_is_the_seed_alone(candidates):
    assert lineage(candidates, 0).index.tolist() == [0]


def test_lineage_refuses_a_missing_candidate(candidates):
    candidates.loc[2, "parents"] = "7"

    with pytest.raises(ValueError, match="Candidate 7 is not in the search record"):
        lineage(candidates, 2)


def test_lineage_refuses_a_candidate_that_is_its_own_ancestor(candidates):
    candidates.loc[0, "parents"] = "2"

    with pytest.raises(ValueError, match="is its own ancestor"):
        lineage(candidates, 2)


def test_refusal_counts_counts_every_reason_in_order(history):
    counts = refusal_counts(history)

    assert counts.index.tolist() == list(REFUSAL_REASONS)
    assert counts.index.name == "reason"
    assert counts.name == "iterations"
    assert counts.to_dict() == {
        "failed_answer": 0,
        "invalid_json": 0,
        "empty_field": 0,
        "too_long": 1,
        "copied_text": 1,
    }


@pytest.mark.parametrize("none", [np.nan, ""])
def test_refusal_counts_of_a_search_that_refused_nothing(history, none):
    # read_search_record gives "" for no refusal; a frame built by hand may hold NaN.
    history["refusal"] = none

    counts = refusal_counts(history)

    assert counts.dtype == "int64"
    assert counts.sum() == 0


def test_refusal_counts_refuses_an_unknown_reason(history):
    history.loc[1, "refusal"] = "bored"

    with pytest.raises(ValueError, match=r"Unknown refusal reasons \['bored'\]"):
        refusal_counts(history)


@pytest.fixture(name="labels")
def labels_fixture():
    """Return the true tags of four problems: two math, one graphs."""
    return pd.DataFrame({"math": [1, 1, 0, 0], "graphs": [0, 0, 1, 0]}, index=PROBLEMS)


def test_probability_shift_compares_each_tags_mean_probability_with_its_base_rate(labels):
    before = pd.DataFrame({"math": [0.9, 0.8, 0.7, 0.6], "graphs": [0.5] * 4}, index=PROBLEMS)
    after = pd.DataFrame(
        {"math": [0.9, 0.7, 0.2, 0.2], "graphs": [0.1, 0.1, 0.9, 0.1]}, index=PROBLEMS
    )

    table = probability_shift(before, after, labels)

    assert table.index.tolist() == TAGS
    assert table.index.name == "tag"
    assert list(table.columns) == list(SHIFT_COLUMNS)
    np.testing.assert_allclose(table["base_rate"], [0.5, 0.25])
    np.testing.assert_allclose(table["mean_before"], [0.75, 0.5])
    np.testing.assert_allclose(table["mean_after"], [0.5, 0.3])


def test_probability_shift_refuses_misaligned_frames_or_values_outside_zero_and_one(labels):
    good = pd.DataFrame(0.5, index=PROBLEMS, columns=TAGS)

    with pytest.raises(ValueError, match="after must have the problems and tags of labels"):
        probability_shift(good, good[["graphs", "math"]], labels)
    with pytest.raises(ValueError, match="before must hold probabilities"):
        probability_shift(good + 1, good, labels)


def test_flipped_lists_the_problems_whose_prediction_changed_on_the_tag(labels):
    before = pd.DataFrame({"math": [1, 0, 1, 0], "graphs": [0, 0, 0, 0]}, index=PROBLEMS)
    after = pd.DataFrame({"math": [1, 1, 0, 0], "graphs": [1, 0, 0, 0]}, index=PROBLEMS)

    table = flipped(before, after, labels, "math")

    assert table.index.name == "sample_id"
    assert table.index.tolist() == [11, 12]
    assert list(table.columns) == list(FLIPPED_COLUMNS)
    assert table.to_dict("index") == {
        11: {"gold": 1, "before": 0, "after": 1, "now_right": True},
        12: {"gold": 0, "before": 1, "after": 0, "now_right": True},
    }
    assert flipped(before, after, labels, "graphs")["now_right"].tolist() == [False]


def test_flipped_refuses_a_value_other_than_0_and_1_or_an_unknown_tag(labels):
    zeros = pd.DataFrame(0, index=PROBLEMS, columns=TAGS)

    with pytest.raises(ValueError, match="before must hold 0/1 tags"):
        flipped(zeros + 2, zeros, labels, "math")
    with pytest.raises(ValueError, match="'trees' is not a tag of labels"):
        flipped(zeros, zeros, labels, "trees")
    with pytest.raises(ValueError, match="before must have the problems and tags of labels"):
        flipped(zeros.iloc[:2], zeros, labels, "math")
