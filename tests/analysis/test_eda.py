"""Tests for the exploratory data analysis helpers."""

import math

import numpy as np
import pandas as pd

from codeforces_tags.analysis.eda import (
    approximate_token_count,
    code_line_count,
    cooccurrence,
    cyrillic_field_counts,
    cyrillic_words,
    distinctive_terms,
    label_counts,
    lift,
    missing_value_counts,
    near_duplicate_summary,
    odds_ratio,
    parses_as_python3,
    profile_columns,
    scraping_artefact_shares,
    summarise_by_label,
    values_by_label,
    word_count,
)
from codeforces_tags.data.labels import multi_hot


def test_profile_columns_summarises_types_missing_and_unique_values():
    frame = pd.DataFrame(
        {"tags": [["b", "a"], ["a", "b"], []], "notes": ["x", None, ""], "unused": [None] * 3}
    )

    profile = profile_columns(frame)

    assert profile.loc["tags", "unique"] == 2  # same tags in another order count once
    assert profile.loc["tags", "empty"] == 1
    assert profile.loc["notes", "missing"] == 1
    assert profile.loc["notes", "example"] == "'x'"
    assert profile.loc["unused", "missing"] == 3
    assert profile.loc["unused", "example"] == "None"


def test_missing_value_counts_covers_none_empty_and_sentinels():
    frame = pd.DataFrame(
        {
            "difficulty": [800, None, -1],
            "tags": [["math"], [], ["dp"]],
            "lang": ["Python 3"] * 3,
        }
    )

    counts = missing_value_counts(frame, sentinels={"difficulty": -1})

    assert counts.index.tolist() == ["tags", "difficulty"]
    assert counts.loc["difficulty"].tolist() == [1, 0, 1]
    assert counts.loc["tags", "empty string / list"] == 1


def test_scraping_artefact_shares_measures_each_artefact():
    frame = pd.DataFrame(
        {
            "prob_desc_description": ["Sort it.Then print", "Sort it. Then print", "Done."],
            "prob_desc_notes": ["NoteIn the first test", "ПримечаниеВ тесте", "Note that n > 1"],
            "tags": [["math", "math"], ["dp"], []],
        }
    )

    shares = scraping_artefact_shares(frame)

    assert shares.tolist() == [1 / 3, 2 / 3, 1 / 3]  # "Note that" is a sentence, not a heading


def test_cyrillic_words_lists_words_with_a_cyrillic_letter():
    assert cyrillic_words("a simplе С++ solution, simplе") == ["simplе", "С"]  # Cyrillic е, С


def test_label_counts_counts_each_label_once_per_problem_and_skips_empty_lists():
    tags = pd.Series([["math", "dp", "math"], ["math"], []])

    assert label_counts(tags).to_dict() == {"math": 2, "dp": 1}


def test_distinctive_terms_ranks_the_word_that_separates_the_label_first():
    texts = pd.Series(["tree root leaf"] * 20 + ["array sum"] * 20 + ["tree array"] * 5)
    labels = pd.DataFrame({"trees": [1] * 20 + [0] * 25})

    terms = distinctive_terms(texts, labels, top_k=2, min_df=5)

    assert terms.loc[1, "trees"].startswith(("leaf", "root"))
    assert "array" not in " ".join(terms["trees"])


def test_distinctive_terms_reports_the_odds_ratio_next_to_each_term():
    texts = pd.Series(["tree"] * 10 + ["other"] * 10)
    labels = pd.DataFrame({"trees": [1] * 10 + [0] * 10})

    terms = distinctive_terms(texts, labels, top_k=1, min_df=5)

    assert terms.loc[1, "trees"].startswith("tree (OR ")


def test_values_by_label_repeats_multi_labelled_rows_and_groups_unlabelled_ones():
    indicators = multi_hot(pd.Series([["a", "b"], ["a"], []]), vocabulary=["a", "b"])
    values = pd.Series([10.0, 20.0, 30.0])

    long = values_by_label(values, indicators, none_label="none")

    assert sorted(map(tuple, long.to_numpy().tolist())) == [
        ("a", 10.0),
        ("a", 20.0),
        ("b", 10.0),
        ("none", 30.0),
    ]


def test_values_by_label_drops_missing_values():
    indicators = multi_hot(pd.Series([["a"], ["a"]]), vocabulary=["a"])

    long = values_by_label(pd.Series([1.0, None]), indicators)

    assert long["value"].tolist() == [1.0]


def test_cooccurrence_counts_and_row_normalisation():
    indicators = multi_hot(pd.Series([["a", "b"], ["a"], ["b"], ["a", "b"]]))

    counts = cooccurrence(indicators)
    conditional = cooccurrence(indicators, normalize=True)

    assert counts.loc["a", "a"] == 3
    assert counts.loc["a", "b"] == 2
    assert conditional.loc["a", "b"] == 2 / 3
    assert conditional.loc["b", "b"] == 1.0


def test_cooccurrence_between_two_label_sets():
    focus = multi_hot(pd.Series([["math"], ["math"], ["trees"]]), vocabulary=["math", "trees"])
    imports = multi_hot(pd.Series([["fractions"], [], ["sys"]]), vocabulary=["fractions", "sys"])

    conditional = cooccurrence(imports, focus, normalize=True)

    assert conditional.loc["fractions", "math"] == 1.0
    assert conditional.loc["sys", "trees"] == 1.0
    assert conditional.loc["sys", "math"] == 0.0


def test_cooccurrence_counts_do_not_overflow():
    indicators = multi_hot(pd.Series([["math"]] * 300))

    assert cooccurrence(indicators).loc["math", "math"] == 300


def test_lift_compares_conditional_and_overall_rates():
    tags = multi_hot(pd.Series([["math"], ["math"], [], []]), vocabulary=["math"])
    modules = multi_hot(pd.Series([["fractions"], [], [], []]), vocabulary=["fractions"])

    assert lift(modules, tags).loc["fractions", "math"] == 2.0  # 1.0 / 0.5


def test_word_count_ignores_math_spans():
    assert word_count("Find $$$n \\le 10$$$ quickly") == 2


def test_approximate_token_count_splits_words_numbers_and_symbols():
    assert approximate_token_count("n <= 10^5, done") == 8  # n < = 10 ^ 5 , done
    assert approximate_token_count("Примечание: café") == 3  # non-ASCII words stay whole


def test_code_line_count_skips_blank_lines():
    assert code_line_count("a = 1\n\n   \nprint(a)\n") == 2


def test_parses_as_python3_rejects_python2_code():
    assert parses_as_python3("print(1)")
    assert not parses_as_python3("print 'hello'")
    assert parses_as_python3("s = '\\d'")  # invalid escape: a warning, not an error


def test_odds_ratio_crude_and_stratified():
    exposed = pd.Series([True, True, False, False, True, False])
    outcome = pd.Series([True, False, True, False, True, False])
    strata = pd.Series([0, 0, 0, 0, 1, 1])

    assert odds_ratio(exposed, outcome) == 4.0  # (2 * 2) / (1 * 1)
    assert math.isclose(odds_ratio(exposed, outcome, strata), (0.25 + 0.5) / 0.25)


def test_odds_ratio_accepts_zero_one_indicators():
    as_ints = odds_ratio(pd.Series([1, 1, 1, 0, 0]), pd.Series([1, 1, 0, 1, 0]))
    as_bools = odds_ratio(
        pd.Series([True, True, True, False, False]), pd.Series([True, True, False, True, False])
    )

    assert as_ints == as_bools == 2.0


def test_odds_ratio_is_inf_or_nan_when_the_table_has_empty_cells():
    assert odds_ratio(pd.Series([True, False]), pd.Series([True, False])) == np.inf
    assert math.isnan(odds_ratio(pd.Series([True, True]), pd.Series([True, False])))


def test_distinctive_terms_pads_when_fewer_terms_than_top_k():
    texts = pd.Series(["tree root"] * 10 + ["array sum"] * 10)
    labels = pd.DataFrame({"trees": [1] * 10 + [0] * 10})

    terms = distinctive_terms(texts, labels, top_k=6, min_df=5)

    assert len(terms) == 6
    assert terms["trees"].tolist()[2:] == ["", "", "", ""]


def test_summarise_by_label_aggregates_rows_per_label_and_overall():
    indicators = multi_hot(pd.Series([["a"], ["a", "b"], []]), vocabulary=["a", "b"])
    values = pd.DataFrame({"limit": [1.0, 3.0, 8.0]})

    summary = summarise_by_label(values, indicators, "median")

    assert summary.loc["limit"].to_dict() == {"a": 2.0, "b": 3.0, "all problems": 3.0}


def test_near_duplicate_summary_counts_groups_and_tag_agreement():
    groups = pd.Series([0, 0, 1, 1, 2])
    tags = pd.Series([["math", "dp"], ["math"], ["trees"], ["trees"], ["games"]])

    summary = near_duplicate_summary(groups, tags, focus_tags=["math", "trees"])

    assert summary.to_dict() == {
        "groups of near-duplicates": 2,
        "largest group size": 2,
        "problems in a group": 4,
        "groups with identical tags": 1,
        "groups with identical focus tags": 2,
    }


def test_cyrillic_field_counts_separates_look_alikes_from_russian():
    frame = pd.DataFrame({"text": ["Use С++ here", "Вася любит задачи", "plain", None]})

    counts = cyrillic_field_counts(frame, fields=["text"])

    assert counts.loc["text"].tolist() == [2, 1]
