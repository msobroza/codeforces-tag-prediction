"""Tests for preprocessing step 2, text cleaning."""

import pandas as pd

from codeforces_tags.preprocessing.text import (
    build_statement,
    clean_text,
    clean_text_fields,
    cyrillic_share,
    is_russian,
    math_spans,
    strip_latex,
    strip_notes_heading,
)


def test_clean_text_restores_the_space_between_glued_paragraphs():
    assert clean_text("rounded up instead.You should") == "rounded up instead. You should"
    assert clean_text("with $$$n$$$.Then") == "with $$$n$$$. Then"


def test_clean_text_keeps_decimals_and_abbreviations():
    assert clean_text("print 3.5 in the U.S.A.") == "print 3.5 in the U.S.A."


def test_clean_text_replaces_cyrillic_look_alikes_in_english_text_only():
    english_with_look_alikes = "Use С++ and a simplе solution"  # Cyrillic С and е
    russian = "Вася любит задачи"

    assert clean_text(english_with_look_alikes) == "Use C++ and a simple solution"
    assert clean_text(russian) == russian


def test_strip_notes_heading_removes_the_heading_once():
    assert strip_notes_heading("NoteIn the first test case") == "In the first test case"
    assert strip_notes_heading("Notegcd(x, y) is the gcd") == "gcd(x, y) is the gcd"
    assert strip_notes_heading("ПримечаниеВ первом тесте") == "В первом тесте"
    assert strip_notes_heading("NoteNote that n is even") == "Note that n is even"


def test_clean_text_fields_cleans_every_text_field_and_keeps_missing_ones():
    df = pd.DataFrame(
        {
            "prob_desc_description": ["Sort it.Then print"],
            "prob_desc_notes": [None],
            "source_code": ["a.B = 1"],
        }
    )

    cleaned = clean_text_fields(df)

    assert cleaned.loc[0, "prob_desc_description"] == "Sort it. Then print"
    assert pd.isna(cleaned.loc[0, "prob_desc_notes"])
    assert cleaned.loc[0, "source_code"] == "a.B = 1"
    assert df.loc[0, "prob_desc_description"] == "Sort it.Then print"


def test_build_statement_joins_present_fields_in_order():
    df = pd.DataFrame(
        {
            "prob_desc_output_spec": ["Print n.", "Print m."],
            "prob_desc_description": ["Find n.", "Find m."],
            "prob_desc_input_spec": ["One integer.", None],
        }
    )

    assert build_statement(df).tolist() == [
        "Find n.\n\nOne integer.\n\nPrint n.",
        "Find m.\n\nPrint m.",
    ]


def test_cyrillic_share_ignores_math_and_handles_text_without_letters():
    assert cyrillic_share("Вася $$$abc$$$") == 1.0
    assert cyrillic_share("ab") == 0.0
    assert cyrillic_share("$$$1$$$ 2 3") == 0.0


def test_strip_latex_removes_triple_dollar_math_spans():
    assert strip_latex("Given $$$n$$$ points and $$$k \\le 10$$$ lines.") == (
        "Given   points and   lines."
    )


def test_math_spans_returns_the_contents_in_order():
    assert math_spans("Given $$$n$$$ and $$$\\gcd(a, b)$$$.") == ["n", "\\gcd(a, b)"]


def test_is_russian_needs_most_letters_to_be_cyrillic():
    assert is_russian("Вася любит задачи")
    assert not is_russian("Use С++ and a simplе solution")
