"""Preprocessing step 2: clean the free-text fields and build the text a model reads."""

from __future__ import annotations

import re

import pandas as pd

from codeforces_tags.config import RUSSIAN_LETTER_SHARE, STATEMENT_FIELDS, TEXT_FIELDS

# Codeforces statements wrap inline math in triple dollars: "$$$1 \le n \le 10^5$$$".
_MATH_SPAN = re.compile(r"\$\$\$(.*?)\$\$\$", re.DOTALL)
# Paragraphs scraped from HTML lost the space between them: "rounded up instead.You should".
GLUED_SENTENCE = re.compile(r"([a-z0-9)$])\.([A-Z])")
# Every notes field starts with its section heading, glued to the text: "NoteIn the first...".
_NOTES_HEADING = re.compile(r"^\s*(?:Note|Примечание)\s*")
_CYRILLIC_LETTER = re.compile("[А-Яа-яЁё]")
_LATIN_LETTER = re.compile("[A-Za-z]")
# Cyrillic letters that look like Latin ones, typed by mistake in English statements.
_LOOKALIKES = str.maketrans("АВЕКМНОРСТХаеорсух", "ABEKMHOPCTXaeopcyx")


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the scraping artefacts of every free-text field.

    Each field of ``TEXT_FIELDS`` goes through ``clean_text``; the notes also lose their
    heading.

    Args:
        df: Samples with typed columns.

    Returns:
        A new DataFrame; missing fields stay missing.
    """
    df = df.copy()
    for column in TEXT_FIELDS:
        if column in df:
            df[column] = df[column].map(clean_text, na_action="ignore")
    if "prob_desc_notes" in df:
        df["prob_desc_notes"] = df["prob_desc_notes"].map(strip_notes_heading, na_action="ignore")
    return df


def build_statement(df: pd.DataFrame) -> pd.Series:
    """Join the statement fields into the problem text a model reads.

    Args:
        df: Samples with any of the ``STATEMENT_FIELDS`` columns.

    Returns:
        One text per row: the fields present, in ``STATEMENT_FIELDS`` order, separated by a
        blank line.
    """
    fields = [column for column in STATEMENT_FIELDS if column in df]
    return df[fields].apply(lambda row: "\n\n".join(row.dropna()), axis=1).rename("statement")


def clean_text(text: str) -> str:
    """Fix the scraping artefacts of one text field.

    Restores the space lost between paragraphs (``"instead.You"`` becomes ``"instead. You"``).
    In English text, also replaces Cyrillic look-alike letters by their Latin twin (``"С++"``
    with a Cyrillic ``С``); Russian text is left as is.

    Args:
        text: One field as scraped.

    Returns:
        The cleaned text.
    """
    text = GLUED_SENTENCE.sub(r"\1. \2", text)
    if not is_russian(text):
        text = text.translate(_LOOKALIKES)
    return text


def strip_notes_heading(notes: str) -> str:
    """Remove the section heading that starts every notes field (``"NoteIn the first..."``)."""
    return _NOTES_HEADING.sub("", notes, count=1)


def is_russian(text: str) -> bool:
    """Tell whether a text is Russian: most of its letters outside math spans are Cyrillic."""
    return cyrillic_share(text) > RUSSIAN_LETTER_SHARE


def cyrillic_share(text: str) -> float:
    """Return the share of letters outside math spans that are Cyrillic (0 without letters)."""
    text = strip_latex(text)
    cyrillic = len(_CYRILLIC_LETTER.findall(text))
    return cyrillic / max(cyrillic + len(_LATIN_LETTER.findall(text)), 1)


def strip_latex(text: str) -> str:
    """Remove the math spans from a statement.

    Args:
        text: Statement with math wrapped in ``$$$...$$$``.

    Returns:
        The text with every math span replaced by a space.
    """
    return _MATH_SPAN.sub(" ", text)


def math_spans(text: str) -> list[str]:
    """Return the contents of the ``$$$...$$$`` math spans, in order of appearance."""
    return _MATH_SPAN.findall(text)
