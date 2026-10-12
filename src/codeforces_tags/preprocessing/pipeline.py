"""Run every preprocessing step, in order."""

from __future__ import annotations

import pandas as pd

from codeforces_tags.config import RAW_INPUT_COLUMNS
from codeforces_tags.preprocessing.columns import clean_columns
from codeforces_tags.preprocessing.source_code import python_major_version
from codeforces_tags.preprocessing.text import build_statement, clean_text_fields


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw samples into clean, model-ready rows.

    Steps, in order:

    1. ``clean_columns``: typed tags, difficulty, dates, limits and examples.
    2. ``clean_text_fields``: fix the scraping artefacts of the free-text fields.
    3. Add ``statement``, the problem text a model reads, and ``python_version`` (2 or 3).

    Each row is processed on its own, so the same function serves the training set and a
    single exercise to predict.

    Args:
        raw: Samples as read by ``codeforces_tags.data.loading.read_samples``.

    Returns:
        A new DataFrame; ``raw`` is left unchanged.
    """
    df = clean_text_fields(clean_columns(raw))
    df["statement"] = build_statement(df)
    if "lang" in df:
        df["python_version"] = df["lang"].map(python_major_version, na_action="ignore")
    return df


def raw_to_problems(raw_problems: pd.DataFrame) -> pd.DataFrame:
    """Check raw problems, then preprocess them: the first step of every tagging method.

    ``preprocess`` silently skips a missing column and is not idempotent, so a tagging method
    calls this function instead: it refuses what ``preprocess`` would get wrong. Its import
    path is frozen: saved tag models refer to it by that path.

    Args:
        raw_problems: Rows of ``codeforces_tags.data.loading.read_samples``, indexed by
            ``sample_id``. Other columns (e.g. the ``fold`` and ``group`` that ``select_part``
            appends, or ``tags``) pass through.

    Returns:
        ``preprocess(raw_problems)``; ``raw_problems`` is left unchanged.

    Raises:
        ValueError: If the problems are already preprocessed (they have a ``statement``
            column), or if a column of ``RAW_INPUT_COLUMNS`` is missing.
    """
    if "statement" in raw_problems:
        raise ValueError(
            "These problems are already preprocessed (they have a statement column): pass raw "
            "rows, as read_samples() returns them, not load_dataset()'s."
        )
    missing = [column for column in RAW_INPUT_COLUMNS if column not in raw_problems]
    if missing:
        raise ValueError(
            f"The raw problems lack the columns {sorted(missing)}: pass rows as read_samples() "
            "returns them."
        )
    return preprocess(raw_problems)
