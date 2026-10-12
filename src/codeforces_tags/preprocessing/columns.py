"""Preprocessing step 1: convert raw fields to typed columns."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

# Limits are free text scraped from Codeforces, partly in Russian ("2 секунды", "256 мегабайт").
_TIME_LIMIT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?|s|секунд[аы]?)", re.IGNORECASE)
_MEMORY_LIMIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:megabytes?|mebibytes?|mb|мегабайт[а]?)", re.IGNORECASE
)


def clean_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw samples to typed columns.

    Columns converted under the same name:

    - ``tags``: repeated tags dropped (order kept); a missing list becomes ``[]``.
    - ``difficulty``: nullable ``Int64``, with ``<NA>`` for unknown ratings.
    - ``prob_desc_created_at``: UTC datetime.
    - ``prob_desc_sample_inputs`` and ``prob_desc_sample_outputs``: lists of strings.

    Columns added: ``time_limit_s`` and ``memory_limit_mb``, parsed from the free-text limits.

    Missing values stay missing instead of raising. Columns absent from ``raw``, such as the
    labels of a test file, are skipped.

    Args:
        raw: Samples as read by ``codeforces_tags.data.loading.read_samples``.

    Returns:
        A new DataFrame; ``raw`` is left unchanged.
    """
    converters = {
        "tags": _dedupe_tags,
        "difficulty": clean_difficulty,
        "prob_desc_created_at": _unix_seconds_to_utc,
        "prob_desc_sample_inputs": _parse_json_lists,
        "prob_desc_sample_outputs": _parse_json_lists,
    }
    parsed_limits = {
        "time_limit_s": ("prob_desc_time_limit", parse_time_limit),
        "memory_limit_mb": ("prob_desc_memory_limit", parse_memory_limit),
    }

    df = raw.copy()
    for column, convert in converters.items():
        if column in df:
            df[column] = convert(df[column])
    for new_column, (source, parse) in parsed_limits.items():
        if source in df:
            df[new_column] = df[source].map(parse)
    return df


def clean_difficulty(difficulty: pd.Series) -> pd.Series:
    """Mark unknown difficulty ratings as missing.

    Codeforces ratings start at 800. ``None`` and non-positive sentinels such as ``-1`` mean
    that the rating is unknown.

    Args:
        difficulty: Raw ratings.

    Returns:
        Ratings as nullable ``Int64``, with ``<NA>`` where unknown.
    """
    numeric = pd.to_numeric(difficulty, errors="coerce")
    return numeric.where(numeric > 0).astype("Int64")


def parse_time_limit(text: object) -> float:
    """Parse a free-text time limit.

    Args:
        text: Limit as scraped, e.g. ``"2 seconds"``, ``"2.0 s"`` or ``"2 секунды"``.

    Returns:
        The limit in seconds, or NaN if ``text`` is missing or unparseable.
    """
    return _parse_quantity(_TIME_LIMIT, text)


def parse_memory_limit(text: object) -> float:
    """Parse a free-text memory limit.

    Units are taken at face value: Codeforces uses megabytes and mebibytes interchangeably.

    Args:
        text: Limit as scraped, e.g. ``"256 megabytes"``, ``"512 MB"`` or ``"256 мегабайт"``.

    Returns:
        The limit in megabytes, or NaN if ``text`` is missing or unparseable.
    """
    return _parse_quantity(_MEMORY_LIMIT, text)


def _parse_quantity(pattern: re.Pattern[str], text: object) -> float:
    """Return the number captured by ``pattern`` in ``text``, or NaN."""
    if not isinstance(text, str):  # None, NaN or pd.NA from a missing field
        return np.nan
    match = pattern.fullmatch(text.strip())
    return float(match.group(1)) if match else np.nan


def _dedupe_tags(tags: pd.Series) -> pd.Series:
    """Drop repeated tags in each list, keeping their order; a missing list becomes ``[]``."""
    return tags.map(lambda row: list(dict.fromkeys(row)) if isinstance(row, list) else [])


def _unix_seconds_to_utc(seconds: pd.Series) -> pd.Series:
    """Convert Unix timestamps (numbers or strings) to UTC datetimes."""
    return pd.to_datetime(pd.to_numeric(seconds, errors="coerce"), unit="s", utc=True)


def _parse_json_lists(column: pd.Series) -> pd.Series:
    """Decode JSON-encoded lists, leaving missing values missing."""
    return column.map(json.loads, na_action="ignore")
