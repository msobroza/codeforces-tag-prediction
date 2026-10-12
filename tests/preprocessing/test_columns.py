"""Tests for preprocessing step 1, typed columns."""

import math

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.data.loading import read_samples
from codeforces_tags.preprocessing.columns import (
    clean_columns,
    clean_difficulty,
    parse_memory_limit,
    parse_time_limit,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2 seconds", 2.0),
        ("1 second", 1.0),
        ("0.5 second", 0.5),
        ("2.0 s", 2.0),
        ("2 секунды", 2.0),
        ("1 секунда", 1.0),
    ],
)
def test_parse_time_limit_handles_every_observed_format(raw, expected):
    assert parse_time_limit(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("256 megabytes", 256.0),
        ("256 мегабайт", 256.0),
        ("512 MB", 512.0),
        ("256 mebibytes", 256.0),
    ],
)
def test_parse_memory_limit_handles_every_observed_format(raw, expected):
    assert parse_memory_limit(raw) == expected


@pytest.mark.parametrize("raw", [None, np.nan, pd.NA, "", "fast", "2 hours"])
def test_unparseable_limits_become_nan(raw):
    assert math.isnan(parse_time_limit(raw))
    assert math.isnan(parse_memory_limit(raw))


def test_clean_difficulty_treats_none_and_non_positive_sentinels_as_missing():
    cleaned = clean_difficulty(pd.Series([800, None, -1, 0, 3500]))

    assert str(cleaned.dtype) == "Int64"
    assert cleaned.isna().tolist() == [False, True, True, True, False]
    assert cleaned.dropna().tolist() == [800, 3500]


def test_clean_columns_types_columns_without_touching_the_raw_frame(write_samples, make_record):
    data_dir = write_samples(
        {0: make_record(tags=["greedy", "math", "greedy"]), 1: make_record(difficulty=-1, tags=[])}
    )
    raw = read_samples(data_dir)

    df = clean_columns(raw)

    assert df["tags"].tolist() == [["greedy", "math"], []]
    assert df["difficulty"].isna().tolist() == [False, True]
    assert df.loc[0, "prob_desc_created_at"] == pd.Timestamp("2020-10-11 09:05:00", tz="UTC")
    assert df.loc[0, "prob_desc_sample_inputs"] == ["1\n4"]
    assert df["time_limit_s"].tolist() == [2.0, 2.0]
    assert df["memory_limit_mb"].tolist() == [256.0, 256.0]
    assert raw.loc[1, "difficulty"] == -1
    assert raw.loc[0, "prob_desc_sample_inputs"] == '["1\\n4"]'


def test_clean_columns_keeps_missing_values_missing_instead_of_raising(write_samples, make_record):
    nulls = dict.fromkeys(
        [
            "prob_desc_time_limit",
            "prob_desc_memory_limit",
            "prob_desc_sample_inputs",
            "prob_desc_created_at",
            "difficulty",
            "tags",
        ]
    )
    data_dir = write_samples({0: make_record(), 1: make_record(**nulls)})

    missing = clean_columns(read_samples(data_dir)).loc[1]

    assert math.isnan(missing["time_limit_s"])
    assert math.isnan(missing["memory_limit_mb"])
    assert pd.isna(missing["prob_desc_sample_inputs"])
    assert pd.isna(missing["prob_desc_created_at"])
    assert pd.isna(missing["difficulty"])
    assert missing["tags"] == []


def test_clean_columns_skips_columns_absent_from_unlabelled_input(write_samples, make_record):
    raw = read_samples(write_samples({0: make_record()}))

    df = clean_columns(raw.drop(columns=["tags", "difficulty"]))

    assert "tags" not in df
    assert df.loc[0, "time_limit_s"] == 2.0


def test_clean_columns_skips_limits_absent_from_the_input(write_samples, make_record):
    raw = read_samples(write_samples({0: make_record()}))

    df = clean_columns(raw.drop(columns=["prob_desc_time_limit"]))

    assert "time_limit_s" not in df
    assert df.loc[0, "memory_limit_mb"] == 256.0
