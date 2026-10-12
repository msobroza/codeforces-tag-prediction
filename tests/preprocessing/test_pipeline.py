"""Tests for the preprocessing pipeline."""

import pandas as pd
import pytest

from codeforces_tags.config import RAW_INPUT_COLUMNS
from codeforces_tags.data.loading import read_samples
from codeforces_tags.preprocessing.pipeline import preprocess, raw_to_problems
from codeforces_tags.splitting import select_part


def test_preprocess_runs_every_step_and_leaves_the_raw_frame_unchanged(write_samples, make_record):
    raw = read_samples(
        write_samples(
            {
                0: make_record(prob_desc_description="Sort it.Then print", lang="PyPy 2"),
                1: make_record(difficulty=-1, tags=["math", "math"]),
            }
        )
    )

    df = preprocess(raw)

    assert df.loc[0, "statement"] == "Sort it. Then print"
    assert df["python_version"].tolist() == [2, 3]
    assert df.loc[1, "tags"] == ["math"]
    assert pd.isna(df.loc[1, "difficulty"])
    assert raw.loc[0, "prob_desc_description"] == "Sort it.Then print"


def test_preprocess_accepts_input_without_a_runtime_column(write_samples, make_record):
    raw = read_samples(write_samples({0: make_record()})).drop(columns=["lang"])

    df = preprocess(raw)

    assert "python_version" not in df
    assert df.loc[0, "statement"] == "Find $$$n$$$."


def test_raw_to_problems_preprocesses_raw_problems(write_samples, make_raw_record):
    raw = read_samples(write_samples({0: make_raw_record(), 1: make_raw_record()}))

    problems = raw_to_problems(raw)

    pd.testing.assert_frame_equal(problems, preprocess(raw))
    assert problems.loc[0, "statement"] == (
        "Find $$$n$$$.\n\nOne integer $$$n$$$.\n\nPrint one integer."
    )


def test_raw_to_problems_names_the_missing_raw_columns(write_samples, make_raw_record):
    raw = read_samples(write_samples({0: make_raw_record()}))

    with pytest.raises(ValueError, match=r"\['prob_desc_notes', 'source_code'\]"):
        raw_to_problems(raw.drop(columns=["source_code", "prob_desc_notes"]))


def test_raw_to_problems_refuses_preprocessed_problems(write_samples, make_raw_record):
    raw = read_samples(write_samples({0: make_raw_record()}))

    with pytest.raises(ValueError, match="already preprocessed"):
        raw_to_problems(preprocess(raw))


def test_raw_to_problems_keeps_extra_columns_and_nan_fields(write_samples, make_raw_record):
    # A raw frame as select_part(read_samples(), split, part) returns it: fold, group and tags
    # columns, and missing notes and input specification (JSON null, read as NaN).
    raw = read_samples(
        write_samples(
            {
                0: make_raw_record(),
                1: make_raw_record(prob_desc_notes=None),
                2: make_raw_record(prob_desc_input_spec=None),
            }
        )
    )
    split = pd.DataFrame(
        {"part": ["dev", "dev", "test"], "fold": [0, 1, -1], "group": [0, 1, 2]},
        index=pd.Index([0, 1, 2], name="sample_id"),
    )
    raw_parts = pd.concat([select_part(raw, split, "dev"), select_part(raw, split, "test")])
    before = raw_parts.copy()

    problems = raw_to_problems(raw_parts)

    assert set(RAW_INPUT_COLUMNS) <= set(raw_parts.columns)
    assert problems["fold"].tolist() == [0, 1, -1]
    assert problems["group"].tolist() == [0, 1, 2]
    assert problems["tags"].tolist() == [["math", "greedy"]] * 3
    assert pd.isna(problems.loc[1, "prob_desc_notes"])
    assert problems.loc[2, "statement"] == "Find $$$n$$$.\n\nPrint one integer."
    pd.testing.assert_frame_equal(raw_parts, before)
