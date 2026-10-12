"""Tests for the dataset as a tagging method's commands read it."""

import argparse

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS, METHODS_ASSETS_DIR, SPLIT_FILE
from codeforces_tags.methods.parts import Parts, add_dataset_arguments, read_parts


def test_read_parts_splits_the_raw_problems_and_labels_every_problem(
    write_samples, make_raw_record, tmp_path
):
    dataset = write_samples(
        {
            1: make_raw_record(tags=["math", "greedy"]),
            2: make_raw_record(tags=["graphs"]),
            3: make_raw_record(tags=["trees", "trees"]),
        }
    )
    split = pd.DataFrame(
        {"part": ["dev", "test", "dev"], "fold": [0, -1, 1], "group": [1, 2, 3]},
        index=pd.Index([1, 2, 3], name="sample_id"),
    )
    split.to_csv(tmp_path / "split.csv")

    parts = read_parts(dataset, tmp_path / "split.csv")

    assert isinstance(parts, Parts)
    assert list(parts.split.index) == [1, 2, 3]
    assert (list(parts.dev.index), list(parts.test.index)) == ([1, 3], [2])
    assert list(parts.dev["fold"]) == [0, 1]  # select_part appends the fold and the group
    assert "statement" not in parts.dev  # raw problems, as tagging methods take them
    assert list(parts.labels.columns) == list(FOCUS_TAGS)
    assert parts.labels.loc[3, "trees"] == 1  # the repeated tag counts once
    assert parts.labels.loc[1].sum() == 1  # "greedy" is not a focus tag


def test_add_dataset_arguments_names_the_dataset_the_split_and_the_frozen_methods(tmp_path):
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)

    defaults = parser.parse_args([])
    given = parser.parse_args(["--data", "d", "--split", "s.csv", "--assets", str(tmp_path)])

    assert (defaults.data, defaults.split, defaults.assets) == (
        None,
        SPLIT_FILE,
        METHODS_ASSETS_DIR,
    )
    assert (given.data, str(given.split), given.assets) == ("d", "s.csv", tmp_path)
