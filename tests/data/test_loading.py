"""Tests for reading the dataset from disk."""

from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.config import DATA_DIR_ENV_VAR
from codeforces_tags.data.loading import (
    dataset_path,
    load_dataset,
    non_sample_files,
    read_sample_file,
    read_samples,
    resolve_data_dir,
)


def test_resolve_data_dir_prefers_argument_then_env_var(tmp_path, monkeypatch):
    from_env = tmp_path / "from_env"
    from_env.mkdir()
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(from_env))

    assert resolve_data_dir(tmp_path) == tmp_path
    assert resolve_data_dir() == from_env


def test_resolve_data_dir_raises_with_a_hint_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=DATA_DIR_ENV_VAR):
        resolve_data_dir(tmp_path / "missing")


def test_dataset_path_resolves_as_resolve_data_dir_without_checking(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path / "from_env"))

    assert dataset_path(tmp_path / "missing") == tmp_path / "missing"
    assert dataset_path() == tmp_path / "from_env"


def test_read_samples_indexes_by_sample_id_and_ignores_other_files(write_samples, make_record):
    data_dir = write_samples({10: make_record(src_uid="b"), 2: make_record(src_uid="a")})
    (data_dir / "notes.json").write_text("{}", encoding="utf-8")

    raw = read_samples(data_dir)

    assert raw.index.name == "sample_id"
    assert raw.index.tolist() == [2, 10]
    assert raw["src_uid"].tolist() == ["a", "b"]
    assert raw.loc[2, "tags"] == ["math", "greedy"]


def test_read_samples_raises_when_directory_has_no_samples(tmp_path):
    with pytest.raises(FileNotFoundError, match="sample_"):
        read_samples(tmp_path)


def test_load_dataset_reads_and_preprocesses_in_one_call(write_samples, make_record):
    data_dir = write_samples({3: make_record(difficulty=-1)})

    df = load_dataset(data_dir)

    assert df.index.tolist() == [3]
    assert pd.isna(df.loc[3, "difficulty"])
    assert df.loc[3, "statement"].startswith("Find $$$n$$$.")


def test_read_sample_file_reads_one_sample_as_read_samples_reads_its_directory(
    write_samples, make_record
):
    data_dir = write_samples({7: make_record(prob_desc_notes=None)})

    raw = read_sample_file(data_dir / "sample_7.json")

    pd.testing.assert_frame_equal(raw, read_samples(data_dir))
    assert raw.index.tolist() == [7]
    assert raw.loc[7, "prob_desc_notes"] is None


def test_read_sample_file_refuses_a_file_whose_name_gives_no_sample_id(tmp_path):
    path = tmp_path / "problem.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"problem\.json is not named sample_<id>\.json"):
        read_sample_file(path)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", r"sample_3\.json is not JSON: Expecting property name"),
        ("[1, 2]", r"sample_3\.json does not hold a JSON object of the exercise's fields"),
        ("null", r"sample_3\.json does not hold a JSON object of the exercise's fields"),
        ("{}", r"sample_3\.json does not hold a JSON object of the exercise's fields"),
    ],
)
def test_a_sample_file_that_is_not_an_object_with_fields_is_named(
    write_samples, make_record, content, match
):
    # Without the check, a file that is not JSON raised a bare JSONDecodeError that named no file,
    # and an empty object read as no row at all.
    data_dir = write_samples({2: make_record()})
    (data_dir / "sample_3.json").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_sample_file(data_dir / "sample_3.json")
    with pytest.raises(ValueError, match=match):
        read_samples(data_dir)


def test_non_sample_files_names_the_entries_that_read_samples_skips(write_samples, make_record):
    data_dir = write_samples({1: make_record()})
    for name in ("README.md", "sample_2.json.bak", "problem.json"):
        (data_dir / name).write_text("{}", encoding="utf-8")
    (data_dir / "nested").mkdir()

    skipped = ["README.md", "nested", "problem.json", "sample_2.json.bak"]
    assert non_sample_files(data_dir) == skipped
    assert read_samples(data_dir).index.tolist() == [1]


def test_read_samples_sorts_samples_that_hold_different_fields_by_sample_id(
    write_samples, make_record
):
    # The frame's index was the union of each field's sample ids, from the first field on: 5,
    # which lacks src_uid, came after 3 and 9.
    data_dir = write_samples({3: make_record(), 5: {"source_code": "print(5)"}, 9: make_record()})

    assert read_samples(data_dir).index.tolist() == [3, 5, 9]


def test_read_samples_refuses_two_files_that_give_one_sample_id(write_samples, make_record):
    data_dir = write_samples({7: make_record()})
    (data_dir / "sample_07.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_samples(data_dir)

    assert str(error.value) == (
        f"sample_07.json and sample_7.json in {data_dir} both give the sample id 7: keep one."
    )


def test_a_folder_named_like_a_sample_file_is_skipped_and_named(write_samples, make_record):
    data_dir = write_samples({1: make_record()})
    (data_dir / "sample_2.json").mkdir()

    assert read_samples(data_dir).index.tolist() == [1]
    assert non_sample_files(data_dir) == ["sample_2.json"]


def test_a_sample_file_that_cannot_be_read_is_named(write_samples, make_record, monkeypatch):
    path = write_samples({4: make_record()}) / "sample_4.json"

    def refuse(self, *_, **__):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_text", refuse)

    with pytest.raises(ValueError, match=r"sample_4\.json cannot be read: Permission denied\.$"):
        read_sample_file(path)
