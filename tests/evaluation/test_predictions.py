"""Tests for the frozen predictions, scores and run records of a tagging method."""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.evaluation.predictions import (
    FROZEN_INDEX,
    PER_PROBLEM_COLUMNS,
    PER_PROBLEM_FILE,
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    freeze_predictions,
    load_per_problem,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)

TAGS = ["math", "number theory"]


def frame(values, ids):
    """Return a frame of the two tags' columns, indexed by sample id."""
    return pd.DataFrame(values, columns=TAGS, index=pd.Index(ids, name="sample_id"))


@pytest.fixture(name="predictions")
def predictions_fixture():
    """Return 0/1 predictions of two dev problems and one test problem."""
    return stack_parts(dev=frame([[1, 0], [0, 1]], [3, 1]), test=frame([[0, 1]], [7]))


@pytest.fixture(name="scores")
def scores_fixture(predictions):
    """Return float scores with the rows and columns of the predictions."""
    values = np.random.default_rng(0).normal(size=predictions.shape)
    return pd.DataFrame(values, index=predictions.index, columns=predictions.columns)


@pytest.fixture(name="record")
def record_fixture(placeholder_record):
    """Return a run record whose thresholds hold an infinity (a tag absent from a fold)."""
    return placeholder_record(
        method_entries={"thresholds": {"math": 0.25, "number theory": math.inf}}
    )


def frozen_bytes(directory):
    """Return the bytes of every file in ``directory``, by name."""
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_stack_parts_indexes_dev_then_test_by_part_and_sample_id(predictions):
    assert list(predictions.index.names) == list(FROZEN_INDEX) == ["part", "sample_id"]
    assert predictions.index.tolist() == [("dev", 3), ("dev", 1), ("test", 7)]
    assert list(predictions.columns) == TAGS


def test_stack_parts_refuses_parts_with_other_columns():
    with pytest.raises(ValueError, match="same columns"):
        stack_parts(dev=frame([[1, 0]], [3]), test=frame([[0, 1]], [7])[TAGS[::-1]])


def test_stack_parts_refuses_a_problem_in_both_parts():
    with pytest.raises(ValueError, match=r"both parts: \[3\]"):
        stack_parts(dev=frame([[1, 0]], [3]), test=frame([[0, 1]], [3]))


def test_freeze_predictions_writes_predictions_scores_and_run_record(
    predictions, scores, record, tmp_path
):
    message = freeze_predictions(predictions, tmp_path / "label_prior", record, scores=scores)

    assert message.startswith("saved 3 predictions and scores to ")
    directory = tmp_path / "label_prior"
    assert sorted(frozen_bytes(directory)) == [PREDICTIONS_FILE, RUN_RECORD_FILE, SCORES_FILE]
    pd.testing.assert_frame_equal(load_predictions(directory), predictions)
    pd.testing.assert_frame_equal(load_scores(directory), scores)
    assert load_run_record(directory) == record


def test_freeze_predictions_writes_infinite_thresholds_as_json_strings(
    predictions, record, tmp_path
):
    freeze_predictions(predictions, tmp_path, record)

    text = (tmp_path / RUN_RECORD_FILE).read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert json.loads(text)["method_entries"]["thresholds"] == {
        "math": 0.25,
        "number theory": "inf",
    }


def test_freeze_predictions_of_a_0_1_method_writes_no_scores(predictions, record, tmp_path):
    message = freeze_predictions(predictions, tmp_path, record)

    assert message.startswith("saved 3 predictions to ")
    assert not (tmp_path / SCORES_FILE).exists()


def test_freeze_predictions_twice_only_checks(predictions, scores, record, tmp_path):
    freeze_predictions(predictions, tmp_path, record, scores=scores)
    before = frozen_bytes(tmp_path)

    message = freeze_predictions(
        predictions, tmp_path, replace(record, created_at="later"), scores=scores
    )

    assert "match the frozen files" in message
    assert frozen_bytes(tmp_path) == before


def test_freeze_predictions_never_overwrites_another_recipe(predictions, record, tmp_path):
    freeze_predictions(predictions, tmp_path, record)
    before = frozen_bytes(tmp_path)

    with pytest.raises(ValueError, match="another recipe"):
        freeze_predictions(1 - predictions, tmp_path, replace(record, recipe_sha256="f" * 64))
    assert frozen_bytes(tmp_path) == before


@pytest.mark.parametrize("missing", [PREDICTIONS_FILE, RUN_RECORD_FILE])
def test_freeze_predictions_refuses_a_directory_that_holds_one_frozen_file(
    predictions, record, tmp_path, missing
):
    # Half frozen (a file deleted, or a freeze stopped between its renames): neither taken for
    # empty, which would overwrite the other file, nor for frozen.
    freeze_predictions(predictions, tmp_path, record)
    (tmp_path / missing).unlink()
    before = frozen_bytes(tmp_path)

    with pytest.raises(ValueError, match=f"holds only one of .* no {missing}: restore it"):
        freeze_predictions(1 - predictions, tmp_path, replace(record, recipe_sha256="f" * 64))
    assert frozen_bytes(tmp_path) == before


def test_freeze_predictions_writes_nothing_for_a_record_json_refuses(predictions, record, tmp_path):
    # A numpy scalar read from a pandas row: the record is serialised before any file is
    # written, so a corrected record can be frozen next.
    bad = replace(record, method_entries={"dev_in_sample": np.bool_(False)})

    with pytest.raises(TypeError, match="bool"):
        freeze_predictions(predictions, tmp_path / "label_prior", bad)

    assert not (tmp_path / "label_prior").exists()
    assert freeze_predictions(predictions, tmp_path / "label_prior", record).startswith("saved")


def test_freeze_predictions_that_fails_while_writing_leaves_no_frozen_file(
    predictions, scores, record, tmp_path, monkeypatch
):
    # A full disk while scores.csv is written: predictions.csv alone would be taken for a
    # frozen directory by every later freeze.
    calls = []
    to_csv = pd.DataFrame.to_csv

    def full_disk(written, *args, **kwargs):
        calls.append(written)
        if len(calls) == 2:
            raise OSError("No space left on device")
        return to_csv(written, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", full_disk)
    with pytest.raises(OSError, match="No space left"):
        freeze_predictions(predictions, tmp_path, record, scores=scores)
    monkeypatch.undo()

    assert not {PREDICTIONS_FILE, SCORES_FILE, RUN_RECORD_FILE} & set(frozen_bytes(tmp_path))
    freeze_predictions(predictions, tmp_path, record, scores=scores)
    assert sorted(frozen_bytes(tmp_path)) == [PREDICTIONS_FILE, RUN_RECORD_FILE, SCORES_FILE]


def test_freeze_predictions_refuses_other_predictions_of_the_same_recipe(
    predictions, record, tmp_path
):
    freeze_predictions(predictions, tmp_path, record)
    changed = predictions.copy()
    changed.loc[("test", 7), "math"] = 1

    with pytest.raises(ValueError, match="predictions differ"):
        freeze_predictions(changed, tmp_path, record)


def test_freeze_predictions_compares_scores_at_the_export_tolerance(
    predictions, scores, record, tmp_path
):
    freeze_predictions(predictions, tmp_path, record, scores=scores)

    freeze_predictions(predictions, tmp_path, record, scores=scores + 5e-10)
    with pytest.raises(ValueError, match=r"scores differ .* more than 1e-09"):
        freeze_predictions(predictions, tmp_path, record, scores=scores + 1e-6)


def test_freeze_predictions_compares_scores_at_the_method_score_atol(
    predictions, scores, record, tmp_path
):
    freeze_predictions(predictions, tmp_path, record, scores=scores)

    freeze_predictions(predictions, tmp_path, record, scores=scores + 1e-6, score_atol=1e-5)
    with pytest.raises(ValueError, match="more than 1e-05"):
        freeze_predictions(predictions, tmp_path, record, scores=scores + 1e-4, score_atol=1e-5)


def test_freeze_predictions_refuses_scores_that_appear_or_disappear(
    predictions, scores, record, tmp_path
):
    freeze_predictions(predictions, tmp_path / "without", record)
    freeze_predictions(predictions, tmp_path / "with", record, scores=scores)

    with pytest.raises(ValueError, match="has no scores.csv"):
        freeze_predictions(predictions, tmp_path / "without", record, scores=scores)
    with pytest.raises(ValueError, match="has a scores.csv"):
        freeze_predictions(predictions, tmp_path / "with", record)


def test_freeze_predictions_refuses_predictions_not_indexed_by_part_and_sample_id(
    predictions, record, tmp_path
):
    with pytest.raises(ValueError, match=r"indexed by \(part, sample_id\)"):
        freeze_predictions(predictions.droplevel("part"), tmp_path, record)


@pytest.mark.parametrize("factor", [2, 1.0])
def test_freeze_predictions_refuses_anything_but_int_0_1_values(
    predictions, record, tmp_path, factor
):
    with pytest.raises(ValueError, match="int 0/1"):
        freeze_predictions(predictions * factor, tmp_path, record)
    assert not tmp_path.joinpath(PREDICTIONS_FILE).exists()


def test_freeze_predictions_refuses_scores_of_other_rows_or_columns(
    predictions, scores, record, tmp_path
):
    with pytest.raises(ValueError, match="rows and columns of the predictions"):
        freeze_predictions(predictions, tmp_path, record, scores=scores.iloc[:2])
    with pytest.raises(ValueError, match="rows and columns of the predictions"):
        freeze_predictions(predictions, tmp_path, record, scores=scores.astype(int))


def test_freeze_predictions_refuses_scores_with_nan(predictions, scores, record, tmp_path):
    # NaN never equals NaN: a frozen NaN would make every later freeze report a change.
    with pytest.raises(ValueError, match="scores must not hold NaN"):
        freeze_predictions(predictions, tmp_path, record, scores=scores.where(scores > 0))
    assert not tmp_path.joinpath(PREDICTIONS_FILE).exists()


def test_freeze_predictions_twice_accepts_infinite_scores(predictions, scores, record, tmp_path):
    infinite = scores.where(scores > 0, -np.inf)
    freeze_predictions(predictions, tmp_path, record, scores=infinite)

    assert "match the frozen files" in freeze_predictions(
        predictions, tmp_path, record, scores=infinite
    )


def test_load_scores_reads_random_float64_scores_back_exactly(record, tmp_path):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 2)) * 10.0 ** rng.integers(-8, 8, size=(500, 2))
    scores = stack_parts(
        dev=frame(values[:400], range(400)), test=frame(values[400:], range(400, 500))
    )
    predictions = (scores >= 0).astype(int)

    freeze_predictions(predictions, tmp_path, record, scores=scores)

    loaded = load_scores(tmp_path)
    assert (loaded.to_numpy() == values).all()
    assert loaded.dtypes.tolist() == ["float64", "float64"]


def test_load_predictions_reads_int_0_1_columns(predictions, record, tmp_path):
    freeze_predictions(predictions, tmp_path, record)

    loaded = load_predictions(tmp_path)

    assert loaded.dtypes.tolist() == ["int64", "int64"]
    assert loaded.index.get_level_values("sample_id").dtype == "int64"


def test_load_predictions_refuses_values_other_than_0_and_1(tmp_path):
    (tmp_path / PREDICTIONS_FILE).write_text("part,sample_id,math\ndev,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="other than 0 and 1"):
        load_predictions(tmp_path)


def test_load_predictions_refuses_a_file_without_part_and_sample_id(tmp_path):
    (tmp_path / PREDICTIONS_FILE).write_text("sample_id,math\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="start with the columns part, sample_id"):
        load_predictions(tmp_path)


def test_per_problem_columns_are_the_recorded_call_of_each_problem():
    assert PER_PROBLEM_COLUMNS == (
        "sample_id",
        "part",
        "provider",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
        "latency_s",
        "served_model",
        "response_id",
        "status",
    )


def test_load_per_problem_reads_each_problem_call_by_part_and_sample_id(tmp_path):
    rows = [
        [3, "dev", "openai", 900, 0, 12, 0.1 + 0.2, 1e-5, 0.30001, 1.25, "gpt-6-luna", "r1", "ok"],
        [7, "test", "openai", 800, 512, 10, 0.002, 2e-5, 0.00202, 0.75, "gpt-6-luna", "r2", "ok"],
    ]
    pd.DataFrame(rows, columns=PER_PROBLEM_COLUMNS).to_csv(tmp_path / PER_PROBLEM_FILE, index=False)

    per_problem = load_per_problem(tmp_path)

    assert per_problem.index.tolist() == [("dev", 3), ("test", 7)]
    assert per_problem.loc[("dev", 3), "input_cost_usd"] == 0.1 + 0.2
    assert per_problem["cached_tokens"].dtype == "int64"
    assert list(per_problem.columns) == list(PER_PROBLEM_COLUMNS[2:])


def test_load_per_problem_refuses_other_columns(tmp_path):
    pd.DataFrame({"sample_id": [3], "part": ["dev"]}).to_csv(
        tmp_path / PER_PROBLEM_FILE, index=False
    )

    with pytest.raises(ValueError, match="must have the columns"):
        load_per_problem(tmp_path)
