"""Freeze a tagging method's predictions once, with its scores and run record, and read them.

Every method's ``freeze`` writes ``assets/methods/<name>/predictions.csv`` (dev rows out of
fold, test rows once), ``scores.csv`` (score methods) and ``run_record.json``, indexed by
``(part, sample_id)``. A second ``freeze`` only checks: test predictions of another recipe
are never overwritten (ADR 0001).

Floats are written with their shortest repr and read with ``float_precision="round_trip"``,
so ``scores.csv`` reads back bit for bit (the default parser changes the last bits of about a
third of random float64 values).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from codeforces_tags.config import TUNING_EXPORT_TOLERANCE
from codeforces_tags.provenance import RunRecord

PREDICTIONS_FILE = "predictions.csv"
SCORES_FILE = "scores.csv"
RUN_RECORD_FILE = "run_record.json"
PER_PROBLEM_FILE = "per_problem.csv"
# The index of every frozen file: the part (dev or test), then the problem.
FROZEN_INDEX = ("part", "sample_id")
# The files whose presence makes a method's directory frozen: either one alone is refused.
_FROZEN_FILES = (PREDICTIONS_FILE, RUN_RECORD_FILE)
# per_problem.csv (API methods): each problem's recorded call, its columns in order.
_PER_PROBLEM_DTYPES = {
    "sample_id": "int64",
    "part": "str",
    "provider": "str",
    "input_tokens": "int64",
    "cached_tokens": "int64",
    "output_tokens": "int64",
    "input_cost_usd": "float64",
    "output_cost_usd": "float64",
    "total_cost_usd": "float64",
    "latency_s": "float64",
    "served_model": "str",
    "response_id": "str",
    "status": "str",
}
PER_PROBLEM_COLUMNS: tuple[str, ...] = tuple(_PER_PROBLEM_DTYPES)


def stack_parts(*, dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Stack the dev and test rows of a method, as its frozen files hold them.

    Args:
        dev: Dev rows (out of fold), indexed by ``sample_id``.
        test: Test rows, indexed by ``sample_id``, with the columns of ``dev`` in order.

    Returns:
        The dev rows then the test rows, indexed by ``(part, sample_id)``.

    Raises:
        ValueError: If the columns differ, or if a problem is in both parts.
    """
    if list(dev.columns) != list(test.columns):
        raise ValueError(
            f"dev and test must have the same columns, in order: {list(dev.columns)} and "
            f"{list(test.columns)}."
        )
    both = dev.index.intersection(test.index)
    if len(both):
        raise ValueError(f"These problems are in both parts: {sorted(both)}.")
    return pd.concat({"dev": dev, "test": test}, names=list(FROZEN_INDEX))


def _check_predictions(predictions: pd.DataFrame) -> None:
    if list(predictions.index.names) != list(FROZEN_INDEX):
        raise ValueError(
            "predictions must be indexed by (part, sample_id), as stack_parts gives them; they "
            f"are indexed by {list(predictions.index.names)}."
        )
    integer = all(pd.api.types.is_integer_dtype(dtype) for dtype in predictions.dtypes)
    if not integer or not predictions.isin((0, 1)).all().all():
        raise ValueError("predictions must hold int 0/1 values only.")


def _same_labels(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """Tell whether two frames have the same rows and columns, in order."""
    return first.index.equals(second.index) and list(first.columns) == list(second.columns)


def _check_scores(scores: pd.DataFrame, predictions: pd.DataFrame) -> None:
    floats = all(pd.api.types.is_float_dtype(dtype) for dtype in scores.dtypes)
    if not floats or not _same_labels(scores, predictions):
        raise ValueError(
            "scores must be float columns with the rows and columns of the predictions, in order."
        )
    if scores.isna().any().any():
        raise ValueError(
            "scores must not hold NaN: NaN never equals itself, so a later freeze could not "
            "check the frozen scores."
        )


def _matches(stored: pd.DataFrame, new: pd.DataFrame, atol: float) -> bool:
    """Tell whether two frames have the same rows and columns and values within atol (rtol 0)."""
    return _same_labels(stored, new) and np.allclose(
        stored.to_numpy(float), new.to_numpy(float), rtol=0, atol=atol
    )


def _write(
    predictions: pd.DataFrame, directory: Path, record: RunRecord, scores: pd.DataFrame | None
) -> None:
    """Write the frozen files, each under a temporary name, then rename them, the record last.

    The record is serialised before anything is written, so a record that JSON refuses leaves
    no file. A failure while writing leaves only temporary files, which the next freeze
    overwrites: never one frozen file without the other.
    """
    text = json.dumps(record.to_dict(), indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    staged = {PREDICTIONS_FILE: _temporary(directory, PREDICTIONS_FILE)}
    predictions.to_csv(staged[PREDICTIONS_FILE])
    if scores is not None:
        staged[SCORES_FILE] = _temporary(directory, SCORES_FILE)
        scores.to_csv(staged[SCORES_FILE])
    staged[RUN_RECORD_FILE] = _temporary(directory, RUN_RECORD_FILE)
    staged[RUN_RECORD_FILE].write_text(text, encoding="utf-8")
    for name, path in staged.items():
        os.replace(path, directory / name)


def _temporary(directory: Path, name: str) -> Path:
    """Return the temporary name a frozen file is written under before it is renamed."""
    return directory / f".{name}.partial"


def _check_half_frozen(directory: Path) -> None:
    """Fail when a directory holds one of predictions.csv and run_record.json without the other."""
    present = [name for name in _FROZEN_FILES if (directory / name).exists()]
    if len(present) == 1:
        [missing] = [name for name in _FROZEN_FILES if name not in present]
        raise ValueError(
            f"{directory} holds only one of {list(_FROZEN_FILES)}: {present[0]}, and no "
            f"{missing}: restore it from git (git checkout -- {directory / missing}). Delete "
            "both only if neither was ever committed: frozen test predictions are never "
            "overwritten (ADR 0001)."
        )


def _check_frozen_scores(directory: Path, scores: pd.DataFrame | None, atol: float) -> None:
    has_file = (directory / SCORES_FILE).exists()
    if scores is not None and not has_file:
        raise ValueError(f"{directory} has no {SCORES_FILE}, but scores were given.")
    if scores is None and has_file:
        raise ValueError(f"{directory} has a {SCORES_FILE}, but no scores were given.")
    if scores is not None and not _matches(load_scores(directory), scores, atol):
        raise ValueError(
            f"The scores differ from the frozen {SCORES_FILE} in {directory} by more than {atol} "
            "(rtol 0), for the same recipe: the code, data or environment changed."
        )


def freeze_predictions(
    predictions: pd.DataFrame,
    directory: Path,
    record: RunRecord,
    *,
    scores: pd.DataFrame | None = None,
    score_atol: float = 0.0,
) -> str:
    """Save a method's predictions once, with its scores and run record; later, only check.

    When ``predictions.csv`` and ``run_record.json`` are absent, it writes them, and
    ``scores.csv`` for a score method, each under a temporary name renamed once all are
    written. When both are present, it writes nothing and raises unless the stored
    ``recipe_sha256`` equals the record's, the stored predictions are identical, and the stored
    scores (present exactly when ``scores`` is given) equal ``scores`` within
    ``atol = max(score_atol, TUNING_EXPORT_TOLERANCE)``, rtol 0. When only one is present, it
    writes nothing and raises. A method with ``score_atol > 0`` never calls it a second time:
    its ``freeze`` checks the frozen files instead.

    Args:
        predictions: Int 0/1 predictions, indexed by ``(part, sample_id)`` (``stack_parts``):
            dev rows out of fold, test rows from the method fitted on dev.
        directory: The method's directory, ``method_dir(name)``; created if missing.
        record: The run record (``make_run_record``).
        scores: A score method's float scores without NaN, with the rows and columns of
            ``predictions``.
        score_atol: The method spec's ``score_atol``: 0.0 for a bit-reproducible method.

    Returns:
        A line to print: the files were saved, or they match.

    Raises:
        ValueError: If ``predictions`` or ``scores`` are malformed, if the directory holds one
            of ``predictions.csv`` and ``run_record.json`` without the other, or if the frozen
            files hold another recipe, other predictions or other scores.
        TypeError: If the record holds a value that is not JSON (e.g. a numpy scalar).
    """
    _check_predictions(predictions)
    if scores is not None:
        _check_scores(scores, predictions)
    directory = Path(directory)
    _check_half_frozen(directory)
    if not (directory / PREDICTIONS_FILE).exists():
        _write(predictions, directory, record, scores)
        what = "predictions and scores" if scores is not None else "predictions"
        return f"saved {len(predictions):,} {what} to {directory}"
    stored = load_run_record(directory).recipe_sha256
    if stored != record.recipe_sha256:
        raise ValueError(
            f"{directory} holds the predictions of another recipe ({stored}, not "
            f"{record.recipe_sha256}): frozen predictions are never overwritten (ADR 0001). "
            "Freeze the new recipe as a new method."
        )
    if not _matches(load_predictions(directory), predictions, atol=0.0):
        raise ValueError(
            f"The predictions differ from the frozen {PREDICTIONS_FILE} in {directory}, for the "
            "same recipe: the code, data or environment changed."
        )
    _check_frozen_scores(directory, scores, max(score_atol, TUNING_EXPORT_TOLERANCE))
    return f"{len(predictions):,} predictions match the frozen files in {directory}"


def _read_frozen(path: Path, dtype: str) -> pd.DataFrame:
    """Read a file indexed by (part, sample_id), its other columns of one dtype."""
    header = list(pd.read_csv(path, nrows=0).columns)
    if header[:2] != list(FROZEN_INDEX):
        raise ValueError(f"{path} must start with the columns part, sample_id; it has {header}.")
    dtypes = {"part": "str", "sample_id": "int64"} | dict.fromkeys(header[2:], dtype)
    return pd.read_csv(
        path, index_col=list(FROZEN_INDEX), dtype=dtypes, float_precision="round_trip"
    )


def load_predictions(directory: Path) -> pd.DataFrame:
    """Read a method's frozen predictions.

    Args:
        directory: The method's directory.

    Returns:
        Int64 0/1 columns, one per tag, indexed by ``(part, sample_id)``.

    Raises:
        ValueError: If the file is not indexed by part and sample id, or holds a value other
            than 0 and 1.
    """
    path = Path(directory) / PREDICTIONS_FILE
    predictions = _read_frozen(path, "int64")
    if not predictions.isin((0, 1)).all().all():
        raise ValueError(f"{path} holds values other than 0 and 1.")
    return predictions


def load_scores(directory: Path) -> pd.DataFrame:
    """Read a score method's frozen scores, bit for bit.

    Args:
        directory: The method's directory.

    Returns:
        Float64 columns, one per tag, indexed by ``(part, sample_id)``.

    Raises:
        ValueError: If the file is not indexed by part and sample id.
    """
    return _read_frozen(Path(directory) / SCORES_FILE, "float64")


def load_run_record(directory: Path) -> RunRecord:
    """Read a method's run record.

    Args:
        directory: The method's directory.

    Returns:
        The record of ``run_record.json``.

    Raises:
        ValueError: If a key of ``RUN_RECORD_KEYS`` is missing or a key is unknown.
    """
    text = (Path(directory) / RUN_RECORD_FILE).read_text(encoding="utf-8")
    return RunRecord.from_dict(json.loads(text))


def load_per_problem(directory: Path) -> pd.DataFrame:
    """Read an API method's recorded call of each problem.

    Args:
        directory: The method's directory.

    Returns:
        The columns of ``PER_PROBLEM_COLUMNS`` after ``sample_id`` and ``part``, indexed by
        ``(part, sample_id)``; tokens are int64 and costs and latencies float64, bit for bit.

    Raises:
        ValueError: If the file's columns are not ``PER_PROBLEM_COLUMNS``, in order.
    """
    path = Path(directory) / PER_PROBLEM_FILE
    header = list(pd.read_csv(path, nrows=0).columns)
    if header != list(PER_PROBLEM_COLUMNS):
        raise ValueError(
            f"{path} must have the columns {list(PER_PROBLEM_COLUMNS)}, in order; it has {header}."
        )
    per_problem = pd.read_csv(path, dtype=_PER_PROBLEM_DTYPES, float_precision="round_trip")
    return per_problem.set_index(list(FROZEN_INDEX))
