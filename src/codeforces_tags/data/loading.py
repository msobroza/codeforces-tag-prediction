"""Read the challenge dataset from disk.

The dataset is a directory of ``sample_<id>.json`` files, one per exercise. ``read_samples``
returns the values as stored, and ``load_dataset`` also runs
``codeforces_tags.preprocessing.pipeline.preprocess`` on them. ``read_sample_file`` reads one
such file as ``read_samples`` reads each, and ``non_sample_files`` names the entries of a
directory that ``read_samples`` skips: any entry that is not a file named ``sample_<id>.json``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from codeforces_tags.config import DATA_DIR_ENV_VAR, DEFAULT_DATA_DIR
from codeforces_tags.preprocessing.pipeline import preprocess

_SAMPLE_FILE = re.compile(r"sample_(\d+)\.json")


def dataset_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Say where the dataset directory is, without checking that it exists.

    Args:
        data_dir: Explicit location. When omitted, ``$CODEFORCES_DATA_DIR`` is used, then
            ``DEFAULT_DATA_DIR`` from the config.

    Returns:
        The path of the dataset directory.
    """
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV_VAR, DEFAULT_DATA_DIR)
    return Path(data_dir)


def resolve_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Find the dataset directory.

    Args:
        data_dir: Explicit location, resolved by ``dataset_path``.

    Returns:
        The dataset directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    path = dataset_path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. Pass data_dir or set ${DATA_DIR_ENV_VAR}."
        )
    return path


def read_samples(data_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Read every ``sample_<id>.json`` file, without any cleaning.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.

    Returns:
        One row per sample, indexed by ``sample_id`` (the number in the file name) and sorted
        by it.

    Raises:
        FileNotFoundError: If the directory is missing or holds no sample file.
        ValueError: If two files give one sample id (``sample_7.json``, ``sample_07.json``), or
            a sample file cannot be read, is not JSON, or does not hold a JSON object of fields;
            the message names the files.
    """
    path = resolve_data_dir(data_dir)
    files = _sample_files(path)
    if not files:
        raise FileNotFoundError(f"No sample_<id>.json files in {path}")
    return _samples_frame({sample_id: _read_record(file) for sample_id, file in files.items()})


def read_sample_file(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read one ``sample_<id>.json`` file, as ``read_samples`` reads each file of a directory.

    Args:
        path: The file; its name gives the sample id.

    Returns:
        One row, indexed by ``sample_id``, with the file's fields as stored.

    Raises:
        ValueError: If the file is not named ``sample_<id>.json``, cannot be read, is not JSON,
            or does not hold a JSON object of fields; the message names the file.
    """
    path = Path(path)
    match = _SAMPLE_FILE.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"{path} is not named sample_<id>.json: name it so, since a sample's id comes from "
            "its file name."
        )
    return _samples_frame({int(match.group(1)): _read_record(path)})


def non_sample_files(data_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Name the entries of a dataset directory that ``read_samples`` skips.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.

    Returns:
        The sorted names of its entries that are not files named ``sample_<id>.json``: other
        files, and folders whatever their name.
    """
    path = resolve_data_dir(data_dir)
    return sorted(entry.name for entry in path.iterdir() if not _is_sample_file(entry))


def load_dataset(data_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Read and preprocess the dataset in one call.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.

    Returns:
        The preprocessed dataset, as returned by ``preprocess``.
    """
    return preprocess(read_samples(data_dir))


def _is_sample_file(entry: Path) -> bool:
    """Tell whether a directory entry is a file named ``sample_<id>.json``."""
    return _SAMPLE_FILE.fullmatch(entry.name) is not None and entry.is_file()


def _sample_files(path: Path) -> dict[int, Path]:
    """Find a directory's sample files by sample id; refuse two files that give one id."""
    files: dict[int, Path] = {}
    for file in sorted(filter(_is_sample_file, path.iterdir())):
        sample_id = int(_SAMPLE_FILE.fullmatch(file.name).group(1))
        if sample_id in files:
            raise ValueError(
                f"{files[sample_id].name} and {file.name} in {path} both give the sample id "
                f"{sample_id}: keep one."
            )
        files[sample_id] = file
    return files


def _read_record(path: Path) -> dict:
    """Read the fields of one sample file, or raise a ValueError that names the file."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{path} cannot be read: {error.strerror or error}.") from error
    except ValueError as error:
        raise ValueError(f"{path} is not JSON: {error}.") from error
    if not isinstance(record, dict) or not record:
        raise ValueError(f"{path} does not hold a JSON object of the exercise's fields.")
    return record


def _samples_frame(records: dict[int, dict]) -> pd.DataFrame:
    """Build the frame of the samples' fields, one row per sample, sorted by sample id."""
    # The index of from_dict is the union of each field's sample ids, first field first: a
    # sample without the first field would come last, hence the reindex.
    raw = pd.DataFrame.from_dict(records, orient="index").reindex(sorted(records))
    raw.index.name = "sample_id"
    return raw
