"""MLflow's environment and store, each set in one place (docs/adr/0002, safeguard 7).

``mlflow_env()`` is the only code that sets MLflow's environment variables, from
``config.MLFLOW_ENV``: no pickle is ever loaded, logged models get explicit requirements and no
``uv.lock``, evaluation and trace export lose no row on sqlite, and nothing is fetched from the
network. ``use_store()`` calls it, then points MLflow at a sqlite store, in WAL journal mode,
and at an experiment whose artifacts live in an explicit folder, so no ``./mlruns`` appears.

MLflow reads ``config.MLFLOW_IMPORT_ENV`` (telemetry, the hint to coding agents) only when it is
imported, so the packages whose modules import mlflow, ``tracking`` and ``tuning``, set those
with ``mlflow_env(MLFLOW_IMPORT_ENV)`` when they are imported. This module imports mlflow only
inside ``use_store``, so that they can.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Collection
from contextlib import closing
from pathlib import Path

from codeforces_tags.config import (
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    MLFLOW_METHODS_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)

_SQLITE = "sqlite:///"


def mlflow_env(names: Collection[str] | None = None) -> None:
    """Set MLflow's environment: ``config.MLFLOW_ENV``, then ``MLFLOW_ENV_DEFAULTS`` if unset.

    MLflow reads most of these variables when it uses them (e.g. the pickle flag on every model
    load), so it is enough to call this before logging or loading anything. It reads
    ``MLFLOW_IMPORT_ENV`` when it is imported: ``mlflow_env(MLFLOW_IMPORT_ENV)`` sets those first.

    Args:
        names: The variables to set, e.g. ``MLFLOW_IMPORT_ENV``; every one if None.
    """
    for name, value in MLFLOW_ENV.items():
        if names is None or name in names:
            os.environ[name] = value
    for name, value in MLFLOW_ENV_DEFAULTS.items():
        if names is None or name in names:
            os.environ.setdefault(name, value)


def use_store(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment: str = MLFLOW_METHODS_EXPERIMENT,
    artifact_dir: Path = MLFLOW_ARTIFACT_DIR,
) -> str:
    """Point MLflow at a sqlite store and an experiment, after setting its environment.

    The store is switched to WAL journal mode once, so readers (the MLflow UI) do not block the
    writer. A new experiment keeps its artifacts in ``artifact_dir``, never in ``./mlruns``.

    Args:
        tracking_uri: The store, a ``sqlite:///`` URI; MLflow stays pointed at it afterwards.
        experiment: The experiment's name, created if missing.
        artifact_dir: The folder of a new experiment's artifacts, created if missing.

    Returns:
        The experiment's id.

    Raises:
        ValueError: If the store is not a sqlite file, or if the experiment was deleted.
    """
    _check_store(tracking_uri)
    mlflow_env()
    import mlflow  # pylint: disable=import-outside-toplevel

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    found = mlflow.get_experiment_by_name(experiment)
    if found is None:
        experiment_id = mlflow.create_experiment(
            experiment, artifact_location=artifact_dir.resolve().as_uri()
        )
    elif found.lifecycle_stage == "deleted":
        raise ValueError(
            f"The experiment {experiment!r} of {tracking_uri} is deleted: restore it in the "
            "MLflow UI, or delete it for good (mlflow gc) to start it again."
        )
    else:
        experiment_id = found.experiment_id
    _use_wal(Path(tracking_uri.removeprefix(_SQLITE)))
    return experiment_id


def _check_store(tracking_uri: str) -> str:
    """Return the store's URI unchanged, or raise ``ValueError`` unless it is ``sqlite:///``."""
    if not tracking_uri.startswith(_SQLITE):
        raise ValueError(
            f"{tracking_uri} is not a sqlite:/// store: every tracking command writes to one "
            "local sqlite file (config.MLFLOW_TRACKING_URI)."
        )
    return tracking_uri


def _store_option(value: str) -> str:
    """Read ``--tracking-uri``, refusing a store that ``use_store`` would refuse."""
    try:
        return _check_store(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _use_wal(database: Path) -> None:
    """Switch a sqlite database to WAL journal mode, unless it is already in it."""
    with closing(sqlite3.connect(database)) as connection:
        if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            connection.execute("PRAGMA journal_mode=WAL")


def add_store_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that name the MLflow store to the parser of a command that logs to it.

    The parser refuses a ``--tracking-uri`` that is not ``sqlite:///``, with ``use_store``'s
    message and exit status 2, so the command does no work it could not log.

    Args:
        parser: The command's parser; it gains ``--tracking-uri`` and ``--artifact-dir``, whose
            defaults are the config's store, for ``use_store`` (``tracking_uri``, ``artifact_dir``).
    """
    parser.add_argument(
        "--tracking-uri",
        type=_store_option,
        default=MLFLOW_TRACKING_URI,
        help="MLflow store, a sqlite:/// URI",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=MLFLOW_ARTIFACT_DIR,
        help="artifact folder of a new MLflow experiment",
    )
