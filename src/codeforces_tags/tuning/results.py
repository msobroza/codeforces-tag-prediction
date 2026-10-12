"""The frozen results of the tuning study, which ``tuning freeze`` and notebook 04 read.

``python -m codeforces_tags.tuning export`` saves them in ``assets/tfidf_baseline/`` as JSON
and CSV, so that reading them needs neither optuna nor mlflow, nor the gitignored
``experiments/`` folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from codeforces_tags.config import TFIDF_BASELINE_DIR
from codeforces_tags.model.params import ModelParams

PARAMS_FILE = "params.json"
THRESHOLDS_FILE = "thresholds.json"
TRIALS_FILE = "trials.csv"
IMPORTANCES_FILE = "importances.csv"
OUT_OF_FOLD_FILE = "out_of_fold.csv"
RESULT_FILES = (PARAMS_FILE, THRESHOLDS_FILE, TRIALS_FILE, IMPORTANCES_FILE, OUT_OF_FOLD_FILE)


@dataclass(frozen=True)
class TuningResults:
    """The configuration the study chose, and how it was chosen.

    Attributes:
        params: The configuration of the best trial.
        thresholds: Its final per-tag thresholds, chosen on all its out-of-fold scores (index
            ``label``: the tags; name ``threshold``); ``inf`` for a tag no dev problem has,
            written ``Infinity`` in JSON.
        trials: One row per trial: ``number``, ``state``, ``value`` (the out-of-fold macro F1),
            ``duration_s``, one ``params_<name>`` column per Optuna parameter (empty when the
            trial did not draw it), ``micro_f1`` and one ``f1/<tag>`` column per tag.
        importances: The seeded fANOVA importance, for the macro F1, of each Optuna parameter
            drawn in every complete trial (the conditional ``c_logistic_regression``,
            ``c_linear_svm`` and ``cue_weight`` are absent); empty when fewer than two
            complete trials differ in value (index ``parameter``; name ``importance``).
        out_of_fold: The out-of-fold scores of the chosen configuration, with cross-fitted
            thresholds: column ``f1``, one row per metric (index ``metric``: ``macro_f1``,
            ``micro_f1``, then each tag).
    """

    params: ModelParams
    thresholds: pd.Series
    trials: pd.DataFrame
    importances: pd.Series
    out_of_fold: pd.DataFrame


def save_tuning_results(results: TuningResults, directory: Path = TFIDF_BASELINE_DIR) -> None:
    """Write the results to the files of ``RESULT_FILES``, replacing any earlier ones.

    Args:
        results: The results, as described in ``TuningResults``.
        directory: Folder of the files; created if missing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / PARAMS_FILE, results.params.to_dict())
    _write_json(directory / THRESHOLDS_FILE, results.thresholds.to_dict())
    results.trials.to_csv(directory / TRIALS_FILE, index=False)
    results.importances.to_csv(
        directory / IMPORTANCES_FILE, index_label="parameter", header=["importance"]
    )
    results.out_of_fold.to_csv(directory / OUT_OF_FOLD_FILE, index_label="metric")


def load_tuning_results(directory: Path = TFIDF_BASELINE_DIR) -> TuningResults:
    """Read the results written by ``save_tuning_results``.

    Args:
        directory: Folder of the files.

    Returns:
        The results, with the index and column names described in ``TuningResults``.

    Raises:
        FileNotFoundError: If a file is missing; the message names it and the command that
            writes it.
    """
    missing = [name for name in RESULT_FILES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{directory} has no {', '.join(missing)}: run "
            "python -m codeforces_tags.tuning export, which writes them."
        )
    thresholds = _read_json(directory / THRESHOLDS_FILE)
    return TuningResults(
        params=ModelParams.from_dict(_read_json(directory / PARAMS_FILE)),
        thresholds=pd.Series(thresholds, name="threshold", dtype=float).rename_axis("label"),
        trials=_read_csv(directory / TRIALS_FILE),
        importances=_read_csv(directory / IMPORTANCES_FILE, index_col="parameter")["importance"],
        out_of_fold=_read_csv(directory / OUT_OF_FOLD_FILE, index_col="metric"),
    )


def _write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, index_col: str | None = None) -> pd.DataFrame:
    """Read a CSV file, with the floats exactly as written."""
    return pd.read_csv(path, index_col=index_col, float_precision="round_trip")
