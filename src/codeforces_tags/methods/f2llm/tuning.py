"""The grid studies of the three tuned frozen-encoder methods, and their export.

Each method's grid lives in ``settings.GRIDS``; its study is an Optuna study whose only
parameter, ``index``, is a configuration's position in that grid, visited by a ``GridSampler``
and kept in the journal ``experiments/<name>/optuna.jsonl``. A trial scores its configuration
with ``cross_fitted_macro_f1``: the out-of-fold macro F1 of the dev set, with cross-fitted
thresholds (docs/adr/0001-model-selection-protocol.md).

The ``GridSampler`` visits the grid in a shuffled order, and ``study.best_trial`` breaks ties by
trial number: neither chooses. ``best_index`` takes the highest value among the COMPLETE trials,
ties broken by the lowest grid index (the grids list the simplest configurations first).
``export`` writes the chosen configuration and its scores to ``experiments/<name>/tuning/``,
which ``freeze`` copies into ``assets/methods/<name>/tuning/`` once its refit has reproduced the
recorded out-of-fold macro F1 (``check_exported_score``).

This module may import optuna (a method's ``tuning.py`` is exempt from TID253); the prediction
path never imports it.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import GridSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from threadpoolctl import threadpool_limits

from codeforces_tags.config import TUNING_BLAS_THREADS, TUNING_EXPORT_TOLERANCE, TUNING_SEED
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.f2llm.encoder import read_json, write_json
from codeforces_tags.methods.f2llm.estimators import make_estimator
from codeforces_tags.methods.f2llm.settings import GRIDS, experiments_dir
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds

TUNING_FOLDER = "tuning"
PARAMS_FILE = "params.json"
TRIALS_FILE = "trials.csv"
OUT_OF_FOLD_FILE = "out_of_fold.csv"
JOURNAL_FILE = "optuna.jsonl"
# The study's only parameter: a configuration's position in the method's grid.
INDEX_PARAM = "index"
# The states of a trial that will not run again: GridSampler counts their cells as visited.
_FINISHED = (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)


def cross_fitted_macro_f1(
    estimator, X: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> tuple[float, pd.DataFrame]:
    """Score an estimator on the dev set, out of fold, with cross-fitted thresholds.

    The fits run with BLAS on ``TUNING_BLAS_THREADS`` threads (torch keeps its own threads).

    Args:
        estimator: An unfitted score estimator (``make_estimator``).
        X: The dev feature frame, indexed by ``sample_id``.
        labels: 0/1 focus labels, with the index of ``X``.
        folds: Dev fold per problem, with the index of ``X``.

    Returns:
        The out-of-fold macro F1, and the out-of-fold table: column ``f1``, one row per metric
        (index ``metric``: ``macro_f1``, ``micro_f1``, then each tag).
    """
    with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
        fitted = CrossFittedTagThresholds(estimator, refit=False).fit(X, labels, groups=folds)
    predictions = pd.DataFrame(fitted.oof_predictions_, index=X.index, columns=labels.columns)
    evaluation = evaluate(labels, predictions)
    return evaluation.macro_f1, evaluation.overall_f1().to_frame("f1")


def journal_path(name: str) -> Path:
    """Return the Optuna journal of a method's study.

    Args:
        name: A tuned method's name.

    Returns:
        ``experiments_dir(name) / JOURNAL_FILE``.
    """
    return experiments_dir(name) / JOURNAL_FILE


def open_study(name: str, journal: Path) -> optuna.Study:
    """Create a method's grid study in its journal, or load it when the journal holds it.

    Args:
        name: A tuned method's name, a key of ``GRIDS``; also the study's name.
        journal: The Optuna journal file; its folder is created if missing.

    Returns:
        The study, which maximises the out-of-fold macro F1, with a ``GridSampler`` over
        ``index`` in ``range(len(GRIDS[name]))``, seeded with ``TUNING_SEED``.
    """
    journal.parent.mkdir(parents=True, exist_ok=True)
    sampler = GridSampler({INDEX_PARAM: list(range(len(GRIDS[name])))}, seed=TUNING_SEED)
    return optuna.create_study(
        storage=JournalStorage(JournalFileBackend(str(journal))),
        sampler=sampler,
        study_name=name,
        direction="maximize",
        load_if_exists=True,
    )


def run_study(  # pylint: disable=too-many-arguments
    name: str,
    dev_frame: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
    *,
    width: int,
    probe_params: dict | None,
    tfidf: ModelParams | None,
    journal: Path,
) -> optuna.Study:
    """Run a method's grid study until every configuration has a COMPLETE trial.

    The ``GridSampler`` runs each configuration not yet run, then any whose trial a crash left
    RUNNING. It counts a FAIL trial's configuration as visited, so each configuration still
    without a COMPLETE trial is then run once more, enqueued. A study already complete runs
    nothing. Each trial keeps its micro F1 and per-tag F1 as user attributes (``micro_f1``,
    ``f1``).

    Args:
        name: ``PROBE``, ``LATE_FUSION`` or ``EARLY_FUSION``.
        dev_frame: The dev feature frame, with ``group`` (late fusion's inner folds read it).
        labels: 0/1 focus labels, with the index of ``dev_frame``.
        folds: Dev fold per problem, with the index of ``dev_frame``.
        width: The embedding width.
        probe_params: The probe's tuned configuration (late fusion), else None.
        tfidf: The frozen TF-IDF configuration (both fusions), else None.
        journal: The Optuna journal file, ``journal_path(name)``.

    Returns:
        The study.
    """
    grid = GRIDS[name]

    def objective(trial: optuna.Trial) -> float:
        index = trial.suggest_categorical(INDEX_PARAM, list(range(len(grid))))
        estimator = make_estimator(
            name, grid[index], width=width, probe_params=probe_params, tfidf=tfidf
        )
        value, table = cross_fitted_macro_f1(estimator, dev_frame, labels, folds)
        trial.set_user_attr("micro_f1", float(table.at["micro_f1", "f1"]))
        trial.set_user_attr("f1", {str(tag): float(table.at[tag, "f1"]) for tag in labels})
        return value

    study = open_study(name, journal)
    if _unfinished(study, len(grid)):
        study.optimize(objective)
    for index in missing_indices(study, len(grid)):
        study.enqueue_trial({INDEX_PARAM: index})
        study.optimize(objective, n_trials=1)
    return study


def missing_indices(study: optuna.Study, size: int) -> list[int]:
    """Return the grid indices that have no COMPLETE trial.

    Args:
        study: A grid study.
        size: The size of its grid.

    Returns:
        The sorted indices from 0 to ``size - 1`` without a COMPLETE trial.
    """
    return sorted(set(range(size)) - _indices(study, (TrialState.COMPLETE,)))


def best_index(study: optuna.Study) -> int:
    """Choose the configuration of a grid study: the best COMPLETE trial, ties to the lowest index.

    Args:
        study: A grid study with at least one COMPLETE trial.

    Returns:
        The lowest grid index among the COMPLETE trials of the highest value. RUNNING and
        FAIL trials are ignored, and so is the order in which the trials ran.

    Raises:
        ValueError: If the study has no COMPLETE trial.
    """
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete:
        raise ValueError(f"The study {study.study_name!r} has no complete trial.")
    best = max(trial.value for trial in complete)
    return min(trial.params[INDEX_PARAM] for trial in complete if trial.value == best)


def _indices(study: optuna.Study, states: tuple[TrialState, ...]) -> set[int]:
    """Return the grid indices of the study's trials in the given states."""
    trials = study.get_trials(deepcopy=False, states=states)
    return {trial.params[INDEX_PARAM] for trial in trials if INDEX_PARAM in trial.params}


def _unfinished(study: optuna.Study, size: int) -> list[int]:
    """Return the grid indices with no finished trial: never run, or left RUNNING by a crash."""
    return sorted(set(range(size)) - _indices(study, _FINISHED))


def export(  # pylint: disable=too-many-arguments
    name: str,
    study: optuna.Study,
    dev_frame: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
    *,
    width: int,
    probe_params: dict | None,
    tfidf: ModelParams | None,
    directory: Path,
) -> Path:
    """Save the chosen configuration of a complete grid study, after scoring it again.

    The configuration of ``best_index`` is cross-fitted again; nothing is written unless it
    scores the study's best value, to within ``TUNING_EXPORT_TOLERANCE`` (otherwise the code,
    data or split differ from the study's).

    Args:
        name: ``PROBE``, ``LATE_FUSION`` or ``EARLY_FUSION``.
        study: The method's grid study, every configuration with a COMPLETE trial.
        dev_frame: The dev feature frame.
        labels: 0/1 focus labels, with the index of ``dev_frame``.
        folds: Dev fold per problem, with the index of ``dev_frame``.
        width: The embedding width.
        probe_params: The probe's tuned configuration (late fusion), else None.
        tfidf: The frozen TF-IDF configuration (both fusions), else None.
        directory: The folder to write, ``tuning_dir(name)``; created if missing.

    Returns:
        ``directory``, holding ``params.json`` (the chosen configuration), ``trials.csv`` (the
        COMPLETE trials, by grid index) and ``out_of_fold.csv`` (its out-of-fold F1 table).

    Raises:
        ValueError: If a configuration has no COMPLETE trial, or if the chosen one scores
            another value than the study recorded.
    """
    grid = GRIDS[name]
    _check_complete(name, study, len(grid))
    index = best_index(study)
    estimator = make_estimator(
        name, grid[index], width=width, probe_params=probe_params, tfidf=tfidf
    )
    value, table = cross_fitted_macro_f1(estimator, dev_frame, labels, folds)
    _check_rescored(index, value, study.best_value)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / PARAMS_FILE, grid[index])
    _trials_table(study, grid).to_csv(directory / TRIALS_FILE, index=False)
    table.to_csv(directory / OUT_OF_FOLD_FILE, index_label="metric")
    return directory


def check_exported_score(
    name: str, labels: pd.DataFrame, predictions: pd.DataFrame, directory: Path
) -> str:
    """Fail unless out-of-fold predictions score the macro F1 that a method's export recorded.

    ``freeze`` refits the exported configuration, then calls it before it scores the test set or
    writes a file: as ``export`` checks its rescore against the study, the refit must score the
    recorded value to within ``TUNING_EXPORT_TOLERANCE`` (NaN never does). Otherwise the code,
    the labels or the configuration changed since the export: late fusion's probe, for one, is
    the frozen probe's at freeze, but the probe of ``experiments/`` at tune and export.

    Args:
        name: A tuned method's name.
        labels: 0/1 focus labels of the dev problems.
        predictions: The refit's cross-fitted out-of-fold 0/1 predictions, with the index and
            columns of ``labels``.
        directory: The method's tuning folder (``tuning_dir(name)``), with ``out_of_fold.csv``.

    Returns:
        A line to print: the macro F1, and the file that records it.

    Raises:
        FileNotFoundError: If the folder has no ``out_of_fold.csv``.
        ValueError: If the macro F1 is not the recorded one.
    """
    path = directory / OUT_OF_FOLD_FILE
    table = pd.read_csv(path, index_col="metric", float_precision="round_trip")
    recorded = float(table.at["macro_f1", "f1"])
    value = evaluate(labels, predictions).macro_f1
    if not abs(value - recorded) <= TUNING_EXPORT_TOLERANCE:  # not >: NaN is never within it
        raise ValueError(
            f"{name}'s refit scores dev out of fold at macro F1 {value!r}, not the {recorded!r} "
            f"that its export recorded in {path}: the code, the labels or the configuration "
            "changed since (late fusion fuses the frozen probe, which may not be the probe of "
            "experiments/ it was tuned with). Nothing was written."
        )
    return f"{name}: the refit scores dev out of fold at macro F1 {value:.4f}, as {path} records"


def tuning_dir(name: str) -> Path:
    """Return the folder that ``export`` writes a method's tuning results to.

    Args:
        name: A tuned method's name.

    Returns:
        ``experiments_dir(name) / TUNING_FOLDER``.
    """
    return experiments_dir(name) / TUNING_FOLDER


def load_params(name: str, directory: Path | None = None) -> dict:
    """Read a tuned method's chosen configuration.

    Args:
        name: A tuned method's name.
        directory: The tuning folder; defaults to ``tuning_dir(name)`` (a frozen method's is
            ``assets/methods/<name>/tuning/``).

    Returns:
        The configuration, one of ``GRIDS[name]``.

    Raises:
        FileNotFoundError: If the folder has no ``params.json``; the message names the
            commands that write it.
        ValueError: If the configuration is not in the method's grid (the grid changed).
    """
    path = (tuning_dir(name) if directory is None else directory) / PARAMS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: run python -m codeforces_tags.methods.f2llm tune --method "
            f"{name}, then export --method {name}."
        )
    params = read_json(path)
    if params not in GRIDS[name]:
        raise ValueError(
            f"{path} holds {params}, which is not a configuration of the {name} grid in "
            "settings.py: the grid changed since the export."
        )
    return params


def _check_complete(name: str, study: optuna.Study, size: int) -> None:
    """Fail unless every configuration of the grid has a COMPLETE trial."""
    missing = missing_indices(study, size)
    if missing:
        raise ValueError(
            f"{len(missing)} of the {size} configurations of {name} have no complete trial "
            f"(indices {missing}): run the tune command again, which runs them."
        )


def _check_rescored(index: int, value: float, recorded: float) -> None:
    """Fail unless the chosen configuration, scored again, gives the study's best value."""
    if not abs(value - recorded) <= TUNING_EXPORT_TOLERANCE:  # not >: NaN is never within it
        raise ValueError(
            f"The chosen configuration (index {index}) scores {value:.6f} out of fold, not the "
            f"study's {recorded:.6f}: the code, data or split differ from the study's. Nothing "
            "was written."
        )


def _trials_table(study: optuna.Study, grid: tuple[dict, ...]) -> pd.DataFrame:
    """Return one row per COMPLETE trial, by grid index: its scores and its configuration."""
    rows = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        index = trial.params[INDEX_PARAM]
        per_tag = {f"f1/{tag}": f1 for tag, f1 in trial.user_attrs.get("f1", {}).items()}
        rows.append(
            {
                "number": trial.number,
                INDEX_PARAM: index,
                "value": trial.value,
                "duration_s": trial.duration.total_seconds(),
                **grid[index],
                "micro_f1": trial.user_attrs.get("micro_f1"),
                **per_tag,
            }
        )
    return pd.DataFrame(rows).sort_values([INDEX_PARAM, "number"], ignore_index=True)
