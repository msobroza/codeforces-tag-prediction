"""Log the tuning study to MLflow.

The workers write their trials to the Optuna journal only. ``log_study`` then replays the
study into MLflow from one process, so that MLflow has a single writer: one parent run per
study, and one nested run per complete trial with its parameters, scores and per-tag
thresholds. A trial already logged is skipped, so logging a study again adds only its new
trials.

``log_study`` points MLflow at its store with ``use_store`` (``tracking/environment.py``),
which sets MLflow's environment first: MLflow never loads a pickle (docs/adr/0002, safeguard
7). No model is logged here: a tag model is a tagging method, logged only through
``log_method``. ``mlflow.autolog`` is never called: it would log a run and a model for each of
the thousands of fits of a study.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
from optuna.trial import FrozenTrial, TrialState

from codeforces_tags.config import MLFLOW_ARTIFACT_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tuning.search import trial_scores


def log_study(
    study: optuna.Study,
    *,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment: str = MLFLOW_EXPERIMENT,
    artifact_dir: Path = MLFLOW_ARTIFACT_DIR,
) -> str:
    """Log each complete trial of a study to MLflow once, as a run nested in the study's run.

    The study's run is the run of the experiment tagged ``study`` with the study's name,
    created if missing. Each complete trial not logged yet gets a nested run tagged ``trial``
    with its number: its Optuna parameters, the metrics ``macro_f1``, ``micro_f1`` and
    ``f1/<tag>``, and ``thresholds.json``. When trials were added, the study's run records the
    best one: metric ``macro_f1`` at the step of the number of complete trials, tag
    ``best_trial`` and ``best_params.json`` (MLflow parameters cannot change once logged, and
    the best trial can).

    Args:
        study: The study, read from its journal.
        tracking_uri: The MLflow store, a ``sqlite:///`` URI; MLflow stays pointed at it
            afterwards.
        experiment: The MLflow experiment, created if missing.
        artifact_dir: Folder of the experiment's artifacts, used when the experiment is
            created; created if missing.

    Returns:
        The id of the study's run.

    Raises:
        ValueError: As ``use_store`` does: if the store is not a sqlite file, or if the
            experiment was deleted.
    """
    experiment_id = use_store(tracking_uri, experiment, artifact_dir)
    parent_id = _study_run_id(experiment_id, study.study_name)
    logged = _logged_trials(experiment_id, parent_id)
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    new = [trial for trial in complete if trial.number not in logged]
    for trial in new:
        _log_trial(trial, experiment_id, parent_id)
    if new:
        _log_best_trial(study, parent_id, len(complete))
    return parent_id


def _study_run_id(experiment_id: str, study_name: str) -> str:
    """Return the id of the run tagged with the study's name, created if missing."""
    runs = mlflow.search_runs(
        [experiment_id], filter_string=f"tags.study = '{study_name}'", output_format="list"
    )
    if runs:
        return runs[0].info.run_id
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=study_name, tags={"study": study_name}
    ) as run:
        return run.info.run_id


def _logged_trials(experiment_id: str, parent_id: str) -> set[int]:
    """Return the numbers of the trials that already have a run in the study's run."""
    runs = mlflow.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
    )
    return {int(run.data.tags["trial"]) for run in runs}


def _log_trial(trial: FrozenTrial, experiment_id: str, parent_id: str) -> None:
    """Log a complete trial as a run nested in the study's run."""
    scores = {"macro_f1": trial.value} | trial_scores(trial)
    with mlflow.start_run(
        experiment_id=experiment_id,
        parent_run_id=parent_id,
        run_name=f"trial {trial.number}",
        tags={"trial": str(trial.number)},
    ):
        mlflow.log_params(trial.params)
        mlflow.log_metrics(scores)
        mlflow.log_dict(trial.user_attrs["thresholds"], "thresholds.json")


def _log_best_trial(study: optuna.Study, parent_id: str, complete_count: int) -> None:
    """Record the study's best trial in its run, at the step of its complete trial count."""
    best = study.best_trial
    with mlflow.start_run(run_id=parent_id):
        mlflow.log_metric("macro_f1", best.value, step=complete_count)
        mlflow.set_tag("best_trial", str(best.number))
        mlflow.log_dict(best.params, "best_params.json")
