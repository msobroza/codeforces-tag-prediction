"""Tests for logging the tuning study to MLflow."""

import math
import os

import mlflow
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState, create_trial

from codeforces_tags.config import MLFLOW_ENV, MLFLOW_EXPERIMENT, TUNING_STUDY
from codeforces_tags.tuning.tracking import log_study


def _trial(value, c, f1):
    """Return a complete trial of a linear SVM, with the user attributes ``objective`` sets."""
    return create_trial(
        value=value,
        params={"classifier": "linear_svm", "c_linear_svm": c},
        distributions={
            "classifier": CategoricalDistribution(("logistic_regression", "linear_svm")),
            "c_linear_svm": FloatDistribution(0.001, 3.0, log=True),
        },
        user_attrs={
            "micro_f1": f1["micro"],
            "f1": {"graphs": f1["graphs"], "strings": f1["strings"]},
            "thresholds": {"graphs": -0.25, "strings": math.inf},
        },
    )


@pytest.fixture(name="study")
def study_fixture():
    """Return a study of three trials: complete, failed, then complete and best."""
    study = optuna.create_study(study_name=TUNING_STUDY, direction="maximize")
    study.add_trial(_trial(0.5, 0.1, {"micro": 0.625, "graphs": 0.25, "strings": 0.75}))
    study.add_trial(create_trial(state=TrialState.FAIL))
    study.add_trial(_trial(0.75, 0.2, {"micro": 0.875, "graphs": 0.5, "strings": 1.0}))
    return study


def _nested_runs(parent_id):
    """Return the runs nested in a parent run, by trial number."""
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
    )
    return {int(run.data.tags["trial"]): run for run in runs}


def test_log_study_logs_a_nested_run_per_complete_trial(study, mlflow_store):
    parent_id = log_study(study, **mlflow_store)

    nested = _nested_runs(parent_id)
    assert sorted(nested) == [0, 2]  # trial 1 failed
    best = nested[2]
    assert best.data.params == {"classifier": "linear_svm", "c_linear_svm": "0.2"}
    assert best.data.metrics == {
        "macro_f1": 0.75,
        "micro_f1": 0.875,
        "f1/graphs": 0.5,
        "f1/strings": 1.0,
    }
    thresholds = mlflow.artifacts.load_dict(f"{best.info.artifact_uri}/thresholds.json")
    assert thresholds == {"graphs": -0.25, "strings": math.inf}


def test_log_study_keeps_the_best_trial_on_the_parent_run(study, mlflow_store):
    parent_id = log_study(study, **mlflow_store)

    parent = mlflow.get_run(parent_id)
    assert parent.data.tags["study"] == TUNING_STUDY
    assert parent.data.tags["best_trial"] == "2"
    assert parent.data.metrics == {"macro_f1": 0.75}
    best_params = mlflow.artifacts.load_dict(f"{parent.info.artifact_uri}/best_params.json")
    assert best_params == {"classifier": "linear_svm", "c_linear_svm": 0.2}


def test_log_study_logs_each_trial_once_however_often_it_runs(study, mlflow_store):
    parent_id = log_study(study, **mlflow_store)
    first = _nested_runs(parent_id)

    assert log_study(study, **mlflow_store) == parent_id
    again = _nested_runs(parent_id)
    study.add_trial(_trial(0.875, 0.3, {"micro": 0.875, "graphs": 0.75, "strings": 1.0}))
    assert log_study(study, **mlflow_store) == parent_id
    later = _nested_runs(parent_id)

    assert {number: run.info.run_id for number, run in again.items()} == {
        number: run.info.run_id for number, run in first.items()
    }
    assert sorted(later) == [0, 2, 3]
    history = mlflow.MlflowClient().get_metric_history(parent_id, "macro_f1")
    assert [(point.step, point.value) for point in history] == [(2, 0.75), (3, 0.875)]


def test_log_study_creates_the_experiment_with_its_artifacts_in_the_given_folder(
    study, mlflow_store
):
    log_study(study, experiment="other experiment", **mlflow_store)

    assert mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None
    experiment = mlflow.get_experiment_by_name("other experiment")
    assert experiment.artifact_location == mlflow_store["artifact_dir"].as_uri()
    assert len(list(mlflow_store["artifact_dir"].rglob("thresholds.json"))) == 2


def test_log_study_sets_mlflows_environment_before_it_logs(study, mlflow_store, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")

    log_study(study, **mlflow_store)

    assert {name: os.environ.get(name) for name in MLFLOW_ENV} == MLFLOW_ENV


def test_log_study_refuses_a_store_that_is_not_sqlite(study, tmp_path):
    store = tmp_path / "mlruns"

    with pytest.raises(ValueError, match="is not a sqlite:/// store"):
        log_study(study, tracking_uri=store.as_uri(), artifact_dir=tmp_path / "artifacts")

    assert not store.exists()
    assert not (tmp_path / "artifacts").exists()
