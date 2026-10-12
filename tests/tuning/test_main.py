"""Tests for the tuning command, end to end on a tiny dataset written to a temporary folder."""

import json
import runpy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import mlflow
import pandas as pd
import pytest
from optuna.trial import TrialState, create_trial
from threadpoolctl import threadpool_info, threadpool_limits

from codeforces_tags.config import (
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    OPTUNA_JOURNAL,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
    TFIDF_BASELINE_DIR,
    TUNING_BLAS_THREADS,
    TUNING_TRIALS,
    TUNING_WORKERS,
)
from codeforces_tags.tuning import __main__ as cli
from codeforces_tags.tuning.results import RESULT_FILES, load_tuning_results
from codeforces_tags.tuning.search import best_params, create_study, worker_commands

# The three kinds of problem of the dataset, in turn: tags, statement, solution, sample input
# and sample output. As in conftest.py, each kind has statement words of its own and every
# solution prints, so that every configuration of the search space can fit the dev set. The
# eight focus tags are spread over the two focus kinds; games is the one only the notes tell,
# on every other strings problem, so that the trials that read the notes score better.
KINDS = (
    (
        ["graphs", "trees", "math", "number theory"],
        "Count the connected components of an undirected graph.",
        "parent = list(range(n))\nprint(len(set(parent)))",
        "4 2\n1 2\n3 4",
        "2",
    ),
    (
        ["strings", "geometry", "probabilities"],
        "Count the distinct letters of a lowercase word.",
        "word = input()\nprint(len(set(word)))",
        "banana",
        "3",
    ),
    (
        ["greedy"],
        "Sort the array, then print its largest element.",
        "values = sorted(map(int, input().split()))\nprint(values[-1])",
        "5 1 4",
        "5",
    ),
)
DEV, TEST = 36, 3  # problems of each part; dev has three folds of four problems of each kind
FOLDS = 3
# One value per Optuna parameter that a logistic regression without cues draws, sorted by name:
# every trial enqueued with it scores the same.
CONFIGURATION = {
    "c_logistic_regression": 1.0,
    "class_weight": "balanced",
    "classifier": "logistic_regression",
    "code_min_df": 2,
    "code_ngram_max": 1,
    "code_sublinear_tf": False,
    "code_weight": 0.5,
    "include_notes": True,
    "statement_max_df": 1.0,
    "statement_min_df": 1,
    "statement_ngram_max": 1,
    "statement_sublinear_tf": False,
    "use_cues": False,
}


@dataclass
class FakeWorker:
    """Stand in for the ``subprocess.Popen`` of a worker: ``run`` gives its exit code.

    The worker runs when the parent waits for it, in this process.
    """

    run: Callable[[], int]
    returncode: int | None = None
    terminated: bool = False

    def __enter__(self):
        """Enter the context, as ``Popen`` does: return the process itself."""
        return self

    def __exit__(self, *_exc_info):
        """Leave the context; unlike ``Popen``, wait for nothing, as nothing runs."""

    def wait(self):
        """Run the worker, and return its exit code."""
        self.returncode = self.run()
        return self.returncode

    def poll(self):
        """Return the exit code, or None while the worker has not been waited for."""
        return self.returncode

    def terminate(self):
        """Record that the parent stopped the worker."""
        self.terminated = True


def _record(make_raw_record, position):
    """Return the raw record of the problem at a position of the dataset, every raw field set."""
    tags, statement, source_code, sample_input, sample_output = KINDS[position % len(KINDS)]
    game = position % (2 * len(KINDS)) == 1  # every other strings problem, two per fold
    return make_raw_record(
        prob_desc_description=statement,
        prob_desc_notes="Alice and Bob play this game." if game else None,
        source_code=source_code,
        prob_desc_sample_inputs=json.dumps([sample_input]),
        prob_desc_sample_outputs=json.dumps([sample_output]),
        tags=[*tags, "games"] if game else tags,
    )


def _split_row(position):
    """Return the split row of the problem at a position: dev for the first DEV problems."""
    dev = position < DEV
    return {
        "sample_id": 100 + position,
        "part": "dev" if dev else "test",
        "fold": position // len(KINDS) % FOLDS if dev else -1,
        "group": 100 + position,
    }


@pytest.fixture(name="paths")
def paths_fixture(tmp_path, journal, write_samples, make_raw_record):
    """Write the dataset and its split to tmp_path; return their paths and the journal's.

    Returns:
        The ``data_dir``, ``split_file`` and ``journal`` keyword arguments of
        ``worker_commands``.
    """
    positions = range(DEV + TEST)
    data_dir = write_samples(
        {100 + position: _record(make_raw_record, position) for position in positions}
    )
    split_file = tmp_path / "split.csv"
    pd.DataFrame([_split_row(position) for position in positions]).to_csv(split_file, index=False)
    return {"data_dir": str(data_dir), "split_file": split_file, "journal": journal}


@pytest.fixture(name="options")
def options_fixture(paths):
    """Return the options that point the command to the dataset, the split and the journal."""
    data_dir, split_file, journal = paths["data_dir"], paths["split_file"], paths["journal"]
    return ["--data", data_dir, "--split", str(split_file), "--journal", str(journal)]


@pytest.fixture(name="mlflow_options")
def mlflow_options_fixture(mlflow_store):
    """Return the options that point the command to the MLflow store of conftest.py."""
    tracking_uri, artifact_dir = mlflow_store["tracking_uri"], mlflow_store["artifact_dir"]
    return ["--tracking-uri", tracking_uri, "--artifact-dir", str(artifact_dir)]


def _study_runs():
    """Return the runs of the default experiment: the study's runs, then the trials' by number."""
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], output_format="list")
    studies = [run for run in runs if "study" in run.data.tags]
    trials = sorted(int(run.data.tags["trial"]) for run in runs if "trial" in run.data.tags)
    return studies, trials


def _blas_threads():
    """Return the thread counts of the BLAS libraries loaded in this process."""
    return {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}


def test_parse_args_defaults_to_the_settings_of_the_config():
    run = cli.parse_args(["run"])

    assert (run.trials, run.workers, run.data) == (TUNING_TRIALS, TUNING_WORKERS, None)
    assert (run.split, run.journal) == (SPLIT_FILE, OPTUNA_JOURNAL)
    assert (run.tracking_uri, run.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)
    assert cli.parse_args(["export"]).out == TFIDF_BASELINE_DIR


def test_parse_args_requires_the_trials_and_seed_of_a_worker():
    with pytest.raises(SystemExit):
        cli.parse_args(["worker", "--trials", "3"])


def test_parse_args_points_freeze_and_verify_to_the_frozen_study_and_methods_not_the_journal():
    freeze = cli.parse_args(["freeze"])
    verify = cli.parse_args(["verify"])

    assert (freeze.results, freeze.assets, freeze.allow_dirty) == (
        TFIDF_BASELINE_DIR,
        METHODS_ASSETS_DIR,
        False,
    )
    assert (freeze.tracking_uri, freeze.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)
    assert (verify.results, verify.assets, verify.data, verify.split) == (
        TFIDF_BASELINE_DIR,
        METHODS_ASSETS_DIR,
        None,
        SPLIT_FILE,
    )
    assert not hasattr(freeze, "journal")
    assert not hasattr(verify, "journal")
    assert not hasattr(verify, "tracking_uri")


def test_main_runs_freeze_and_verify(monkeypatch):
    received = []
    for command in ("freeze", "verify"):
        monkeypatch.setattr(cli, command, lambda args: received.append(args.command) or 0)

    assert cli.main(["freeze"]) == 0
    assert cli.main(["verify"]) == 0

    assert received == ["freeze", "verify"]


def test_worker_adds_trials_scored_on_the_focus_tags_of_the_dev_set(options, journal):
    assert cli.main(["worker", "--trials", "2", "--seed", "1", *options]) == 0

    trials = create_study(journal).trials
    assert [trial.state for trial in trials] == [TrialState.COMPLETE] * 2
    assert list(trials[0].user_attrs["f1"]) == list(FOCUS_TAGS)


def test_worker_scores_the_trials_on_the_raw_dev_rows(options, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "run_worker", lambda _study, _trials, *dev: received.append(dev))

    assert cli.main(["worker", "--trials", "1", "--seed", "0", *options]) == 0

    assert len(received) == 1
    raw_problems, labels, folds = received[0]
    assert "statement" not in raw_problems  # read_samples() rows: the pipeline preprocesses them
    assert set(RAW_INPUT_COLUMNS) <= set(raw_problems.columns)
    assert raw_problems.index.tolist() == [100 + position for position in range(DEV)]
    assert labels.index.equals(raw_problems.index)
    assert list(labels.columns) == list(FOCUS_TAGS)
    pd.testing.assert_series_equal(folds, raw_problems["fold"])


def test_worker_limits_the_blas_threads_while_it_runs(options, monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "run_worker", lambda *_: observed.append(_blas_threads()))

    with threadpool_limits(limits=TUNING_BLAS_THREADS + 1, user_api="blas"):
        assert cli.main(["worker", "--trials", "1", "--seed", "0", *options]) == 0
        after = _blas_threads()

    assert observed == [{TUNING_BLAS_THREADS}]
    assert after == {TUNING_BLAS_THREADS + 1}  # the limit ends with the worker


def test_run_starts_the_workers_then_logs_each_complete_trial_to_mlflow(
    paths, options, mlflow_options, monkeypatch, capsys
):
    started = []

    def popen(command):
        """Stand in for subprocess.Popen: the worker runs in this process, when waited for."""
        started.append(command)
        return FakeWorker(run=lambda: cli.main(command[3:]))

    monkeypatch.setattr(cli, "Popen", popen)

    assert cli.main(["run", "--trials", "3", "--workers", "2", *options, *mlflow_options]) == 0

    assert started == worker_commands(3, 2, **paths)
    trials = create_study(paths["journal"]).trials
    assert [trial.state for trial in trials] == [TrialState.COMPLETE] * 3
    studies, logged = _study_runs()
    assert len(studies) == 1
    assert logged == [0, 1, 2]
    assert "best macro F1" in capsys.readouterr().out


def test_run_exits_with_1_and_logs_nothing_when_a_worker_fails(
    options, mlflow_options, monkeypatch, capsys
):
    exit_codes = iter([0, 1])
    monkeypatch.setattr(cli, "Popen", lambda command: FakeWorker(run=partial(next, exit_codes)))

    assert cli.main(["run", "--workers", "2", *options, *mlflow_options]) == 1

    assert "1 of 2 workers failed" in capsys.readouterr().err
    assert mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None


def test_run_exits_with_1_and_logs_nothing_when_the_workers_complete_no_trial(
    options, mlflow_options, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "Popen", lambda command: FakeWorker(run=lambda: 0))

    assert cli.main(["run", "--workers", "2", *options, *mlflow_options]) == 1

    assert "has no complete trial" in capsys.readouterr().err
    assert mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None


def test_run_starts_no_worker_for_a_target_of_no_trial(
    options, mlflow_options, monkeypatch, capsys
):
    started = []
    monkeypatch.setattr(cli, "Popen", started.append)

    assert cli.main(["run", "--trials", "0", *options, *mlflow_options]) == 1

    assert "--trials must be at least 1, not 0" in capsys.readouterr().err
    assert not started


def test_run_starts_no_worker_for_fewer_than_one_worker(
    options, mlflow_options, monkeypatch, capsys
):
    started = []
    monkeypatch.setattr(cli, "Popen", started.append)

    assert cli.main(["run", "--workers", "0", *options, *mlflow_options]) == 1

    assert "--workers must be at least 1, not 0" in capsys.readouterr().err
    assert not started


def test_run_refuses_a_store_that_is_not_sqlite_before_it_starts_a_worker(
    options, journal, monkeypatch, capsys
):
    started = []
    monkeypatch.setattr(cli, "Popen", started.append)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", *options, "--tracking-uri", "http://127.0.0.1:5000"])

    assert exit_info.value.code == 2
    assert "http://127.0.0.1:5000 is not a sqlite:/// store" in capsys.readouterr().err
    assert not started
    assert not journal.parent.exists()  # no study created


def test_run_stops_the_workers_it_started_when_starting_another_fails(
    options, mlflow_options, monkeypatch
):
    started = []

    def popen(_command):
        """Stand in for subprocess.Popen: start the first worker, then run out of processes."""
        if started:
            raise OSError("Resource temporarily unavailable")
        started.append(FakeWorker(run=lambda: 0))
        return started[-1]

    monkeypatch.setattr(cli, "Popen", popen)

    with pytest.raises(OSError, match="Resource temporarily unavailable"):
        cli.main(["run", "--workers", "2", *options, *mlflow_options])

    assert len(started) == 1
    assert started[0].terminated


def test_export_writes_the_results_of_the_best_trial_and_logs_the_study_but_no_model(
    options, mlflow_options, journal, tmp_path
):
    study = create_study(journal)
    for include_notes in (False, True):  # the only difference between the two trials
        study.enqueue_trial(CONFIGURATION | {"include_notes": include_notes})
    cli.main(["worker", "--trials", "3", "--seed", "0", *options])
    out = tmp_path / "tfidf_baseline"

    assert cli.main(["export", "--out", str(out), *options, *mlflow_options]) == 0

    assert study.trials[1].value > study.trials[0].value  # reading the notes tells games
    assert sorted(path.name for path in out.iterdir()) == sorted(RESULT_FILES)
    results = load_tuning_results(out)
    assert results.params == best_params(study)
    assert results.out_of_fold.at["macro_f1", "f1"] == study.best_value
    assert list(results.thresholds.index) == list(FOCUS_TAGS)
    assert list(results.trials.columns[:4]) == ["number", "state", "value", "duration_s"]
    assert list(results.trials.columns[-9:]) == ["micro_f1", *(f"f1/{tag}" for tag in FOCUS_TAGS)]
    parameters = {column.removeprefix("params_") for column in results.trials.columns[4:-9]}
    assert parameters == set().union(*(trial.params for trial in study.trials))
    assert set(results.importances.index) <= parameters
    studies, logged = _study_runs()
    assert logged == [0, 1, 2]
    # The tag model is logged as a tagging method, through log_method (docs/adr/0002), not here.
    artifacts = mlflow.artifacts.list_artifacts(run_id=studies[0].info.run_id)
    assert [artifact.path for artifact in artifacts] == ["best_params.json"]
    experiment_ids = [studies[0].info.experiment_id]
    assert mlflow.search_logged_models(experiment_ids, output_format="list") == []


def test_export_fails_when_the_journal_holds_no_complete_trial(
    options, mlflow_options, tmp_path, capsys
):
    out = tmp_path / "tfidf_baseline"

    assert cli.main(["export", "--out", str(out), *options, *mlflow_options]) == 1

    assert "no complete trial" in capsys.readouterr().err
    assert not out.exists()


def test_export_writes_nothing_when_the_best_configuration_scores_another_value(
    options, mlflow_options, journal, tmp_path, capsys
):
    study = create_study(journal)
    study.enqueue_trial(CONFIGURATION)
    cli.main(["worker", "--trials", "1", "--seed", "0", *options])
    scored = study.trials[0]
    # The same configuration, recorded with a better value: as if the study had run with other
    # code, data or split, its value is not what the configuration scores now.
    study.add_trial(
        create_trial(
            value=scored.value + 0.01,
            params=scored.params,
            distributions=scored.distributions,
            user_attrs=scored.user_attrs,
        )
    )
    out = tmp_path / "tfidf_baseline"

    assert cli.main(["export", "--out", str(out), *options, *mlflow_options]) == 1

    assert "differs from the study's best value" in capsys.readouterr().err
    assert not out.exists()
    assert mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None


def test_export_refuses_a_store_that_is_not_sqlite_before_it_writes_a_result_file(
    options, journal, tmp_path, capsys
):
    study = create_study(journal)
    study.enqueue_trial(CONFIGURATION)
    cli.main(["worker", "--trials", "1", "--seed", "0", *options])
    out, store = tmp_path / "tfidf_baseline", (tmp_path / "mlruns").as_uri()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["export", "--out", str(out), *options, "--tracking-uri", store])

    assert exit_info.value.code == 2
    assert f"{store} is not a sqlite:/// store" in capsys.readouterr().err
    assert not out.exists()


@pytest.fixture(name="equal_trials", params=[1, 2], ids=["one trial", "two equal trials"])
def equal_trials_fixture(request, options, journal):
    """Run the worker on CONFIGURATION, once or twice; return the study, all of equal value."""
    study = create_study(journal)
    for _ in range(request.param):
        study.enqueue_trial(CONFIGURATION)
    cli.main(["worker", "--trials", str(request.param), "--seed", "0", *options])
    return study


def test_export_writes_no_importances_unless_two_trials_differ_in_value(
    equal_trials, options, mlflow_options, tmp_path, capsys
):
    out = tmp_path / "tfidf_baseline"

    assert cli.main(["export", "--out", str(out), *options, *mlflow_options]) == 0

    assert len({trial.value for trial in equal_trials.trials}) == 1
    importances = load_tuning_results(out).importances
    assert importances.empty
    assert (importances.name, importances.index.name) == ("importance", "parameter")
    assert "no parameter importances" in capsys.readouterr().out


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.tuning", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.tuning.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.tuning", run_name="__main__")

    assert exit_info.value.code == 0
