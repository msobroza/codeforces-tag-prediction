"""Tests for the grid studies of the tuned frozen-encoder methods and their export."""

import json
import math

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial
from threadpoolctl import threadpool_info, threadpool_limits

from codeforces_tags.config import FOCUS_TAGS, TUNING_BLAS_THREADS
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.f2llm import tuning
from codeforces_tags.methods.f2llm.encoder import write_json
from codeforces_tags.methods.f2llm.estimators import make_probe
from codeforces_tags.methods.f2llm.settings import GRIDS, PROBE, experiments_dir
from codeforces_tags.methods.f2llm.tuning import (
    INDEX_PARAM,
    best_index,
    cross_fitted_macro_f1,
    export,
    journal_path,
    load_params,
    missing_indices,
    open_study,
    run_study,
    tuning_dir,
)
from codeforces_tags.model.thresholds import CrossFittedTagThresholds

# Three probe configurations that differ, and one configuration four times: an all-tie grid.
TINY_GRID = (
    {"head": "ridge", "c": 1.0, "class_weight": "none", "solution_weight": 0.0},
    {"head": "ridge", "c": 0.01, "class_weight": "balanced", "solution_weight": 1.0},
    {"head": "linear_svm", "c": 1.0, "class_weight": "none", "solution_weight": 0.5},
)
TIE_GRID = (TINY_GRID[0],) * 4


@pytest.fixture(name="tiny_grid")
def tiny_grid_fixture(monkeypatch):
    """Make ``TINY_GRID`` the probe's grid for the test."""
    monkeypatch.setitem(GRIDS, PROBE, TINY_GRID)
    return TINY_GRID


@pytest.fixture(name="journal")
def journal_fixture(tmp_path):
    """Return the path of an Optuna journal, in a folder that does not exist yet."""
    return tmp_path / "experiments" / PROBE / "optuna.jsonl"


@pytest.fixture(name="run")
def run_fixture(embedding_frame, synthetic_labels, embedding_width, journal):
    """Return a runner of the probe's study on the synthetic dev set."""

    def run():
        return run_study(
            PROBE,
            embedding_frame,
            synthetic_labels,
            embedding_frame["fold"],
            width=embedding_width,
            probe_params=None,
            tfidf=None,
            journal=journal,
        )

    return run


def _blas_threads():
    """Return the thread counts of the BLAS libraries loaded in this process."""
    return {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}


def _hand_built_study(values):
    """Return an in-memory study with one COMPLETE trial per ``(index, value)``, in that order."""
    study = optuna.create_study(direction="maximize")
    distribution = CategoricalDistribution(list(range(len(TINY_GRID) + 1)))
    for index, value in values:
        study.add_trial(
            create_trial(
                params={INDEX_PARAM: index},
                distributions={INDEX_PARAM: distribution},
                value=value,
            )
        )
    return study


def _complete_indices(study):
    """Return the grid index of each COMPLETE trial, by trial number."""
    trials = study.get_trials(states=(TrialState.COMPLETE,))
    return [trial.params[INDEX_PARAM] for trial in trials]


def test_cross_fitted_macro_f1_scores_the_cross_fitted_predictions(
    embedding_frame, synthetic_labels, embedding_width
):
    probe = make_probe(TINY_GRID[0], embedding_width)
    folds = embedding_frame["fold"]

    value, table = cross_fitted_macro_f1(probe, embedding_frame, synthetic_labels, folds)

    fitted = CrossFittedTagThresholds(probe, refit=False).fit(
        embedding_frame, synthetic_labels, groups=folds
    )
    predictions = pd.DataFrame(
        fitted.oof_predictions_, index=embedding_frame.index, columns=synthetic_labels.columns
    )
    assert value == evaluate(synthetic_labels, predictions).macro_f1
    assert table.index.name == "metric"
    assert list(table.index) == ["macro_f1", "micro_f1", *FOCUS_TAGS]
    assert table.at["macro_f1", "f1"] == value


def test_cross_fitted_macro_f1_fits_with_blas_on_the_tuning_threads(
    embedding_frame, synthetic_labels, embedding_width, monkeypatch
):
    observed = []

    class Recording(CrossFittedTagThresholds):
        """Records the BLAS threads it is fitted with."""

        def fit(self, X, y, *, groups):
            observed.append(_blas_threads())
            return super().fit(X, y, groups=groups)

    monkeypatch.setattr(tuning, "CrossFittedTagThresholds", Recording)
    probe = make_probe(TINY_GRID[0], embedding_width)

    with threadpool_limits(limits=TUNING_BLAS_THREADS + 1, user_api="blas"):
        cross_fitted_macro_f1(probe, embedding_frame, synthetic_labels, embedding_frame["fold"])
        after = _blas_threads()

    assert observed == [{TUNING_BLAS_THREADS}]
    assert after == {TUNING_BLAS_THREADS + 1}


def test_run_study_scores_every_configuration_of_the_grid_once(
    tiny_grid, run, embedding_frame, synthetic_labels, embedding_width
):
    study = run()

    assert sorted(_complete_indices(study)) == [0, 1, 2]
    for trial in study.trials:
        probe = make_probe(tiny_grid[trial.params[INDEX_PARAM]], embedding_width)
        expected, table = cross_fitted_macro_f1(
            probe, embedding_frame, synthetic_labels, embedding_frame["fold"]
        )
        assert trial.value == expected
        assert trial.user_attrs["micro_f1"] == table.at["micro_f1", "f1"]
        assert list(trial.user_attrs["f1"]) == list(FOCUS_TAGS)


def test_run_study_on_a_complete_study_runs_nothing(tiny_grid, run):
    first = run()
    again = run()

    assert len(again.trials) == len(first.trials) == len(tiny_grid)


def test_run_study_runs_a_crashed_trial_s_configuration_again(tiny_grid, run, journal):
    crashed = open_study(PROBE, journal).ask()
    index = crashed.suggest_categorical(INDEX_PARAM, list(range(len(tiny_grid))))

    study = run()

    states = {trial.number: trial.state for trial in study.trials}
    assert states[crashed.number] == TrialState.RUNNING  # a crash never tells the journal
    assert missing_indices(study, len(tiny_grid)) == []
    assert _complete_indices(study).count(index) == 1


def test_run_study_runs_a_failed_trial_s_configuration_again(tiny_grid, run, journal):
    study = open_study(PROBE, journal)
    failed = study.ask()
    index = failed.suggest_categorical(INDEX_PARAM, list(range(len(tiny_grid))))
    study.tell(failed, state=TrialState.FAIL)

    study = run()

    assert missing_indices(study, len(tiny_grid)) == []
    reruns = [
        trial.number
        for trial in study.get_trials(states=(TrialState.COMPLETE,))
        if trial.params[INDEX_PARAM] == index
    ]
    assert len(reruns) == 1
    assert reruns[0] > failed.number


def test_run_study_on_an_all_tie_grid_chooses_the_first_configuration(monkeypatch, run, journal):
    monkeypatch.setitem(GRIDS, PROBE, TIE_GRID)

    study = run()

    assert len({trial.value for trial in study.trials}) == 1
    assert sorted(_complete_indices(study)) == [0, 1, 2, 3]
    # Optuna's own best trial is the first one run, whichever cell the GridSampler drew.
    assert study.best_trial.number == 0
    assert best_index(study) == 0
    assert journal.is_file()


def test_best_index_breaks_ties_by_the_lowest_index_not_the_first_trial():
    study = _hand_built_study([(3, 0.5), (1, 0.5), (2, 0.5), (0, 0.5)])

    assert study.best_trial.params[INDEX_PARAM] == 3
    assert best_index(study) == 0


def test_best_index_takes_the_highest_value_of_the_complete_trials():
    study = _hand_built_study([(0, 0.5), (2, 0.7), (1, 0.6)])
    study.add_trial(
        create_trial(
            params={INDEX_PARAM: 3},
            distributions={INDEX_PARAM: CategoricalDistribution([0, 1, 2, 3])},
            state=TrialState.FAIL,
        )
    )

    assert best_index(study) == 2


def test_best_index_needs_a_complete_trial():
    with pytest.raises(ValueError, match="no complete trial"):
        best_index(_hand_built_study([]))


def test_the_journal_is_in_the_method_s_experiments_folder():
    assert journal_path(PROBE) == experiments_dir(PROBE) / "optuna.jsonl"


@pytest.fixture(name="export_to")
def export_to_fixture(embedding_frame, synthetic_labels, embedding_width):
    """Return an exporter of a probe study into a folder."""

    def export_to(study, directory):
        return export(
            PROBE,
            study,
            embedding_frame,
            synthetic_labels,
            embedding_frame["fold"],
            width=embedding_width,
            probe_params=None,
            tfidf=None,
            directory=directory,
        )

    return export_to


def test_export_writes_the_chosen_configuration_its_trials_and_its_scores(
    tiny_grid, run, export_to, tmp_path
):
    study = run()
    chosen = best_index(study)

    directory = export_to(study, tmp_path / "tuning")

    assert directory == tmp_path / "tuning"
    params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
    assert params == tiny_grid[chosen]
    assert load_params(PROBE, directory) == tiny_grid[chosen]
    trials = pd.read_csv(directory / "trials.csv")
    assert list(trials[INDEX_PARAM]) == [0, 1, 2]
    assert list(trials.columns[:8]) == [
        "number",
        INDEX_PARAM,
        "value",
        "duration_s",
        "head",
        "c",
        "class_weight",
        "solution_weight",
    ]
    assert list(trials.columns[8:]) == ["micro_f1", *(f"f1/{tag}" for tag in FOCUS_TAGS)]
    out_of_fold = pd.read_csv(
        directory / "out_of_fold.csv", index_col="metric", float_precision="round_trip"
    )
    np.testing.assert_allclose(out_of_fold.at["macro_f1", "f1"], study.best_value, rtol=0)


def test_export_lists_the_complete_trials_only(tiny_grid, run, journal, export_to, tmp_path):
    crashed = open_study(PROBE, journal).ask()
    crashed.suggest_categorical(INDEX_PARAM, list(range(len(tiny_grid))))

    directory = export_to(run(), tmp_path / "tuning")

    trials = pd.read_csv(directory / "trials.csv")
    assert crashed.number not in set(trials["number"])
    assert len(trials) == len(tiny_grid)


def test_export_refuses_a_study_with_a_configuration_without_a_complete_trial(
    tiny_grid, export_to, tmp_path
):
    study = _hand_built_study([(0, 0.5)])

    with pytest.raises(ValueError, match=f"2 of the {len(tiny_grid)} configurations"):
        export_to(study, tmp_path / "tuning")
    assert not (tmp_path / "tuning").exists()


def test_export_refuses_a_configuration_that_scores_another_value_than_recorded(
    tiny_grid, export_to, tmp_path
):
    study = _hand_built_study([(index, 0.123) for index in range(len(tiny_grid))])

    with pytest.raises(ValueError, match="Nothing was written"):
        export_to(study, tmp_path / "tuning")
    assert not (tmp_path / "tuning").exists()


def test_export_refuses_a_configuration_that_scores_nan_when_scored_again(
    tiny_grid, export_to, tmp_path, monkeypatch
):
    study = _hand_built_study([(index, 0.5) for index in range(len(tiny_grid))])
    monkeypatch.setattr(tuning, "cross_fitted_macro_f1", lambda *rows: (math.nan, pd.DataFrame()))

    with pytest.raises(ValueError, match="Nothing was written"):
        export_to(study, tmp_path / "tuning")
    assert not (tmp_path / "tuning").exists()


def test_the_tuning_folder_is_in_the_method_s_experiments_folder():
    assert tuning_dir(PROBE) == experiments_dir(PROBE) / "tuning"


def test_load_params_reads_the_tuning_folder_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "tuning_dir", lambda name: tmp_path / name)
    (tmp_path / PROBE).mkdir()
    write_json(tmp_path / PROBE / "params.json", GRIDS[PROBE][5])

    assert load_params(PROBE) == GRIDS[PROBE][5]


def test_load_params_names_the_commands_when_there_is_no_params_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="tune --method f2llm_probe"):
        load_params(PROBE, tmp_path)


def test_load_params_refuses_a_configuration_outside_the_grid(tmp_path):
    write_json(tmp_path / "params.json", {"head": "ridge", "c": 3.0})

    with pytest.raises(ValueError, match="grid changed"):
        load_params(PROBE, tmp_path)
