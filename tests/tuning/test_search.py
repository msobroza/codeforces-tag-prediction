"""Tests for the search space, the Optuna study and its workers."""

import sys
import warnings
from pathlib import Path

import pytest
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import FixedTrial, TrialState, create_trial

from codeforces_tags.config import TUNING_SEED, TUNING_SPACE, TUNING_STUDY
from codeforces_tags.evaluation.report import ALL_PROBLEMS, evaluate
from codeforces_tags.model.params import BLOCKS, CLASS_WEIGHTS, CLASSIFIERS, ModelParams
from codeforces_tags.model.training import build_pipeline, cross_fit
from codeforces_tags.tuning.search import (
    best_params,
    create_study,
    objective,
    run_worker,
    suggest_params,
    trial_scores,
    worker_commands,
)

# One value per Optuna parameter, both classifiers' C included: a logistic regression on the
# three blocks. The values are small enough for the tiny dev set of conftest.py.
VALUES = {
    "classifier": "logistic_regression",
    "c_logistic_regression": 2.5,
    "c_linear_svm": 0.02,
    "class_weight": "balanced",
    "include_notes": True,
    "statement_ngram_max": 2,
    "statement_min_df": 3,
    "statement_max_df": 0.8,
    "statement_sublinear_tf": True,
    "code_weight": 0.7,
    "code_ngram_max": 3,
    "code_min_df": 4,
    "code_sublinear_tf": False,
    "use_cues": True,
    "cue_weight": 0.3,
}
# The parameters searched besides the ranges of TUNING_SPACE: the choices.
CHOICES = {
    "classifier": CLASSIFIERS,
    "class_weight": CLASS_WEIGHTS,
    "include_notes": (False, True),
    "statement_sublinear_tf": (False, True),
    "code_sublinear_tf": (False, True),
    "use_cues": (False, True),
}


def _model_params(**overrides):
    """Return the ModelParams of VALUES, with fields overridable by keyword."""
    values = {
        "classifier": "logistic_regression",
        "c": 2.5,
        "class_weight": "balanced",
        "include_notes": True,
        "statement_ngram_max": 2,
        "statement_min_df": 3,
        "statement_max_df": 0.8,
        "statement_sublinear_tf": True,
        "code_weight": 0.7,
        "code_ngram_max": 3,
        "code_min_df": 4,
        "code_sublinear_tf": False,
        "cue_weight": 0.3,
        "blocks": BLOCKS,
    }
    return ModelParams(**(values | overrides))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, {}),
        ({"classifier": "linear_svm"}, {"classifier": "linear_svm", "c": 0.02}),
        ({"use_cues": False}, {"cue_weight": 0.0, "blocks": ("statement", "code")}),
    ],
    ids=["logistic regression with cues", "linear SVM", "without cues"],
)
def test_suggest_params_turns_the_values_of_a_trial_into_model_params(overrides, expected):
    assert suggest_params(FixedTrial(VALUES | overrides)) == _model_params(**expected)


def test_suggest_params_searches_the_ranges_of_the_tuning_space_and_the_choices():
    distributions = {}
    for overrides in ({}, {"classifier": "linear_svm"}):
        trial = FixedTrial(VALUES | overrides)
        suggest_params(trial)
        distributions |= trial.distributions

    ranges = {name: (distributions[name].low, distributions[name].high) for name in TUNING_SPACE}
    assert ranges == TUNING_SPACE
    assert distributions["c_logistic_regression"].log and distributions["c_linear_svm"].log
    choices = {name: distributions[name].choices for name in CHOICES}
    assert choices == CHOICES
    assert set(distributions) == set(TUNING_SPACE) | set(CHOICES)


def test_suggest_params_draws_only_the_c_of_the_classifier_and_the_cue_weight_with_cues():
    trial = FixedTrial(VALUES | {"classifier": "linear_svm", "use_cues": False})

    suggest_params(trial)

    assert "c_linear_svm" in trial.distributions
    assert "c_logistic_regression" not in trial.distributions
    assert "cue_weight" not in trial.distributions


def test_create_study_loads_the_study_already_in_the_journal(journal):
    first = create_study(journal)
    first.tell(first.ask(), 0.5)

    again = create_study(journal)

    assert again.study_name == TUNING_STUDY
    assert again.direction == StudyDirection.MAXIMIZE
    assert [trial.value for trial in again.trials] == [0.5]


def test_create_study_seeds_its_sampler(tmp_path):
    def first_params(name, seed):
        return suggest_params(create_study(tmp_path / f"{name}.jsonl", seed=seed).ask())

    assert first_params("first", seed=1) == first_params("same seed", seed=1)
    assert first_params("first", seed=1) != first_params("other seed", seed=2)


def test_create_study_silences_the_experimental_group_warning_only_while_creating(journal):
    filters = list(warnings.filters)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        create_study(journal)
    create_study(journal)

    assert warnings.filters == filters
    with pytest.warns(ExperimentalWarning, match="group"):  # Optuna still flags the option
        TPESampler(multivariate=True, group=True)


def test_best_params_gives_the_configuration_of_the_best_trial(journal):
    suggested = {}

    def fake_objective(trial):
        suggested[trial.number] = suggest_params(trial)
        return suggested[trial.number].c

    study = create_study(journal)
    study.optimize(fake_objective, n_trials=5)

    assert best_params(study) == suggested[study.best_trial.number]


def test_objective_returns_the_out_of_fold_macro_f1_and_keeps_the_other_scores(
    raw_dev, labels, folds
):
    labels.loc[[100, 103], "graphs"] = 0  # graphs is then learnt less well than strings
    trial = FixedTrial(VALUES)

    value = objective(trial, raw_dev, labels, folds)

    out_of_fold = cross_fit(build_pipeline(_model_params()), raw_dev, labels, folds)
    evaluation = evaluate(labels, out_of_fold.predictions)
    assert value == evaluation.macro_f1 != evaluation.summary.at[ALL_PROBLEMS, "micro_f1"]
    assert trial.user_attrs == {
        "micro_f1": evaluation.summary.at[ALL_PROBLEMS, "micro_f1"],
        "f1": evaluation.per_tag.loc[ALL_PROBLEMS, "f1"].to_dict(),
        "thresholds": out_of_fold.thresholds.to_dict(),
    }


def test_trial_scores_name_the_micro_f1_then_each_tag_f1_a_trial_keeps():
    complete = create_trial(
        value=0.5,
        user_attrs={"micro_f1": 0.625, "f1": {"graphs": 0.25, "strings": 0.75}, "thresholds": {}},
    )

    scores = trial_scores(complete)

    assert list(scores) == ["micro_f1", "f1/graphs", "f1/strings"]
    assert scores == {"micro_f1": 0.625, "f1/graphs": 0.25, "f1/strings": 0.75}
    assert trial_scores(create_trial(state=TrialState.FAIL)) == {"micro_f1": None}


def test_run_worker_adds_trials_until_the_study_has_the_target_number(
    journal, raw_dev, labels, folds
):
    run_worker(create_study(journal), 3, raw_dev, labels, folds)
    run_worker(create_study(journal, seed=TUNING_SEED + 1), 4, raw_dev, labels, folds)

    trials = create_study(journal).trials
    assert [trial.state for trial in trials] == [TrialState.COMPLETE] * 4
    assert all(set(trial.user_attrs) == {"micro_f1", "f1", "thresholds"} for trial in trials)


def test_run_worker_adds_no_trial_once_the_target_is_reached(journal, raw_dev, labels, folds):
    run_worker(create_study(journal), 2, raw_dev, labels, folds)
    run_worker(create_study(journal), 2, raw_dev, labels, folds)

    assert len(create_study(journal).trials) == 2


def test_worker_commands_start_one_worker_per_seed():
    commands = worker_commands(
        200, 2, data_dir=None, split_file=Path("split.csv"), journal=Path("journal.jsonl")
    )

    assert commands == [
        [sys.executable, "-m", "codeforces_tags.tuning", "worker", "--trials", "200", "--seed"]
        + [str(TUNING_SEED + worker), "--split", "split.csv", "--journal", "journal.jsonl"]
        for worker in range(2)
    ]


def test_worker_commands_pass_the_data_folder_only_when_given():
    paths = {"split_file": Path("split.csv"), "journal": Path("journal.jsonl")}

    with_data = worker_commands(10, 3, data_dir="data", **paths)
    without_data = worker_commands(10, 3, data_dir=None, **paths)

    assert [command[:-2] for command in with_data] == without_data
    assert [command[-2:] for command in with_data] == [["--data", "data"]] * 3
