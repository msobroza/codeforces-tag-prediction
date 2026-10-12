"""Tests for the one-fold tuning study of the ModernBERT methods, on tiny CPU models."""

import json
from dataclasses import asdict, replace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState, create_trial
from sklearn.metrics import average_precision_score

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.modernbert import tuning
from codeforces_tags.methods.modernbert.folds import read_dev
from codeforces_tags.methods.modernbert.modeling import (
    classifier_model,
    load_tokenizer,
    predict_scores,
)
from codeforces_tags.methods.modernbert.settings import (
    FULL,
    HISTORY_FILE,
    IMPORTANCES_FILE,
    LORA,
    OUT_OF_FOLD_FILE,
    PARAMS_FILE,
    TRIALS_FILE,
    VARIANTS,
    study_recipe,
)
from codeforces_tags.methods.modernbert.training import FullParams, LoraParams, TrainingSetup
from codeforces_tags.methods.modernbert.tuning import (
    DivergedTrial,
    SameEpochsMedianPruner,
    TuningData,
    epoch_reporter,
    export_study,
    held_out_metrics,
    history_table,
    open_study,
    read_study,
    run_study,
    suggest_params,
    trials_table,
    tuning_data,
)
from codeforces_tags.methods.modernbert.weights import snapshot_folder
from codeforces_tags.model.thresholds import cross_fitted_predictions

TAGS = list(FOCUS_TAGS)
EPOCHS = {"epochs": IntDistribution(1, 5)}
# One draw of each method's search space, in the order the notebooks drew it.
DRAWS = {
    LORA: {
        "learning_rate": 2e-4,
        "lora_rank": 16,
        "lora_dropout": 0.1,
        "epochs": 3,
        "batch_size": 32,
        "warmup_ratio": 0.1,
        "pos_weight_power": 0.5,
    },
    FULL: {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "classifier_dropout": 0.1,
        "epochs": 4,
        "batch_size": 16,
        "warmup_ratio": 0.05,
        "pos_weight_power": 0.25,
    },
}


def dev_problems(tiny_problems):
    """Return 24 raw dev problems, shuffled: 16 in the tuning fold 0, two in each other fold.

    The tuning fold holds every focus tag twice: once alone in its group, once in a group of two
    problems. Sample id ``s`` has the focus tag ``TAGS[(s - 1) % 8]``.
    """
    copies = [
        tiny_problems.set_axis(pd.RangeIndex(8 * k + 1, 8 * k + 9, name="sample_id"))
        for k in range(3)
    ]
    raw = pd.concat(copies)
    position = np.arange(8)
    raw["fold"] = np.concatenate([np.zeros(16, dtype=int), 1 + position // 2])
    raw["group"] = np.concatenate([position, 100 + position // 2, 200 + position])
    return raw.iloc[np.random.default_rng(0).permutation(len(raw))]


def entry(epoch, macro_f1, macro_ap):
    """Return one epoch's line of a trial's history."""
    return {"epoch": epoch, "macro_f1": macro_f1, "macro_ap": macro_ap}


def peer(epochs, values, state=TrialState.COMPLETE):
    """Return a trial of ``epochs`` epochs that reported ``values`` (macro F1 by epoch)."""
    return create_trial(
        state=state,
        value=None if state == TrialState.FAIL else values[max(values)],
        params={"epochs": epochs},
        distributions=EPOCHS,
        intermediate_values=values,
    )


def running(epochs, values):
    """Return a running trial of ``epochs`` epochs that reported ``values`` so far."""
    return create_trial(
        state=TrialState.RUNNING,
        params={"epochs": epochs},
        distributions=EPOCHS,
        intermediate_values=values,
    )


def study_of(*trials):
    """Return an in-memory study that holds these trials."""
    study = optuna.create_study(direction="maximize")
    for trial in trials:
        study.add_trial(trial)
    return study


def add_trial(study, variant, state, macro_f1s):
    """Run a trial of a method's search space by hand, with one macro F1 per epoch it ran."""
    trial = study.ask()
    params = suggest_params(trial, variant)
    history = [entry(epoch, f1, f1 / 2) for epoch, f1 in enumerate(macro_f1s, start=1)]
    trial.set_user_attr("history", history)
    for line in history:
        trial.report(line["macro_f1"], line["epoch"])
    study.tell(trial, macro_f1s[-1] if state == TrialState.COMPLETE else None, state=state)
    return params


class SeededSampler(TPESampler):
    """A TPE sampler that remembers its seed."""

    def __init__(self, *, seed, **kwargs):
        """Build the sampler and keep its seed."""
        super().__init__(seed=seed, **kwargs)
        self.seed = seed


@pytest.fixture(name="variant")
def variant_fixture(tiny_variants):
    """Return modernbert_lora's tiny settings, with trials of one or two epochs."""
    variant = tiny_variants[LORA]
    return replace(variant, search_space={**variant.search_space, "epochs": (1, 2)})


@pytest.fixture(name="setup")
def setup_fixture(caches, variant, tmp_path):
    """Return where a trial trains: the tiny snapshot, on the CPU."""
    del caches  # points the pretrained cache at the tiny snapshot
    snapshot = snapshot_folder(variant.model_repo, variant.model_revision)
    return TrainingSetup(
        variant, None, snapshot, load_tokenizer(snapshot), torch.device("cpu"), tmp_path / "out"
    )


@pytest.fixture(name="dev")
def dev_fixture(setup, tiny_problems):
    """Return the dev data of ``dev_problems``."""
    return read_dev(dev_problems(tiny_problems), setup.tokenizer, setup.variant)


@pytest.fixture(name="data")
def data_fixture(setup, dev):
    """Return the tuning data of the dev data."""
    return tuning_data(setup.variant, dev)


@pytest.fixture(name="model")
def model_fixture(setup):
    """Return an untrained classifier on the tiny snapshot."""
    torch.manual_seed(0)
    return classifier_model(setup.snapshot, weights=True)


@pytest.fixture(name="held_out")
def held_out_fixture():
    """Return tuning data of 16 problems: each focus tag once in each of two sub-folds."""
    index = pd.RangeIndex(1, 17, name="sample_id")
    labels = pd.DataFrame(np.tile(np.eye(8, dtype=int), (2, 1)), index=index, columns=TAGS)
    subfolds = pd.Series(np.repeat([0, 1], 8), index=index, name="subfold")
    return TuningData([], np.empty((0, 8)), [[0]] * 16, labels, subfolds)


@pytest.fixture(name="fake_training")
def fake_training_fixture(monkeypatch, data):
    """Replace training by its epoch hooks on fixed scores; return the configurations trained."""
    trained = []
    scores = data.held_out_labels.to_numpy() * 0.8 + 0.1

    def train(setup, train_ids, train_labels, *, seed, epoch_hook):
        """Call the epoch hook after each epoch of the trial's configuration."""
        del train_ids, train_labels, seed
        trained.append(setup.params)
        for epoch in range(1, setup.params.epochs + 1):
            epoch_hook(epoch, None)

    monkeypatch.setattr(tuning, "train_model", train)
    monkeypatch.setattr(tuning, "predict_scores", lambda *args, **kwargs: scores)
    return trained


def test_tuning_data_scores_the_tuning_fold_and_trains_on_the_others(dev, data):
    ids = dict(zip(dev.folds.index, dev.input_ids, strict=True))
    held_out = list(dev.folds.index[dev.folds == 0])
    training = list(dev.folds.index[dev.folds != 0])
    assert sorted(held_out) == list(range(1, 17))
    assert list(data.held_out_labels.index) == held_out
    assert list(data.held_out_labels.idxmax(axis=1)) == [TAGS[(s - 1) % 8] for s in held_out]
    assert data.held_out_ids == [ids[s] for s in held_out]
    assert data.train_ids == [ids[s] for s in training]
    np.testing.assert_array_equal(data.train_labels, dev.labels.loc[training].to_numpy())
    assert [TAGS[column] for column in data.train_labels.argmax(axis=1)] == [
        TAGS[(s - 1) % 8] for s in training
    ]


def test_tuning_data_splits_the_tuning_fold_into_sub_folds_that_keep_groups_whole(dev, data):
    groups = dev.groups.loc[data.held_out_labels.index]
    assert data.subfolds.name == "subfold"
    assert data.subfolds.index.equals(data.held_out_labels.index)
    assert sorted(data.subfolds.unique()) == [0, 1]
    assert groups.value_counts().max() == 2
    assert (data.subfolds.groupby(groups).nunique() == 1).all()


def test_pruner_never_prunes_a_trial_before_it_reports_an_epoch():
    study = study_of(peer(3, {1: 0.9}))
    assert not SameEpochsMedianPruner(1).prune(study, running(3, {}))


def test_pruner_never_prunes_a_trial_at_its_last_epoch():
    study = study_of(peer(2, {1: 0.9, 2: 0.9}))
    pruner = SameEpochsMedianPruner(1)
    assert pruner.prune(study, running(2, {1: 0.1}))
    assert not pruner.prune(study, running(2, {1: 0.1, 2: 0.1}))


def test_pruner_waits_for_enough_finished_peers_of_the_same_epoch_count():
    one = study_of(peer(3, {1: 0.9}))
    two = study_of(peer(3, {1: 0.9}), peer(3, {1: 0.8}, TrialState.PRUNED))
    pruner = SameEpochsMedianPruner(2)
    assert not pruner.prune(one, running(3, {1: 0.1}))
    assert pruner.prune(two, running(3, {1: 0.1}))


@pytest.mark.parametrize(("value", "pruned"), [(0.39, True), (0.4, False), (0.6, False)])
def test_pruner_prunes_below_the_median_of_its_peers_and_keeps_at_or_above_it(value, pruned):
    study = study_of(
        peer(3, {1: 0.2, 2: 0.1}),
        peer(3, {1: 0.4}, TrialState.PRUNED),
        peer(3, {1: 0.6, 2: 0.7, 3: 0.8}),
    )
    assert SameEpochsMedianPruner(3).prune(study, running(3, {1: value})) == pruned


def test_pruner_ignores_peers_of_another_epoch_count_unfinished_or_without_that_epoch():
    others = [
        peer(4, {1: 0.9, 2: 0.9}),  # another epoch count
        peer(3, {1: 0.9, 2: 0.9}, TrialState.FAIL),  # failed
        running(3, {1: 0.9, 2: 0.9}),  # still running
        peer(3, {1: 0.9}, TrialState.PRUNED),  # pruned before epoch 2
        create_trial(value=0.9, intermediate_values={2: 0.9}),  # no epoch count
    ]
    trial = running(3, {1: 0.1, 2: 0.1})
    pruner = SameEpochsMedianPruner(1)
    assert not pruner.prune(study_of(*others), trial)
    assert pruner.prune(study_of(*others, peer(3, {1: 0.9, 2: 0.9})), trial)


@pytest.mark.parametrize(("name", "kind"), [(LORA, LoraParams), (FULL, FullParams)])
def test_suggest_params_draws_a_methods_configuration_in_the_notebooks_order(name, kind):
    trial = optuna.trial.FixedTrial(DRAWS[name])
    assert suggest_params(trial, VARIANTS[name]) == kind(**DRAWS[name])
    assert list(trial.params) == list(DRAWS[name])


def test_suggest_params_draws_from_the_methods_search_space_with_log_scaled_rates():
    lora, full = VARIANTS[LORA].search_space, VARIANTS[FULL].search_space
    shared = {
        "batch_size": CategoricalDistribution(lora["batch_size"]),
        "warmup_ratio": FloatDistribution(*lora["warmup_ratio"]),
        "pos_weight_power": FloatDistribution(*lora["pos_weight_power"]),
    }
    expected = {
        LORA: {
            "learning_rate": FloatDistribution(*lora["learning_rate"], log=True),
            "lora_rank": CategoricalDistribution(lora["lora_rank"]),
            "lora_dropout": FloatDistribution(*lora["lora_dropout"]),
            "epochs": IntDistribution(*lora["epochs"]),
            **shared,
        },
        FULL: {
            "learning_rate": FloatDistribution(*full["learning_rate"], log=True),
            "weight_decay": FloatDistribution(*full["weight_decay"], log=True),
            "classifier_dropout": FloatDistribution(*full["classifier_dropout"]),
            "epochs": IntDistribution(*full["epochs"]),
            **shared,
        },
    }
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for name, distributions in expected.items():
        trial = study.ask()
        suggest_params(trial, VARIANTS[name])
        assert trial.distributions == distributions


def test_held_out_metrics_are_one_when_the_scores_separate_every_tag(held_out):
    noise = np.random.default_rng(0).uniform(0.0, 0.1, size=(16, 8))
    metrics = held_out_metrics(held_out.held_out_labels.to_numpy() * 0.8 + noise, held_out)
    assert metrics == {"macro_f1": 1.0, "macro_ap": 1.0}


def test_held_out_metrics_score_cross_fitted_predictions_and_average_precision(held_out):
    scores = np.random.default_rng(1).random((16, 8))
    truth = held_out.held_out_labels
    frame = pd.DataFrame(scores, index=truth.index, columns=truth.columns)
    _, predicted = cross_fitted_predictions(frame, truth, held_out.subfolds)
    assert held_out_metrics(scores, held_out) == {
        "macro_f1": evaluate(truth, predicted).macro_f1,
        "macro_ap": average_precision_score(truth, scores, average="macro"),
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_held_out_metrics_refuse_scores_that_are_not_finite(held_out, bad):
    scores = np.full((16, 8), 0.5)
    scores[3, 2] = bad
    with pytest.raises(DivergedTrial, match="non-finite"):
        held_out_metrics(scores, held_out)


def test_epoch_reporter_scores_the_tuning_fold_records_its_history_and_reports_it(
    setup, data, model
):
    study = optuna.create_study(direction="maximize", pruner=SameEpochsMedianPruner(1))
    trial = study.ask(fixed_distributions={"epochs": IntDistribution(2, 2)})
    hook = epoch_reporter(trial, setup.variant, data, setup.tokenizer)
    hook(1, model)
    hook(2, model)
    variant = setup.variant
    scores = predict_scores(
        model,
        data.held_out_ids,
        setup.tokenizer,
        batch_size=variant.score_batch,
        pad_to_multiple_of=variant.pad_to_multiple_of,
    )
    metrics = held_out_metrics(scores, data)
    frozen = study.trials[0]
    assert frozen.user_attrs["history"] == [{"epoch": 1, **metrics}, {"epoch": 2, **metrics}]
    assert frozen.intermediate_values == {1: metrics["macro_f1"], 2: metrics["macro_f1"]}


def test_epoch_reporter_stops_a_trial_its_pruner_prunes(setup, data, model):
    study = optuna.create_study(direction="maximize", pruner=SameEpochsMedianPruner(1))
    study.add_trial(peer(2, {1: 2.0}))  # above any macro F1
    trial = study.ask(fixed_distributions={"epochs": IntDistribution(2, 2)})
    hook = epoch_reporter(trial, setup.variant, data, setup.tokenizer)
    with pytest.raises(optuna.TrialPruned, match="pruned after epoch 1"):
        hook(1, model)
    assert [line["epoch"] for line in study.trials[1].user_attrs["history"]] == [1]


def test_epoch_reporter_reports_nothing_for_a_diverged_training(setup, data, monkeypatch):
    monkeypatch.setattr(tuning, "predict_scores", lambda *args, **kwargs: np.full((16, 8), np.nan))
    study = optuna.create_study(direction="maximize")
    hook = epoch_reporter(study.ask(), setup.variant, data, setup.tokenizer)
    with pytest.raises(DivergedTrial):
        hook(1, None)
    assert study.trials[0].intermediate_values == {}


def test_open_study_creates_a_study_that_records_its_recipe(variant, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "TPESampler", SeededSampler)
    journal = tmp_path / "experiments" / variant.name / "optuna.jsonl"
    study = open_study(variant, journal)
    assert journal.is_file()
    assert study.study_name == variant.study_name
    assert study.direction == StudyDirection.MAXIMIZE
    assert study.user_attrs == {"recipe": study_recipe(variant), "interrupted_trials": []}
    assert isinstance(study.pruner, SameEpochsMedianPruner)
    assert study.pruner.min_peers == variant.pruner_min_peers
    assert study.sampler.seed == variant.seed


def test_open_study_resumes_a_journal_with_a_sampler_seeded_by_the_trials_run(
    variant, tmp_path, monkeypatch
):
    monkeypatch.setattr(tuning, "TPESampler", SeededSampler)
    journal = tmp_path / "optuna.jsonl"
    first = open_study(variant, journal)
    first.tell(first.ask(), 0.5)
    first.tell(first.ask(), 0.7)
    resumed = open_study(variant, journal)
    assert [trial.value for trial in resumed.trials] == [0.5, 0.7]
    assert resumed.user_attrs == first.user_attrs
    assert resumed.sampler.seed == variant.seed + 2


def test_open_study_closes_and_records_the_trials_a_killed_process_left_running(variant, tmp_path):
    journal = tmp_path / "optuna.jsonl"
    first = open_study(variant, journal)
    first.ask()  # left running when its process was killed
    first.tell(first.ask(), 0.5)
    second = open_study(variant, journal)
    assert [trial.state for trial in second.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert second.user_attrs["interrupted_trials"] == [0]
    second.ask()  # killed again
    third = open_study(variant, journal)
    states = [TrialState.FAIL, TrialState.COMPLETE, TrialState.FAIL]
    assert [trial.state for trial in third.trials] == states
    assert third.user_attrs["interrupted_trials"] == [0, 2]


def test_open_study_refuses_a_journal_of_another_recipe(variant, tmp_path):
    journal = tmp_path / "optuna.jsonl"
    open_study(variant, journal)
    with pytest.raises(ValueError, match="another recipe"):
        open_study(replace(variant, subfolds=variant.subfolds + 1), journal)


def test_read_study_needs_a_journal(variant, tmp_path):
    with pytest.raises(FileNotFoundError, match="run the tune command first"):
        read_study(variant, tmp_path / "optuna.jsonl")


def test_read_study_refuses_a_journal_of_another_recipe(variant, tmp_path):
    journal = tmp_path / "optuna.jsonl"
    open_study(variant, journal)
    with pytest.raises(ValueError, match="another recipe"):
        read_study(replace(variant, seed=variant.seed + 1), journal)


def test_read_study_reads_a_study_without_writing_to_its_journal(variant, tmp_path):
    journal = tmp_path / "optuna.jsonl"
    study = open_study(variant, journal)
    study.tell(study.ask(), 0.5)
    study.ask()  # left running: opening the study to write would close it
    before = journal.read_bytes()
    read = read_study(variant, journal)
    assert journal.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [journal.name]
    assert [trial.state for trial in read.trials] == [TrialState.COMPLETE, TrialState.RUNNING]
    assert read.user_attrs["recipe"] == study_recipe(variant)


def test_run_study_trains_trials_until_enough_have_finished(setup, data, tmp_path):
    seen = []
    study = run_study(
        setup,
        tmp_path / "optuna.jsonl",
        data,
        callback=lambda study, trial: seen.append(trial.number),
    )
    assert seen == [0, 1]
    assert all(trial.state in tuning.FINISHED for trial in study.trials)
    histories = [trial.user_attrs["history"] for trial in study.trials]
    assert [[line["epoch"] for line in history] for history in histories] == [
        list(range(1, len(history) + 1)) for history in histories
    ]
    assert [trial.value for trial in study.trials] == [h[-1]["macro_f1"] for h in histories]
    assert all(
        len(trial.user_attrs["history"]) == trial.params["epochs"]
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    )


def test_run_study_runs_no_trial_when_enough_have_finished(setup, data, tmp_path, fake_training):
    journal = tmp_path / "optuna.jsonl"
    run_study(setup, journal, data)
    assert len(fake_training) == 2
    seen = []
    study = run_study(setup, journal, data, callback=lambda study, trial: seen.append(trial))
    assert not seen
    assert len(fake_training) == 2
    assert len(study.trials) == 2


def test_run_study_resumes_after_a_killed_trial_and_runs_only_the_missing_ones(
    setup, data, tmp_path, fake_training
):
    journal = tmp_path / "optuna.jsonl"
    one_trial = setup._replace(variant=replace(setup.variant, tuning_trials=1))
    run_study(one_trial, journal, data)
    open_study(setup.variant, journal).ask()  # left running when its process was killed
    study = run_study(setup, journal, data)
    assert [trial.state for trial in study.trials] == [
        TrialState.COMPLETE,
        TrialState.FAIL,
        TrialState.COMPLETE,
    ]
    assert study.user_attrs["interrupted_trials"] == [1]
    assert fake_training == [
        suggest_params(optuna.trial.FixedTrial(study.trials[number].params), setup.variant)
        for number in (0, 2)
    ]
    assert study.trials[2].user_attrs["history"][-1]["macro_f1"] == study.trials[2].value


def test_run_study_fails_when_too_few_trials_finish(setup, data, tmp_path, monkeypatch):
    errors = iter([torch.OutOfMemoryError("CUDA out of memory"), *[DivergedTrial("nan")] * 2])

    def fail(*args, **kwargs):
        """Fail as a trial out of GPU memory, then as diverged ones."""
        del args, kwargs
        raise next(errors)

    monkeypatch.setattr(tuning, "train_model", fail)
    journal = tmp_path / "optuna.jsonl"
    with pytest.raises(RuntimeError, match="Only 0 of 2 trials finished"):
        run_study(setup, journal, data)
    study = read_study(setup.variant, journal)
    assert [trial.state for trial in study.trials] == [TrialState.FAIL] * 3


def test_trials_table_has_a_row_per_trial_and_no_duration_for_an_interrupted_one(variant, tmp_path):
    journal = tmp_path / "optuna.jsonl"
    first = open_study(variant, journal)
    done = first.ask(fixed_distributions={"epochs": IntDistribution(2, 2)})
    done.set_user_attr("history", [entry(1, 0.4, 0.5), entry(2, 0.6, 0.7)])
    first.tell(done, 0.6)
    first.ask(fixed_distributions={"epochs": IntDistribution(1, 1)})  # its process was killed
    study = open_study(variant, journal)
    study.ask()  # running: no duration yet
    table = trials_table(study)
    assert list(table.columns) == [
        "number",
        "state",
        "value",
        "epochs_run",
        "macro_ap",
        "minutes",
        "epochs",
    ]
    assert list(table["number"]) == [0, 1, 2]
    assert list(table["state"]) == ["COMPLETE", "FAIL", "RUNNING"]
    assert table["value"].tolist()[0] == 0.6
    assert table["value"].iloc[1:].isna().all()
    assert list(table["epochs_run"]) == [2, 0, 0]
    assert table["macro_ap"].iloc[0] == 0.7
    assert table["macro_ap"].iloc[1:].isna().all()
    assert table.at[0, "minutes"] == study.trials[0].duration.total_seconds() / 60
    assert study.trials[1].duration is not None  # closed when the study resumed
    assert table["minutes"].iloc[1:].isna().all()
    assert table["epochs"].iloc[:2].tolist() == [2, 1]
    assert np.isnan(table.at[2, "epochs"])


def test_history_table_holds_each_epoch_of_each_finished_trial():
    lines = [entry(1, 0.1, 0.2), entry(2, 0.3, 0.4)]
    study = study_of(
        create_trial(value=0.3, user_attrs={"history": lines}),
        create_trial(state=TrialState.FAIL, user_attrs={"history": lines}),
        create_trial(state=TrialState.PRUNED, value=0.5, user_attrs={"history": lines[:1]}),
        create_trial(state=TrialState.RUNNING, user_attrs={"history": lines}),
        create_trial(value=0.9),  # no history
    )
    expected = pd.DataFrame(
        {
            "number": [0, 0, 2],
            "epoch": [1, 2, 1],
            "macro_f1": [0.1, 0.3, 0.1],
            "macro_ap": [0.2, 0.4, 0.2],
        }
    )
    pd.testing.assert_frame_equal(history_table(study), expected)


def test_history_table_of_a_study_without_finished_trials_is_empty():
    table = history_table(study_of(create_trial(state=TrialState.FAIL)))
    assert table.empty
    assert list(table.columns) == ["number", "epoch", "macro_f1", "macro_ap"]


@pytest.mark.parametrize("name", [LORA, FULL])
def test_export_study_freezes_the_chosen_trial_and_the_studys_tables(name, tmp_path):
    variant = VARIANTS[name]
    study = open_study(variant, tmp_path / "optuna.jsonl")
    drawn = [
        add_trial(study, variant, TrialState.COMPLETE, [0.3, 0.5]),
        add_trial(study, variant, TrialState.COMPLETE, [0.6, 0.8]),
        add_trial(study, variant, TrialState.PRUNED, [0.2]),
        add_trial(study, variant, TrialState.FAIL, []),
        add_trial(study, variant, TrialState.COMPLETE, [0.4, 0.7]),
    ]
    directory = tmp_path / "assets" / name / "tuning"
    written = export_study(study, variant, directory)
    assert written == [PARAMS_FILE, TRIALS_FILE, HISTORY_FILE, IMPORTANCES_FILE, OUT_OF_FOLD_FILE]
    assert sorted(path.name for path in directory.iterdir()) == sorted(written)
    text = (directory / PARAMS_FILE).read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == {
        "params": asdict(drawn[1]),
        "trial": 1,
        "tuning_macro_f1": 0.8,
        "recipe": study_recipe(variant),
    }
    trials = pd.read_csv(directory / TRIALS_FILE)
    pd.testing.assert_frame_equal(trials, trials_table(study))
    assert list(trials["state"]) == ["COMPLETE", "COMPLETE", "PRUNED", "FAIL", "COMPLETE"]
    history = pd.read_csv(directory / HISTORY_FILE)
    pd.testing.assert_frame_equal(history, history_table(study))
    assert list(zip(history["number"], history["epoch"], strict=True)) == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 1),
        (4, 1),
        (4, 2),
    ]
    importances = pd.read_csv(directory / IMPORTANCES_FILE, index_col="parameter")["importance"]
    assert sorted(importances.index) == sorted(asdict(drawn[0]))
    assert (importances >= 0).all()
    assert importances.sum() == pytest.approx(1.0)
    out_of_fold = pd.read_csv(directory / OUT_OF_FOLD_FILE, index_col="metric")["value"]
    assert out_of_fold.to_dict() == {"macro_f1": 0.8, "macro_ap": 0.4}


def test_export_study_writes_the_same_importances_twice(tmp_path):
    variant = VARIANTS[LORA]
    study = open_study(variant, tmp_path / "optuna.jsonl")
    for values in ([0.3], [0.8], [0.5], [0.6]):
        add_trial(study, variant, TrialState.COMPLETE, values)
    export_study(study, variant, tmp_path / "first")
    export_study(study, variant, tmp_path / "second")
    first = (tmp_path / "first" / IMPORTANCES_FILE).read_bytes()
    assert (tmp_path / "second" / IMPORTANCES_FILE).read_bytes() == first
