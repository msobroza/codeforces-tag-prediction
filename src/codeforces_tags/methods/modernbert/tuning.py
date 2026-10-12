"""The tuning study of a ModernBERT method, on one dev fold (docs/adr/0005).

A trial trains on the dev folds other than ``tuning_fold`` and scores that fold after each epoch:
its macro F1 with per-tag thresholds cross-fitted across ``subfolds`` stratified group sub-folds of
it, and its macro average precision. The study maximises the macro F1 after the last epoch. TPE
(multivariate, seeded) takes over after ``startup_trials`` random trials; a trial is pruned below
the median, at an epoch, of the trials trained for as many epochs. The journal is
``experiments/<name>/optuna.jsonl``; it records the study's recipe, and the study refuses to resume
under another. A trial left running by a killed process is marked failed (and recorded), and the
sampler is seeded again with the number of trials run, so a resumed study draws new
configurations. ``export_study`` freezes the results in ``assets/methods/<name>/tuning/``.

This module and ``__main__`` are the only ones of the package that import optuna.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState
from sklearn.metrics import average_precision_score

from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.modernbert.folds import DevData
from codeforces_tags.methods.modernbert.inputs import take
from codeforces_tags.methods.modernbert.modeling import predict_scores
from codeforces_tags.methods.modernbert.settings import (
    HISTORY_FILE,
    IMPORTANCES_FILE,
    OUT_OF_FOLD_FILE,
    PARAMS_FILE,
    TRIALS_FILE,
    Variant,
    study_recipe,
)
from codeforces_tags.methods.modernbert.training import (
    Params,
    TrainingSetup,
    params_from,
    train_model,
)
from codeforces_tags.model.thresholds import cross_fitted_predictions
from codeforces_tags.splitting import stratified_group_folds

FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)


class DivergedTrial(RuntimeError):
    """A trial whose held-out scores are not finite: its training diverged."""


class TuningData(NamedTuple):
    """What every trial reads: the training folds, the scored fold and its sub-folds."""

    train_ids: list  # token ids of the training problems
    train_labels: np.ndarray  # their 0/1 labels
    held_out_ids: list  # token ids of the scored fold
    held_out_labels: pd.DataFrame  # its 0/1 labels, indexed by sample_id
    subfolds: pd.Series  # its sub-fold per problem, indexed like held_out_labels


def tuning_data(variant: Variant, dev: DevData) -> TuningData:
    """Split the dev data as every trial reads it: the tuning fold against the others.

    Args:
        variant: The method's settings (tuning fold, sub-folds, seed).
        dev: The dev data.

    Returns:
        The training folds, the tuning fold and its stratified group sub-folds, which keep
        near-duplicate groups whole.
    """
    in_fold = (dev.folds == variant.tuning_fold).to_numpy()
    held_out_labels = dev.labels[in_fold]
    subfolds = stratified_group_folds(
        held_out_labels, dev.groups[in_fold], variant.subfolds, variant.seed
    ).rename("subfold")
    return TuningData(
        train_ids=take(dev.input_ids, ~in_fold),
        train_labels=dev.labels[~in_fold].to_numpy(),
        held_out_ids=take(dev.input_ids, in_fold),
        held_out_labels=held_out_labels,
        subfolds=subfolds,
    )


class SameEpochsMedianPruner(optuna.pruners.BasePruner):  # pylint: disable=too-few-public-methods
    """Prune a trial below the median, at an epoch, of the trials trained for as many epochs.

    The learning rate falls to 0 at a trial's last epoch, so comparing a trial with trials of
    another epoch count at the same epoch would favour the short ones.
    """

    def __init__(self, min_peers: int) -> None:
        """Keep the number of comparable trials needed before pruning.

        Args:
            min_peers: Finished trials of the same epoch count that reported the same epoch.
        """
        self.min_peers = min_peers

    def prune(self, study: optuna.Study, trial: FrozenTrial) -> bool:
        """Decide whether to stop the trial after its last reported epoch.

        Args:
            study: The study.
            trial: The running trial.

        Returns:
            True when the trial is below the median of its peers at this epoch; never after its
            last epoch, which ends it anyway.
        """
        epoch = trial.last_step
        if epoch is None or epoch >= trial.params["epochs"]:
            return False
        peers = [
            peer.intermediate_values[epoch]
            for peer in study.get_trials(deepcopy=False, states=FINISHED)
            if peer.params.get("epochs") == trial.params["epochs"]
            and epoch in peer.intermediate_values
        ]
        if len(peers) < self.min_peers:
            return False
        return trial.intermediate_values[epoch] < float(np.median(peers))


def suggest_params(trial: optuna.Trial, variant: Variant) -> Params:
    """Draw a configuration from a method's search space, in the notebooks' order.

    Args:
        trial: The trial that draws it.
        variant: The method's settings.

    Returns:
        ``LoraParams`` or ``FullParams``.
    """
    space = variant.search_space
    values = {
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True)
    }
    if variant.lora:
        values["lora_rank"] = trial.suggest_categorical("lora_rank", space["lora_rank"])
        values["lora_dropout"] = trial.suggest_float("lora_dropout", *space["lora_dropout"])
    else:
        values["weight_decay"] = trial.suggest_float(
            "weight_decay", *space["weight_decay"], log=True
        )
        values["classifier_dropout"] = trial.suggest_float(
            "classifier_dropout", *space["classifier_dropout"]
        )
    values["epochs"] = trial.suggest_int("epochs", *space["epochs"])
    values["batch_size"] = trial.suggest_categorical("batch_size", space["batch_size"])
    values["warmup_ratio"] = trial.suggest_float("warmup_ratio", *space["warmup_ratio"])
    values["pos_weight_power"] = trial.suggest_float("pos_weight_power", *space["pos_weight_power"])
    return params_from(variant, values)


def held_out_metrics(scores: np.ndarray, data: TuningData) -> dict[str, float]:
    """Score the tuning fold: macro F1 with cross-fitted thresholds, and macro AP.

    Args:
        scores: The fold's probabilities, one row per problem of ``data.held_out_labels``.
        data: The tuning data.

    Returns:
        ``macro_f1`` and ``macro_ap``.

    Raises:
        DivergedTrial: If a score is not finite.
    """
    if not np.isfinite(scores).all():
        raise DivergedTrial("non-finite held-out scores")
    truth = data.held_out_labels
    frame = pd.DataFrame(scores, index=truth.index, columns=truth.columns)
    _, predicted = cross_fitted_predictions(frame, truth, data.subfolds)
    return {
        "macro_f1": evaluate(truth, predicted).macro_f1,
        "macro_ap": float(average_precision_score(truth, frame, average="macro")),
    }


def epoch_reporter(
    trial: optuna.Trial, variant: Variant, data: TuningData, tokenizer
) -> Callable[[int, torch.nn.Module], None]:
    """Return the epoch hook of a trial: score the tuning fold, record, report, maybe prune.

    Args:
        trial: The trial being trained.
        variant: The method's settings.
        data: The tuning data.
        tokenizer: The tokenizer.

    Returns:
        ``hook(epoch, model)``, which raises ``optuna.TrialPruned`` when the pruner stops the
        trial and ``DivergedTrial`` on non-finite scores.
    """
    history: list[dict[str, float]] = []

    def hook(epoch: int, model: torch.nn.Module) -> None:
        scores = predict_scores(
            model,
            data.held_out_ids,
            tokenizer,
            batch_size=variant.score_batch,
            pad_to_multiple_of=variant.pad_to_multiple_of,
        )
        history.append({"epoch": epoch, **held_out_metrics(scores, data)})
        trial.set_user_attr("history", history)
        trial.report(history[-1]["macro_f1"], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned after epoch {epoch}")

    return hook


def open_study(variant: Variant, journal: Path) -> optuna.Study:
    """Open (or create) a method's study, refusing one made under another recipe.

    A trial left running by a killed process is marked failed and recorded in the user attribute
    ``interrupted_trials``; a resumed study gets a sampler seeded with the number of trials run.

    Args:
        variant: The method's settings.
        journal: The Optuna journal file.

    Returns:
        The study.

    Raises:
        ValueError: If the journal holds a study made under another recipe.
    """
    journal.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=variant.study_name,
        storage=JournalStorage(JournalFileBackend(str(journal))),
        direction="maximize",
        sampler=_sampler(variant, 0),
        pruner=SameEpochsMedianPruner(variant.pruner_min_peers),
        load_if_exists=True,
    )
    recipe = study_recipe(variant)
    if "recipe" not in study.user_attrs:
        study.set_user_attr("recipe", recipe)
    if study.user_attrs["recipe"] != recipe:
        raise ValueError(f"{journal} holds a study made under another recipe.")
    stale = [t.number for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))]
    for number in stale:
        study.tell(number, state=TrialState.FAIL)
    interrupted = sorted({*study.user_attrs.get("interrupted_trials", []), *stale})
    study.set_user_attr("interrupted_trials", interrupted)
    if study.trials:  # resumed: a sampler seeded as before would draw the first trials again
        study.sampler = _sampler(variant, len(study.trials))
    return study


def read_study(variant: Variant, journal: Path) -> optuna.Study:
    """Load a method's study without writing to its journal, refusing another recipe's.

    Args:
        variant: The method's settings.
        journal: The Optuna journal file.

    Returns:
        The study, read only.

    Raises:
        FileNotFoundError: If there is no journal.
        ValueError: If the study was made under another recipe.
    """
    if not journal.is_file():
        raise FileNotFoundError(f"{journal} does not exist: run the tune command first.")
    study = optuna.load_study(
        study_name=variant.study_name, storage=JournalStorage(JournalFileBackend(str(journal)))
    )
    if study.user_attrs.get("recipe") != study_recipe(variant):
        raise ValueError(f"{journal} holds a study made under another recipe.")
    return study


def _sampler(variant: Variant, offset: int) -> TPESampler:
    return TPESampler(
        seed=variant.seed + offset, n_startup_trials=variant.startup_trials, multivariate=True
    )


def run_study(
    setup: TrainingSetup,
    journal: Path,
    data: TuningData,
    *,
    callback: Callable[[optuna.Study, FrozenTrial], None] | None = None,
) -> optuna.Study:
    """Run trials until ``tuning_trials`` have finished (complete or pruned).

    A trial that runs out of GPU memory or diverges fails without stopping the study, up to
    ``max_failed_trials`` of them.

    Args:
        setup: Where to train; each trial draws its own configuration (``setup.params`` is
            unused).
        journal: The Optuna journal file.
        data: The tuning data.
        callback: Called after each trial (e.g. to print a line), or None.

    Returns:
        The study.

    Raises:
        RuntimeError: If fewer than ``tuning_trials`` trials finished.
    """
    variant = setup.variant
    study = open_study(variant, journal)

    def objective(trial: optuna.Trial) -> float:
        train_model(
            setup._replace(params=suggest_params(trial, variant)),
            data.train_ids,
            data.train_labels,
            seed=variant.seed,
            epoch_hook=epoch_reporter(trial, variant, data, setup.tokenizer),
        )
        return trial.user_attrs["history"][-1]["macro_f1"]

    remaining = variant.tuning_trials - len(study.get_trials(deepcopy=False, states=FINISHED))
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining + variant.max_failed_trials,
            gc_after_trial=True,
            callbacks=[
                *([callback] if callback else []),
                MaxTrialsCallback(variant.tuning_trials, states=FINISHED),
            ],
            catch=(torch.OutOfMemoryError, DivergedTrial),
        )
    finished = len(study.get_trials(deepcopy=False, states=FINISHED))
    if finished < variant.tuning_trials:
        raise RuntimeError(f"Only {finished} of {variant.tuning_trials} trials finished.")
    return study


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """Return one row per trial: state, value, epochs run, macro AP, minutes and parameters.

    A trial interrupted by a killed process has no duration: it was closed when the study
    resumed, so its duration would count the time the run was down.

    Args:
        study: The study.

    Returns:
        Columns ``number``, ``state``, ``value``, ``epochs_run``, ``macro_ap``, ``minutes``,
        then the parameters.
    """
    interrupted = set(study.user_attrs.get("interrupted_trials", []))
    rows = []
    for trial in study.trials:
        history = trial.user_attrs.get("history", [])
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "epochs_run": len(history),
                "macro_ap": history[-1]["macro_ap"] if history else math.nan,
                "minutes": (
                    math.nan
                    if trial.number in interrupted or trial.duration is None
                    else trial.duration.total_seconds() / 60
                ),
                **trial.params,
            }
        )
    return pd.DataFrame(rows)


def history_table(study: optuna.Study) -> pd.DataFrame:
    """Return the tuning fold's scores after each epoch of each finished trial.

    Args:
        study: The study.

    Returns:
        Columns ``number``, ``epoch``, ``macro_f1``, ``macro_ap``.
    """
    rows = [
        {"number": trial.number, **entry}
        for trial in study.get_trials(deepcopy=False, states=FINISHED)
        for entry in trial.user_attrs.get("history", [])
    ]
    return pd.DataFrame(rows, columns=["number", "epoch", "macro_f1", "macro_ap"])


def export_study(study: optuna.Study, variant: Variant, directory: Path) -> list[str]:
    """Freeze a finished study's results in a method's ``tuning/`` folder.

    Args:
        study: The finished study, from ``read_study`` (which checks its recipe).
        variant: The method's settings.
        directory: ``assets/methods/<name>/tuning``.

    Returns:
        The names of the files written: ``params.json`` (the chosen trial's parameters, number,
        tuning-fold macro F1 and the study's recipe), ``trials.csv``, ``history.csv``,
        ``importances.csv`` (seeded fANOVA) and ``out_of_fold.csv`` (the chosen trial's
        tuning-fold scores).

    """
    best = study.best_trial
    params = asdict(params_from(variant, best.params))
    directory.mkdir(parents=True, exist_ok=True)
    chosen = {
        "params": params,
        "trial": best.number,
        "tuning_macro_f1": best.value,
        "recipe": study_recipe(variant),
    }
    (directory / PARAMS_FILE).write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    trials_table(study).to_csv(directory / TRIALS_FILE, index=False)
    history_table(study).to_csv(directory / HISTORY_FILE, index=False)
    importances = pd.Series(
        get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=variant.seed)),
        name="importance",
    ).rename_axis("parameter")
    importances.to_csv(directory / IMPORTANCES_FILE)
    last = best.user_attrs["history"][-1]
    pd.Series(
        {"macro_f1": last["macro_f1"], "macro_ap": last["macro_ap"]}, name="value"
    ).rename_axis("metric").to_csv(directory / OUT_OF_FOLD_FILE)
    return [PARAMS_FILE, TRIALS_FILE, HISTORY_FILE, IMPORTANCES_FILE, OUT_OF_FOLD_FILE]
