"""The search space of the tag model, and the Optuna study that explores it.

``suggest_params`` is the only place where Optuna's parameter names exist: it turns a trial
into ``ModelParams``. Each trial scores its configuration with ``objective``: the out-of-fold
macro F1 of the dev set, with cross-fitted thresholds (docs/adr/0001-model-selection-protocol.md).
``trial_scores`` reads back the other scores it keeps.

The study lives in a journal file of JSON lines, which several worker processes can share:
each one runs ``run_worker`` with a sampler of its own seed, and ``worker_commands`` gives
their command lines.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from codeforces_tags.config import (
    OPTUNA_JOURNAL,
    TUNING_SEED,
    TUNING_SPACE,
    TUNING_STARTUP_TRIALS,
    TUNING_STUDY,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.model.params import BLOCKS, CLASS_WEIGHTS, CLASSIFIERS, ModelParams
from codeforces_tags.model.training import build_pipeline, cross_fit

YES_NO = (False, True)  # the choices of a yes/no parameter


def suggest_params(trial: optuna.trial.BaseTrial) -> ModelParams:
    """Draw one configuration of the tag model from the search space.

    The ranges are those of ``TUNING_SPACE``. The statement block is always kept, with weight
    1, and so is the code block. Each classifier has a C of its own (``c_logistic_regression``,
    ``c_linear_svm``), on a log scale, and the cue weight is drawn only when ``use_cues`` keeps
    the cue block.

    Args:
        trial: The trial that draws the values, or a ``FixedTrial`` that holds them.

    Returns:
        The configuration; its ``cue_weight`` is 0 when the cue block is dropped.
    """
    classifier = trial.suggest_categorical("classifier", CLASSIFIERS)
    use_cues = trial.suggest_categorical("use_cues", YES_NO)
    return ModelParams(
        classifier=classifier,
        c=_suggest_float(trial, f"c_{classifier}", log=True),
        class_weight=trial.suggest_categorical("class_weight", CLASS_WEIGHTS),
        include_notes=trial.suggest_categorical("include_notes", YES_NO),
        statement_ngram_max=_suggest_int(trial, "statement_ngram_max"),
        statement_min_df=_suggest_int(trial, "statement_min_df"),
        statement_max_df=_suggest_float(trial, "statement_max_df"),
        statement_sublinear_tf=trial.suggest_categorical("statement_sublinear_tf", YES_NO),
        code_weight=_suggest_float(trial, "code_weight"),
        code_ngram_max=_suggest_int(trial, "code_ngram_max"),
        code_min_df=_suggest_int(trial, "code_min_df"),
        code_sublinear_tf=trial.suggest_categorical("code_sublinear_tf", YES_NO),
        cue_weight=_suggest_float(trial, "cue_weight") if use_cues else 0.0,
        blocks=BLOCKS if use_cues else ("statement", "code"),
    )


def _suggest_float(trial: optuna.trial.BaseTrial, name: str, log: bool = False) -> float:
    """Draw a float in the range ``TUNING_SPACE[name]``."""
    low, high = TUNING_SPACE[name]
    return trial.suggest_float(name, low, high, log=log)


def _suggest_int(trial: optuna.trial.BaseTrial, name: str) -> int:
    """Draw an integer in the range ``TUNING_SPACE[name]``, both ends included."""
    low, high = TUNING_SPACE[name]
    return trial.suggest_int(name, low, high)


def best_params(study: optuna.Study) -> ModelParams:
    """Return the configuration of the study's best trial.

    Args:
        study: A study with at least one complete trial.

    Returns:
        The configuration ``suggest_params`` drew in the trial with the highest macro F1.
    """
    return suggest_params(optuna.trial.FixedTrial(study.best_params))


def create_study(
    journal: Path = OPTUNA_JOURNAL, name: str = TUNING_STUDY, seed: int = TUNING_SEED
) -> optuna.Study:
    """Create the study in the journal file, or load it when the journal already holds it.

    The sampler is a multivariate TPE: its first ``TUNING_STARTUP_TRIALS`` trials are random,
    and later ones model the parameters jointly. ``group=True`` gives each set of parameters
    drawn together (each classifier's C, the cue weight) a joint model of its own. With
    ``constant_liar=True``, a worker treats the trials still running in other workers as bad
    ones, so that the workers do not all try the same region.

    Args:
        journal: Optuna journal file; its folder is created if missing.
        name: Name of the study in the journal.
        seed: Seed of this process's sampler; each worker has its own.

    Returns:
        The study, which maximises the macro F1.
    """
    journal.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal)))
    with warnings.catch_warnings():
        # Optuna 5.0 flags `group` as experimental, at every sampler creation: the warning
        # says nothing about this study, and would repeat in every worker.
        warnings.filterwarnings("ignore", "Argument ``group``", ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=TUNING_STARTUP_TRIALS,
        )
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        study_name=name,
        direction="maximize",
        load_if_exists=True,
    )


def objective(
    trial: optuna.trial.BaseTrial,
    raw_problems: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
) -> float:
    """Score one configuration of the search space on the dev set, out of fold.

    Besides the macro F1 it returns, the trial keeps its micro F1, each tag's F1 and the final
    per-tag thresholds as user attributes: ``micro_f1``, ``f1`` and ``thresholds`` (the last
    two map each tag to a float).

    Args:
        trial: The trial that draws the configuration.
        raw_problems: Raw dev problems, ``select_part(read_samples(), split, "dev")``.
        labels: 0/1 focus-tag labels, with the index of ``raw_problems``.
        folds: Dev fold per problem, with the index of ``raw_problems``.

    Returns:
        The macro F1 of the out-of-fold predictions, with cross-fitted thresholds.
    """
    out_of_fold = cross_fit(build_pipeline(suggest_params(trial)), raw_problems, labels, folds)
    evaluation = evaluate(labels, out_of_fold.predictions)
    trial.set_user_attr("micro_f1", evaluation.micro_f1)
    trial.set_user_attr("f1", _per_tag(evaluation.tag_f1))
    trial.set_user_attr("thresholds", _per_tag(out_of_fold.thresholds))
    return evaluation.macro_f1


def _per_tag(values: pd.Series) -> dict[str, float]:
    """Return the values of a series indexed by tag as a JSON-able dict."""
    return {str(tag): float(value) for tag, value in values.items()}


def trial_scores(trial: optuna.trial.FrozenTrial) -> dict[str, float | None]:
    """Return the micro F1 and the per-tag F1 that ``objective`` keeps in a trial.

    Args:
        trial: A trial of the study; one that did not complete has a micro F1 of None and no
            per-tag F1.

    Returns:
        ``micro_f1``, then one ``f1/<tag>`` per tag, in the tag order of the labels.
    """
    per_tag = trial.user_attrs.get("f1", {})
    return {"micro_f1": trial.user_attrs.get("micro_f1")} | {
        f"f1/{tag}": f1 for tag, f1 in per_tag.items()
    }


def run_worker(
    study: optuna.Study,
    target_trials: int,
    raw_problems: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
) -> None:
    """Add trials to the study until it has ``target_trials`` complete ones.

    The count includes the trials of every worker sharing the journal. Workers stop after the
    trial they are running, so a study run by several workers can end a few trials past its
    target. A study that has reached its target gets no new trial.

    Args:
        study: The study, from ``create_study``.
        target_trials: Complete trials the study should have.
        raw_problems: Raw dev problems, ``select_part(read_samples(), split, "dev")``.
        labels: 0/1 focus-tag labels, with the index of ``raw_problems``.
        folds: Dev fold per problem, with the index of ``raw_problems``.
    """
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if len(complete) >= target_trials:
        return
    study.optimize(
        lambda trial: objective(trial, raw_problems, labels, folds),
        callbacks=[MaxTrialsCallback(target_trials, states=(TrialState.COMPLETE,))],
    )


def worker_commands(
    trials: int, workers: int, *, data_dir: str | None, split_file: Path, journal: Path
) -> list[list[str]]:
    """Return the command line of each worker process; worker i samples with TUNING_SEED + i.

    Args:
        trials: Complete trials the study should have when the workers stop.
        workers: Worker processes.
        data_dir: Dataset folder, passed on as ``--data``; None leaves it out, so that each
            worker uses the default (``$CODEFORCES_DATA_DIR`` or ``code_classification_dataset``).
        split_file: The frozen split.
        journal: The Optuna journal the workers share.

    Returns:
        One argument list per worker, for ``subprocess.Popen``.
    """
    data_args = [] if data_dir is None else ["--data", data_dir]
    return [
        [sys.executable, "-m", "codeforces_tags.tuning", "worker"]
        + ["--trials", str(trials), "--seed", str(TUNING_SEED + worker)]
        + ["--split", str(split_file), "--journal", str(journal), *data_args]
        for worker in range(workers)
    ]
