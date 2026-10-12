"""Command line of the two ModernBERT methods (docs/adr/0005, 0006).

    uv run --group finetuning python -m codeforces_tags.methods.modernbert COMMAND --method NAME

NAME is modernbert_lora (ModernBERT-large with LoRA) or modernbert_base_full (ModernBERT-base,
every weight trained). The commands, in the order of a run from scratch:

- fetch: download the pinned pretrained snapshot into the pretrained cache.
- tune: run the one-fold Optuna study; it resumes from its journal, experiments/NAME/optuna.jsonl.
  The frozen methods reuse the studies that the former notebooks 05 and 06 ran (their journals
  are in experiments/NAME/): no command re-runs them.
- export: freeze the study's results in assets/methods/NAME/tuning/ (params.json and tables),
  reading the journal without writing to it.
- train: train one fold model per dev fold with tuning/params.json, into the trained cache; it
  resumes after the last complete fold model, and refuses the cache of a frozen method (its
  weights are restored by copying them: training again makes other weights). adopt copies the
  fold models that the former notebooks 05 and 06 trained instead (--source), after checking
  their recipe and configuration.
- freeze [--allow-dirty]: the first time, it reads the fold models' out-of-fold scores, cross-fits
  the dev predictions' thresholds, saves the method folder (the caches' weight lists, no weight),
  checks that the loaded bag's fold models give back their out-of-fold scores, scores test once
  with the bag, and freezes predictions.csv, scores.csv and run_record.json, and each fold model's
  test scores in analysis/. A second run re-derives nothing (score_atol > 0): it checks the frozen
  files. Every run then logs the frozen record and scores to MLflow.
- verify: offline, it re-derives the dev predictions and the thresholds from scores.csv, re-scores
  each dev fold with its fold model and test with the bag (within the registry's score_atol, 1e-2),
  and prints the scores. It re-scores on the CPU only, in float32 (hours for modernbert_lora): one
  batch of many problems held the GPU at full power and rebooted a machine once. Away from the GPU
  batches that froze them, the scores move by up to 8.25e-3 (docs/adr/0005). --retrain also trains
  the fold models again, each from the seed that trained it (GPU, about an hour for
  modernbert_lora), into the trained cache's retrain/ folder. GPU training is not bit-reproducible,
  so it passes on the agreement bounds that the settings declare (retrain_*): score correlations
  and gaps, flipped predictions, and the paired difference of the dev macro F1. It also prints the
  tuning-fold objective of the retrained fold model that the study's tuning fold leaves out: the
  chosen trial's training, run again.
- latency-cpu: time the method folder on the CPU as tracking latency times it (on CUDA when it is
  available), into analysis/cpu/latency.csv.

Every command takes --data DIR (the dataset), --split FILE (the frozen split) and --assets DIR (the
folder of the frozen methods); freeze also takes --tracking-uri URI and --artifact-dir DIR (the
MLflow store). The defaults come from codeforces_tags.config.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from codeforces_tags.config import (
    EXPERIMENTS_DIR,
    FOCUS_TAGS,
    LATENCY_PROBLEMS,
    LATENCY_SEED,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)
from codeforces_tags.evaluation.report import compare, evaluate
from codeforces_tags.evaluation.timing import (
    LATENCY_FILE,
    Latency,
    latency_summary,
    save_latency,
    time_bulk,
    time_single,
)
from codeforces_tags.methods import (
    METHODS,
    MODEL_FOLDER,
    load_method,
    save_method,
    sibling_methods,
)
from codeforces_tags.methods.modernbert.folds import (
    DevData,
    adopt_folds,
    fold_name,
    read_dev,
    read_fold_models,
    train_folds,
)
from codeforces_tags.methods.modernbert.method import FineTunedBag, load
from codeforces_tags.methods.modernbert.modeling import bag_mean, load_tokenizer, resolve_device
from codeforces_tags.methods.modernbert.settings import (
    ANALYSIS_FOLDER,
    CONFIG_FILES,
    CPU_LATENCY_FOLDER,
    ENCODER_FILE,
    FOLD_TEST_SCORES_FILE,
    FOLD_WEIGHTS_FILE,
    JOURNAL_FILE,
    PARAMS_FILE,
    PRETRAINED_FILE,
    RETRAIN_FOLDER,
    TRAINED_FILE,
    TUNING_FOLDER,
    VARIANTS,
    Variant,
    settings,
)
from codeforces_tags.methods.modernbert.training import TrainingSetup, params_from
from codeforces_tags.methods.modernbert.tuning import (
    export_study,
    held_out_metrics,
    read_study,
    run_study,
    tuning_data,
)
from codeforces_tags.methods.modernbert.weights import (
    pretrained_weight_list,
    snapshot_folder,
    trained_root,
    trained_weight_list,
)
from codeforces_tags.methods.parts import Parts, add_dataset_arguments, read_parts
from codeforces_tags.model.thresholds import cross_fitted_predictions, per_tag_thresholds
from codeforces_tags.provenance import (
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    make_run_record,
    sha256_json,
)
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.methods import log_method, log_run_extras, part_metrics

# The folder of fold models that the former notebooks 05 and 06 trained, in experiments/<name>/.
NOTEBOOK_FOLDERS = {True: "adapters", False: "models"}
# What the dev rows of a frozen ModernBERT method are (docs/adr/0005).
DEV_ESTIMATE = (
    "Each dev problem is scored by the fold model that did not train on it, and predicted with "
    "thresholds cross-fitted across the dev folds. The bag scores test only, so the dev rows "
    "estimate one fold model, not the bag."
)
# The run metric of the dev folds that the study did not score: chosen on none of them.
OUTSIDE_TUNING_FOLD_METRIC = "outside_tuning_fold/macro_f1"


class _DevRows(NamedTuple):
    """What freeze derives for dev from the fold models' out-of-fold scores."""

    records: dict  # each fold model's training.json, by name
    scores: pd.DataFrame  # out-of-fold scores, float64
    predictions: pd.DataFrame  # 0/1, thresholds cross-fitted across the dev folds
    thresholds: pd.Series  # chosen on every out-of-fold score: the bag's, for test


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, ``method``, ``data``, ``split`` and ``assets``, and
        each command's own options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.methods.modernbert",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download the pinned pretrained snapshot")
    tune = commands.add_parser("tune", help="run the one-fold study")
    export = commands.add_parser("export", help="freeze the study's results in tuning/")
    for command in (tune, export):
        command.add_argument("--journal", type=Path, help="Optuna journal (default: experiments/)")
    train = commands.add_parser("train", help="train one fold model per dev fold")
    train.add_argument("--root", type=Path, help="folder of the fold models (default: the cache)")
    adopt = commands.add_parser("adopt", help="copy the fold models a notebook trained")
    adopt.add_argument("--source", type=Path, help="the notebook's folder of fold models")
    freeze = commands.add_parser("freeze", help="freeze the files once, then log to MLflow")
    freeze.add_argument(
        "--allow-dirty", action="store_true", help="freeze from uncommitted code (recorded)"
    )
    add_store_arguments(freeze)
    verify = commands.add_parser("verify", help="re-derive the frozen files, report their scores")
    verify.add_argument("--retrain", action="store_true", help="also train the fold models again")
    commands.add_parser("latency-cpu", help="time the method folder on the CPU")
    for command in commands.choices.values():
        command.add_argument("--method", required=True, choices=sorted(VARIANTS))
        add_dataset_arguments(command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of a ModernBERT method.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status, 0; a refused dataset, a dirty tree, a missing or changed weight file, or
        a frozen file that the code no longer reproduces raises.
    """
    args = parse_args(argv)
    commands = {
        "fetch": _fetch,
        "tune": _tune,
        "export": _export,
        "train": _train,
        "adopt": _adopt,
        "freeze": _freeze,
        "verify": _verify,
        "latency-cpu": _latency_cpu,
    }
    return commands[args.command](args, VARIANTS[args.method])


def recipe_sha256(variant: Variant, chosen: Mapping, pretrained: Mapping) -> str:
    """Hash what defines a frozen ModernBERT method: committed inputs only (docs/adr/0004).

    Not its weights or thresholds, which are outcomes of training: ``trained.json`` pins the
    weights, and the thresholds are re-derived from the frozen scores.

    Args:
        variant: The method's settings.
        chosen: The content of ``tuning/params.json``: the configuration its study chose.
        pretrained: The content of ``pretrained.json``: the pretrained files it reads.

    Returns:
        The sha256 hex digest, the ``recipe_sha256`` of its run record.
    """
    return sha256_json(
        {
            "method": variant.name,
            "tags": list(FOCUS_TAGS),
            "settings": settings(variant),
            "tuning": dict(chosen),
            "pretrained": dict(pretrained),
        }
    )


def supersedes(assets: Path, name: str) -> list[str]:
    """Name the sibling methods whose predictions are committed, as the freeze records them.

    The siblings are the other methods of this package (``sibling_methods``, ADR 0004); both
    frozen before either is committed give ``[]``.

    Args:
        assets: The folder of the frozen methods.
        name: The method being frozen.

    Returns:
        The sorted names of the siblings whose ``predictions.csv`` is tracked by git and
        unchanged since ``HEAD``.
    """
    return sorted(
        other
        for other in sibling_methods(name, METHODS)
        if committed_unchanged(assets / other / PREDICTIONS_FILE)
    )


def _fetch(args: argparse.Namespace, variant: Variant) -> int:
    """Download the pinned snapshot: its configuration, tokenizer and encoder weights."""
    del args  # every option is the method's
    folder = snapshot_folder(variant.model_repo, variant.model_revision)
    snapshot_download(
        variant.model_repo,
        revision=variant.model_revision,
        local_dir=folder,
        allow_patterns=[*CONFIG_FILES, ENCODER_FILE],
    )
    weight_list = pretrained_weight_list(
        variant.model_repo, variant.model_revision, variant.pretrained_files
    )
    print(
        f"fetched {variant.model_repo}@{variant.model_revision} into {folder}; "
        f"{variant.name} pins {sorted(weight_list['files'])}"
    )
    return 0


def _tune(args: argparse.Namespace, variant: Variant) -> int:
    """Run the one-fold study until its trials are finished, printing a line per trial."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    with tempfile.TemporaryDirectory() as output_dir:
        setup = _setup(variant, None, Path(output_dir))
        study = run_study(
            setup,
            args.journal or _journal(variant),
            tuning_data(variant, read_dev(parts.dev, setup.tokenizer, variant)),
            callback=_print_trial,
        )
    print(
        f"best: trial {study.best_trial.number}, tuning-fold macro F1 {study.best_value:.4f}; "
        "run export next"
    )
    return 0


def _print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print a finished trial's state and value."""
    del study
    print(f"trial {trial.number}: {trial.state.name}, tuning-fold macro F1 {trial.value}")


def _export(args: argparse.Namespace, variant: Variant) -> int:
    """Freeze the study's results in the method's tuning/ folder."""
    study = read_study(variant, args.journal or _journal(variant))
    directory = args.assets / variant.name / TUNING_FOLDER
    written = export_study(study, variant, directory)
    print(
        f"exported trial {study.best_trial.number} (tuning-fold macro F1 "
        f"{study.best_value:.4f}) to {directory}: {', '.join(written)}"
    )
    return 0


def _train(args: argparse.Namespace, variant: Variant) -> int:
    """Train the fold models with the exported configuration, into the trained cache."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    chosen = _chosen(args.assets, variant)
    if args.root is None and (args.assets / variant.name / MODEL_FOLDER / TRAINED_FILE).exists():
        raise ValueError(
            f"{variant.name} is frozen: its fold models are the weights its {TRAINED_FILE} lists, "
            "restored by copying them. Training again makes other weights: pass --root to train "
            "them elsewhere, as a new method."
        )
    root = args.root or trained_root() / variant.name
    with tempfile.TemporaryDirectory() as output_dir:
        setup = _setup(variant, chosen, Path(output_dir))
        dev = read_dev(parts.dev, setup.tokenizer, variant)
        for line in train_folds(setup, chosen, dev, root):
            print(line)
    return 0


def _adopt(args: argparse.Namespace, variant: Variant) -> int:
    """Copy the fold models a notebook trained into the trained cache."""
    chosen = _chosen(args.assets, variant)
    source = args.source or EXPERIMENTS_DIR / variant.name / NOTEBOOK_FOLDERS[variant.lora]
    split = load_split(args.split)
    folds = sorted(int(fold) for fold in split.loc[split["part"] == "dev", "fold"].unique())
    for line in adopt_folds(variant, chosen, source, trained_root() / variant.name, folds):
        print(line)
    return 0


def _freeze(args: argparse.Namespace, variant: Variant) -> int:
    """Freeze the predictions, scores, run record and method folder once, then log to MLflow."""
    check_dataset(args.data)
    check_clean_tree(args.allow_dirty)
    parts = read_parts(args.data, args.split)
    chosen = _chosen(args.assets, variant)
    directory = args.assets / variant.name
    if (directory / PREDICTIONS_FILE).exists():
        _check_frozen(directory, variant, chosen, parts.split)
        print(
            f"{directory} is frozen under the code's recipe: nothing re-derived (verify "
            "re-scores it)"
        )
    else:
        _first_freeze(args, variant, chosen, parts)
    record = load_run_record(directory)
    model_id = log_method(
        directory / MODEL_FOLDER,
        record,
        part_metrics(parts.labels, load_predictions(directory), FOCUS_TAGS),
        split=parts.split,
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
    )
    log_run_extras(
        record.recipe_sha256,
        metrics={OUTSIDE_TUNING_FOLD_METRIC: record.method_entries["outside_tuning_fold_macro_f1"]},
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
    )
    print(f"logged {variant.name} to MLflow: model {model_id}")
    return 0


def _first_freeze(
    args: argparse.Namespace, variant: Variant, chosen: Mapping, parts: Parts
) -> None:
    """Derive dev from the fold models, save the folder, score test with it, and freeze."""
    directory = args.assets / variant.name
    dev = _dev_rows(variant, chosen, parts)
    pretrained = pretrained_weight_list(
        variant.model_repo, variant.model_revision, variant.pretrained_files
    )
    trained = trained_weight_list(
        variant.name, {name: f"{name}/{FOLD_WEIGHTS_FILE}" for name in dev.records}
    )
    folder = directory / MODEL_FOLDER
    save_method(
        FineTunedBag.build(
            variant, chosen["params"], dev.thresholds.to_numpy(), pretrained, trained
        ),
        folder,
    )
    _free_memory()  # the built bag is gone: the folder is loaded as the CLI loads it
    method = load_method(folder)
    print(_check_held_out(method, parts.dev, dev.scores))
    fold_test = method.fold_scores(parts.test)
    test_scores = method.frame(bag_mean(fold_test), parts.test)
    record = make_run_record(
        variant.name,
        recipe_sha256(variant, chosen, pretrained),
        allow_dirty=args.allow_dirty,
        requirements=METHODS[variant.name].requirements,
        method_entries=_method_entries(args.assets, variant, chosen, parts, dev),
        data_dir=args.data,
        split_path=args.split,
    )
    print(
        freeze_predictions(
            stack_parts(dev=dev.predictions, test=_predict(test_scores, method.thresholds_)),
            directory,
            record,
            scores=stack_parts(dev=dev.scores, test=test_scores),
            score_atol=METHODS[variant.name].score_atol,
        )
    )
    print(_save_fold_test_scores(directory, fold_test, method.tags, parts.test))


def _dev_rows(variant: Variant, chosen: Mapping, parts: Parts) -> _DevRows:
    """Read the fold models' out-of-fold scores, and predict dev with cross-fitted thresholds."""
    folds = parts.dev["fold"]
    records, scores = read_fold_models(variant, chosen, folds, trained_root() / variant.name)
    labels = parts.labels.loc[scores.index]
    _, predictions = cross_fitted_predictions(scores, labels, folds)
    return _DevRows(records, scores, predictions, per_tag_thresholds(scores, labels))


def _method_entries(
    assets: Path, variant: Variant, chosen: Mapping, parts: Parts, dev: _DevRows
) -> dict:
    """Return what a ModernBERT method adds to its run record, as plain JSON values."""
    device = resolve_device(variant.device)
    return {
        "settings": settings(variant),
        "supersedes": supersedes(assets, variant.name),
        "dev_in_sample": False,
        "dev_estimate": DEV_ESTIMATE,
        "tuning": {"trial": chosen["trial"], "tuning_macro_f1": chosen["tuning_macro_f1"]},
        "params": dict(chosen["params"]),
        "device": _device_name(device),
        "dtype": "fp16-autocast" if device.type == "cuda" else "float32",
        "fold_models": {name: _fold_entry(variant, record) for name, record in dev.records.items()},
        "outside_tuning_fold_macro_f1": _outside_tuning_fold_f1(variant, parts, dev.predictions),
    }


def _fold_entry(variant: Variant, record: Mapping) -> dict:
    """Summarise a fold model's training.json for the run record."""
    return {
        "seed": record["seed"],
        # Seeds are tried in order, so the ones before the seed that trained it collapsed.
        "collapsed_seeds": list(range(variant.seed, record["seed"])),
        "train_macro_ap": record["train_macro_ap"],
        "minutes": record["minutes"],
        "adopted_from": record.get("adopted_from"),
    }


def _outside_tuning_fold_f1(variant: Variant, parts: Parts, predictions: pd.DataFrame) -> float:
    """Score the dev folds that the study did not score: the configuration was chosen on none."""
    outside = parts.dev.index[parts.dev["fold"] != variant.tuning_fold]
    return evaluate(parts.labels.loc[outside], predictions.loc[outside]).macro_f1


def _device_name(device: torch.device) -> str:
    """Name the device that scores, with the GPU's model on CUDA."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def _predict(scores: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Predict each tag whose score reaches its threshold, as ``FineTunedBag.predict`` does."""
    return (scores >= thresholds).astype(int)


def _save_fold_test_scores(
    directory: Path,
    fold_scores: Mapping[str, np.ndarray],
    tags: tuple[str, ...],
    raw_test: pd.DataFrame,
) -> str:
    """Write each fold model's test scores, float64, to analysis/, for the notebooks."""
    frame = pd.concat(
        {
            name: pd.DataFrame(scores.astype(np.float64), index=raw_test.index, columns=list(tags))
            for name, scores in fold_scores.items()
        },
        names=["fold_model"],
    )
    path = directory / ANALYSIS_FOLDER / FOLD_TEST_SCORES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return f"saved each fold model's test scores to {path}"


def _check_frozen(directory: Path, variant: Variant, chosen: Mapping, split: pd.DataFrame) -> None:
    """Fail unless the frozen files are complete, consistent, and hold the code's recipe."""
    check_frozen_method(variant.name, assets_dir=directory.parent, split=split)
    pretrained = _read_json(directory / MODEL_FOLDER / PRETRAINED_FILE)
    stored = load_run_record(directory).recipe_sha256
    if stored != recipe_sha256(variant, chosen, pretrained):
        raise ValueError(
            f"{directory} was frozen with another recipe ({stored}): the settings of "
            f"{variant.name}, its {PARAMS_FILE} or its pretrained files changed since."
        )


def _verify(args: argparse.Namespace, variant: Variant) -> int:
    """Check the frozen files against the dataset, the code and the caches; report the scores."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    chosen = _chosen(args.assets, variant)
    directory = args.assets / variant.name
    _check_frozen(directory, variant, chosen, parts.split)
    print(f"the frozen files of {variant.name} are complete, consistent, under the code's recipe")
    frozen, scores = load_predictions(directory), load_scores(directory)
    thresholds = _check_dev_rows(parts, frozen.loc["dev"], scores.loc["dev"])
    print("re-derived the dev predictions and the thresholds from scores.csv: they match")
    method = load(directory / MODEL_FOLDER, device="cpu")  # never the GPU: see the docstring
    print(_check_held_out(method, parts.dev, scores.loc["dev"]))
    print(_check_test_rows(method, parts.test, frozen.loc["test"], scores.loc["test"], thresholds))
    del method
    _free_memory()
    for dataset, metrics in part_metrics(parts.labels, frozen, FOCUS_TAGS).items():
        print(f"{dataset}: macro F1 {metrics['macro_f1']:.4f}, micro F1 {metrics['micro_f1']:.4f}")
    if args.retrain:
        print(_retrain(variant, chosen, parts, directory))
    return 0


def _check_dev_rows(parts: Parts, frozen: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """Re-derive the dev predictions from the frozen scores, and return the bag's thresholds."""
    labels = parts.labels.loc[scores.index]
    _, predictions = cross_fitted_predictions(scores, labels, parts.dev["fold"].loc[scores.index])
    if not predictions.equals(frozen):
        raise ValueError(
            "The dev predictions re-derived from the frozen scores differ from "
            f"{PREDICTIONS_FILE}: the thresholds' code, the data or the split changed since the "
            "freeze."
        )
    return per_tag_thresholds(scores, labels)


def _check_held_out(method: FineTunedBag, raw_dev: pd.DataFrame, scores: pd.DataFrame) -> str:
    """Re-score each dev fold with the fold model that left it out, against its frozen scores.

    This ties the loaded bag to the out-of-fold scores its fold models gave after training (and
    so to the dev rows): the same fold model on the same device and batches gives them back
    exactly, and another device or batching within ``score_atol``.

    Args:
        method: The loaded method.
        raw_dev: The raw dev problems, with their fold.
        scores: Their out-of-fold scores, frozen or read from the fold models' folders.

    Returns:
        A line to print.

    Raises:
        ValueError: If a score moves by more than ``score_atol``.
    """
    atol = max(METHODS[method.name].score_atol, 1e-9)
    gap = 0.0
    for fold in sorted(raw_dev["fold"].unique()):
        rows = raw_dev.index[raw_dev["fold"] == fold]
        name = fold_name(fold)
        again = method.fold_scores(raw_dev.loc[rows], [name])[name]
        gap = max(gap, float(np.abs(again - scores.loc[rows].to_numpy()).max()))
    if gap > atol:
        raise ValueError(
            f"A fold model of the method folder moves an out-of-fold score by {gap:.2e}, more "
            f"than {atol}: the folder's weights are not those that scored dev."
        )
    return f"each fold model gives back its out-of-fold scores: largest gap {gap:.2e}"


def _check_test_rows(
    method: FineTunedBag,
    raw_test: pd.DataFrame,
    frozen: pd.DataFrame,
    scores: pd.DataFrame,
    thresholds: pd.Series,
) -> str:
    """Check the folder's thresholds and test predictions, then re-score test within score_atol.

    A re-scored problem within ``score_atol`` of its frozen score can flip a tag only where the
    frozen score lies within ``score_atol`` of the threshold; the flips are counted.
    """
    if not np.array_equal(method.thresholds_, thresholds.to_numpy()):
        raise ValueError("The method folder's thresholds differ from those of the frozen scores.")
    if not _predict(scores, method.thresholds_).equals(frozen):
        raise ValueError("The frozen test predictions are not the frozen test scores thresholded.")
    atol = max(METHODS[method.name].score_atol, 1e-9)
    rescored = method.scores(raw_test)
    gap = float((rescored - scores).abs().to_numpy().max())
    if gap > atol:
        raise ValueError(
            f"Re-scoring test moves a score by {gap:.2e}, more than {atol}: the weights, the code "
            "or the environment changed since the freeze."
        )
    flips = int((_predict(rescored, method.thresholds_) != frozen).to_numpy().sum())
    return (
        f"re-scored {len(raw_test):,} test problems with the bag: largest gap {gap:.2e} "
        f"(score_atol {atol}), {flips} predictions flipped"
    )


def _retrain(variant: Variant, chosen: Mapping, parts: Parts, directory: Path) -> str:
    """Train the fold models again, each from its frozen seed, and check them against the frozen."""
    frozen = (load_predictions(directory).loc["dev"], load_scores(directory).loc["dev"])
    fold_models = load_run_record(directory).method_entries["fold_models"]
    root = trained_root() / RETRAIN_FOLDER / variant.name
    with tempfile.TemporaryDirectory() as output_dir:
        setup = _setup(variant, chosen, Path(output_dir))
        dev = read_dev(parts.dev, setup.tokenizer, variant)
        seeds = {name: entry["seed"] for name, entry in fold_models.items()}
        for line in train_folds(setup, chosen, dev, root, seeds):
            print(line)
    _, retrained = read_fold_models(variant, chosen, dev.folds, root)
    print(_tuning_objectives(variant, chosen, dev, retrained, frozen[1]))
    return compare_retrained(variant, parts, frozen, retrained)


def _tuning_objectives(
    variant: Variant, chosen: Mapping, dev: DevData, retrained: pd.DataFrame, frozen: pd.DataFrame
) -> str:
    """Score the tuning fold as the study did, with the retrained and the frozen fold model.

    The fold model that leaves out the tuning fold trains on the chosen trial's problems, with its
    configuration and seed: retraining it runs the chosen trial's training again.
    """
    data = tuning_data(variant, dev)
    rows = data.held_out_labels.index
    again = held_out_metrics(retrained.loc[rows].to_numpy(), data)["macro_f1"]
    before = held_out_metrics(frozen.loc[rows].to_numpy(), data)["macro_f1"]
    return (
        f"tuning-fold macro F1 of {fold_name(variant.tuning_fold)}: retrained {again:.4f}, frozen "
        f"{before:.4f}; the study's trial {chosen['trial']}: {chosen['tuning_macro_f1']:.4f} (the "
        "same configuration, seed and problems; GPU training is not bit-reproducible)"
    )


def compare_retrained(
    variant: Variant,
    parts: Parts,
    frozen: tuple[pd.DataFrame, pd.DataFrame],
    retrained: pd.DataFrame,
) -> str:
    """Check retrained out-of-fold scores against the frozen ones, on the declared bounds.

    GPU training is not bit-reproducible, so a tolerance per score cannot hold. The bounds of
    the settings (``retrain_*``, docs/adr/0005) were declared before any retraining, so that two
    different models of the same macro F1 fail them.

    Args:
        variant: The method's settings, with the bounds.
        parts: The dataset.
        frozen: The frozen dev predictions and scores.
        retrained: The retrained fold models' out-of-fold scores, indexed like them.

    Returns:
        One line per bound, each passed.

    Raises:
        ValueError: If a bound fails; the message holds every line.
    """
    stats = _retrain_statistics(parts, frozen, retrained)
    margin = variant.retrain_f1_margin
    checks = {
        f"lowest correlation of a fold model's tag {stats['correlation']:.3f}, at least "
        f"{variant.retrain_min_correlation}": stats["correlation"]
        >= variant.retrain_min_correlation,
        f"median score gap {stats['median_gap']:.4f}, at most {variant.retrain_max_median_gap}": (
            stats["median_gap"] <= variant.retrain_max_median_gap
        ),
        f"flipped predictions {stats['flips']:.2%}, at most {variant.retrain_max_flips:.1%}": (
            stats["flips"] <= variant.retrain_max_flips
        ),
        f"dev macro F1 {stats['candidate']:.4f} retrained, {stats['baseline']:.4f} frozen: "
        f"difference {stats['delta']:+.4f} [{stats['delta_low']:+.4f}, "
        f"{stats['delta_high']:+.4f}], within +/-{margin}": -margin <= stats["delta_low"]
        and stats["delta_high"] <= margin,
    }
    lines = "\n".join(f"{'pass' if ok else 'FAIL'}: {text}" for text, ok in checks.items())
    if not all(checks.values()):
        raise ValueError(f"The retrained fold models fail a declared bound:\n{lines}")
    return lines


def _retrain_statistics(
    parts: Parts, frozen: tuple[pd.DataFrame, pd.DataFrame], retrained: pd.DataFrame
) -> dict[str, float]:
    """Measure how retrained out-of-fold scores and predictions differ from the frozen ones."""
    predictions, scores = frozen
    folds = parts.dev["fold"].loc[scores.index]
    labels = parts.labels.loc[scores.index]
    again = cross_fitted_predictions(retrained, labels, folds)[1]
    correlations = [
        retrained[folds == fold].corrwith(scores[folds == fold]) for fold in folds.unique()
    ]
    row = compare(labels, predictions, again, parts.dev["group"].loc[scores.index]).loc["macro_f1"]
    return {
        "correlation": float(np.min(correlations)),  # NaN, so failing, for a constant fold model
        "median_gap": float(np.median((retrained - scores).abs().to_numpy())),
        "flips": float((again != predictions).to_numpy().mean()),
        **{
            key: float(row[key])
            for key in ("baseline", "candidate", "delta", "delta_low", "delta_high")
        },
    }


def _latency_cpu(args: argparse.Namespace, variant: Variant) -> int:
    """Time the method folder on the CPU, as tracking latency does, once."""
    directory = args.assets / variant.name / ANALYSIS_FOLDER / CPU_LATENCY_FOLDER
    if (directory / LATENCY_FILE).exists():
        raise FileExistsError(
            f"{directory / LATENCY_FILE} exists: the committed timings stay the record."
        )
    check_dataset(args.data)
    dev = select_part(read_samples(args.data), load_split(args.split), "dev")
    problems = dev.sample(n=LATENCY_PROBLEMS, random_state=LATENCY_SEED).sort_index()
    problems = problems[list(METHODS[variant.name].input_columns)]
    start = perf_counter()
    method = load(args.assets / variant.name / MODEL_FOLDER, device="cpu")
    load_s = perf_counter() - start
    timings = Latency(time_single(method, problems), time_bulk(method, problems), load_s)
    directory.mkdir(parents=True, exist_ok=True)
    save_latency(directory, timings)
    summary = latency_summary(timings.seconds)
    print(
        f"saved {directory / LATENCY_FILE}: {len(problems)} problems on the CPU "
        f"({torch.get_num_threads()} threads), median {summary['median_s']:.2f} s, max "
        f"{summary['max_s']:.2f} s; one call {timings.bulk_s:.2f} s; load {load_s:.2f} s"
    )
    return 0


def _chosen(assets: Path, variant: Variant) -> dict:
    """Read the configuration the study chose, ``tuning/params.json``."""
    path = assets / variant.name / TUNING_FOLDER / PARAMS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run export --method {variant.name} first.")
    return _read_json(path)


def _setup(variant: Variant, chosen: Mapping | None, output_dir: Path) -> TrainingSetup:
    """Gather what the chosen configuration (none for a study) trains with."""
    snapshot = _training_snapshot(variant)
    return TrainingSetup(
        variant=variant,
        params=None if chosen is None else params_from(variant, chosen["params"]),
        snapshot=snapshot,
        tokenizer=load_tokenizer(snapshot),
        device=resolve_device(variant.device),
        output_dir=output_dir,
    )


def _training_snapshot(variant: Variant) -> Path:
    """Return the fetched snapshot's folder, after checking it holds what training reads."""
    pretrained_weight_list(
        variant.model_repo, variant.model_revision, (*CONFIG_FILES, ENCODER_FILE)
    )
    return snapshot_folder(variant.model_repo, variant.model_revision)


def _journal(variant: Variant) -> Path:
    """Return the study's journal, experiments/<name>/optuna.jsonl."""
    return EXPERIMENTS_DIR / variant.name / JOURNAL_FILE


def _free_memory() -> None:
    """Release a model just dropped, so that the next one fits on the GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
