"""Freeze the tuned tag model as the tagging method ``tfidf``, and verify its frozen files.

    python -m codeforces_tags.tuning freeze [--allow-dirty]
    python -m codeforces_tags.tuning verify

Neither re-runs the study: both read its frozen results (``assets/tfidf_baseline/``, the
``--results`` option), the configuration and thresholds it chose.

``freeze`` fits the tag model on the dev set (``fit_tag_model``: each problem scored out of fold,
the thresholds chosen on those scores, then the pipeline refit on every dev problem) and, before
writing anything, checks that it reproduces the study: every row of ``out_of_fold.csv`` exactly
and the thresholds within ``TUNING_EXPORT_TOLERANCE`` of ``thresholds.json``; then, the study
reproduced, the test macro F1 equal to ``TFIDF_TEST_MACRO_F1``, the frozen configuration's, which
it does not print otherwise. It then writes ``assets/methods/tfidf/`` once: ``predictions.csv``
and ``scores.csv`` (dev rows out of fold, never the refit's in-sample ones; test rows from the
refit, scored once), ``run_record.json`` and the method folder ``model/``, which must predict
every dev and test problem as the fitted tag model does, and score them within
``max(score_atol, TUNING_EXPORT_TOLERANCE)`` of it; then it logs the method to MLflow
(``log_method``). A second run rewrites nothing: it derives everything again, checks it against
the frozen files, and mirrors the frozen record to MLflow. Before it fits anything, it refuses a
method folder left without the frozen files beside it, which may come from another fit, and
another recipe than the frozen run record's: another configuration is a new model
(docs/adr/0004), which ``freeze`` does not freeze as ``tfidf``.

``verify``, offline and writing nothing, checks the frozen files at the two levels of the
checklist (CLAUDE.md, Tagging methods, step 7): (a) without fitting, from ``scores.csv``,
``predictions.csv`` and the method folder; (b) by deriving everything again, as ``freeze`` does.

This module imports mlflow only through ``tracking.methods``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from codeforces_tags.config import (
    FOCUS_TAGS,
    TFIDF_BASELINE_DIR,
    TFIDF_TEST_MACRO_F1,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods import METHOD_FILE, METHODS, MODEL_FOLDER, load_method, save_method
from codeforces_tags.methods.parts import Parts, read_parts
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.tag_model import NAME, TagModel
from codeforces_tags.model.thresholds import cross_fitted_predictions, per_tag_thresholds
from codeforces_tags.model.training import fit_tag_model, out_of_fold
from codeforces_tags.provenance import (
    check_clean_tree,
    check_dataset,
    make_run_record,
    settings_snapshot,
    sha256_json,
)
from codeforces_tags.tracking.methods import log_method, part_metrics
from codeforces_tags.tuning.results import (
    OUT_OF_FOLD_FILE,
    PARAMS_FILE,
    THRESHOLDS_FILE,
    TuningResults,
    load_tuning_results,
)


class _Study(NamedTuple):
    """The frozen study: its folder and its results."""

    directory: Path
    results: TuningResults


class _Rows(NamedTuple):
    """A method's 0/1 predictions and scores of the same problems."""

    predictions: pd.DataFrame
    scores: pd.DataFrame


class _Derived(NamedTuple):
    """What freeze derives: the fitted tag model, and the rows it freezes."""

    tag_model: TagModel
    rows: _Rows  # indexed by (part, sample_id): dev out of fold, test from the refit


def recipe_sha256(results_dir: Path = TFIDF_BASELINE_DIR) -> str:
    """Hash what defines the method: the study's choices, the recorded settings, the focus tags.

    The study's choices are ``params.json`` and ``thresholds.json``, as read: committed inputs,
    never floats derived again (docs/adr/0004).

    Args:
        results_dir: The folder of the frozen study.

    Returns:
        The sha256 hex digest, the ``recipe_sha256`` of the run record.
    """
    return sha256_json(
        {
            "params": _read_json(results_dir / PARAMS_FILE),
            "settings": settings_snapshot(),
            "thresholds": _read_json(results_dir / THRESHOLDS_FILE),
            "tags": list(FOCUS_TAGS),
        }
    )


def method_entries(
    tag_model: TagModel, results_dir: Path = TFIDF_BASELINE_DIR
) -> dict[str, object]:
    """Return what the run record of ``tfidf`` adds: the study's choices and the fit's iterations.

    Args:
        tag_model: The tag model fitted on the dev set.
        results_dir: The folder of the frozen study.

    Returns:
        ``params`` (``params.json`` as read); ``thresholds``, the tag model's ``{tag: float}``;
        ``max_n_iter``, the most iterations any tag's linear model of the refit ran (below the
        record's ``MODEL_MAX_ITER`` setting, every one converged); ``dev_in_sample`` False, as
        the study chose on out-of-fold scores (docs/adr/0001); and ``supersedes`` ``[]``, as
        ``tfidf`` is in no method package (docs/adr/0004).
    """
    classifier = tag_model.pipeline.named_steps["classifier"]
    return {
        "params": _read_json(results_dir / PARAMS_FILE),
        "thresholds": {tag: float(value) for tag, value in tag_model.thresholds.items()},
        "max_n_iter": max(int(np.max(model.n_iter_)) for model in classifier.estimators_),
        "dev_in_sample": False,
        "supersedes": [],
    }


def freeze(args: argparse.Namespace) -> int:
    """Freeze the tag model's predictions, scores, run record and method folder; log it.

    Args:
        args: The parsed arguments: ``data``, ``split``, ``results``, ``assets``,
            ``allow_dirty``, ``tracking_uri`` and ``artifact_dir``.

    Returns:
        The exit status, 0.

    Raises:
        ValueError: If the dataset is refused; before any fit, if a method folder has no
            frozen predictions beside it or the frozen files hold another recipe; if the fitted
            tag model does not reproduce the frozen study or its test macro F1 (nothing is then
            written or logged); or if the frozen files or the method folder hold other
            predictions or other scores.
        RuntimeError: If the code has uncommitted changes and ``allow_dirty`` is False.
    """
    check_dataset(args.data)
    check_clean_tree(args.allow_dirty)
    directory = args.assets / NAME
    _check_folder_is_frozen(directory)
    if (directory / RUN_RECORD_FILE).exists():
        # Before the fit and its test score: another configuration is another recipe.
        _check_recipe(directory, args.results)
    parts = read_parts(args.data, args.split)
    study = _Study(args.results, load_tuning_results(args.results))
    derived = _derive(parts, study.results.params)
    what = "the fitted tag model"
    print(_check_study(study, parts, derived.rows.predictions, derived.tag_model.thresholds, what))
    record = make_run_record(
        NAME,
        recipe_sha256(args.results),
        allow_dirty=args.allow_dirty,
        requirements=METHODS[NAME].requirements,
        method_entries=method_entries(derived.tag_model, args.results),
        data_dir=args.data,
        split_path=args.split,
    )
    print(
        freeze_predictions(
            derived.rows.predictions,
            directory,
            record,
            scores=derived.rows.scores,
            score_atol=METHODS[NAME].score_atol,
        )
    )
    folder = directory / MODEL_FOLDER
    print(_freeze_folder(derived.tag_model, folder, parts))
    store = {"tracking_uri": args.tracking_uri, "artifact_dir": args.artifact_dir}
    # What is frozen, not the record made now: a second run's has another created_at.
    frozen_record = load_run_record(directory)
    metrics = part_metrics(parts.labels, derived.rows.predictions, FOCUS_TAGS)
    model_id = log_method(folder, frozen_record, metrics, split=parts.split, **store)
    print(f"logged {NAME} to MLflow: model {model_id}")
    return 0


def verify(args: argparse.Namespace) -> int:
    """Check the frozen files of ``tfidf`` at both levels, offline, and report their scores.

    Args:
        args: The parsed arguments: ``data``, ``split``, ``results`` and ``assets``.

    Returns:
        The exit status, 0.

    Raises:
        ConformanceError: If a frozen file is missing, malformed or inconsistent with the split.
        ValueError: If the dataset is refused, if the files hold another recipe, or if a check
            of level (a) or (b) fails; the message names it.
    """
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    directory = args.assets / NAME
    check_frozen_method(NAME, assets_dir=args.assets, split=parts.split)
    print(f"the frozen files of {NAME} are complete and consistent with the split")
    _check_recipe(directory, args.results)
    print(f"they hold the recipe of {args.results}")
    study = _Study(args.results, load_tuning_results(args.results))
    frozen = _Rows(load_predictions(directory), load_scores(directory))
    _verify_from_files(directory, study, parts, frozen)
    _verify_by_fitting(directory, study, parts, frozen)
    for dataset, metrics in part_metrics(parts.labels, frozen.predictions, FOCUS_TAGS).items():
        print(f"{dataset}: macro F1 {metrics['macro_f1']!r}, micro F1 {metrics['micro_f1']!r}")
    return 0


def _read_json(path: Path) -> dict:
    """Read a JSON file of the frozen study."""
    return json.loads(path.read_text(encoding="utf-8"))


def _derive(parts: Parts, params: ModelParams) -> _Derived:
    """Fit the tag model on dev, and take its dev rows out of fold and its test rows."""
    folds, dev_labels = parts.dev["fold"], parts.labels.loc[parts.dev.index]
    tag_model = fit_tag_model(params, parts.dev, dev_labels, folds)
    dev = out_of_fold(tag_model.estimator, dev_labels, folds)
    rows = _Rows(
        predictions=stack_parts(dev=dev.predictions, test=tag_model.predict(parts.test)),
        scores=stack_parts(dev=dev.scores, test=tag_model.scores(parts.test)),
    )
    return _Derived(tag_model, rows)


def _check_study(
    study: _Study, parts: Parts, predictions: pd.DataFrame, thresholds: pd.Series, what: str
) -> str:
    """Fail unless predictions and thresholds reproduce the frozen study, then its test score.

    The test macro F1 is compared only once the study is reproduced, and never printed: a
    configuration whose test score differs from ``TFIDF_TEST_MACRO_F1`` is not frozen, so its
    test score is not read (docs/adr/0001).

    Args:
        study: The frozen study.
        parts: The dataset.
        predictions: Dev (out of fold) and test predictions, indexed by ``(part, sample_id)``.
        thresholds: The final per-tag thresholds, chosen on the dev scores.
        what: What gave them, for the messages.

    Returns:
        The line to print.

    Raises:
        ValueError: Naming every row of ``out_of_fold.csv`` that the dev predictions do not
            score exactly and every threshold further than ``TUNING_EXPORT_TOLERANCE`` from
            ``thresholds.json``; or, these matching, if the test macro F1 is not
            ``TFIDF_TEST_MACRO_F1``.
    """
    dev, test = predictions.loc["dev"], predictions.loc["test"]
    scored = evaluate(parts.labels.loc[dev.index], dev).overall_f1()
    differences = [
        *_row_differences(study.results.out_of_fold["f1"], scored),
        *_threshold_differences(study.results.thresholds, thresholds),
    ]
    if differences:
        raise ValueError(
            f"The frozen study in {study.directory} is not reproduced by {what}: "
            f"{'; '.join(differences)}. The code, data, split or frozen files differ from the "
            "study's; nothing was written or logged."
        )
    if evaluate(parts.labels.loc[test.index], test).macro_f1 != TFIDF_TEST_MACRO_F1:
        raise ValueError(
            f"The frozen study in {study.directory} is reproduced by {what}, whose test "
            f"predictions do not score TFIDF_TEST_MACRO_F1 = {TFIDF_TEST_MACRO_F1!r}, the test "
            f"macro F1 of the configuration frozen as {NAME}: either the code changed the "
            f"refit, or {study.directory} holds another configuration, a new model that "
            f"freeze does not freeze as {NAME} (docs/adr/0004). Their macro F1 is not printed: "
            "a configuration's test score is read once it is frozen (docs/adr/0001). Nothing "
            "was written or logged."
        )
    return (
        f"the frozen study in {study.directory} is reproduced by {what}: every row of "
        f"{OUT_OF_FOLD_FILE} (out-of-fold macro F1 {float(scored['macro_f1'])!r}), the thresholds "
        f"within {TUNING_EXPORT_TOLERANCE} of {THRESHOLDS_FILE}, and the test macro F1 "
        f"(TFIDF_TEST_MACRO_F1 = {TFIDF_TEST_MACRO_F1!r})"
    )


def _row_differences(frozen: pd.Series, scored: pd.Series) -> list[str]:
    """Name each row of ``out_of_fold.csv`` that the scored F1 does not equal exactly."""
    return [
        f"{metric} is {float(scored[metric])!r} out of fold, not {float(value)!r} "
        f"({OUT_OF_FOLD_FILE})"
        for metric, value in frozen.items()
        if scored[metric] != value
    ]


def _threshold_differences(frozen: pd.Series, thresholds: pd.Series) -> list[str]:
    """Name each tag whose threshold is further than the tolerance from ``thresholds.json``."""
    return [
        f"the threshold of {tag} is {float(thresholds[tag])!r}, further than "
        f"{TUNING_EXPORT_TOLERANCE} from {float(value)!r} ({THRESHOLDS_FILE})"
        for tag, value in frozen.items()
        if not np.isclose(thresholds[tag], value, rtol=0, atol=TUNING_EXPORT_TOLERANCE)
    ]


def _check_rows(what: str, rows: str, derived: _Rows, frozen: _Rows, directory: Path) -> str:
    """Fail unless ``what`` predicts the frozen rows exactly and scores them within tolerance.

    The tolerance of scores is ``_score_atol()``.

    Returns:
        The line to print.
    """
    atol = _score_atol()
    if not derived.predictions.equals(frozen.predictions):
        raise ValueError(
            f"On the {rows}, {what} predicts other tags than {PREDICTIONS_FILE} in {directory}: "
            "the code, data or environment changed since the freeze, or the files come from "
            "another fit."
        )
    if not np.allclose(derived.scores.to_numpy(), frozen.scores.to_numpy(), rtol=0, atol=atol):
        raise ValueError(
            f"On the {rows}, {what} scores further than {atol} (rtol 0) from {SCORES_FILE} in "
            f"{directory}: the code, data or environment changed since the freeze, or the files "
            "come from another fit."
        )
    return (
        f"{what} scores the {len(frozen.predictions):,} {rows} within {atol} of {SCORES_FILE} "
        f"and predicts them as {PREDICTIONS_FILE}"
    )


def _score_atol() -> float:
    """Return how far scores may move: ``max(score_atol, TUNING_EXPORT_TOLERANCE)``, rtol 0.

    Another machine's BLAS moves the scores by rounding errors only.
    """
    return max(METHODS[NAME].score_atol, TUNING_EXPORT_TOLERANCE)


def _check_folder(folder: Path, tag_model: TagModel, parts: Parts) -> str:
    """Fail unless the method folder predicts and scores every problem as the tag model does.

    Returns:
        What it checked, to end a line to print.
    """
    loaded, atol = load_method(folder), _score_atol()
    for part, problems in (("dev", parts.dev), ("test", parts.test)):
        if not loaded.predict(problems).equals(tag_model.predict(problems)):
            raise ValueError(
                f"The method folder {folder} predicts other tags for the {part} problems than "
                "the fitted tag model: the folder and the frozen files come from different fits."
            )
        scores, expected = loaded.scores(problems).to_numpy(), tag_model.scores(problems)
        if not np.allclose(scores, expected.to_numpy(), rtol=0, atol=atol):
            raise ValueError(
                f"The method folder {folder} scores the {part} problems further than {atol} "
                "(rtol 0) from the fitted tag model: the folder and the frozen files come from "
                "different fits."
            )
    return (
        f"predicts every dev and test problem as the fitted tag model does, and scores them "
        f"within {atol} of it"
    )


def _check_folder_is_frozen(directory: Path) -> None:
    """Fail if the method folder of ``directory`` has no frozen predictions beside it."""
    folder = directory / MODEL_FOLDER
    if (folder / METHOD_FILE).exists() and not (directory / PREDICTIONS_FILE).exists():
        raise ValueError(
            f"{directory} holds the method folder {MODEL_FOLDER}/ but no {PREDICTIONS_FILE}: "
            "the folder may come from another fit, and freeze saves it only with the frozen "
            f"files. Remove {folder} to freeze again; nothing was written or logged."
        )


def _freeze_folder(tag_model: TagModel, folder: Path, parts: Parts) -> str:
    """Save the method folder once; then check that it predicts and scores as the tag model."""
    saved = not (folder / METHOD_FILE).exists()
    if saved:
        save_method(tag_model, folder)
    checked = _check_folder(folder, tag_model, parts)
    return f"{'saved' if saved else 'checked'} the method folder {folder}: it {checked}"


def _check_recipe(directory: Path, results_dir: Path) -> None:
    """Fail unless the frozen run record holds the recipe of the study and the code."""
    stored = load_run_record(directory).recipe_sha256
    expected = recipe_sha256(results_dir)
    if stored != expected:
        raise ValueError(
            f"{directory} was frozen with another recipe ({stored}, not {expected}): "
            f"{PARAMS_FILE} or {THRESHOLDS_FILE} in {results_dir}, the recorded settings or the "
            "focus tags changed since. Frozen predictions are never overwritten: another "
            f"configuration is a new model (docs/adr/0004), not a new freeze of {NAME}."
        )


def _verify_from_files(directory: Path, study: _Study, parts: Parts, frozen: _Rows) -> None:
    """Level (a): check the frozen files against each other and the method folder, unfitted."""
    dev_scores = frozen.scores.loc["dev"]
    labels = parts.labels.loc[dev_scores.index]
    _, dev_predictions = cross_fitted_predictions(
        dev_scores, labels, parts.dev["fold"].loc[dev_scores.index]
    )
    if not dev_predictions.equals(frozen.predictions.loc["dev"]):
        raise ValueError(
            f"The thresholds cross-fitted on the dev rows of {SCORES_FILE} predict other tags "
            f"than the dev rows of {PREDICTIONS_FILE} in {directory}: the two files come from "
            "different fits."
        )
    print(
        f"the thresholds cross-fitted on the dev rows of {SCORES_FILE} give the dev rows of "
        f"{PREDICTIONS_FILE}"
    )
    thresholds = per_tag_thresholds(dev_scores, labels)
    files = f"{SCORES_FILE} and {PREDICTIONS_FILE}"
    print(_check_study(study, parts, frozen.predictions, thresholds, files))
    method = load_method(directory / MODEL_FOLDER)
    on_test = _Rows(method.predict(parts.test), method.scores(parts.test))
    test_rows = _Rows(frozen.predictions.loc["test"], frozen.scores.loc["test"])
    what = f"the method folder {directory / MODEL_FOLDER}"
    print(_check_rows(what, "test problems", on_test, test_rows, directory))


def _verify_by_fitting(directory: Path, study: _Study, parts: Parts, frozen: _Rows) -> None:
    """Level (b): derive everything again as freeze does, and compare it with the frozen files."""
    derived = _derive(parts, study.results.params)
    what = "the fitted tag model"
    print(_check_study(study, parts, derived.rows.predictions, derived.tag_model.thresholds, what))
    print(_check_rows(what, "dev and test problems", derived.rows, frozen, directory))
    folder = directory / MODEL_FOLDER
    print(f"the method folder {folder} {_check_folder(folder, derived.tag_model, parts)}")
