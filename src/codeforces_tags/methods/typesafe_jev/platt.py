"""The dev-only record of a calibrated twin's threshold rule (docs/adr/0011).

A calibrated twin (``settings.PLATT_BASES``) reads its base's committed answers and turns the
same probabilities into 0/1 by another threshold rule, Platt + Lipton (R2,
``model.platt_lipton``), declared with R0 and R1 before this record was computed.
``export_platt`` writes the record once in the twin's ``tuning/`` folder, before the twin is
frozen, from the dev part only, out of fold:

- ``rules.csv``: three threshold rules, each cross-fitted over the dev folds: R0 (the F1
  argmax, the base's own rule), R1 (Platt, then a calibrated probability of at least 0.5) and
  R2. For each, the macro F1, the micro F1 and each focus tag's F1 with group-bootstrap
  intervals, and its paired difference from R0 (``analysis.comparison.method_scores``);
- ``nested.csv``: for each outer fold, the rule that the cross-fitted macro F1 of the other
  folds chooses, its F1 on the outer fold once fitted on those folds, and the nested
  out-of-fold macro F1 of these choices;
- ``stability.csv``: for R0 and R2, each tag's fold thresholds, their spread, the final
  threshold and the number of dev problems whose 0/1 depends on which folds chose it;
- ``params.json``: the rule and its settings, the base, the sha256 and commit of each base file
  read, R2's final thresholds and the nested choices.

Each rule's fold thresholds are ``model.platt_lipton.cross_fitted_rows`` of its per-tag choice,
and each fold is predicted with its own row by ``apply_fold_thresholds``, as the twin's
``PlattLiptonThresholds`` does.

The nested choice is reported, never acted on: the twin's rule is R2 whatever it says. Nothing
here reads a test row: the caller passes the dev rows of the base's scores, and a row without
a dev fold is refused (``check_dev_folds``).

The twin's freeze and verify check its inputs against that record (``check_base_files``,
``check_thresholds``), record its base, rule and disclosure in its run record
(``twin_entries``) and take its base's ``latency.csv`` and ``per_problem.csv`` as its own
(``BASE_COPIES``, ``copy_from_base``, ``check_copy``): a twin sends its base's requests. It
imports neither gepa nor mlflow.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import expit  # pylint: disable=no-name-in-module  # a ufunc, built in C

from codeforces_tags.analysis.comparison import method_scores
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.predictions import PER_PROBLEM_FILE, RUN_RECORD_FILE, SCORES_FILE
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.evaluation.timing import LATENCY_FILE
from codeforces_tags.llm.store import answers_file
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev.search_record import write_once
from codeforces_tags.methods.typesafe_jev.settings import (
    PARAMS_FILE,
    PLATT_BASES,
    PLATT_FILES,
    TEST_INFORMED,
)
from codeforces_tags.model.platt_lipton import (
    RULE_SETTINGS,
    apply_fold_thresholds,
    cross_fitted_rows,
    lipton_threshold,
    platt_fit,
    snap_to_grid,
)
from codeforces_tags.model.thresholds import best_threshold, check_dev_folds
from codeforces_tags.provenance import json_safe
from codeforces_tags.splitting import PARTS

# The threshold rules the record compares, in the order of its tables: R0 the F1 argmax (the
# base's rule), R1 Platt then a calibrated probability of at least 0.5, R2 Platt + Lipton.
RULES: tuple[str, ...] = ("R0", "R1", "R2")
# The twin's rule, declared before the record was computed (docs/adr/0011).
TWIN_RULE = "R2"
# The rules whose fold thresholds stability.csv compares: the base's and the twin's.
STABILITY_RULES: tuple[str, ...] = ("R0", "R2")
# The tables of a twin's tuning/ record, after its params.json (settings.PLATT_FILES).
RULES_FILE, NESTED_FILE, STABILITY_FILE = PLATT_FILES[1:]
# The columns of nested.csv, after its index, the outer fold.
NESTED_COLUMNS: tuple[str, ...] = (
    "chosen",
    *(f"inner_macro_f1_{rule}" for rule in RULES),
    "fold_macro_f1",
    "nested_macro_f1",
)
# The files of a base that its twin's freeze copies byte for byte: the twin sends its base's
# requests, so its latency and each problem's recorded call are its base's.
BASE_COPIES: tuple[str, ...] = (LATENCY_FILE, PER_PROBLEM_FILE)
# Why a twin refuses a base file whose sha256 or last commit is not the recorded one, even once
# its bytes are restored: the end of each message that refuses it.
BASE_CHANGED = (
    "A revert does not restore a changed base file: it brings the bytes back at a new commit, so "
    "the file's last commit is no longer the recorded one. A frozen file never changes "
    "(docs/adr/0003): a base file changed after its twin was exported or frozen means a new "
    "twin, under a new name (docs/adr/0011)."
)


class DevScores(NamedTuple):
    """The dev part as a twin's record reads it: every frame and series indexed by sample id.

    Attributes:
        scores: The base's dev probabilities, a column per focus tag: the dev rows of its
            ``scores.csv``. Other columns and rows (another problem's) are not read.
        labels: The 0/1 focus tags of the dev problems, one column per focus tag.
        folds: Each dev problem's fold, indexed like ``labels``.
        groups: Each dev problem's near-duplicate group, which the bootstrap resamples.
    """

    scores: pd.DataFrame
    labels: pd.DataFrame
    folds: pd.Series
    groups: pd.Series


def platt_half_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose one tag's threshold by R1: Platt on the logit, then a calibrated 0.5.

    Args:
        scores: One probability per problem, on Jev's 0.01 grid.
        labels: 0/1 true label per problem, aligned on ``scores``.

    Returns:
        ``snap_to_grid(expit(-b / a))``, the raw probability that Platt's map (slope ``a``,
        intercept ``b``) takes to 0.5; ``inf`` when no problem has the tag.

    Raises:
        ValueError: As ``platt_fit``: a score outside [0, 1], every problem has the tag, or the
            slope is not above 0.
    """
    if not np.asarray(labels, dtype=bool).any():
        return math.inf
    slope, intercept = platt_fit(scores, labels)
    return snap_to_grid(float(expit(-intercept / slope)))


# How each rule chooses one tag's threshold from its scores and 0/1 labels.
_CHOOSERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "R0": best_threshold,
    "R1": platt_half_threshold,
    "R2": lipton_threshold,
}


def rule_thresholds(rule: str, scores: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Choose each tag's threshold by one rule, on the scores and labels of the same problems.

    Args:
        rule: One of ``RULES``.
        scores: Probabilities, one column per tag, one row per problem.
        labels: 0/1 labels, with the index and columns of ``scores``.

    Returns:
        One threshold per tag, indexed by the columns of ``scores`` and named ``threshold``;
        ``inf`` for a tag that no problem has.

    Raises:
        ValueError: If ``rule`` is not one of ``RULES``, if ``labels`` is not indexed like
            ``scores``, or as the rule's Platt fit, naming the tag.
    """
    if rule not in _CHOOSERS:
        raise ValueError(f"{rule!r} is not a threshold rule of the record: one of {list(RULES)}.")
    if not (scores.index.equals(labels.index) and scores.columns.equals(labels.columns)):
        raise ValueError(
            "labels must have the index and columns of scores: the same problems and tags, in "
            "order."
        )
    values = [_choose(rule, scores[tag], labels[tag]) for tag in scores.columns]
    return pd.Series(values, index=scores.columns, name="threshold", dtype=float)


def _choose(rule: str, scores: pd.Series, labels: pd.Series) -> float:
    """Choose one tag's threshold by one rule; a Platt fit's refusal names the tag."""
    try:
        return _CHOOSERS[rule](scores.to_numpy(dtype=float), labels.to_numpy(dtype=int))
    except ValueError as error:
        raise ValueError(
            f"{rule} cannot choose the threshold of {scores.name!r}: {error}"
        ) from error


def cross_fitted_rule_thresholds(
    rule: str, scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Choose each fold's thresholds by one rule, on the problems of the other folds.

    Args:
        rule: One of ``RULES``.
        scores: Out-of-fold probabilities, one column per tag, one row per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.
        folds: Each problem's dev fold, with the index of ``scores``.

    Returns:
        ``cross_fitted_rows`` of ``rule_thresholds``: one row per sorted fold id (index
        ``fold``), with the columns of ``scores``; row k was chosen without fold k's labels.

    Raises:
        ValueError: As ``check_dev_folds`` (a test problem's fold, -1, is refused) and
            ``rule_thresholds``, naming the fold whose thresholds failed.
    """
    check_dev_folds(scores.index, labels, folds)

    def choose(kept_scores: pd.DataFrame, kept_labels: pd.DataFrame) -> pd.Series:
        try:
            return rule_thresholds(rule, kept_scores, kept_labels)
        except ValueError as error:
            fold = folds.drop(index=kept_scores.index).iloc[0]  # the one fold left out
            raise ValueError(f"On the folds other than fold {fold}: {error}") from error

    return cross_fitted_rows(scores, labels, folds, choose).rename_axis("fold")


def cross_fitted_rule_predictions(
    rule: str, scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Predict each fold with the thresholds one rule chose on the other folds.

    Args:
        rule: One of ``RULES``.
        scores: Out-of-fold probabilities, one column per tag, one row per dev problem.
        labels: 0/1 labels, with the index and columns of ``scores``.
        folds: Each problem's dev fold, with the index of ``scores``.

    Returns:
        0/1 ints, with the index and columns of ``scores``.

    Raises:
        ValueError: As ``cross_fitted_rule_thresholds``.
    """
    fold_thresholds = cross_fitted_rule_thresholds(rule, scores, labels, folds)
    return apply_fold_thresholds(scores, fold_thresholds, folds)


def rules_table(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Score each rule's cross-fitted dev predictions, with its paired difference from R0.

    Args:
        scores: Out-of-fold probabilities, one column per focus tag, one row per dev problem.
        labels: 0/1 focus tags, with the index and columns of ``scores``.
        folds: Each problem's dev fold, with the index of ``scores``.
        groups: Each problem's near-duplicate group, with the index of ``scores``.

    Returns:
        ``analysis.comparison.method_scores`` of the rules' predictions with R0 as the
        baseline, indexed by ``rule`` (``RULES``) and ``metric`` (``macro_f1``, ``micro_f1``,
        then each tag), with the columns ``f1``, ``f1_low``, ``f1_high``, ``delta``,
        ``delta_low`` and ``delta_high`` (``config.BOOTSTRAP_RESAMPLES`` group resamples).

    Raises:
        ValueError: As ``cross_fitted_rule_thresholds`` and ``method_scores``.
    """
    predictions = {
        rule: cross_fitted_rule_predictions(rule, scores, labels, folds) for rule in RULES
    }
    table = method_scores(labels, predictions, groups, baseline=RULES[0])
    return table.rename_axis(index=["rule", "metric"])


def nested_table(scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Choose a rule for each outer fold inside the other folds, then score the choices.

    For outer fold k, each rule is cross-fitted over the other folds alone and scored by its
    macro F1 there; the best rule (the first of ``RULES`` among equal scores) is fitted on those
    folds and predicts fold k. The nested macro F1 scores these predictions of every dev problem:
    whether choosing among the rules on dev adds optimism.

    Args:
        scores: Out-of-fold probabilities, one column per focus tag, one row per dev problem.
        labels: 0/1 focus tags, with the index and columns of ``scores``.
        folds: Each problem's dev fold, with the index of ``scores``: at least three folds.

    Returns:
        One row per sorted outer fold (index ``fold``), with the columns of ``NESTED_COLUMNS``:
        the rule chosen, each rule's macro F1 inside the other folds, the chosen rule's macro F1
        on the outer fold and the nested macro F1, the same on every row.

    Raises:
        ValueError: As ``cross_fitted_rule_thresholds``, naming the outer fold too.
    """
    check_dev_folds(scores.index, labels, folds)
    rows, chosen_thresholds = {}, {}
    for fold in np.sort(folds.unique()):
        try:
            rows[fold], chosen_thresholds[fold] = _nested_choice(scores, labels, folds, fold)
        except ValueError as error:
            raise ValueError(f"Inside the folds other than outer fold {fold}: {error}") from error
    fold_thresholds = pd.DataFrame.from_dict(chosen_thresholds, orient="index")
    predicted = apply_fold_thresholds(scores, fold_thresholds, folds)
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("fold")
    outer = {fold: (folds == fold).to_numpy() for fold in table.index}
    table["fold_macro_f1"] = [_macro_f1(labels[outer[k]], predicted[outer[k]]) for k in outer]
    table["nested_macro_f1"] = _macro_f1(labels, predicted)
    return table[list(NESTED_COLUMNS)]


def _nested_choice(
    scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series, fold: int
) -> tuple[dict[str, object], pd.Series]:
    """Choose a rule inside the folds other than ``fold``; return it and its thresholds there."""
    inner = (folds != fold).to_numpy()
    inner_f1 = {
        rule: _macro_f1(
            labels[inner],
            cross_fitted_rule_predictions(rule, scores[inner], labels[inner], folds[inner]),
        )
        for rule in RULES
    }
    chosen = max(RULES, key=inner_f1.__getitem__)  # max keeps the first of equal scores
    row = {"chosen": chosen, **{f"inner_macro_f1_{rule}": f1 for rule, f1 in inner_f1.items()}}
    return row, rule_thresholds(chosen, scores[inner], labels[inner])


def _macro_f1(truth: pd.DataFrame, predicted: pd.DataFrame) -> float:
    """Return the macro F1 of 0/1 predictions, without intervals."""
    return float(evaluate(truth, predicted).macro_f1)


def stability_table(scores: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Show how far each tag's threshold moves with the folds that choose it, under R0 and R2.

    Args:
        scores: Out-of-fold probabilities, one column per focus tag, one row per dev problem.
        labels: 0/1 focus tags, with the index and columns of ``scores``.
        folds: Each problem's dev fold, with the index of ``scores``.

    Returns:
        One row per tag and rule of ``STABILITY_RULES`` (index ``tag``, ``rule``, the tags in
        the order of the columns of ``scores``): ``fold_<k>``, the threshold fold k is predicted
        with (chosen on the other folds); ``spread``, the largest of them minus the smallest (0
        when all are equal, ``inf`` when some but not all are ``inf``); ``final``, the
        threshold chosen on every problem; ``flips``, the dev problems that some of the fold
        thresholds predict the tag for and others do not.

    Raises:
        ValueError: As ``cross_fitted_rule_thresholds``.
    """
    fold_thresholds = {
        rule: cross_fitted_rule_thresholds(rule, scores, labels, folds) for rule in STABILITY_RULES
    }
    finals = {rule: rule_thresholds(rule, scores, labels) for rule in STABILITY_RULES}
    rows = {
        (tag, rule): _stability_row(scores[tag], fold_thresholds[rule][tag], finals[rule][tag])
        for tag in scores.columns
        for rule in STABILITY_RULES
    }
    table = pd.DataFrame(list(rows.values()))
    table.index = pd.MultiIndex.from_tuples(list(rows), names=["tag", "rule"])
    return table


def _stability_row(
    scores: pd.Series, fold_thresholds: pd.Series, final: float
) -> dict[str, float | int]:
    """Return one tag's row of stability.csv under one rule."""
    values = fold_thresholds.to_numpy(dtype=float)
    reached = scores.to_numpy(dtype=float)[:, np.newaxis] >= values
    flips = int((reached.any(axis=1) & ~reached.all(axis=1)).sum())
    spread = 0.0 if (values == values[0]).all() else float(values.max() - values.min())
    columns = {f"fold_{k}": value for k, value in zip(fold_thresholds.index, values, strict=True)}
    return {**columns, "spread": spread, "final": float(final), "flips": flips}


def platt_files(
    params: Mapping[str, object],
    rules: pd.DataFrame,
    nested: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict[str, bytes]:
    """Render every file of a twin's record, by its name in ``tuning/``.

    Args:
        params: What params.json holds.
        rules: ``rules_table``'s table.
        nested: ``nested_table``'s table.
        stability: ``stability_table``'s table.

    Returns:
        The bytes of each file of ``settings.PLATT_FILES``: params.json as indented JSON (an
        infinite threshold as ``"inf"``), each table as CSV with its index.
    """
    text = json.dumps(json_safe(params), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tables = {RULES_FILE: rules, NESTED_FILE: nested, STABILITY_FILE: stability}
    csv = {
        name: table.to_csv(lineterminator="\n").encode("utf-8") for name, table in tables.items()
    }
    return {PARAMS_FILE: text.encode("utf-8"), **csv}


def export_platt(
    twin: str, out: Path, dev: DevScores, base_files: Mapping[str, Mapping[str, str]]
) -> Path:
    """Write a calibrated twin's dev-only record in ``out`` once; later, only compare it.

    Args:
        twin: A calibrated twin, a key of ``settings.PLATT_BASES``.
        out: The record's folder, the twin's ``tuning/``.
        dev: The base's dev scores, and the labels, folds and groups of the dev problems. Only
            the focus tags of the labels' problems are read from the scores, in the labels'
            order.
        base_files: The sha256 and last commit of each base file the scores come from
            (``base_paths``), by file name: params.json records them.

    Returns:
        ``out``, written once; a second export compares its files byte for byte.

    Raises:
        ValueError: If ``twin`` is not a calibrated twin; before any fit, as ``check_dev_folds``
            on the labels and folds (folds not indexed like the labels, or a test problem's
            fold, -1), if the groups are not indexed like the labels, or if the scores lack a
            dev problem or a focus tag or hold NaN; if a Platt fit fails (a slope not above 0,
            or a tag on every problem of the folds it is fitted on; the message names the twin,
            the tag and the fold); or if ``out`` holds another record.
    """
    if twin not in PLATT_BASES:
        raise ValueError(f"{twin!r} is not a calibrated twin: one of {list(PLATT_BASES)}.")
    tags = list(FOCUS_TAGS)
    labels = dev.labels[tags]
    # The caller's folds and groups are checked here, so a misaligned one is never framed as a
    # Platt fit that failed.
    check_dev_folds(labels.index, labels, dev.folds)
    if not dev.groups.index.equals(labels.index):
        raise ValueError(
            "groups must have the same index as problems: the same problems, in order."
        )
    scores = dev.scores.reindex(index=labels.index, columns=tags)
    if scores.isna().any().any():
        raise ValueError(
            f"The dev scores of {PLATT_BASES[twin]} lack a dev problem or a focus tag, or hold "
            "NaN: a twin's record reads the dev rows of its base's frozen scores.csv, a "
            "probability of every focus tag for every dev problem (docs/adr/0011)."
        )
    try:
        tables = _record_tables(dev._replace(scores=scores, labels=labels))
    except ValueError as error:
        raise ValueError(
            f"The record of {twin} cannot be computed on the dev scores of {PLATT_BASES[twin]}: "
            f"{error} The rule was declared before the record was computed (docs/adr/0011), so "
            "nothing falls back to another one: report the failure."
        ) from error
    thresholds, nested = tables["thresholds"], tables["nested"]
    params = {
        "rule": TWIN_RULE,
        "threshold_rule": RULE_SETTINGS,
        "base": PLATT_BASES[twin],
        "base_files": base_files,
        "thresholds": {tag: float(value) for tag, value in thresholds.items()},
        "nested_choice": nested["chosen"].tolist(),
        "nested_macro_f1": float(nested["nested_macro_f1"].iloc[0]),
    }
    files = platt_files(params, tables["rules"], nested, tables["stability"])
    write_once(files, Path(out), owner=f"the threshold rule of {twin}")
    return Path(out)


def _record_tables(dev: DevScores) -> dict[str, pd.DataFrame | pd.Series]:
    """Compute what a twin's record holds: the three tables and R2's final thresholds.

    The tables come first: a Platt fit that fails on every dev problem also fails on the folds
    of one of them, whose message names the fold.
    """
    return {
        "nested": nested_table(dev.scores, dev.labels, dev.folds),
        "rules": rules_table(dev.scores, dev.labels, dev.folds, dev.groups),
        "stability": stability_table(dev.scores, dev.labels, dev.folds),
        "thresholds": rule_thresholds(TWIN_RULE, dev.scores, dev.labels),
    }


def base_paths(base: str) -> dict[str, Path]:
    """Name the frozen files of a Jev method that its calibrated twin reads.

    Args:
        base: The twin's base, a frozen Jev method.

    Returns:
        Its answers file of each part, then its ``scores.csv`` and ``run_record.json``, by file
        name.
    """
    directory = method_dir(base)
    paths = {answers_file(base, part).name: answers_file(base, part) for part in PARTS}
    return paths | {name: directory / name for name in (SCORES_FILE, RUN_RECORD_FILE)}


def read_platt_params(tuning: Path) -> dict[str, object]:
    """Read a twin's params.json back.

    Args:
        tuning: The twin's ``tuning/`` folder, as ``export_platt`` wrote it.

    Returns:
        What params.json holds; an infinite threshold is the string ``"inf"``.

    Raises:
        FileNotFoundError: If the record has not been exported.
    """
    return json.loads((Path(tuning) / PARAMS_FILE).read_text(encoding="utf-8"))


def check_base_files(tuning: Path, current: Mapping[str, Mapping[str, str]]) -> None:
    """Fail unless a twin's params.json names its base's files as they are now.

    Each file must have the sha256 and, as its last commit, the commit that params.json records:
    a change and its revert leave the bytes as they were, not the commit, and still fail.

    Args:
        tuning: The twin's ``tuning/`` folder.
        current: The sha256 and last commit of each of the base's files now, by file name
            (``platt_commands.base_files``).

    Raises:
        ValueError: If a file's sha256 or last commit differs from params.json's, or a file is
            named on one side only.
        FileNotFoundError: If the record has not been exported.
    """
    params = read_platt_params(tuning)
    stated = params["base_files"]
    differ = sorted(
        name for name in stated.keys() | current.keys() if stated.get(name) != current.get(name)
    )
    if differ:
        raise ValueError(
            f"{Path(tuning) / PARAMS_FILE} names other files of {params['base']} than those "
            f"committed now: {differ} differ, and each must have the sha256 and, as its last "
            "commit, the commit that params.json records. If params.json was edited, restore "
            f"it from git. {BASE_CHANGED}"
        )


def check_thresholds(tuning: Path, thresholds: Sequence[float]) -> None:
    """Fail unless the thresholds the twin's rule gives now are those of its params.json.

    Args:
        tuning: The twin's ``tuning/`` folder.
        thresholds: One threshold per focus tag, in ``FOCUS_TAGS`` order: the ``thresholds_``
            of the twin's ``PlattLiptonThresholds``, fitted on its base's dev answers.

    Raises:
        ValueError: If a threshold differs from params.json's (snapped thresholds compare
            exactly).
        FileNotFoundError: If the record has not been exported.
    """
    params = read_platt_params(tuning)
    stated = [params["thresholds"][tag] for tag in FOCUS_TAGS]
    found = json_safe([float(value) for value in thresholds])
    if found != stated:
        raise ValueError(
            f"{Path(tuning) / PARAMS_FILE} holds the thresholds {stated}, but the twin's rule "
            f"gives {found} on its base's answers files: the record was exported from other "
            "scores than those answers give (docs/adr/0011)."
        )


def twin_entries(
    twin: str, base_files: Mapping[str, Mapping[str, str]], base_entries: Mapping[str, object]
) -> dict[str, object]:
    """Return what a twin's run record keeps of its base, its threshold rule and its disclosure.

    Args:
        twin: A calibrated twin, a key of ``settings.PLATT_BASES``.
        base_files: The sha256 and last commit of each of its base's files, by file name.
        base_entries: The ``method_entries`` of its base's run record.

    Returns:
        ``base`` (the base's name and ``base_files``), ``served_models`` (the base's: the
        twin's scores are its answers), ``threshold_rule`` (``RULE_SETTINGS``),
        ``dev_in_sample`` (the base's: a twin's dev rows are out of fold exactly when its
        base's are), ``test_informed`` (``settings.TEST_INFORMED``: the rule was proposed after
        a test score was read, so the command line never names a twin eligible) and
        ``latency_from`` (the base, whose ``latency.csv`` the twin's is).
    """
    base = PLATT_BASES[twin]
    return {
        "base": {"name": base, "files": dict(base_files)},
        "served_models": list(base_entries["served_models"]),
        "threshold_rule": dict(RULE_SETTINGS),
        "dev_in_sample": base_entries.get("dev_in_sample") is True,
        "test_informed": dict(TEST_INFORMED),
        "latency_from": base,
    }


def copy_from_base(base_directory: Path, directory: Path, file_name: str) -> str:
    """Copy one of ``BASE_COPIES`` from a base's directory into its twin's once; later, compare.

    A twin sends its base's requests, so its latency and each problem's recorded call are its
    base's: nothing is timed or called again. The copy is written under a temporary name, then
    renamed.

    Args:
        base_directory: The base's directory, which holds its committed file.
        directory: The twin's directory; it must exist.
        file_name: The file, one of ``BASE_COPIES``.

    Returns:
        A line to print: the file was copied, or it is the base's.

    Raises:
        ValueError: If the twin's file exists and is not the base's, byte for byte.
    """
    source, target = Path(base_directory) / file_name, Path(directory) / file_name
    if target.exists():
        check_copy(base_directory, directory, file_name)
        return f"{target} is {source}, byte for byte"
    partial = Path(directory) / f".{file_name}.partial"
    partial.write_bytes(source.read_bytes())
    os.replace(partial, target)
    return f"copied {source} to {target}: the twin sends its base's requests"


def check_copy(base_directory: Path, directory: Path, file_name: str) -> None:
    """Fail unless a twin's copy of one of ``BASE_COPIES`` is its base's, byte for byte.

    Args:
        base_directory: The base's directory.
        directory: The twin's directory.
        file_name: The file, one of ``BASE_COPIES``.

    Raises:
        ValueError: If the twin has no such file, or another one than its base's.
    """
    source, target = Path(base_directory) / file_name, Path(directory) / file_name
    if not target.is_file() or target.read_bytes() != source.read_bytes():
        raise ValueError(
            f"{target} is not {source}, byte for byte: a calibrated twin sends its base's "
            "requests, so its latency and each problem's recorded call are its base's "
            "(docs/adr/0011). Restore it from git."
        )
