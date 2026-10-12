"""Log the calibration of the tag model's out-of-fold scores to its MLflow run.

``python -m codeforces_tags.tracking calibrate`` computes what notebook 04's section 5 shows, from
the committed files and the dev problems only (ADR 0001): the out-of-fold scores of the frozen
``tfidf`` (its method folder is loaded, never refit) and the first look's, cross-fitted on the 8
focus tags. It builds the four probabilities (``probabilities_by_method``), their calibration
table and the two reliability diagrams, and adds them to the method run of ``tfidf``
(``log_run_extras``): a metric per estimate of the table (``calibration_metrics``), the table as
``calibration/calibration_table.csv`` and the figures beside it. The run keeps any key or file it
already holds, never updated, so a second run logs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    path_not_unique,
    validate_param_and_metric_name,
)

from codeforces_tags.analysis.plotting import plot_reliability, plot_reliability_grid, setup_style
from codeforces_tags.config import BOOTSTRAP_CONFIDENCE, SPLIT_FILE
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.calibration import (
    METRICS,
    POOLED,
    calibration_table,
    probabilities_by_method,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE, load_run_record
from codeforces_tags.methods import load_method, method_dir, model_dir
from codeforces_tags.model.first_look import first_look_pipeline
from codeforces_tags.model.tag_model import NAME, TagModel
from codeforces_tags.model.training import cross_fit
from codeforces_tags.provenance import RunRecord, check_dataset, file_sha256
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.methods import log_run_extras, method_run_id

# Where the method run of tfidf holds the calibration table and the reliability diagrams.
TABLE_PATH = "calibration/calibration_table.csv"
FIGURE_PATHS = {
    "reliability": "calibration/reliability.png",
    "reliability_by_tag": "calibration/reliability_by_tag.png",
}


def calibration_metrics(table: pd.DataFrame) -> dict[str, float]:
    """Name each estimate of a calibration table as a run metric.

    Args:
        table: The table of ``calibration_table``, indexed by ``method`` and ``tag``.

    Returns:
        ``<metric>/<method>/<tag>`` for each metric of ``METRICS`` and each row of the table,
        e.g. ``brier_skill/svm_platt/all``, mapped to the row's estimate: 4 methods x 9 rows x 3
        metrics = 108 keys for the focus tags.

    Raises:
        ValueError: Naming every key MLflow would refuse, before anything is logged: a key with
            a character other than a word character, ``/``, ``.``, ``-``, ``:`` or a space, of
            more than ``MAX_ENTITY_KEY_LENGTH`` characters, or that a path normalisation changes
            (e.g. an empty tag). ``client.log_metric`` raises on such a key only once the
            earlier keys are logged, and so again at every later run.
    """
    metrics = {
        f"{metric}/{method}/{tag}": float(table.at[(method, tag), metric])
        for metric in METRICS
        for method, tag in table.index
    }
    refused = [key for key in metrics if not _accepted(key)]
    if refused:
        raise ValueError(
            f"MLflow refuses the metric keys {refused}: a key holds word characters, '/', '.', "
            f"'-', ':' and spaces, at most {MAX_ENTITY_KEY_LENGTH} of them, and no part that is "
            "empty, '.' or '..'."
        )
    return metrics


def _accepted(key: str) -> bool:
    """Tell whether MLflow accepts a metric key, by its own checks."""
    return (
        bool(validate_param_and_metric_name(key))
        and len(key) <= MAX_ENTITY_KEY_LENGTH
        and not path_not_unique(key)
    )


def run_calibration(
    *, data_dir: str | Path | None = None, split_path: Path = SPLIT_FILE, **store: Any
) -> str:
    """Compute the calibration of the frozen ``tfidf``, and add it to its method run.

    It reads the run record of ``tfidf``, checks that the split is the one ``tfidf`` was frozen
    with and finds its method run first, before it hashes the dataset or fits anything: the
    split's folds and groups drive every probability and interval, which the run keeps for good.
    It then checks the dataset, reads the dev problems of the split, loads the committed method
    folder for its out-of-fold scores, cross-fits the first look on the 8 focus tags, and builds
    the four probabilities, their calibration table and the two reliability diagrams, in the
    notebooks' style. Everything is computed again at every run, and ``log_run_extras`` skips
    each metric key and file the run holds: a run never updates one. After a change of code,
    the run keeps the old values, and nothing logged shows the difference.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.
        split_path: The frozen split.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        A line to print: the run, and each method's pooled Brier skill with its interval.

    Raises:
        FileNotFoundError: If ``tfidf`` is not frozen (no run record).
        ValueError: If the split's sha256 is not the run record's, if the store holds no
            method run of its recipe, if the dataset is refused, if its out-of-fold scores are
            not those of the split's dev problems, in order, or if a metric key would be
            refused (``calibration_metrics``).
        TypeError: If its method folder holds another method than the tag model.
    """
    record = _frozen_record()
    _check_frozen_split(record, split_path)
    run_id = method_run_id(record.recipe_sha256, **store)
    if run_id is None:
        raise ValueError(
            f"No method run has the recipe {record.recipe_sha256} of {NAME}: run "
            "`python -m codeforces_tags.tuning freeze` first, which logs it."
        )
    check_dataset(data_dir)
    dev_raw = select_part(read_samples(data_dir), load_split(split_path), "dev")
    labels = focus_labels(dev_raw)
    svm_scores = _out_of_fold_scores(dev_raw)
    logits = cross_fit(first_look_pipeline(), dev_raw, labels, dev_raw["fold"]).scores
    probabilities = probabilities_by_method(svm_scores, logits, labels, dev_raw["fold"])
    table = calibration_table(probabilities, labels, dev_raw["group"])
    _log_calibration(record.recipe_sha256, probabilities, labels, table, **store)
    return (
        f"calibration of {NAME} logged to run {run_id}, keeping any key or file it held; "
        f"{_pooled_skill(table)}"
    )


def _frozen_record() -> RunRecord:
    """Read the run record of the frozen ``tfidf``."""
    directory = method_dir(NAME)
    if not (directory / RUN_RECORD_FILE).is_file():
        raise FileNotFoundError(
            f"{directory} holds no {RUN_RECORD_FILE}: run `python -m codeforces_tags.tuning "
            "freeze` first."
        )
    return load_run_record(directory)


def _check_frozen_split(record: RunRecord, split_path: Path) -> None:
    """Fail unless the split file is the one the run record of ``tfidf`` names, by its sha256."""
    given = file_sha256(split_path)
    if given != record.split_sha256:
        raise ValueError(
            f"{split_path} has the sha256 {given}, not the {record.split_sha256} of the split "
            f"{NAME} was frozen with: pass that split (--split)."
        )


def _out_of_fold_scores(dev_raw: pd.DataFrame) -> pd.DataFrame:
    """Return the committed tag model's out-of-fold scores of the dev problems, in their order."""
    folder = model_dir(NAME)
    tag_model = load_method(folder)
    if not isinstance(tag_model, TagModel):
        raise TypeError(f"{folder} holds the method {tag_model.name!r}, not the tag model {NAME}.")
    scores = tag_model.out_of_fold_scores()
    if not scores.index.equals(dev_raw.index):
        raise ValueError(
            f"The out-of-fold scores of {NAME} score {len(scores)} problems, not the "
            f"{len(dev_raw)} dev problems of the split, in their order: its method folder was "
            "not frozen with the split its run record names."
        )
    return scores


def _log_calibration(
    recipe_sha256: str,
    probabilities: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    table: pd.DataFrame,
    **store: Any,
) -> None:
    """Draw the two reliability diagrams, and log them, the table and its metrics to the run."""
    metrics = calibration_metrics(table)  # every key is checked before anything is logged
    figures = {}
    with plt.rc_context():  # the notebooks' style, for these figures only
        setup_style()
        try:
            figures[FIGURE_PATHS["reliability"]] = plot_reliability(probabilities, labels)
            figures[FIGURE_PATHS["reliability_by_tag"]] = plot_reliability_grid(
                probabilities, labels
            )
            log_run_extras(
                recipe_sha256, metrics=metrics, figures=figures, tables={TABLE_PATH: table}, **store
            )
        finally:
            for figure in figures.values():
                plt.close(figure)


def _pooled_skill(table: pd.DataFrame) -> str:
    """Describe each method's Brier skill over all problem-tag pairs, with its interval."""
    pooled = table.xs(POOLED, level="tag")
    skills = [
        f"{method} {row.brier_skill:.3f} [{row.brier_skill_low:.3f}, {row.brier_skill_high:.3f}]"
        for method, row in pooled.iterrows()
    ]
    return f"pooled Brier skill [{BOOTSTRAP_CONFIDENCE:.0%} interval]: " + ", ".join(skills)
