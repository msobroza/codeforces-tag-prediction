"""Tests for tracking calibrate: the calibration of tfidf's out-of-fold scores, in its MLflow run.

Each test freezes a small tag model as ``tfidf`` in a folder of its own (``log_tag_model``), with
a dataset of its 54 dev problems and two test problems, and runs the command on a temporary store.
"""

import dataclasses
import io
import json
import re
import shutil
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from codeforces_tags import methods
from codeforces_tags.config import CALIBRATION_METHODS, FOCUS_TAGS
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.evaluation.calibration import (
    CALIBRATION_COLUMNS,
    METRICS,
    POOLED,
    calibration_table,
    probabilities_by_method,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.methods import method_dir, model_dir, save_method
from codeforces_tags.methods.label_prior.method import LabelPrior
from codeforces_tags.model.first_look import first_look_pipeline
from codeforces_tags.model.tag_model import NAME
from codeforces_tags.model.training import cross_fit
from codeforces_tags.provenance import file_sha256
from codeforces_tags.tracking import calibration
from codeforces_tags.tracking.calibration import (
    FIGURE_PATHS,
    TABLE_PATH,
    calibration_metrics,
    run_calibration,
)
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import method_run_id

# The test problems of the dataset: the command reads their files (select_part needs every
# problem of the split) and drops them before it computes anything.
TEST_IDS = [200, 201]


class Env(NamedTuple):
    """What a calibrate test works with: a dataset, its split, a store, and what the spy saw."""

    dataset: Path
    split_file: Path
    store: dict
    seen: dict


@pytest.fixture(name="dataset")
def dataset_fixture(focus_raw_dev, make_raw_record, write_samples):
    """Write the 54 dev problems of ``focus_raw_dev`` and two test problems; return the folder."""
    dev = focus_raw_dev.drop(columns=["fold", "group"]).to_dict(orient="index")
    test = {
        sample_id: make_raw_record(prob_desc_description=f"Test problem {sample_id}.")
        for sample_id in TEST_IDS
    }
    return write_samples(dev | test)


@pytest.fixture(name="split_file")
def split_file_fixture(focus_raw_dev, tmp_path):
    """Write the split of the dataset: the dev folds and groups of ``focus_raw_dev``, then test."""
    dev = focus_raw_dev[["fold", "group"]].assign(part="dev")
    test = pd.DataFrame(
        {"fold": -1, "group": [1000, 1001], "part": "test"},
        index=pd.Index(TEST_IDS, name="sample_id"),
    )
    path = tmp_path / "split.csv"
    pd.concat([dev, test])[["part", "fold", "group"]].to_csv(path)
    return path


@pytest.fixture(name="spy")
def spy_fixture(monkeypatch):
    """Record the dataset checks and the first looks fitted, and let the fits run.

    The dataset check only records its argument: the test dataset is not the challenge's.
    """
    seen = {"dataset": [], "fits": []}
    fit = calibration.cross_fit

    def cross_fit_and_record(estimator, *args):
        seen["fits"].append(estimator)
        return fit(estimator, *args)

    monkeypatch.setattr(calibration, "check_dataset", seen["dataset"].append)
    monkeypatch.setattr(calibration, "cross_fit", cross_fit_and_record)
    return seen


@pytest.fixture(name="env")
def env_fixture(dataset, split_file, store, spy):
    """Bundle the dataset, its split, the store and what the spy sees."""
    return Env(dataset, split_file, store, spy)


@pytest.fixture(name="record")
def record_fixture(monkeypatch, placeholder_record, split_file, tmp_path):
    """Point ``method_dir`` at a folder of the test; return the run record of ``tfidf``.

    The record names the split of the test by its sha256, as ``freeze`` names the split it read.
    """
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path / "assets")
    return placeholder_record(NAME, split_sha256=file_sha256(split_file))


def write_record(record):
    """Write the run record of ``tfidf`` into its method directory, as ``freeze`` does."""
    method_dir(NAME).mkdir(parents=True, exist_ok=True)
    text = json.dumps(record.to_dict(), allow_nan=False)
    (method_dir(NAME) / RUN_RECORD_FILE).write_text(text, encoding="utf-8")


@pytest.fixture(name="frozen")
def frozen_fixture(record, log_tag_model):
    """Freeze the focus tag model as ``tfidf``: run record, method folder, method run; return it."""
    write_record(record)
    log_tag_model(model_dir(NAME))
    return record


@pytest.fixture(name="table")
def table_fixture(focus_raw_dev, focus_tag_model):
    """Compute here the calibration table of the focus tag model's out-of-fold scores."""
    labels = focus_labels(focus_raw_dev)
    folds = focus_raw_dev["fold"]
    logits = cross_fit(first_look_pipeline(), focus_raw_dev, labels, folds).scores
    scores = focus_tag_model.out_of_fold_scores()
    probabilities = probabilities_by_method(scores, logits, labels, folds)
    return calibration_table(probabilities, labels, focus_raw_dev["group"])


def run(env, split_file=None):
    """Run the command on the dataset and store of ``env``, with its split unless another."""
    split_path = split_file or env.split_file
    return run_calibration(data_dir=env.dataset, split_path=split_path, **env.store)


def calibration_keys(run_id):
    """Return the run's metrics that ``calibration_metrics`` names: ``<metric>/<method>/<tag>``."""
    logged = mlflow.MlflowClient().get_run(run_id).data.metrics
    return {key: value for key, value in logged.items() if key.split("/")[0] in METRICS}


def logged_artifacts(run_id):
    """Return the path and size of each artifact under the run's ``calibration/`` folder."""
    items = mlflow.MlflowClient().list_artifacts(run_id, "calibration")
    return [(item.path, item.file_size) for item in items]


def logged_table_text(run_id):
    """Return the text of the calibration table the run holds."""
    return mlflow.artifacts.load_text(f"runs:/{run_id}/{TABLE_PATH}")


def test_calibration_metrics_name_each_metric_of_each_row(probabilities, calibration_data):
    table = calibration_table(
        probabilities, calibration_data.labels, calibration_data.groups, resamples=20
    )

    metrics = calibration_metrics(table)

    rows = [(method, tag) for method in CALIBRATION_METHODS for tag in [*FOCUS_TAGS, POOLED]]
    assert len(metrics) == 108
    assert set(metrics) == {
        f"{metric}/{method}/{tag}" for metric in METRICS for method, tag in rows
    }
    assert (
        metrics["brier/svm_platt/number theory"]
        == table.at[("svm_platt", "number theory"), "brier"]
    )
    assert metrics["log_loss/svm_isotonic/all"] == table.at[("svm_isotonic", POOLED), "log_loss"]
    assert max(len(key) for key in metrics) == len("brier_skill/logistic_regression/probabilities")


@pytest.mark.parametrize(
    "tag",
    [
        pytest.param("*special", id="a character MLflow refuses"),
        pytest.param("x" * 250, id="more than 250 characters"),
        pytest.param("", id="a path normpath changes"),
    ],
)
def test_calibration_metrics_refuse_a_key_mlflow_would_refuse(tag, store):
    # Client.log_metric raises on such a key after the earlier keys are logged,
    # so a half-logged run would stop on it again at every run. Every key is checked first.
    index = pd.MultiIndex.from_tuples([("svm_platt", tag)], names=["method", "tag"])
    table = pd.DataFrame(0.5, index=index, columns=list(CALIBRATION_COLUMNS))
    keys = [f"{metric}/svm_platt/{tag}" for metric in METRICS]

    with pytest.raises(ValueError, match=re.escape(f"MLflow refuses the metric keys {keys}")):
        calibration_metrics(table)

    experiment_id = use_store(**store)
    client = mlflow.MlflowClient()
    run_id = client.create_run(experiment_id).info.run_id
    for key in keys:
        with pytest.raises(MlflowException):
            client.log_metric(run_id, key, 0.5)


def test_calibrate_logs_every_metric_figure_and_table_to_the_tfidf_run(frozen, env, table):
    with plt.rc_context({"figure.figsize": (3.0, 2.0)}):
        message = run(env)
        assert tuple(plt.rcParams["figure.figsize"]) == (3.0, 2.0)  # the style stays inside

    run_id = method_run_id(frozen.recipe_sha256, **env.store)
    logged = calibration_keys(run_id)
    assert logged == calibration_metrics(table)
    assert len(logged) == 108
    assert "brier/svm_platt/number theory" in logged
    paths = [path for path, size in logged_artifacts(run_id) if size > 0]
    assert sorted(paths) == sorted([TABLE_PATH, *FIGURE_PATHS.values()])
    logged_table = pd.read_csv(
        io.StringIO(logged_table_text(run_id)),
        index_col=["method", "tag"],
        float_precision="round_trip",
    )
    pd.testing.assert_frame_equal(logged_table, table, check_exact=True)
    # A resample of 54 problems can draw no positive of a tag. Its skill bounds
    # are then NaN, never infinite, and the logged table keeps them.
    assert logged_table["brier_skill_low"].isna().any()
    assert env.seen["dataset"] == [env.dataset]
    assert len(env.seen["fits"]) == 1  # the first look, on the eight focus tags
    assert plt.get_fignums() == []
    assert message.startswith(f"calibration of {NAME} logged to run {run_id}")
    for method, row in table.xs(POOLED, level="tag").iterrows():
        low, high = row.brier_skill_low, row.brier_skill_high
        assert f"{method} {row.brier_skill:.3f} [{low:.3f}, {high:.3f}]" in message


@pytest.mark.usefixtures("frozen")
def test_calibrate_logs_each_figure_under_its_path_in_the_notebooks_style(env, monkeypatch):
    # The logged files are only checked to exist above: keep the figures the command hands over.
    figures = {}
    log = calibration.log_run_extras

    def keep_figures(recipe_sha256, **extras):
        figures.update(extras["figures"])
        return log(recipe_sha256, **extras)

    monkeypatch.setattr(calibration, "log_run_extras", keep_figures)

    # The style in place says normal titles: a bold one comes from the notebooks' style.
    with plt.rc_context({"axes.titleweight": "normal"}):
        run(env)

    pooled = figures[FIGURE_PATHS["reliability"]]
    by_tag = figures[FIGURE_PATHS["reliability_by_tag"]]
    assert pooled.axes[0].get_title() == "Reliability diagram: all problem-tag pairs"
    assert len(pooled.axes) == 1 + len(CALIBRATION_METHODS)  # the diagram, then the histograms
    assert [ax.get_title() for ax in by_tag.axes] == list(FOCUS_TAGS)
    assert {ax.title.get_fontweight() for ax in [*pooled.axes, *by_tag.axes]} == {"bold"}


def test_calibrate_twice_logs_nothing_new(frozen, env, monkeypatch):
    # The second run computes other values, as it would after a change of code:
    # the run keeps the first ones, and logs nothing.
    run(env)
    run_id = method_run_id(frozen.recipe_sha256, **env.store)
    first = calibration_keys(run_id)
    artifacts = logged_artifacts(run_id)
    text = logged_table_text(run_id)
    computed = calibration.calibration_table
    monkeypatch.setattr(calibration, "calibration_table", lambda *args: computed(*args) + 0.5)

    run(env)

    client = mlflow.MlflowClient()
    for key, value in first.items():
        assert [metric.value for metric in client.get_metric_history(run_id, key)] == [value]
    assert calibration_keys(run_id) == first
    assert logged_artifacts(run_id) == artifacts
    assert logged_table_text(run_id) == text
    assert len(env.seen["fits"]) == 2  # everything is computed again
    assert plt.get_fignums() == []


@pytest.mark.usefixtures("record")
def test_calibrate_refuses_a_tfidf_that_is_not_frozen(env):
    with pytest.raises(FileNotFoundError, match="holds no run_record.json: run `python -m"):
        run(env)

    assert env.seen == {"dataset": [], "fits": []}  # refused before the dataset is hashed


def test_calibrate_refuses_a_tfidf_without_a_method_run(record, env):
    write_record(record)

    with pytest.raises(ValueError, match=f"No method run has the recipe {record.recipe_sha256}"):
        run(env)

    assert env.seen == {"dataset": [], "fits": []}


@pytest.mark.usefixtures("frozen")
def test_calibrate_refuses_a_method_folder_that_holds_another_method(env):
    shutil.rmtree(model_dir(NAME))
    save_method(LabelPrior(("math",)), model_dir(NAME))

    with pytest.raises(TypeError, match="holds the method 'label_prior', not the tag model tfidf"):
        run(env)

    assert env.seen["fits"] == []


@pytest.mark.usefixtures("frozen")
def test_calibrate_refuses_a_split_other_than_the_one_tfidf_was_frozen_with(env, tmp_path):
    # The same dev problems, in the same order, dealt to other folds: the scores' index matches
    # the split's dev problems, but its folds would drive the scaling, the cross-fits and the
    # bootstrap, and the run would keep what they give for good.
    split = pd.read_csv(env.split_file, index_col="sample_id")
    dev = split["part"] == "dev"
    split.loc[dev, "fold"] = (split.loc[dev, "fold"] + 1) % 3
    other_split = tmp_path / "other_split.csv"
    split.to_csv(other_split)
    frozen_with, given = file_sha256(env.split_file), file_sha256(other_split)

    with pytest.raises(ValueError, match=f"sha256 {given}, not the {frozen_with} of the split"):
        run(env, other_split)

    assert env.seen == {"dataset": [], "fits": []}  # refused before the dataset is hashed


def test_calibrate_refuses_out_of_fold_scores_of_other_dev_problems(frozen, env, tmp_path):
    # A run record that names a split whose dev part lost a problem, beside a method folder
    # frozen on the full one: the scores are not the problems of the split the record names.
    split = pd.read_csv(env.split_file, index_col="sample_id")
    split.loc[100, ["part", "fold", "group"]] = ["test", -1, 1002]  # its pair stays in dev
    other_split = tmp_path / "other_split.csv"
    split.to_csv(other_split)
    write_record(dataclasses.replace(frozen, split_sha256=file_sha256(other_split)))

    with pytest.raises(ValueError, match="score 54 problems, not the 53 dev problems of the split"):
        run(env, other_split)

    assert env.seen["fits"] == []


@pytest.mark.usefixtures("frozen")
def test_calibrate_closes_its_figures_when_the_logging_fails(env, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("the store went away")

    monkeypatch.setattr(calibration, "log_run_extras", fail)

    with pytest.raises(RuntimeError, match="the store went away"):
        run(env)

    assert plt.get_fignums() == []
