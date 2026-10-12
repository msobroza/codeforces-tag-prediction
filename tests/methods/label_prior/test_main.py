"""Tests for the label_prior commands, on a tiny dataset written to a temporary folder."""

import dataclasses
import itertools
import json
import runpy
import sys

import mlflow
import pandas as pd
import pytest
from mlflow.data.pandas_dataset import from_pandas

from codeforces_tags import provenance
from codeforces_tags.config import (
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
)
from codeforces_tags.conformance import ConformanceError, check_frozen_method
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    SCORES_FILE,
    load_predictions,
    load_run_record,
)
from codeforces_tags.methods import METHOD_FILE, METHODS, MODEL_FOLDER, MethodSpec
from codeforces_tags.methods.label_prior import __main__ as cli
from codeforces_tags.methods.label_prior.settings import NAME, PRIOR_FILE
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import load_logged_method, log_method, method_run_id

# sample_id -> (part, fold, tags): two dev problems per fold, then four test problems. "trees" has
# its only dev positive in fold 0, so that fold is predicted without it.
PROBLEMS = {
    1: ("dev", 0, ["math", "trees"]),
    2: ("dev", 0, ["graphs"]),
    3: ("dev", 1, ["math"]),
    4: ("dev", 1, ["graphs"]),
    5: ("dev", 2, ["math"]),
    6: ("dev", 2, []),
    7: ("dev", 3, ["math"]),
    8: ("dev", 3, ["implementation"]),
    9: ("dev", 4, []),
    10: ("dev", 4, ["greedy"]),
    11: ("test", -1, ["math"]),
    12: ("test", -1, ["strings"]),
    13: ("test", -1, []),
    14: ("test", -1, ["trees", "graphs"]),
}


def row(*tags):
    """Return the 0/1 row of the focus tags, with a 1 for each tag given."""
    return [int(tag in tags) for tag in FOCUS_TAGS]


# Dev fold 0 is fitted on the other folds, which have no "trees"; every other prediction is the
# tags with a positive among all of dev: math, graphs and trees.
FROZEN_ROWS = [row("math", "graphs")] * 2 + [row("math", "graphs", "trees")] * 12


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record):
    """Write the problems as sample files, and return the dataset folder."""
    return write_samples(
        {sid: make_raw_record(tags=tags) for sid, (_, _, tags) in PROBLEMS.items()}
    )


@pytest.fixture(name="split_file")
def split_file_fixture(tmp_path):
    """Write the frozen split of the problems, and return its path."""
    split = pd.DataFrame(
        {
            "part": [part for part, _, _ in PROBLEMS.values()],
            "fold": [fold for _, fold, _ in PROBLEMS.values()],
            "group": list(PROBLEMS),
        },
        index=pd.Index(list(PROBLEMS), name="sample_id"),
    )
    path = tmp_path / "split.csv"
    split.to_csv(path)
    return path


@pytest.fixture(name="assets")
def assets_fixture(tmp_path):
    """Return the folder of the frozen methods; nothing is frozen in it yet."""
    return tmp_path / "assets"


@pytest.fixture(name="common")
def common_fixture(dataset, split_file, assets):
    """Return the arguments both commands take."""
    return ["--data", str(dataset), "--split", str(split_file), "--assets", str(assets)]


@pytest.fixture(name="freeze_args")
def freeze_args_fixture(common, store):
    """Return the freeze command's arguments; the tree may be dirty while the tests run."""
    return [
        "freeze",
        *common,
        "--allow-dirty",
        "--tracking-uri",
        store["tracking_uri"],
        "--artifact-dir",
        str(store["artifact_dir"]),
    ]


@pytest.fixture(name="verify_args")
def verify_args_fixture(common):
    """Return the verify command's arguments."""
    return ["verify", *common]


@pytest.fixture(name="calls", autouse=True)
def calls_fixture(monkeypatch):
    """Replace the dataset check, which only the challenge dataset passes, and the MLflow log.

    Returns the arguments each was called with. The MLflow test puts the real ``log_method`` back.
    """
    calls = {"check_dataset": [], "log_method": [], "split": []}

    def log(folder, record, metrics, *, split, **store):
        calls["log_method"].append((folder, record, metrics, store))
        calls["split"].append(split)
        return "model-id"

    monkeypatch.setattr(cli, "check_dataset", calls["check_dataset"].append)
    monkeypatch.setattr(cli, "log_method", log)
    return calls


def snapshot(directory):
    """Map each file under ``directory`` to its bytes and modification time."""
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def tamper_with_the_method_folder(assets):
    """Overwrite the saved method folder's prior with a method that predicts only ``math``."""
    prior = assets / NAME / MODEL_FOLDER / PRIOR_FILE
    prior.write_text(json.dumps({"predicted": ["math"]}), encoding="utf-8")


def test_parse_args_defaults_to_the_committed_files_and_the_experiments_store():
    freeze = cli.parse_args(["freeze"])
    verify = cli.parse_args(["verify"])

    assert (freeze.command, freeze.data, freeze.split, freeze.assets) == (
        "freeze",
        None,
        SPLIT_FILE,
        METHODS_ASSETS_DIR,
    )
    assert (freeze.allow_dirty, freeze.tracking_uri, freeze.artifact_dir) == (
        False,
        MLFLOW_TRACKING_URI,
        MLFLOW_ARTIFACT_DIR,
    )
    assert (verify.command, verify.assets) == ("verify", METHODS_ASSETS_DIR)
    assert not hasattr(verify, "tracking_uri")


def test_freeze_saves_the_predictions_the_run_record_and_the_method_folder(
    freeze_args, assets, calls, capsys
):
    assert cli.main(freeze_args) == 0

    directory = assets / NAME
    assert sorted(path.name for path in directory.iterdir()) == [
        MODEL_FOLDER,
        PREDICTIONS_FILE,
        "run_record.json",
    ]  # a fitted 0/1 method freezes no scores
    assert sorted(path.name for path in (directory / MODEL_FOLDER).iterdir()) == [
        PRIOR_FILE,
        METHOD_FILE,
    ]
    assert calls["check_dataset"] == [freeze_args[2]]
    assert capsys.readouterr().out.splitlines() == [
        f"saved 14 predictions to {directory}",
        f"saved the method folder {directory / MODEL_FOLDER}",
        "logged label_prior to MLflow: model model-id",
    ]


def test_freeze_predicts_dev_out_of_fold_and_test_once(freeze_args, assets):
    cli.main(freeze_args)

    predictions = load_predictions(assets / NAME)
    assert list(predictions.columns) == list(FOCUS_TAGS)
    assert list(predictions.index) == [("dev", sid) for sid in range(1, 11)] + [
        ("test", sid) for sid in range(11, 15)
    ]
    assert predictions.to_numpy().tolist() == FROZEN_ROWS
    saved = json.loads((assets / NAME / MODEL_FOLDER / PRIOR_FILE).read_text(encoding="utf-8"))
    assert saved == {"predicted": ["math", "graphs", "trees"]}
    assert not (assets / NAME / SCORES_FILE).exists()


def test_freeze_records_the_recipe_and_the_methods_entries(freeze_args, assets, split_file):
    cli.main(freeze_args)

    record = load_run_record(assets / NAME)
    assert record.method == NAME
    assert record.split_sha256 == provenance.file_sha256(split_file)  # --split, not SPLIT_FILE
    assert record.recipe_sha256 == cli.recipe_sha256()
    assert record.allow_dirty is True
    assert record.method_entries == {
        "settings": {"min_positives": 1},
        "dev_in_sample": False,
        "supersedes": [],
    }


def test_the_recipe_hashes_the_name_the_tags_and_the_settings(monkeypatch):
    recipe = cli.recipe_sha256()

    assert recipe == provenance.sha256_json(
        {"method": "label_prior", "tags": list(FOCUS_TAGS), "settings": {"min_positives": 1}}
    )
    monkeypatch.setattr(cli, "SETTINGS", {"min_positives": 2})
    assert cli.recipe_sha256() != recipe


def test_the_frozen_files_pass_check_frozen_method(freeze_args, assets, split_file):
    cli.main(freeze_args)

    check_frozen_method(NAME, assets_dir=assets, split=load_split(split_file))


def test_freeze_logs_the_frozen_record_and_each_parts_metrics(
    freeze_args, assets, calls, store, split_file
):
    cli.main(freeze_args)

    [(folder, record, metrics, stored)] = calls["log_method"]
    assert folder == assets / NAME / MODEL_FOLDER
    assert record == load_run_record(assets / NAME)
    assert stored == store
    [split] = calls["split"]
    pd.testing.assert_frame_equal(split, load_split(split_file))
    assert list(metrics) == ["dev-oof", "test"]
    # Dev is out of fold: the only dev "trees" problem is in fold 0, which is fitted without it.
    assert metrics["dev-oof"]["f1/trees"] == 0
    assert metrics["dev-oof"]["f1/math"] == pytest.approx(2 * 0.4 / (1 + 0.4))
    assert metrics["dev-oof"]["f1/graphs"] == pytest.approx(2 * 0.2 / (1 + 0.2))
    assert metrics["dev-oof"]["macro_f1"] == pytest.approx((4 / 7 + 1 / 3) / 8)
    # Test is scored once, by the method fitted on all of dev: one positive in four problems.
    assert metrics["test"]["f1/math"] == pytest.approx(2 * 0.25 / (1 + 0.25))
    assert metrics["test"]["f1/strings"] == 0  # no "strings" in dev: never predicted
    assert metrics["test"]["macro_f1"] == pytest.approx(3 * 0.4 / 8)


def test_label_prior_freeze_twice_only_checks(freeze_args, assets, calls, capsys, monkeypatch):
    # A second freeze rewrites no frozen file and logs the frozen record.
    stamps = itertools.count(1)
    make = cli.make_run_record
    monkeypatch.setattr(
        cli,
        "make_run_record",
        lambda *args, **kwargs: dataclasses.replace(
            make(*args, **kwargs), created_at=f"2026-10-0{next(stamps)}T00:00:00+00:00"
        ),
    )
    cli.main(freeze_args)
    frozen = snapshot(assets)
    capsys.readouterr()

    assert cli.main(freeze_args) == 0

    assert snapshot(assets) == frozen
    assert load_run_record(assets / NAME).created_at == "2026-10-01T00:00:00+00:00"
    assert [record.created_at for _, record, _, _ in calls["log_method"]] == [
        "2026-10-01T00:00:00+00:00"
    ] * 2
    assert capsys.readouterr().out.splitlines()[:2] == [
        f"14 predictions match the frozen files in {assets / NAME}",
        f"the method folder {assets / NAME / MODEL_FOLDER} predicts the frozen test rows",
    ]


def test_freeze_refuses_other_predictions_of_the_same_recipe(
    freeze_args, assets, calls, dataset, make_raw_record
):
    cli.main(freeze_args)
    frozen = snapshot(assets)
    changed = make_raw_record(tags=["math", "trees"])  # problem 3 gains "trees"
    (dataset / "sample_3.json").write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="The predictions differ from the frozen predictions.csv"):
        cli.main(freeze_args)

    assert snapshot(assets) == frozen
    assert len(calls["log_method"]) == 1


def test_freeze_refuses_another_recipe(freeze_args, assets, calls, monkeypatch):
    cli.main(freeze_args)
    frozen = snapshot(assets)
    monkeypatch.setattr(cli, "SETTINGS", {"min_positives": 2})

    with pytest.raises(ValueError, match="predictions of another recipe"):
        cli.main(freeze_args)

    assert snapshot(assets) == frozen
    assert len(calls["log_method"]) == 1


def test_freeze_refuses_a_method_folder_that_predicts_other_tags(freeze_args, assets, calls):
    cli.main(freeze_args)
    tamper_with_the_method_folder(assets)

    with pytest.raises(ValueError, match="predicts other tags"):
        cli.main(freeze_args)

    assert len(calls["log_method"]) == 1


def test_freeze_checks_the_dataset_then_refuses_a_dirty_tree(
    freeze_args, assets, calls, monkeypatch
):
    dirty = {"commit": "0" * 40, "code_dirty": True}
    monkeypatch.setattr(provenance, "git_state", lambda root: dirty)
    without_allow_dirty = [argument for argument in freeze_args if argument != "--allow-dirty"]

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        cli.main(without_allow_dirty)

    assert calls["check_dataset"] == [freeze_args[2]]
    assert not assets.exists()
    assert calls["log_method"] == []


def test_freeze_names_the_committed_siblings_of_its_package_it_supersedes(
    freeze_args, assets, monkeypatch
):
    def sibling(package):
        return MethodSpec(
            module=f"codeforces_tags.methods.{package}.method", kind="fitted", requirements=()
        )

    siblings = {
        "label_prior_committed": sibling("label_prior"),
        "label_prior_new": sibling("label_prior"),
        "another_package": sibling("another"),
    }
    monkeypatch.setattr(cli, "METHODS", METHODS | siblings)
    monkeypatch.setattr(
        cli, "committed_unchanged", lambda path: path.parent.name != "label_prior_new"
    )

    cli.main(freeze_args)

    record = load_run_record(assets / NAME)
    assert record.method_entries["supersedes"] == ["label_prior_committed"]


def test_freeze_refuses_a_store_that_is_not_sqlite_before_any_work(
    common, assets, calls, tmp_path, capsys
):
    store = (tmp_path / "mlruns").as_uri()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["freeze", *common, "--allow-dirty", "--tracking-uri", store])

    assert exit_info.value.code == 2
    assert f"{store} is not a sqlite:/// store" in capsys.readouterr().err
    assert calls["check_dataset"] == []  # nothing read, fitted or written
    assert not assets.exists()


def test_freeze_mirrors_the_method_to_mlflow_once(freeze_args, assets, store, monkeypatch):
    monkeypatch.setattr(cli, "log_method", log_method)

    cli.main(freeze_args)
    cli.main(freeze_args)

    record = load_run_record(assets / NAME)
    assert method_run_id(record.recipe_sha256, **store) is not None
    [model] = mlflow.search_logged_models([use_store(**store)], output_format="list")
    metrics = mlflow.get_logged_model(model.model_id).metrics
    assert len({(metric.dataset_name, metric.key) for metric in metrics}) == len(metrics)
    test_scores = {metric.key: metric.value for metric in metrics if metric.dataset_name == "test"}
    assert test_scores["macro_f1"] == pytest.approx(3 * 0.4 / 8)
    # The metrics' datasets hold the sample ids of the --split parts they were computed on.
    split = load_split(assets.parent / "split.csv")
    run = mlflow.get_run(model.source_run_id)
    digests = {each.dataset.name: each.dataset.digest for each in run.inputs.dataset_inputs}
    assert digests == {
        name: from_pandas(
            pd.DataFrame({"sample_id": split.index[split["part"] == part]}), name=name
        ).digest
        for part, name in (("dev", "dev-oof"), ("test", "test"))
    }
    raw = read_samples(assets.parent)
    test_problems = select_part(raw, split, "test")
    logged = load_logged_method(f"models:/{model.model_id}", **store)
    predicted = logged.predict(test_problems[list(RAW_INPUT_COLUMNS)])
    assert predicted.to_numpy().tolist() == [row("math", "graphs", "trees")] * 4


def test_verify_re_derives_the_frozen_files_and_reports_the_scores(
    freeze_args, verify_args, assets, calls, capsys
):
    cli.main(freeze_args)
    capsys.readouterr()

    assert cli.main(verify_args) == 0

    assert calls["check_dataset"] == [freeze_args[2]] * 2
    assert capsys.readouterr().out.splitlines() == [
        "the frozen files of label_prior are complete and consistent with the split",
        "re-derived 14 predictions: they equal the frozen ones",
        f"the method folder {assets / NAME / MODEL_FOLDER} predicts the frozen test rows",
        "dev-oof: macro F1 0.1131, micro F1 0.3429",
        "test: macro F1 0.1500, micro F1 0.3750",
    ]


def test_verify_writes_and_logs_nothing(freeze_args, verify_args, assets, calls):
    cli.main(freeze_args)
    frozen = snapshot(assets)

    cli.main(verify_args)

    assert snapshot(assets) == frozen
    assert len(calls["log_method"]) == 1


def test_verify_refuses_frozen_files_that_are_missing(verify_args):
    with pytest.raises(ConformanceError, match="holds no method.json"):
        cli.main(verify_args)


def test_verify_refuses_another_recipe(freeze_args, verify_args, monkeypatch):
    cli.main(freeze_args)
    monkeypatch.setattr(cli, "SETTINGS", {"min_positives": 2})

    with pytest.raises(ValueError, match="another recipe"):
        cli.main(verify_args)


def test_verify_refuses_a_frozen_prediction_that_is_not_the_re_derived_one(
    freeze_args, verify_args, assets
):
    cli.main(freeze_args)
    path = assets / NAME / PREDICTIONS_FILE
    frozen = pd.read_csv(path, index_col=["part", "sample_id"])
    frozen.loc[("dev", 1), "trees"] = 1  # fold 0 is fitted without "trees"
    frozen.to_csv(path)

    with pytest.raises(ValueError, match="re-derived predictions differ"):
        cli.main(verify_args)


def test_verify_refuses_a_method_folder_that_predicts_other_tags(freeze_args, verify_args, assets):
    cli.main(freeze_args)
    tamper_with_the_method_folder(assets)

    with pytest.raises(ValueError, match="predicts other tags"):
        cli.main(verify_args)


def test_the_committed_label_prior_is_frozen_with_the_recipe_of_the_code():
    # Needs no dataset: a changed setting makes the committed recipe stale until it is frozen again.
    directory = METHODS_ASSETS_DIR / NAME

    check_frozen_method(NAME)

    record = load_run_record(directory)
    assert record.recipe_sha256 == cli.recipe_sha256()
    assert record.method_entries == {
        "settings": {"min_positives": 1},
        "dev_in_sample": False,
        "supersedes": [],
    }
    predictions = load_predictions(directory)
    assert predictions.index.get_level_values("part").value_counts().to_dict() == {
        "dev": 3988,
        "test": 994,
    }
    # Every focus tag has a positive in each dev fold, so every problem gets every tag.
    assert (predictions == 1).all().all()


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.methods.label_prior", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.methods.label_prior.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.methods.label_prior", run_name="__main__")

    assert exit_info.value.code == 0
