"""Tests for the tracking command line."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from codeforces_tags import methods
from codeforces_tags.config import (
    DATASET_SHA256,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    SPLIT_FILE,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.tracking import __main__ as cli
from codeforces_tags.tracking import calibration
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import log_method


def test_parse_args_defaults_to_the_frozen_split_and_the_experiments_store():
    args = cli.parse_args(["latency", "--method", "label_prior"])

    assert (args.command, args.method, args.data, args.split, args.max_cost_usd) == (
        "latency",
        "label_prior",
        None,
        SPLIT_FILE,
        None,
    )
    assert (args.tracking_uri, args.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)


def test_the_latency_command_runs_the_harness_and_prints_its_line(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_run_latency(name, **kwargs):
        calls.append((name, kwargs))
        return "timed it"

    monkeypatch.setattr(cli, "run_latency", fake_run_latency)

    status = cli.main(
        [
            "latency",
            "--method",
            "label_prior",
            "--data",
            "dataset",
            "--split",
            "split.csv",
            "--max-cost-usd",
            "0.25",
            "--tracking-uri",
            "sqlite:///store.db",
            "--artifact-dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "timed it\n"
    assert calls == [
        (
            "label_prior",
            {
                "max_cost_usd": 0.25,
                "data_dir": "dataset",
                "split_path": Path("split.csv"),
                "tracking_uri": "sqlite:///store.db",
                "artifact_dir": tmp_path,
            },
        )
    ]


def test_the_latency_command_needs_a_method(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["latency"])

    assert exit_info.value.code == 2
    assert "--method" in capsys.readouterr().err


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.tracking", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.tracking.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.tracking", run_name="__main__")

    assert exit_info.value.code == 0


def test_the_command_line_imports_calibrate_and_the_libraries_it_needs_only_when_it_runs():
    # tracking latency times the loading of a method in this process: with the tag model, skops,
    # torch and matplotlib imported before the timer starts, load_s would leave out their import.
    # A fresh interpreter, so that the imports of other tests do not count.
    heavy = ("codeforces_tags.tracking.calibration", "skops", "torch", "matplotlib")
    code = (
        "import sys; import codeforces_tags.tracking.__main__; "
        f"print(sorted(set({heavy!r}) & set(sys.modules)))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


def test_parse_args_of_datasets_and_evaluate_default_to_the_frozen_split_and_every_method():
    datasets = cli.parse_args(["datasets"])
    evaluate = cli.parse_args(["evaluate", "--part", "test"])

    assert (datasets.command, datasets.data, datasets.split, datasets.tracking_uri) == (
        "datasets",
        None,
        SPLIT_FILE,
        MLFLOW_TRACKING_URI,
    )
    assert (evaluate.command, evaluate.part, evaluate.method, evaluate.artifact_dir) == (
        "evaluate",
        "test",
        None,
        MLFLOW_ARTIFACT_DIR,
    )


def test_the_evaluate_command_takes_a_part_of_the_split_only(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["evaluate", "--part", "train"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'train'" in capsys.readouterr().err


def test_the_evaluate_command_takes_a_registered_method_only(capsys):
    # A path-like name would read the frozen files of another folder (or of an alias).
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["evaluate", "--part", "test", "--method", "label_prior/"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'label_prior/'" in capsys.readouterr().err


def test_the_discard_command_discards_the_incomplete_run(monkeypatch, capsys):
    calls = []

    def fake_discard(run_id, **store):
        calls.append((run_id, store))
        return 3

    monkeypatch.setattr(cli, "discard_evaluation_run", fake_discard)

    status = cli.main(["discard", "--run", "abc123", "--tracking-uri", "sqlite:///s.db"])

    assert status == 0
    assert calls == [
        ("abc123", {"tracking_uri": "sqlite:///s.db", "artifact_dir": MLFLOW_ARTIFACT_DIR})
    ]
    assert capsys.readouterr().out == "discarded the evaluation run abc123 and its 3 traces\n"


def test_the_datasets_command_checks_the_dataset_then_makes_the_datasets(monkeypatch, capsys):
    calls = []
    split = pd.DataFrame({"part": ["dev", "dev", "dev", "test", "test"]})
    monkeypatch.setattr(cli, "check_dataset", lambda data: calls.append(("check", data)))
    monkeypatch.setattr(cli, "load_dataset", lambda data: f"problems of {data}")
    monkeypatch.setattr(cli, "load_split", lambda path: split)
    monkeypatch.setattr(cli, "file_sha256", lambda path: f"sha256 of {path}")

    def fake_ensure_datasets(problems, split_, **kwargs):
        calls.append(("ensure", problems, split_ is split, kwargs))
        return {
            part: SimpleNamespace(name=f"codeforces-{part}", tags={"content_sha256": f"{part}-c"})
            for part in ("dev", "test")
        }

    monkeypatch.setattr(cli, "ensure_datasets", fake_ensure_datasets)

    status = cli.main(
        [
            "datasets",
            "--data",
            "dataset",
            "--split",
            "split.csv",
            "--tracking-uri",
            "sqlite:///s.db",
        ]
    )

    assert status == 0
    assert calls == [
        ("check", "dataset"),
        (
            "ensure",
            "problems of dataset",
            True,
            {
                "dataset_sha256": DATASET_SHA256,
                "split_sha256": "sha256 of split.csv",
                "tracking_uri": "sqlite:///s.db",
                "artifact_dir": MLFLOW_ARTIFACT_DIR,
            },
        ),
    ]
    assert capsys.readouterr().out == (
        "codeforces-dev: 3 records, content_sha256 dev-c\n"
        "codeforces-test: 2 records, content_sha256 test-c\n"
    )


def fake_evaluate_method(calls):
    """Return a stand-in for evaluate_method that records its calls in ``calls``."""

    def evaluate_method(name, part, **store):
        calls.append((name, part, store))
        return f"run-of-{name}"

    return evaluate_method


def test_the_evaluate_command_evaluates_the_method_on_the_part(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "evaluate_method", fake_evaluate_method(calls))

    status = cli.main(["evaluate", "--part", "dev", "--method", "label_prior"])

    assert status == 0
    store = {"tracking_uri": MLFLOW_TRACKING_URI, "artifact_dir": MLFLOW_ARTIFACT_DIR}
    assert calls == [("label_prior", "dev", store)]
    assert (
        capsys.readouterr().out == "eval-label_prior-dev: run run-of-label_prior, traces checked\n"
    )


def test_the_evaluate_command_evaluates_every_frozen_registered_method(
    monkeypatch, capsys, tmp_path
):
    # The check of the store is stubbed: the run records hold {}, and the default store is the
    # repository's, which no test opens.
    calls, checked = [], []
    monkeypatch.setattr(cli, "evaluate_method", fake_evaluate_method(calls))

    def every_method_logged(names, **kwargs):
        checked.append((names, kwargs))
        return []

    monkeypatch.setattr(cli, "unlogged_methods", every_method_logged)
    registry = dict.fromkeys(["zeta", "never_frozen", "alpha"])
    monkeypatch.setattr(cli, "METHODS", registry)
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path)
    for name in ("zeta", "alpha", "unregistered"):
        (tmp_path / name).mkdir()
        (tmp_path / name / RUN_RECORD_FILE).write_text("{}", encoding="utf-8")

    cli.main(["evaluate", "--part", "test"])

    store = {"tracking_uri": MLFLOW_TRACKING_URI, "artifact_dir": MLFLOW_ARTIFACT_DIR}
    assert checked == [(["alpha", "zeta"], {"registry": registry, **store})]
    assert [(name, part) for name, part, _ in calls] == [("alpha", "test"), ("zeta", "test")]
    assert capsys.readouterr().out.splitlines() == [
        "eval-alpha-test: run run-of-alpha, traces checked",
        "eval-zeta-test: run run-of-zeta, traces checked",
    ]


@pytest.fixture(name="frozen_fakes")
def frozen_fakes_fixture(registry, folders, placeholder_record, monkeypatch, tmp_path):
    """Freeze the two fakes as their run records alone, in the command line's registry.

    Before its first run, a bare evaluate reads only the run records and the store. Returns a
    namespace whose ``log(name, store)`` logs a fake to a store as its freeze does, with no
    metrics, so no split is read.
    """
    monkeypatch.setattr(cli, "METHODS", registry)
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path / "assets")
    records = {name: placeholder_record(name) for name in registry}
    for name, record in records.items():
        (tmp_path / "assets" / name).mkdir(parents=True)
        text = json.dumps(record.to_dict())
        (tmp_path / "assets" / name / RUN_RECORD_FILE).write_text(text, encoding="utf-8")

    def log(name, store):
        split = pd.DataFrame()
        log_method(folders[name], records[name], {}, split=split, registry=registry, **store)

    return SimpleNamespace(log=log)


def store_options(store):
    """Return the command line options that name a store."""
    return ["--tracking-uri", store["tracking_uri"], "--artifact-dir", str(store["artifact_dir"])]


def test_evaluate_without_a_method_refuses_before_any_run_when_a_frozen_method_was_never_logged(
    frozen_fakes, store, monkeypatch
):
    # Without the check, the loop would evaluate code_rule_fake, then stop at keyword_fake.
    calls = []
    monkeypatch.setattr(cli, "evaluate_method", fake_evaluate_method(calls))
    evaluate_dev = ["evaluate", "--part", "dev", *store_options(store)]

    with pytest.raises(ValueError, match=r"for \['code_rule_fake', 'keyword_fake'\]: log each"):
        cli.main(evaluate_dev)
    frozen_fakes.log("code_rule_fake", store)
    with pytest.raises(ValueError, match=r"No logged model in the store for \['keyword_fake'\]"):
        cli.main(evaluate_dev)

    assert not calls
    runs = mlflow.search_runs([use_store(**store)], output_format="list")
    assert [run.info.run_name for run in runs] == ["code_rule_fake"]


def test_evaluate_without_a_method_evaluates_every_frozen_method_once_each_is_logged(
    frozen_fakes, store, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(cli, "evaluate_method", fake_evaluate_method(calls))
    for name in ("keyword_fake", "code_rule_fake"):
        frozen_fakes.log(name, store)

    assert cli.main(["evaluate", "--part", "dev", *store_options(store)]) == 0

    assert [(name, part) for name, part, _ in calls] == [
        ("code_rule_fake", "dev"),
        ("keyword_fake", "dev"),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "eval-code_rule_fake-dev: run run-of-code_rule_fake, traces checked",
        "eval-keyword_fake-dev: run run-of-keyword_fake, traces checked",
    ]


def test_evaluate_with_a_method_keeps_evaluate_methods_refusal_of_an_unlogged_method(store):
    # label_prior's committed run record, and a store with no method run: evaluate_method's
    # message names the recipe, where the bare command's would name the method.
    with pytest.raises(ValueError, match="No method run of the recipe .* run the method's freeze"):
        cli.main(["evaluate", "--part", "dev", "--method", "label_prior", *store_options(store)])

    assert mlflow.search_runs([use_store(**store)], output_format="list") == []


def test_parse_args_of_calibrate_default_to_the_frozen_split_and_the_experiments_store():
    args = cli.parse_args(["calibrate"])

    assert (args.command, args.data, args.split) == ("calibrate", None, SPLIT_FILE)
    assert (args.tracking_uri, args.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)


def test_the_calibrate_command_runs_the_calibration_and_prints_its_line(
    monkeypatch, capsys, tmp_path
):
    calls = []

    def fake_run_calibration(**kwargs):
        calls.append(kwargs)
        return "calibrated it"

    monkeypatch.setattr(calibration, "run_calibration", fake_run_calibration)

    status = cli.main(
        [
            "calibrate",
            "--data",
            "dataset",
            "--split",
            "split.csv",
            "--tracking-uri",
            "sqlite:///store.db",
            "--artifact-dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "calibrated it\n"
    assert calls == [
        {
            "data_dir": "dataset",
            "split_path": Path("split.csv"),
            "tracking_uri": "sqlite:///store.db",
            "artifact_dir": tmp_path,
        }
    ]
