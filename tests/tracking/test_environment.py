"""Tests for MLflow's environment and the choice of store."""

import argparse
import ast
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import mlflow
import pytest
from mlflow import environment_variables as variables

import codeforces_tags
from codeforces_tags.config import (
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    MLFLOW_IMPORT_ENV,
    MLFLOW_METHODS_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)
from codeforces_tags.methods import METHODS, installed_methods
from codeforces_tags.tracking.environment import add_store_arguments, mlflow_env, use_store

PACKAGE = Path(codeforces_tags.__file__).parent
# Each variable mlflow_env() sets, and the value MLflow then reads from it.
READ_AS = (
    (variables.MLFLOW_ALLOW_PICKLE_DESERIALIZATION, False),
    (variables.MLFLOW_UV_AUTO_DETECT, False),
    (variables.MLFLOW_LOG_UV_FILES, False),
    (variables.MLFLOW_GENAI_EVAL_MAX_WORKERS, 1),
    (variables.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING, False),
    (variables.MLFLOW_MODEL_CATALOG_URI, ""),
    (variables.MLFLOW_DISABLE_AGENT_HINT, True),
    (variables.MLFLOW_SERVER_ENABLE_JOB_EXECUTION, False),
    (variables.MLFLOW_DISABLE_TELEMETRY, True),
)
# The method packages whose methods' dependency groups this environment lacks, e.g. the
# ModernBERT methods' finetuning group in the default job: their commands cannot be imported
# here, and their own job checks them (tests/methods/modernbert/test_main.py; the f2llm job
# runs this file's fresh-import test with its group installed).
UNINSTALLED_PACKAGES = {
    METHODS[name].module.rpartition(".")[0] for name in METHODS.keys() - installed_methods().keys()
}
# Every command of the package that this environment can import, as python -m imports it in a
# fresh interpreter.
COMMANDS = sorted(
    command
    for command in (
        ".".join((PACKAGE.name, *path.relative_to(PACKAGE).with_suffix("").parts))
        for path in PACKAGE.rglob("__main__.py")
    )
    if command.removesuffix(".__main__") not in UNINSTALLED_PACKAGES
)
# Imports mlflow, then prints every environment variable it read meanwhile, as JSON.
READS_ON_IMPORT = """
import json, os
read = set()
class Recorded(type(os.environ)):
    def __getitem__(self, name):
        read.add(name)
        return super().__getitem__(name)
os.environ.__class__ = Recorded
import mlflow
print(json.dumps(sorted(name for name in read if isinstance(name, str))))
"""
# The methods of os.environ that change it.
ENVIRON_WRITES = {"__delitem__", "__setitem__", "clear", "pop", "popitem", "setdefault", "update"}


def is_os(node, name):
    """Tell whether ``node`` is ``os.<name>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def may_be_mlflow(key):
    """Tell whether an environment key may name an MLflow variable: it is not a str literal.

    A str literal names one when it starts with ``MLFLOW_``.
    """
    literal = isinstance(key, ast.Constant) and isinstance(key.value, str)
    return not literal or key.value.startswith("MLFLOW_")


def sets_mlflow_environment(node):
    """Tell whether an AST node writes, or may write, an MLflow variable of the environment."""
    if isinstance(node, ast.Subscript):
        written = is_os(node.value, "environ") and not isinstance(node.ctx, ast.Load)
        return written and may_be_mlflow(node.slice)
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    environ_write = isinstance(function, ast.Attribute) and (
        (is_os(function.value, "environ") and function.attr in ENVIRON_WRITES)
        or is_os(function, "putenv")
        or is_os(function, "unsetenv")
    )
    return environ_write and (not node.args or may_be_mlflow(node.args[0]))


def mlflow_environment_writers(package):
    """Return the modules of a package that set an MLflow variable, relative to the package."""
    return {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if any(map(sets_mlflow_environment, ast.walk(ast.parse(path.read_text(encoding="utf-8")))))
    }


def journal_mode(tracking_uri):
    """Return the sqlite journal mode of a ``sqlite:///`` store."""
    with closing(sqlite3.connect(tracking_uri.removeprefix("sqlite:///"))) as connection:
        return connection.execute("PRAGMA journal_mode").fetchone()[0]


def test_mlflow_env_sets_every_variable_whatever_the_environment_says(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "10")

    mlflow_env()

    assert {name: os.environ[name] for name in MLFLOW_ENV} == MLFLOW_ENV


def test_mlflow_reads_each_variable_as_intended():
    mlflow_env()

    assert [(variable.name, variable.get()) for variable, _ in READ_AS] == [
        (variable.name, value) for variable, value in READ_AS
    ]
    assert {variable.name for variable, _ in READ_AS} == {*MLFLOW_ENV, *MLFLOW_ENV_DEFAULTS}


def test_mlflow_env_turns_telemetry_off_unless_the_environment_sets_it(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")

    mlflow_env()

    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "false"


@pytest.mark.parametrize("names", [MLFLOW_IMPORT_ENV, ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION",)])
def test_mlflow_env_sets_only_the_variables_it_is_given(names):
    mlflow_env(names)

    values = MLFLOW_ENV_DEFAULTS | MLFLOW_ENV
    assert {name: os.environ[name] for name in values if name in os.environ} == {
        name: values[name] for name in names
    }


def test_mlflow_import_env_turns_telemetry_off_unless_the_environment_sets_it(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")

    mlflow_env(MLFLOW_IMPORT_ENV)

    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "false"


def test_mlflow_import_env_names_the_variables_mlflow_reads_when_it_is_imported(agent_python):
    output, _ = agent_python(READS_ON_IMPORT)

    read = set(json.loads(output))

    assert read & {*MLFLOW_ENV, *MLFLOW_ENV_DEFAULTS} == set(MLFLOW_IMPORT_ENV)


def test_a_bare_import_of_mlflow_starts_telemetry_and_hints_in_the_agent_shell(fresh_import):
    # The control of every fresh-import check (the next test, and the Jev commands' in
    # tests/methods/typesafe_jev/test_search_commands.py): there, only the package can turn both
    # off.
    assert fresh_import("mlflow") == {"telemetry": True, "agent_hint": True}


@pytest.mark.parametrize("command", COMMANDS)
def test_no_command_lets_mlflow_start_telemetry_or_hint(command, fresh_import):
    assert fresh_import(command) == {"telemetry": False, "agent_hint": False}


def test_use_store_creates_the_experiment_once_with_its_artifacts_in_the_given_folder(store):
    experiment_id = use_store(**store)

    assert use_store(**store) == experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == MLFLOW_METHODS_EXPERIMENT
    assert experiment.artifact_location == store["artifact_dir"].resolve().as_uri()
    assert store["artifact_dir"].is_dir()
    assert mlflow.get_tracking_uri() == store["tracking_uri"]


def test_use_store_takes_any_experiment_name(store):
    experiment_id = use_store(experiment="label_prior-training", **store)

    assert mlflow.get_experiment(experiment_id).name == "label_prior-training"
    assert mlflow.get_experiment_by_name(MLFLOW_METHODS_EXPERIMENT) is None


def test_use_store_sets_mlflow_env(store, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")

    use_store(**store)

    assert os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] == "false"


def test_use_store_switches_a_sqlite_store_to_wal_once(store):
    assert journal_mode(store["tracking_uri"]) == "delete"

    use_store(**store)
    assert journal_mode(store["tracking_uri"]) == "wal"
    use_store(**store)

    assert journal_mode(store["tracking_uri"]) == "wal"


def test_use_store_refuses_a_store_that_is_not_sqlite(tmp_path):
    with pytest.raises(ValueError, match="sqlite:///"):
        use_store("http://127.0.0.1:5000", artifact_dir=tmp_path / "artifacts")

    assert not (tmp_path / "artifacts").exists()


def test_use_store_refuses_a_deleted_experiment(store):
    mlflow.delete_experiment(use_store(**store))

    with pytest.raises(ValueError, match="deleted"):
        use_store(**store)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('os.environ["MLFLOW_TRACKING_URI"] = uri', True),
        ('del os.environ["MLFLOW_TRACKING_URI"]', True),
        ("os.environ[name] = value", True),
        ('os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")', True),
        ("os.environ.update(SETTINGS)", True),
        ("os.environ.clear()", True),
        ('os.putenv("MLFLOW_TRACKING_URI", uri)', True),
        ('os.environ["HF_HUB_OFFLINE"] = "1"', False),
        ('os.environ.setdefault("HF_HUB_OFFLINE", "1")', False),
        ('uri = os.environ["MLFLOW_TRACKING_URI"]', False),
        ('uri = os.environ.get("MLFLOW_TRACKING_URI")', False),
        ("settings.update(SETTINGS)", False),
        ("print(uri)", False),
    ],
)
def test_sets_mlflow_environment_finds_every_kind_of_write(source, expected):
    assert any(map(sets_mlflow_environment, ast.walk(ast.parse(source)))) is expected


def test_only_mlflow_env_sets_mlflow_environment():
    # Safeguard 7 (docs/adr/0002): one place sets MLflow's environment.
    assert mlflow_environment_writers(PACKAGE) == {"tracking/environment.py"}


def test_add_store_arguments_adds_the_stores_options_with_the_configs_defaults():
    parser = argparse.ArgumentParser()
    add_store_arguments(parser)

    defaults = parser.parse_args([])
    chosen = parser.parse_args(["--tracking-uri", "sqlite:///other.db", "--artifact-dir", "out"])

    assert (defaults.tracking_uri, defaults.artifact_dir) == (
        MLFLOW_TRACKING_URI,
        MLFLOW_ARTIFACT_DIR,
    )
    assert (chosen.tracking_uri, chosen.artifact_dir) == ("sqlite:///other.db", Path("out"))


def test_add_store_arguments_refuses_a_store_that_is_not_sqlite_with_use_stores_message(capsys):
    # The parser refuses it, so a command finds out before it does any work, not when it logs.
    parser = argparse.ArgumentParser()
    add_store_arguments(parser)

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--tracking-uri", "http://127.0.0.1:5000"])

    assert exit_info.value.code == 2
    with pytest.raises(ValueError) as refused:
        use_store("http://127.0.0.1:5000")
    assert f"argument --tracking-uri: {refused.value}" in capsys.readouterr().err
