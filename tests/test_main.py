"""Tests for ``python -m codeforces_tags`` and the ``codeforces-tags`` console script."""

import runpy
import sys
import tomllib

import pytest

from codeforces_tags import __main__ as entry_point
from codeforces_tags import cli
from codeforces_tags.config import DATA_DIR_ENV_VAR, PYPROJECT_FILE


def test_python_dash_m_codeforces_tags_runs_the_command_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags", "methods"])
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags", run_name="__main__")

    assert exit_info.value.code == 0
    assert "tfidf" in capsys.readouterr().out


def test_python_dash_m_codeforces_tags_dot_cli_runs_the_command_line_too(
    monkeypatch, tmp_path, capsys
):
    # PROBLEM.md sets the goal of a command line in one Python module: cli.py is that module.
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.cli", "methods"])
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.delitem(sys.modules, "codeforces_tags.cli")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.cli", run_name="__main__")

    assert exit_info.value.code == 0
    assert "tfidf" in capsys.readouterr().out


def test_the_module_and_the_console_script_run_the_same_main():
    scripts = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts == {cli.PROG: "codeforces_tags.cli:main"}
    assert entry_point.main is cli.main
