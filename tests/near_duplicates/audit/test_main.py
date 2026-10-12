"""Tests for the audit's command line."""

import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.near_duplicates.audit import __main__ as cli


def test_parse_args_reads_each_command():
    judge = cli.parse_args(["judge", "first_pass", "out", "--dry-run"])

    assert (judge.command, judge.pass_name, judge.folder, judge.dry_run) == (
        "judge", "first_pass", Path("out"), True,
    )  # fmt: skip
    assert cli.parse_args(["merge", "out"]).command == "merge"


def test_parse_args_rejects_an_unknown_pass():
    with pytest.raises(SystemExit):
        cli.parse_args(["judge", "third_pass", "out"])


def test_main_runs_the_requested_step(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_dataset", lambda: "dataset")
    monkeypatch.setattr(
        cli, "build_pairs", lambda df, folder, recall_check: calls.append((df, recall_check)) or [1]
    )
    monkeypatch.setattr(
        cli, "run_pass", lambda name, folder, dry_run: calls.append(dry_run) or ["p"]
    )
    monkeypatch.setattr(cli, "merge", lambda folder: pd.DataFrame({"final": ["duplicate"]}))

    cli.main(["build-pairs", "out", "--recall-check"])
    cli.main(["judge", "first_pass", "out", "--dry-run"])
    cli.main(["merge", "out"])

    assert calls == [("dataset", True), True]
    assert capsys.readouterr().out.splitlines() == [
        "1 pairs to judge in out",
        "first_pass: 1 prompts written in out",
        "1 duplicates in out",
    ]


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.near_duplicates.audit", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.near_duplicates.audit.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.near_duplicates.audit", run_name="__main__")

    assert exit_info.value.code == 0
