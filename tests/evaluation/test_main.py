"""Tests for the evaluation command, on a tiny dataset written to a temporary folder."""

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.config import SELF_IDENTIFYING_FILE, SPLIT_FILE
from codeforces_tags.evaluation import __main__ as cli
from codeforces_tags.evaluation.contamination import load_slice

# Problem 1 (dev) names its problem by URL, problem 3 (test) by a CodeforcesTask class.
CODES = {
    1: "# https://codeforces.com/contest/1228/problem/D\nprint(1)",
    2: "print(2)",
    3: "class CodeforcesTask355ASolution:\n    pass",
    4: "print(4)",
}
PARTS = {1: "dev", 2: "dev", 3: "test", 4: "test"}


@pytest.fixture(name="args")
def args_fixture(tmp_path, make_record, write_samples):
    """Write the dataset and its split; return the slices command's arguments, out last."""
    data = write_samples({sid: make_record(source_code=code) for sid, code in CODES.items()})
    split = pd.DataFrame(
        {
            "part": list(PARTS.values()),
            "fold": [1, 2, -1, -1],
            "group": list(PARTS),
        },
        index=pd.Index(list(PARTS), name="sample_id"),
    )
    split.to_csv(tmp_path / "split.csv")
    out = tmp_path / "slices" / "self_identifying_solutions.csv"
    return [
        "slices",
        "--data",
        str(data),
        "--split",
        str(tmp_path / "split.csv"),
        "--out",
        str(out),
    ]


@pytest.fixture(name="checked")
def checked_fixture(monkeypatch):
    """Replace the dataset check, which only the challenge dataset passes; return its calls."""
    calls = []
    monkeypatch.setattr(cli, "check_dataset", calls.append)
    return calls


def test_parse_args_defaults_to_the_frozen_split_and_slice():
    args = cli.parse_args(["slices"])

    assert (args.command, args.data, args.split, args.out) == (
        "slices",
        None,
        SPLIT_FILE,
        SELF_IDENTIFYING_FILE,
    )


def test_slices_checks_the_dataset_then_freezes_the_slice(args, checked, capsys):
    assert cli.main(args) == 0

    assert checked == [args[2]]
    slice_ = load_slice(args[-1])
    assert slice_.to_dict("index") == {
        1: {"part": "dev", "rules": "codeforces_url"},
        3: {"part": "test", "rules": "task_class"},
    }
    assert capsys.readouterr().out.splitlines() == [
        "saved 2 self-identifying problems to self_identifying_solutions.csv",
        "2 problems (1 dev, 1 test); by rule: codeforces_url 1, task_class 1, "
        "round_or_id_header 0, cf_problem_id 0, quotes_description 0",
    ]


def test_slices_then_only_checks_the_frozen_slice(args, checked, capsys, make_record):
    cli.main(args)
    capsys.readouterr()

    assert cli.main(args) == 0

    assert capsys.readouterr().out.splitlines()[0] == (
        "self_identifying_solutions.csv matches the slice (2 problems)"
    )
    sample = Path(args[2]) / "sample_2.json"
    sample.write_text(json.dumps(make_record(source_code="# cf 1228 D")), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer gives the frozen slice"):
        cli.main(args)
    assert len(checked) == 3


def test_slices_refuses_another_dataset(args):
    with pytest.raises(ValueError, match="The dataset has 4 problem files"):
        cli.main(args)

    assert not Path(args[-1]).exists()


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.evaluation", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.evaluation.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.evaluation", run_name="__main__")

    assert exit_info.value.code == 0
