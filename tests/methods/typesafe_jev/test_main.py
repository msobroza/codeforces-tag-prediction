"""Tests for the commands of the TypeSafe Jev methods, on a tiny dataset and a fake Jev.

The four commands share one module and these fixtures, so their tests are one file, longer than
pylint's limit on a module.

No test sends a request: the ``jev_api`` fixture (conftest.py) puts a fake Jev in place of
``client.jev_session``, and it answers from the words of each state. git is faked too
(``FakeGit``), since the split, the slice and the answers files live under ``tmp_path``, and so
is the dataset check, which only the challenge's dataset passes. ``REAL_SESSION`` is the real
``client.jev_session``: the tests that prove a command needs no SDK and no key put it back.
"""

# pylint: disable=too-many-lines

import asyncio
import contextlib
import dataclasses
import json
import math
import re
import runpy
import shutil
import sys
import types
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from codeforces_tags import methods, provenance
from codeforces_tags.config import FOCUS_TAGS, MLFLOW_ARTIFACT_DIR, MLFLOW_TRACKING_URI
from codeforces_tags.conformance import ConformanceError, check_frozen_method
from codeforces_tags.data.loading import load_dataset, read_samples
from codeforces_tags.evaluation.contamination import freeze_slice, self_identifying_solutions
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    load_per_problem,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)
from codeforces_tags.evaluation.timing import LATENCY_FILE, Latency, save_latency
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.store import answers_file, read_answers
from codeforces_tags.methods import METHODS, MODEL_FOLDER, load_method, method_dir, save_method
from codeforces_tags.methods.typesafe_jev import __main__ as cli
from codeforces_tags.methods.typesafe_jev import (
    client,
    platt,
    platt_commands,
    questions,
    runner,
    search_commands,
)
from codeforces_tags.methods.typesafe_jev import words as jev_words
from codeforces_tags.methods.typesafe_jev.answers import probabilities
from codeforces_tags.methods.typesafe_jev.client import jev_session as REAL_SESSION
from codeforces_tags.methods.typesafe_jev.method import (
    JevProbabilities,
    JevTagger,
    fitted_thresholds,
    load,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    FROZEN_NAMES,
    GEPA_NAME,
    GEPA_SEED_NAME,
    MAX_COST_USD,
    MODEL,
    NAMES,
    PARAMS_FILE,
    PLATT_BASES,
    PLATT_FILES,
    PROMPT,
    PROMPT_NAME,
    QUESTION,
    SETTINGS,
    TAG_DEFINITIONS,
    TEST_INFORMED,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.tracking import link_jev_prompt, register_jev_prompt
from codeforces_tags.methods.typesafe_jev.tuning import SearchOutcome
from codeforces_tags.methods.typesafe_jev.words import method_settings
from codeforces_tags.model.platt_lipton import (
    RULE_SETTINGS,
    PlattLiptonThresholds,
    lipton_thresholds,
)
from codeforces_tags.model.thresholds import (
    apply_thresholds,
    cross_fitted_predictions,
    per_tag_thresholds,
)
from codeforces_tags.provenance import file_sha256, json_safe, sha256_json
from codeforces_tags.splitting import PARTS, load_split, select_part
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import log_method, logged_model_id, method_run_id
from codeforces_tags.tracking.prompts import LINKED_PROMPTS_TAG

# sample_id -> (part, fold, tags, description, solution); problem 6's solution names its
# Codeforces problem (the self-identifying slice); no focus tag is positive on every dev problem;
# geometry and probabilities have no dev positive (inf thresholds); statements are distinct.
# Problem 1's solution names the vertices, so a twin's Platt map never reads equal scores of both
# classes (graphs, once fold 1 is held out): its fit would have no unique maximum, and
# newton-cholesky would warn of a singular Hessian.
PROBLEMS = {
    1: ("dev", 0, ["trees", "graphs"], "Count the leaves of the tree.", "print(leaves, vertices)"),
    2: ("dev", 0, ["strings"], "Reverse the letters.", "print(input()[::-1])"),
    3: ("dev", 1, ["math", "number theory"], "Print the gcd of a and b.", "print(gcd(a, b))"),
    4: ("dev", 1, ["graphs"], "Link the vertices.", "print(n - 1)"),
    5: ("dev", 2, ["math"], "Print the sum.", "print(sum(a))"),
    6: ("dev", 2, [], "Sort the array.", "# codeforces.com/contest/6/problem/A\nprint(sorted(a))"),
    7: ("dev", 3, ["strings", "math"], "Count the letters, then their sum.", "print(len(s))"),
    8: ("dev", 3, ["trees"], "Remove the leaves.", "print(0)"),
    9: ("dev", 4, ["games"], "Who wins the game?", "print('Alice')"),
    10: ("dev", 4, ["math"], "Print the sum of two numbers.", "print(a + b)"),
    11: ("test", -1, ["number theory"], "Print the gcd of three numbers.", "print(g)"),
    12: ("test", -1, ["graphs", "trees"], "Count the vertices and the leaves.", "print(n, k)"),
    13: ("test", -1, ["strings"], "Print the letters twice.", "print(s * 2)"),
}
DEV = [sample_id for sample_id, problem in PROBLEMS.items() if problem[0] == "dev"]
TEST = [sample_id for sample_id, problem in PROBLEMS.items() if problem[0] == "test"]
# What the fake prompt registry answers: the version of jev-tags that the methods read.
FAKE_PROMPT = types.SimpleNamespace(uri="prompts:/jev-tags/1")


class FakeGit:
    """Stands for git in the commands: each commit keeps the bytes of the files it was given.

    Attributes:
        commits: One mapping per commit, oldest first, from each file it changed to its bytes;
            a test rewrites it to stand for a rewritten history (e.g. an amended commit).
    """

    def __init__(self):
        """Start with no commit."""
        self.commits = []

    def commit(self, *paths):
        """Record one commit that holds the given files as they are on disk now."""
        self.commits.append({Path(path): Path(path).read_bytes() for path in paths})

    def committed_versions(self, path):
        """Return ``(sha, bytes)`` of a file at each commit that holds it, oldest first."""
        return [
            (f"{number:040x}", files[Path(path)])
            for number, files in enumerate(self.commits, start=1)
            if Path(path) in files
        ]

    def committed_unchanged(self, path):
        """Tell whether a file is committed, and on disk with the bytes of its last commit."""
        versions = self.committed_versions(path)
        if not versions or not Path(path).is_file():
            return False
        return Path(path).read_bytes() == versions[-1][1]

    def last_commit(self, path):
        """Return the sha of the last commit that holds a file."""
        return self.committed_versions(path)[-1][0]


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(monkeypatch, tmp_path):
    """Keep the frozen files of the methods, answers files included, in a folder of tmp_path."""
    folder = tmp_path / "assets"
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", folder)
    return folder


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record, monkeypatch):
    """Write the problems as sample files where read_samples() looks; return their folder."""
    folder = write_samples(
        {
            sample_id: make_raw_record(prob_desc_description=text, tags=tags, source_code=code)
            for sample_id, (_, _, tags, text, code) in PROBLEMS.items()
        }
    )
    monkeypatch.setenv("CODEFORCES_DATA_DIR", str(folder))
    return folder


@pytest.fixture(name="split_file")
def split_file_fixture(monkeypatch, tmp_path):
    """Write the committed split, each problem its own group, and point the commands at it."""
    rows = [f"{sid},{part},{fold},{sid}" for sid, (part, fold, *_) in PROBLEMS.items()]
    path = tmp_path / "split.csv"
    path.write_text("\n".join(["sample_id,part,fold,group", *rows]) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "SPLIT_FILE", path)
    return path


@pytest.fixture(name="slice_file")
def slice_file_fixture(monkeypatch, tmp_path, dataset, split_file):
    """Freeze the self-identifying slice (problem 6), and point the commands at it."""
    found = self_identifying_solutions(load_dataset(dataset), load_split(split_file))
    path = tmp_path / "self_identifying_solutions.csv"
    freeze_slice(found, path)
    monkeypatch.setattr(cli, "SELF_IDENTIFYING_FILE", path)
    return path


@pytest.fixture(name="git", autouse=True)
def git_fixture(monkeypatch, split_file, slice_file):
    """Fake git in the commands, with the split and the slice committed; return the fake.

    A calibrated twin's gates (platt_commands) read git through the same fake.
    """
    git = FakeGit()
    for module in (cli, platt_commands):
        for name in ("committed_versions", "committed_unchanged", "last_commit"):
            monkeypatch.setattr(module, name, getattr(git, name))
    git.commit(split_file, slice_file)
    return git


@pytest.fixture(name="events", autouse=True)
def events_fixture(monkeypatch, api):
    """Record the dataset check, the MLflow log and the prompt registry instead of running them.

    They are in order; the record of a registration holds how many requests were sent before it.
    The MLflow tests of run and freeze put the real ``log_method``, ``register_jev_prompt`` and
    ``link_jev_prompt`` back. Returns the records.
    """
    events = []
    monkeypatch.setattr(cli, "check_dataset", lambda: events.append(("check_dataset",)))

    def log(folder, record, metrics, *, split, registry, **_store):
        events.append(("log_method", folder, record, metrics, split, registry))
        return "model-id"

    def register(**_store):
        events.append(("register_prompt", len(api.sent)))
        return FAKE_PROMPT

    def link(model_id, **_store):
        events.append(("link_prompt", model_id))
        return FAKE_PROMPT

    monkeypatch.setattr(cli, "log_method", log)
    monkeypatch.setattr(cli, "register_jev_prompt", register)
    monkeypatch.setattr(cli, "link_jev_prompt", link)
    return events


@pytest.fixture(name="api", autouse=True)
def api_fixture(jev_api):
    """Answer every command's requests through the fake Jev; return it."""
    return jev_api


@pytest.fixture(name="jev")
def jev_fixture(store):
    """Return a runner of one command on one method, as ``main(command, method, *options)``.

    run and freeze also get --allow-dirty (the tree may be dirty while the tests run) and the
    test's MLflow store.
    """
    tracking = ["--tracking-uri", store["tracking_uri"]]
    artifacts = ["--artifact-dir", str(store["artifact_dir"])]
    added = {command: ["--allow-dirty", *tracking, *artifacts] for command in ("run", "freeze")}

    def main(command, method, *options):
        return cli.main([command, "--method", method, *options, *added.get(command, [])])

    return main


def answer_every_part(jev, git):
    """Answer each part of both frozen methods, dev first, committing each part's answers files."""
    for part in PARTS:
        for name in FROZEN_NAMES:
            assert jev("run", name, "--part", part) == 0
        git.commit(*(answers_file(name, part) for name in FROZEN_NAMES))


# The words of the searched method in these tests: the seed's, with another convention.
GEPA_WORDS = dataclasses.replace(
    jev_words.SEED_WORDS, convention="Codeforces tags name the ideas of a problem's solution."
)


def write_params(words=GEPA_WORDS, **changes):
    """Write the searched method's tuning/params.json, as export does; return its path.

    It holds the chosen words and a few of export's other keys, which run and freeze read none
    of. Each keyword of ``changes`` replaces or adds one key of the file (e.g. ``chosen=0``).
    """
    path = method_dir(GEPA_NAME) / TUNING_FOLDER / PARAMS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "words": words.settings(),
        "words_sha256": words.sha256(),
        "chosen": 3,
        "cost_usd": {"jev": 1.0, "luna": 0.5, "total": 1.5},
    } | changes
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def requests_of(name, part):
    """Build a part's requests from the test's dataset and split, as the commands build them."""
    raw = select_part(read_samples(), load_split(cli.SPLIT_FILE), part)
    return questions.requests(raw, name)


def test_parse_args_reads_each_command_and_its_defaults():
    render = cli.parse_args(
        ["render", "--method", "jev-statement", "--part", "test", "--limit", "3", "--out", "x"]
    )
    run = cli.parse_args(["run", "--method", "jev-statement-code", "--part", "dev"])
    freeze = cli.parse_args(["freeze", "--method", "jev-statement"])
    verify = cli.parse_args(["verify", "--method", "jev-statement-code"])

    assert (render.method, render.part, render.limit, render.out) == (
        "jev-statement",
        "test",
        3,
        Path("x"),
    )
    assert (run.method, run.part, run.allow_dirty) == ("jev-statement-code", "dev", False)
    assert run.max_cost_usd == MAX_COST_USD == 2.0
    for command in (run, freeze):
        assert (command.allow_dirty, command.tracking_uri, command.artifact_dir) == (
            False,
            MLFLOW_TRACKING_URI,
            MLFLOW_ARTIFACT_DIR,
        )
    assert vars(verify) == {"command": "verify", "method": "jev-statement-code"}


@pytest.mark.parametrize("option", ["--data", "--split"])
@pytest.mark.parametrize(
    "command",
    [
        ["render", "--part", "dev", "--limit", "1", "--out", "x"],
        ["run", "--part", "test"],
        ["freeze"],
        ["verify"],
    ],
)
def test_parse_args_takes_no_data_or_split_which_could_lift_the_gates(command, option, capsys):
    # Another split could park an unanswered dev problem outside dev, and another dataset could
    # hold another slice: every command reads $CODEFORCES_DATA_DIR and the committed split only.
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args([*command, "--method", "jev-statement", option, "x"])

    assert exit_info.value.code == 2
    assert f"unrecognized arguments: {option} x" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["render", "--part", "dev", "--limit", "0", "--out", "x"], "at least 1 problem, not 0"),
        (["render", "--part", "dev", "--limit", "-2", "--out", "x"], "at least 1 problem, not -2"),
        (["run", "--part", "dev", "--max-cost-usd", "0"], "finite and above 0 USD, not 0"),
        (["run", "--part", "dev", "--max-cost-usd", "-1"], "finite and above 0 USD, not -1"),
        (["run", "--part", "dev", "--max-cost-usd", "nan"], "finite and above 0 USD, not nan"),
        (["run", "--part", "dev", "--max-cost-usd", "inf"], "finite and above 0 USD, not inf"),
    ],
)
def test_parse_args_refuses_a_limit_below_one_and_a_cap_not_finite_above_zero(
    options, message, capsys
):
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args([*options, "--method", "jev-statement"])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_parse_args_takes_only_the_jev_methods(capsys):
    assert cli.parse_args(["verify", "--method", GEPA_NAME]).method == GEPA_NAME
    for twin in PLATT_BASES:
        assert cli.parse_args(["freeze", "--method", twin]).method == twin
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args(["verify", "--method", "label_prior"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'label_prior'" in capsys.readouterr().err


@pytest.mark.usefixtures("dataset", "split_file")
def test_render_writes_the_body_of_the_first_problems_and_sends_nothing(jev, api, tmp_path, capsys):
    out = tmp_path / "render"

    assert (
        jev("render", "jev-statement-code", "--part", "test", "--limit", "2", "--out", str(out))
        == 0
    )

    folder = out / "jev-statement-code" / "test"
    expected = requests_of("jev-statement-code", "test")[:2]
    assert sorted(path.name for path in folder.iterdir()) == ["11.json", "12.json"]
    for request in expected:
        text = (folder / f"{request.sample_id}.json").read_text(encoding="utf-8")
        assert text == json.dumps(request.body(), indent=2, ensure_ascii=False) + "\n"
    assert capsys.readouterr().out.splitlines() == [
        f"{folder / f'{request.sample_id}.json'}: request {request.sha256()[:12]}"
        for request in expected
    ]
    assert list(expected[1].state) == ["statement", "solution"]
    assert expected[1].state["solution"] == "print(n, k)"
    assert expected[1].model == MODEL
    assert (api.sent, api.sessions) == ([], [])


@pytest.fixture(name="read_events")
def read_events_fixture(events, monkeypatch):
    """Record each read of the dataset in the events, instead of reading it; return the events."""
    monkeypatch.setattr(cli, "_read_data", lambda: events.append(("read_data",)))
    return events


@pytest.mark.parametrize("twin", list(PLATT_BASES))
@pytest.mark.parametrize("command", ["render", "run"])
def test_render_and_run_refuse_a_calibrated_twin_before_anything_else(
    jev, read_events, tmp_path, command, twin
):
    options = {"render": ["--limit", "1", "--out", str(tmp_path / "render")], "run": []}
    message = (
        f"{twin}: a calibrated twin sends no request: it reads {PLATT_BASES[twin]}'s answers "
        r"\(docs/adr/0011-platt-lipton-threshold-rule\.md\)\."
    )

    with pytest.raises(ValueError, match=rf"^{message}$"):
        jev(command, twin, "--part", "dev", *options[command])

    # Refused before the dataset is checked or read, any session or any file: nothing was asked,
    # read or written.
    assert read_events == []
    assert not (tmp_path / "render").exists()
    assert not answers_file(twin, "dev").exists()


def test_run_sends_each_request_of_the_part_and_writes_its_answer(jev, api, events, capsys):
    assert jev("run", "jev-statement", "--part", "dev") == 0

    path = answers_file("jev-statement", "dev")
    answers = read_answers(path)
    assert api.sessions == ["jev-statement"]
    assert api.sent == [("jev-statement", sample_id) for sample_id in DEV]
    assert [answer.sample_id for answer in answers] == DEV
    assert [answer.request_sha256 for answer in answers] == [
        request.sha256() for request in requests_of("jev-statement", "dev")
    ]
    assert json.loads(answers[2].output_text)["number theory"] == 0.8  # "gcd" in its statement
    # The prompt is registered before the first request is sent.
    assert events == [("check_dataset",), ("register_prompt", 0)]
    assert not answers_file("jev-statement", "test").exists()
    assert (
        f"{path}: 10 of 10 answers (0 failed), 25,000 input tokens, 0 output tokens"
        in capsys.readouterr().out
    )


def test_run_with_nothing_to_send_needs_no_sdk_and_no_key(jev, git, monkeypatch, capsys):
    answer_every_part(jev, git)
    files = [answers_file(name, part) for name in FROZEN_NAMES for part in PARTS]
    before = [path.read_bytes() for path in files]
    capsys.readouterr()
    # No SDK and no key: the real session would fail to import typesafe_sdk if it were opened.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(client, "jev_session", REAL_SESSION)

    assert [jev("run", name, "--part", "test") for name in FROZEN_NAMES] == [0, 0]

    assert [path.read_bytes() for path in files] == before
    assert capsys.readouterr().out.splitlines() == [
        f"{answers_file(name, 'test')}: every request has an answer: nothing to send."
        for name in FROZEN_NAMES
    ]
    with pytest.raises(ImportError):
        asyncio.run(contextlib.AsyncExitStack().enter_async_context(REAL_SESSION("jev-statement")))


def test_run_resumes_after_a_transient_error_and_exits_with_1_until_done(jev, api, capsys):
    api.outcomes[3] = TransientApiError("TypeSafe did not answer the request of problem 3.")

    assert jev("run", "jev-statement-code", "--part", "dev") == 1

    path = answers_file("jev-statement-code", "dev")
    assert sorted(answer.sample_id for answer in read_answers(path)) == [1, 2, *DEV[3:]]
    assert (
        f"{path}: 1 of 10 requests have no answer yet: run again to send them."
        in capsys.readouterr().err
    )
    api.sent.clear()
    assert jev("run", "jev-statement-code", "--part", "dev") == 0
    assert api.sent == [("jev-statement-code", 3)]
    assert sorted(answer.sample_id for answer in read_answers(path)) == DEV


def test_run_exits_with_1_at_the_cost_cap_and_sends_the_rest_next_time(
    jev, api, monkeypatch, capsys
):
    # One request at a time, or every dev request would be in flight before the first is priced.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)

    # An answer costs 2,500 input tokens at 0.042 USD per million: 0.000105 USD.
    assert jev("run", "jev-statement", "--part", "dev", "--max-cost-usd", "0.0001") == 1

    assert len(api.sent) == 1
    err = capsys.readouterr().err
    assert "the cost cap of $0.0001 was reached" in err
    assert "9 of 10 requests have no answer yet: run again to send them." in err
    assert jev("run", "jev-statement", "--part", "dev") == 0
    assert len(api.sent) == 10
    assert len(read_answers(answers_file("jev-statement", "dev"))) == 10


def test_run_registers_the_prompt_in_its_store_with_no_key(jev, api, store, monkeypatch):
    monkeypatch.setattr(cli, "register_jev_prompt", register_jev_prompt)

    for _ in range(2):
        assert jev("run", "jev-statement", "--part", "dev") == 0

    use_store(**store)
    [version] = mlflow.MlflowClient().search_prompt_versions(PROMPT_NAME)
    assert (version.version, version.template) == (1, PROMPT.text)
    assert len(api.sent) == len(DEV)


def test_run_sends_nothing_when_the_registry_refuses_the_prompt(jev, api, monkeypatch):
    def refuse(**_store):
        raise ValueError("The prompt 'jev-tags' holds the versions [1], whose content hashes ...")

    monkeypatch.setattr(cli, "register_jev_prompt", refuse)

    with pytest.raises(ValueError, match="The prompt 'jev-tags' holds the versions"):
        jev("run", "jev-statement", "--part", "dev")

    assert (api.sent, api.sessions) == ([], [])
    assert not answers_file("jev-statement", "dev").exists()


def test_run_checks_the_dataset_then_refuses_a_dirty_tree(api, events, monkeypatch):
    monkeypatch.setattr(
        provenance, "git_state", lambda root=None: {"commit": "e" * 40, "code_dirty": True}
    )

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        cli.main(["run", "--method", "jev-statement", "--part", "dev"])  # no --allow-dirty

    assert events == [("check_dataset",)]
    assert (api.sent, api.sessions) == ([], [])


def test_run_refuses_a_split_that_is_not_committed_and_unchanged(jev, api, split_file):
    # Parking an unanswered dev problem outside dev would let the test part through.
    text = split_file.read_text(encoding="utf-8")
    split_file.write_text(text.replace("\n9,dev,4,9\n", "\n9,parked,4,9\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="split.csv is not committed, or has changed since"):
        jev("run", "jev-statement", "--part", "test")
    assert api.sent == []


def test_run_refuses_a_slice_that_is_not_committed(jev, api, git, slice_file):
    del git.commits[0][slice_file]

    with pytest.raises(ValueError, match="solutions.csv is not committed, or has changed since"):
        jev("run", "jev-statement", "--part", "dev")
    assert api.sent == []


def test_run_refuses_a_slice_that_is_not_the_one_of_the_data(jev, api, git, slice_file):
    header = slice_file.read_text(encoding="utf-8").splitlines()[0]
    slice_file.write_text(f"{header}\n", encoding="utf-8")  # problem 6 left out, and committed
    git.commit(slice_file)

    with pytest.raises(ValueError, match="is not the slice that self_identifying_solutions finds"):
        jev("run", "jev-statement", "--part", "dev")
    assert api.sent == []


def test_run_refuses_an_answers_file_that_lost_a_committed_line(jev, api, git):
    # A refused request's line is final: removing it would have the request asked again.
    api.outcomes[4] = {"status": "api_error:422"}
    assert jev("run", "jev-statement", "--part", "dev") == 0
    path = answers_file("jev-statement", "dev")
    git.commit(path)
    kept = [line for line in path.read_bytes().splitlines(True) if b'"sample_id": 4,' not in line]
    path.write_bytes(b"".join(kept))
    api.sent.clear()

    with pytest.raises(ValueError, match="the file on disk lacks or changes a line of its commit"):
        jev("run", "jev-statement", "--part", "dev")
    git.commit(path)  # committing the loss leaves it in the history
    with pytest.raises(ValueError, match=r"its commit \w{12} lacks or changes a line of its"):
        jev("run", "jev-statement", "--part", "dev")
    assert api.sent == []


def test_run_refuses_answers_that_are_not_to_the_current_requests(jev, api, monkeypatch):
    assert jev("run", "jev-statement", "--part", "dev") == 0
    dev = answers_file("jev-statement", "dev")
    test = answers_file("jev-statement", "test")
    test.write_bytes(dev.read_bytes())
    api.sent.clear()

    with pytest.raises(ValueError, match=r"answers for the problems \[1, 2, .*\], which have no"):
        jev("run", "jev-statement", "--part", "dev")
    test.unlink()
    # Every request holds every tag definition: a changed word is a new method (docs/adr/0003).
    monkeypatch.setitem(
        TAG_DEFINITIONS["games"]["false"], "definition", "No two players play against each other."
    )
    with pytest.raises(
        ValueError, match=r"The requests of the problems \[1, 2, .*\] are not the ones that"
    ):
        jev("run", "jev-statement", "--part", "dev")
    assert api.sent == []


def test_run_refuses_a_part_holding_an_answer_from_another_model(jev, api):
    api.outcomes[5] = {"served_model": "jev-latest"}

    with pytest.raises(ValueError, match="jev-latest") as written:
        jev("run", "jev-statement", "--part", "dev")

    path = answers_file("jev-statement", "dev")
    assert 5 in [answer.sample_id for answer in read_answers(path)]  # the paid line is kept
    sent = len(api.sent)
    with pytest.raises(ValueError, match="jev-latest") as refused:
        jev("run", "jev-statement", "--part", "dev")
    assert len(api.sent) == sent  # refused before any request
    assert "[5]" in str(written.value)
    assert "[5]" in str(refused.value)


def test_run_asks_the_test_part_only_once_every_jev_methods_dev_answers_are_committed_and_complete(
    jev, api, git
):
    with pytest.raises(ValueError, match="/jev-statement/answers_dev.jsonl is not committed"):
        jev("run", "jev-statement", "--part", "test")
    assert jev("run", "jev-statement", "--part", "dev") == 0
    git.commit(answers_file("jev-statement", "dev"))

    # The dev answers of jev-statement are committed and complete, not those of the other.
    with pytest.raises(ValueError, match="/jev-statement-code/answers_dev.jsonl is not committed"):
        jev("run", "jev-statement", "--part", "test")
    api.outcomes[9] = TransientApiError("TypeSafe did not answer the request of problem 9.")
    assert jev("run", "jev-statement-code", "--part", "dev") == 1
    git.commit(answers_file("jev-statement-code", "dev"))
    with pytest.raises(
        ValueError,
        match=r"no answer yet for the problems \[9\]: send their requests with `python -m "
        r"codeforces_tags.methods.typesafe_jev run --method jev-statement-code --part dev` first\.",
    ):
        jev("run", "jev-statement", "--part", "test")
    assert not answers_file("jev-statement", "test").exists()
    assert jev("run", "jev-statement-code", "--part", "dev") == 0
    git.commit(answers_file("jev-statement-code", "dev"))
    api.sent.clear()
    assert jev("run", "jev-statement", "--part", "test") == 0
    assert api.sent == [("jev-statement", sample_id) for sample_id in TEST]


def test_run_closes_the_dev_part_once_the_test_part_has_answers(jev, api, git):
    answer_every_part(jev, git)
    api.sent.clear()

    for name in FROZEN_NAMES:  # even with no dev request left to send
        with pytest.raises(
            ValueError, match=f"answers_test.jsonl has answers: the dev part of {name} is closed"
        ):
            jev("run", name, "--part", "dev")
    assert api.sent == []


def test_run_resumes_a_part_whose_last_line_was_cut_short(jev, api, git, capsys):
    assert jev("run", "jev-statement", "--part", "dev") == 0
    path = answers_file("jev-statement", "dev")
    *whole, last = path.read_bytes().splitlines(True)
    path.write_bytes(b"".join(whole))
    git.commit(path)  # nine lines committed, then a run killed while it wrote the tenth
    path.write_bytes(b"".join(whole) + last[: len(last) // 2])
    api.sent.clear()

    assert jev("run", "jev-statement", "--part", "dev") == 0

    assert api.sent == [("jev-statement", json.loads(last)["sample_id"])]
    assert path.read_bytes() == b"".join(whole) + last
    assert "ignored a last line of" in capsys.readouterr().err


def test_run_refuses_the_searched_method_until_its_words_are_committed(jev, api, git):
    with pytest.raises(ValueError, match=r"tuning/params\.json is not committed, or has changed"):
        jev("run", GEPA_NAME, "--part", "dev")
    path = write_params()
    with pytest.raises(ValueError, match="sends the words its search chose"):
        jev("run", GEPA_NAME, "--part", "dev")
    git.commit(path)

    assert jev("run", GEPA_NAME, "--part", "dev") == 0

    answered = read_answers(answers_file(GEPA_NAME, "dev"))
    raw = select_part(read_samples(), load_split(cli.SPLIT_FILE), "dev")
    asked = questions.requests(raw, GEPA_NAME, GEPA_WORDS)
    assert [answer.request_sha256 for answer in answered] == [r.sha256() for r in asked]
    seed = {request.sha256() for request in requests_of("jev-statement-code", "dev")}
    assert not seed & {answer.request_sha256 for answer in answered}
    assert api.sessions == [GEPA_NAME]
    write_params(dataclasses.replace(GEPA_WORDS, convention="Tags name ideas."))
    with pytest.raises(ValueError, match=r"params\.json is not committed, or has changed since"):
        jev("run", GEPA_NAME, "--part", "dev")


def test_the_searched_methods_test_part_waits_only_for_its_own_dev_answers(jev, api, git):
    git.commit(write_params())
    with pytest.raises(
        ValueError, match="/jev-statement-code-gepa/answers_dev.jsonl is not committed"
    ) as refused:
        jev("run", GEPA_NAME, "--part", "test")
    assert jev("run", GEPA_NAME, "--part", "dev") == 0
    git.commit(answers_file(GEPA_NAME, "dev"))
    api.sent.clear()

    # No frozen method has answered anything: they are another study.
    assert jev("run", GEPA_NAME, "--part", "test") == 0

    assert "every method of its study (jev-statement-code-gepa)" in str(refused.value)
    assert api.sent == [(GEPA_NAME, sample_id) for sample_id in TEST]
    assert not answers_file("jev-statement-code", "dev").exists()


def test_a_frozen_methods_test_part_never_waits_for_the_searched_method(jev, api, git):
    for name in FROZEN_NAMES:
        assert jev("run", name, "--part", "dev") == 0
    git.commit(*(answers_file(name, "dev") for name in FROZEN_NAMES))
    api.sent.clear()

    # The searched method has neither words nor answers yet.
    assert jev("run", "jev-statement", "--part", "test") == 0

    assert api.sent == [("jev-statement", sample_id) for sample_id in TEST]
    assert not method_dir(GEPA_NAME).exists()


def test_freeze_saves_the_searched_methods_words_and_records_it_in_sample(jev, git, events, capsys):
    git.commit(write_params())
    for part in PARTS:
        assert jev("run", GEPA_NAME, "--part", part) == 0
        git.commit(answers_file(GEPA_NAME, part))
    capsys.readouterr()

    assert jev("freeze", GEPA_NAME) == 0

    # No prompt is named after its model: the searched method reads no version of jev-tags.
    logged = capsys.readouterr().out.splitlines()[-1]
    assert logged == f"logged {GEPA_NAME} to MLflow: model model-id"
    directory = method_dir(GEPA_NAME)
    folder = directory / MODEL_FOLDER
    assert sorted(path.name for path in folder.iterdir()) == [
        "jev.json",
        "method.json",
        "words.json",
    ]
    assert load_method(folder).words == GEPA_WORDS
    record = load_run_record(directory)
    settings = method_settings(GEPA_NAME, GEPA_WORDS)
    assert record.method_entries["settings"] == {
        "model": MODEL,
        "state_fields": ["statement", "solution"],
        "settings_sha256": sha256_json(settings),
    }
    assert record.method_entries["dev_in_sample"] is True
    assert record.recipe_sha256 == cli.recipe_sha256(GEPA_NAME)
    assert record.recipe_sha256 == sha256_json(
        {"method": GEPA_NAME, "tags": list(FOCUS_TAGS), "settings": settings}
    )
    assert [event[2].method for event in events if event[0] == "log_method"] == [GEPA_NAME]
    # Its words are its search's record, not a version of jev-tags: neither its run nor its
    # freeze registers the prompt, its record pins none, and none is linked to it.
    assert [event for event in events if event[0] in ("register_prompt", "link_prompt")] == []
    assert not record.prompts
    assert jev("verify", GEPA_NAME) == 0


def test_verify_and_freeze_refuse_a_searched_method_folder_with_other_words(jev, git):
    git.commit(write_params())
    for part in PARTS:
        assert jev("run", GEPA_NAME, "--part", part) == 0
        git.commit(answers_file(GEPA_NAME, part))
    assert jev("freeze", GEPA_NAME) == 0
    folder = method_dir(GEPA_NAME) / MODEL_FOLDER
    other = dataclasses.replace(GEPA_WORDS, convention="Tags name ideas.")
    # Saved again with other words and the same thresholds: the folder loads, its jev.json hashes
    # its words.json, yet these words never had the answers that chose the thresholds.
    save_method(JevTagger(GEPA_NAME, load_method(folder).thresholds, other), folder)
    assert load_method(folder).words == other

    for command in ("verify", "freeze"):
        with pytest.raises(ValueError, match=f"holds other words than those {GEPA_NAME} sends"):
            jev(command, GEPA_NAME)


def test_module_runs_as_a_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.methods.typesafe_jev", "--help"])
    # Run the module afresh, as `python -m` does.
    monkeypatch.delitem(sys.modules, "codeforces_tags.methods.typesafe_jev.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.methods.typesafe_jev", run_name="__main__")

    assert exit_info.value.code == 0
    assert "usage: python -m codeforces_tags.methods.typesafe_jev" in capsys.readouterr().out


# What freeze makes of the fake Jev's answers, as jev-statement asks them: each tag with a dev
# positive gets the threshold halfway between the fake's 0.8 and 0.1, and geometry and
# probabilities, which no dev problem has, get an infinite one.
STATEMENT_THRESHOLDS = [0.45, 0.45, 0.45, 0.45, 0.45, math.inf, 0.45, math.inf]


@pytest.fixture(name="frozen")
def frozen_fixture(jev, git):
    """Answer both parts of both methods, commit the answers, then freeze jev-statement.

    Returns the directory of its frozen files.
    """
    answer_every_part(jev, git)
    assert jev("freeze", "jev-statement") == 0
    return method_dir("jev-statement")


def snapshot(directory):
    """Return the bytes and the modification time of each file under a folder, by its path."""
    files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns) for path in files
    }


def label_rows(sample_ids):
    """Return the 0/1 focus tags of the problems, one row per problem, from PROBLEMS."""
    return [[int(tag in PROBLEMS[sample_id][2]) for tag in FOCUS_TAGS] for sample_id in sample_ids]


def logged_records(events):
    """Return the run records the MLflow log was called with, in order."""
    return [event[2] for event in events if event[0] == "log_method"]


def summary_of(lines):
    """Return the run record's summary of an answers file of the fake's completed answers."""
    return {
        "lines": lines,
        "statuses": {"completed": lines},
        "failed": 0,
        "served_models": {"jev-1.13.0": lines},
        "input_tokens": 2500 * lines,
        "output_tokens": 0,
        "cost_usd": pytest.approx(lines * 2500 * 0.042 / 1e6),
    }


def test_the_recipe_tells_the_two_methods_apart_and_hashes_the_settings(monkeypatch):
    # Both methods hash the same SETTINGS, and log_method finds a method's run by
    # its recipe alone, so the name must be in it.
    recipes = {name: cli.recipe_sha256(name) for name in FROZEN_NAMES}

    assert recipes == {
        "jev-statement": "43464dbef429e0641aa0a461bc08e2a5b52c787d413fb07a4c95629bb2575069",
        "jev-statement-code": "5f73ca5686a4d241f12ee566991de4d8bba4f139cf025e8c1365558036d782ca",
    }
    assert recipes["jev-statement-code"] == sha256_json(
        {"method": "jev-statement-code", "tags": list(FOCUS_TAGS), "settings": SETTINGS}
    )
    monkeypatch.setitem(SETTINGS, "model", "jev-latest")
    assert all(cli.recipe_sha256(name) != recipe for name, recipe in recipes.items())


def test_freeze_saves_the_predictions_scores_calls_record_and_folder(jev, git, capsys):
    answer_every_part(jev, git)
    capsys.readouterr()

    assert jev("freeze", "jev-statement") == 0

    directory = method_dir("jev-statement")
    assert capsys.readouterr().out.splitlines() == [
        f"saved 13 predictions and scores to {directory}",
        f"saved each problem's call to {directory / PER_PROBLEM_FILE} (13 problems)",
        f"saved the method folder {directory / MODEL_FOLDER}",
        "logged jev-statement to MLflow: model model-id, prompt prompts:/jev-tags/1",
    ]
    # No staged file or folder is left behind.
    saved = [PREDICTIONS_FILE, SCORES_FILE, PER_PROBLEM_FILE, RUN_RECORD_FILE, MODEL_FOLDER]
    answered = [answers_file("jev-statement", part).name for part in PARTS]
    assert sorted(path.name for path in directory.iterdir()) == sorted(saved + answered)
    model = directory / MODEL_FOLDER
    assert sorted(path.name for path in model.iterdir()) == ["jev.json", "method.json"]
    calls = load_per_problem(directory)
    assert list(calls.index) == list(load_predictions(directory).index)
    assert calls["total_cost_usd"].sum() == pytest.approx(13 * 2500 * 0.042 / 1e6)


def test_freeze_predicts_dev_out_of_fold_and_test_with_the_final_thresholds(jev, git):
    answer_every_part(jev, git)

    assert jev("freeze", "jev-statement-code") == 0

    dev = pd.Index(DEV, name="sample_id")
    test = pd.Index(TEST, name="sample_id")
    answered = {part: read_answers(answers_file("jev-statement-code", part)) for part in PARTS}
    dev_scores = probabilities(answered["dev"], dev)
    test_scores = probabilities(answered["test"], test)
    truth = pd.DataFrame(label_rows(DEV), index=dev, columns=list(FOCUS_TAGS))
    folds = pd.Series([PROBLEMS[sample_id][1] for sample_id in DEV], index=dev)
    _, out_of_fold = cross_fitted_predictions(dev_scores, truth, folds)
    final = apply_thresholds(test_scores, per_tag_thresholds(dev_scores, truth))
    frozen = load_predictions(method_dir("jev-statement-code"))
    assert frozen.equals(stack_parts(dev=out_of_fold, test=final))
    assert load_scores(method_dir("jev-statement-code")).equals(
        stack_parts(dev=dev_scores, test=test_scores)
    )
    # Problems 3 and 9 are the only dev problems with number theory and with games: each one's
    # fold is predicted with thresholds chosen without it, so neither tag is predicted.
    assert frozen.loc[("dev", 3), "number theory"] == frozen.loc[("dev", 9), "games"] == 0
    assert frozen.loc["test"].to_numpy().tolist() == label_rows(TEST)


def test_freeze_records_the_answers_files_their_commits_and_summaries(frozen, git, split_file):
    record = load_run_record(frozen)

    files = {part: answers_file("jev-statement", part) for part in PARTS}
    assert record.method_entries == {
        "settings": {
            "model": "jev-1.13.0",
            "state_fields": ["statement"],
            "settings_sha256": sha256_json(SETTINGS),
        },
        "answers_sha256": {part: file_sha256(path) for part, path in files.items()},
        "answers_commit": {part: git.last_commit(path) for part, path in files.items()},
        "answers": {"dev": summary_of(10), "test": summary_of(3)},
        "served_models": ["jev-1.13.0"],
        "dev_in_sample": False,
    }
    assert record.recipe_sha256 == cli.recipe_sha256("jev-statement")
    assert record.split_sha256 == file_sha256(split_file)
    # A record frozen now pins the version of the prompt its requests were built from.
    assert (record.allow_dirty, record.prompts) == (True, ("prompts:/jev-tags/1",))


def test_freeze_pins_the_prompt_in_a_new_record_then_links_it_to_the_logged_model(jev, git, events):
    answer_every_part(jev, git)
    events.clear()

    assert jev("freeze", "jev-statement") == 0

    assert [event[0] for event in events] == [
        "check_dataset",
        "register_prompt",
        "log_method",
        "link_prompt",
    ]
    assert events[-1] == ("link_prompt", "model-id")
    assert logged_records(events)[0].prompts == ("prompts:/jev-tags/1",)


def test_freeze_keeps_a_record_that_pins_no_prompt_and_links_the_prompt_after_the_log(
    frozen, jev, events
):
    # The records frozen before the prompt registry pin no prompt, and are never rewritten.
    path = frozen / RUN_RECORD_FILE
    rewrite_json(path, lambda stored: stored.update(prompts=[]))
    before = path.read_bytes()
    events.clear()

    assert jev("freeze", "jev-statement") == 0

    assert path.read_bytes() == before
    assert [event[0] for event in events] == ["check_dataset", "log_method", "link_prompt"]
    assert logged_records(events)[0].prompts == ()


def test_freeze_writes_an_infinite_threshold_for_a_tag_without_a_dev_positive(frozen):
    text = (frozen / MODEL_FOLDER / "jev.json").read_text(encoding="utf-8")

    # JSON has no infinity: json_safe writes it as the string "inf".
    assert json.loads(text)["thresholds"] == [0.45, 0.45, 0.45, 0.45, 0.45, "inf", 0.45, "inf"]
    loaded = load_method(frozen / MODEL_FOLDER)
    assert loaded.thresholds.thresholds_.tolist() == STATEMENT_THRESHOLDS
    never = load_predictions(frozen)[["geometry", "probabilities"]]
    assert not never.to_numpy().any()


def test_freeze_scores_a_failed_answer_0_on_every_tag_which_a_0_threshold_meets(jev, git, api):
    # Problem 4 is one of the two dev problems with graphs; jev-statement's dev part is asked
    # first, so its answer to problem 4 is the refused one.
    api.outcomes[4] = {"status": "api_error:422"}
    assert jev("run", "jev-statement", "--part", "dev") == 0
    answer_every_part(jev, git)

    assert jev("freeze", "jev-statement") == 0

    directory = method_dir("jev-statement")
    assert load_scores(directory).loc[("dev", 4)].tolist() == [0.0] * len(FOCUS_TAGS)
    call = load_per_problem(directory).loc[("dev", 4)]
    assert (call["status"], call["total_cost_usd"]) == ("api_error:422", 0.0)
    summary = load_run_record(directory).method_entries["answers"]["dev"]
    assert (summary["statuses"], summary["failed"]) == ({"api_error:422": 1, "completed": 9}, 1)
    # The best graphs threshold on dev is now 0.0, the failed answer's score: every test problem
    # reaches it.
    assert load_method(directory / MODEL_FOLDER).thresholds.thresholds_[1] == 0.0
    assert load_predictions(directory).loc["test", "graphs"].tolist() == [1, 1, 1]


def test_freeze_logs_the_frozen_record_and_each_parts_metrics(frozen, events, split_file):
    [(_, folder, record, metrics, split, registry)] = [
        event for event in events if event[0] == "log_method"
    ]

    assert (folder, record) == (frozen / MODEL_FOLDER, load_run_record(frozen))
    assert registry is METHODS
    pd.testing.assert_frame_equal(split, load_split(split_file))
    assert list(metrics) == ["dev-oof", "test"]
    # The fake answers 0.8 to each test problem's tags only: test is predicted without error.
    assert (metrics["test"]["macro_f1"], metrics["test"]["micro_f1"]) == (0.5, 1.0)
    assert metrics["dev-oof"]["micro_f1"] == pytest.approx(16 / 21)
    assert metrics["dev-oof"]["recall/number theory"] == 0.0


def test_freeze_a_second_time_rewrites_nothing_and_logs_again(
    jev, git, events, capsys, monkeypatch
):
    days = iter(["2026-10-08T09:00:00+00:00", "2026-10-09T09:00:00+00:00"])
    make = cli.make_run_record

    def dated(*args, **kwargs):
        return dataclasses.replace(make(*args, **kwargs), created_at=next(days))

    monkeypatch.setattr(cli, "make_run_record", dated)
    answer_every_part(jev, git)
    jev("freeze", "jev-statement")
    directory = method_dir("jev-statement")
    before = snapshot(directory)
    capsys.readouterr()

    assert jev("freeze", "jev-statement") == 0

    assert snapshot(directory) == before
    # Both logs get the frozen record, never the one the second run made.
    created = [record.created_at for record in logged_records(events)]
    assert created == ["2026-10-08T09:00:00+00:00"] * 2
    assert capsys.readouterr().out.splitlines() == [
        f"13 predictions match the frozen files in {directory}",
        f"{directory / PER_PROBLEM_FILE} holds the calls of the answers files",
        f"the method folder {directory / MODEL_FOLDER} holds the re-derived thresholds",
        "logged jev-statement to MLflow: model model-id, prompt prompts:/jev-tags/1",
    ]


def test_freeze_refuses_answers_files_that_are_not_committed_or_complete(jev, git, events):
    for name in FROZEN_NAMES:
        assert jev("run", name, "--part", "dev") == 0
    git.commit(*(answers_file(name, "dev") for name in FROZEN_NAMES))
    assert jev("run", "jev-statement", "--part", "test") == 0
    path = answers_file("jev-statement", "test")

    with pytest.raises(ValueError, match="answers_test.jsonl is not committed, or has changed"):
        jev("freeze", "jev-statement")

    kept = [line for line in path.read_bytes().splitlines(True) if b'"sample_id": 13,' not in line]
    path.write_bytes(b"".join(kept))
    git.commit(path)
    missing = r"the problems \[13\].* --method jev-statement --part test"
    with pytest.raises(ValueError, match=missing):
        jev("freeze", "jev-statement")
    assert not (method_dir("jev-statement") / PREDICTIONS_FILE).exists()
    assert logged_records(events) == []


def test_freeze_refuses_an_answer_from_another_model(jev, git, events, monkeypatch):
    # The code now pins another model than the one that served the answers.
    answer_every_part(jev, git)
    monkeypatch.setattr("codeforces_tags.methods.typesafe_jev.answers.MODEL", "jev-1.14.0")

    with pytest.raises(ValueError, match=r"served by \['jev-1.13.0'\], not jev-1.14.0"):
        jev("freeze", "jev-statement")

    assert not (method_dir("jev-statement") / PREDICTIONS_FILE).exists()
    assert logged_records(events) == []


def test_freeze_refuses_a_run_record_that_holds_other_answers_files(frozen, jev, git, events):
    path = answers_file("jev-statement", "test")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"latency_s": 0.4', '"latency_s": 0.5', 1), encoding="utf-8")
    # A rewritten history (e.g. an amended commit) shows the edited file as its only version.
    for commit in git.commits:
        commit.pop(path, None)
    git.commit(path)

    with pytest.raises(ValueError, match="holds other answers_sha256 than the answers files"):
        jev("freeze", "jev-statement")

    assert len(logged_records(events)) == 1
    assert load_run_record(frozen).method_entries["answers_sha256"]["test"] != file_sha256(path)


def test_freeze_refuses_a_per_problem_file_that_is_not_the_answers(frozen, jev, events):
    path = frozen / PER_PROBLEM_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(",0.4,", ",0.5,", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="per_problem.csv is not the calls of the answers files"):
        jev("freeze", "jev-statement")

    assert len(logged_records(events)) == 1


def test_freeze_saves_no_method_folder_that_fails_its_check(jev, git, monkeypatch):
    answer_every_part(jev, git)
    target = "codeforces_tags.methods.typesafe_jev.method.load"
    # A load that loses the thresholds: the saved folder would not predict the frozen test rows.
    lost = fitted_thresholds([0.0] * len(FOCUS_TAGS), FOCUS_TAGS)
    monkeypatch.setattr(target, lambda folder: JevTagger("jev-statement", lost))

    with pytest.raises(ValueError, match="holds other thresholds"):
        jev("freeze", "jev-statement")

    directory = method_dir("jev-statement")
    assert not (directory / MODEL_FOLDER).exists()
    assert not list(directory.glob(".saving-*"))  # nor the temporary folder it was saved in
    monkeypatch.setattr(target, load)
    assert jev("freeze", "jev-statement") == 0
    assert load_method(directory / MODEL_FOLDER).thresholds.thresholds_.tolist() == (
        STATEMENT_THRESHOLDS
    )


def test_the_frozen_files_pass_check_frozen_method(frozen, assets, split_file):
    assert frozen.parent == assets

    check_frozen_method("jev-statement", assets_dir=assets, split=load_split(split_file))


def use_real_mlflow(monkeypatch):
    """Put back the real MLflow log and prompt registry, under a registry that pins no SDK.

    log_method pins every requirement from uv.lock, and the default job does not install
    typesafe-sdk: this registry leaves it out.
    """
    local = {
        name: dataclasses.replace(
            METHODS[name],
            requirements=tuple(
                requirement
                for requirement in METHODS[name].requirements
                if requirement != "typesafe-sdk"
            ),
        )
        for name in NAMES
    }
    monkeypatch.setattr(cli, "METHODS", {**METHODS, **local})
    monkeypatch.setattr(cli, "log_method", log_method)
    monkeypatch.setattr(cli, "register_jev_prompt", register_jev_prompt)
    monkeypatch.setattr(cli, "link_jev_prompt", link_jev_prompt)


def linked_prompts(tags):
    """Return the prompt versions that a run's or a model's tags say are linked to it."""
    return json.loads(tags.get(LINKED_PROMPTS_TAG, "[]"))


def test_freeze_mirrors_both_methods_to_their_own_mlflow_runs(jev, git, store, monkeypatch, capsys):
    # On a real MLflow store.
    use_real_mlflow(monkeypatch)
    answer_every_part(jev, git)
    capsys.readouterr()

    for name in (*FROZEN_NAMES, "jev-statement"):
        assert jev("freeze", name) == 0

    runs = {name: method_run_id(cli.recipe_sha256(name), **store) for name in FROZEN_NAMES}
    assert None not in runs.values()
    assert runs["jev-statement"] != runs["jev-statement-code"]
    assert {name: mlflow.get_run(run).data.tags["method"] for name, run in runs.items()} == {
        name: name for name in FROZEN_NAMES
    }
    models = mlflow.search_logged_models([use_store(**store)], output_format="list")
    assert sorted(model.source_run_id for model in models) == sorted(runs.values())
    # The second freeze of jev-statement logged nothing new: the same model, no third one.
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("logged")]
    assert lines[2] == lines[0]
    # Each method run and its model are linked to the one version of jev-tags, once.
    registry = mlflow.MlflowClient()
    expected = [{"name": PROMPT_NAME, "version": "1"}]
    assert [linked_prompts(model.tags) for model in models] == [expected] * 2
    run_tags = [registry.get_run(run).data.tags for run in runs.values()]
    assert [linked_prompts(tags) for tags in run_tags] == [expected] * 2
    assert len(registry.search_prompt_versions(PROMPT_NAME)) == 1


def test_freeze_links_the_prompt_to_a_method_whose_record_pins_none(jev, git, store, monkeypatch):
    # The frozen methods' records pin no prompt, so log_method links none: freeze links it.
    use_real_mlflow(monkeypatch)
    make = cli.make_run_record
    monkeypatch.setattr(
        cli, "make_run_record", lambda *args, **kwargs: make(*args, **(kwargs | {"prompts": ()}))
    )
    answer_every_part(jev, git)

    for _ in range(2):
        assert jev("freeze", "jev-statement") == 0

    assert not load_run_record(method_dir("jev-statement")).prompts
    run_id = method_run_id(cli.recipe_sha256("jev-statement"), **store)
    registry = mlflow.MlflowClient()
    expected = [{"name": PROMPT_NAME, "version": "1"}]
    assert linked_prompts(registry.get_logged_model(logged_model_id(run_id)).tags) == expected
    assert linked_prompts(registry.get_run(run_id).data.tags) == expected


def test_freeze_checks_the_dataset_then_refuses_a_dirty_tree(jev, git, events, monkeypatch):
    answer_every_part(jev, git)
    events.clear()
    monkeypatch.setattr(
        provenance, "git_state", lambda root=None: {"commit": "d" * 40, "code_dirty": True}
    )

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        cli.main(["freeze", "--method", "jev-statement"])  # no --allow-dirty

    assert events == [("check_dataset",)]
    assert not (method_dir("jev-statement") / PREDICTIONS_FILE).exists()


def test_freeze_refuses_a_split_that_is_not_committed_and_unchanged(jev, git, events, split_file):
    answer_every_part(jev, git)
    split_file.write_text(split_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split.csv is not committed, or has changed since"):
        jev("freeze", "jev-statement")

    assert not (method_dir("jev-statement") / PREDICTIONS_FILE).exists()
    assert logged_records(events) == []


def test_freeze_needs_no_sdk_and_no_key(jev, git, monkeypatch):
    answer_every_part(jev, git)
    # The real session would fail to import typesafe_sdk, or to find the key, if freeze opened it.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(client, "jev_session", REAL_SESSION)

    assert jev("freeze", "jev-statement") == 0


def rewrite_json(path, edit):
    """Rewrite a JSON file of a frozen method after ``edit`` changed the values it holds."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    edit(stored)
    path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")


def append_a_newline(path):
    """Add an empty line at the end of a file, so that its bytes change."""
    path.write_bytes(path.read_bytes() + b"\n")


def serve_by_another_model(directory, _monkeypatch):
    """Make the committed answer to test problem 12 one of jev-latest's, on disk only."""
    path = directory / "answers_test.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    edited = [
        line.replace('"served_model": "jev-1.13.0"', '"served_model": "jev-latest"')
        if '"sample_id": 12,' in line
        else line
        for line in lines
    ]
    path.write_text("".join(edited), encoding="utf-8")


def pin_another_model(_directory, monkeypatch):
    """Make the code pin another model than the one that served every answer."""
    monkeypatch.setattr("codeforces_tags.methods.typesafe_jev.answers.MODEL", "jev-1.14.0")


def flip_a_prediction(directory):
    """Predict games for dev problem 1 in the frozen predictions.csv, a well-formed file still."""
    frozen = load_predictions(directory)
    frozen.loc[("dev", 1), "games"] = 1
    frozen.to_csv(directory / PREDICTIONS_FILE)


def change_a_score(directory):
    """Change dev problem 1's probability of games in the frozen scores.csv."""
    frozen = load_scores(directory)
    frozen.loc[("dev", 1), "games"] = 0.2
    frozen.to_csv(directory / SCORES_FILE)


def change_a_latency(directory):
    """Change the first recorded latency in the frozen per_problem.csv."""
    path = directory / PER_PROBLEM_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(",0.4,", ",0.5,", 1), encoding="utf-8")


def test_verify_re_derives_the_frozen_files_and_reports_the_scores(frozen, jev, events, capsys):
    before = snapshot(frozen)
    capsys.readouterr()

    assert jev("verify", "jev-statement") == 0

    assert snapshot(frozen) == before
    assert len(logged_records(events)) == 1  # the freeze's: verify logs nothing
    assert capsys.readouterr().out.splitlines() == [
        "the frozen files of jev-statement are complete and consistent with the split",
        "the run record of jev-statement holds the recipe of the code",
        "rebuilt 13 requests: each has its answer, under its hash",
        "every completed answer was served by jev-1.13.0",
        "each answers file holds every line of its committed versions, then new ones",
        "the run record holds the hashes, the commits and the summaries of the answers files",
        "re-derived 13 predictions, their scores and per_problem.csv from the answers: they equal "
        "the frozen ones",
        f"the method folder {frozen / MODEL_FOLDER} holds the re-derived thresholds",
        # Dev is out of fold: number theory and games, each on one dev problem only, are missed.
        "dev-oof: macro F1 0.4196, micro F1 0.7619",
        "test: macro F1 0.5000, micro F1 1.0000",
    ]


@pytest.mark.usefixtures("frozen")
def test_verify_needs_no_sdk_and_no_key(jev, monkeypatch):
    # The real session would fail to import typesafe_sdk, or to find the key, if verify opened it.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(client, "jev_session", REAL_SESSION)

    assert jev("verify", "jev-statement") == 0


@pytest.mark.usefixtures("frozen")
def test_verify_refuses_another_recipe(jev, monkeypatch):
    monkeypatch.setitem(SETTINGS, "convention", "Tags name techniques.")

    with pytest.raises(ValueError, match="was frozen with another recipe"):
        jev("verify", "jev-statement")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (serve_by_another_model, r"\[12\] were served by \['jev-latest'\], not jev-1.13.0"),
        (pin_another_model, r"were served by \['jev-1.13.0'\], not jev-1.14.0"),
    ],
    ids=["on disk", "in the code"],
)
def test_verify_refuses_an_answer_from_another_model(frozen, jev, monkeypatch, change, message):
    # The served-model check runs before the append-only check, so an answers
    # file edited on disk reaches it.
    change(frozen, monkeypatch)

    with pytest.raises(ValueError, match=message):
        jev("verify", "jev-statement")


@pytest.mark.parametrize("committed", [False, True], ids=["on disk", "committed"])
def test_verify_refuses_an_answers_file_whose_committed_line_was_edited(
    frozen, jev, git, committed
):
    path = frozen / "answers_dev.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"latency_s": 0.4', '"latency_s": 0.5', 1), encoding="utf-8")
    if committed:
        git.commit(path)
    later = r"its commit \w{12}" if committed else "the file on disk"

    with pytest.raises(ValueError, match=f"answers_dev.jsonl: {later} lacks or changes a line"):
        jev("verify", "jev-statement")


@pytest.mark.parametrize(
    ("entry", "value", "message"),
    [
        ("answers_sha256", {"dev": "0" * 64}, "holds other answers_sha256 than"),
        ("answers", {"dev": {"lines": 9}}, "holds other answers than"),
        ("answers_commit", {"test": "f" * 40}, "names the commit f{40} for answers_test.jsonl"),
    ],
)
def test_verify_refuses_a_run_record_whose_answers_hashes_or_commits_differ(
    frozen, jev, entry, value, message
):
    rewrite_json(
        frozen / RUN_RECORD_FILE,
        lambda record: record["method_entries"][entry].update(value),
    )

    with pytest.raises(ValueError, match=message):
        jev("verify", "jev-statement")


@pytest.mark.parametrize(
    ("change", "file_name"),
    [
        (flip_a_prediction, PREDICTIONS_FILE),
        (change_a_score, SCORES_FILE),
        (change_a_latency, PER_PROBLEM_FILE),
    ],
)
def test_verify_refuses_a_changed_predictions_scores_or_per_problem_file(
    frozen, jev, change, file_name
):
    change(frozen)

    with pytest.raises(ValueError, match=re.escape(f"The frozen files ['{file_name}']")):
        jev("verify", "jev-statement")


def test_verify_refuses_a_method_folder_with_other_thresholds(frozen, jev):
    def lower_math(stored):
        stored["thresholds"][0] = 0.05

    rewrite_json(frozen / MODEL_FOLDER / "jev.json", lower_math)

    with pytest.raises(ValueError, match="holds other thresholds"):
        jev("verify", "jev-statement")


@pytest.mark.usefixtures("frozen")
def test_verify_refuses_answers_to_requests_that_are_not_the_current_ones(jev, monkeypatch):
    # Only the seed words that the requests are built from hold the patched question: every
    # request's hash changes, the recipe (SETTINGS) does not, so the request check refuses.
    question = QUESTION.replace("Would", "Will")
    seed = jev_words.SEED_WORDS
    monkeypatch.setattr(jev_words, "SEED_WORDS", dataclasses.replace(seed, question=question))

    with pytest.raises(ValueError, match="are not the ones that were answered"):
        jev("verify", "jev-statement")


def test_verify_refuses_missing_frozen_files(jev, git):
    answer_every_part(jev, git)

    with pytest.raises(ConformanceError, match="holds no method.json"):
        jev("verify", "jev-statement")


# What the searched method's search cost, as write_params records it in tuning/params.json.
SEARCH_COST = {"jev": 1.0, "luna": 0.5, "total": 1.5}
# The cost of another search, which a params.json changed after the freeze records.
CHANGED_COST = SEARCH_COST | {"total": 10.0}


def freeze_the_seed(jev, git):
    """Answer both frozen methods, freeze jev-statement-code, the search's seed, and commit it.

    Returns the directory of the seed's frozen files.
    """
    for part in PARTS:
        for name in FROZEN_NAMES:
            assert jev("run", name, "--part", part) == 0
        git.commit(*(answers_file(name, part) for name in FROZEN_NAMES))
    assert jev("freeze", GEPA_SEED_NAME) == 0
    directory = method_dir(GEPA_SEED_NAME)
    git.commit(directory / SCORES_FILE, directory / RUN_RECORD_FILE)
    return directory


def test_tune_checks_the_gates_then_searches_from_the_seeds_committed_dev_answers(
    jev, git, events, monkeypatch, tmp_path
):
    freeze_the_seed(jev, git)
    events.clear()
    calls = []

    def search(data, folder, **options):
        calls.append((data, folder, options))
        return SearchOutcome(
            stop_reason="iterations",
            iterations=2,
            best_index=1,
            cost_usd={"jev": 0.5, "luna": 0.25, "total": 0.75},
        )

    monkeypatch.setattr(search_commands, "run_search", search)
    monkeypatch.setattr(search_commands, "search_data", lambda raw: raw)
    monkeypatch.setattr(search_commands, "EXPERIMENTS_DIR", tmp_path / "experiments")

    assert cli.main(["tune", "--method", GEPA_NAME, "--allow-dirty", "--no-mlflow"]) == 0

    assert len(calls) == 1
    raw_dev, folder, options = calls[0]
    assert raw_dev.index.tolist() == DEV
    assert raw_dev["fold"].tolist() == [PROBLEMS[sample_id][1] for sample_id in DEV]
    assert folder == tmp_path / "experiments" / GEPA_NAME / "search"
    assert options["committed_seed_answers"] == read_answers(answers_file(GEPA_SEED_NAME, "dev"))
    assert options["live"] is None
    assert events == [("check_dataset",)]


@pytest.mark.parametrize("file_name", ["answers_dev.jsonl", SCORES_FILE, RUN_RECORD_FILE])
def test_tune_and_export_refuse_until_the_seeds_frozen_files_are_committed(
    jev, git, monkeypatch, file_name
):
    append_a_newline(freeze_the_seed(jev, git) / file_name)  # changed since its commit
    started = []
    monkeypatch.setattr(search_commands, "run_search", lambda *args, **options: started.append(1))
    monkeypatch.setattr(search_commands, "export_search", lambda *args: started.append(1))

    for command in ("tune", "export"):
        with pytest.raises(ValueError, match=f"{file_name} is not committed, or has changed since"):
            cli.main([command, "--method", GEPA_NAME, "--allow-dirty"])

    assert not started


@pytest.mark.parametrize("command", ["tune", "export"])
def test_tune_and_export_check_the_dataset_then_refuse_a_dirty_tree(events, monkeypatch, command):
    monkeypatch.setattr(
        provenance, "git_state", lambda root=None: {"commit": "d" * 40, "code_dirty": True}
    )

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        cli.main([command, "--method", GEPA_NAME])  # no --allow-dirty

    assert events == [("check_dataset",)]


def test_export_hands_over_the_seeds_dev_scores_and_the_texts_of_every_problem(
    jev, git, monkeypatch, tmp_path
):
    seed = freeze_the_seed(jev, git)
    calls = []

    def export(folder, data, seed_scores, texts, out):
        calls.append((folder, data, seed_scores, texts, out))
        out.mkdir(parents=True)
        params = {"chosen": 2, "chosen_selection_mean": 0.85, "seed_selection_mean": 0.8}
        (out / PARAMS_FILE).write_text(json.dumps(params), encoding="utf-8")
        return out

    monkeypatch.setattr(search_commands, "export_search", export)
    monkeypatch.setattr(search_commands, "search_data", lambda raw: raw)
    monkeypatch.setattr(search_commands, "EXPERIMENTS_DIR", tmp_path / "experiments")

    assert cli.main(["export", "--method", GEPA_NAME, "--allow-dirty"]) == 0

    assert len(calls) == 1
    folder, raw_dev, seed_scores, texts, out = calls[0]
    assert folder == tmp_path / "experiments" / GEPA_NAME / "search"
    assert raw_dev.index.tolist() == DEV
    assert seed_scores.equals(load_scores(seed).loc["dev"])
    assert len(texts) == 3 * len(PROBLEMS)  # each problem's statement, notes and solution
    assert "print(leaves, vertices)" in texts
    assert out == method_dir(GEPA_NAME) / TUNING_FOLDER


@pytest.fixture(name="searched_method")
def searched_method_fixture(jev, git):
    """Commit the searched method's params.json, answer and commit both parts, then freeze it.

    Returns the directory of its frozen files.
    """
    git.commit(write_params())
    for part in PARTS:
        assert jev("run", GEPA_NAME, "--part", part) == 0
        git.commit(answers_file(GEPA_NAME, part))
    assert jev("freeze", GEPA_NAME) == 0
    return method_dir(GEPA_NAME)


def test_freeze_records_the_search_and_the_held_out_folds_scores_of_the_searched_method(
    searched_method,
):
    entries = load_run_record(searched_method).method_entries

    assert entries["dev_in_sample"] is True
    assert entries["search"] == {
        "held_out_fold": 0,
        "selection_fold": 4,
        "learning_folds": [1, 2, 3],
        "chosen": 3,
        "params_sha256": file_sha256(searched_method / TUNING_FOLDER / PARAMS_FILE),
        "cost_usd": SEARCH_COST,
        # Fold 0 holds problems 1 (trees, graphs) and 2 (strings): with thresholds chosen on
        # folds 1-4, trees and strings are found and graphs is missed.
        "held_out_macro_f1": 0.25,
        "held_out_f1": dict.fromkeys(FOCUS_TAGS, 0.0) | {"strings": 1.0, "trees": 1.0},
    }


@pytest.mark.parametrize(
    ("committed", "message"),
    [
        (False, "params.json is not committed, or has changed since"),
        (True, "holds another search than tuning/params.json"),
    ],
    ids=["on disk", "committed"],
)
def test_freeze_refuses_a_params_file_changed_after_the_searched_method_was_frozen(
    searched_method, jev, git, committed, message
):
    path = write_params(cost_usd=CHANGED_COST)
    if committed:
        git.commit(path)

    with pytest.raises(ValueError, match=message):
        jev("freeze", GEPA_NAME)

    assert load_run_record(searched_method).method_entries["search"]["cost_usd"] == SEARCH_COST


@pytest.mark.usefixtures("searched_method")
def test_verify_checks_the_search_entries_of_the_searched_method(jev, capsys):
    capsys.readouterr()

    assert jev("verify", GEPA_NAME) == 0

    assert (
        "the run record holds the sha256 of tuning/params.json and the held-out fold's scores"
        in capsys.readouterr().out.splitlines()
    )


@pytest.mark.usefixtures("searched_method")
def test_verify_refuses_a_params_file_changed_after_the_searched_method_was_frozen(jev):
    write_params(cost_usd=CHANGED_COST)  # the same words, on disk

    with pytest.raises(ValueError, match="holds another search than tuning/params.json"):
        jev("verify", GEPA_NAME)


# The calibrated twin of jev-statement-code (docs/adr/0011), and the files of its base it reads.
TWIN = "jev-statement-code-platt"
BASE_FILE_NAMES = ["answers_dev.jsonl", "answers_test.jsonl", SCORES_FILE, RUN_RECORD_FILE]


def export_twin(twin=TWIN):
    """Run export on a calibrated twin, with --allow-dirty (the tree may be dirty in tests)."""
    return cli.main(["export", "--method", twin, "--allow-dirty"])


def dev_labels():
    """Return the 0/1 focus tags of the dev problems, indexed by sample id, from PROBLEMS."""
    index = pd.Index(DEV, name="sample_id")
    return pd.DataFrame(label_rows(DEV), index=index, columns=list(FOCUS_TAGS))


def test_export_writes_a_twins_dev_only_record_from_its_bases_committed_files(jev, git, capsys):
    base = freeze_the_seed(jev, git)
    capsys.readouterr()

    assert export_twin() == 0

    tuning = method_dir(TWIN) / TUNING_FOLDER
    assert sorted(path.name for path in tuning.iterdir()) == sorted(PLATT_FILES)
    assert sorted(path.name for path in method_dir(TWIN).iterdir()) == [TUNING_FOLDER]
    params = platt.read_platt_params(tuning)
    assert (params["rule"], params["base"]) == ("R2", "jev-statement-code")
    assert params["base_files"] == {
        name: {"sha256": file_sha256(base / name), "commit": git.last_commit(base / name)}
        for name in BASE_FILE_NAMES
    }
    expected = lipton_thresholds(load_scores(base).loc["dev"], dev_labels())
    assert params["thresholds"] == json_safe(expected.to_dict())
    # No dev problem has geometry or probabilities.
    assert params["thresholds"]["geometry"] == params["thresholds"]["probabilities"] == "inf"
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(
        f"{tuning} now holds the record of the threshold rule R2 on the dev scores of "
        "jev-statement-code; the rule chosen inside each fold's other folds would be ["
    )
    assert lines[1] == (
        "Commit it, then freeze the twin with `python -m codeforces_tags.methods.typesafe_jev "
        f"freeze --method {TWIN}`."
    )


def test_export_of_a_twin_never_reads_a_test_score(jev, git):
    # The base's test rows are poisoned with NaN, and committed: the record is
    # the same, but for the sha256 and commit of scores.csv that params.json names.
    base = freeze_the_seed(jev, git)
    assert export_twin() == 0
    tuning = method_dir(TWIN) / TUNING_FOLDER
    clean = {name: (tuning / name).read_bytes() for name in PLATT_FILES}
    shutil.rmtree(method_dir(TWIN))
    scores = load_scores(base)
    scores.loc[scores.index.get_level_values("part") == "test"] = np.nan
    scores.to_csv(base / SCORES_FILE)
    git.commit(base / SCORES_FILE)

    assert export_twin() == 0

    for name in PLATT_FILES[1:]:
        assert (tuning / name).read_bytes() == clean[name]
    params = platt.read_platt_params(tuning)
    before = json.loads(clean[PARAMS_FILE])
    assert params["thresholds"] == before["thresholds"]
    assert params["base_files"][SCORES_FILE]["sha256"] == file_sha256(base / SCORES_FILE)
    assert params["base_files"][SCORES_FILE] != before["base_files"][SCORES_FILE]


@pytest.mark.parametrize("file_name", BASE_FILE_NAMES)
def test_export_of_a_twin_refuses_until_its_bases_files_are_committed(jev, git, file_name):
    append_a_newline(freeze_the_seed(jev, git) / file_name)  # changed since its commit

    with pytest.raises(
        ValueError, match=f"{file_name} is not committed, or has changed since: {TWIN} reads"
    ):
        export_twin()

    assert not method_dir(TWIN).exists()


def test_export_of_a_twin_a_second_time_writes_nothing(jev, git, capsys):
    freeze_the_seed(jev, git)
    assert export_twin() == 0
    tuning = method_dir(TWIN) / TUNING_FOLDER
    before = snapshot(tuning)
    capsys.readouterr()

    assert export_twin() == 0

    assert snapshot(tuning) == before
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith(f"{tuning} holds the same record of the threshold rule R2")


def test_export_of_a_twin_checks_the_dataset_then_refuses_a_dirty_tree(events, monkeypatch):
    monkeypatch.setattr(
        provenance, "git_state", lambda root=None: {"commit": "d" * 40, "code_dirty": True}
    )

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        cli.main(["export", "--method", TWIN])  # no --allow-dirty

    assert events == [("check_dataset",)]


# The calibrated twin of the searched method.
GEPA_TWIN = "jev-statement-code-gepa-platt"
# The files of a base that its twin's freeze copies, byte for byte.
BASE_COPIES = [LATENCY_FILE, PER_PROBLEM_FILE]


def commit_base_calls(git, directory):
    """Write a base's latency.csv, as tracking latency does; commit it and its per_problem.csv."""
    seconds = pd.Series([0.25, 0.5], index=pd.Index(DEV[:2], name="sample_id"), name="seconds")
    save_latency(directory, Latency(seconds, bulk_s=1.0, load_s=0.125))
    git.commit(*(directory / name for name in BASE_COPIES))


def export_and_commit(git, twin=TWIN):
    """Export a twin's dev-only record, then commit its files."""
    assert export_twin(twin) == 0
    git.commit(*(method_dir(twin) / TUNING_FOLDER / name for name in PLATT_FILES))


def squash_history(git):
    """Rewrite the fake history as one commit of every file, as a squash merge does."""
    files = {}
    for commit in git.commits:
        files |= commit
    git.commits = [files]


@pytest.fixture(name="twin_base")
def twin_base_fixture(jev, git):
    """Freeze jev-statement-code with its latency.csv, then export TWIN; commit all of it.

    Returns the directory of the base's frozen files.
    """
    base = freeze_the_seed(jev, git)
    commit_base_calls(git, base)
    export_and_commit(git)
    return base


@pytest.fixture(name="frozen_twin")
def frozen_twin_fixture(jev, twin_base):  # pylint: disable=unused-argument
    """Freeze TWIN over its committed base; return the directory of its frozen files."""
    assert jev("freeze", TWIN) == 0
    return method_dir(TWIN)


def test_freeze_of_a_twin_reads_its_bases_files_and_writes_only_its_own(jev, twin_base, capsys):
    before = snapshot(twin_base)
    capsys.readouterr()

    assert jev("freeze", TWIN) == 0

    directory = method_dir(TWIN)
    assert snapshot(twin_base) == before  # nothing is written in the base's folder
    copied = [
        f"copied {twin_base / name} to {directory / name}: the twin sends its base's requests"
        for name in BASE_COPIES
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"saved 13 predictions and scores to {directory}",
        *copied,
        f"saved the method folder {directory / MODEL_FOLDER}",
        f"logged {TWIN} to MLflow: model model-id",
    ]
    saved = [*BASE_COPIES, MODEL_FOLDER, PREDICTIONS_FILE, RUN_RECORD_FILE, SCORES_FILE]
    assert sorted(path.name for path in directory.iterdir()) == sorted([*saved, TUNING_FOLDER])
    for name in BASE_COPIES:
        assert (directory / name).read_bytes() == (twin_base / name).read_bytes()
    assert load_scores(directory).equals(load_scores(twin_base))
    model = directory / MODEL_FOLDER
    assert sorted(path.name for path in model.iterdir()) == ["jev.json", "method.json"]


def test_freeze_of_a_twin_predicts_with_its_platt_lipton_thresholds(frozen_twin, twin_base):
    scores = load_scores(twin_base)
    dev_scores, test_scores = scores.loc["dev"], scores.loc["test"]
    folds = pd.Series([PROBLEMS[sample_id][1] for sample_id in DEV], index=dev_scores.index)
    wrapper = PlattLiptonThresholds(JevProbabilities()).fit(dev_scores, dev_labels(), groups=folds)
    tags = list(FOCUS_TAGS)

    assert load_predictions(frozen_twin).equals(
        stack_parts(
            dev=pd.DataFrame(wrapper.oof_predictions_, index=dev_scores.index, columns=tags),
            test=pd.DataFrame(wrapper.predict(test_scores), index=test_scores.index, columns=tags),
        )
    )
    stored = json.loads((frozen_twin / MODEL_FOLDER / "jev.json").read_text(encoding="utf-8"))
    params = platt.read_platt_params(frozen_twin / TUNING_FOLDER)
    assert stored["thresholds"] == [params["thresholds"][tag] for tag in FOCUS_TAGS]
    # Each finite threshold is a midpoint of Jev's grid between the fake's 0.1 and 0.8: the
    # test problems are predicted without error, as by the base.
    finite = [value for value in stored["thresholds"] if value != "inf"]
    assert all(0.1 < value < 0.8 and round(value * 1000) % 10 == 5 for value in finite)
    assert load_predictions(frozen_twin).loc["test"].to_numpy().tolist() == label_rows(TEST)


def test_freeze_of_a_twin_records_its_base_its_rule_and_its_disclosure(
    frozen_twin, twin_base, git, events
):
    record = load_run_record(frozen_twin)

    files = {
        name: {"sha256": file_sha256(twin_base / name), "commit": git.last_commit(twin_base / name)}
        for name in BASE_FILE_NAMES
    }
    assert record.method_entries == {
        "settings": {
            "model": MODEL,
            "state_fields": ["statement", "solution"],
            "settings_sha256": sha256_json(method_settings(TWIN, jev_words.SEED_WORDS)),
        },
        "base": {"name": "jev-statement-code", "files": files},
        "served_models": ["jev-1.13.0"],
        "threshold_rule": json_safe(RULE_SETTINGS),
        "dev_in_sample": False,  # its base's dev rows are out of fold, and so are its own
        "test_informed": TEST_INFORMED,
        "latency_from": "jev-statement-code",
    }
    assert record.recipe_sha256 == cli.recipe_sha256(TWIN)
    assert platt.read_platt_params(frozen_twin / TUNING_FOLDER)["base_files"] == files
    assert [logged.method for logged in logged_records(events)] == ["jev-statement-code", TWIN]
    # A twin sends no request of its own: its record pins no prompt, and its freeze registers
    # and links none; only its base's run and freeze did.
    assert not record.prompts
    prompt_events = [i for i, event in enumerate(events) if event[0].endswith("_prompt")]
    twin_log = next(
        i for i, event in enumerate(events) if event[0] == "log_method" and event[2].method == TWIN
    )
    assert prompt_events and max(prompt_events) < twin_log


def test_the_gepa_twin_inherits_the_in_sample_dev_of_its_base(jev, git, searched_method):
    git.commit(searched_method / SCORES_FILE, searched_method / RUN_RECORD_FILE)
    commit_base_calls(git, searched_method)
    export_and_commit(git, GEPA_TWIN)

    assert jev("freeze", GEPA_TWIN) == 0

    directory = method_dir(GEPA_TWIN)
    entries = load_run_record(directory).method_entries
    assert (entries["base"]["name"], entries["dev_in_sample"]) == (GEPA_NAME, True)
    assert entries["test_informed"] == TEST_INFORMED
    assert load_method(directory / MODEL_FOLDER).words == GEPA_WORDS
    assert jev("verify", GEPA_TWIN) == 0


@pytest.mark.parametrize("file_name", PLATT_FILES)
def test_freeze_of_a_twin_refuses_until_every_file_of_its_record_is_committed(jev, git, file_name):
    # Regression: rules.csv, nested.csv and stability.csv are pinned by no hash, so each is gated.
    commit_base_calls(git, freeze_the_seed(jev, git))
    assert export_twin() == 0
    tuning = method_dir(TWIN) / TUNING_FOLDER
    git.commit(*(tuning / name for name in PLATT_FILES if name != file_name))

    with pytest.raises(ValueError, match=f"{file_name} is not committed, or has changed since: a"):
        jev("freeze", TWIN)

    assert not (method_dir(TWIN) / PREDICTIONS_FILE).exists()


@pytest.mark.parametrize("file_name", BASE_COPIES)
def test_freeze_of_a_twin_refuses_until_its_bases_calls_are_committed(jev, git, file_name):
    base = freeze_the_seed(jev, git)
    commit_base_calls(git, base)
    export_and_commit(git)
    append_a_newline(base / file_name)  # changed since its commit

    with pytest.raises(ValueError, match=f"{file_name} is not committed, or has changed since"):
        jev("freeze", TWIN)

    assert not (method_dir(TWIN) / PREDICTIONS_FILE).exists()


@pytest.mark.parametrize(
    ("committed", "message"),
    [
        (False, "scores.csv is not committed, or has changed since"),
        (True, re.escape("than those committed now: ['scores.csv'] differ")),
    ],
    ids=["on disk", "committed"],
)
def test_freeze_of_a_twin_refuses_a_base_file_changed_after_its_export(
    twin_base, jev, git, committed, message
):
    change_a_score(twin_base)
    if committed:
        git.commit(twin_base / SCORES_FILE)

    with pytest.raises(ValueError, match=message):
        jev("freeze", TWIN)

    assert not (method_dir(TWIN) / PREDICTIONS_FILE).exists()


@pytest.mark.parametrize("command", ["export", "freeze", "verify"])
def test_a_twin_refuses_a_base_whose_recorded_commit_a_rewritten_history_lost(
    twin_base, jev, git, command
):
    # Regression: the twin names its base's files by commit; a squash or a rebase of the pull
    # request that holds them keeps every byte but drops those commits from HEAD's history.
    if command == "verify":
        assert jev("freeze", TWIN) == 0
    tuning = method_dir(TWIN) / TUNING_FOLDER
    commit = platt.read_platt_params(tuning)["base_files"]["answers_dev.jsonl"]["commit"]
    squash_history(git)

    with pytest.raises(ValueError, match="which the history of HEAD no longer holds") as caught:
        if command == "export":
            export_twin()
        else:
            jev(command, TWIN)

    path = twin_base / "answers_dev.jsonl"
    assert str(caught.value).startswith(f"{path} is named by the commit {commit}, which the")
    assert "merge it with a merge commit, never squash or rebase it" in str(caught.value)


def test_export_of_a_twin_a_second_time_names_a_base_file_changed_since(twin_base, git):
    # Regression: a second export blamed the twin's record, which it could not write again, for a
    # base file committed again since; it names the base's file, as freeze and verify do.
    tuning = method_dir(TWIN) / TUNING_FOLDER
    before = snapshot(tuning)
    change_a_score(twin_base)
    git.commit(twin_base / SCORES_FILE)

    with pytest.raises(ValueError, match=re.escape("committed now: ['scores.csv'] differ")):
        export_twin()

    assert snapshot(tuning) == before


@pytest.mark.parametrize("command", ["export", "freeze", "verify"])
def test_a_twin_refuses_a_base_file_changed_then_reverted_and_says_a_revert_cannot_help(
    twin_base, jev, git, command
):
    # Regression: the refusal said to restore the base's files from git; a revert brings the bytes
    # back at a new commit, and a twin names each base file by its last commit.
    if command == "verify":
        assert jev("freeze", TWIN) == 0
    path = twin_base / SCORES_FILE
    exported = path.read_bytes()
    change_a_score(twin_base)
    git.commit(path)
    path.write_bytes(exported)  # the revert
    git.commit(path)

    with pytest.raises(ValueError, match="other files of jev-statement-code than those") as caught:
        if command == "export":
            export_twin()
        else:
            jev(command, TWIN)

    message = str(caught.value)
    assert "A revert does not restore a changed base file" in message
    assert "means a new twin, under a new name (docs/adr/0011)" in message


def test_freeze_of_a_twin_refuses_base_scores_that_are_not_its_bases_answers(jev, git):
    base = freeze_the_seed(jev, git)
    commit_base_calls(git, base)
    change_a_score(base)
    git.commit(base / SCORES_FILE)
    export_and_commit(git)  # exported from the changed scores

    with pytest.raises(ValueError, match="give are not its frozen scores.csv: run"):
        jev("freeze", TWIN)


@pytest.mark.usefixtures("twin_base")
def test_freeze_of_a_twin_refuses_other_thresholds_than_its_records(jev, git):
    change_the_recorded_thresholds(method_dir(TWIN))
    git.commit(method_dir(TWIN) / TUNING_FOLDER / PARAMS_FILE)

    with pytest.raises(ValueError, match="but the twin's rule gives"):
        jev("freeze", TWIN)

    assert not (method_dir(TWIN) / PREDICTIONS_FILE).exists()


def test_freeze_of_a_twin_a_second_time_rewrites_nothing(frozen_twin, jev, capsys):
    before = snapshot(frozen_twin)
    capsys.readouterr()

    assert jev("freeze", TWIN) == 0

    assert snapshot(frozen_twin) == before
    base = method_dir("jev-statement-code")
    assert capsys.readouterr().out.splitlines()[:3] == [
        f"13 predictions match the frozen files in {frozen_twin}",
        *(f"{frozen_twin / name} is {base / name}, byte for byte" for name in BASE_COPIES),
    ]


def test_the_frozen_files_of_a_twin_pass_check_frozen_method(frozen_twin, assets, split_file):
    # A composite method over an API method: no answers file of its own.
    assert not list(frozen_twin.glob("answers_*.jsonl"))

    check_frozen_method(TWIN, assets_dir=assets, split=load_split(split_file))


def test_verify_of_a_twin_re_derives_its_files_from_its_bases(frozen_twin, jev, events, capsys):
    before = snapshot(frozen_twin)
    capsys.readouterr()

    assert jev("verify", TWIN) == 0

    assert snapshot(frozen_twin) == before
    assert len(logged_records(events)) == 2  # the two freezes': verify logs nothing
    lines = capsys.readouterr().out.splitlines()
    assert lines[:-2] == [
        f"the frozen files of {TWIN} are complete and consistent with the split",
        f"the run record of {TWIN} holds the recipe of the code",
        "rebuilt 13 requests: each has its answer, under its hash",
        "every completed answer was served by jev-1.13.0",
        "each answers file holds every line of its committed versions, then new ones",
        "the run record and tuning/params.json name the files of jev-statement-code by their "
        "sha256 and last commit, a commit of HEAD's history, and latency.csv and per_problem.csv "
        "are jev-statement-code's",
        "the answers of jev-statement-code give its scores.csv, and the twin's rule the "
        "thresholds of tuning/params.json",
        f"re-derived the tuning/ record of {TWIN} from the dev scores of jev-statement-code: it "
        "is equal",
        "re-derived 13 predictions, their scores and per_problem.csv from the answers: they "
        "equal the frozen ones",
        f"the method folder {frozen_twin / MODEL_FOLDER} holds the re-derived thresholds",
    ]
    assert lines[-2].startswith("dev-oof: macro F1 ")
    assert lines[-1] == "test: macro F1 0.5000, micro F1 1.0000"


def change_the_latency(directory):
    """Add a line to a twin's copy of its base's latency.csv."""
    append_a_newline(directory / LATENCY_FILE)


def change_the_calls(directory):
    """Add a line to a twin's copy of its base's per_problem.csv."""
    append_a_newline(directory / PER_PROBLEM_FILE)


def change_the_rules(directory):
    """Rename a rule in a twin's tuning/rules.csv."""
    path = directory / TUNING_FOLDER / "rules.csv"
    path.write_bytes(path.read_bytes().replace(b"R1", b"R9", 1))


def change_the_recorded_base(directory):
    """Make a twin's run record name another commit of its base's scores.csv."""

    def edit(record):
        record["method_entries"]["base"]["files"][SCORES_FILE]["commit"] = "f" * 40

    rewrite_json(directory / RUN_RECORD_FILE, edit)


def change_the_recorded_thresholds(directory):
    """Make a twin's params.json hold another threshold of math."""
    rewrite_json(
        directory / TUNING_FOLDER / PARAMS_FILE,
        lambda params: params["thresholds"].update(math=0.995),
    )


def drop_the_disclosure(directory):
    """Delete test_informed from a twin's run record, the entry that keeps it never eligible."""
    rewrite_json(
        directory / RUN_RECORD_FILE, lambda record: record["method_entries"].pop("test_informed")
    )


def change_the_recorded_dev(directory):
    """Make a twin's run record say that its dev rows are in sample, unlike its base's."""
    rewrite_json(
        directory / RUN_RECORD_FILE,
        lambda record: record["method_entries"].update(dev_in_sample=True),
    )


def change_the_recorded_state_fields(directory):
    """Make a twin's run record say that its requests send the statement alone."""
    rewrite_json(
        directory / RUN_RECORD_FILE,
        lambda record: record["method_entries"]["settings"].update(state_fields=["statement"]),
    )


def entries_differ(*keys):
    """Return the message of a twin's run record whose entries ``keys`` differ from its freeze's."""
    return re.escape(f"other entries than the freeze of {TWIN} records now: {list(keys)} differ")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (change_the_latency, "latency.csv is not .* byte for byte"),
        (change_the_calls, "per_problem.csv is not .* byte for byte"),
        (change_the_rules, f"holds another record of the threshold rule of {TWIN}"),
        (change_the_recorded_base, "names other files of jev-statement-code than those"),
        (drop_the_disclosure, entries_differ("test_informed")),
        (change_the_recorded_dev, entries_differ("dev_in_sample")),
        (change_the_recorded_state_fields, entries_differ("settings")),
        (change_the_recorded_thresholds, "but the twin's rule gives"),
        (flip_a_prediction, re.escape("The frozen files ['predictions.csv']")),
        (change_a_score, re.escape("The frozen files ['scores.csv']")),
    ],
)
def test_verify_of_a_twin_refuses_a_changed_file(frozen_twin, jev, change, message):
    change(frozen_twin)

    with pytest.raises(ValueError, match=message):
        jev("verify", TWIN)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (drop_the_disclosure, entries_differ("test_informed")),
        (change_the_recorded_dev, entries_differ("dev_in_sample")),
        (change_the_recorded_state_fields, entries_differ("settings")),
    ],
)
def test_freeze_of_a_twin_a_second_time_refuses_a_changed_run_record(
    frozen_twin, jev, change, message
):
    # Regression: test_informed keeps both twins out of the command line's eligible methods (and
    # dev_in_sample the GEPA twin too), and settings names what the twin's requests send.
    change(frozen_twin)
    before = snapshot(frozen_twin)

    with pytest.raises(ValueError, match=message):
        jev("freeze", TWIN)

    assert snapshot(frozen_twin) == before
