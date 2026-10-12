"""Fixtures shared by the test folders: datasets, run records, answers, model parameters, MLflow.

Among the datasets, ``raw_dev`` and ``focus_raw_dev`` are small dev sets whose words make focus
tags learnable; ``focus_tag_model`` is a tag model fitted on the second, and ``log_tag_model``
saves and logs it as ``freeze`` does. ``cli_jev`` stands in for TypeSafe's Jev behind the
command line, and ``cli_challenge`` for the challenge dataset. ``agent_python`` runs a script in a
fresh interpreter of a coding agent's shell, where a bare ``import mlflow`` starts MLflow's
telemetry and its hint, and ``fresh_import`` tells what importing a module there let MLflow do.
Every test also fails once the session has opened a connection outside localhost. The session
draws its figures on matplotlib's Agg backend, as CI does.
"""

import ipaddress
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from importlib.metadata import version
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

# MLflow decides once, when it is first imported, whether to send telemetry and hint at coding
# agents. Importing the tracking package turns both off; this file imports mlflow only inside its
# fixtures, so the package comes first, before any test file imports mlflow.
import codeforces_tags.tracking  # noqa: F401  # pylint: disable=unused-import
from codeforces_tags.config import DATASET_SHA256, FOCUS_TAGS, MLFLOW_ENV, MLFLOW_ENV_DEFAULTS
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.evaluation.calibration import probabilities_by_method
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.methods import save_method
from codeforces_tags.methods.typesafe_jev.settings import MODEL as JEV_MODEL
from codeforces_tags.model.params import BLOCKS, ModelParams
from codeforces_tags.model.tag_model import NAME
from codeforces_tags.model.training import fit_tag_model, out_of_fold
from codeforces_tags.provenance import RunRecord, sha256_json
from codeforces_tags.tracking.environment import use_store

# Agg draws without a window, as on CI, which has no display. With one, matplotlib picks TkAgg,
# whose closed figure leaves Tk variables and images in reference cycles: the garbage collector
# may then delete them on a worker thread (MLflow's evaluation runs its predictions on one), where
# Tk raises in their __del__ and pytest reports an unraisable exception. Set before any test file
# imports pyplot.
matplotlib.use("Agg")


def remote_host(event, args):
    """Return the host a socket audit event connects to or resolves, unless it is local.

    Local means no host, ``localhost``, a loopback address or a Unix socket; any other audit
    event returns None too.
    """
    if event == "socket.getaddrinfo":
        host = args[0]
    elif event == "socket.connect" and isinstance(args[1], tuple):
        host = args[1][0]
    else:
        return None
    if isinstance(host, bytes):
        host = host.decode()
    if host in (None, "", "localhost"):
        return None
    with suppress(ValueError):
        if ipaddress.ip_address(host).is_loopback:
            return None
    return host


@pytest.fixture(name="remote_connections", scope="session")
def remote_connections_fixture():
    """Return the connections the session opens outside localhost from now on, as a list.

    An audit hook, which stays until the process ends, appends ``(event, host, thread)`` for
    each: MLflow's telemetry, for one, connects from threads of its own.
    """
    opened = []

    def record(event, args):
        host = remote_host(event, args)
        if host is not None:
            opened.append((event, host, threading.current_thread().name))

    # CPython runs an audit hook with tracing off unless it says so (PEP 578): coverage sees it.
    record.__cantrace__ = True
    sys.addaudithook(record)
    return opened


@pytest.fixture(autouse=True)
def no_connection_outside_localhost(remote_connections):
    """Fail a test if, by its end, the session has opened a connection outside localhost."""
    yield
    opened = remote_connections.copy()
    remote_connections.clear()
    assert not opened, f"connections outside localhost: {opened}"


@pytest.fixture(name="make_record")
def make_record_fixture():
    """Return a builder for one valid sample, with fields overridable by keyword."""

    def make(**overrides):
        record = {
            "src_uid": "uid-0",
            "prob_desc_description": "Find $$$n$$$.",
            "prob_desc_time_limit": "2 seconds",
            "prob_desc_memory_limit": "256 megabytes",
            "prob_desc_sample_inputs": '["1\\n4"]',
            "prob_desc_sample_outputs": '["2"]',
            "prob_desc_created_at": "1602407100",
            "difficulty": 1000,
            "lang": "Python 3",
            "source_code": "print(1)",
            "tags": ["math", "greedy"],
        }
        return record | overrides

    return make


@pytest.fixture(name="make_raw_record")
def make_raw_record_fixture(make_record):
    """Return a builder for one valid sample with every raw column a tagging method reads.

    It adds the input and output specifications and the notes (``RAW_INPUT_COLUMNS``) to
    ``make_record``'s fields, all overridable by keyword.
    """

    def make(**overrides):
        specs = {
            "prob_desc_input_spec": "One integer $$$n$$$.",
            "prob_desc_output_spec": "Print one integer.",
            "prob_desc_notes": "NoteThe answer is $$$2$$$.",
        }
        return make_record(**(specs | overrides))

    return make


@pytest.fixture(name="placeholder_record")
def placeholder_record_fixture():
    """Return a builder of run records with made-up fingerprints, fields overridable by keyword.

    ``placeholder_record(method, **changes)`` returns the ``RunRecord`` of ``method``
    (``label_prior`` by default), whose recipe is ``sha256_json({"method": method})``.
    """

    def build(method="label_prior", **changes):
        values = {
            "method": method,
            "recipe_sha256": sha256_json({"method": method}),
            "git": {"commit": "0" * 40, "code_dirty": False},
            "environment": {"python": "3.12.12", "packages": {"pandas": version("pandas")}},
            "dataset_sha256": DATASET_SHA256,
            "split_sha256": "1" * 64,
            "verdicts_sha256": "2" * 64,
            "settings": {"FOCUS_TAGS": list(FOCUS_TAGS)},
            "allow_dirty": False,
            "created_at": "2026-10-07T00:00:00+00:00",
            "method_entries": {"supersedes": []},
        }
        return RunRecord(**(values | changes))

    return build


@pytest.fixture(name="write_samples")
def write_samples_fixture(tmp_path):
    """Return a writer that saves ``{sample_id: record}`` as JSON files and returns the dir."""

    def write(records):
        for sample_id, record in records.items():
            path = tmp_path / f"sample_{sample_id}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
        return tmp_path

    return write


@pytest.fixture(name="mlflow_environment")
def mlflow_environment_fixture(monkeypatch, tmp_path):
    """Clear MLflow's environment for the test, then restore it and forget the store's URI.

    Until the test calls ``use_store``, MLflow's store is a file under ``tmp_path``: a client
    made too early never writes MLflow's default ``./mlflow.db`` into the repository. The
    ``store`` fixture asks for it, and every tracking test does, in ``tests/tracking/conftest.py``.
    """
    for name in (*MLFLOW_ENV, *MLFLOW_ENV_DEFAULTS):
        # setenv records the value to restore, even an absent one, which delenv alone would
        # not: what mlflow_env() sets later would outlive the test.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'default.db'}")
    yield
    import mlflow  # pylint: disable=import-outside-toplevel

    mlflow.set_tracking_uri(None)


@pytest.fixture(name="blank_database", scope="session")
def blank_database_fixture(tmp_path_factory):
    """Return a sqlite file holding MLflow's tables, made once: migrating them takes seconds."""
    import mlflow  # pylint: disable=import-outside-toplevel

    database = tmp_path_factory.mktemp("blank_store") / "mlflow.db"
    mlflow.MlflowClient(f"sqlite:///{database}").search_experiments()
    return database


@pytest.fixture(name="store")
def store_fixture(tmp_path, blank_database, mlflow_environment):  # pylint: disable=unused-argument
    """Return the keyword arguments of ``use_store`` for a fresh store under ``tmp_path``.

    Each test's store is a copy of an empty sqlite store migrated once per session, and MLflow's
    environment is restored after the test, whatever ``mlflow_env()`` set.
    """
    database = tmp_path / "store" / "mlflow.db"
    database.parent.mkdir()
    shutil.copy(blank_database, database)
    return {"tracking_uri": f"sqlite:///{database}", "artifact_dir": tmp_path / "artifacts"}


@pytest.fixture(name="logged_model")
def logged_model_fixture(store):
    """Return an empty logged model, logged in a run of the store's experiment."""
    import mlflow  # pylint: disable=import-outside-toplevel

    experiment_id = use_store(**store)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        pass
    return mlflow.initialize_logged_model(
        name="tagging_method", source_run_id=run.info.run_id, experiment_id=experiment_id
    )


# The variables that turn MLflow's telemetry off when it is imported, besides its own: pytest sets
# PYTEST_CURRENT_TEST while a test runs, and GitHub Actions sets CI and GITHUB_ACTIONS.
TELEMETRY_OFF = {"DO_NOT_TRACK", "PYTEST_CURRENT_TEST", "CI", "GITHUB_ACTIONS"}
# The logger of MLflow's hint to coding agents, in each line it prints.
AGENT_HINT = "mlflow.agent.hint"
# Imports the module named by argv[1], then prints whether MLflow made the telemetry client that
# sends its usage events.
FRESH_IMPORT = """
import importlib, sys
importlib.import_module(sys.argv[1])
client = sys.modules.get("mlflow.telemetry.client")
print(client is not None and client.get_telemetry_client() is not None)
"""
# What FRESH_IMPORT may print, read as whether MLflow made its telemetry client: anything else
# is an error, never a pass.
TELEMETRY_PRINTED = {"True": True, "False": False}


@pytest.fixture(name="agent_python")
def agent_python_fixture(tmp_path):
    """Return a runner of scripts in fresh interpreters of a coding agent's shell.

    In that shell a bare ``import mlflow`` starts telemetry and hints: a coding agent drives it
    (``CLAUDECODE``), and it holds no MLflow variable and none of ``TELEMETRY_OFF``. MLflow's
    installation id is written under ``tmp_path``, not ``~/.config``. The runner takes a script
    and its arguments, runs them with ``python -c`` and returns the output and the error output;
    a script that fails raises ``subprocess.CalledProcessError``.
    """
    kept = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("MLFLOW_") and name not in TELEMETRY_OFF
    }
    env = kept | {"CLAUDECODE": "1", "XDG_CONFIG_HOME": str(tmp_path)}

    def run(script, *args):
        result = subprocess.run(
            [sys.executable, "-c", script, *args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout, result.stderr

    return run


@pytest.fixture(name="fresh_import")
def fresh_import_fixture(agent_python):
    """Return a function that imports a module in a fresh interpreter of the agent shell.

    It takes the module's name and returns ``{"telemetry": bool, "agent_hint": bool}``: whether
    MLflow made the telemetry client that sends its usage events, and whether it printed its hint
    to coding agents. tests/tracking/test_environment.py checks that a bare ``import mlflow``
    does both there: the control of every check that a module does neither.
    """

    def fresh_import(module):
        output, errors = agent_python(FRESH_IMPORT, module)
        return {"telemetry": TELEMETRY_PRINTED[output.strip()], "agent_hint": AGENT_HINT in errors}

    return fresh_import


@pytest.fixture(name="raw_problems")
def raw_problems_fixture():
    """Return three raw problems as ``select_part(read_samples(), split, part)`` gives them.

    They come with their tags, fold and group columns, out of sample id order; problem 30 has no
    notes and problem 20 no input specification.
    """
    return pd.DataFrame(
        {
            "prob_desc_description": [
                "Count the leaves of a tree.",
                "Reverse each of the strings.",
                "Print the gcd, a number theory classic.",
            ],
            "prob_desc_input_spec": ["A tree.", "Some strings.", np.nan],
            "prob_desc_output_spec": ["One integer.", "The strings.", "One integer."],
            "prob_desc_notes": [np.nan, "NoteEach string is short.", "NoteUse math.gcd."],
            "prob_desc_sample_inputs": ['["3\\n1 2\\n1 3"]', '["abc"]', '["4 6"]'],
            "prob_desc_sample_outputs": ['["2"]', '["cba"]', '["2"]'],
            "source_code": [
                "n = int(input())\nprint(n - 1)\n",
                "s = input()\nprint(s[::-1])\n",
                "print(2)",
            ],
            "tags": [["trees"], ["strings"], ["math", "number theory"]],
            "fold": [0, 1, 2],
            "group": [5, 6, 7],
        },
        index=pd.Index([30, 10, 20], name="sample_id"),
    )


@pytest.fixture(name="make_answer")
def make_answer_fixture():
    """Return a builder of the answer to problem 1, with fields overridable by keyword.

    The answer is a complete one of ``fake-text``: its text names two tags, its model is one that
    ``config.LLM_PRICES`` prices, and the fields that the runner stamps are left at their defaults.
    """

    def make(**changes):
        values = {
            "request_sha256": "a" * 64,
            "sample_id": 1,
            "method": "fake-text",
            "provider": "openai",
            "output_text": "dp, math",
            "status": "completed",
            "served_model": "gpt-6-luna",
            "response_id": "resp_1",
            "request_id": "req_1",
            "usage": {
                "input_tokens": 100,
                "cached_tokens": 0,
                "output_tokens": 5,
                "reasoning_tokens": 0,
            },
            "extra": {"openai-processing-ms": "321"},
            "started_ns": 1_791_000_000_000_000_000,
            "latency_s": 0.5,
            "retries": 0,
            "created_at": "2026-10-07T10:00:00+00:00",
            "completed_at": "2026-10-07T10:00:00.500000+00:00",
        }
        return LlmAnswer(**(values | changes))

    return make


# What the command line's fake Jev answers about every problem, by focus tag: math and number
# theory reach the frozen thresholds of jev-statement-code (0.335 and 0.695), no other tag does.
CLI_JEV_PROBABILITIES = dict(zip(FOCUS_TAGS, (0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1), strict=True))
# The input tokens it bills each answer: Jev's output tokens are free.
CLI_JEV_INPUT_TOKENS = 3000


@dataclass
class CliJev:
    """Stands in for ``client.jev_session`` in the command line's tests: nothing leaves the process.

    Each request gets at once an answer of ``probabilities``, billed ``input_tokens`` input
    tokens, unless ``failure`` is set: the request raises it, or only the requests of the
    problems of ``failing`` when it is set. The problems of ``refused`` get the line of a request
    the API refused for itself alone (``api_error:422``, no text, no token), as ``client`` writes
    it.

    Attributes:
        build: The ``make_answer`` builder.
        asked: The sample id of every request, in the order they came.
        failure: What a request raises; None to answer them.
        failing: The sample ids whose requests raise ``failure``; None for every request.
        refused: The sample ids whose requests the API refuses.
        probabilities: Each focus tag's probability, in every answer.
        input_tokens: The input tokens billed for each answer.
    """

    build: Callable[..., LlmAnswer]
    asked: list[int] = field(default_factory=list)
    failure: BaseException | None = None
    failing: set[int] | None = None
    refused: set[int] = field(default_factory=set)
    probabilities: dict[str, float] = field(default_factory=lambda: dict(CLI_JEV_PROBABILITIES))
    input_tokens: int = CLI_JEV_INPUT_TOKENS

    @asynccontextmanager
    async def session(self, name):
        """Yield the ask of method ``name``, which answers or raises ``failure``."""

        async def ask(request):
            self.asked.append(request.sample_id)
            if self.failure is not None and (
                self.failing is None or request.sample_id in self.failing
            ):
                raise self.failure
            return self._answer(request, name, refused=request.sample_id in self.refused)

        yield ask

    def _answer(self, request, name, *, refused):
        """Build the line of a request: Jev's answer, or the line of a refused request."""
        tokens = 0 if refused else self.input_tokens
        return self.build(
            request_sha256=request.sha256(),
            sample_id=request.sample_id,
            method=name,
            provider="typesafe",
            status="api_error:422" if refused else "completed",
            output_text=None if refused else json.dumps(self.probabilities, separators=(",", ":")),
            served_model=None if refused else JEV_MODEL,
            usage={
                "input_tokens": tokens,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
            extra={},
            failed=refused,
        )


@pytest.fixture(name="cli_jev")
def cli_jev_fixture(monkeypatch, make_answer):
    """Replace TypeSafe's client session by a ``CliJev`` and return it."""
    fake = CliJev(make_answer)
    monkeypatch.setattr("codeforces_tags.methods.typesafe_jev.client.jev_session", fake.session)
    return fake


@pytest.fixture(name="cli_challenge")
def cli_challenge_fixture(tmp_path, write_samples, make_raw_record, monkeypatch):
    """Stand in for the challenge dataset in the command line: 16 problems and their split.

    Dev problems 1-8 and test problems 9-16, problem i tagged with the ((i - 1) % 8)-th focus tag
    (and greedy), so a method that predicts the truth scores a macro F1 of 1. In
    ``codeforces_tags.cli``, ``check_dataset`` passes on these problems and ``SPLIT_FILE`` is
    their split.

    Returns:
        ``data_dir`` and ``split_path``; ``labels``, the problems' focus labels; ``dev`` and
        ``test``, the sample ids of each part.
    """
    dev, test = list(range(1, 9)), list(range(9, 17))
    labels = pd.DataFrame(
        np.tile(np.eye(len(FOCUS_TAGS), dtype=np.int64), (2, 1)),
        index=pd.Index(dev + test, name="sample_id"),
        columns=list(FOCUS_TAGS),
    )
    tags = {sample_id: [FOCUS_TAGS[(sample_id - 1) % 8], "greedy"] for sample_id in labels.index}
    data_dir = write_samples(
        {sample_id: make_raw_record(tags=tag) for sample_id, tag in tags.items()}
    )
    split = pd.DataFrame(
        {"part": ["dev"] * 8 + ["test"] * 8, "fold": [1, 2, 3, 4, 5, 1, 2, 3] + [0] * 8},
        index=labels.index,
    ).assign(group=range(16))
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path)
    monkeypatch.setattr("codeforces_tags.cli.check_dataset", lambda data_dir: None)
    monkeypatch.setattr("codeforces_tags.cli.SPLIT_FILE", split_path)
    return {
        "data_dir": data_dir,
        "split_path": split_path,
        "labels": labels,
        "dev": dev,
        "test": test,
    }


@pytest.fixture(name="make_params")
def make_params_fixture():
    """Return a builder of valid ``ModelParams`` for small data, with fields overridable by keyword.

    Unigrams, every term kept (``min_df`` 1, ``max_df`` 1.0), all three blocks.
    """

    def make(**overrides):
        values = {
            "classifier": "logistic_regression",
            "c": 1.0,
            "class_weight": "none",
            "include_notes": False,
            "statement_ngram_max": 1,
            "statement_min_df": 1,
            "statement_max_df": 1.0,
            "statement_sublinear_tf": False,
            "code_weight": 0.5,
            "code_ngram_max": 1,
            "code_min_df": 1,
            "code_sublinear_tf": False,
            "cue_weight": 0.5,
            "blocks": BLOCKS,
        }
        return ModelParams(**(values | overrides))

    return make


@pytest.fixture(name="calibration_data")
def calibration_data_fixture():
    """Return synthetic out-of-fold inputs of the calibration, on the 8 focus tags.

    A namespace of 240 problems indexed by ``sample_id``, in 3 folds and 120 near-duplicate
    groups of 1, 2, 3 and 2 problems in turn (a group never spans two folds), so that the
    resamples of the group bootstrap draw different numbers of problems, as on the dev set:
    ``labels`` (0/1, each tag with positives and negatives in every fold), ``svm_scores``
    (margins that rise with the label, mostly within [-2, 2]), ``logits`` (wider scores, as a
    logistic regression gives), ``folds`` and ``groups``.
    """
    rng = np.random.default_rng(0)
    n_problems = 240
    index = pd.Index(np.arange(n_problems) * 3 + 7, name="sample_id")
    tags = pd.Index(FOCUS_TAGS, name="label")
    labels = (rng.random((n_problems, len(tags))) < np.linspace(0.4, 0.15, len(tags))).astype(int)
    margins = (2 * labels - 1) * 0.6 + rng.normal(0, 0.7, labels.shape)
    n_groups = 120
    groups = np.repeat(np.arange(n_groups), np.resize([1, 2, 3, 2], n_groups))
    return SimpleNamespace(
        labels=pd.DataFrame(labels, index=index, columns=tags),
        svm_scores=pd.DataFrame(margins, index=index, columns=tags),
        logits=pd.DataFrame(2.5 * margins - 1, index=index, columns=tags),
        folds=pd.Series(groups % 3, index=index, name="fold"),
        groups=pd.Series(groups, index=index, name="group"),
    )


@pytest.fixture(name="probabilities")
def probabilities_fixture(calibration_data):
    """Return the four probabilities of the calibration data (isotonic's hold exact 0s)."""
    data = calibration_data
    return probabilities_by_method(data.svm_scores, data.logits, data.labels, data.folds)


# The four kinds of problem of raw_dev, in turn. Each focus kind has its own statement and
# solution words and its own sample I/O cue (graph edge list, letters, player word); the last
# kind has no focus tag, and every other one of its problems has no tag at all.
KINDS = (
    {
        "tags": ["graphs", "dfs and similar"],
        "statement": "Given a graph with $$$n$$$ vertices and $$$m$$$ edges, find the shortest "
        "path from vertex $$$1$$$ to every other vertex.",
        "source_code": "import sys\nn, m = map(int, sys.stdin.readline().split())\n"
        "adj = [[] for _ in range(n + 1)]\nfor _ in range(m):\n"
        "    u, v = map(int, sys.stdin.readline().split())\n    adj[u].append(v)\n"
        "print(len(adj[1]))",
        "sample_input": "3 3\n1 2\n2 3\n3 1",
        "sample_output": "2",
    },
    {
        "tags": ["strings", "implementation"],
        "statement": "You are given a string $$$s$$$ of lowercase letters. Find the longest "
        "palindrome substring of $$$s$$$.",
        "source_code": "s = input().strip()\nprint(s[::-1] if s == s[::-1] else s[:3])",
        "sample_input": "abacaba",
        "sample_output": "aba",
    },
    {
        "tags": ["games"],
        "statement": "Alice and Bob play a game with a pile of $$$n$$$ stones and take turns. "
        "Who wins if both players play optimally?",
        "source_code": "n = int(input())\nprint('Alice' if n % 2 else 'Bob')",
        "sample_input": "3",
        "sample_output": "Alice",
    },
    {
        "tags": ["greedy"],
        "statement": "Given an array of $$$n$$$ integers, find the maximum sum of its elements "
        "after at most one swap.",
        "source_code": "n = int(input())\na = list(map(int, input().split()))\nprint(sum(a))",
        "sample_input": "3\n1 2 3",
        "sample_output": "6",
    },
)
# Sentences shared by every kind, so that the tags are not the only words that vary.
FILLERS = (
    "Print the answer modulo $$$10^9+7$$$.",
    "The input has several test cases.",
    "Output a single integer.",
    "It is guaranteed that $$$1 \\le n \\le 10^5$$$.",
    "Read the input from standard input.",
)
# The kinds of the focus tags that KINDS lacks, for focus_raw_dev.
MORE_FOCUS_KINDS = (
    {
        "tags": ["math"],
        "statement": "Compute the sum of the digits of the number $$$n$$$ and their product.",
        "source_code": "n = input().strip()\nprint(sum(int(digit) for digit in n))",
        "sample_input": "123",
        "sample_output": "6",
    },
    {
        "tags": ["math", "number theory"],
        "statement": "Find the greatest common divisor of $$$a$$$ and $$$b$$$, and tell whether "
        "it is a prime number.",
        "source_code": "from math import gcd\na, b = map(int, input().split())\nprint(gcd(a, b))",
        "sample_input": "12 18",
        "sample_output": "6",
    },
    {
        "tags": ["trees", "dfs and similar"],
        "statement": "Given a rooted tree with $$$n$$$ vertices, find the depth of every leaf of "
        "the tree.",
        "source_code": "n = int(input())\nparent = [0] + list(map(int, input().split()))\n"
        "depth = [0] * n\nfor v in range(1, n):\n    depth[v] = depth[parent[v] - 1] + 1\n"
        "print(max(depth))",
        "sample_input": "4\n1 1 2",
        "sample_output": "2",
    },
    {
        "tags": ["geometry"],
        "statement": "Given three points on the plane, find the area of the triangle whose "
        "corners they are.",
        "source_code": "x1, y1, x2, y2, x3, y3 = map(int, input().split())\n"
        "print(abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2)",
        "sample_input": "0 0 1 0 0 1",
        "sample_output": "0.5",
    },
    {
        "tags": ["probabilities"],
        "statement": "A fair coin is tossed $$$n$$$ times. Find the expected number of heads and "
        "the probability of no tail.",
        "source_code": "n = int(input())\nprint(n / 2, 0.5 ** n)",
        "sample_input": "3",
        "sample_output": "1.5 0.125",
    },
)
# Every focus tag's kind, then the kind without a focus tag.
FOCUS_KINDS = (*KINDS[:3], *MORE_FOCUS_KINDS, KINDS[3])
PROBLEMS = 60  # 15 of each kind
FOCUS_PROBLEMS = 54  # 6 of each focus kind: 2 in each fold
FOLDS = 3


def raw_rows(kinds, count):
    """Return ``count`` raw problems, one of each kind in turn, as ``read_samples()`` gives them.

    Only the fields a tagging method reads (``RAW_INPUT_COLUMNS``: the samples as JSON strings,
    a missing field NaN) and the tags. Index ``sample_id`` from 100; within each kind, every
    third problem has no input specification, every other one no notes, and every other problem
    of a kind without a focus tag has no tag at all.
    """
    rows = []
    for position in range(count):
        kind = kinds[position % len(kinds)]
        rank = position // len(kinds)  # the problem's rank within its kind
        untagged = not set(kind["tags"]) & set(FOCUS_TAGS) and rank % 2 == 1
        rows.append(
            {
                "prob_desc_description": f"{kind['statement']} {FILLERS[position % len(FILLERS)]}",
                "prob_desc_input_spec": "The first line holds $$$n$$$." if rank % 3 else np.nan,
                "prob_desc_output_spec": "Print the answer.",
                "prob_desc_notes": np.nan if rank % 2 else "NoteThe first sample is explained.",
                "prob_desc_sample_inputs": json.dumps([kind["sample_input"]]),
                "prob_desc_sample_outputs": json.dumps([kind["sample_output"]]),
                "source_code": kind["source_code"],
                "tags": [] if untagged else list(kind["tags"]),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(range(100, 100 + count), name="sample_id"))


@pytest.fixture(name="raw_dev")
def raw_dev_fixture():
    """Return 60 raw dev problems, 15 of each kind: graphs, strings, games, no focus tag.

    Raw rows as ``read_samples()`` gives them (``raw_rows``).
    """
    return raw_rows(KINDS, PROBLEMS)


@pytest.fixture(name="focus_raw_dev")
def focus_raw_dev_fixture():
    """Return 54 raw dev problems carrying all eight focus tags, as ``select_part`` gives them.

    6 problems of each kind of ``FOCUS_KINDS`` (``raw_rows``), with a ``fold`` column (each
    kind's problems dealt to folds 0, 1, 2 in turn: every focus tag is positive in every fold)
    and a ``group`` column (27 near-duplicate pairs: each problem shares its group with the
    problem of its kind 27 places on, in the same fold), so that a group bootstrap and a
    bootstrap of single problems give different intervals.
    """
    rows = raw_rows(FOCUS_KINDS, FOCUS_PROBLEMS)
    position = np.arange(FOCUS_PROBLEMS)
    folds = position // len(FOCUS_KINDS) % FOLDS
    return rows.assign(fold=folds, group=position % (FOCUS_PROBLEMS // 2))


@pytest.fixture(name="focus_tag_model")
def focus_tag_model_fixture(focus_raw_dev, make_params):
    """Return a tag model of the eight focus tags, fitted with every block and the notes.

    It reads every raw text field, so a missing one handed over as anything but missing would
    change its statement.
    """
    labels = focus_labels(focus_raw_dev)
    params = make_params(include_notes=True)
    return fit_tag_model(params, focus_raw_dev, labels, focus_raw_dev["fold"])


@pytest.fixture(name="log_tag_model")
def log_tag_model_fixture(focus_raw_dev, focus_tag_model, store, placeholder_record):
    """Return a function that saves the focus tag model and logs it as ``freeze`` does.

    ``log_tag_model(folder)`` saves ``focus_tag_model`` to ``folder`` (``save_method``), logs it
    to the test's store with ``log_method``, under ``placeholder_record("tfidf")`` and with its
    out-of-fold dev metrics, and returns the id of the logged model.
    """
    from codeforces_tags.tracking.methods import (  # pylint: disable=import-outside-toplevel
        log_method,
        standard_metrics,
    )

    def log(folder):
        save_method(focus_tag_model, folder)
        labels = focus_labels(focus_raw_dev)
        oof = out_of_fold(focus_tag_model.estimator, labels, focus_raw_dev["fold"])
        metrics = {"dev-oof": standard_metrics(labels, oof.predictions, focus_tag_model.tags)}
        split = focus_raw_dev[["fold", "group"]].assign(part="dev")
        return log_method(folder, placeholder_record(NAME), metrics, split=split, **store)

    return log
