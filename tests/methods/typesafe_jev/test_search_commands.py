"""Tests for the commands of the search over Jev's words: tune, export and log-search.

The handlers are called with what ``__main__``'s gates hand them; test_main.py tests the gates.
No test searches: ``FakeSearch`` stands for ``run_search`` and ``FakeLiveLog`` for the live
MLflow log; ``export`` runs the real ``export_search`` on the search that the ``search_run``
fixture (conftest.py) writes by hand. ``log-search`` runs once with a recorder for
``log_search`` and once with the real one, on the fresh store of the ``store`` fixture.
"""

import functools
import json
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags import methods
from codeforces_tags.config import MLFLOW_ARTIFACT_DIR, MLFLOW_TRACKING_URI
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev import __main__ as cli
from codeforces_tags.methods.typesafe_jev import client, search_commands
from codeforces_tags.methods.typesafe_jev.export import export_search
from codeforces_tags.methods.typesafe_jev.jev_cache import CostCapSpent, SearchInterrupted
from codeforces_tags.methods.typesafe_jev.search_record import read_search_record
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    PARAMS_FILE,
    RUN_ID_FILE,
    SEARCH_EXPERIMENT,
    SEARCH_FOLDER,
    SEARCH_MAX_COST_USD,
    SEARCH_MAX_ITERATIONS,
    SEARCH_RUN_NAME,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.tuning import SearchOutcome

# What the fake search returns: it stopped at its cost cap after 12 iterations.
OUTCOME = SearchOutcome(
    stop_reason="cost",
    iterations=12,
    best_index=3,
    cost_usd={"jev": 1.25, "luna": 0.5, "total": 1.75},
)
# What tune's gates hand its handler: the dev rows, with their fold, and the seed's answers.
RAW_DEV = pd.DataFrame({"fold": [1, 4]}, index=pd.Index([5, 6], name="sample_id"))
SEED_ANSWERS = ["the committed answer to problem 5"]


class FakeSearch:  # pylint: disable=too-few-public-methods
    """Stands for ``run_search``: records each call, then raises ``error`` or returns OUTCOME.

    Attributes:
        error: What each call raises; None to return ``OUTCOME``.
        calls: The data, folder and options of each call, in order.
    """

    def __init__(self):
        """Start with no call and no error."""
        self.error = None
        self.calls = []

    def __call__(self, data, folder, **options):
        """Record the call, then raise the error or return the outcome."""
        self.calls.append({"data": data, "folder": folder, **options})
        if self.error is not None:
            raise self.error
        return OUTCOME


class FakeLiveLog:  # pylint: disable=too-few-public-methods
    """Stands for ``search_tracking.LiveSearchLog``: keeps its store and counts its closings."""

    def __init__(self, logs, **store):
        """Keep the store's options and join the list of opened logs."""
        self.store = store
        self.closed = 0
        logs.append(self)

    def close(self):
        """Count one closing."""
        self.closed += 1


@pytest.fixture(name="paths", autouse=True)
def paths_fixture(monkeypatch, tmp_path):
    """Keep the search folder and the methods' frozen files under tmp_path; return its root."""
    monkeypatch.setattr(search_commands, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path / "assets")
    return tmp_path


@pytest.fixture(name="search")
def search_fixture(monkeypatch):
    """Replace run_search by a FakeSearch, and search_data by a tag of its rows; return the fake."""
    fake = FakeSearch()
    monkeypatch.setattr(search_commands, "run_search", fake)
    monkeypatch.setattr(search_commands, "search_data", lambda raw: ("data of", list(raw.index)))
    return fake


@pytest.fixture(name="live_logs")
def live_logs_fixture(monkeypatch):
    """Replace the live MLflow log of the search by FakeLiveLog; return the logs it opens."""
    logs = []
    fake = functools.partial(FakeLiveLog, logs)
    monkeypatch.setattr(search_commands.search_tracking, "LiveSearchLog", fake)
    return logs


def run_tune(*options):
    """Run tune's handler with the options, on RAW_DEV and SEED_ANSWERS; return its status."""
    args = cli.parse_args(["tune", "--method", GEPA_NAME, *options])
    return search_commands.tune(args, RAW_DEV, SEED_ANSWERS)


def test_parse_args_reads_the_search_commands_and_their_defaults():
    tune = cli.parse_args(["tune", "--method", GEPA_NAME])
    export = cli.parse_args(["export", "--method", GEPA_NAME])
    log = cli.parse_args(["log-search", "--method", GEPA_NAME])

    assert vars(tune) == {
        "command": "tune",
        "method": GEPA_NAME,
        "max_cost_usd": SEARCH_MAX_COST_USD,
        "max_iterations": SEARCH_MAX_ITERATIONS,
        "no_mlflow": False,
        "allow_dirty": False,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_dir": MLFLOW_ARTIFACT_DIR,
    }
    assert (SEARCH_MAX_COST_USD, SEARCH_MAX_ITERATIONS) == (15.0, 200)
    assert vars(export) == {"command": "export", "method": GEPA_NAME, "allow_dirty": False}
    assert vars(log) == {
        "command": "log-search",
        "method": GEPA_NAME,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_dir": MLFLOW_ARTIFACT_DIR,
    }


@pytest.mark.parametrize(
    ("command", "recorded"),
    [("run", True), ("freeze", True), ("tune", False), ("export", False)],
)
def test_allow_dirty_says_recorded_only_where_a_frozen_file_records_the_git_state(
    command, recorded, monkeypatch, capsys
):
    # run stamps each answer with the commit, freeze writes it in the run record; neither
    # outcome.json nor the exported tuning/ holds it.
    monkeypatch.setenv("COLUMNS", "200")

    with pytest.raises(SystemExit):
        cli.parse_args([command, "--help"])

    help_text = capsys.readouterr().out
    assert "accept uncommitted code" in help_text
    assert ("(recorded)" in help_text) is recorded


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["tune", "--method", "jev-statement-code"], "invalid choice: 'jev-statement-code'"),
        (["export", "--method", "jev-statement"], "invalid choice: 'jev-statement'"),
        (["tune", "--method", "jev-statement-code-platt"], "invalid choice"),
        (["log-search", "--method", "jev-statement-code-gepa-platt"], "invalid choice"),
        (["tune", "--method", GEPA_NAME, "--max-iterations", "0"], "at least 1 iteration, not 0"),
        (["tune", "--method", GEPA_NAME, "--max-cost-usd", "nan"], "above 0 USD, not nan"),
        (["log-search", "--method", GEPA_NAME, "--allow-dirty"], "unrecognized arguments"),
    ],
)
def test_parse_args_refuses_another_method_or_option_for_the_search_commands(
    options, message, capsys
):
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args(options)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "method", [GEPA_NAME, "jev-statement-code-platt", "jev-statement-code-gepa-platt"]
)
def test_export_takes_the_searched_method_or_a_calibrated_twin(method):
    assert cli.parse_args(["export", "--method", method]).method == method


def test_tune_searches_from_the_seed_answers_and_logs_live(search, live_logs, paths, capsys):
    assert run_tune("--max-cost-usd", "3", "--max-iterations", "7") == 0

    folder = paths / "experiments" / GEPA_NAME / SEARCH_FOLDER
    [log] = live_logs
    assert search.calls == [
        {
            "data": ("data of", [5, 6]),
            "folder": folder,
            "max_cost_usd": 3.0,
            "max_iterations": 7,
            "committed_seed_answers": SEED_ANSWERS,
            "jev_session": client.jev_session,
            "live": log,
        }
    ]
    assert log.store == {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_dir": MLFLOW_ARTIFACT_DIR,
        "run_file": folder / RUN_ID_FILE,
    }
    assert log.closed == 1
    assert folder.is_dir()
    assert capsys.readouterr().out.splitlines() == [
        "the search stopped at its cost cap after 12 iterations, for $1.75 (Jev $1.25, Luna "
        "$0.50); GEPA's best candidate is 3.",
        f"Next: python -m codeforces_tags.methods.typesafe_jev export --method {GEPA_NAME}",
    ]


# What tune says when a rerun goes on as it is, and when only a higher cap goes on.
RERUN = "Run tune again: it replays every cached answer for free, then goes on."
RERUN_HIGHER = (
    "Run tune again with a higher --max-cost-usd: it replays every cached answer for free, "
    "then goes on."
)


@pytest.mark.parametrize(
    ("error", "rerun"),
    [
        (
            CostCapSpent(
                "Jev's answers of this search cost $3.0002, the cap of $3.0000: run the search "
                "again with a higher cap to go on."
            ),
            RERUN_HIGHER,
        ),
        (
            SearchInterrupted(
                "Jev left some of 40 requests unanswered (a transient error or a run of failed "
                "answers): run the search again to send them."
            ),
            RERUN,
        ),
        (TransientApiError("OpenAI did not answer the reflection of iteration 4."), RERUN),
    ],
    ids=["cost cap spent inside a call", "requests left unanswered", "transient error"],
)
def test_tune_exits_with_1_and_says_how_a_rerun_goes_on_when_the_search_is_interrupted(
    search, live_logs, error, rerun, capsys
):
    search.error = error

    assert run_tune() == 1

    assert [log.closed for log in live_logs] == [1]
    err = capsys.readouterr().err
    assert f"{type(error).__name__}: {error}" in err
    assert err.rstrip("\n").endswith(rerun)


def test_tune_closes_the_live_log_when_the_search_fails(search, live_logs):
    search.error = ValueError("a cap must be above 0 USD")

    with pytest.raises(ValueError, match="a cap must be above 0 USD"):
        run_tune()

    assert [log.closed for log in live_logs] == [1]


def test_tune_without_mlflow_opens_no_log(search, live_logs):
    assert run_tune("--no-mlflow") == 0

    assert live_logs == []
    assert search.calls[0]["live"] is None


def test_tune_refuses_a_search_that_was_exported(search, live_logs):
    params = method_dir(GEPA_NAME) / TUNING_FOLDER / PARAMS_FILE
    params.parent.mkdir(parents=True)
    params.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="the search was exported, so its words are final"):
        run_tune()

    assert (search.calls, live_logs) == ([], [])


def test_export_freezes_the_record_once_then_only_checks_it(search_run, monkeypatch, capsys):
    search_run.write_finished()
    monkeypatch.setattr(search_commands, "search_data", lambda raw: search_run.data)
    # The robot's text is a statement: candidate 2 copies it, and candidate 1 is chosen.
    problems = pd.DataFrame(
        {
            "statement": search_run.texts,
            "prob_desc_notes": [None] * len(search_run.texts),
            "source_code": ["print(1)"] * len(search_run.texts),
        }
    )

    assert search_commands.export(RAW_DEV, problems, search_run.seed_scores) == 0
    assert search_commands.export(RAW_DEV, problems, search_run.seed_scores) == 0

    out = method_dir(GEPA_NAME) / TUNING_FOLDER
    assert json.loads((out / PARAMS_FILE).read_text(encoding="utf-8"))["chosen"] == 1
    scores = (
        "candidate 1 is chosen, with a mean score of 0.9500 on the selection fold, against the "
        f"seed's {search_run.seed_line()['selection_mean']:.4f}."
    )
    assert capsys.readouterr().out.splitlines() == [
        f"{out} now holds the record of the search: {scores}",
        "Commit it, then answer the dev part with `python -m "
        f"codeforces_tags.methods.typesafe_jev run --method {GEPA_NAME} --part dev`.",
        f"{out} holds the same record of the search: {scores}",
    ]


def test_log_search_replays_the_exported_record(search_run, monkeypatch, paths, capsys):
    search_run.write_finished()
    out = method_dir(GEPA_NAME) / TUNING_FOLDER
    export_search(search_run.folder, search_run.data, search_run.seed_scores, search_run.texts, out)
    logged = []

    def log(record, **store):
        logged.append((record, store))
        return "run-7"

    monkeypatch.setattr(search_commands.search_tracking, "log_search", log)
    uri = f"sqlite:///{paths / 'mlflow.db'}"

    artifacts = str(paths / "artifacts")
    status = cli.main(
        ["log-search", "--method", GEPA_NAME, "--tracking-uri", uri, "--artifact-dir", artifacts]
    )

    assert status == 0
    assert len(logged) == 1
    record, store = logged[0]
    assert record.params == read_search_record(out).params
    assert sorted(record.words) == [0, 1, 3]
    assert store == {
        "tracking_uri": uri,
        "artifact_dir": Path(artifacts),
        "run_file": paths / "experiments" / GEPA_NAME / SEARCH_FOLDER / RUN_ID_FILE,
    }
    assert capsys.readouterr().out == (
        f"logged the search's record to MLflow: run run-7 of the experiment {SEARCH_EXPERIMENT}\n"
    )


def test_log_search_names_the_replayed_run_in_run_json_after_a_store_rebuild(
    search_run, store, paths, capsys
):
    search_run.write_finished()
    out = method_dir(GEPA_NAME) / TUNING_FOLDER
    export_search(search_run.folder, search_run.data, search_run.seed_scores, search_run.texts, out)
    # The search ran live into a store that was since rebuilt: run.json names the old store's run.
    run_file = paths / "experiments" / GEPA_NAME / SEARCH_FOLDER / RUN_ID_FILE
    run_file.write_text(json.dumps({"run_id": "a-run-of-the-old-store"}) + "\n", encoding="utf-8")
    options = ["--tracking-uri", store["tracking_uri"]]
    options += ["--artifact-dir", str(store["artifact_dir"])]

    assert cli.main(["log-search", "--method", GEPA_NAME, *options]) == 0

    import mlflow  # pylint: disable=import-outside-toplevel

    run_id = json.loads(run_file.read_text(encoding="utf-8"))["run_id"]
    tracking = mlflow.MlflowClient(store["tracking_uri"])
    run = tracking.get_run(run_id)
    assert run.info.run_name == SEARCH_RUN_NAME
    assert tracking.get_experiment(run.info.experiment_id).name == SEARCH_EXPERIMENT
    assert capsys.readouterr().out == (
        f"logged the search's record to MLflow: run {run_id} of the experiment "
        f"{SEARCH_EXPERIMENT}\n"
    )


def test_importing_the_commands_lets_mlflow_start_neither_telemetry_nor_its_hint(fresh_import):
    # tests/tracking/test_environment.py leaves the Jev commands out of the default job, which
    # lacks their llm-typesafe group. __main__ imports search_commands, and through it tuning
    # (gepa) and search_tracking; the fixture's shell is the one where a bare import of mlflow
    # does both.
    module = "codeforces_tags.methods.typesafe_jev.__main__"

    assert fresh_import(module) == {"telemetry": False, "agent_hint": False}
