"""Tests for the MLflow UI capture (notebooks/figures/mlflow_ui.py), all but its browser.

The capture is one script, and its tests share these stand-ins (a fake server, a fake browser
context, a stand-in playwright), so this one file is longer than pylint's limit on a module, as
``tests/test_cli.py`` is.
"""

# pylint: disable=too-many-lines

import json
import os
import re
import socket
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import mlflow
import numpy as np
import pytest
from mlflow import environment_variables
from PIL import Image

from codeforces_tags.config import (
    FOCUS_TAGS,
    MLFLOW_DATASETS,
    MLFLOW_ENV,
    MLFLOW_SCREEN_GEPA_FOLDER,
    MLFLOW_SCREEN_GEPA_TAG,
    MLFLOW_SCREENS_DIR,
    PROJECT_ROOT,
)

PORT = 5517
NOTEBOOK = PROJECT_ROOT / "notebooks" / "05_zero_shot_llm.ipynb"
NOTEBOOK_10 = PROJECT_ROOT / "notebooks" / "10_jev_gepa.ipynb"
# The search's views of a store whose chosen words changed the criteria of the tag the views
# compare (MLFLOW_SCREEN_GEPA_TAG) and the convention.
GEPA_FILES = [
    "01_gepa_search_chart.png",
    "02_gepa_candidates.png",
    "03_gepa_prompt_compare.png",
    "04_gepa_convention_compare.png",
]
# The prompt of that tag's criteria, and a focus tag whose criteria the gepa store's search kept.
GEPA_PROMPT = "jev-gepa-" + MLFLOW_SCREEN_GEPA_TAG.replace(" ", "-")
KEPT_TAG = next(tag for tag in FOCUS_TAGS if tag != MLFLOW_SCREEN_GEPA_TAG)
# Variables the capture's server never gets: credentials of LLM providers and clouds, proxies.
CREDENTIALS_AND_PROXIES = (
    "OPENAI_API_KEY",
    "TYPESAFE_API_KEY",
    "gemini_api_key",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "DATABASE_PASSWORD",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HTTP_PROXY",
    "https_proxy",
    "ALL_PROXY",
)
# A screen as notebook 05 shows it: the file name in an Image of MLFLOW_SCREENS_DIR.
SCREEN_IMAGE = re.compile(r'Image\(filename=MLFLOW_SCREENS_DIR / "([^"]+)"')
# A screen as notebook 10 shows it: the file name in an Image of MLFLOW_SCREEN_GEPA_FOLDER.
GEPA_SCREEN_IMAGE = re.compile(r'Image\(filename=MLFLOW_SCREEN_GEPA_FOLDER / "([^"]+)"')
# A server that answers GET /health, ignores SIGTERM and starts a child process, as mlflow server
# starts uvicorn: only a SIGKILL to its whole process group stops both.
STUBBORN_SERVER = """
import http.server, signal, subprocess, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # its child leaves no zombie
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""
# Loads the script by path in a fresh interpreter, then prints whether playwright and mlflow are
# imported.
FRESH_LOAD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mlflow_ui", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules["mlflow_ui"] = module
spec.loader.exec_module(module)
print("playwright" in sys.modules, "mlflow" in sys.modules)
"""


def at(clock):
    """Return a time of 2026-10-09 UTC (``"09:40:05"``) in milliseconds since the epoch."""
    return round(datetime.fromisoformat(f"2026-10-09T{clock}+00:00").timestamp() * 1000)


def run_info(run_id, start, end):
    """Return a run's info as dev_window reads it: its id, its start and its end (None: running)."""
    return SimpleNamespace(run_id=run_id, start_time=at(start), end_time=end and at(end))


def free_port():
    """Return a port of 127.0.0.1 that nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def listener():
    """Listen on a port of 127.0.0.1 while the context lasts; yield the port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


def fake_server(url, events):
    """Return a stand-in for mlflow_server that yields ``url`` and records its start and stop."""

    @contextmanager
    def serve(tracking_uri, artifact_dir, port):
        events.append(("start", tracking_uri, artifact_dir, port))
        try:
            yield url
        finally:
            events.append(("stop",))

    return serve


def record_processes(mlflow_ui, monkeypatch):
    """Record each process the capture starts, and the environment it got, in the list returned."""
    started, popen = [], subprocess.Popen

    def recorded_popen(*args, **kwargs):
        process = popen(*args, **kwargs)  # pylint: disable=consider-using-with
        started.append((process, kwargs.get("env")))
        return process

    monkeypatch.setattr(mlflow_ui.subprocess, "Popen", recorded_popen)
    return started


def write_pngs(views, folder):
    """Write a stand-in PNG for each view into the capture's folder."""
    for view in views:
        (folder / view.file).write_bytes(f"png of {view.route}".encode())


def options(screen_store, out):
    """Return the command line options of a capture of the screen store into ``out``."""
    return [
        "--tracking-uri",
        screen_store.tracking_uri,
        "--artifact-dir",
        str(screen_store.artifact_dir),
        "--out",
        str(out),
    ]


def old_pngs(tmp_path):
    """Return a folder of PNGs from an earlier capture, and its contents by file name."""
    out = tmp_path / "out"
    out.mkdir()
    contents = {"01_datasets.png": b"old datasets", "13_prompt_compare.png": b"old compare"}
    for name, content in contents.items():
        (out / name).write_bytes(content)
    return out, contents


def leftovers(out):
    """Return the temporary folders a capture left beside its output folder."""
    return list(out.parent.glob(f".{out.name}-*"))


def contents_of(folder):
    """Return the files of a folder by name."""
    return {path.name: path.read_bytes() for path in folder.iterdir()}


def dataset_of(screen_store, part):
    """Return the evaluation dataset of a part of the screen store."""
    mlflow.set_tracking_uri(screen_store.tracking_uri)
    return mlflow.genai.datasets.search_datasets(
        experiment_ids=[screen_store.experiment_id],
        filter_string=f"name = '{MLFLOW_DATASETS[part]}'",
    )[0]


def traces_of(screen_store, method, part):
    """Return the traces of a method's evaluation run on a part, with their spans."""
    client = mlflow.MlflowClient(screen_store.tracking_uri)
    return client.search_traces(
        locations=[screen_store.experiment_id], run_id=screen_store.runs[method, part]
    )


def test_importing_the_script_imports_mlflow_but_no_playwright(mlflow_ui):
    # CI has no playwright: only the browser half imports it, when it runs.
    result = subprocess.run(
        [sys.executable, "-c", FRESH_LOAD, mlflow_ui.__file__],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["False", "True"]


def test_screen_ids_read_the_ids_of_the_views_from_the_store(mlflow_ui, screen_store):
    dev = dataset_of(screen_store, "dev")
    rows = dev.to_df()
    [record_id] = rows["dataset_record_id"][rows["inputs"].str["sample_id"] == 0]
    traces = traces_of(screen_store, "descode2tag-json", "dev")
    [trace] = [trace for trace in traces if json.loads(trace.data.request)["sample_id"] == 0]

    ids = mlflow_ui.screen_ids(screen_store.tracking_uri)

    assert ids == mlflow_ui.ScreenIds(
        experiment_id=screen_store.experiment_id,
        dataset_id=dev.dataset_id,
        record_id=record_id,
        dev_runs={
            "descode2tag-json": screen_store.runs["descode2tag-json", "dev"],
            "tfidf": screen_store.runs["tfidf", "dev"],
        },
        trace_id=trace.info.trace_id,
        prompt_name="descode2tag",
        prompt_version="2",
        window=("2026-10-09T09:40:00.000Z", "2026-10-09T09:53:00.000Z"),
    )
    assert list(ids.dev_runs) == ["descode2tag-json", "tfidf"]


def test_screens_open_only_dev_runs_and_the_dev_dataset(mlflow_ui, screen_store):
    test_runs = [screen_store.runs[method, "test"] for method in ("tfidf", "descode2tag-json")]
    test_traces = [
        trace.info.trace_id
        for method in ("tfidf", "descode2tag-json")
        for trace in traces_of(screen_store, method, "test")
    ]
    test_dataset = dataset_of(screen_store, "test").dataset_id

    ids = mlflow_ui.screen_ids(screen_store.tracking_uri)
    routes = "\n".join(view.route for view in mlflow_ui.screens(ids))

    assert len(test_traces) == 4
    assert not [name for name in (*test_runs, *test_traces, test_dataset) if name in routes]
    # Every run a route names (after a / or an =) is a dev run.
    assert set(re.findall(r"(?<=[/=])[0-9a-f]{32}", routes)) == {
        screen_store.runs["tfidf", "dev"],
        screen_store.runs["descode2tag-json", "dev"],
    }
    assert dataset_of(screen_store, "dev").dataset_id in routes


def test_the_thirteen_views_their_files_routes_viewports_and_steps(mlflow_ui):
    ids = mlflow_ui.ScreenIds(
        experiment_id="1",
        dataset_id="d-dev",
        record_id="dr-0",
        dev_runs={"code2tag-text": "C", "descode2tag-json": "R", "tfidf": "T"},
        trace_id="tr-0",
        prompt_name="descode2tag",
        prompt_version="2",
        window=("2026-10-09T09:40:00.000Z", "2026-10-09T10:57:00.000Z"),
    )
    window = (
        "startTimeLabel=CUSTOM&startTime=2026-10-09T09:40:00.000Z&endTime=2026-10-09T10:57:00.000Z"
    )
    false_number_theory = "ASSESSMENT%3A%3A%3D%3A%3Afalse%3A%3Acorrect_number_theory"

    views = mlflow_ui.screens(ids, tag="number theory")

    assert [(view.file, view.route, view.viewport, view.steps) for view in views] == [
        ("01_datasets.png", "experiments/1/datasets", (1600, 1000), ()),
        ("02_dataset_records.png", "experiments/1/datasets/d-dev", (1600, 1000), ()),
        ("03_record_drawer.png", "experiments/1/datasets/d-dev?recordId=dr-0", (1600, 1000), ()),
        (
            "04_evaluation_runs.png",
            "experiments/1/evaluation-runs?selectedRunUuid=R",
            (1600, 1000),
            ("filter_dev_runs",),
        ),
        (
            "05_evaluation_runs_charts.png",
            "experiments/1/evaluation-runs?viewMode=charts&selectedRunUuid=R",
            (1600, 1000),
            ("filter_dev_runs",),
        ),
        ("06_run_traces.png", "experiments/1/runs/R/traces", (1600, 1000), ()),
        (
            "07_trace_drawer.png",
            "experiments/1/runs/R/traces?selectedEvaluationId=tr-0",
            (1600, 1800),
            ("open_assessments",),
        ),
        (
            "08_filter_false.png",
            f"experiments/1/runs/R/traces?filter={false_number_theory}",
            (1600, 1000),
            ("load_every_trace",),
        ),
        ("09_compare.png", "experiments/1/runs/T/traces?compareToRunUuid=R", (1600, 1000), ()),
        ("10_overview_usage.png", f"experiments/1/overview/usage?{window}", (1600, 1800), ()),
        (
            "11_traces_cost.png",
            f"experiments/1/traces?{window}",
            (2000, 1000),
            ("show_cost_column",),
        ),
        ("12_prompt_versions.png", "prompts/descode2tag?promptVersion=2", (1600, 1000), ()),
        (
            "13_prompt_compare.png",
            "prompts/descode2tag",
            (1600, 1000),
            ("compare_prompt_versions",),
        ),
    ]
    # The last step is the search's views' only one.
    assert {step for view in views for step in view.steps} == set(mlflow_ui.STEPS[:-1])


@pytest.mark.parametrize(
    ("last_end", "window_end"),
    [
        ("10:56:00", "10:56:00.000Z"),
        ("10:56:00.001", "10:57:00.000Z"),
        ("10:56:59", "10:57:00.000Z"),
    ],
)
def test_dev_window_spans_the_dev_runs_in_whole_minutes(mlflow_ui, last_end, window_end):
    dev = [run_info("a", "09:52:00", last_end), run_info("b", "09:40:05", "09:51:59.999")]
    # Test runs before the window, or after it, are left out of it.
    test = [run_info("c", "09:30:00", "09:39:59.999"), run_info("d", "10:58:00", "11:01:00")]

    assert mlflow_ui.dev_window(dev, test) == (
        "2026-10-09T09:40:00.000Z",
        f"2026-10-09T{window_end}",
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("09:45:00", "09:46:00"),
        # It starts after the last dev run ends, within the minute the window rounds up to.
        ("10:56:30", "10:59:00"),
        # It ends before the first dev run starts, within the minute the window rounds down to.
        ("09:38:00", "09:40:00.500"),
        ("09:30:00", None),
    ],
)
def test_dev_window_refuses_a_test_run_that_meets_it(mlflow_ui, start, end):
    dev = [run_info("a", "09:40:05", "10:56:10")]

    with pytest.raises(ValueError, match=r"\['t'\] meet the dev window.*a 2-minute pause, then"):
        mlflow_ui.dev_window(dev, [run_info("t", start, end)])


def test_dev_window_refuses_a_dev_run_with_no_end(mlflow_ui):
    dev = [run_info("a", "09:40:05", "10:56:10"), run_info("b", "10:56:20", None)]

    with pytest.raises(ValueError, match=r"\['b'\] have no end time"):
        mlflow_ui.dev_window(dev, [])


def add_incomplete_test_run(client, screen_store):
    """Start a test evaluation run of tfidf, as a stopped tracking evaluate leaves it."""
    tags = {"run_kind": "evaluation", "method": "tfidf", "part": "test"}
    client.create_run(screen_store.experiment_id, tags=tags)


def add_incomplete_dev_run(client, screen_store):
    """Start a dev evaluation run of label_prior, as a stopped tracking evaluate leaves it."""
    tags = {"run_kind": "evaluation", "method": "label_prior", "part": "dev"}
    client.create_run(screen_store.experiment_id, tags=tags)


def add_second_dev_run(client, screen_store):
    """Add a second complete dev evaluation run of descode2tag-json, of another recipe."""
    tags = {"run_kind": "evaluation", "method": "descode2tag-json", "part": "dev"}
    run = client.create_run(screen_store.experiment_id, tags=tags)
    client.set_terminated(run.info.run_id)
    client.set_tag(run.info.run_id, "evaluation_complete", "true")


def delete_tfidf_dev_run(client, screen_store):
    """Delete the dev evaluation run of tfidf."""
    client.delete_run(screen_store.runs["tfidf", "dev"])


def delete_compared_dev_run(client, screen_store):
    """Delete the dev evaluation run of descode2tag-json."""
    client.delete_run(screen_store.runs["descode2tag-json", "dev"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (add_incomplete_test_run, r"incomplete: discard each \(python -m codeforces_tags.tracking"),
        (add_incomplete_dev_run, r"incomplete: discard each \(python -m codeforces_tags.tracking"),
        (add_second_dev_run, r"\['descode2tag-json'\] have more than one dev evaluation run"),
        (delete_tfidf_dev_run, r"No dev evaluation run of \['tfidf'\]: run .* evaluate --part dev"),
        (delete_compared_dev_run, r"No dev evaluation run of \['descode2tag-json'\]"),
    ],
)
def test_screen_ids_refuse_a_store_without_exactly_one_complete_dev_run_per_method(
    mlflow_ui, screen_store, change, message
):
    # An incomplete run would show in the Evaluation runs table, and its traces in the dev
    # window; with two dev runs of a method, or none, the views would show an arbitrary one.
    change(mlflow.MlflowClient(screen_store.tracking_uri), screen_store)

    with pytest.raises(ValueError, match=message):
        mlflow_ui.screen_ids(screen_store.tracking_uri)


def test_screen_ids_refuse_a_store_whose_complete_test_run_meets_the_dev_window(
    mlflow_ui, screen_store
):
    # Screenshots 10 and 11 show every trace of the dev window, which ends at 09:53: a test run
    # from 09:52:30 would show its traces (ADR 0001).
    client = mlflow.MlflowClient(screen_store.tracking_uri)
    tags = {"run_kind": "evaluation", "method": "tfidf", "part": "test"}
    run_id = client.create_run(
        screen_store.experiment_id, start_time=at("09:52:30"), tags=tags
    ).info.run_id
    client.set_terminated(run_id, end_time=at("09:54:00"))
    client.set_tag(run_id, "evaluation_complete", "true")

    with pytest.raises(ValueError, match=rf"\['{run_id}'\] meet the dev window"):
        mlflow_ui.screen_ids(screen_store.tracking_uri)


def delete_dev_dataset(screen_store):
    """Delete the codeforces-dev evaluation dataset."""
    dataset_id = dataset_of(screen_store, "dev").dataset_id
    mlflow.genai.datasets.delete_dataset(dataset_id=dataset_id)


def delete_sample_trace(screen_store):
    """Delete the trace of problem 0 from the dev evaluation run of descode2tag-json."""
    traces = traces_of(screen_store, "descode2tag-json", "dev")
    [trace] = [trace for trace in traces if json.loads(trace.data.request)["sample_id"] == 0]
    client = mlflow.MlflowClient(screen_store.tracking_uri)
    client.delete_traces(screen_store.experiment_id, trace_ids=[trace.info.trace_id])


def keep_store(screen_store):  # pylint: disable=unused-argument
    """Leave the store as it is."""


@pytest.mark.parametrize(
    ("change", "settings", "message"),
    [
        (
            delete_dev_dataset,
            {},
            r"0 evaluation datasets named codeforces-dev, not one: run python -m .* datasets",
        ),
        (keep_store, {"sample_id": 3}, r"0 records of the problem 3, not one: .* a dev problem"),
        (delete_sample_trace, {}, r"0 traces of the problem 0, not one: rebuild the store"),
        (keep_store, {"compare": "tfidf"}, r"of tfidf pins the prompts \[\], not one"),
    ],
)
def test_screen_ids_refuse_a_store_that_lacks_what_a_view_opens(
    mlflow_ui, screen_store, change, settings, message
):
    # A test problem has no record in codeforces-dev; tfidf's run record pins no prompt.
    change(screen_store)

    with pytest.raises(ValueError, match=message):
        mlflow_ui.screen_ids(screen_store.tracking_uri, **settings)


def test_screen_ids_refuse_a_store_with_no_methods_experiment(mlflow_ui, store):
    with pytest.raises(ValueError, match="holds no tag-methods experiment: rebuild the store"):
        mlflow_ui.screen_ids(store["tracking_uri"])


def guard_of(mlflow_ui):
    """Return what a guarded context of a stand-in browser does with a URL.

    The function returned tells whether its request route and its WebSocket route take the URL,
    and whether its request listener records the URL as one that left the server.
    """
    context, outside = mlflow_ui.guarded_context(SimpleNamespace(new_context=FakeContext), PORT)
    [(leaves, _)] = context.routes
    [(socket_leaves, _)] = context.socket_routes

    def decide(url):
        context.emit("request", SimpleNamespace(url=url))
        return leaves(url), socket_leaves(url), url in outside

    return decide


@pytest.mark.parametrize(
    "url",
    [
        f"http://127.0.0.1:{PORT}/",
        f"http://127.0.0.1:{PORT}/ajax-api/2.0/mlflow/runs/search",
        f"https://127.0.0.1:{PORT}/static-files/main.js",
        f"ws://127.0.0.1:{PORT}/socket",
        f"wss://127.0.0.1:{PORT}/socket",
        "data:image/png;base64,iVBORw0KGgo=",
        f"blob:http://127.0.0.1:{PORT}/6f1c0b2e",
        "about:blank",
    ],
)
def test_requests_to_the_capture_server_and_inline_data_are_allowed(mlflow_ui, url):
    assert mlflow_ui.is_allowed(url, PORT)
    # Neither route of the guarded context takes the URL, and its request listener records none.
    assert guard_of(mlflow_ui)(url) == (False, False, False)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:5000/",
        "http://127.0.0.1/",
        f"http://localhost:{PORT}/",
        f"http://[::1]:{PORT}/",
        f"http://127.0.0.2:{PORT}/",
        f"http://127.0.0.1.example.com:{PORT}/",
        f"http://127.0.0.1:{PORT}@example.com/",
        "https://fonts.example.com/inter.woff2",
        f"ws://example.com:{PORT}/socket",
        "wss://example.com/socket",
        f"ftp://127.0.0.1:{PORT}/",
        "file:///etc/hosts",
        "http://127.0.0.1:99999/",
        f"http://127.0.0.1:{PORT}x/",
        "chrome-extension://abcdef/script.js",
    ],
)
def test_only_requests_to_the_capture_server_are_allowed(mlflow_ui, url):
    # The host and port are compared exactly once parsed, never as a prefix of the URL: the
    # userinfo form's host is example.com.
    assert not mlflow_ui.is_allowed(url, PORT)
    # Both routes of the guarded context take the URL, and its request listener records it.
    assert guard_of(mlflow_ui)(url) == (True, True, True)


def test_server_command_serves_the_store_on_127_0_0_1_with_one_worker(
    mlflow_ui, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    command = mlflow_ui.server_command("sqlite:///s/mlflow.db", "s/mlartifacts", 5517)

    assert command == [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///s/mlflow.db",
        "--default-artifact-root",
        (tmp_path / "s" / "mlartifacts").as_uri(),
        "--host",
        "127.0.0.1",
        "--port",
        "5517",
        "--workers",
        "1",
    ]


def test_server_env_is_the_environment_under_mlflow_env(mlflow_ui, monkeypatch):
    monkeypatch.setenv("CODEFORCES_DATA_DIR", "/data")

    env = mlflow_ui.server_env()

    left_out = [name for name in os.environ if mlflow_ui.LEFT_OUT.search(name)]
    assert env == {name: value for name, value in os.environ.items() if name not in left_out}
    assert {name: env[name] for name in MLFLOW_ENV} == MLFLOW_ENV
    assert env["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert env["CODEFORCES_DATA_DIR"] == "/data"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true"),
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "1"),
        ("MLFLOW_DISABLE_TELEMETRY", "false"),
        ("MLFLOW_DISABLE_TELEMETRY", "0"),
        ("MLFLOW_DISABLE_TELEMETRY", "FALSE"),
        # Values MLflow cannot read as a boolean.
        ("MLFLOW_DISABLE_TELEMETRY", "yes"),
        ("MLFLOW_DISABLE_TELEMETRY", ""),
        ("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true"),
        ("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "True"),
    ],
)
def test_server_env_is_mlflow_envs_and_refuses_pickle_telemetry_or_the_job_runner_on(
    mlflow_ui, monkeypatch, name, value
):
    # mlflow_env() sets pickle and the job runner off whatever the environment says, and turns
    # telemetry off only when it is unset: the capture checks them, and never sets one itself.
    monkeypatch.delitem(MLFLOW_ENV, name, raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=re.escape(f"holds {{'{name}': '{value}'}}")):
        mlflow_ui.server_env()


@pytest.mark.parametrize(
    ("name", "value", "read"),
    [
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "0", False),
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "False", False),
        ("MLFLOW_DISABLE_TELEMETRY", "1", True),
        ("MLFLOW_DISABLE_TELEMETRY", "TRUE", True),
        ("MLFLOW_DISABLE_TELEMETRY", "True", True),
        ("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "0", False),
        ("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "FALSE", False),
    ],
)
def test_server_env_accepts_every_spelling_that_mlflow_reads_as_off(
    mlflow_ui, monkeypatch, name, value, read
):
    # MLflow reads a boolean variable as true or 1, false or 0, in any case.
    monkeypatch.delitem(MLFLOW_ENV, name, raising=False)
    monkeypatch.setenv(name, value)

    env = mlflow_ui.server_env()

    assert env[name] == value
    assert getattr(environment_variables, name).get() is read


def test_check_safeguards_refuses_an_environment_that_leaves_them_unset(mlflow_ui):
    # MLflow's defaults load pickles, send telemetry and run jobs.
    unset = dict.fromkeys(mlflow_ui.SERVER_SAFEGUARDS)

    with pytest.raises(ValueError, match=re.escape(f"holds {unset}")):
        mlflow_ui.check_safeguards({})


@pytest.mark.parametrize("name", CREDENTIALS_AND_PROXIES)
def test_server_env_holds_no_credential_and_no_proxy(mlflow_ui, monkeypatch, name):
    # With a credential, the server's LLM features could spend money; through a proxy, its
    # requests to 127.0.0.1 would leave the machine. This process keeps both.
    monkeypatch.setenv(name, "placeholder")

    env = mlflow_ui.server_env()

    assert name not in env
    assert os.environ[name] == "placeholder"


def test_port_in_use_tells_whether_something_listens_on_the_port(mlflow_ui):
    with listener() as port:
        assert mlflow_ui.port_in_use(port)

    assert not mlflow_ui.port_in_use(port)


def test_main_refuses_a_port_in_use_and_starts_no_server(
    mlflow_ui, screen_store, monkeypatch, tmp_path
):
    # Another server on the port, e.g. a user's own mlflow ui, would answer instead of ours.
    started = []
    monkeypatch.setattr(mlflow_ui.subprocess, "Popen", lambda *args, **kwargs: started.append(args))

    with listener() as port, pytest.raises(RuntimeError, match=f"127.0.0.1:{port} is in use"):
        mlflow_ui.main([*options(screen_store, tmp_path / "out"), "--port", str(port)])

    assert not started
    assert not (tmp_path / "out").exists()


def test_main_sets_mlflows_environment_then_refuses_a_missing_store_and_creates_none(
    mlflow_ui, tmp_path
):
    # The README's first step moves the store aside: MLflow would create an empty one there.
    events, database = [], tmp_path / "mlflow.db"
    store = SimpleNamespace(tracking_uri=f"sqlite:///{database}", artifact_dir=tmp_path / "art")

    with pytest.raises(ValueError, match=f"{re.escape(str(database))} does not exist: rebuild"):
        mlflow_ui.main(options(store, tmp_path / "out"), server=fake_server("", events))

    assert os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] == "false"
    assert not database.exists()
    assert not events


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true"),
        ("MLFLOW_DISABLE_TELEMETRY", "false"),
        ("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true"),
    ],
)
def test_main_refuses_pickle_telemetry_or_the_job_runner_on_before_reading_the_store(
    mlflow_ui, monkeypatch, tmp_path, name, value
):
    # With telemetry on, the script's own first read of the store (a dataset's to_df) would start
    # MLflow's telemetry client, which fetches its config from outside the machine.
    reads, events = [], []
    monkeypatch.setattr(mlflow_ui, "screen_ids", lambda *args, **kwargs: reads.append(args))
    monkeypatch.delitem(MLFLOW_ENV, name, raising=False)
    monkeypatch.setenv(name, value)
    database = tmp_path / "mlflow.db"
    store = SimpleNamespace(tracking_uri=f"sqlite:///{database}", artifact_dir=tmp_path)

    with pytest.raises(ValueError, match=re.escape(f"holds {{'{name}': '{value}'}}")):
        mlflow_ui.main(options(store, tmp_path / "out"), server=fake_server("", events))

    assert not reads
    assert not events


def test_main_exits_1_when_the_server_serves_another_store(
    mlflow_ui, screen_store, monkeypatch, tmp_path, capsys
):
    # The server answers other run ids than the store of --tracking-uri: another server bound
    # the port between the check and our start, and ours died.
    read_ids, events, shots = mlflow_ui.screen_ids, [], []

    def other_ids_from_the_server(tracking_uri, **settings):
        ids = read_ids(screen_store.tracking_uri, **settings)
        if tracking_uri == screen_store.tracking_uri:
            return ids
        return replace(ids, dev_runs={**ids.dev_runs, "tfidf": "0" * 32})

    monkeypatch.setattr(mlflow_ui, "screen_ids", other_ids_from_the_server)
    out, before = old_pngs(tmp_path)
    server = fake_server("http://127.0.0.1:5517", events)

    status = mlflow_ui.main(
        options(screen_store, out), server=server, shoot=lambda *args: shots.append(args)
    )

    assert status == 1
    assert "serves another store" in capsys.readouterr().err
    assert not shots
    assert events[-1] == ("stop",)
    assert contents_of(out) == before


def test_main_exits_2_writes_no_png_and_stops_the_server_when_a_request_left_it(
    mlflow_ui, screen_store, tmp_path, capsys
):
    events = []

    def leaking_capture(views, base_url, folder):  # pylint: disable=unused-argument
        write_pngs(views, folder)
        return ["https://fonts.example.com/inter.woff2"]

    out, before = old_pngs(tmp_path)
    server = fake_server(screen_store.tracking_uri, events)

    status = mlflow_ui.main(options(screen_store, out), server=server, shoot=leaking_capture)

    assert status == mlflow_ui.OUTSIDE_REQUEST_EXIT == 2
    assert "https://fonts.example.com/inter.woff2" in capsys.readouterr().err
    assert events[-1] == ("stop",)
    assert contents_of(out) == before
    assert not leftovers(out)


def capture_with_one_view_missing(views, base_url, folder):  # pylint: disable=unused-argument
    """Write every view's PNG but the last one, and report no outside request."""
    write_pngs(views[:-1], folder)
    return []


def capture_with_an_extra_file(views, base_url, folder):  # pylint: disable=unused-argument
    """Write every view's PNG and one more, and report no outside request."""
    write_pngs(views, folder)
    (folder / "14_stale.png").write_bytes(b"png")
    return []


@pytest.mark.parametrize("shoot", [capture_with_one_view_missing, capture_with_an_extra_file])
def test_main_exits_1_and_writes_no_png_when_the_capture_misses_a_file(
    mlflow_ui, screen_store, tmp_path, capsys, shoot
):
    events = []
    out, before = old_pngs(tmp_path)
    server = fake_server(screen_store.tracking_uri, events)

    status = mlflow_ui.main(options(screen_store, out), server=server, shoot=shoot)

    assert status == 1
    assert "no PNG was replaced" in capsys.readouterr().err
    assert events[-1] == ("stop",)
    assert contents_of(out) == before
    assert not leftovers(out)


def test_main_exits_1_and_writes_no_png_when_the_capture_fails(mlflow_ui, screen_store, tmp_path):
    # A missing control raises halfway through the views: Python exits with 1.
    events = []

    def failing_capture(views, base_url, folder):  # pylint: disable=unused-argument
        write_pngs(views[:5], folder)
        raise RuntimeError("no control mlflow.traces-v4.display.trigger")

    out, before = old_pngs(tmp_path)
    server = fake_server(screen_store.tracking_uri, events)

    with pytest.raises(RuntimeError, match="no control"):
        mlflow_ui.main(options(screen_store, out), server=server, shoot=failing_capture)

    assert events[-1] == ("stop",)
    assert contents_of(out) == before
    assert not leftovers(out)


def test_main_replaces_the_pngs_once_every_view_was_captured(
    mlflow_ui, screen_store, tmp_path, capsys
):
    events, shot = [], []

    def complete_capture(views, base_url, folder):
        shot.append((views, base_url))
        write_pngs(views, folder)
        return []

    out, _ = old_pngs(tmp_path)
    server = fake_server(screen_store.tracking_uri, events)
    command_line = [*options(screen_store, out), "--port", "6123"]

    status = mlflow_ui.main(command_line, server=server, shoot=complete_capture)

    views = mlflow_ui.screens(mlflow_ui.screen_ids(screen_store.tracking_uri))
    assert status == 0
    assert shot == [(views, screen_store.tracking_uri)]
    assert contents_of(out) == {view.file: f"png of {view.route}".encode() for view in views}
    start = ("start", screen_store.tracking_uri, screen_store.artifact_dir, 6123)
    assert events == [start, ("stop",)]
    assert not leftovers(out)
    assert capsys.readouterr().out == f"Wrote 13 PNGs to {out}\n"


def test_mlflow_server_serves_the_store_on_127_0_0_1_and_stops(
    mlflow_ui, screen_store, monkeypatch
):
    started = record_processes(mlflow_ui, monkeypatch)
    port = free_port()
    for name in CREDENTIALS_AND_PROXIES:
        monkeypatch.setenv(name, "placeholder")
    # MLflow's REST reads would go through this proxy, which listens nowhere, and fail at once.
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{free_port()}")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

    with mlflow_ui.mlflow_server(screen_store.tracking_uri, screen_store.artifact_dir, port) as url:
        assert url == f"http://127.0.0.1:{port}"
        served = mlflow_ui.screen_ids(url)

    assert "HTTP_PROXY" not in os.environ
    assert served == mlflow_ui.screen_ids(screen_store.tracking_uri)
    # The server got no credential and no proxy, and MLflow's safeguards on.
    assert len(started) == 1
    process, env = started[0]
    assert sorted(set(CREDENTIALS_AND_PROXIES) & set(env)) == []
    assert {name: env[name] for name in mlflow_ui.SERVER_SAFEGUARDS} == {
        "MLFLOW_ALLOW_PICKLE_DESERIALIZATION": "false",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
    }
    # uvicorn is a child process of mlflow server: the whole group must be gone.
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    assert not mlflow_ui.port_in_use(port)


def test_mlflow_server_kills_a_server_group_that_ignores_sigterm(mlflow_ui, monkeypatch, tmp_path):
    def stubborn_command(tracking_uri, artifact_dir, port):  # pylint: disable=unused-argument
        return [sys.executable, "-c", STUBBORN_SERVER, str(port)]

    started = record_processes(mlflow_ui, monkeypatch)
    monkeypatch.setattr(mlflow_ui, "server_command", stubborn_command)
    monkeypatch.setattr(mlflow_ui, "SERVER_STOP_S", 1)
    port = free_port()

    with mlflow_ui.mlflow_server("sqlite:///unused.db", tmp_path, port) as url:
        assert mlflow_ui.port_in_use(port)

    assert url == f"http://127.0.0.1:{port}"
    assert len(started) == 1
    process, _ = started[0]
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    assert not mlflow_ui.port_in_use(port)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("print('no such store'); raise SystemExit(3)", r"exited with 3:\nno such store"),
        ("import time; time.sleep(30)", r"did not answer http://127.0.0.1:\d+/health in 1 s"),
    ],
)
def test_mlflow_server_refuses_a_server_that_exits_or_never_answers(
    mlflow_ui, monkeypatch, tmp_path, script, message
):
    def command(tracking_uri, artifact_dir, port):  # pylint: disable=unused-argument
        return [sys.executable, "-c", script]

    monkeypatch.setattr(mlflow_ui, "server_command", command)
    monkeypatch.setattr(mlflow_ui, "SERVER_START_S", 1)

    with ExitStack() as stack, pytest.raises(RuntimeError, match=message):
        stack.enter_context(mlflow_ui.mlflow_server("sqlite:///unused.db", tmp_path, free_port()))


def test_never_click_holds_the_llm_features_and_none_of_clicks(mlflow_ui):
    assert set(mlflow_ui.CLICKS).isdisjoint(mlflow_ui.NEVER_CLICK)
    assert len(set(mlflow_ui.NEVER_CLICK)) == len(mlflow_ui.NEVER_CLICK) == 15
    assert {"mlflow.assistant.fab", "mlflow.traces-v4.actions.run-judges"} <= set(
        mlflow_ui.NEVER_CLICK
    )


def test_the_steps_reach_a_control_only_through_clicks(mlflow_ui):
    # _control is the steps' only way to a control by its id: it refuses every id of NEVER_CLICK.
    page = SimpleNamespace(locator=lambda selector: ("locator", selector))
    control = mlflow_ui._control  # pylint: disable=protected-access

    for component_id in mlflow_ui.CLICKS:
        located = control(page, component_id)
        assert located == ("locator", f'[data-component-id="{component_id}"]')
    for component_id in mlflow_ui.NEVER_CLICK:
        with pytest.raises(ValueError, match=f"{re.escape(component_id)} is not in CLICKS"):
            control(page, component_id)


def test_the_init_script_removes_shared_workers_sets_the_presets_and_keeps_the_browser_online(
    mlflow_ui,
):
    # A shared worker's requests are neither routed nor reported to the context, and the UI
    # starts one for its own telemetry unless its setting is off. Dismissed, shown, opened once,
    # closed: localStorage holds each value as JSON text.
    assert mlflow_ui.INIT_SCRIPT.startswith("delete window.SharedWorker;\n")
    assert {key: json.loads(value) for key, value in mlflow_ui.UI_PRESETS.items()} == {
        "mlflow.settings.telemetry.enabled_v1": False,
        "mlflow.telemetry.info.alert.dismissed_v1": True,
        "mlflow.detectIssues.guidanceShown_v1": True,
        "mlflow.assistant.fab.autoOpenedOnFirstLoad_v1": True,
        "mlflow.assistant.panelOpen_v1": False,
    }
    assert json.dumps(mlflow_ui.UI_PRESETS) in mlflow_ui.INIT_SCRIPT
    assert 'defineProperty(Navigator.prototype, "onLine", {get: () => true' in mlflow_ui.INIT_SCRIPT
    assert mlflow_ui.HIDDEN.startswith('button[data-assistant-ui="true"]{display:none')


class FakeContext:
    """A browser context that records how the capture wires it."""

    def __init__(self, **settings):
        """Keep the settings the context was opened with; nothing is wired yet."""
        self.options, self.scripts, self.listeners = settings, [], {}
        self.routes, self.socket_routes = [], []

    def add_init_script(self, script):
        """Record a script run in every page before its own."""
        self.scripts.append(script)

    def route(self, url, handler):
        """Record a route of the requests whose URL matches."""
        self.routes.append((url, handler))

    def route_web_socket(self, url, handler):
        """Record a route of the WebSockets whose URL matches."""
        self.socket_routes.append((url, handler))

    def on(self, event, handler):
        """Record a handler of an event."""
        self.listeners.setdefault(event, []).append(handler)

    def emit(self, event, value):
        """Call every handler of an event with the value, in the order they were added."""
        for handler in self.listeners.get(event, []):
            handler(value)


class FakeRoute:  # pylint: disable=too-few-public-methods
    """A routed request or WebSocket, which records whether it was aborted.

    It has no ``connect_to_server``: a guard that connected a WebSocket would raise
    ``AttributeError``.
    """

    def __init__(self, url):
        """Route ``url``, not aborted yet."""
        self.url, self.request, self.calls = url, SimpleNamespace(url=url), []

    def abort(self):
        """Abort the request."""
        self.calls.append("abort")


class StandInBrowser:
    """A browser that opens fake contexts and records whether it was closed."""

    def __init__(self):
        """Open no context yet."""
        self.contexts, self.closed = [], False

    def new_context(self, **settings):
        """Open a fake context with the settings."""
        self.contexts.append(FakeContext(**settings))
        return self.contexts[-1]

    def close(self):
        """Close the browser."""
        self.closed = True


def stand_in_playwright(monkeypatch, browser):
    """Make ``browser`` the Chrome that ``capture`` launches, through a stand-in playwright."""

    @contextmanager
    def sync_playwright():
        yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda **options: browser))

    sync_api = SimpleNamespace(Error=RuntimeError, sync_playwright=sync_playwright)
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace(sync_api=sync_api))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def random_pngs(folder, count):
    """Write PNGs of random colours, which a 256-colour palette shrinks; return their paths."""
    rng = np.random.default_rng(0)
    paths = [folder / f"{index:02d}.png" for index in range(count)]
    for path in paths:
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    return paths


def test_guarded_context_is_light_us_english_utc_and_blocks_service_workers(mlflow_ui):
    # Dates and counts in the PNGs follow the context, not the machine's locale and timezone;
    # a service worker's requests would escape the routes.
    browser = SimpleNamespace(new_context=FakeContext)

    context, outside = mlflow_ui.guarded_context(browser, PORT)

    assert context.options == {
        "color_scheme": "light",
        "device_scale_factor": 1,
        "locale": "en-US",
        "timezone_id": "UTC",
        "service_workers": "block",
    }
    assert context.scripts == [mlflow_ui.INIT_SCRIPT]
    assert outside == []


def test_guarded_context_aborts_and_records_every_request_that_leaves_the_server(mlflow_ui):
    context, outside = mlflow_ui.guarded_context(SimpleNamespace(new_context=FakeContext), PORT)
    [(leaves, abort)] = context.routes
    [(socket_leaves, hold)] = context.socket_routes
    image, socket_url = "https://cdn.example.com/a.png", "wss://example.com/live"
    unrouted = f"http://localhost:{PORT}/from-a-worker"
    request, socket_route = FakeRoute(image), FakeRoute(socket_url)

    abort(request)
    hold(socket_route)
    for url in (image, unrouted, f"http://127.0.0.1:{PORT}/ajax-api/2.0/mlflow/runs/search"):
        context.emit("request", SimpleNamespace(url=url))

    assert [leaves(url) for url in (image, f"http://127.0.0.1:{PORT}/")] == [True, False]
    assert [socket_leaves(url) for url in (socket_url, f"ws://127.0.0.1:{PORT}/s")] == [True, False]
    # A routed WebSocket that never connects to its server stays in the browser.
    assert (request.calls, socket_route.calls) == (["abort"], [])
    # Each outside URL once: the aborted request is also seen as a request.
    assert outside == [image, socket_url, unrouted]


def test_capture_reports_a_request_that_a_page_sends_as_the_browser_closes(
    mlflow_ui, monkeypatch, tmp_path
):
    # A closing page can send a request (a beacon) after the last view: the guard aborts and
    # records it, and the capture must report it, not its PNGs.
    contexts, beacon = [], "https://telemetry.example.com/beacon"

    def new_context(**settings):
        contexts.append(FakeContext(**settings))
        return contexts[-1]

    def close():
        [(_, abort)] = contexts[0].routes
        abort(FakeRoute(beacon))

    stand_in_playwright(monkeypatch, SimpleNamespace(new_context=new_context, close=close))
    monkeypatch.setattr(mlflow_ui, "_shoot", lambda context, view, url, png: png.write_bytes(b""))
    views = [mlflow_ui.Screen("01_datasets.png", "experiments/1/datasets")]

    assert mlflow_ui.capture(views, f"http://127.0.0.1:{PORT}", tmp_path) == [beacon]


def test_capture_fits_the_pngs_and_counts_the_requests_of_a_clean_run(
    mlflow_ui, monkeypatch, tmp_path, capsys
):
    browser, folder = StandInBrowser(), tmp_path / "shots"
    views = [
        mlflow_ui.Screen("01_datasets.png", "experiments/1/datasets"),
        mlflow_ui.Screen("02_dataset_records.png", "experiments/1/datasets/d-dev"),
    ]
    shots = dict(zip(views, random_pngs(tmp_path, len(views)), strict=True))
    shot = sum(path.stat().st_size for path in shots.values())

    def shoot(context, view, url, png):
        # Two requests to the server and one inline, for each view.
        for request in (url, f"http://127.0.0.1:{PORT}/ajax-api/", "data:image/png;,"):
            context.emit("request", SimpleNamespace(url=request))
        png.write_bytes(shots[view].read_bytes())

    stand_in_playwright(monkeypatch, browser)
    monkeypatch.setattr(mlflow_ui, "_shoot", shoot)
    monkeypatch.setattr(mlflow_ui, "SCREENS_MAX_BYTES", shot - 1)
    folder.mkdir()

    assert mlflow_ui.capture(views, f"http://127.0.0.1:{PORT}", folder) == []

    fitted = sum(path.stat().st_size for path in folder.iterdir())
    assert capsys.readouterr().out == (
        f"4 requests, all to 127.0.0.1:{PORT}, and 2 inline\n"
        f"The PNGs weigh {shot:,} bytes, {fitted:,} once fitted to {shot - 1:,}\n"
    )
    for view in views:
        with Image.open(folder / view.file) as image:
            assert image.mode == "P"
    assert browser.closed
    # The guard lets through the server's port, which capture read from its URL, and no other.
    [(leaves, _)] = browser.contexts[0].routes
    assert [leaves(f"http://127.0.0.1:{PORT}/"), leaves("http://127.0.0.1:1/")] == [False, True]


def test_capture_reports_a_request_it_aborted_rather_than_the_failure_that_followed(
    mlflow_ui, monkeypatch, tmp_path, capsys
):
    # A page that needs an outside script or font never loads once the guard aborts it, and its
    # view times out: the capture reports the request, which explains the timeout.
    browser, needed = StandInBrowser(), "https://cdn.example.com/needed.js"

    def shoot(context, view, url, png):  # pylint: disable=unused-argument
        [(_, abort)] = context.routes
        abort(FakeRoute(needed))
        raise TimeoutError("Timeout 60000ms exceeded.")

    stand_in_playwright(monkeypatch, browser)
    monkeypatch.setattr(mlflow_ui, "_shoot", shoot)
    views = [mlflow_ui.Screen("01_datasets.png", "experiments/1/datasets")]

    assert mlflow_ui.capture(views, f"http://127.0.0.1:{PORT}", tmp_path) == [needed]
    assert "Timeout 60000ms exceeded." in capsys.readouterr().err
    assert browser.closed
    assert not list(tmp_path.iterdir())


def test_capture_raises_the_failure_of_a_view_when_no_request_left_the_server(
    mlflow_ui, monkeypatch, tmp_path
):
    browser = StandInBrowser()

    def shoot(context, view, url, png):  # pylint: disable=unused-argument
        raise TimeoutError("Timeout 60000ms exceeded.")

    stand_in_playwright(monkeypatch, browser)
    monkeypatch.setattr(mlflow_ui, "_shoot", shoot)
    views = [mlflow_ui.Screen("01_datasets.png", "experiments/1/datasets")]

    with pytest.raises(TimeoutError, match="Timeout 60000ms exceeded."):
        mlflow_ui.capture(views, f"http://127.0.0.1:{PORT}", tmp_path)

    assert browser.closed


def test_check_dev_runs_accepts_the_dev_runs_of_the_unfiltered_table(mlflow_ui):
    before = ["eval-b-test", "eval-a-test", "eval-b-dev", "eval-a-dev"]

    mlflow_ui.check_dev_runs(before, ["eval-b-dev", "eval-a-dev"])


@pytest.mark.parametrize(
    "after",
    [
        ["eval-b-test", "eval-a-test", "eval-b-dev", "eval-a-dev"],
        ["eval-b-dev"],
        ["eval-b-dev", "eval-a-dev", "eval-c-dev"],
        [],
    ],
)
def test_check_dev_runs_refuses_a_table_that_is_not_every_dev_run(mlflow_ui, after):
    # The filter has not applied, or the table lists another set of runs than the dev ones.
    before = ["eval-b-test", "eval-a-test", "eval-b-dev", "eval-a-dev"]

    with pytest.raises(RuntimeError, match=r"lists \["):
        mlflow_ui.check_dev_runs(before, after)


def test_fit_budget_leaves_pngs_within_the_budget_as_they_are(mlflow_ui, tmp_path):
    paths = random_pngs(tmp_path, 2)
    before = {path: path.read_bytes() for path in paths}
    total = sum(len(content) for content in before.values())

    assert mlflow_ui.fit_budget(paths, total) == total
    assert {path: path.read_bytes() for path in paths} == before


def test_fit_budget_quantizes_every_png_to_a_palette_when_they_weigh_more(mlflow_ui, tmp_path):
    paths = random_pngs(tmp_path, 2)
    total = sum(path.stat().st_size for path in paths)

    fitted = mlflow_ui.fit_budget(paths, total - 1)

    assert fitted == sum(path.stat().st_size for path in paths) < total - 1
    for path in paths:
        with Image.open(path) as image:
            assert (image.mode, image.size) == ("P", (64, 64))
            assert len(image.getcolors(maxcolors=256)) <= 256


def test_fit_budget_refuses_pngs_over_the_budget_even_quantized(mlflow_ui, tmp_path):
    paths = random_pngs(tmp_path, 2)

    with pytest.raises(ValueError, match="over the budget of 1,000 bytes"):
        mlflow_ui.fit_budget(paths, 1_000)


def test_every_step_of_the_views_has_its_action(mlflow_ui):
    assert tuple(mlflow_ui.STEP_ACTIONS) == mlflow_ui.STEPS
    assert all(callable(action) for action in mlflow_ui.STEP_ACTIONS.values())


def test_notebook_05_shows_each_screen_once_and_the_folder_holds_only_the_screens(mlflow_ui):
    # The file names do not depend on the ids a store gives the views.
    ids = mlflow_ui.ScreenIds(
        experiment_id="1",
        dataset_id="d-dev",
        record_id="dr-0",
        dev_runs={"descode2tag-json": "R", "tfidf": "T"},
        trace_id="tr-0",
        prompt_name="descode2tag",
        prompt_version="2",
        window=("2026-10-09T09:40:00.000Z", "2026-10-09T10:57:00.000Z"),
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    planned = [view.file for view in mlflow_ui.screens(ids)]
    shown = [file for source in code for file in SCREEN_IMAGE.findall(source)]

    assert len(planned) == 13
    assert shown == planned
    # The search's views have a folder of their own inside it (MLFLOW_SCREEN_GEPA_FOLDER).
    assert sorted(path.name for path in MLFLOW_SCREENS_DIR.iterdir() if path.is_file()) == planned


def drop_chosen_tag(client, built):
    """Remove the search run's chosen tag, as a store the live log alone wrote."""
    client.delete_tag(built.run_id, "chosen")


def add_search_run(client, built):
    """Add a second search run of the method to the experiment."""
    client.create_run(built.experiment_id, tags={"search": "jev-statement-code-gepa"})


def drop_tag_alias(client, built):  # pylint: disable=unused-argument
    """Remove the chosen alias of the prompt of the compared tag's criteria."""
    client.delete_prompt_alias(GEPA_PROMPT, "chosen")


def keep_gepa_store(client, built):  # pylint: disable=unused-argument
    """Leave the store as it is."""


def test_gepa_screen_ids_read_the_search_run_and_the_chosen_versions(mlflow_ui, gepa_store):
    built = gepa_store(chosen=2)

    ids = mlflow_ui.gepa_screen_ids(built.tracking_uri)

    assert ids == mlflow_ui.GepaScreenIds(
        experiment_id=built.experiment_id,
        run_id=built.run_id,
        tag_prompt=GEPA_PROMPT,
        tag_version="2",
        convention_version="2",
    )


def test_gepa_screen_ids_give_version_1_of_a_convention_the_chosen_words_keep(
    mlflow_ui, gepa_store
):
    built = gepa_store(chosen=1)

    ids = mlflow_ui.gepa_screen_ids(built.tracking_uri)

    assert (ids.tag_version, ids.convention_version) == ("2", "1")


@pytest.mark.parametrize(
    ("change", "settings", "message"),
    [
        (drop_chosen_tag, {}, r"The search run \w+ names no chosen candidate: replay the frozen"),
        (add_search_run, {}, r"holds 2 search runs of jev-statement-code-gepa, not one: rebuild"),
        (drop_tag_alias, {}, rf"The prompt {GEPA_PROMPT} has no chosen version: replay"),
        (
            keep_gepa_store,
            {"tag": KEPT_TAG},
            rf"The chosen words keep the seed's {re.escape(KEPT_TAG)} criteria",
        ),
    ],
)
def test_gepa_screen_ids_refuse_a_store_that_lacks_what_a_view_opens(
    mlflow_ui, gepa_store, change, settings, message
):
    built = gepa_store(chosen=2)
    change(mlflow.MlflowClient(built.tracking_uri), built)

    with pytest.raises(ValueError, match=message):
        mlflow_ui.gepa_screen_ids(built.tracking_uri, **settings)


def test_gepa_screen_ids_refuse_a_store_with_no_search_and_name_log_search(mlflow_ui, store):
    # The methods' store of notebook 05 holds no search until log-search replays it.
    with pytest.raises(ValueError, match=r"holds no jev-gepa experiment: replay .* log-search"):
        mlflow_ui.gepa_screen_ids(store["tracking_uri"])


def test_gepa_screen_ids_refuse_a_missing_store_and_create_none(mlflow_ui, tmp_path):
    database = tmp_path / "mlflow.db"

    with pytest.raises(ValueError, match=f"{re.escape(str(database))} does not exist: rebuild"):
        mlflow_ui.gepa_screen_ids(f"sqlite:///{database}")

    assert not database.exists()


def test_the_gepa_views_their_files_routes_viewports_and_steps(mlflow_ui):
    ids = mlflow_ui.GepaScreenIds(
        experiment_id="7",
        run_id="S",
        tag_prompt="jev-gepa-graphs",
        tag_version="2",
        convention_version="3",
    )
    candidates = "tags.%60mlflow.parentRunId%60%20%3D%20%27S%27"
    order = "metrics.%60selection_mean%60"

    views = mlflow_ui.gepa_screens(ids)

    assert [(view.file, view.route, view.viewport, view.steps) for view in views] == [
        ("01_gepa_search_chart.png", "experiments/7/runs/S/model-metrics", (1600, 1800), ()),
        (
            "02_gepa_candidates.png",
            f"experiments/7/runs?searchFilter={candidates}&orderByKey={order}&orderByAsc=false",
            (2000, 1000),
            (),
        ),
        (
            "03_gepa_prompt_compare.png",
            "prompts/jev-gepa-graphs?promptVersion=2",
            (1600, 1000),
            ("compare_with_seed",),
        ),
        (
            "04_gepa_convention_compare.png",
            "prompts/jev-gepa-convention?promptVersion=3",
            (1600, 1000),
            ("compare_with_seed",),
        ),
    ]
    assert {step for view in views for step in view.steps} == {mlflow_ui.STEPS[-1]}


def test_the_gepa_views_leave_out_the_convention_when_the_chosen_words_keep_it(mlflow_ui):
    ids = mlflow_ui.GepaScreenIds("7", "S", "jev-gepa-graphs", "2", "1")

    assert [view.file for view in mlflow_ui.gepa_screens(ids)] == GEPA_FILES[:3]


def test_the_set_option_names_the_views_and_their_default_folder(mlflow_ui):
    parser = mlflow_ui._parser()  # pylint: disable=protected-access

    defaults = parser.parse_args([])

    assert (defaults.set, defaults.out) == ("methods", None)
    assert parser.parse_args(["--set", "gepa"]).set == "gepa"
    assert mlflow_ui.VIEW_FOLDERS == {
        "methods": MLFLOW_SCREENS_DIR,
        "gepa": MLFLOW_SCREEN_GEPA_FOLDER,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["--set", "everything"])


def test_main_replaces_the_gepa_pngs_in_their_folder_once_every_view_was_captured(
    mlflow_ui, gepa_store, monkeypatch, tmp_path, capsys
):
    built = gepa_store(chosen=2)
    events, shot = [], []

    def complete_capture(views, base_url, folder):  # pylint: disable=unused-argument
        shot.append(views)
        write_pngs(views, folder)
        return []

    out = tmp_path / "mlflow" / "gepa"
    monkeypatch.setitem(mlflow_ui.VIEW_FOLDERS, "gepa", out)
    command_line = [
        "--set",
        "gepa",
        "--tracking-uri",
        built.tracking_uri,
        "--artifact-dir",
        str(built.artifact_dir),
    ]

    status = mlflow_ui.main(
        command_line, server=fake_server(built.tracking_uri, events), shoot=complete_capture
    )

    views = mlflow_ui.gepa_screens(mlflow_ui.gepa_screen_ids(built.tracking_uri))
    assert status == 0
    assert shot == [views]
    assert sorted(contents_of(out)) == [view.file for view in views] == GEPA_FILES
    assert events[-1] == ("stop",)
    assert not leftovers(out)
    assert capsys.readouterr().out == f"Wrote 4 PNGs to {out}\n"


def test_main_exits_1_when_the_server_serves_another_search(
    mlflow_ui, gepa_store, monkeypatch, tmp_path, capsys
):
    built = gepa_store(chosen=2)
    read_ids, events, shots = mlflow_ui.gepa_screen_ids, [], []

    def another_search_from_the_server(tracking_uri, **settings):
        ids = read_ids(built.tracking_uri, **settings)
        if tracking_uri == built.tracking_uri:
            return ids
        return replace(ids, run_id="0" * 32)

    monkeypatch.setattr(mlflow_ui, "gepa_screen_ids", another_search_from_the_server)
    out = tmp_path / "gepa"

    status = mlflow_ui.main(
        ["--set", "gepa", *options(built, out)],
        server=fake_server("http://127.0.0.1:5517", events),
        shoot=lambda *args: shots.append(args),
    )

    assert status == 1
    assert "serves another store" in capsys.readouterr().err
    assert not shots
    assert events[-1] == ("stop",)
    assert not out.exists()


def test_notebook_10_shows_each_gepa_screen_once_and_the_folder_holds_only_them(mlflow_ui):
    # The views depend on whether the chosen words changed the convention: the folder says so.
    saved = sorted(path.name for path in MLFLOW_SCREEN_GEPA_FOLDER.iterdir())
    convention = "2" if "04_gepa_convention_compare.png" in saved else "1"
    ids = mlflow_ui.GepaScreenIds("1", "S", "jev-gepa-graphs", "2", convention)
    notebook = json.loads(NOTEBOOK_10.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    planned = [view.file for view in mlflow_ui.gepa_screens(ids)]
    shown = [file for source in code for file in GEPA_SCREEN_IMAGE.findall(source)]

    assert shown == planned
    assert saved == planned
