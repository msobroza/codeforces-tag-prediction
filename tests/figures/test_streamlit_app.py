"""Tests for the Streamlit app's capture (notebooks/figures/streamlit_app.py), all but its browser.

CI installs neither playwright nor streamlit: the browser's half runs on a stand-in playwright,
and the app's server is a stand-in that answers Streamlit's health check.
"""

import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from codeforces_tags.config import CLI_REQUIRED_FIELDS, PROJECT_ROOT
from codeforces_tags.serving.problems import FORM_FIELDS, problem_from_form

APP = PROJECT_ROOT / "apps" / "streamlit_app.py"
PORT = 8599
SERVER_URL = "http://127.0.0.1:8080"
UI_URL = "http://127.0.0.1:5000"
# Loads the capture by path in a fresh interpreter, then says which heavy libraries it imported.
FRESH_LOAD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("streamlit_capture", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules["streamlit_capture"] = module
spec.loader.exec_module(module)
print(*(name in sys.modules for name in ("playwright", "streamlit", "mlflow", "torch")))
"""
# Stands in for the app's server: it answers Streamlit's health check, and nothing else.
HEALTH_SERVER = """
import http.server, sys
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/_stcore/health" else 404)
        self.end_headers()
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""
# Variables the app's server never gets: credentials of LLM providers, clouds, model hubs and
# databases.
CREDENTIALS = (
    "OPENAI_API_KEY",
    "gemini_api_key",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "DATABASE_PASSWORD",
    "AZURE_CREDENTIALS",
)
# The labels of the form's text areas, as the app shows them, in its order.
FORM_LABELS = [
    "Description (required)",
    "Input specification",
    "Output specification",
    "Notes",
    "Solution (required)",
]


class RecordingContext:
    """A browser context that records each call the capture makes on it, by the method's name."""

    def __init__(self, **options):
        """Keep the options the context was opened with; no call is recorded yet."""
        self.options, self.calls = options, defaultdict(list)

    def __getattr__(self, name):
        """Return a method that records its arguments under ``name`` (``route``, ``on``...)."""
        return lambda *args: self.calls[name].append(args)

    def emit(self, event, value):
        """Call each handler that ``on`` registered for the event, in the order of ``on``."""
        for handler in [handler for name, handler in self.calls["on"] if name == event]:
            handler(value)


def routed(url, aborted):
    """Return a routed request or WebSocket of ``url``; its abort appends ``url`` to ``aborted``."""
    return SimpleNamespace(
        url=url, request=SimpleNamespace(url=url), abort=lambda: aborted.append(url)
    )


def recorded_popen(streamlit_capture, monkeypatch):
    """Record each process the capture starts, with the options it got, in the list returned."""
    started, popen = [], subprocess.Popen

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)  # pylint: disable=consider-using-with
        started.append((process, kwargs))
        return process

    monkeypatch.setattr(streamlit_capture.subprocess, "Popen", record)
    return started


def fake_server(app_url, events):
    """Return a stand-in for streamlit_server: it yields ``app_url``, recording start and stop."""

    @contextmanager
    def serve(url, mlflow_url, port):
        events.append(("start", url, mlflow_url, port))
        try:
            yield app_url
        finally:
            events.append(("stop",))

    return serve


def old_pngs(tmp_path):
    """Return a folder of PNGs from an earlier capture, and its contents by file name."""
    out = tmp_path / "out"
    out.mkdir()
    contents = {"test_problem.png": b"old test problem", "new_problem.png": b"old new problem"}
    for name, content in contents.items():
        (out / name).write_bytes(content)
    return out, contents


def contents_of(folder):
    """Return the files of a folder by name."""
    return {path.name: path.read_bytes() for path in folder.iterdir()}


def leftovers(out):
    """Return the temporary folders a capture left beside its output folder."""
    return list(out.parent.glob(f".{out.name}-*"))


def capture_one_screenshot(app_url, folder):  # pylint: disable=unused-argument
    """Write the first screenshot only, and report no request outside the app's server."""
    (folder / "test_problem.png").write_bytes(b"png")
    return []


def capture_an_extra_file(app_url, folder):  # pylint: disable=unused-argument
    """Write both screenshots and one more file, and report no request outside."""
    for name in ("test_problem.png", "new_problem.png", "stale.png"):
        (folder / name).write_bytes(b"png")
    return []


@pytest.fixture(autouse=True)
def no_server_variables(monkeypatch):
    """Clear the servers' variables that the shell may set: the capture would read them."""
    monkeypatch.delenv("CODEFORCES_SERVING_URL", raising=False)
    monkeypatch.delenv("CODEFORCES_MLFLOW_UI_URL", raising=False)


@pytest.fixture(name="servers_answer")
def servers_answer_fixture(streamlit_capture, monkeypatch):
    """Make the model server and the MLflow UI answer; return the URLs they were asked about."""
    asked = []

    def answer(url):
        asked.append(url)
        return True

    monkeypatch.setattr(streamlit_capture, "ping", answer)
    monkeypatch.setattr(streamlit_capture, "ui_alive", answer)
    return asked


@pytest.fixture(name="browser")
def browser_fixture(monkeypatch):
    """Make a stand-in browser the Chrome that ``capture`` launches, through a stand-in playwright.

    The browser opens ``RecordingContext``s, kept in ``contexts``, and sets ``closed`` once closed.
    """
    browser = SimpleNamespace(contexts=[], closed=False)

    def new_context(**options):
        browser.contexts.append(RecordingContext(**options))
        return browser.contexts[-1]

    def close():
        browser.closed = True

    browser.new_context, browser.close = new_context, close
    launcher = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **options: browser))
    sync_api = SimpleNamespace(Error=RuntimeError, sync_playwright=lambda: nullcontext(launcher))
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace(sync_api=sync_api))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    return browser


def test_importing_the_capture_imports_neither_playwright_nor_streamlit_nor_mlflow(
    streamlit_capture,
):
    # CI has neither playwright nor streamlit: only the browser's half imports playwright, when it
    # runs, and the app runs in a process of its own.
    result = subprocess.run(
        [sys.executable, "-c", FRESH_LOAD, streamlit_capture.__file__],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["False", "False", "False", "False"]


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080", "http://127.0.0.1:5000/"])
def test_local_url_keeps_an_http_url_of_127_0_0_1(streamlit_capture, url):
    assert streamlit_capture.local_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080",
        "https://127.0.0.1:8080",
        "http://127.0.0.2:8080",
        "http://127.0.0.1:8080@example.com/",
        "http://127.0.0.1:port",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:0",
    ],
)
def test_local_url_refuses_another_host_scheme_or_port(streamlit_capture, url):
    with pytest.raises(ValueError, match="is not an http URL of 127.0.0.1"):
        streamlit_capture.local_url(url)


@pytest.mark.parametrize(
    ("model_server", "ui", "message"),
    [
        (False, True, r"No model server answers http://127\.0\.0\.1:8080/ping"),
        (
            True,
            False,
            r"No MLflow UI answers http://127\.0\.0\.1:5000/health: start it with uv run "
            r"python -m codeforces_tags\.serving mlflow-ui",
        ),
    ],
)
def test_check_servers_refuses_a_server_that_does_not_answer(
    streamlit_capture, monkeypatch, model_server, ui, message
):
    monkeypatch.setattr(streamlit_capture, "ping", lambda _url: model_server)
    monkeypatch.setattr(streamlit_capture, "ui_alive", lambda _url: ui)

    with pytest.raises(RuntimeError, match=message):
        streamlit_capture.check_servers(SERVER_URL, UI_URL)


def test_streamlit_command_serves_the_app_headless_on_127_0_0_1_without_usage_statistics(
    streamlit_capture,
):
    assert streamlit_capture.streamlit_command(PORT) == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8599",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
        "--theme.base",
        "light",
    ]


@pytest.mark.parametrize("name", CREDENTIALS)
def test_app_environment_holds_no_credential(streamlit_capture, monkeypatch, name):
    monkeypatch.setenv(name, "placeholder")

    env = streamlit_capture.app_environment(SERVER_URL, UI_URL)

    assert name not in env
    assert os.environ[name] == "placeholder"


def test_app_environment_names_both_servers_keeps_the_dataset_and_blocks_every_proxy(
    streamlit_capture, monkeypatch
):
    monkeypatch.setenv("CODEFORCES_DATA_DIR", "/data")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    monkeypatch.setenv("no_proxy", "example.com")

    env = streamlit_capture.app_environment(SERVER_URL, UI_URL)

    assert env["CODEFORCES_SERVING_URL"] == SERVER_URL
    assert env["CODEFORCES_MLFLOW_UI_URL"] == UI_URL
    assert env["CODEFORCES_DATA_DIR"] == "/data"
    assert {name: env[name] for name in streamlit_capture.PROXY_VARIABLES} == dict.fromkeys(
        streamlit_capture.PROXY_VARIABLES, "http://127.0.0.1:9"
    )
    assert (env["NO_PROXY"], env["no_proxy"]) == ("127.0.0.1,localhost", "127.0.0.1,localhost")


def test_the_blocking_proxy_stops_a_request_to_another_host_at_once(streamlit_capture, monkeypatch):
    # Streamlit checks its latest version on PyPI with a library that honours the proxy
    # variables, as urllib's default opener does: in the app's server, nothing reaches PyPI.
    env = streamlit_capture.app_environment(SERVER_URL, UI_URL)
    for name in (*streamlit_capture.PROXY_VARIABLES, *streamlit_capture.NO_PROXY_VARIABLES):
        monkeypatch.setenv(name, env[name])

    with pytest.raises(urllib.error.URLError, match="Connection refused"):
        urllib.request.build_opener().open("https://pypi.org/pypi/streamlit/json", timeout=5)


def test_streamlit_server_serves_the_app_on_127_0_0_1_and_stops_its_process_group(
    streamlit_capture, monkeypatch
):
    started = recorded_popen(streamlit_capture, monkeypatch)
    monkeypatch.setattr(
        streamlit_capture,
        "streamlit_command",
        lambda port: [sys.executable, "-c", HEALTH_SERVER, str(port)],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    port = streamlit_capture.free_port()

    with streamlit_capture.streamlit_server(SERVER_URL, UI_URL, port) as url:
        served = streamlit_capture._healthy(url)  # pylint: disable=protected-access

    assert (url, served) == (f"http://127.0.0.1:{port}", True)
    assert len(started) == 1
    process, options = started[0]
    assert (options["cwd"], options["stdin"], options["start_new_session"]) == (
        PROJECT_ROOT,
        subprocess.DEVNULL,
        True,
    )
    assert options["env"] == streamlit_capture.app_environment(SERVER_URL, UI_URL)
    assert "OPENAI_API_KEY" not in options["env"]
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("print('no such app'); raise SystemExit(3)", r"streamlit exited with 3:\nno such app"),
        (
            "import time; time.sleep(30)",
            r"streamlit did not answer http://127\.0\.0\.1:\d+/_stcore/health in 1 s",
        ),
    ],
)
def test_streamlit_server_refuses_an_app_that_exits_or_never_answers(
    streamlit_capture, monkeypatch, script, message
):
    monkeypatch.setattr(
        streamlit_capture, "streamlit_command", lambda _port: [sys.executable, "-c", script]
    )
    monkeypatch.setattr(streamlit_capture, "APP_START_S", 1)
    port = streamlit_capture.free_port()

    with ExitStack() as stack, pytest.raises(RuntimeError, match=message):
        stack.enter_context(streamlit_capture.streamlit_server(SERVER_URL, UI_URL, port))


def test_form_index_finds_each_text_field_of_the_form_by_its_label(streamlit_capture):
    found = {column: streamlit_capture.form_index(FORM_LABELS, column) for column in FORM_FIELDS}

    assert found == {
        "prob_desc_description": 0,
        "prob_desc_input_spec": 1,
        "prob_desc_output_spec": 2,
        "prob_desc_notes": 3,
        "source_code": 4,
    }


@pytest.mark.parametrize("labels", [FORM_LABELS[1:], [*FORM_LABELS, "Description, again"]])
def test_form_index_refuses_a_form_whose_labels_changed(streamlit_capture, labels):
    with pytest.raises(RuntimeError, match="not one is 'description'"):
        streamlit_capture.form_index(labels, "prob_desc_description")


def test_the_tagged_problem_is_the_first_of_the_picker_with_a_true_focus_tag(streamlit_capture):
    # The picker's order, not the sample ids': problem 3 comes after problem 7 in it.
    truth = pd.DataFrame(
        {"math": [0, 0, 1, 0], "strings": [0, 1, 0, 1]}, index=pd.Index([20, 7, 3, 9])
    )

    assert streamlit_capture.first_tagged_problem(truth) == 7


def test_the_tagged_problem_is_refused_when_no_problem_has_a_true_focus_tag(streamlit_capture):
    truth = pd.DataFrame({"math": [0, 0], "strings": [0, 0]}, index=pd.Index([13, 14]))

    with pytest.raises(RuntimeError, match="no test-set problem has a true focus tag"):
        streamlit_capture.first_tagged_problem(truth)


def test_content_clip_keeps_the_page_width_down_to_the_content_and_a_margin(streamlit_capture):
    width, height = streamlit_capture.VIEWPORT
    margin = streamlit_capture.CONTENT_MARGIN

    assert streamlit_capture.content_clip(1000.2) == {
        "x": 0,
        "y": 0,
        "width": width,
        "height": 1001 + margin,
    }
    assert streamlit_capture.content_clip(height - margin)["height"] == height


@pytest.mark.parametrize(
    ("bottom", "message"),
    [
        (float("-inf"), "found no content"),
        (float("nan"), "found no content"),
        (0.0, "found no content"),
        (1800.5 - 40, "reaches the bottom of the 1700 x 1800 viewport"),
    ],
)
def test_content_clip_refuses_a_page_without_content_or_with_content_the_viewport_cuts(
    streamlit_capture, bottom, message
):
    with pytest.raises(RuntimeError, match=message):
        streamlit_capture.content_clip(bottom)


def test_the_typed_problem_fills_the_form_but_its_notes_and_the_form_accepts_it(
    streamlit_capture,
):
    problem = problem_from_form(streamlit_capture.NEW_PROBLEM, [])

    assert set(streamlit_capture.NEW_PROBLEM) == set(FORM_FIELDS) - {"prob_desc_notes"}
    assert set(CLI_REQUIRED_FIELDS) <= set(streamlit_capture.NEW_PROBLEM)
    assert problem.loc[0, "prob_desc_notes"] is None


def test_the_app_holds_every_text_the_capture_clicks_or_waits_for(streamlit_capture):
    source = APP.read_text(encoding="utf-8")

    assert [text for text in streamlit_capture.APP_TEXTS if text not in source] == []


def test_the_screenshots_are_notebook_12s_two_in_their_folder(streamlit_capture):
    assert list(streamlit_capture.SHOTS) == ["test_problem.png", "new_problem.png"]
    assert streamlit_capture.SCREENS_DIR == PROJECT_ROOT / "notebooks" / "figures" / "streamlit"


@pytest.mark.parametrize(
    "url",
    [
        f"http://127.0.0.1:{PORT}/",
        f"ws://127.0.0.1:{PORT}/_stcore/stream",
        "data:image/png;base64,",
        f"blob:http://127.0.0.1:{PORT}/1",
        "about:blank",
    ],
)
def test_the_browser_may_load_the_app_server_and_inline_data(streamlit_capture, url):
    assert streamlit_capture.is_allowed(url, PORT)


@pytest.mark.parametrize(
    "url",
    [
        f"http://localhost:{PORT}/",
        "http://127.0.0.1:8080/",
        "https://fonts.example.com/a.woff2",
        f"http://127.0.0.1:{PORT}@example.com/",
        "http://127.0.0.1:port/",
        "wss://example.com/live",
    ],
)
def test_the_browser_may_load_nothing_else(streamlit_capture, url):
    assert not streamlit_capture.is_allowed(url, PORT)


def test_guarded_context_is_light_us_english_utc_and_blocks_service_workers(streamlit_capture):
    browser = SimpleNamespace(new_context=RecordingContext)

    context, outside = streamlit_capture.guarded_context(browser, PORT)

    # The page's size first, then the guard, then what the screenshots' rendering depends on.
    assert context.options == {
        "viewport": {"width": 1700, "height": 1800},
        "service_workers": "block",
        "color_scheme": "light",
        "timezone_id": "UTC",
        "locale": "en-US",
        "device_scale_factor": 1,
    }
    assert outside == []


def test_guarded_context_aborts_and_records_every_request_that_leaves_the_app_server(
    streamlit_capture,
):
    browser = SimpleNamespace(new_context=RecordingContext)
    context, outside = streamlit_capture.guarded_context(browser, PORT)
    [(leaves, abort)] = context.calls["route"]
    [(socket_leaves, hold)] = context.calls["route_web_socket"]
    font, live = "https://fonts.example.com/inter.woff2", "wss://example.com/live"
    aborted = []

    abort(routed(font, aborted))
    hold(routed(live, aborted))
    for url in (font, "https://pypi.org/pypi/streamlit/json", f"http://127.0.0.1:{PORT}/"):
        context.emit("request", SimpleNamespace(url=url))

    assert [leaves(font), leaves(f"http://127.0.0.1:{PORT}/_stcore/health")] == [True, False]
    assert [socket_leaves(live), socket_leaves(f"ws://127.0.0.1:{PORT}/_stcore/stream")] == [
        True,
        False,
    ]
    # Only the request is aborted: a routed WebSocket that never connects stays in the browser.
    assert aborted == [font]
    assert outside == [font, live, "https://pypi.org/pypi/streamlit/json"]


def test_capture_takes_both_screenshots_counts_their_requests_and_closes_the_browser(
    streamlit_capture, browser, monkeypatch, tmp_path, capsys
):
    shots = []

    def shoot(context, app_url, png, steps):
        # A request to the app, its WebSocket and inline data, for each screenshot.
        for url in (app_url, f"ws://127.0.0.1:{PORT}/_stcore/stream", "data:font/woff2;,"):
            context.emit("request", SimpleNamespace(url=url))
        shots.append((png.name, steps))
        png.write_bytes(b"png")

    monkeypatch.setattr(streamlit_capture, "_shoot", shoot)

    assert streamlit_capture.capture(f"http://127.0.0.1:{PORT}", tmp_path) == []

    assert shots == list(streamlit_capture.SHOTS.items())
    assert capsys.readouterr().out == (
        f"4 requests, all to 127.0.0.1:{PORT}, and 2 inline\n"
        "test_problem.png: 3 bytes\n"
        "new_problem.png: 3 bytes\n"
    )
    assert browser.closed
    # The guard lets through the app's port, which capture read from its URL, and no other.
    [(leaves, _)] = browser.contexts[0].calls["route"]
    assert [leaves(f"http://127.0.0.1:{PORT}/"), leaves("http://127.0.0.1:1/")] == [False, True]


def test_capture_stops_at_the_first_screenshot_that_sent_a_request_outside(
    streamlit_capture, browser, monkeypatch, tmp_path
):
    shots, beacon = [], "https://telemetry.example.com/beacon"

    def shoot(context, app_url, png, steps):  # pylint: disable=unused-argument
        [(_, abort)] = context.calls["route"]
        abort(routed(beacon, []))
        shots.append(png.name)

    monkeypatch.setattr(streamlit_capture, "_shoot", shoot)

    assert streamlit_capture.capture(f"http://127.0.0.1:{PORT}", tmp_path) == [beacon]
    assert shots == ["test_problem.png"]
    assert browser.closed


def test_capture_reports_a_request_it_aborted_rather_than_the_failure_that_followed(
    streamlit_capture, browser, monkeypatch, tmp_path, capsys
):
    # A page that needs an outside script never loads once the guard aborts it, and its
    # screenshot times out: the capture reports the request, which explains the timeout.
    needed = "https://cdn.example.com/needed.js"

    def shoot(context, app_url, png, steps):  # pylint: disable=unused-argument
        [(_, abort)] = context.calls["route"]
        abort(routed(needed, []))
        raise TimeoutError("Timeout 120000ms exceeded.")

    monkeypatch.setattr(streamlit_capture, "_shoot", shoot)

    assert streamlit_capture.capture(f"http://127.0.0.1:{PORT}", tmp_path) == [needed]
    assert "Timeout 120000ms exceeded." in capsys.readouterr().err
    assert (browser.closed, list(tmp_path.iterdir())) == (True, [])


def test_capture_raises_the_failure_of_a_screenshot_when_no_request_left(
    streamlit_capture, browser, monkeypatch, tmp_path
):
    def shoot(context, app_url, png, steps):  # pylint: disable=unused-argument
        raise TimeoutError("Timeout 120000ms exceeded.")

    monkeypatch.setattr(streamlit_capture, "_shoot", shoot)

    with pytest.raises(TimeoutError, match="Timeout 120000ms exceeded."):
        streamlit_capture.capture(f"http://127.0.0.1:{PORT}", tmp_path)

    assert browser.closed


def test_main_replaces_both_pngs_once_both_screenshots_succeeded(
    streamlit_capture, servers_answer, monkeypatch, tmp_path, capsys
):
    events, shots = [], []

    def complete_capture(app_url, folder):
        shots.append(app_url)
        for name in streamlit_capture.SHOTS:
            (folder / name).write_bytes(f"png of {name}".encode())
        return []

    monkeypatch.setattr(streamlit_capture, "free_port", lambda: PORT)
    out, _ = old_pngs(tmp_path)
    command_line = ["--url", "http://127.0.0.1:8081/", "--mlflow-url", "http://127.0.0.1:5001"]
    server = fake_server(f"http://127.0.0.1:{PORT}", events)

    status = streamlit_capture.main(
        [*command_line, "--out", str(out)], server=server, shoot=complete_capture
    )

    assert status == 0
    assert servers_answer == ["http://127.0.0.1:8081", "http://127.0.0.1:5001"]
    assert events == [("start", "http://127.0.0.1:8081", "http://127.0.0.1:5001", PORT), ("stop",)]
    assert shots == [f"http://127.0.0.1:{PORT}"]
    assert contents_of(out) == {
        "test_problem.png": b"png of test_problem.png",
        "new_problem.png": b"png of new_problem.png",
    }
    assert not leftovers(out)
    assert capsys.readouterr().out == f"Wrote 2 PNGs to {out}\n"


def test_main_asks_the_servers_of_the_environment_by_default(
    streamlit_capture, servers_answer, monkeypatch, tmp_path
):
    events = []
    monkeypatch.setenv("CODEFORCES_SERVING_URL", "http://127.0.0.1:18080")
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", "http://127.0.0.1:5517")
    out, _ = old_pngs(tmp_path)

    status = streamlit_capture.main(
        ["--out", str(out)], server=fake_server("", events), shoot=capture_one_screenshot
    )

    assert status == 1
    assert servers_answer == ["http://127.0.0.1:18080", "http://127.0.0.1:5517"]
    assert events[0][1:3] == ("http://127.0.0.1:18080", "http://127.0.0.1:5517")


@pytest.mark.usefixtures("servers_answer")
def test_main_exits_2_and_writes_no_png_when_a_request_left_the_app_server(
    streamlit_capture, tmp_path, capsys
):
    events = []

    def leaking_capture(app_url, folder):  # pylint: disable=unused-argument
        for name in streamlit_capture.SHOTS:
            (folder / name).write_bytes(b"png")
        return ["https://pypi.org/pypi/streamlit/json"]

    out, before = old_pngs(tmp_path)

    status = streamlit_capture.main(
        ["--out", str(out)], server=fake_server("", events), shoot=leaking_capture
    )

    assert status == streamlit_capture.OUTSIDE_REQUEST_EXIT == 2
    assert "https://pypi.org/pypi/streamlit/json" in capsys.readouterr().err
    assert (events[-1], contents_of(out), leftovers(out)) == (("stop",), before, [])


@pytest.mark.usefixtures("servers_answer")
@pytest.mark.parametrize("shoot", [capture_one_screenshot, capture_an_extra_file])
def test_main_exits_1_and_writes_no_png_when_the_capture_missed_a_file(
    streamlit_capture, tmp_path, capsys, shoot
):
    events = []
    out, before = old_pngs(tmp_path)

    status = streamlit_capture.main(
        ["--out", str(out)], server=fake_server("", events), shoot=shoot
    )

    assert status == 1
    assert "no PNG was replaced" in capsys.readouterr().err
    assert (events[-1], contents_of(out), leftovers(out)) == (("stop",), before, [])


@pytest.mark.usefixtures("servers_answer")
def test_main_keeps_the_pngs_and_stops_the_app_when_the_capture_fails(streamlit_capture, tmp_path):
    events = []

    def failing_capture(app_url, folder):  # pylint: disable=unused-argument
        (folder / "test_problem.png").write_bytes(b"png")
        raise RuntimeError("The form's text areas are labelled []: not one is 'description'.")

    out, before = old_pngs(tmp_path)

    with pytest.raises(RuntimeError, match="text areas are labelled"):
        streamlit_capture.main(
            ["--out", str(out)], server=fake_server("", events), shoot=failing_capture
        )

    assert (events[-1], contents_of(out), leftovers(out)) == (("stop",), before, [])


@pytest.mark.parametrize("option", ["--url", "--mlflow-url"])
def test_main_refuses_a_server_off_127_0_0_1_before_asking_it_or_starting_the_app(
    streamlit_capture, servers_answer, tmp_path, option
):
    events = []

    with pytest.raises(ValueError, match="http://localhost:8080 is not an http URL of 127.0.0.1"):
        streamlit_capture.main(
            [option, "http://localhost:8080", "--out", str(tmp_path / "out")],
            server=fake_server("", events),
        )

    assert not servers_answer
    assert not events
    assert not (tmp_path / "out").exists()


def test_main_refuses_a_model_server_that_does_not_answer_before_starting_the_app(
    streamlit_capture, monkeypatch, tmp_path
):
    events = []
    monkeypatch.setattr(streamlit_capture, "ping", lambda _url: False)

    with pytest.raises(RuntimeError, match="No model server answers"):
        streamlit_capture.main(
            ["--url", SERVER_URL, "--out", str(tmp_path / "out")], server=fake_server("", events)
        )

    assert not events
    assert not (tmp_path / "out").exists()
