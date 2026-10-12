"""Capture the Streamlit app's two screenshots for notebook 12: a test-set problem and a new one.

The app (``apps/streamlit_app.py``) asks a model server for a problem's focus tags and links the
served tagging method to what MLflow tracked about it. This script starts the app itself, in
Streamlit's headless server on a free port of 127.0.0.1, against a model server and an MLflow UI
that already run on this machine (``--url`` and ``--mlflow-url``), opens it in headless Chrome
and writes two PNGs to ``notebooks/figures/streamlit/`` (``SCREENS_DIR``), which notebook 12
shows (``SHOTS``):

- ``test_problem.png``: the mode "Test-set problem" on the first problem of the picker that has a
  true focus tag (``first_tagged_problem``: a fixed rule, which never looks at a prediction;
  problem 14, whose true focus tag is strings, with the frozen split), once the model server
  answered it: the predicted, true and frozen focus tags, each focus tag's score beside its
  per-tag threshold, and the sidebar's links to the ``tag-methods`` experiment, the method's run
  and the prompt registry;
- ``new_problem.png``: the mode "New problem", with ``NEW_PROBLEM`` typed into the form, once the
  model server answered it.

Each screenshot is clipped to the page's content (``content_clip``): the whole width, down to the
lowest content of the sidebar and of the main area, and a margin.

The served method must be a score method (``f2llm_late_fusion``): each view waits for the chart
of the scores. Run it from the repository root, with the browser's and the app's dependency
groups, once both servers answer (notebooks/README.md, The Streamlit screenshots)::

    uv run --group figures --group app python notebooks/figures/streamlit_app.py

Nothing leaves the machine through the capture. Both URLs must be http URLs of 127.0.0.1
(``local_url``) and must answer before the app starts (``check_servers``): the model server's
``/ping`` and the MLflow UI's ``/health``, so that the screenshots show the sidebar's links live.
The app's server gets no credential, and its proxy variables point at a closed port of 127.0.0.1
(``BLOCKING_PROXY``): a library's request to another host, such as Streamlit's check of its
latest version on PyPI, fails at once, while the app's own requests go to 127.0.0.1 through no
proxy. The browser aborts every request that leaves the app's server (``guarded_context``).

Exit status: 0 when both PNGs were replaced; 1 when the capture missed a file; 2 when a request
left the app's server. Any other failure is raised: a URL off 127.0.0.1, a server that does not
answer, a control missing from the page. In every failure both PNGs stay as they were, and the
app's server is stopped in every case. ``tests/figures/`` tests all but the real browser, since CI
installs neither playwright nor streamlit. This is a render tool, like ``mlflow_ui.py``, not a
module the notebooks import.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from codeforces_tags.config import (
    MLFLOW_SERVER_LEFT_OUT,
    PROJECT_ROOT,
    SERVING_MLFLOW_UI_URL_ENV_VAR,
    SERVING_URL_ENV_VAR,
)
from codeforces_tags.serving.client import ping, server_url
from codeforces_tags.serving.local import LocalServer, free_port, stop
from codeforces_tags.serving.mlflow_links import mlflow_ui_url, ui_alive
from codeforces_tags.serving.problems import FORM_FIELDS, load_test_set

if TYPE_CHECKING:
    import pandas as pd
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Locator,
        Page,
        Request,
        Route,
        WebSocketRoute,
    )

# The app, and the folder of its screenshots.
APP = PROJECT_ROOT / "apps" / "streamlit_app.py"
SCREENS_DIR = PROJECT_ROOT / "notebooks" / "figures" / "streamlit"
# The exit status when a request left the app's server: nothing is written.
OUTSIDE_REQUEST_EXIT = 2
# The proxy of the app's server: the discard port of 127.0.0.1, where nothing listens. A request
# that honours the proxy variables and goes elsewhere than 127.0.0.1 fails there at once.
BLOCKING_PROXY = "http://127.0.0.1:9"
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")
NO_PROXY_HOSTS = "127.0.0.1,localhost"
# Streamlit's server answers GET HEALTH_PATH with 200 once it serves the app, which must happen
# within APP_START_S of its start.
HEALTH_PATH = "/_stcore/health"
APP_START_S = 120
# The page's size: wide enough for every column of the per-tag table beside the statement, and
# high enough for the whole answer and the sidebar, since Streamlit scrolls its main area inside
# the page, which a full-page screenshot would not show. Each screenshot keeps the page's width
# down to its content's bottom and CONTENT_MARGIN pixels more (content_clip).
VIEWPORT = (1700, 1800)
CONTENT_MARGIN = 40
# The longest wait for a control or an answer (the test set loads first, and a problem takes
# seconds on the CPU), then the time the chart of the scores takes to draw.
LOAD_TIMEOUT_MS = 120_000
SETTLE_MS = 2_000
# The texts of the app that the capture clicks or waits for: tests/figures/ checks that the app
# holds each.
NEW_MODE = "New problem"
PICKER = "Problem ("
PREDICT = "Predict"
ANSWER_HEADING = "Prediction of "
REGISTRY_LINK = "Prompt registry"
MISSING_LINK = "not logged in this store"
APP_TEXTS = (NEW_MODE, PICKER, PREDICT, ANSWER_HEADING, REGISTRY_LINK, MISSING_LINK)
# The chart of the scores, once Vega has drawn it: the canvas or svg that Vega classes "marks"
# (the chart's menu has an svg of its own).
CHART = ".vega-embed .marks"
# The problem typed into the form for new_problem.png, by raw field; the notes are left blank.
NEW_PROBLEM: dict[str, str] = {
    "prob_desc_description": (
        "You are given an integer $$$n$$$. Count the integers $$$k$$$ with "
        "$$$1 \\le k \\le n$$$ such that $$$\\gcd(k, n) = 1$$$."
    ),
    "prob_desc_input_spec": (
        "The only line contains one integer $$$n$$$ ($$$1 \\le n \\le 10^{12}$$$)."
    ),
    "prob_desc_output_spec": "Print the number of such integers $$$k$$$.",
    "source_code": (
        "n = int(input())\n"
        "result, rest, p = n, n, 2\n"
        "while p * p <= rest:\n"
        "    if rest % p == 0:\n"
        "        while rest % p == 0:\n"
        "            rest //= p\n"
        "        result -= result // p\n"
        "    p += 1\n"
        "if rest > 1:\n"
        "    result -= result // rest\n"
        "print(result)\n"
    ),
}
_HOST = "127.0.0.1"
LOCAL_SCHEMES = ("data", "blob", "about")
SERVER_SCHEMES = ("http", "https", "ws", "wss")
# True once Streamlit has finished its run: the app says it does not run (when it says so), no
# element of an earlier run is left stale, and no status ("Running...") shows.
_IDLE = """() => {
  const app = document.querySelector('[data-testid="stApp"]');
  const state = app ? app.getAttribute("data-test-script-state") : null;
  return (state === null || state === "notRunning")
    && !document.querySelector('[data-stale="true"]')
    && !document.querySelector('[data-testid="stStatusWidget"]');
}"""
# The lowest bottom, in pixels from the page's top, of the content of the sidebar and of the main
# area (the outermost vertical block of each); -Infinity when the page has neither.
_CONTENT_BOTTOM = """() => Math.max(...[
  '[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]',
  '[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]',
].map((selector) => document.querySelector(selector))
  .filter((element) => element !== null)
  .map((element) => element.getBoundingClientRect().bottom))"""
# The text of the Statement tab of the test-set view, which changes once a picked problem shows.
_STATEMENT_TEXT = """() => {
  const panel = document.querySelector('[role="tabpanel"]');
  return panel ? panel.innerText : null;
}"""
# Each text area scrolled back to its first line: a fill leaves it scrolled to its last.
_SCROLL_TO_TOP = "(areas) => areas.forEach((area) => { area.scrollTop = 0; })"


def local_url(url: str) -> str:
    """Check that a URL is an http URL of 127.0.0.1: the capture talks to nothing else.

    Args:
        url: The model server's or the MLflow UI's URL.

    Returns:
        The URL, without a trailing slash.

    Raises:
        ValueError: If its scheme is not http, its host is not exactly 127.0.0.1 (``localhost``
            and ``http://127.0.0.1:8080@example.com/``, whose host is ``example.com``, are
            refused), or its port is not a number from 1 to 65535.
    """
    parts = urlsplit(url)
    try:
        port_parsed = parts.port is None or parts.port > 0
    except ValueError:  # not a number, or out of range
        port_parsed = False
    if parts.scheme != "http" or parts.hostname != _HOST or not port_parsed:
        raise ValueError(f"{url} is not an http URL of {_HOST}: the capture talks to {_HOST} only.")
    return url.rstrip("/")


def check_servers(url: str, mlflow_url: str) -> None:
    """Refuse a model server or an MLflow UI that does not answer, before the app starts.

    Args:
        url: The model server.
        mlflow_url: The MLflow UI.

    Raises:
        RuntimeError: If the model server does not answer ``/ping`` with 200 (it is down, or
            still loading its method), or the MLflow UI ``/health``: the screenshots would show
            an error instead of the answer, or no link. The message says how to start it.
    """
    if not ping(url):
        raise RuntimeError(
            f"No model server answers {url}/ping: start one first (notebooks/README.md, The "
            "Streamlit screenshots), or pass --url."
        )
    if not ui_alive(mlflow_url):
        raise RuntimeError(
            f"No MLflow UI answers {mlflow_url}/health: start it with uv run python -m "
            "codeforces_tags.serving mlflow-ui, or pass --mlflow-url."
        )


def streamlit_command(port: int) -> list[str]:
    """Return the command that serves the app on 127.0.0.1, headless.

    Args:
        port: The port on 127.0.0.1.

    Returns:
        ``streamlit run apps/streamlit_app.py``, run by this interpreter: headless, with no
        usage statistics, no file watcher, the light theme and the minimal toolbar.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        _HOST,
        "--server.port",
        str(port),
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


def app_environment(url: str, mlflow_url: str) -> dict[str, str]:
    """Return the environment of the app's server.

    It is this one without any variable ``config.MLFLOW_SERVER_LEFT_OUT`` names (API keys,
    tokens, secrets, cloud keys, proxies); the proxy variables then point at ``BLOCKING_PROXY``,
    127.0.0.1 and localhost excepted, and the app reads the two servers' URLs from
    ``$CODEFORCES_SERVING_URL`` and ``$CODEFORCES_MLFLOW_UI_URL``. The dataset's variable,
    ``$CODEFORCES_DATA_DIR``, is kept: the app reads the test set from it.

    Args:
        url: The model server.
        mlflow_url: The MLflow UI.

    Returns:
        The environment.
    """
    left_out = re.compile(MLFLOW_SERVER_LEFT_OUT, re.IGNORECASE)
    kept = {name: value for name, value in os.environ.items() if not left_out.search(name)}
    blocked = dict.fromkeys(PROXY_VARIABLES, BLOCKING_PROXY)
    exempt = dict.fromkeys(NO_PROXY_VARIABLES, NO_PROXY_HOSTS)
    servers = {SERVING_URL_ENV_VAR: url, SERVING_MLFLOW_UI_URL_ENV_VAR: mlflow_url}
    return kept | blocked | exempt | servers


@contextmanager
def streamlit_server(url: str, mlflow_url: str, port: int) -> Iterator[str]:
    """Serve the app on 127.0.0.1 while the context lasts.

    The server runs from the repository root (Streamlit reads ``.streamlit/config.toml`` there),
    in its own session with no standard input, and is stopped with its whole process group.

    Args:
        url: The model server the app asks.
        mlflow_url: The MLflow UI the app links to.
        port: The port on 127.0.0.1.

    Yields:
        The app's URL, ``http://127.0.0.1:<port>``.

    Raises:
        RuntimeError: If the server exits, or does not answer ``HEALTH_PATH`` within
            ``APP_START_S``, with the last lines of its log.
    """
    with tempfile.TemporaryDirectory(prefix="streamlit-capture-") as folder:
        log = Path(folder) / "streamlit.log"
        with log.open("wb") as output:
            # The server outlives this block: stop() ends its process group.
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                streamlit_command(port),
                cwd=PROJECT_ROOT,
                env=app_environment(url, mlflow_url),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        server = LocalServer(process, f"http://{_HOST}:{port}", log)
        try:
            _wait_until_served(server)
            yield server.url
        finally:
            stop(server)


def _wait_until_served(server: LocalServer) -> None:
    """Wait until the app's server answers HEALTH_PATH; refuse one that exits or never does."""
    deadline = time.monotonic() + APP_START_S
    while time.monotonic() < deadline:
        healthy = _healthy(server.url)
        if server.process.poll() is not None:
            raise RuntimeError(
                f"streamlit exited with {server.process.returncode}:\n{_tail(server.log)}"
            )
        if healthy:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"streamlit did not answer {server.url}{HEALTH_PATH} in {APP_START_S} s:\n"
        f"{_tail(server.log)}"
    )


def _healthy(url: str) -> bool:
    """Tell whether Streamlit's server answers HEALTH_PATH with 200, through no proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}{HEALTH_PATH}", timeout=2) as response:
            return response.status == 200
    except OSError:  # refused, reset or timed out, or an error status
        return False


def _tail(log: Path, lines: int = 20) -> str:
    """Return the last lines of the app's server's log."""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def is_allowed(url: str, port: int) -> bool:
    """Tell whether the browser may load a URL: inline data, or the app's server.

    Args:
        url: A URL a page asks for.
        port: The app's port on 127.0.0.1.

    Returns:
        True for a ``data:``, ``blob:`` or ``about:`` URL, and for an http(s) or ws(s) URL of
        127.0.0.1 at ``port``; False for any other.
    """
    parts = urlsplit(url)
    if parts.scheme in LOCAL_SCHEMES:
        return True
    try:
        url_port = parts.port
    except ValueError:  # not a number, or out of range
        return False
    return parts.scheme in SERVER_SCHEMES and parts.hostname == _HOST and url_port == port


def guarded_context(browser: Browser, port: int) -> tuple[BrowserContext, list[str]]:
    """Open a browser context that loads nothing but the app's server.

    The context is light, in US English and UTC, of ``VIEWPORT``'s size, and blocks service
    workers, whose requests no route sees. Every request of a page to another address than
    ``127.0.0.1:<port>`` (``is_allowed``) is aborted and recorded; such a WebSocket is recorded
    and never connected. A request the routes do not see is recorded once it was sent.

    Args:
        browser: The browser.
        port: The app's port on 127.0.0.1.

    Returns:
        The context, and the list it fills with every URL that left the app's server, each once.
    """
    context = browser.new_context(
        color_scheme="light",
        device_scale_factor=1,
        locale="en-US",
        timezone_id="UTC",
        service_workers="block",
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
    )
    outside: list[str] = []

    def leaves(url: str) -> bool:
        return not is_allowed(url, port)

    def record(url: str) -> None:
        if url not in outside:
            outside.append(url)

    def abort(route: Route) -> None:
        record(route.request.url)
        route.abort()

    def hold(socket: WebSocketRoute) -> None:
        record(socket.url)

    def seen(request: Request) -> None:
        if leaves(request.url):
            record(request.url)

    context.route(leaves, abort)
    context.route_web_socket(leaves, hold)
    context.on("request", seen)
    return context, outside


def form_index(labels: list[str], column: str) -> int:
    """Find the form's text area of a raw field by its label.

    The app labels each text area with the field's name in ``FORM_FIELDS``, capitalised, and
    marks the required ones (``Description (required)``).

    Args:
        labels: The labels of the page's text areas, in the page's order.
        column: A raw field of ``FORM_FIELDS``.

    Returns:
        The index of the one label that starts with the field's name, in any case.

    Raises:
        RuntimeError: Unless exactly one label starts with it: the app's form changed.
    """
    name = FORM_FIELDS[column]
    found = [index for index, label in enumerate(labels) if label.lower().startswith(name)]
    if len(found) != 1:
        raise RuntimeError(f"The form's text areas are labelled {labels}: not one is {name!r}.")
    return found[0]


def first_tagged_problem(truth: pd.DataFrame) -> int:
    """Choose the test-set problem of ``test_problem.png``, by a fixed rule.

    The rule never looks at a prediction: the first problem, in the picker's order (the order of
    ``load_test_set``, which the app's picker keeps under "any focus tag"), whose true focus tags
    are not empty.

    Args:
        truth: The test-set problems' true focus tags, as ``load_test_set`` gives them (0/1, one
            column per focus tag, indexed by ``sample_id``).

    Returns:
        The problem's sample id.

    Raises:
        RuntimeError: If no problem has a true focus tag.
    """
    tagged = truth.index[truth.sum(axis="columns") > 0]
    if tagged.empty:
        raise RuntimeError("no test-set problem has a true focus tag: nothing to show.")
    return int(tagged[0])


def content_clip(bottom: float) -> dict[str, int]:
    """Return the part of the page a screenshot keeps: its width, down to its content.

    Args:
        bottom: The lowest bottom of the sidebar's and the main area's content, in pixels from the
            page's top (``_CONTENT_BOTTOM``).

    Returns:
        Playwright's clip: the page's whole width, from its top to ``CONTENT_MARGIN`` pixels below
        ``bottom``.

    Raises:
        RuntimeError: If no content was found (the page differs from what the capture expects), or
            if the content reaches the viewport's bottom: Streamlit scrolls its main area inside
            the page, so the screenshot would cut it (raise ``VIEWPORT``'s height).
    """
    if not math.isfinite(bottom) or bottom <= 0:
        raise RuntimeError(f"The capture found no content on the page (its bottom is {bottom}).")
    width, height = VIEWPORT
    kept = math.ceil(bottom) + CONTENT_MARGIN
    if kept > height:
        raise RuntimeError(
            f"The content reaches the bottom of the {width} x {height} viewport ({bottom:.0f} px): "
            "raise VIEWPORT's height."
        )
    return {"x": 0, "y": 0, "width": width, "height": kept}


def capture(app_url: str, folder: Path) -> list[str]:
    """Take the screenshots of ``SHOTS`` of the app served at ``app_url`` into ``folder``.

    Each screenshot opens a new page of one guarded context (``guarded_context``). The capture
    stops at the first that sent a request outside the app's server, and reports that request
    rather than a failure that follows it.

    Args:
        app_url: The app's URL.
        folder: Where the PNGs go, named as ``SHOTS`` says.

    Returns:
        The requests aborted because they left the app's server, up to the browser's close; an
        empty list when none did.

    Raises:
        Exception: What a screenshot raised (a timeout, a missing control), when no request had
            left the app's server.
    """
    # CI installs no playwright: only the browser's half imports it, when it runs.
    from playwright.sync_api import (  # pylint: disable=import-outside-toplevel,import-error
        Error,
        sync_playwright,
    )

    port = urlsplit(app_url).port
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome")
        except Error:  # no system Chrome
            browser = playwright.chromium.launch()
        try:
            context, outside = guarded_context(browser, port)
            requests: list[str] = []
            context.on("request", lambda request: requests.append(request.url))
            _shoot_all(context, app_url, folder, outside)
        finally:
            browser.close()
    if outside:  # sent by a page, or by a page as it closed
        return list(outside)
    served = sum(urlsplit(url).scheme in SERVER_SCHEMES for url in requests)
    print(f"{served} requests, all to {_HOST}:{port}, and {len(requests) - served} inline")
    for file in SHOTS:
        print(f"{file}: {(folder / file).stat().st_size:,} bytes")
    return []


def _shoot_all(context: BrowserContext, app_url: str, folder: Path, outside: list[str]) -> None:
    """Take each screenshot, up to the first that sent a request outside the app's server."""
    try:
        for file, steps in SHOTS.items():
            _shoot(context, app_url, folder / file, steps)
            if outside:
                return
    except Exception as error:
        if outside:
            print(
                f"A screenshot failed once a request had left the server: {error}", file=sys.stderr
            )
            return
        raise


def _shoot(context: BrowserContext, app_url: str, png: Path, steps: Callable[[Page], None]) -> None:
    """Open the app in a new page, take a screenshot's steps, wait for the answer, shoot it."""
    page = context.new_page()
    try:
        page.goto(app_url)
        steps(page)
        _wait_answered(page)
        page.mouse.move(0, 0)
        page.locator("textarea").evaluate_all(_SCROLL_TO_TOP)
        page.screenshot(path=png, clip=content_clip(page.evaluate(_CONTENT_BOTTOM)))
    finally:
        page.close()


def _test_problem(page: Page) -> None:
    """Pick the test-set problem of ``first_tagged_problem``, then ask the model server about it."""
    _, truth = load_test_set()
    sample_id = first_tagged_problem(truth)
    print(f"test_problem.png: problem {sample_id}, the first of the picker with a true focus tag")
    _pick(page, sample_id)
    _predict(page)


def _pick(page: Page, sample_id: int) -> None:
    """Select a problem in the picker, then wait until its statement shows and the run ends."""
    field = page.get_by_test_id("stSelectbox").filter(has_text=PICKER).locator("input")
    field.wait_for(timeout=LOAD_TIMEOUT_MS)
    _wait_idle(page)
    option = f"{sample_id}: "
    if field.input_value().startswith(option):  # already selected
        return
    shown = page.evaluate(_STATEMENT_TEXT)
    field.click()
    field.fill(option)  # the picker lists the options whose label holds the typed text
    page.get_by_role("option", name=re.compile(f"^{re.escape(option)}")).click()
    page.wait_for_function(
        f"(shown) => ({_STATEMENT_TEXT})() !== shown", arg=shown, timeout=LOAD_TIMEOUT_MS
    )
    _wait_idle(page)


def _new_problem(page: Page) -> None:
    """Type ``NEW_PROBLEM`` into the form of the mode "New problem", then ask about it."""
    mode = page.get_by_text(NEW_MODE, exact=True)
    mode.wait_for(timeout=LOAD_TIMEOUT_MS)
    _wait_idle(page)
    mode.click()
    areas = page.locator("textarea[aria-label]")
    areas.nth(len(FORM_FIELDS) - 1).wait_for(timeout=LOAD_TIMEOUT_MS)
    _wait_idle(page)
    labels = [area.get_attribute("aria-label") or "" for area in areas.all()]
    for column, text in NEW_PROBLEM.items():
        area = areas.nth(form_index(labels, column))
        area.fill(text)
        area.blur()  # Streamlit takes a text area's value when it loses the focus
        _wait_idle(page)
    _predict(page)


def _predict(page: Page) -> None:
    """Click Predict once the page is idle."""
    button = _predict_button(page)
    button.wait_for(timeout=LOAD_TIMEOUT_MS)
    _wait_idle(page)
    button.click()


def _predict_button(page: Page) -> Locator:
    """Return the page's Predict button (one per mode)."""
    return page.get_by_role("button", name=PREDICT, exact=True)


def _wait_answered(page: Page) -> None:
    """Wait for the answer, its chart and the sidebar's links, and refuse a link the store lacks."""
    heading = re.compile(f"^{re.escape(ANSWER_HEADING)}")
    page.get_by_role("heading", name=heading).wait_for(timeout=LOAD_TIMEOUT_MS)
    page.get_by_role("link", name=REGISTRY_LINK, exact=True).wait_for(timeout=LOAD_TIMEOUT_MS)
    page.locator(CHART).first.wait_for(timeout=LOAD_TIMEOUT_MS)
    _wait_idle(page)
    missing = page.get_by_text(MISSING_LINK).count()
    if missing:
        raise RuntimeError(
            f"The sidebar says {missing} links are {MISSING_LINK}: mirror the served method into "
            "the store first (notebooks/README.md, The Streamlit screenshots)."
        )
    page.wait_for_timeout(SETTLE_MS)


def _wait_idle(page: Page) -> None:
    """Wait until Streamlit has finished its run, LOAD_TIMEOUT_MS at most."""
    page.wait_for_function(_IDLE, timeout=LOAD_TIMEOUT_MS)


# Each screenshot, in the order notebook 12 shows them, and the steps that lead to its answer.
SHOTS: dict[str, Callable[[Page], None]] = {
    "test_problem.png": _test_problem,
    "new_problem.png": _new_problem,
}


def main(
    argv: list[str] | None = None,
    *,
    server: Callable[[str, str, int], AbstractContextManager[str]] = streamlit_server,
    shoot: Callable[[str, Path], list[str]] = capture,
) -> int:
    """Capture the app's two screenshots into ``--out``, both of them or none.

    The URLs are checked first (``local_url``), then the servers (``check_servers``), before the
    app starts. The PNGs are written to a temporary folder beside ``--out``, and replace those in
    it only when both were captured and no request left the app's server.

    Args:
        argv: The command line arguments; ``sys.argv[1:]`` if None.
        server: A context manager like ``streamlit_server``, which the tests replace.
        shoot: A function like ``capture``, which the tests replace.

    Returns:
        0 when both PNGs were replaced; 1 when the capture missed a file;
        ``OUTSIDE_REQUEST_EXIT`` when a request left the app's server.
    """
    args = _parser().parse_args(argv)
    url = local_url(server_url(args.url))
    mlflow_url = local_url(mlflow_ui_url(args.mlflow_url))
    check_servers(url, mlflow_url)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (
        server(url, mlflow_url, free_port()) as app_url,
        tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as folder,
    ):
        outside = shoot(app_url, Path(folder))
        if outside:
            print(
                "Requests left the app's server; no PNG was written:",
                *outside,
                sep="\n",
                file=sys.stderr,
            )
            return OUTSIDE_REQUEST_EXIT
        return replace_pngs(Path(folder), out)


def replace_pngs(folder: Path, out: Path) -> int:
    """Move the screenshots into ``out`` if ``folder`` holds exactly the files of ``SHOTS``.

    Args:
        folder: The capture's temporary folder.
        out: The screenshots' folder.

    Returns:
        0 once both moved; 1, with nothing moved, when the capture wrote other files.
    """
    written = sorted(path.name for path in folder.iterdir())
    expected = sorted(SHOTS)
    if written != expected:
        print(f"The capture wrote {written}, not {expected}: no PNG was replaced.", file=sys.stderr)
        return 1
    for file in SHOTS:
        os.replace(folder / file, out / file)
    print(f"Wrote {len(SHOTS)} PNGs to {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=None,
        help="the model server (default: $CODEFORCES_SERVING_URL, else http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--mlflow-url",
        default=None,
        help="the MLflow UI (default: $CODEFORCES_MLFLOW_UI_URL, else http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--out", type=Path, default=SCREENS_DIR, help="the PNGs' folder (default: %(default)s)"
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
