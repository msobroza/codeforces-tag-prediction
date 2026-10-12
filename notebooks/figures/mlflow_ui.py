"""Capture the MLflow UI's views of every problem (notebook 05) or of the search (notebook 10).

The store (``experiments/``) holds every problem of the frozen split, in the evaluation datasets
``codeforces-dev`` and ``codeforces-test``, and one evaluation run per method and part with a
trace per problem (``python -m codeforces_tags.tracking datasets`` and ``evaluate``). This script
starts a local ``mlflow server`` on that store, opens 13 views of its UI in headless Chrome
(``--set methods``, the default; ``--set gepa``, below, opens 3 or 4 others), and writes them as
PNGs to ``notebooks/figures/mlflow/`` (``config.MLFLOW_SCREENS_DIR``), which notebook 05 shows.
Run it from the repository root, once the store is rebuilt (``notebooks/README.md``, Figures)::

    uv run --group figures python notebooks/figures/mlflow_ui.py

Every view is of the dev part (ADR 0001): no route names the test dataset, a test run or a test
trace, and the pages that list traces by time show the window of the dev runs, which
``dev_window`` refuses when a test run meets it.

Nothing leaves the machine. Before its first MLflow call, the script refuses an environment with
telemetry, pickle loading or the job runner on (``check_safeguards``). The server listens on
127.0.0.1 (``config.MLFLOW_UI_PORT``, refused when in use) with one worker, no job runner, and no
credential or proxy in its environment; the script reads it through no proxy. The browser
aborts every request to another address that its routes see, and records any other (a redirect's
target); the script then exits with code 2 and writes no PNG. The UI's LLM features (Assistant,
Analyze, Detect Issues, Judges, Optimize, Playground) and the controls that write to the store are
never clicked (``NEVER_CLICK``). The committed PNGs come from a run inside a network namespace
that holds only loopback (``unshare -rn``), the backstop for what the script cannot see.

Exit status: 0 when every PNG of the set was replaced (13 for ``--set methods``, 3 or 4 for
``--set gepa``); 1 on any refusal or failure; 2 when a request left the server. In both failures
every PNG stays as it was, and the server is stopped in every case.
The PNGs are retaken, not byte-reproducible (timestamps, relative times, run colours), and
quantized to 256 colours when together they pass ``SCREENS_MAX_BYTES``. ``tests/figures/`` tests
all but the real browser, since CI does not install playwright: ``guarded_context`` on a stand-in
browser, and ``capture`` on a stand-in playwright, only for what it reports once the browser is
closed. This is a render tool, like ``estimator_diagram.py``, not a module the notebooks import.

``--set gepa`` takes the search's views instead, for notebook 10, into
``notebooks/figures/mlflow/gepa/`` (``config.MLFLOW_SCREEN_GEPA_FOLDER``), from a store into which
``python -m codeforces_tags.methods.typesafe_jev log-search --method jev-statement-code-gepa``
replayed the frozen search (``gepa_screen_ids``): the search run's chart, its candidates' runs
sorted by their selection-fold score, and the prompt of ``MLFLOW_SCREEN_GEPA_TAG``'s criteria,
then of the convention when the chosen words changed it, each comparing version 1 (the seed's
words) with the chosen version. The search reads dev folds 1 to 4 only, so these views show no
test problem (ADR 0001, docs/adr/0010). Every safeguard above holds for them too, and they are
replaced all together or not at all.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING
from urllib.parse import quote, urlsplit

# The tracking package sets the variables MLflow reads once, when it is imported (no telemetry,
# no hint to coding agents): it is imported before mlflow.
import codeforces_tags.tracking  # noqa: F401

# isort: split

import mlflow
from mlflow.exceptions import MlflowException
from PIL import Image

from codeforces_tags.config import (
    MLFLOW_DATASETS,
    MLFLOW_METHODS_EXPERIMENT,
    MLFLOW_SCREEN_COMPARE,
    MLFLOW_SCREEN_GEPA_FOLDER,
    MLFLOW_SCREEN_GEPA_TAG,
    MLFLOW_SCREEN_SAMPLE_ID,
    MLFLOW_SCREEN_TAG,
    MLFLOW_SCREENS_DIR,
    MLFLOW_SERVER_LEFT_OUT,
    MLFLOW_UI_PORT,
)
from codeforces_tags.evaluation.predictions import load_run_record
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev.search_tracking import (
    CHOSEN,
    PARENT_RUN_TAG,
    SEARCH_TAG,
    prompt_name,
)
from codeforces_tags.methods.typesafe_jev.settings import GEPA_NAME, SEARCH_EXPERIMENT
from codeforces_tags.tracking.environment import add_store_arguments, mlflow_env
from codeforces_tags.tracking.evaluation import COMPLETE_TAG, EVALUATION_RUN_KIND
from codeforces_tags.tracking.methods import RUN_KIND_TAG

if TYPE_CHECKING:
    from mlflow.entities import Run, RunInfo
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Locator,
        Page,
        Request,
        Route,
        WebSocketRoute,
    )

# The exit status when a request left the server: nothing is written.
OUTSIDE_REQUEST_EXIT = 2
# Each page waits at least this long: the UI polls, so "networkidle" never comes. Then no loading
# skeleton or spinner may show within LOAD_TIMEOUT_MS: the real store's tables of 3,988 traces
# load slower than a small store's.
WAIT_MS = 5_000
LOAD_TIMEOUT_MS = 60_000
# The server must answer GET /health within SERVER_START_S of its start; it is stopped with
# SIGTERM, then SIGKILL to what is left of its process group after SERVER_STOP_S.
SERVER_START_S = 60
SERVER_STOP_S = 10
# The variables of MLflow's environment that the capture checks, never sets (mlflow_env() sets
# MLflow's environment, ADR 0002 safeguard 7), in its own process and in the server's, and the
# value MLflow must read from each: no pickle loaded, no telemetry, no job runner. Unset, each
# reads as MLflow's default, which is the other value.
SERVER_SAFEGUARDS: dict[str, bool] = {
    "MLFLOW_ALLOW_PICKLE_DESERIALIZATION": False,
    "MLFLOW_DISABLE_TELEMETRY": True,
    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": False,
}
# How MLflow reads a boolean variable (mlflow.environment_variables), in any case: it refuses any
# other value.
_BOOLEANS = {"true": True, "1": True, "false": False, "0": False}
# The variables the server never gets: credentials, with which its LLM features (its gateway, its
# judges) could spend money with a provider or a cloud, and proxies (PROXY), through which its
# requests to 127.0.0.1 would leave the machine. The rule is config's, which `serving mlflow-ui`
# shares.
LEFT_OUT = re.compile(MLFLOW_SERVER_LEFT_OUT)
PROXY = re.compile(r"_PROXY$", re.IGNORECASE)
# The method every view compares MLFLOW_SCREEN_COMPARE with: the TF-IDF baseline of notebook 04.
BASELINE = "tfidf"
# The UI's localStorage, set before it loads (key_v<version>: JSON value): the UI's telemetry off
# (on, it starts a shared worker to send it) and its alert dismissed, the Detect Issues "Got it"
# popover shown, the Assistant panel already opened once and closed.
UI_PRESETS: dict[str, str] = {
    "mlflow.settings.telemetry.enabled_v1": "false",
    "mlflow.telemetry.info.alert.dismissed_v1": "true",
    "mlflow.detectIssues.guidanceShown_v1": "true",
    "mlflow.assistant.fab.autoOpenedOnFirstLoad_v1": "true",
    "mlflow.assistant.panelOpen_v1": "false",
}
# Run in every page before the UI: no SharedWorker, whose requests are neither routed nor reported
# to the context (the UI catches its absence, and then sends no telemetry); the presets (a page
# with no origin, such as about:blank, has no localStorage); and a browser that says it is online:
# offline, Chrome reports navigator.onLine false, and the UI then makes no API call.
INIT_SCRIPT = (
    "delete window.SharedWorker;\n"
    "try {\n"
    f"  for (const [key, value] of Object.entries({json.dumps(UI_PRESETS)})) {{\n"
    "    localStorage.setItem(key, value);\n"
    "  }\n"
    "} catch (error) {}\n"
    'Object.defineProperty(Navigator.prototype, "onLine", {get: () => true, configurable: true});\n'
)
# The Assistant's floating button and the trace drawer's Assistant button, hidden.
HIDDEN = 'button[data-assistant-ui="true"]{display:none!important}'
# The named interactions a view takes after it loads (STEP_ACTIONS holds what each does); the
# last is the search's views' only one.
STEPS = (
    "filter_dev_runs",
    "open_assessments",
    "load_every_trace",
    "show_cost_column",
    "compare_prompt_versions",
    "compare_with_seed",
)
# Every data-component-id the steps click: the trace drawer's assessments pane; the traces
# table's display menu, its Cost and Run name columns and its group of assessment columns; the
# prompt page's Compare mode, its switch of sides and its Text rendering. The steps click nothing
# else but the display menu's "Columns" entry and the prompt versions table's "Select as compared
# version" radios (compare_with_seed), which have no id.
CLICKS: tuple[str, ...] = (
    "shared.model-trace-explorer.assessments-pane-toggle",
    "mlflow.traces-v4.display.trigger",
    "mlflow.traces-v4.column-selector.item.cost",
    "mlflow.traces-v4.column-selector.item.run_name",
    "mlflow.traces-v4.display.columns.group-toggle-all",
    "mlflow.prompts.details.mode",
    "mlflow.prompts.details.switch_sides",
    "mlflow.prompts.compare.toggle-markdown-rendering",
)
# The controls of the opened pages that call an LLM or write to the store: never clicked.
NEVER_CLICK: tuple[str, ...] = (
    "mlflow.experiment.overview.detect-issues-button",
    "mlflow.traces-v4.detect-issues-button",
    "mlflow.chat-sessions.detect-issues-button",
    "mlflow.run-view-evaluations.analyze-button",
    "mlflow.evaluations_review.modal.analyze-trace",
    "mlflow.assistant.fab",
    "mlflow.assistant.trace_header_button",
    "mlflow.traces-v4.actions.run-judges",
    "mlflow.eval-runs.start-run-button",
    "mlflow.prompts.details.preview.optimize",
    "mlflow.home.demo-banner.launch",
    "mlflow.eval-datasets-v2.drawer.save",
    "mlflow.traces-v4.actions.add-to-dataset",
    "mlflow.traces-v4.actions.delete",
    "mlflow.traces-v4.save_view.save_button",
)
# The viewports: most views; the trace drawer and the Overview, which run long; the experiment
# Traces page, whose Cost column falls off a narrower screen.
VIEWPORT = (1600, 1000)
TALL = (1600, 1800)
WIDE = (2000, 1000)
# The schemes of the URLs a page may load besides the server's: inline data, objects in the
# page's memory, and blank pages.
LOCAL_SCHEMES = ("data", "blob", "about")
SERVER_SCHEMES = ("http", "https", "ws", "wss")
# The loading indicators of the 3.16.1 UI: skeletons, spinners and busy regions. A view waits
# until none shows inside the window (the charts below it load only when scrolled to).
LOADING = '.du-bois-light-skeleton, .du-bois-light-spin-spinning, [aria-busy="true"]'
# The UI's Assistant (its panel and buttons), which no view may show.
ASSISTANT = '[data-component-id^="mlflow.assistant."], [data-assistant-ui="true"]'
# The filter typed into the Evaluation runs table's search box, and that table's run names.
DEV_RUNS_FILTER = "tags.part = 'dev'"
RUN_NAMES = '[data-component-id="mlflow.eval-runs.run-name-cell"]'
# A run's traces table loads 100 traces at a time as it is scrolled: each scroll waits this long
# for the next ones.
SCROLL_WAIT_MS = 3_000
# Each set of views and the folder of its PNGs: notebook 05's 13 views of every problem, and
# notebook 10's views of the search.
VIEW_FOLDERS: dict[str, Path] = {"methods": MLFLOW_SCREENS_DIR, "gepa": MLFLOW_SCREEN_GEPA_FOLDER}
# A set's PNGs together weigh at most this much, or are quantized to 256 colours: notebook 05
# embeds its 13 in base64 and stays under 5 MB.
SCREENS_MAX_BYTES = 2_500_000
_HOST = "127.0.0.1"
_SQLITE = "sqlite:///"
_MINUTE_MS = 60_000
_PROMPT_URI = re.compile(r"prompts:/(?P<name>[^/]+)/(?P<version>\d+)")
_LOADED_COUNT = re.compile(r"^\d[\d,]* of \d[\d,]*$")
_COMPARING = re.compile(r"^Comparing version (\d+) with version (\d+)$")
# The accessible name of the prompt versions table's control that makes a row's version the
# compared one (on the right): in Compare mode, every row has one, a button of role radio.
_COMPARED_RADIO = "Select as compared version"
# What a refusal of the search's views asks for: the frozen search, replayed into the store.
_REPLAY = (
    "replay the frozen search into the store (python -m codeforces_tags.methods.typesafe_jev "
    f"log-search --method {GEPA_NAME})."
)
# True when no element the selector matches shows inside the window.
_NOTHING_SHOWN = """selector => ![...document.querySelectorAll(selector)].some(element => {
  const box = element.getBoundingClientRect();
  return box.width > 0 && box.height > 0 && box.bottom > 0 && box.right > 0
    && box.top < window.innerHeight && box.left < window.innerWidth;
})"""
# True when the run name cells are there, and all of dev runs.
_ONLY_DEV_RUNS = """selector => {
  const names = [...document.querySelectorAll(selector)].map(cell => cell.textContent);
  return names.length > 0 && names.every(name => name.endsWith("-dev"));
}"""
# Scrolls every scrolled region of the page to its bottom ("bottom") or its top.
_SCROLL = """to => {
  for (const element of document.querySelectorAll("*")) {
    const overflow = getComputedStyle(element).overflowY;
    if (element.scrollHeight > element.clientHeight && ["auto", "scroll"].includes(overflow)) {
      element.scrollTop = to === "bottom" ? element.scrollHeight : 0;
    }
  }
}"""


@dataclass(frozen=True)
class ScreenIds:
    """The ids the views open, read from the store.

    Attributes:
        experiment_id: The ``tag-methods`` experiment.
        dataset_id: The ``codeforces-dev`` evaluation dataset.
        record_id: Its record of the sample shown.
        dev_runs: Each method's one complete dev evaluation run, by method, for every method of
            the store.
        trace_id: The sample's trace in the dev run of the compared method.
        prompt_name: The prompt the compared method's run record pins.
        prompt_version: The version it pins.
        window: The dev window, ``dev_window``'s two ISO 8601 times.
    """

    experiment_id: str
    dataset_id: str
    record_id: str
    dev_runs: Mapping[str, str]
    trace_id: str
    prompt_name: str
    prompt_version: str
    window: tuple[str, str]


@dataclass(frozen=True)
class GepaScreenIds:
    """The ids the search's views open, read from the store.

    Attributes:
        experiment_id: The ``jev-gepa`` experiment.
        run_id: The search's parent run.
        tag_prompt: The prompt of ``MLFLOW_SCREEN_GEPA_TAG``'s criteria.
        tag_version: Its version that the chosen words hold (its ``chosen`` alias), never 1.
        convention_version: The convention's version that the chosen words hold: 1 when they
            keep the seed's.
    """

    experiment_id: str
    run_id: str
    tag_prompt: str
    tag_version: str
    convention_version: str


@dataclass(frozen=True)
class Screen:
    """One view of the UI, and the PNG it is captured to.

    Attributes:
        file: The PNG's name in its set's folder (``VIEW_FOLDERS``).
        route: The UI's hash route, after ``/#/``.
        viewport: The page's width and height, in CSS pixels.
        steps: The interactions taken after the page loads, from ``STEPS``.
    """

    file: str
    route: str
    viewport: tuple[int, int] = VIEWPORT
    steps: tuple[str, ...] = ()


def screen_ids(
    tracking_uri: str,
    *,
    sample_id: int = MLFLOW_SCREEN_SAMPLE_ID,
    compare: str = MLFLOW_SCREEN_COMPARE,
) -> ScreenIds:
    """Read the ids the views open from a store: over REST from a server, or from sqlite.

    The evaluation runs are those ``tracking evaluate`` tags (``run_kind=evaluation``, ``method``,
    ``part``, ``evaluation_complete``). Each must be complete, and each method may have one dev
    run at most; ``tfidf`` and ``compare`` need theirs. The prompt is the one ``compare``'s run
    record pins. It first removes every proxy variable (``PROXY``) from this process's
    environment: MLflow's REST client would send its reads of a server on 127.0.0.1 through one.

    Args:
        tracking_uri: The store, or the server that serves it. The evaluation datasets API reads
            MLflow's global tracking URI, so MLflow stays pointed at this one.
        sample_id: The dev problem of the record and trace drawers.
        compare: The zero-shot method shown, and compared with ``tfidf``.

    Returns:
        The ids.

    Raises:
        ValueError: If the store is a sqlite file that does not exist (MLflow would create it
            empty); if it has no ``tag-methods`` experiment, not one ``codeforces-dev``
            dataset, or not one record of the sample in it; if an evaluation run is incomplete,
            a method has two dev runs, or ``tfidf`` or ``compare`` none; if ``compare``'s dev run
            has not one trace of the sample, or its run record pins not one prompt; or if a test
            run meets the dev window (``dev_window``). Each message names the command that makes
            or fixes what is missing.
    """
    _drop_proxies()
    _check_exists(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_METHODS_EXPERIMENT)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(
            f"{tracking_uri} holds no {MLFLOW_METHODS_EXPERIMENT} experiment: rebuild the store "
            "from committed files (notebooks/README.md, Figures)."
        )
    experiment_id = experiment.experiment_id
    runs = client.search_runs(
        [experiment_id], filter_string=f"tags.{RUN_KIND_TAG} = '{EVALUATION_RUN_KIND}'"
    )
    dev_runs = _dev_runs(runs, compare)
    test_runs = [run.info for run in runs if run.data.tags.get("part") == "test"]
    dataset_id, record_id = _dataset_record(experiment_id, sample_id)
    pinned_name, pinned_version = _pinned_prompt(compare)
    return ScreenIds(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        record_id=record_id,
        dev_runs={method: run.info.run_id for method, run in sorted(dev_runs.items())},
        trace_id=_sample_trace(client, experiment_id, dev_runs[compare], sample_id),
        prompt_name=pinned_name,
        prompt_version=pinned_version,
        window=dev_window([run.info for run in dev_runs.values()], test_runs),
    )


def _drop_proxies() -> None:
    """Remove every proxy variable from this process's environment.

    ``requests``, MLflow's REST client, sends a request to 127.0.0.1 through ``HTTP_PROXY``,
    ``HTTPS_PROXY`` or ``ALL_PROXY`` unless ``NO_PROXY`` names it.
    """
    for name in [name for name in os.environ if PROXY.search(name)]:
        del os.environ[name]


def _check_exists(tracking_uri: str) -> None:
    """Refuse a sqlite store whose file does not exist, before MLflow creates it empty."""
    if not tracking_uri.startswith(_SQLITE):
        return
    database = Path(tracking_uri.removeprefix(_SQLITE))
    if not database.is_file():
        raise ValueError(
            f"{database} does not exist: rebuild the store from committed files "
            "(notebooks/README.md, Figures)."
        )


def _dev_runs(runs: Sequence[Run], compare: str) -> dict[str, Run]:
    """Return each method's one dev evaluation run, refusing an incomplete or missing one."""
    incomplete = [run.info.run_id for run in runs if run.data.tags.get(COMPLETE_TAG) != "true"]
    if incomplete:
        raise ValueError(
            f"The evaluation runs {incomplete} are incomplete: discard each (python -m "
            "codeforces_tags.tracking discard --run RUN_ID), then evaluate its method again. "
            "The views would show it, and its traces in the dev window."
        )
    by_method: dict[str, list[Run]] = {}
    for run in runs:
        if run.data.tags.get("part") == "dev":
            by_method.setdefault(run.data.tags["method"], []).append(run)
    repeated = sorted(method for method, found in by_method.items() if len(found) > 1)
    if repeated:
        raise ValueError(
            f"{repeated} have more than one dev evaluation run: the views show one per method. "
            "Rebuild the store from committed files (notebooks/README.md, Figures)."
        )
    missing = [method for method in (BASELINE, compare) if method not in by_method]
    if missing:
        raise ValueError(
            f"No dev evaluation run of {missing}: run python -m codeforces_tags.tracking "
            "evaluate --part dev --method NAME for each, before any test run."
        )
    return {method: found[0] for method, found in by_method.items()}


def _dataset_record(experiment_id: str, sample_id: int) -> tuple[str, str]:
    """Return the dev dataset's id and its record of the sample."""
    name = MLFLOW_DATASETS["dev"]
    found = mlflow.genai.datasets.search_datasets(
        experiment_ids=[experiment_id], filter_string=f"name = '{name}'"
    )
    if len(found) != 1:
        raise ValueError(
            f"The store holds {len(found)} evaluation datasets named {name}, not one: run "
            "python -m codeforces_tags.tracking datasets."
        )
    rows = found[0].to_df()
    records = [
        record_id
        for record_id, inputs in zip(rows["dataset_record_id"], rows["inputs"], strict=True)
        if inputs.get("sample_id") == sample_id
    ]
    if len(records) != 1:
        raise ValueError(
            f"The dataset {name} holds {len(records)} records of the problem {sample_id}, not "
            "one: MLFLOW_SCREEN_SAMPLE_ID must be a dev problem."
        )
    return found[0].dataset_id, records[0]


def _sample_trace(client: mlflow.MlflowClient, experiment_id: str, run: Run, sample_id: int) -> str:
    """Return the trace of the sample in an evaluation run: its inputs start with its id."""
    traces = client.search_traces(
        locations=[experiment_id],
        run_id=run.info.run_id,
        filter_string=f"metadata.`mlflow.traceInputs` LIKE '{{\"sample_id\": {sample_id},%'",
        include_spans=False,
    )
    if len(traces) != 1:
        raise ValueError(
            f"The evaluation run {run.info.run_id} holds {len(traces)} traces of the problem "
            f"{sample_id}, not one: rebuild the store from committed files (notebooks/README.md, "
            "Figures)."
        )
    return traces[0].info.trace_id


def _pinned_prompt(method: str) -> tuple[str, str]:
    """Return the name and version of the one prompt a method's run record pins."""
    prompts = load_run_record(method_dir(method)).prompts
    matches = [_PROMPT_URI.fullmatch(uri) for uri in prompts]
    if len(matches) != 1 or matches[0] is None:
        raise ValueError(
            f"The run record of {method} pins the prompts {list(prompts)}, not one: "
            "MLFLOW_SCREEN_COMPARE must be a zero-shot method."
        )
    return matches[0]["name"], matches[0]["version"]


def dev_window(dev_runs: Sequence[RunInfo], test_runs: Sequence[RunInfo]) -> tuple[str, str]:
    """Return the dev window: the dev evaluation runs' time, rounded out to whole minutes.

    The Overview and the Traces page show every trace of a time window, which the UI's date
    picker sets in minutes: from the first dev run's start, rounded down, to the last one's end,
    rounded up, every dev trace is in. No test run may meet that window, or its traces would show
    (ADR 0001).

    Args:
        dev_runs: The dev evaluation runs.
        test_runs: The test evaluation runs.

    Returns:
        The window's start and end, in ISO 8601 UTC (``2026-10-10T08:01:00.000Z``), as the UI
        reads them from its route.

    Raises:
        ValueError: If a dev run has no end time, or a test run meets the window.
    """
    unfinished = [run.run_id for run in dev_runs if run.end_time is None]
    if unfinished:
        raise ValueError(f"The dev evaluation runs {unfinished} have no end time: they stopped.")
    start = min(run.start_time for run in dev_runs) // _MINUTE_MS * _MINUTE_MS
    end = -(-max(run.end_time for run in dev_runs) // _MINUTE_MS) * _MINUTE_MS
    met = [
        run.run_id
        for run in test_runs
        if run.start_time <= end and (run.end_time is None or run.end_time >= start)
    ]
    if met:
        raise ValueError(
            f"The test evaluation runs {met} meet the dev window, whose views would show their "
            "traces: rebuild the store with every dev run, a 2-minute pause, then the test runs "
            "(notebooks/README.md, Figures)."
        )
    return _iso(start), _iso(end)


def _iso(milliseconds: int) -> str:
    """Return a time in milliseconds since the epoch as ISO 8601 UTC, to the millisecond."""
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def screens(
    ids: ScreenIds, *, compare: str = MLFLOW_SCREEN_COMPARE, tag: str = MLFLOW_SCREEN_TAG
) -> tuple[Screen, ...]:
    """Return the 13 views, in the order notebook 05 shows them.

    The Evaluation runs table selects ``compare``'s dev run, which is shown everywhere else, and
    is filtered to the dev runs; the runs' pages are those of dev runs; the Overview and the
    experiment's Traces page show the dev window.

    Args:
        ids: The ids read from the store (``screen_ids``).
        compare: The zero-shot method shown, and compared with ``tfidf``.
        tag: The focus tag whose wrong answers a view filters.

    Returns:
        The views.
    """
    experiment = f"experiments/{ids.experiment_id}"
    dataset = f"{experiment}/datasets/{ids.dataset_id}"
    selected = f"selectedRunUuid={ids.dev_runs[compare]}"
    traces = f"{experiment}/runs/{ids.dev_runs[compare]}/traces"
    wrong = quote(f"ASSESSMENT::=::false::correct_{tag.replace(' ', '_')}", safe="")
    window = "startTimeLabel=CUSTOM&startTime={}&endTime={}".format(*ids.window)
    baseline = f"{experiment}/runs/{ids.dev_runs[BASELINE]}/traces"
    prompt = f"prompts/{ids.prompt_name}"
    return (
        Screen("01_datasets.png", f"{experiment}/datasets"),
        Screen("02_dataset_records.png", dataset),
        Screen("03_record_drawer.png", f"{dataset}?recordId={ids.record_id}"),
        Screen(
            "04_evaluation_runs.png",
            f"{experiment}/evaluation-runs?{selected}",
            steps=("filter_dev_runs",),
        ),
        Screen(
            "05_evaluation_runs_charts.png",
            f"{experiment}/evaluation-runs?viewMode=charts&{selected}",
            steps=("filter_dev_runs",),
        ),
        Screen("06_run_traces.png", traces),
        Screen(
            "07_trace_drawer.png",
            f"{traces}?selectedEvaluationId={ids.trace_id}",
            TALL,
            ("open_assessments",),
        ),
        Screen("08_filter_false.png", f"{traces}?filter={wrong}", steps=("load_every_trace",)),
        Screen("09_compare.png", f"{baseline}?compareToRunUuid={ids.dev_runs[compare]}"),
        Screen("10_overview_usage.png", f"{experiment}/overview/usage?{window}", TALL),
        Screen("11_traces_cost.png", f"{experiment}/traces?{window}", WIDE, ("show_cost_column",)),
        Screen("12_prompt_versions.png", f"{prompt}?promptVersion={ids.prompt_version}"),
        Screen("13_prompt_compare.png", prompt, steps=("compare_prompt_versions",)),
    )


def gepa_screen_ids(tracking_uri: str, *, tag: str = MLFLOW_SCREEN_GEPA_TAG) -> GepaScreenIds:
    """Read the ids the search's views open from a store: over REST from a server, or from sqlite.

    The search is the ``jev-gepa`` experiment's one run tagged ``search`` with the method's name,
    and its ``chosen`` tag and the prompts' ``chosen`` aliases are those ``log_search`` writes.
    It first removes every proxy variable from this process's environment, as ``screen_ids``
    does.

    Args:
        tracking_uri: The store, or the server that serves it; MLflow stays pointed at it.
        tag: The focus tag whose criteria the third view compares; the chosen words must have
            changed them.

    Returns:
        The ids.

    Raises:
        ValueError: If the store is a sqlite file that does not exist; if it has no ``jev-gepa``
            experiment, not one search run, or a search run with no chosen candidate; if the
            tag's or the convention's prompt has no ``chosen`` version; or if the chosen words
            keep the tag's criteria of the seed. Each message names what makes or fixes it.
    """
    _drop_proxies()
    _check_exists(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(SEARCH_EXPERIMENT)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(f"{tracking_uri} holds no {SEARCH_EXPERIMENT} experiment: {_REPLAY}")
    run = _search_run(client, experiment.experiment_id)
    tag_prompt = prompt_name(tag)
    tag_version = _chosen_version(client, tag_prompt)
    if tag_version == "1":
        raise ValueError(
            f"The chosen words keep the seed's {tag} criteria: version 1 has nothing to be "
            "compared with. Set MLFLOW_SCREEN_GEPA_TAG to a tag whose criteria the search "
            "changed (the chosen words, tuning/params.json)."
        )
    return GepaScreenIds(
        experiment_id=experiment.experiment_id,
        run_id=run.info.run_id,
        tag_prompt=tag_prompt,
        tag_version=tag_version,
        convention_version=_chosen_version(client, prompt_name("convention")),
    )


def _search_run(client: mlflow.MlflowClient, experiment_id: str) -> Run:
    """Return the experiment's one search run, refusing one that names no chosen candidate."""
    runs = client.search_runs([experiment_id], filter_string=f"tags.{SEARCH_TAG} = '{GEPA_NAME}'")
    if len(runs) != 1:
        raise ValueError(
            f"The {SEARCH_EXPERIMENT} experiment holds {len(runs)} search runs of {GEPA_NAME}, "
            "not one: rebuild the store from committed files (notebooks/README.md, Figures)."
        )
    if CHOSEN not in runs[0].data.tags:
        raise ValueError(
            f"The search run {runs[0].info.run_id} names no chosen candidate: {_REPLAY}"
        )
    return runs[0]


def _chosen_version(client: mlflow.MlflowClient, name: str) -> str:
    """Return the number of a prompt's ``chosen`` version."""
    try:
        version = client.get_prompt_version_by_alias(name, CHOSEN)
    except MlflowException as error:
        raise ValueError(f"The prompt {name} has no {CHOSEN} version: {_REPLAY}") from error
    return str(version.version)


def gepa_screens(ids: GepaScreenIds) -> tuple[Screen, ...]:
    """Return the search's views, in the order notebook 10 shows them.

    The search run's model metrics; its candidates' runs, best selection-fold score first; then
    the prompt of the tag's criteria, and of the convention unless the chosen words keep the
    seed's, each comparing version 1 with the chosen version.

    Args:
        ids: The ids read from the store (``gepa_screen_ids``).

    Returns:
        The views: three, or four when the convention changed.
    """
    experiment = f"experiments/{ids.experiment_id}"
    candidates = quote(f"tags.`{PARENT_RUN_TAG}` = '{ids.run_id}'", safe="")
    order = quote("metrics.`selection_mean`", safe="")
    compare = ("compare_with_seed",)
    views = [
        Screen("01_gepa_search_chart.png", f"{experiment}/runs/{ids.run_id}/model-metrics", TALL),
        Screen(
            "02_gepa_candidates.png",
            f"{experiment}/runs?searchFilter={candidates}&orderByKey={order}&orderByAsc=false",
            WIDE,
        ),
        Screen(
            "03_gepa_prompt_compare.png",
            f"prompts/{ids.tag_prompt}?promptVersion={ids.tag_version}",
            steps=compare,
        ),
    ]
    if ids.convention_version != "1":
        convention = prompt_name("convention")
        views.append(
            Screen(
                "04_gepa_convention_compare.png",
                f"prompts/{convention}?promptVersion={ids.convention_version}",
                steps=compare,
            )
        )
    return tuple(views)


def is_allowed(url: str, port: int) -> bool:
    """Tell whether the browser may load a URL: inline data, or the capture's server.

    The host and port are compared exactly, after parsing: ``localhost``, another port, another
    loopback address, ``127.0.0.1.example.com`` and ``http://127.0.0.1:<port>@example.com/``
    (whose host is ``example.com``) are all refused.

    Args:
        url: A URL a page or worker asks for.
        port: The server's port on 127.0.0.1.

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


def server_command(tracking_uri: str, artifact_dir: Path, port: int) -> list[str]:
    """Return the command that serves a store's UI on 127.0.0.1, with one worker.

    Args:
        tracking_uri: The store, a ``sqlite:///`` URI.
        artifact_dir: The artifact folder of a new experiment (the store's experiments name
            their own).
        port: The port on 127.0.0.1.

    Returns:
        The command, run by this interpreter.
    """
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        tracking_uri,
        "--default-artifact-root",
        Path(artifact_dir).resolve().as_uri(),
        "--host",
        _HOST,
        "--port",
        str(port),
        "--workers",
        "1",
    ]


def check_safeguards(env: Mapping[str, str]) -> None:
    """Refuse an environment in which MLflow would load a pickle, send telemetry or run jobs.

    Each variable is read as MLflow reads a boolean: ``true`` or ``1``, ``false`` or ``0``, in
    any case; an unset one or any other value is refused.

    Args:
        env: The environment, e.g. ``os.environ`` once ``mlflow_env()`` set it.

    Raises:
        ValueError: If pickle loading, telemetry or the job runner is on (``SERVER_SAFEGUARDS``),
            e.g. telemetry when ``MLFLOW_DISABLE_TELEMETRY=false`` was set.
    """
    wrong = {
        name: env.get(name)
        for name, wanted in SERVER_SAFEGUARDS.items()
        if _BOOLEANS.get(env.get(name, "").lower()) is not wanted
    }
    if wrong:
        raise ValueError(
            f"The environment holds {wrong}, where the capture needs MLflow to read "
            f"{SERVER_SAFEGUARDS} (true or 1, false or 0): unset these variables."
        )


def server_env() -> dict[str, str]:
    """Return the server's environment: this one under ``mlflow_env()``, without credentials.

    A credential would let the server's LLM features spend money, and through a proxy its
    requests to 127.0.0.1 would leave the machine: every variable ``LEFT_OUT`` names (API keys,
    tokens, secrets, passwords, credentials, cloud keys, proxies) is left out. The MLflow
    variables are checked (``check_safeguards``), never set here (ADR 0002, safeguard 7).

    Returns:
        The environment.

    Raises:
        ValueError: If pickle loading, telemetry or the job runner is on (``SERVER_SAFEGUARDS``).
    """
    mlflow_env()
    env = {name: value for name, value in os.environ.items() if not LEFT_OUT.search(name)}
    check_safeguards(env)
    return env


def port_in_use(port: int) -> bool:
    """Tell whether something listens on a port of 127.0.0.1.

    Args:
        port: The port.

    Returns:
        True if a TCP connection to it succeeds.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex((_HOST, port)) == 0


@contextmanager
def mlflow_server(tracking_uri: str, artifact_dir: Path, port: int) -> Iterator[str]:
    """Serve a store's UI on 127.0.0.1 while the context lasts.

    The server runs in its own session, so that stopping it stops its whole process group:
    ``mlflow server`` runs uvicorn as a child process, which would otherwise keep the port.

    Args:
        tracking_uri: The store, a ``sqlite:///`` URI.
        artifact_dir: The artifact folder of a new experiment.
        port: The port on 127.0.0.1.

    Yields:
        The server's URL, ``http://127.0.0.1:<port>``.

    Raises:
        RuntimeError: If the port is in use, before anything starts (another server there would
            be screenshotted); or if the server exits, or does not answer ``/health`` within
            ``SERVER_START_S``.
    """
    if port_in_use(port):
        raise RuntimeError(
            f"{_HOST}:{port} is in use: stop what listens there, or pass another --port. "
            "Another server there would be screenshotted instead of this store."
        )
    url = f"http://{_HOST}:{port}"
    command = server_command(tracking_uri, artifact_dir, port)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command, env=server_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _wait_until_served(process, url, log)
            yield url
        finally:
            _stop(process)


def _wait_until_served(process: subprocess.Popen, url: str, log: IO[bytes]) -> None:
    """Wait until the server answers /health, refusing one that exits or never answers.

    The process is checked after each answer: if it exited, another server answered.
    """
    deadline = time.monotonic() + SERVER_START_S
    while time.monotonic() < deadline:
        healthy = _healthy(url)
        if process.poll() is not None:
            raise RuntimeError(f"mlflow server exited with {process.returncode}:\n{_tail(log)}")
        if healthy:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"mlflow server did not answer {url}/health in {SERVER_START_S} s:\n{_tail(log)}"
    )


def _healthy(url: str) -> bool:
    """Tell whether a server answers GET /health with 200, through no proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/health", timeout=2) as response:
            return response.status == 200
    except OSError:  # refused, reset or timed out
        return False


def _tail(log: IO[bytes], lines: int = 20) -> str:
    """Return the last lines of the server's log."""
    log.seek(0)
    return "\n".join(log.read().decode(errors="replace").splitlines()[-lines:])


def _stop(process: subprocess.Popen) -> None:
    """Stop the server's process group: SIGTERM, then SIGKILL to what is left of it."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signum)
        deadline = time.monotonic() + SERVER_STOP_S
        while _group_alive(process) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not _group_alive(process):
            return


def _group_alive(process: subprocess.Popen) -> bool:
    """Tell whether a process of the server's group is left, after reaping the server itself."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def guarded_context(browser: Browser, port: int) -> tuple[BrowserContext, list[str]]:
    """Open a browser context that loads nothing but the capture's server.

    The context is light, in US English and UTC, so that the PNGs' dates and counts do not follow
    the machine's locale; it blocks service workers, whose requests no route sees, and its init
    script removes shared workers, whose requests neither a route nor the context's request
    events see (``INIT_SCRIPT``). Every request of a page or a dedicated worker to another address
    than ``127.0.0.1:<port>`` (``is_allowed``) is aborted and recorded. Such a WebSocket is
    recorded and never connected to its server: Playwright holds a routed WebSocket in the browser
    until its handler connects it (closing it from the handler would hang the sync API). A request
    the routes do not see, such as the target of a redirect the server answers, is only recorded,
    once it was sent: the capture then exits with code 2, and the run inside a network namespace
    that holds only loopback is the backstop.

    Args:
        browser: The browser.
        port: The server's port on 127.0.0.1.

    Returns:
        The context, and the list it fills with every URL that left the server, each once.
    """
    context = browser.new_context(
        color_scheme="light",
        device_scale_factor=1,
        locale="en-US",
        timezone_id="UTC",
        service_workers="block",
    )
    context.add_init_script(script=INIT_SCRIPT)
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


def capture(views: Sequence[Screen], base_url: str, folder: Path) -> list[str]:
    """Screenshot each view of the UI served at ``base_url`` into ``folder``.

    Each view opens in a new page of one guarded context (``guarded_context``): the page waits
    ``WAIT_MS``, then until no loading indicator shows, hides the Assistant's buttons, takes the
    view's steps, checks that no Assistant panel or "Got it" popover shows, and is screenshotted.
    The capture stops at the first view that sent a request outside the server, and reports that
    request rather than a failure that follows it: a page that needs an aborted script or font
    times out. Once the browser is closed, the PNGs are fitted to ``SCREENS_MAX_BYTES``
    (``fit_budget``), unless a page sent such a request as it closed.

    Args:
        views: The views (``screens``).
        base_url: The server's URL.
        folder: Where the PNGs go, named as the views say.

    Returns:
        The requests aborted because they left the server, up to the browser's close; an empty
        list when none did.

    Raises:
        Exception: What a view raised (a timeout, a missing control), when no request had left
            the server.
    """
    from playwright.sync_api import Error, sync_playwright  # CI has no playwright

    port = urlsplit(base_url).port
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome")
        except Error:  # no system Chrome
            browser = playwright.chromium.launch()
        try:
            context, outside = guarded_context(browser, port)
            requests: list[str] = []
            context.on("request", lambda request: requests.append(request.url))
            _shoot_views(context, views, base_url, folder, outside)
        finally:
            browser.close()
    if outside:  # sent by a view, or by a page as it closed
        return list(outside)
    served = sum(urlsplit(url).scheme in SERVER_SCHEMES for url in requests)
    print(f"{served} requests, all to {_HOST}:{port}, and {len(requests) - served} inline")
    pngs = [folder / view.file for view in views]
    shot = sum(png.stat().st_size for png in pngs)
    fitted = fit_budget(pngs, SCREENS_MAX_BYTES)
    print(f"The PNGs weigh {shot:,} bytes, {fitted:,} once fitted to {SCREENS_MAX_BYTES:,}")
    return []


def _shoot_views(
    context: BrowserContext,
    views: Sequence[Screen],
    base_url: str,
    folder: Path,
    outside: Sequence[str],
) -> None:
    """Shoot each view, up to the first that sent a request outside the server.

    A view that fails once such a request was aborted is reported by that request (``capture``
    returns it), and its failure is printed: the page may have waited for what was aborted.
    """
    try:
        for view in views:
            _shoot(context, view, f"{base_url}/#/{view.route}", folder / view.file)
            if outside:
                return
    except Exception as error:
        if outside:
            print(f"A view failed once a request had left the server: {error}", file=sys.stderr)
            return
        raise


def _shoot(context: BrowserContext, view: Screen, url: str, png: Path) -> None:
    """Open a view in a new page, take its steps and screenshot it."""
    page = context.new_page()
    try:
        page.set_viewport_size({"width": view.viewport[0], "height": view.viewport[1]})
        page.goto(url)
        page.wait_for_timeout(WAIT_MS)
        _wait_loaded(page)
        page.add_style_tag(content=HIDDEN)
        for step in view.steps:
            STEP_ACTIONS[step](page)
            _wait_loaded(page)
        _check_no_assistant(page)
        page.mouse.move(0, 0)
        page.evaluate("() => document.activeElement && document.activeElement.blur()")
        page.screenshot(path=png)
    finally:
        page.close()


def _wait_loaded(page: Page) -> None:
    """Wait until no loading indicator shows in the window, LOAD_TIMEOUT_MS at most."""
    page.wait_for_function(_NOTHING_SHOWN, arg=LOADING, timeout=LOAD_TIMEOUT_MS)


def _check_no_assistant(page: Page) -> None:
    """Refuse a page that shows the Assistant or the Detect Issues "Got it" popover."""
    shown = page.locator(ASSISTANT).filter(visible=True).count()
    got_it = page.get_by_role("button", name="Got it").filter(visible=True).count()
    if shown or got_it:
        raise RuntimeError(
            f"{page.url} shows {shown} Assistant elements and {got_it} Got it popovers: "
            "the UI_PRESETS keys no longer dismiss them."
        )


def _control(page: Page, component_id: str) -> Locator:
    """Return the control of a component id the steps may click (``CLICKS``)."""
    if component_id not in CLICKS:
        raise ValueError(f"{component_id} is not in CLICKS: the steps never click it.")
    return page.locator(f'[data-component-id="{component_id}"]')


def _set_checked(item: Locator, checked: bool) -> None:
    """Click a menu checkbox unless it is already as wanted; refuse one that does not follow."""
    state = "true" if checked else "false"
    if item.get_attribute("aria-checked") != state:
        item.click()
    item.and_(item.page.locator(f'[aria-checked="{state}"]')).wait_for(timeout=LOAD_TIMEOUT_MS)


def _option(group: Locator, value: str) -> Locator:
    """Return the option of a segmented control whose input has the value."""
    return group.locator("label", has=group.page.locator(f'input[value="{value}"]'))


def _filter_dev_runs(page: Page) -> None:
    """Filter the Evaluation runs table to the dev runs, and check that it lists every one."""
    names = page.locator(RUN_NAMES)
    before = names.all_inner_texts()
    search = page.locator('input[type="search"]')
    search.fill(DEV_RUNS_FILTER)
    search.press("Enter")
    page.wait_for_function(_ONLY_DEV_RUNS, arg=RUN_NAMES, timeout=LOAD_TIMEOUT_MS)
    check_dev_runs(before, names.all_inner_texts())


def check_dev_runs(before: Sequence[str], after: Sequence[str]) -> None:
    """Check that a filtered Evaluation runs table lists the dev runs of the unfiltered one.

    The Run Name column truncates the names in the PNG; the DOM holds them whole.

    Args:
        before: The run names of the table before the filter (every evaluation run).
        after: Its run names after it.

    Raises:
        RuntimeError: Unless ``after`` holds the names of ``before`` that end in ``-dev``, and
            there is one at least.
    """
    expected = sorted(name for name in before if name.endswith("-dev"))
    if not expected or sorted(after) != expected:
        raise RuntimeError(
            f"The filtered Evaluation runs table lists {list(after)}, not the dev runs "
            f"{expected} of the unfiltered one."
        )


def _open_assessments(page: Page) -> None:
    """Open the trace drawer's assessments pane."""
    _control(page, "shared.model-trace-explorer.assessments-pane-toggle").click()
    pane = '[data-component-id="shared.model-trace-explorer.close-assessments-pane"]'
    page.locator(pane).wait_for(timeout=LOAD_TIMEOUT_MS)


def _load_every_trace(page: Page) -> None:
    """Scroll a run's traces table until it loaded every trace it lists, then back to its top.

    Its count (``<loaded> of <traces of the run>``) then shows how many traces the filter keeps.
    """
    count = page.get_by_text(_LOADED_COUNT)
    shown = ""
    while shown != count.inner_text():
        shown = count.inner_text()
        page.evaluate(_SCROLL, "bottom")
        page.wait_for_timeout(SCROLL_WAIT_MS)
    page.evaluate(_SCROLL, "top")


def _show_cost_column(page: Page) -> None:
    """Show the experiment traces table's Run name and Cost columns, and hide its assessments."""
    _control(page, "mlflow.traces-v4.display.trigger").click()
    page.get_by_role("menuitem", name="Columns", exact=True).click()
    _set_checked(_control(page, "mlflow.traces-v4.column-selector.item.run_name"), True)
    _set_checked(_control(page, "mlflow.traces-v4.column-selector.item.cost"), True)
    assessments = _control(page, "mlflow.traces-v4.display.columns.group-toggle-all")
    _set_checked(assessments.filter(has_text="Assessments"), False)
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")
    headers = page.locator('[data-component-id="web-shared.traces-table.header"]')
    if not {"Run name", "Cost"} <= set(headers.all_inner_texts()):
        raise RuntimeError(f"The traces table shows the columns {headers.all_inner_texts()}.")


def _compare_prompt_versions(page: Page) -> None:
    """Compare the prompt's versions, the older on the left, as text."""
    _option(_control(page, "mlflow.prompts.details.mode"), "compare").click()
    left, right = _COMPARING.fullmatch(page.get_by_text(_COMPARING).inner_text()).groups()
    _control(page, "mlflow.prompts.details.switch_sides").click()
    page.get_by_text(f"Comparing version {right} with version {left}", exact=True).wait_for()
    if int(right) > int(left):
        raise RuntimeError(f"The switch of sides compares version {right} with version {left}.")
    text = _option(_control(page, "mlflow.prompts.compare.toggle-markdown-rendering"), "false")
    text.click()
    text.locator("input:checked").wait_for(state="attached", timeout=LOAD_TIMEOUT_MS)


def _compare_with_seed(page: Page) -> None:
    """Compare version 1, the seed's words, on the left with the version the route selects.

    The route selects the chosen version, which Compare mode puts on the left; the last row's
    "Select as compared version" radio puts version 1 (the table lists the newest first) on the
    right, where it already is when the prompt has two versions, and the switch of sides puts it
    on the left. The text is shown as text.
    """
    _option(_control(page, "mlflow.prompts.details.mode"), "compare").click()
    chosen, _ = _COMPARING.fullmatch(page.get_by_text(_COMPARING).inner_text()).groups()
    if chosen == "1":
        raise RuntimeError(f"{page.url} selects version 1: it has nothing to be compared with.")
    page.get_by_role("radio", name=_COMPARED_RADIO, exact=True).last.click()
    page.get_by_text(f"Comparing version {chosen} with version 1", exact=True).wait_for()
    _control(page, "mlflow.prompts.details.switch_sides").click()
    page.get_by_text(f"Comparing version 1 with version {chosen}", exact=True).wait_for()
    text = _option(_control(page, "mlflow.prompts.compare.toggle-markdown-rendering"), "false")
    text.click()
    text.locator("input:checked").wait_for(state="attached", timeout=LOAD_TIMEOUT_MS)


# What each step of STEPS does to a page.
STEP_ACTIONS: dict[str, Callable[[Page], None]] = {
    "filter_dev_runs": _filter_dev_runs,
    "open_assessments": _open_assessments,
    "load_every_trace": _load_every_trace,
    "show_cost_column": _show_cost_column,
    "compare_prompt_versions": _compare_prompt_versions,
    "compare_with_seed": _compare_with_seed,
}


def fit_budget(paths: Sequence[Path], budget: int = SCREENS_MAX_BYTES) -> int:
    """Quantize PNGs to a 256-colour palette if together they weigh more than the budget.

    Args:
        paths: The PNGs.
        budget: The most they may weigh together, in bytes.

    Returns:
        What they weigh together afterwards, in bytes.

    Raises:
        ValueError: If they weigh more than the budget even quantized.
    """
    total = sum(path.stat().st_size for path in paths)
    if total <= budget:
        return total
    for path in paths:
        with Image.open(path) as image:
            palette = image.convert("RGB").quantize(colors=256, dither=Image.Dither.NONE)
        palette.save(path, optimize=True)
    total = sum(path.stat().st_size for path in paths)
    if total > budget:
        raise ValueError(
            f"The PNGs weigh {total:,} bytes even quantized, over the budget of {budget:,} "
            "bytes: notebook 05 would pass 5 MB."
        )
    return total


def main(
    argv: Sequence[str] | None = None,
    *,
    server: Callable[[str, Path, int], AbstractContextManager[str]] = mlflow_server,
    shoot: Callable[[Sequence[Screen], str, Path], list[str]] = capture,
) -> int:
    """Capture the views of a store's UI into ``--out``, all of them or none.

    MLflow's environment is set first, as every tracking command sets it (``mlflow_env``: no
    pickle loaded, no model catalog fetched), then checked before any MLflow call
    (``check_safeguards``): with telemetry on, the script's first read of the store would start
    MLflow's telemetry client, which fetches its config from outside the machine. The ids of the
    set of views that ``--set`` names (``methods``, notebook 05's, by default; ``gepa``, the
    search's) are read from the store, which must exist, then from the server, which must serve
    that store: another server on the port would answer other ids. The PNGs are written to a
    temporary folder beside ``--out`` (by default the set's folder, ``VIEW_FOLDERS``), and replace
    those in it only when every view was captured and no request left the server.

    Args:
        argv: The command line arguments; ``sys.argv[1:]`` if None.
        server: A context manager like ``mlflow_server``, which the tests replace.
        shoot: A function like ``capture``, which the tests replace.

    Returns:
        0 when every PNG was replaced; 1 when another store answered or a view was not
        captured; ``OUTSIDE_REQUEST_EXIT`` when a request left the server.

    Raises:
        ValueError: If pickle loading, telemetry or the job runner is on (``check_safeguards``),
            before the store is read and the server started.
    """
    args = _parser().parse_args(argv)
    out = args.out or VIEW_FOLDERS[args.set]
    mlflow_env()
    check_safeguards(os.environ)
    expected = _read_ids(args.set, args.tracking_uri)
    with server(args.tracking_uri, args.artifact_dir, args.port) as base_url:
        ids = _read_ids(args.set, base_url)
        if _identity(ids) != _identity(expected):
            print(
                f"{base_url} serves another store than {args.tracking_uri}: it answers "
                f"{_identity(ids)}.",
                file=sys.stderr,
            )
            return 1
        views = gepa_screens(ids) if isinstance(ids, GepaScreenIds) else screens(ids)
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as folder:
            outside = shoot(views, base_url, Path(folder))
            if outside:
                print(
                    "Requests left the server; no PNG was written:",
                    *outside,
                    sep="\n",
                    file=sys.stderr,
                )
                return OUTSIDE_REQUEST_EXIT
            return _replace_pngs(views, Path(folder), out)


def _read_ids(view_set: str, tracking_uri: str) -> ScreenIds | GepaScreenIds:
    """Read the ids of a set of views (``VIEW_FOLDERS``) from a store or its server."""
    return gepa_screen_ids(tracking_uri) if view_set == "gepa" else screen_ids(tracking_uri)


def _identity(ids: ScreenIds | GepaScreenIds) -> tuple[object, ...]:
    """Return what two reads of one store share: its experiment, and its runs the views open."""
    if isinstance(ids, GepaScreenIds):
        return ids.experiment_id, ids.run_id
    return ids.experiment_id, dict(ids.dev_runs)


def _replace_pngs(views: Sequence[Screen], folder: Path, out: Path) -> int:
    """Move the PNGs into ``out`` if ``folder`` holds the views' files only; return the status."""
    written = sorted(path.name for path in folder.iterdir())
    expected = sorted(view.file for view in views)
    if written != expected:
        print(f"The capture wrote {written}, not {expected}: no PNG was replaced.", file=sys.stderr)
        return 1
    for view in views:
        os.replace(folder / view.file, out / view.file)
    print(f"Wrote {len(views)} PNGs to {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_store_arguments(parser)
    parser.add_argument(
        "--port", type=int, default=MLFLOW_UI_PORT, help="the server's port on 127.0.0.1"
    )
    parser.add_argument(
        "--set",
        choices=tuple(VIEW_FOLDERS),
        default="methods",
        help="the views: notebook 05's (methods) or the search's, of notebook 10 (gepa)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="the PNGs' folder (default: the set's own)"
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
