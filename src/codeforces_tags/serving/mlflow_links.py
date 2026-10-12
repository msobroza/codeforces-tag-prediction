"""Links from a served tagging method to what MLflow tracked about it (docs/adr/0012).

The Streamlit app links the served method to the ``tag-methods`` experiment, to the method's run
(its dev out-of-fold metrics and frozen latencies, the calibration of ``tfidf``, and its logged
model, which holds the run record's parameters and the test metrics) and to the prompt versions
of a prompted or Jev method, in the UI of a local ``mlflow server`` on the project's store
(``python -m codeforces_tags.serving mlflow-ui``).

This module imports no mlflow: it reads MLflow's REST API with the standard library, through no
proxy, since the UI is a server of this machine. The experiment's id comes from
``experiments/get-by-name`` and the method run from ``runs/search`` (the run ``log_method``
creates: tagged ``run_kind=method`` and named after the method). The prompt versions are first
those the committed ``run_record.json`` pins, read with no request, then the others linked to
the method run in the store, which MLflow lists in the run's ``mlflow.linkedPrompts`` tag: the
run records of ``jev-statement`` and ``jev-statement-code`` pin no prompt, and their ``freeze``
links the version of ``jev-tags`` they read to their method run. A store not yet rebuilt from
committed files lacks the experiment or the run: their links are None, never an error, and only
the pinned versions are linked.

``mlflow_ui_command`` and ``mlflow_ui_environment`` start that UI: on 127.0.0.1 with one worker,
under ``mlflow_env()``, and without any variable ``config.MLFLOW_SERVER_LEFT_OUT`` names
(credentials, proxies), the rule of the screenshot tool ``notebooks/figures/mlflow_ui.py`` too.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote, urlencode

from codeforces_tags.config import (
    METHODS_ASSETS_DIR,
    MLFLOW_METHODS_EXPERIMENT,
    MLFLOW_SERVER_LEFT_OUT,
    MLFLOW_TRACKING_URI,
    SERVING_MLFLOW_UI_URL,
    SERVING_MLFLOW_UI_URL_ENV_VAR,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE, load_run_record
from codeforces_tags.methods import method_dir
from codeforces_tags.serving.client import ServingError
from codeforces_tags.tracking.environment import mlflow_env

# The tag that log_method puts on a method run, with the value "method":
# tracking.methods.RUN_KIND_TAG, which this module cannot import (tracking.methods imports
# mlflow). A test pins the two equal.
RUN_KIND_TAG = "run_kind"
# The tag in which MLflow lists the prompt versions linked to a run, as the JSON text of a list of
# {"name": ..., "version": ...}: tracking.prompts.LINKED_PROMPTS_TAG, which this module cannot
# import either (tracking.prompts imports mlflow). A test pins the two equal.
LINKED_PROMPTS_TAG = "mlflow.linkedPrompts"
# The command that starts the UI the links open: the app shows it when the UI does not answer.
UI_START_COMMAND = "uv run python -m codeforces_tags.serving mlflow-ui"
_HOST = "127.0.0.1"
_EXPERIMENT_ROUTE = "api/2.0/mlflow/experiments/get-by-name"
_RUNS_ROUTE = "api/2.0/mlflow/runs/search"
# A pinned prompt version of a run record.
_PROMPT_URI = re.compile(r"prompts:/(?P<name>[^/]+)/(?P<version>\d+)")
_LEFT_OUT = re.compile(MLFLOW_SERVER_LEFT_OUT)
# What a request to the UI raises when it gets no answer: a refused, reset or timed-out connection
# (OSError), a peer that does not speak HTTP (HTTPException), or a URL that is not one
# (ValueError, which urllib raises before it opens a socket).
_NO_ANSWER = (OSError, http.client.HTTPException, ValueError)
# Requests to the UI go through no proxy: it is a server of this machine.
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class MlflowLinks:
    """The pages of an MLflow UI that show what MLflow tracked about one tagging method.

    Each link is an absolute URL of one of the UI's hash routes.

    Attributes:
        experiment: The ``tag-methods`` experiment, ``<ui>/#/experiments/<id>``; None when the
            store has no such experiment.
        run: The method's run, ``<ui>/#/experiments/<id>/runs/<run id>``; None when the store
            has no method run of that name (or no experiment).
        prompts: Each prompt version of the method, as ``(label, link)``, e.g.
            ``("jev-tags v1", "<ui>/#/prompts/jev-tags?promptVersion=1")``: first those its
            committed run record pins, in the record's order, then the others linked to its run
            in the store, in the order MLflow lists them, each version once; empty for a method
            with neither.
        registry: The prompt registry, ``<ui>/#/prompts``.
    """

    experiment: str | None
    run: str | None
    prompts: tuple[tuple[str, str], ...]
    registry: str


def mlflow_ui_url(url: str | None = None) -> str:
    """Return the URL of the MLflow UI that the links open.

    Args:
        url: The UI's URL, e.g. an option of the app; None or empty to read the environment.

    Returns:
        ``url`` when given, else ``$CODEFORCES_MLFLOW_UI_URL`` when it is set and not empty
        (read at call time), else ``config.SERVING_MLFLOW_UI_URL``.
    """
    return url or os.environ.get(SERVING_MLFLOW_UI_URL_ENV_VAR) or SERVING_MLFLOW_UI_URL


def method_links(
    ui_url: str, method: str, *, assets_dir: Path = METHODS_ASSETS_DIR, timeout: float = 5.0
) -> MlflowLinks:
    """Return the links of a tagging method to its experiment, run and prompts in an MLflow UI.

    Args:
        ui_url: The UI, e.g. ``mlflow_ui_url()``.
        method: The method's registry key, e.g. the ``method`` of a served prediction.
        assets_dir: The folder of the frozen methods: the committed
            ``<assets_dir>/<method>/run_record.json`` names the prompt versions the method pins.
            A method without one pins none.
        timeout: Seconds to wait for each of the UI's answers.

    Returns:
        The links; the experiment's and the run's are None when the store lacks them.

    Raises:
        ValueError: If ``method`` is not a method name (it goes into a path and a search
            filter), before any request; or if the run record pins a prompt that is not a
            version, ``prompts:/<name>/<version>``.
        ServingError: If the UI does not answer, or ``ui_url`` is not a URL (status None); if it
            answers an error other than a missing experiment (its status, with MLflow's
            message); or if it answers a body that is not MLflow's JSON (status 200).
    """
    method_dir(method)  # refuses a name that is not a method name, e.g. one holding a quote
    base = ui_url.rstrip("/")
    pinned = _pinned_versions(Path(assets_dir) / method)
    registry = f"{base}/#/prompts"
    experiment_id = _experiment_id(base, timeout)
    if experiment_id is None:
        return MlflowLinks(None, None, _prompt_links(base, pinned), registry)
    experiment = f"{base}/#/experiments/{experiment_id}"
    run_id, linked = _method_run(base, experiment_id, method, timeout)
    run = None if run_id is None else f"{experiment}/runs/{run_id}"
    return MlflowLinks(experiment, run, _prompt_links(base, [*pinned, *linked]), registry)


def ui_alive(ui_url: str, timeout: float = 2.0) -> bool:
    """Tell whether an MLflow UI answers ``GET /health`` with 200, through no proxy.

    Args:
        ui_url: The UI, e.g. ``mlflow_ui_url()``.
        timeout: Seconds to wait for its answer.

    Returns:
        True if it answered 200; False for any other answer, a refused connection, a timeout
        or a URL that is not one.
    """
    try:
        with _NO_PROXY.open(f"{ui_url.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == HTTPStatus.OK
    except _NO_ANSWER:  # refused, reset, timed out, an error status, a URL that is not one
        return False


def mlflow_ui_command(port: int) -> list[str]:
    """Return the command that serves the project's MLflow store on 127.0.0.1, with one worker.

    The store's experiments keep their ``file://`` artifact locations, which a server on this
    machine reads.

    Args:
        port: The port on 127.0.0.1.

    Returns:
        ``mlflow server`` on ``config.MLFLOW_TRACKING_URI``, run by this interpreter.
    """
    serve = [sys.executable, "-m", "mlflow", "server", "--backend-store-uri", MLFLOW_TRACKING_URI]
    return [*serve, "--host", _HOST, "--port", str(port), "--workers", "1"]


def mlflow_ui_environment() -> dict[str, str]:
    """Return the environment of ``mlflow_ui_command``: this one under ``mlflow_env()``.

    ``mlflow_env()`` sets MLflow's variables first (no pickle loaded, no telemetry, no hint to
    coding agents, no job runner); every variable that ``config.MLFLOW_SERVER_LEFT_OUT`` names
    (API keys, tokens, secrets, passwords, credentials, cloud keys, proxies) is then left out,
    as the screenshot tool leaves them out of its own server's. This process keeps them.

    Returns:
        The environment.
    """
    mlflow_env()
    return {name: value for name, value in os.environ.items() if not _LEFT_OUT.search(name)}


def _pinned_versions(folder: Path) -> list[tuple[str, str]]:
    """Return the name and version of each prompt version a method's run record pins."""
    if not (folder / RUN_RECORD_FILE).is_file():
        return []
    versions = []
    for uri in load_run_record(folder).prompts:
        pinned = _PROMPT_URI.fullmatch(uri)
        if pinned is None:
            raise ValueError(
                f"{folder / RUN_RECORD_FILE} pins {uri!r}, which is not a prompt version "
                "(prompts:/<name>/<version>)."
            )
        versions.append((pinned["name"], pinned["version"]))
    return versions


def _prompt_links(base: str, versions: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Return the label and link of each prompt version, in order, each version once."""
    links = []
    for name, version in dict.fromkeys(versions):
        route = f"prompts/{quote(name, safe='')}?promptVersion={quote(version, safe='')}"
        links.append((f"{name} v{version}", f"{base}/#/{route}"))
    return tuple(links)


def _experiment_id(base: str, timeout: float) -> str | None:
    """Return the id of the ``tag-methods`` experiment, or None when the store has none."""
    query = urlencode({"experiment_name": MLFLOW_METHODS_EXPERIMENT})
    try:
        answer = _rest(base, f"{_EXPERIMENT_ROUTE}?{query}", None, timeout)
    except ServingError as error:
        if error.status == HTTPStatus.NOT_FOUND:  # MLflow's RESOURCE_DOES_NOT_EXIST
            return None
        raise
    try:
        return str(answer["experiment"]["experiment_id"])
    except (KeyError, TypeError):
        raise _unexpected(base, _EXPERIMENT_ROUTE) from None


def _method_run(
    base: str, experiment_id: str, method: str, timeout: float
) -> tuple[str | None, list[tuple[str, str]]]:
    """Return the id of the method's run and the prompt versions linked to it, or None and none."""
    search = {
        "experiment_ids": [experiment_id],
        "filter": f"tags.{RUN_KIND_TAG} = 'method' AND attributes.run_name = '{method}'",
        "max_results": 1,
    }
    answer = _rest(base, _RUNS_ROUTE, search, timeout)
    try:
        runs = answer.get("runs", [])  # MLflow answers {} when no run matches
        if not runs:
            return None, []
        tags = {tag["key"]: tag["value"] for tag in runs[0].get("data", {}).get("tags", [])}
        linked = json.loads(tags.get(LINKED_PROMPTS_TAG, "[]"))
        versions = [(str(entry["name"]), str(entry["version"])) for entry in linked]
        return str(runs[0]["info"]["run_id"]), versions
    except (AttributeError, KeyError, TypeError, ValueError):  # ValueError: a tag not JSON
        raise _unexpected(base, _RUNS_ROUTE) from None


def _rest(base: str, route: str, payload: Mapping[str, object] | None, timeout: float) -> object:
    """Send one request to MLflow's REST API, a POST of ``payload`` or a GET, and read its JSON."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    try:
        request = urllib.request.Request(
            f"{base}/{route}", data=data, headers={"Content-Type": "application/json"}
        )
        with _NO_PROXY.open(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise ServingError(error.code, _error_message(error)) from None
    except _NO_ANSWER as error:  # refused, reset, timed out, or a URL that is not one
        reason = getattr(error, "reason", error)
        raise ServingError(
            None,
            f"the MLflow UI at {base} does not answer ({reason}): start it with {UI_START_COMMAND}",
        ) from None
    try:
        return json.loads(body)
    except ValueError:  # not JSON, or not UTF-8
        raise ServingError(
            HTTPStatus.OK.value, f"the MLflow UI at {base} answered {route} with no JSON"
        ) from None


def _unexpected(base: str, route: str) -> ServingError:
    """Return the error of an answer whose JSON is not what MLflow answers."""
    return ServingError(
        HTTPStatus.OK.value,
        f"{base} answered {route} with JSON that MLflow does not write: is it an MLflow UI?",
    )


def _error_message(error: urllib.error.HTTPError) -> str:
    """Return MLflow's message of an error answer, else its status line."""
    with error:
        body = error.read()
    try:
        return str(json.loads(body)["message"])
    except (ValueError, KeyError, TypeError):  # not JSON, or JSON without a message
        return f"{error.code} {error.reason}"
