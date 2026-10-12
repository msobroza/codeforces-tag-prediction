"""The HTTP client of a model server, for the Streamlit app, ``serving smoke`` and notebook 12.

A model server is MLflow's scoring server serving one packaged tagging method (docs/adr/0012).
``POST /invocations`` answers each problem's focus tags, with each tag's score and per-tag
threshold for a score method, and each answer's cost and failure for a method that calls an API;
``GET /ping`` answers 200 once the method has loaded (nginx answers 502 until then).

``request_payload`` writes problems as MLflow's ``dataframe_split``: every raw input column, JSON
null for a missing or blank optional field (never ``"nan"`` or ``""``: an empty note changes what
an encoder reads), the sample inputs and outputs as the JSON text of a list, and every other text
as typed, character for character. ``predict`` posts it and reads the answer. Every failure (a
refused connection, a timeout, a peer that does not answer HTTP, an answer other than 200, a body
it cannot read) is one ``ServingError`` with the HTTP status and a one-line message, never a
traceback.

Its HTTP goes through the standard library only (``urllib.request``): the client needs no
dependency beyond the project's, and imports neither mlflow nor torch. It never goes through a
proxy: the model server runs on this machine (compose, ``serving.local``) or behind a
``kubectl port-forward``, and a proxy of the environment would otherwise receive the problems'
statements and solutions.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from codeforces_tags.cli import check_fields
from codeforces_tags.config import (
    CLI_JSON_LIST_FIELDS,
    FOCUS_TAGS,
    RAW_INPUT_COLUMNS,
    SERVING_DEFAULT_URL,
    SERVING_TIMEOUT_S,
    SERVING_URL_ENV_VAR,
)
from codeforces_tags.serving.records import (
    COST_COLUMN,
    FAILED_COLUMN,
    METHOD_COLUMN,
    score_column,
    threshold_column,
)

# The two paths of a model server that the client calls.
_PING_PATH = "/ping"
_INVOCATIONS_PATH = "/invocations"
# What a request that got no answer raises: a refused, reset or closed connection and a timeout
# (OSError, urllib's URLError among them); a peer that does not answer HTTP (BadStatusLine) and a
# URL that http.client cannot use (InvalidURL), both HTTPExceptions; a URL that urllib cannot
# parse ("Invalid IPv6 URL") and a body that is not JSON, both ValueErrors.
_NO_ANSWER = (OSError, http.client.HTTPException, ValueError)
# MLflow's scoring server writes the input frame into the message of an error the method raised
# ("Failed to predict data '<frame>'. \nError: <message>", status 500): the method's own message
# is what the client keeps.
_PREDICT_FAILURE = re.compile(r"Failed to predict data '.*'\. \nError: (.*)", flags=re.DOTALL)


class ServingError(Exception):
    """A model server that gave no answer, an error, or an answer the client cannot read.

    Attributes:
        status: The answer's HTTP status (200 for a body that cannot be read); None when no
            answer came: a refused or closed connection, a timeout, a peer that does not answer
            HTTP, a URL that is not one.
        message: One line that says what happened, for the app or a command to show.
    """

    def __init__(self, status: int | None, message: str) -> None:
        """Keep the status and the message.

        Args:
            status: The answer's HTTP status, or None when no answer came.
            message: One line that says what happened.
        """
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ServedPrediction:
    """What a model server answered about one problem.

    Attributes:
        method: The served tagging method's name.
        tags: The predicted focus tags, in ``FOCUS_TAGS`` order.
        scores: Each focus tag's score, for a score method; empty for any other method.
        thresholds: Each focus tag's per-tag threshold (``inf`` for a tag the method never
            predicts), for a score method; empty for any other method.
        failed: For a method that calls an API, whether its answer failed (the problem then
            gets no tag); None for a local method.
        cost_usd: For a method that calls an API, what its request cost; None for a local
            method.
        seconds: The round trip of the request that asked about the problem, the same for every
            problem of one request.
    """

    method: str
    tags: tuple[str, ...]
    scores: dict[str, float]
    thresholds: dict[str, float]
    failed: bool | None
    cost_usd: float | None
    seconds: float


def server_url(url: str | None = None) -> str:
    """Say which model server to call.

    Args:
        url: The model server's URL, e.g. ``http://127.0.0.1:8080``; None or empty for the
            others.

    Returns:
        ``url`` when given, else ``$CODEFORCES_SERVING_URL`` read now (unset or empty counts as
        unset), else ``SERVING_DEFAULT_URL``.
    """
    if url:
        return url
    return os.environ.get(SERVING_URL_ENV_VAR) or SERVING_DEFAULT_URL


def request_payload(problems: pd.DataFrame, columns: Sequence[str] = RAW_INPUT_COLUMNS) -> dict:
    """Write problems as the body of a request to ``/invocations``: MLflow's ``dataframe_split``.

    Args:
        problems: Raw problems indexed by ``sample_id``, as ``read_samples()`` or
            ``problems.problem_from_form`` gives them; other columns (tags, fold, group) are left
            out.
        columns: The raw input columns to send, in order; one that ``problems`` lacks is sent
            as null.

    Returns:
        ``{"dataframe_split": {"columns": [...], "data": [[...], ...]}}``, one row per problem in
        the order of ``problems``: None (JSON null) for a missing (NaN or None) or blank optional
        text; the sample inputs and outputs as the JSON text of a list (a list is written with
        ``json.dumps``, a JSON text is kept as it is); every other text as it is, character for
        character.

    Raises:
        CliError: As ``cli.check_fields``: a problem without a description or a solution, or
            with a field that no method can read; the message names the problems.
        ValueError: If a column is not one of ``RAW_INPUT_COLUMNS``.
    """
    unknown = [column for column in columns if column not in RAW_INPUT_COLUMNS]
    if unknown:
        raise ValueError(f"{unknown} are not raw input columns: send some of {RAW_INPUT_COLUMNS}.")
    check_fields(problems)
    frame = problems.reindex(columns=list(columns))
    data = [
        [_sent_value(column, value) for column, value in zip(columns, row, strict=True)]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {"dataframe_split": {"columns": list(columns), "data": data}}


def ping(url: str, timeout: float = 5.0) -> bool:
    """Tell whether a model server has loaded its method: ``GET /ping`` answers 200.

    Args:
        url: The model server's URL (``server_url``).
        timeout: The most seconds to wait for the connection, then for the answer.

    Returns:
        True on a 200; False on any other status (nginx answers 502 while the method loads), a
        refused or closed connection, a timeout, a peer that does not answer HTTP, or a URL that
        is not one.
    """
    try:
        with _open(_path_url(url, _PING_PATH), timeout) as response:
            return response.status == 200
    except (ServingError, *_NO_ANSWER):
        return False


def predict(
    url: str, problems: pd.DataFrame, timeout: float = SERVING_TIMEOUT_S
) -> list[ServedPrediction]:
    """Ask a model server for the focus tags of problems, in one request.

    Args:
        url: The model server's URL (``server_url``).
        problems: Raw problems indexed by ``sample_id`` (``request_payload``). A method that
            calls an API takes one problem per request.
        timeout: The most seconds to wait for the connection, then for each read of the answer.

    Returns:
        One prediction per problem, in the order of ``problems``; ``seconds`` is the request's
        round trip, the same for each.

    Raises:
        CliError: As ``request_payload``, before anything is sent.
        ServingError: If no answer came (status None: a refused or closed connection, a
            timeout, a peer that does not answer HTTP, a URL that is not ``http://`` or
            ``https://`` or that cannot be used); if the answer is not a 200 (its status, with
            the first line of MLflow's JSON ``message`` when the body holds one, else the status
            line: nginx's 502 and 504 are HTML pages, a 415 plain text); or if a 200's body is
            not one record per problem, each with the method's name and the focus tags (status
            200).
    """
    body = json.dumps(request_payload(problems), allow_nan=False).encode("utf-8")
    invocations_url = _path_url(url, _INVOCATIONS_PATH)
    started = time.perf_counter()
    answer = _answer_body(invocations_url, body, timeout)
    seconds = time.perf_counter() - started
    return _predictions(answer, len(problems), seconds, invocations_url)


def _sent_value(column: str, value: object) -> str | None:
    """Write one field of a problem as the request sends it (``request_payload``'s rules)."""
    if column in CLI_JSON_LIST_FIELDS and isinstance(value, list):
        return json.dumps(value)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _path_url(url: str, path: str) -> str:
    """Join a model server's URL and one of its paths; refuse a URL that is not HTTP."""
    if not url.startswith(("http://", "https://")):
        raise ServingError(
            None, f"{url!r} is not the URL of a model server: give http://<host>:<port>."
        )
    return url.rstrip("/") + path


def _open(request: urllib.request.Request | str, timeout: float) -> http.client.HTTPResponse:
    """Open a request through no proxy (the module docstring says why)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _answer_body(url: str, body: bytes, timeout: float) -> bytes:
    """Post a JSON body and return the body of its 200 answer, or raise one ServingError."""
    try:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with _open(request, timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ServingError(error.code, _error_message(error)) from None
    except _NO_ANSWER as error:
        # One line: http.client quotes the first line of a peer that does not answer HTTP, line
        # end included.
        reason = " ".join(str(getattr(error, "reason", error)).split())
        raise ServingError(None, f"no answer from the model server at {url}: {reason}") from None


def _error_message(error: urllib.error.HTTPError) -> str:
    """Say what an answer other than 200 means: MLflow's JSON message, else the status line."""
    answered = f"the model server answered {error.code} {error.reason}"
    try:
        message = json.loads(error.read())["message"]
        match = _PREDICT_FAILURE.fullmatch(message)
    except (*_NO_ANSWER, KeyError, TypeError):
        return answered
    lines = (match.group(1) if match else message).strip().splitlines()
    return f"{answered}: {lines[0].strip()}" if lines else answered


def _predictions(body: bytes, count: int, seconds: float, url: str) -> list[ServedPrediction]:
    """Read the answer of a request about ``count`` problems, or raise a ServingError (200)."""
    try:
        records = json.loads(body)["predictions"]
        predictions = [_prediction(record, seconds) for record in records]
    except (*_NO_ANSWER, KeyError, TypeError, AttributeError) as error:
        raise ServingError(
            200,
            f"the answer of {url} cannot be read ({type(error).__name__}): it is not one "
            "prediction per problem, each with the method's name and its focus tags",
        ) from None
    if len(predictions) != count:
        raise ServingError(
            200,
            f"the answer of {url} does not hold one prediction per problem: {len(predictions)} "
            f"for {count}",
        )
    return predictions


def _prediction(record: dict, seconds: float) -> ServedPrediction:
    """Read one record of a model server's answer, as ``records.served_records`` writes it."""
    failed = record.get(FAILED_COLUMN)
    cost_usd = record.get(COST_COLUMN)
    return ServedPrediction(
        method=str(record[METHOD_COLUMN]),
        tags=tuple(tag for tag in FOCUS_TAGS if record[tag] == 1),
        scores=_by_tag(record, score_column),
        thresholds=_by_tag(record, threshold_column),
        failed=None if failed is None else bool(failed),
        cost_usd=None if cost_usd is None else float(cost_usd),
        seconds=seconds,
    )


def _by_tag(record: dict, column: Callable[[str], str]) -> dict[str, float]:
    """Read the float of each focus tag whose column ``column(tag)`` the record holds."""
    return {tag: float(record[column(tag)]) for tag in FOCUS_TAGS if column(tag) in record}
