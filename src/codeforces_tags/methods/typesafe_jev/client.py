"""Ask TypeSafe's Jev about a problem, and turn what it gave into an answers-file line.

``jev_session(name)`` opens one ``AsyncTypeSafeClient`` in the running event loop and yields
the ``Ask`` of method ``name``: it sends one ``JevRequest`` with ``system_one`` and returns its
``LlmAnswer``, the LLM toolkit's line (docs/adr/0007). The SDK (dependency group
``llm-typesafe``) is imported when a session opens, and reads ``TYPESAFE_API_KEY`` then:
building requests, loading a method folder and reading answers need neither.

What a request becomes, once the SDK's own retries are spent:

- a 200 that holds a probability in [0, 1] for every focus tag: a ``completed`` line;
- a 200 that lacks a tag's Noul answer, holds a probability that is not finite or not in
  [0, 1], or whose body fails the SDK's validation: a final ``malformed`` line;
- an HTTP 400 or 422, a request the API refused for itself alone (e.g. a state over its
  budget): a final ``api_error:<status>`` line, with no text, no model and no tokens;
- a timeout, a lost connection, an HTTP 408, 409, 429 or any status from 500 up:
  ``TransientApiError``, and no line (the next run sends the request again);
- any other HTTP error (e.g. 401, 402, 403, 404, 413), and a missing key: the SDK's own
  error, and the run stops.

An SDK error's text and body can echo the state, so neither reaches a line or a message.
"""

from __future__ import annotations

import functools
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, NamedTuple

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import USAGE_KEYS, LlmAnswer, TransientApiError
from codeforces_tags.methods.typesafe_jev.questions import JevRequest, question_id
from codeforces_tags.methods.typesafe_jev.settings import (
    MAX_RETRIES,
    MODEL,
    NAMES,
    PROVIDER,
    TIMEOUT_S,
)

# Sends one request and returns its answer line: jev_session's, or a test's fake.
Ask = Callable[[JevRequest], Awaitable[LlmAnswer]]
# HTTP statuses the API may answer later, besides every status from 500 up (529 included): a
# request timeout, a conflict and a rate limit.
TRANSIENT_STATUSES: tuple[int, ...] = (408, 409, 429)
# HTTP statuses of a request the API refused for itself alone: a final api_error line.
REFUSED_STATUSES: tuple[int, ...] = (400, 422)


class JevOutcome(NamedTuple):
    """What one call gave, before it is a line: the fields of a line that vary by outcome.

    Attributes:
        status: ``"completed"``, ``"malformed"`` or ``"api_error:<HTTP status>"``.
        output_text: When completed, the compact JSON of the probabilities, in ``FOCUS_TAGS``
            order; None otherwise.
        served_model: The model that answered, when a 200 body was read; None otherwise.
        request_id: TypeSafe's ``x-typesafe-request-id`` header; None when it was absent.
        usage: The tokens, by ``USAGE_KEYS``: TypeSafe reports no cached or reasoning ones.
        extra: Empty, except for ``malformed``: the tags whose answer is ``missing`` and those
            whose answer is ``invalid``.
    """

    status: str
    output_text: str | None
    served_model: str | None
    request_id: str | None
    usage: dict[str, int]
    extra: dict[str, object]


def response_outcome(response: Any, request_id: str | None) -> JevOutcome:
    """Read a 200 response: a probability for each focus tag, or what kept it from being one.

    The probabilities are read from ``response.nouls`` by question id. A tag with no Noul
    answer there (absent, or answered with another type) is missing; a probability that is not
    finite or not in [0, 1], which the SDK accepts, is invalid. The tokens are kept either
    way: the request was answered, and billed.

    Args:
        response: The SDK's ``SystemOneResponse``.
        request_id: Its ``x-typesafe-request-id`` header, or None.

    Returns:
        A ``completed`` outcome, or a ``malformed`` one whose ``extra`` names the missing and
        the invalid tags, each in ``FOCUS_TAGS`` order.
    """
    found = _noul_values(response.nouls)
    unread = _unread_tags(found)
    usage = _usage(response.usage)
    if unread:
        return JevOutcome("malformed", None, response.model, request_id, usage, unread)
    text = json.dumps(
        {tag: float(found[tag]) for tag in FOCUS_TAGS}, separators=(",", ":"), allow_nan=False
    )
    return JevOutcome("completed", text, response.model, request_id, usage, {})


def failure_outcome(status: str, request_id: str | None) -> JevOutcome:
    """Describe a request that got no readable answer.

    Args:
        status: ``"api_error:<HTTP status>"``, or ``"malformed"`` for a body that the SDK's
            validation refused.
        request_id: TypeSafe's ``x-typesafe-request-id`` header, or None.

    Returns:
        The outcome, with no text, no served model, zero tokens and nothing extra.
    """
    return JevOutcome(status, None, None, request_id, dict.fromkeys(USAGE_KEYS, 0), {})


def jev_answer(
    request: JevRequest, name: str, outcome: JevOutcome, timing: Mapping[str, object]
) -> LlmAnswer:
    """Write one outcome as the toolkit's answer line.

    TypeSafe gives one id, the ``x-typesafe-request-id`` header, which is both the response's
    and the request's id. The SDK does not report its retries: they count 0. The runner stamps
    ``git_commit`` and ``code_dirty``; ``tags`` and ``unknown_pieces`` belong to tag-list
    answers and stay empty.

    Args:
        request: The request answered.
        name: The method's registry key.
        outcome: What the call gave.
        timing: ``started_ns``, ``latency_s``, ``created_at`` and ``completed_at`` of the call.

    Returns:
        The line, ``failed`` unless its status is ``completed``.
    """
    return LlmAnswer(
        request_sha256=request.sha256(),
        method=name,
        sample_id=request.sample_id,
        provider=PROVIDER,
        status=outcome.status,
        output_text=outcome.output_text,
        served_model=outcome.served_model,
        response_id=outcome.request_id,
        request_id=outcome.request_id,
        usage=dict(outcome.usage),
        extra=dict(outcome.extra),
        started_ns=timing["started_ns"],
        latency_s=timing["latency_s"],
        retries=0,
        created_at=timing["created_at"],
        completed_at=timing["completed_at"],
        failed=outcome.status != "completed",
    )


@asynccontextmanager
async def jev_session(name: str) -> AsyncIterator[Ask]:
    """Open one TypeSafe client for a method's requests, in the running event loop.

    The SDK is imported, and reads ``TYPESAFE_API_KEY``, only now: a command with nothing left
    to send never opens a session. The client sends the pinned model, retries at most
    ``MAX_RETRIES`` times within the SDK's own budget of 30 s per call, and gives each attempt
    ``TIMEOUT_S`` seconds; it is closed when the session ends, in the loop that opened it.
    Tests replace the whole session through the module: ``client.jev_session``.

    Args:
        name: The method whose requests the session sends, one of ``NAMES``.

    Yields:
        The ``Ask`` that sends one ``JevRequest`` through the client and returns its line.

    Raises:
        ValueError: If ``name`` is not a TypeSafe Jev method; nothing is imported then.
        ImportError: If the SDK is not installed (``uv sync --group llm-typesafe``).
        typesafe_sdk.TypeSafeError: If ``TYPESAFE_API_KEY`` is missing or malformed.
    """
    if name not in NAMES:
        raise ValueError(f"{name!r} is not a TypeSafe Jev method: expected one of {NAMES}.")
    import typesafe_sdk  # pylint: disable=import-outside-toplevel,import-error

    retry = typesafe_sdk.RetryPolicy(max_retries=MAX_RETRIES)
    async with typesafe_sdk.AsyncTypeSafeClient(
        model=MODEL, retry=retry, timeout=TIMEOUT_S
    ) as sdk_client:
        yield functools.partial(_ask, typesafe_sdk, sdk_client, name)


async def _ask(sdk: Any, sdk_client: Any, name: str, request: JevRequest) -> LlmAnswer:
    """Send one request through an open client, and return its line or raise (module doc)."""
    sent_ns, sent_at = time.time_ns(), time.perf_counter()
    try:
        response = await sdk_client.system_one(**request.body())
    except sdk.TypeSafeAPIResponseValidationError as error:
        # A 200 whose body is no answer. Its base class, TypeSafeAPIError, is caught next.
        outcome = failure_outcome("malformed", error.request_id)
    except sdk.TypeSafeAPIError as error:
        outcome = failure_outcome(_refused_status(name, request, error), error.request_id)
    except sdk.TypeSafeAPIConnectionError as error:
        # Timeouts too: TypeSafeAPITimeoutError subclasses it, and neither is a TypeSafeAPIError.
        raise _no_answer(name, request, error) from error
    else:
        outcome = response_outcome(response, _request_id(sdk, response))
    return jev_answer(request, name, outcome, _timing(sent_ns, sent_at))


def _refused_status(name: str, request: JevRequest, error: Any) -> str:
    """Return the line status of a request the API refused alone, or raise (module doc)."""
    if error.status in REFUSED_STATUSES:
        return f"api_error:{error.status}"
    if error.status in TRANSIENT_STATUSES or error.status >= 500:
        raise _no_answer(name, request, error) from error
    raise error


def _no_answer(name: str, request: JevRequest, error: Exception) -> TransientApiError:
    """Return the error of a request that TypeSafe did not answer once the SDK gave up.

    It names the SDK error's class and status, never its text, which can echo the state.
    """
    status = getattr(error, "status", None)
    kind = type(error).__name__ if status is None else f"{type(error).__name__} {status}"
    return TransientApiError(
        f"TypeSafe did not answer the request of problem {request.sample_id} of {name} after "
        f"the SDK's retries (at most {MAX_RETRIES}): {kind}."
    )


def _request_id(sdk: Any, response: Any) -> str | None:
    """Return a response's x-typesafe-request-id header, or None when it was absent."""
    try:
        return response.request_id
    except sdk.TypeSafeError:
        # The SDK raises when the header is missing; the API does not promise it.
        return None


def _timing(sent_ns: int, sent_at: float) -> dict[str, object]:
    """Return a call's timing fields, from the wall and performance clocks read at sending."""
    latency_s = time.perf_counter() - sent_at
    created_at = datetime.fromtimestamp(sent_ns / 1e9, UTC)
    return {
        "started_ns": sent_ns,
        "latency_s": latency_s,
        "created_at": created_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _noul_values(nouls: Mapping[str, Any]) -> dict[str, float]:
    """Return each focus tag's Noul probability, by tag, for the tags that have one."""
    return {tag: nouls[question_id(tag)].noul for tag in FOCUS_TAGS if question_id(tag) in nouls}


def _unread_tags(found: Mapping[str, float]) -> dict[str, object]:
    """Name the focus tags with no answer, then those whose answer is no probability."""
    named = {
        "missing": [tag for tag in FOCUS_TAGS if tag not in found],
        "invalid": [tag for tag, value in found.items() if not _is_probability(value)],
    }
    return {key: tags for key, tags in named.items() if tags}


def _is_probability(value: float) -> bool:
    """Tell whether a Noul answer is a probability: finite, and between 0 and 1."""
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _usage(usage: Any) -> dict[str, int]:
    """Return a response's tokens by USAGE_KEYS; a count the API left out (None) is 0."""
    tokens = dict.fromkeys(USAGE_KEYS, 0)
    tokens["input_tokens"] = usage.input_tokens or 0
    tokens["output_tokens"] = usage.output_tokens or 0
    return tokens
