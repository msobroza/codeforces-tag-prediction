"""The OpenAI adapter: the Responses API, through the official SDK.

The SDK (dependency group ``llm``) is imported inside ``ask`` only, so loading a prompted method
or building its requests never needs it. Each request is one user message, without
instructions; the transport settings come from ``config.LLM_*``: no response is stored at
OpenAI, no implicit cache breakpoint is written, and the SDK retries with backoff, at most
``LLM_MAX_RETRIES`` times. ``LLM_TIMEOUT_S`` bounds each attempt, not the request: a timed-out
attempt is retried too, so a request can take about ``(LLM_MAX_RETRIES + 1) * LLM_TIMEOUT_S``
seconds plus the backoff before it fails. The SDK may also retry less, or not at all (a server
error with ``x-should-retry: false``, a ``retry-after`` over its own limit).

What a failed request becomes:

- a timeout, a lost connection, a rate limit or a server error, once the SDK gave up retrying:
  ``TransientApiError``, and no answer (the next run sends it again);
- an HTTP 400 or 422 that names a setting of the request (its ``param``, e.g. ``temperature``
  or ``text.format.schema``, or an ``unsupported_*`` code): ``RejectedSetting``, and the run
  stops;
- any other HTTP 400 or 422 (e.g. the input is too long): an answer with the status
  ``api_error:<HTTP status>``, written once, like any response;
- any other HTTP error (a missing key, an unknown model), and a 429 ``insufficient_quota`` (no
  credit left, which no retry fixes, unlike a rate limit): the SDK's error, and the run stops.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from types import ModuleType
from typing import Any

from codeforces_tags.config import (
    LLM_MAX_RETRIES,
    LLM_PROMPT_CACHE_MODE,
    LLM_SERVICE_TIER,
    LLM_STORE,
    LLM_TIMEOUT_S,
)
from codeforces_tags.llm.client import (
    USAGE_KEYS,
    Ask,
    LlmAnswer,
    RejectedSetting,
    TransientApiError,
)
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.llm.settings import LlmSettings

SDK_MODULE = "openai"
PROVIDER = "openai"
# The response header that holds OpenAI's own processing time, in milliseconds.
PROCESSING_MS_HEADER = "openai-processing-ms"
# HTTP statuses the SDK retries, besides every status from 500 up: a request timeout, a lock
# timeout and a rate limit. Once it gives up, the request may well succeed later.
_TRANSIENT_STATUSES = (408, 409, 429)
# HTTP statuses of a request the API refused: for a setting, or for this request alone.
_REFUSED_STATUSES = (400, 422)
# Error codes of a refused setting, whatever parameter they name.
_SETTING_ERROR_CODES = ("unsupported_parameter", "unsupported_value")
# The code (and type) of a 429 that is no rate limit: the account has no credit left, or reached
# its spend limit. Every request would fail alike, so the run stops.
_NO_CREDIT = "insufficient_quota"
# The top-level parameter that an error's dotted or indexed "param" names.
_TOP_LEVEL_PARAM = re.compile(r"[^.\[]*")


def request_body(request: LlmRequest) -> dict[str, object]:
    """Build the Responses API body of a request.

    Args:
        request: The request.

    Returns:
        The keyword arguments of ``responses.create``: the model, the input as one user
        message, the transport settings, then the reasoning effort, temperature and top_p when
        set, and, for the ``json`` output format, the schema as a strict ``text.format``.
    """
    settings = request.settings
    body: dict[str, object] = {
        "model": settings.model,
        "input": [{"role": "user", "content": request.input}],
        "max_output_tokens": settings.max_output_tokens,
        "store": LLM_STORE,
        "service_tier": LLM_SERVICE_TIER,
        "prompt_cache_options": {"mode": LLM_PROMPT_CACHE_MODE},
    }
    if settings.reasoning_effort is not None:
        body["reasoning"] = {"effort": settings.reasoning_effort}
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.top_p is not None:
        body["top_p"] = settings.top_p
    if settings.output_format == "json":
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.schema_name,
                "schema": request.schema,
                "strict": True,
            }
        }
    return body


def ask(settings: LlmSettings) -> Ask:
    """Return the ``Ask`` that sends a method's requests to OpenAI.

    Args:
        settings: The settings of every request it will send.

    Returns:
        An ``OpenAiAsk``: open it with ``client.session`` for a batch of requests.

    Raises:
        ImportError: If the ``openai`` package is not installed (``uv sync --group llm``).
    """
    import openai  # pylint: disable=import-outside-toplevel,import-error

    return OpenAiAsk(openai, settings)


class OpenAiAsk:
    """Sends requests of one method's settings to OpenAI.

    Inside ``client.session``, its requests share one ``AsyncOpenAI`` client, opened in the
    running event loop and closed at the session's end. Called outside one, each call opens
    and closes its own client.
    """

    def __init__(self, sdk: ModuleType, settings: LlmSettings) -> None:
        """Bind the SDK and the settings.

        Args:
            sdk: The ``openai`` module.
            settings: The settings of every request it will send.
        """
        self.sdk = sdk
        self.settings = settings

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Ask]:
        """Open a client in the running loop.

        Yields:
            The ``Ask`` that sends through it.
        """
        options = {"max_retries": LLM_MAX_RETRIES, "timeout": LLM_TIMEOUT_S}
        async with self.sdk.AsyncOpenAI(**options) as client:
            yield partial(self._send, client)

    async def __call__(self, request: LlmRequest) -> LlmAnswer:
        """Send one request through a client of its own.

        Args:
            request: The request.

        Returns:
            Its answer.
        """
        async with self.open() as send:
            return await send(request)

    async def _send(self, client: Any, request: LlmRequest) -> LlmAnswer:
        """Send one request through an open client, and turn its response into an answer."""
        if request.settings != self.settings:
            raise ValueError(
                f"This Ask sends requests of {self.settings}, not of {request.settings}."
            )
        body = request_body(request)
        started = (time.time_ns(), time.perf_counter())
        try:
            raw = await client.responses.with_raw_response.create(**body)
        except self.sdk.APIConnectionError as error:
            raise _transient(request, error) from error
        except self.sdk.APIStatusError as error:
            _raise_unless_answered(request, body, error)
            return _error_answer(request, error, _timing(started))
        return _answer(request, raw, _timing(started))


def _transient(request: LlmRequest, error: Exception) -> TransientApiError:
    """Return the error of a request that OpenAI did not answer, after the SDK's retries."""
    return TransientApiError(
        f"OpenAI did not answer the request of problem {request.sample_id} of "
        f"{request.method}, after the SDK's retries (at most {LLM_MAX_RETRIES}): {error}"
    )


def _raise_unless_answered(request: LlmRequest, body: dict[str, object], error: Any) -> None:
    """Raise unless an HTTP error refused this request alone, which an answer then records."""
    status = error.status_code
    if _NO_CREDIT in (error.code, error.type):
        raise error
    if status in _TRANSIENT_STATUSES or status >= 500:
        raise _transient(request, error) from error
    if status not in _REFUSED_STATUSES:
        raise error
    param = _TOP_LEVEL_PARAM.match(error.param or "").group(0)
    if param != "input" and (param in body or error.code in _SETTING_ERROR_CODES):
        raise RejectedSetting(
            f"OpenAI rejected a setting of {request.method}'s requests "
            f"({error.param or error.code}): {error.message}"
        ) from error


def _timing(started: tuple[int, float]) -> dict[str, object]:
    """Return an answer's timing fields, from the wall and performance clocks at sending."""
    started_ns, start = started
    return {
        "started_ns": started_ns,
        "latency_s": time.perf_counter() - start,
        "created_at": datetime.fromtimestamp(started_ns / 1e9, UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _answer(request: LlmRequest, raw: Any, timing: dict[str, object]) -> LlmAnswer:
    """Turn a raw response into an answer."""
    response = raw.parse()
    refusal = _refusal(response)
    incomplete = response.incomplete_details
    return LlmAnswer(
        request_sha256=request.sha256(),
        sample_id=request.sample_id,
        method=request.method,
        provider=PROVIDER,
        output_text=response.output_text,
        status=response.status if refusal is None else "refused",
        served_model=response.model,
        response_id=response.id,
        request_id=raw.request_id,
        usage=_usage(response.usage),
        extra={
            PROCESSING_MS_HEADER: raw.headers.get(PROCESSING_MS_HEADER),
            "service_tier": response.service_tier,
            "incomplete_reason": incomplete.reason if incomplete is not None else None,
            "refusal": refusal,
        },
        retries=raw.retries_taken,
        **timing,
    )


def _error_answer(request: LlmRequest, error: Any, timing: dict[str, object]) -> LlmAnswer:
    """Turn the API's refusal of this request into a failed answer, with no text."""
    return LlmAnswer(
        request_sha256=request.sha256(),
        sample_id=request.sample_id,
        method=request.method,
        provider=PROVIDER,
        output_text=None,
        status=f"api_error:{error.status_code}",
        served_model=None,
        response_id=None,
        request_id=error.request_id,
        usage=dict.fromkeys(USAGE_KEYS, 0),
        extra={"error_code": error.code, "error_param": error.param, "error_type": error.type},
        retries=0,
        **timing,
    )


def _refusal(response: Any) -> str | None:
    """Return the model's refusal, if its message holds one instead of text."""
    refusals = [
        content.refusal
        for item in response.output
        if item.type == "message"
        for content in item.content
        if content.type == "refusal"
    ]
    return refusals[0] if refusals else None


def _usage(usage: Any) -> dict[str, int]:
    """Return the token counts of a response, by ``USAGE_KEYS``: zeros when it has none."""
    if usage is None:
        return dict.fromkeys(USAGE_KEYS, 0)
    return {
        "input_tokens": usage.input_tokens,
        "cached_tokens": usage.input_tokens_details.cached_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
    }
