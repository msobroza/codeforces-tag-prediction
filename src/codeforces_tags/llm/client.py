"""The seam between the toolkit and a provider: an answer, and the ``Ask`` that gets it.

An ``Ask`` sends one ``LlmRequest`` and returns its ``LlmAnswer``. A provider's adapter builds
one with ``ask(settings)`` (``llm.providers``), and ``ask_for(settings)`` finds that adapter;
tests inject a fake ``Ask`` instead, as the near-duplicate audit injects its ``Query``. An
adapter's ``Ask`` holds an HTTP client, which must live in one event loop: ``session`` opens it
in the running loop for a batch of requests and closes it there.

An answer is provider-neutral: what any API returns (status, text, served model, ids, token
usage, timing, retries), plus ``extra``, a dict of the provider's own fields.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

from codeforces_tags.llm import providers
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.llm.settings import LlmSettings

# The token counts of an answer's usage: every input token, the cached ones among them, every
# output token, and the reasoning ones among them.
USAGE_KEYS: tuple[str, ...] = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
# What str.splitlines() takes for a line break and json.dumps(..., ensure_ascii=False) leaves as
# it is: escaped, so an answer is one line of an answers file whatever reads it.
_LINE_BREAKS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


@dataclass(frozen=True)
class LlmAnswer:  # pylint: disable=too-many-instance-attributes
    """One frozen API response for one problem and method, keyed by its request's hash.

    It holds no statement or code text: the dataset is not redistributed.

    Attributes:
        request_sha256: ``LlmRequest.sha256()`` of the request it answers.
        sample_id: The problem asked about.
        method: The prompted method's registry key.
        provider: The provider that answered, e.g. ``"openai"``.
        output_text: The answer's text; None when the API refused the request.
        status: ``"completed"`` for a whole answer; else the API's status (e.g.
            ``"incomplete"``), ``"refused"`` for a refusal, or ``"api_error:<HTTP status>"``
            when the API refused the request.
        served_model: The model that answered, as the API names it (e.g. a dated snapshot).
        response_id: The API's id of the response.
        request_id: The API's id of the HTTP request.
        usage: The tokens, by ``USAGE_KEYS``; zeros when the API reported none.
        extra: The provider's own fields, e.g. OpenAI's ``openai-processing-ms``.
        started_ns: The client's wall clock just before sending, in ns since the epoch.
        latency_s: The wall-clock time from sending to the whole answer, in seconds.
        retries: How many times the provider's SDK sent the request again.
        created_at: When the request was sent, ISO 8601 in UTC.
        completed_at: When the answer came back, ISO 8601 in UTC.
        git_commit: The commit of the code that ran the request, stamped by the runner.
        code_dirty: Whether that code had uncommitted changes, stamped by the runner.
        tags: The tags parsed from the answer, a convenience copy written by the runner.
        failed: Whether the answer failed (``llm.answers``), written by the runner.
        unknown_pieces: The pieces of the answer that name no tag, written by the runner.
    """

    request_sha256: str
    sample_id: int
    method: str
    provider: str
    output_text: str | None
    status: str
    served_model: str | None
    response_id: str | None
    request_id: str | None
    usage: dict[str, int]
    extra: dict[str, object]
    started_ns: int
    latency_s: float
    retries: int
    created_at: str
    completed_at: str
    git_commit: str = ""
    code_dirty: bool = False
    tags: tuple[str, ...] = ()
    failed: bool = False
    unknown_pieces: int = 0

    def to_json(self) -> str:
        """Return the answer as one line of JSON, without its newline.

        Keys are sorted and non-ASCII text is kept as it is, so the answers file reads and
        diffs well; the characters that ``str.splitlines()`` takes for line breaks are escaped.

        Returns:
            The line.
        """
        line = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        for character, escape in _LINE_BREAKS.items():
            line = line.replace(character, escape)
        return line

    @classmethod
    def from_json(cls, line: str) -> LlmAnswer:
        """Read an answer back from ``to_json``'s line.

        Args:
            line: The line, with or without its newline.

        Returns:
            The answer.

        Raises:
            ValueError: If the line is not JSON.
            TypeError: If the line is not a JSON object, if a field that has no default is
                missing, or if a key is not a field.
        """
        values = json.loads(line)
        if not isinstance(values, dict):
            raise TypeError(f"An answer is a JSON object, not a {type(values).__name__}.")
        if "tags" in values:
            values = values | {"tags": tuple(values["tags"])}
        return cls(**values)


# Sends one request and returns its answer: a provider's adapter, or a test's fake.
Ask = Callable[[LlmRequest], Awaitable[LlmAnswer]]


class TransientApiError(RuntimeError):
    """The provider did not answer, after its SDK's retries: no line is written.

    A timeout, a rate limit or a server error. The next ``llm run`` sends the request again.
    """


class RejectedSetting(RuntimeError):
    """The provider rejected a setting of the request, e.g. the temperature: the run stops.

    Every request of the method has the setting, so no line is written for any of them.
    """


@asynccontextmanager
async def session(ask: Ask) -> AsyncIterator[Ask]:
    """Open an ``Ask`` for a batch of requests, in the running event loop.

    An adapter's ``Ask`` has an ``open()`` method, an async context manager that opens its
    client and gives the ``Ask`` that sends through it; the client is closed when the session
    ends, so it never outlives the loop. Any other ``Ask`` (e.g. a test's coroutine function)
    is used as it is.

    Args:
        ask: The ``Ask`` to open.

    Yields:
        The ``Ask`` to send the batch's requests with.
    """
    opener = getattr(ask, "open", None)
    if opener is None:
        yield ask
        return
    async with opener() as opened:
        yield opened


def ask_for(settings: LlmSettings) -> Ask:
    """Return the ``Ask`` of the settings' provider.

    The provider's adapter module, which ``llm.providers.PROVIDERS`` names, is imported now:
    a method that never sends a request never imports it, nor its SDK.

    Args:
        settings: The settings of the requests to send.

    Returns:
        The adapter's ``ask(settings)``.

    Raises:
        ValueError: If ``settings.provider`` has no adapter.
    """
    return providers.provider_module(settings.provider).ask(settings)
