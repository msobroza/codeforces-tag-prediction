"""Tests for the OpenAI adapter, on a fake ``openai`` module: no request leaves the process.

The fake stands in for the parts of the SDK (openai 3.26.0) the adapter uses: ``AsyncOpenAI``
as an async context manager, ``responses.with_raw_response.create`` returning a raw response
(``headers``, ``request_id``, ``retries_taken``, ``parse()``), and the errors
``APIConnectionError`` and ``APIStatusError`` (``status_code``, ``code``, ``param``, ``type``,
``request_id``).
"""

import asyncio
import dataclasses
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from codeforces_tags.config import LLM_MAX_RETRIES, LLM_TIMEOUT_S
from codeforces_tags.llm.client import (
    USAGE_KEYS,
    LlmAnswer,
    RejectedSetting,
    TransientApiError,
    session,
)
from codeforces_tags.llm.providers.openai import SDK_MODULE, ask, request_body

USAGE = SimpleNamespace(
    input_tokens=100,
    input_tokens_details=SimpleNamespace(cached_tokens=10),
    output_tokens=5,
    output_tokens_details=SimpleNamespace(reasoning_tokens=2),
)


class FakeConnectionError(Exception):
    """Stands in for ``openai.APIConnectionError``, and its subclass ``APITimeoutError``."""


class FakeStatusError(Exception):
    """Stands in for ``openai.APIStatusError``: the API answered with an HTTP error."""

    def __init__(self, status_code, *, code=None, param=None, error_type="invalid_request_error"):
        """Hold the HTTP status and the error's code, parameter and type, as the API gives them."""
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.code = code
        self.param = param
        self.type = error_type
        self.message = f"Error code: {status_code}"
        self.request_id = f"req_{status_code}"


@dataclass
class Server:
    """What the fake API answers and what it was sent.

    Each call takes the next outcome, a raw response or an error; the bodies sent and the
    clients opened are kept in order.
    """

    outcomes: list = field(default_factory=list)
    bodies: list = field(default_factory=list)
    clients: list = field(default_factory=list)


class FakeClient:
    """Stands in for ``openai.AsyncOpenAI``: usable only in the loop that opened it."""

    def __init__(self, server, **options):
        """Record the client's options; it is opened by ``async with``, as the SDK's is."""
        self.server = server
        self.options = options
        self.loop = None
        self.closed = False
        self.responses = SimpleNamespace(with_raw_response=SimpleNamespace(create=self.create))
        server.clients.append(self)

    async def __aenter__(self):
        """Open the client in the running loop."""
        self.loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, *exc_info):
        """Close the client, in the loop that opened it."""
        assert asyncio.get_running_loop() is self.loop
        self.closed = True

    async def create(self, **body):
        """Record the body, and return the next outcome or raise it.

        Reusing a client in another event loop fails every other call in httpx: a client is
        opened in the running loop and used only there, until it is closed.
        """
        assert asyncio.get_running_loop() is self.loop and not self.closed
        self.server.bodies.append(body)
        outcome = self.server.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(name="server")
def server_fixture(monkeypatch):
    """Put a fake ``openai`` module in ``sys.modules``, and return its fake API."""
    server = Server()
    sdk = types.ModuleType("openai")
    sdk.AsyncOpenAI = lambda **options: FakeClient(server, **options)
    sdk.APIConnectionError = FakeConnectionError
    sdk.APIStatusError = FakeStatusError
    monkeypatch.setitem(sys.modules, "openai", sdk)
    return server


def raw_response(*, content=None, status="completed", usage=USAGE, retries=0, incomplete=None):
    """Return a raw response as ``with_raw_response.create`` gives it, fields overridable."""
    content = content or [SimpleNamespace(type="output_text", text="dp, math")]
    response = SimpleNamespace(
        id="resp_1",
        model="model-1-2026-09-01",
        status=status,
        output=[SimpleNamespace(type="message", content=content)],
        output_text="".join(part.text for part in content if part.type == "output_text"),
        usage=usage,
        service_tier="default",
        incomplete_details=incomplete,
    )
    return SimpleNamespace(
        headers={"openai-processing-ms": "321", "x-request-id": "req_1"},
        request_id="req_1",
        retries_taken=retries,
        parse=lambda: response,
    )


def send(request):
    """Answer one request through the adapter, in a session of its own."""

    async def run():
        async with session(ask(request.settings)) as send_one:
            return await send_one(request)

    return asyncio.run(run())


def test_the_sdk_of_the_adapter_is_openai():
    assert SDK_MODULE == "openai"


def test_the_body_of_a_text_request_is_one_user_message_with_the_settings(make_request):
    assert request_body(make_request()) == {
        "model": "model-1",
        "input": [{"role": "user", "content": "Tag this."}],
        "max_output_tokens": 512,
        "store": False,
        "service_tier": "default",
        "prompt_cache_options": {"mode": "explicit"},
        "reasoning": {"effort": "none"},
        "temperature": 0.3157894736842105,
        "top_p": 1.0,
    }


def test_a_request_to_openai_has_the_body_of_the_adapter(make_request):
    request = make_request("json")

    assert request.body() == request_body(request)


def test_the_body_of_a_json_request_adds_its_schema_as_a_strict_format(make_request):
    request = make_request("json")

    assert request_body(request)["text"] == {
        "format": {
            "type": "json_schema",
            "name": "problem_tags",
            "schema": {"type": "object", "required": ["tags"]},
            "strict": True,
        }
    }
    assert "text" not in request_body(make_request())


def test_the_body_leaves_out_the_effort_the_temperature_and_the_top_p_when_unset(
    make_request, make_settings
):
    settings = make_settings(reasoning_effort=None, temperature=None, top_p=None)

    body = request_body(make_request(settings=settings))

    assert not {"reasoning", "temperature", "top_p"} & set(body)


def test_ask_sends_the_body_and_fills_the_answer(server, make_request):
    request = make_request()
    server.outcomes.append(raw_response())

    answer = send(request)

    assert server.bodies == [request_body(request)]
    timing = ("started_ns", "latency_s", "created_at", "completed_at")
    assert dataclasses.replace(answer, **dict.fromkeys(timing, None)) == LlmAnswer(
        request_sha256=request.sha256(),
        sample_id=1,
        method="fake-text",
        provider="openai",
        output_text="dp, math",
        status="completed",
        served_model="model-1-2026-09-01",
        response_id="resp_1",
        request_id="req_1",
        usage={"input_tokens": 100, "cached_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2},
        extra={
            "openai-processing-ms": "321",
            "service_tier": "default",
            "incomplete_reason": None,
            "refusal": None,
        },
        started_ns=None,
        latency_s=None,
        retries=0,
        created_at=None,
        completed_at=None,
    )
    assert answer.started_ns > 0 and answer.latency_s >= 0
    assert answer.created_at <= answer.completed_at


def test_ask_opens_its_client_with_the_retries_and_the_timeout_of_the_config(server, make_request):
    server.outcomes.append(raw_response())

    send(make_request())

    assert [client.options for client in server.clients] == [
        {"max_retries": LLM_MAX_RETRIES, "timeout": LLM_TIMEOUT_S}
    ]


def test_a_session_sends_its_batch_through_one_client_closed_at_its_end(server, make_request):
    server.outcomes += [raw_response(), raw_response()]
    adapter = ask(make_request().settings)

    async def run():
        async with session(adapter) as send_one:
            answers = await asyncio.gather(
                send_one(make_request(sample_id=1)), send_one(make_request(sample_id=2))
            )
            return answers, server.clients[0].closed

    answers, closed_inside = asyncio.run(run())

    assert [answer.sample_id for answer in answers] == [1, 2]
    assert len(server.clients) == 1
    assert (closed_inside, server.clients[0].closed) == (False, True)


def test_no_client_outlives_its_event_loop(server, make_request):
    # The same Ask in two event loops, as two predict calls in one process: each loop opens
    # its own client and closes it before the loop ends.
    server.outcomes += [raw_response(), raw_response()]
    adapter = ask(make_request().settings)

    async def run():
        async with session(adapter) as send_one:
            return await send_one(make_request())

    first, second = asyncio.run(run()), asyncio.run(run())

    assert first.status == second.status == "completed"
    assert [client.closed for client in server.clients] == [True, True]
    assert server.clients[0].loop is not server.clients[1].loop


def test_an_ask_called_outside_a_session_opens_a_client_for_that_call(server, make_request):
    server.outcomes.append(raw_response())

    answer = asyncio.run(ask(make_request().settings)(make_request()))

    assert answer.status == "completed"
    assert [client.closed for client in server.clients] == [True]


def test_ask_records_the_retries_the_sdk_took(server, make_request):
    server.outcomes.append(raw_response(retries=2))

    assert send(make_request()).retries == 2


def test_ask_records_a_refusal_as_a_refused_answer(server, make_request):
    # The API marks the response completed, with a refusal instead of text: it is not an answer
    # the parser may read as an empty tag list.
    refusal = SimpleNamespace(type="refusal", refusal="I can't help with that.")
    server.outcomes.append(raw_response(content=[refusal]))

    answer = send(make_request("json"))

    assert (answer.status, answer.output_text) == ("refused", "")
    assert answer.extra["refusal"] == "I can't help with that."


def test_ask_keeps_an_incomplete_answer_with_its_reason_and_no_usage(server, make_request):
    incomplete = SimpleNamespace(reason="max_output_tokens")
    server.outcomes.append(raw_response(status="incomplete", usage=None, incomplete=incomplete))

    answer = send(make_request())

    assert answer.status == "incomplete"
    assert answer.extra["incomplete_reason"] == "max_output_tokens"
    assert answer.usage == dict.fromkeys(USAGE_KEYS, 0)


@pytest.mark.parametrize(
    "error",
    [
        FakeConnectionError("Request timed out."),
        FakeStatusError(408),
        FakeStatusError(409),
        FakeStatusError(429, code="rate_limit_exceeded"),
        FakeStatusError(500),
        FakeStatusError(503),
    ],
)
def test_ask_raises_a_transient_error_once_the_sdk_gave_up_retrying(server, make_request, error):
    # No line is written: the next llm run sends the request again.
    server.outcomes.append(error)

    with pytest.raises(TransientApiError, match="request of problem 1 of fake-text") as raised:
        send(make_request())

    assert raised.value.__cause__ is error
    # The SDK may not have retried at all (e.g. a 5xx with x-should-retry: false).
    assert f"after the SDK's retries (at most {LLM_MAX_RETRIES})" in str(raised.value)


@pytest.mark.parametrize(
    "error",
    [
        FakeStatusError(400, code="unsupported_parameter", param="temperature"),
        FakeStatusError(400, code="invalid_value", param="reasoning.effort"),
        FakeStatusError(400, code="invalid_json_schema", param="text.format.schema"),
        FakeStatusError(422, param="max_output_tokens"),
        FakeStatusError(400, code="unsupported_value"),
    ],
)
def test_ask_stops_the_run_when_the_api_rejects_a_setting(server, make_request, error):
    # Every request has the setting: the run stops before any line is written.
    server.outcomes.append(error)

    with pytest.raises(RejectedSetting, match="OpenAI rejected a setting of fake-json"):
        send(make_request("json"))


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (FakeStatusError(400, code="context_length_exceeded", param="input"), "api_error:400"),
        (FakeStatusError(400, code="invalid_prompt"), "api_error:400"),
        (FakeStatusError(422), "api_error:422"),
    ],
)
def test_ask_answers_a_refused_request_once_with_an_api_error(server, make_request, error, status):
    # This request alone was refused: a failed answer is written once, and it is final.
    request = make_request()
    server.outcomes.append(error)

    answer = send(request)

    assert (answer.status, answer.output_text, answer.request_sha256) == (
        status,
        None,
        request.sha256(),
    )
    assert (answer.served_model, answer.response_id, answer.request_id) == (
        None,
        None,
        error.request_id,
    )
    assert answer.usage == dict.fromkeys(USAGE_KEYS, 0)
    assert answer.extra == {
        "error_code": error.code,
        "error_param": error.param,
        "error_type": "invalid_request_error",
    }


@pytest.mark.parametrize(
    "error",
    [
        FakeStatusError(401),
        FakeStatusError(403),
        FakeStatusError(404),
        FakeStatusError(429, code="insufficient_quota", error_type="insufficient_quota"),
        FakeStatusError(429, error_type="insufficient_quota"),
    ],
)
def test_ask_lets_an_error_of_every_request_stop_the_run(server, make_request, error):
    # A missing key, a forbidden project, an unknown model, or no credit left (a 429 that no
    # retry can help, unlike a rate limit): no answer is written.
    server.outcomes.append(error)

    with pytest.raises(FakeStatusError) as raised:
        send(make_request())

    assert raised.value is error


def test_ask_refuses_a_request_of_other_settings(server, make_request, make_settings):
    adapter = ask(make_settings())

    with pytest.raises(ValueError, match="model-2"):
        asyncio.run(adapter(make_request(settings=make_settings(model="model-2"))))
    assert not server.bodies


def test_ask_needs_the_openai_package(monkeypatch, make_settings):
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ImportError, match="openai"):
        ask(make_settings())
