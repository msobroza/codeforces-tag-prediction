"""Tests for the Jev client, through a fake ``typesafe_sdk`` module: no request leaves the process.

Every test but those of ``jev_answer`` and ``failure_outcome`` opens the real ``jev_session``
over the fake SDK of the ``server`` fixture (``conftest.py``): the mapping tested is a run's.
"""

import asyncio
import dataclasses
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import USAGE_KEYS, LlmAnswer, TransientApiError
from codeforces_tags.methods.typesafe_jev import client
from codeforces_tags.methods.typesafe_jev.questions import JevRequest, noul_questions
from codeforces_tags.methods.typesafe_jev.settings import MODEL

# One request of each method; problem 5's state holds no word the tests look for.
STATEMENT = {"statement": "Print the gcd of a and b."}
REQUEST = JevRequest(5, MODEL, STATEMENT, noul_questions())
CODE_REQUEST = JevRequest(
    6, MODEL, {"statement": "Sort the array.", "solution": "print(sorted(a))"}, noul_questions()
)
# Text that an SDK error can echo from the state, and that no line or message may hold.
ECHOED = "the state echoed back by the API"
# A probability per focus tag, all distinct, in FOCUS_TAGS order.
DISTINCT = dict(zip(FOCUS_TAGS, (0.9, 0.05, 0.0, 0.7, 1.0, 0.01, 0.25, 0.125), strict=True))


def ask_jev(*requests, name="jev-statement"):
    """Send the requests one after the other through one real session, in a loop of their own.

    Returns:
        Their answer lines, in order.
    """

    async def ask_in_turn():
        async with client.jev_session(name) as ask:
            return [await ask(request) for request in requests]

    return asyncio.run(ask_in_turn())


def test_jev_session_builds_one_client_with_the_pinned_model_retries_and_timeout(
    server, noul_response
):
    server.outcomes += [noul_response(), noul_response(request_id="req-2")]

    lines = ask_jev(REQUEST, CODE_REQUEST, name="jev-statement-code")

    # One client for the session: the pinned model, 5 retries within the SDK's own 30 s budget
    # per call (the policy's default), and 60 s per attempt; closed once the session ends.
    assert server.clients == [
        {"model": MODEL, "retry": server.sdk.RetryPolicy(max_retries=5), "timeout": 60.0}
    ]
    assert server.closed == 1
    assert [call["model"] for call in server.calls] == [MODEL, MODEL]
    assert [(line.sample_id, line.method) for line in lines] == [
        (5, "jev-statement-code"),
        (6, "jev-statement-code"),
    ]


def test_a_complete_answer_is_a_completed_line_with_the_probabilities_in_focus_order(
    server, noul_response
):
    server.outcomes.append(noul_response(DISTINCT, input_tokens=2412, request_id="req-7"))
    clock_before = datetime.now(UTC)

    [line] = ask_jev(REQUEST)

    assert server.calls == [{"state": STATEMENT, "model": MODEL, "questions": REQUEST.questions}]
    unclocked = dataclasses.replace(
        line, started_ns=0, latency_s=0.0, created_at="", completed_at=""
    )
    assert unclocked == LlmAnswer(
        request_sha256=REQUEST.sha256(),
        sample_id=5,
        method="jev-statement",
        provider="typesafe",
        output_text=(
            '{"math":0.9,"graphs":0.05,"strings":0.0,"number theory":0.7,"trees":1.0,'
            '"geometry":0.01,"games":0.25,"probabilities":0.125}'
        ),
        status="completed",
        served_model="jev-1.13.0",
        response_id="req-7",
        request_id="req-7",
        usage={"input_tokens": 2412, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        extra={},
        started_ns=0,
        latency_s=0.0,
        retries=0,
        created_at="",
        completed_at="",
        git_commit="",
        code_dirty=False,
        tags=(),
        failed=False,
        unknown_pieces=0,
    )
    # started_ns is the wall clock at sending; created_at and completed_at are UTC times.
    created = datetime.fromtimestamp(line.started_ns / 1e9, UTC)
    assert line.created_at == created.isoformat()
    assert abs(created - clock_before) < timedelta(minutes=1)
    assert datetime.fromisoformat(line.completed_at).tzinfo == UTC
    assert line.latency_s >= 0


def test_a_response_without_request_id_or_token_counts_is_still_completed(server, noul_response):
    # The SDK raises on response.request_id without the header, and the API may leave a count
    # out: the paid answer is still written, with no id and the missing counts as 0.
    server.outcomes.append(noul_response(input_tokens=None, output_tokens=None, request_id=None))

    [line] = ask_jev(REQUEST)

    assert (line.status, line.failed, line.served_model) == ("completed", False, MODEL)
    assert (line.request_id, line.response_id) == (None, None)
    assert line.usage == dict.fromkeys(USAGE_KEYS, 0)
    assert json.loads(line.output_text) == dict.fromkeys(FOCUS_TAGS, 0.5)


def test_a_missing_or_non_noul_answer_is_malformed_and_names_the_missing_tags(
    server, noul_response
):
    answered = {tag: 0.5 for tag in FOCUS_TAGS if tag not in ("number theory", "games")}
    choice = SimpleNamespace(type="choice", choice="yes")
    server.outcomes.append(noul_response(answered, input_tokens=2300, others={"games": choice}))

    [line] = ask_jev(REQUEST)

    assert (line.status, line.output_text, line.failed) == ("malformed", None, True)
    assert line.extra == {"missing": ["number theory", "games"]}
    # The request was answered, and billed: its model and tokens are kept.
    assert (line.served_model, line.usage["input_tokens"]) == (MODEL, 2300)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.5])
def test_a_probability_outside_0_1_or_not_finite_is_malformed(server, noul_response, value):
    server.outcomes.append(
        noul_response({**dict.fromkeys(FOCUS_TAGS, 0.5), "trees": value}, input_tokens=2600)
    )

    [line] = ask_jev(REQUEST)

    assert (line.status, line.output_text) == ("malformed", None)
    assert line.extra == {"invalid": ["trees"]}
    assert (line.served_model, line.usage["input_tokens"]) == (MODEL, 2600)


def test_a_malformed_line_names_the_missing_tags_then_the_invalid_ones(server, noul_response):
    answered = {tag: 0.5 for tag in FOCUS_TAGS if tag != "math"} | {"geometry": -1.0}
    server.outcomes.append(noul_response(answered))

    [line] = ask_jev(REQUEST)

    assert line.extra == {"missing": ["math"], "invalid": ["geometry"]}


def test_a_missing_key_fails_when_the_session_opens_before_any_request(server):
    server.key_error = server.sdk.TypeSafeError(
        "No API key was provided. Pass api_key or set the TYPESAFE_API_KEY environment variable."
    )

    with pytest.raises(server.sdk.TypeSafeError, match="No API key") as raised:
        ask_jev(REQUEST)

    assert raised.value is server.key_error
    assert (server.clients, server.calls) == ([], [])


@pytest.mark.parametrize(
    ("error_name", "status"),
    [
        ("TypeSafeAuthenticationError", 401),
        ("TypeSafeAPIError", 402),
        ("TypeSafePermissionDeniedError", 403),
        ("TypeSafeNotFoundError", 404),
        ("TypeSafeAPIError", 413),
    ],
)
def test_other_http_errors_are_re_raised_as_they_are(server, error_name, status):
    # Every request would get the same error (a refused key, no credit, an unknown model): the
    # run stops, and the session still closes its client.
    error = getattr(server.sdk, error_name)(status, request_id="req-x", body=ECHOED)
    server.outcomes.append(error)

    with pytest.raises(server.sdk.TypeSafeAPIError) as raised:
        ask_jev(REQUEST)

    assert raised.value is error
    assert server.closed == 1


def test_the_sdk_is_imported_only_when_a_session_opens(monkeypatch):
    # The client module was imported without the SDK, and holds no name of it.
    assert "typesafe_sdk" not in vars(client)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ImportError, match="typesafe_sdk"):
        ask_jev(REQUEST)


def test_jev_session_refuses_an_unknown_method(monkeypatch):
    # The name is checked before the SDK is imported: no SDK is needed to get this error.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ValueError, match="'label_prior' is not a TypeSafe Jev method"):
        ask_jev(REQUEST, name="label_prior")


def test_jev_answer_fills_every_key_of_an_llm_answer():
    outcome = client.JevOutcome(
        status="malformed",
        output_text=None,
        served_model=MODEL,
        request_id="req-3",
        usage={"input_tokens": 2300, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        extra={"missing": ["trees"]},
    )
    timing = {
        "started_ns": 1_791_000_000_000_000_000,
        "latency_s": 0.25,
        "created_at": "2026-10-08T10:00:00+00:00",
        "completed_at": "2026-10-08T10:00:00.250000+00:00",
    }

    line = client.jev_answer(CODE_REQUEST, "jev-statement-code", outcome, timing)

    assert dataclasses.asdict(line) == {
        "request_sha256": CODE_REQUEST.sha256(),
        "sample_id": 6,
        "method": "jev-statement-code",
        "provider": "typesafe",
        "output_text": None,
        "status": "malformed",
        "served_model": "jev-1.13.0",
        "response_id": "req-3",
        "request_id": "req-3",
        "usage": {
            "input_tokens": 2300,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        },
        "extra": {"missing": ["trees"]},
        "started_ns": 1_791_000_000_000_000_000,
        "latency_s": 0.25,
        "retries": 0,
        "created_at": "2026-10-08T10:00:00+00:00",
        "completed_at": "2026-10-08T10:00:00.250000+00:00",
        "git_commit": "",
        "code_dirty": False,
        "tags": (),
        "failed": True,
        "unknown_pieces": 0,
    }


def test_failure_outcome_has_no_text_model_or_usage():
    assert client.failure_outcome("api_error:422", "req-9") == client.JevOutcome(
        status="api_error:422",
        output_text=None,
        served_model=None,
        request_id="req-9",
        usage=dict.fromkeys(USAGE_KEYS, 0),
        extra={},
    )
    assert client.failure_outcome("malformed", None).request_id is None


def test_a_body_that_fails_the_sdks_validation_is_a_malformed_line_with_no_usage(server):
    # A 200 whose body is no answer (no usage, no model, an HTML page): the SDK raises its
    # validation error, a TypeSafeAPIError of status 200, which must not be read as api_error.
    server.outcomes.append(
        server.sdk.TypeSafeAPIResponseValidationError(request_id="req-5", field_path="usage")
    )

    [line] = ask_jev(REQUEST)

    assert (line.status, line.output_text, line.served_model) == ("malformed", None, None)
    assert line.usage == dict.fromkeys(USAGE_KEYS, 0)
    assert (line.request_id, line.response_id, line.extra, line.failed) == (
        "req-5",
        "req-5",
        {},
        True,
    )


@pytest.mark.parametrize(
    ("error_name", "status", "solution"),
    [
        ("TypeSafeBadRequestError", 400, "print(gcd(a, b))"),
        ("TypeSafeUnprocessableEntityError", 422, "print(gcd(a, b))"),
        # A state over the API's budget: the largest solution of the dataset has 58,387
        # characters. Refused at no cost, its line is final, small and holds no state text.
        ("TypeSafeUnprocessableEntityError", 422, "x = 1\n" * 10_000),
    ],
    # Short ids: by default pytest names a case by its 60,000-character solution.
    ids=["400", "422", "422-over-the-budget"],
)
def test_a_400_or_422_is_a_final_api_error_line_with_no_text_usage_or_model(
    server, error_name, status, solution
):
    state = {"statement": "Print the gcd of a and b.", "solution": solution}
    request = JevRequest(7, MODEL, state, noul_questions())
    server.outcomes.append(
        getattr(server.sdk, error_name)(status, request_id=f"req-{status}", body=ECHOED)
    )

    [line] = ask_jev(request, name="jev-statement-code")

    assert (line.status, line.failed) == (f"api_error:{status}", True)
    assert (line.output_text, line.served_model, line.extra) == (None, None, {})
    assert line.usage == dict.fromkeys(USAGE_KEYS, 0)
    assert (line.request_id, line.response_id) == (f"req-{status}", f"req-{status}")
    assert line.request_sha256 == request.sha256()
    assert solution not in line.to_json() and len(line.to_json()) < 1_000


def test_no_error_text_reaches_a_line(server):
    error = server.sdk.TypeSafeUnprocessableEntityError(422, request_id="req-1", body=ECHOED)
    assert ECHOED in str(error)  # the SDK's message carries the body
    server.outcomes.append(error)

    [line] = ask_jev(REQUEST)

    assert ECHOED not in line.to_json()


@pytest.mark.parametrize(
    ("error_name", "status"),
    [
        ("TypeSafeAPITimeoutError", None),
        ("TypeSafeAPIConnectionError", None),
        ("TypeSafeAPIError", 408),
        ("TypeSafeAPIError", 409),
        ("TypeSafeRateLimitError", 429),
        ("TypeSafeInternalServerError", 500),
        ("TypeSafeInternalServerError", 529),
    ],
)
def test_transient_failures_raise_transient_api_error_without_the_error_text(
    server, error_name, status
):
    # No line is written: the next run sends the request again. A 529 has no class of its own.
    error_class = getattr(server.sdk, error_name)
    if status is None:
        error = error_class(f"Connection error: {ECHOED}")
    else:
        error = error_class(status, request_id="req-t", body=ECHOED)
    server.outcomes.append(error)

    with pytest.raises(TransientApiError) as raised:
        ask_jev(REQUEST)

    kind = error_name if status is None else f"{error_name} {status}"
    assert str(raised.value) == (
        "TypeSafe did not answer the request of problem 5 of jev-statement after the SDK's "
        f"retries (at most 5): {kind}."
    )
    assert raised.value.__cause__ is error
    assert server.closed == 1
