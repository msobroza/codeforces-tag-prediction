"""Tests for the HTTP client of a model server: the payload, the answers and every failure."""

import http.client
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.scoring_server import invocations

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import (
    FOCUS_TAGS,
    RAW_INPUT_COLUMNS,
    SERVING_DEFAULT_URL,
    SERVING_URL_ENV_VAR,
)
from codeforces_tags.serving.client import (
    ServedPrediction,
    ServingError,
    ping,
    predict,
    request_payload,
    server_url,
)

# nginx's page while uvicorn behind it is not listening yet (the method still loads).
NGINX_502 = (
    b"<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n"
    b"<center><h1>502 Bad Gateway</h1></center>\r\n<hr><center>nginx/1.22.1</center>\r\n"
    b"</body>\r\n</html>\r\n"
)
# The columns of raw_problems that are not raw input columns: never sent.
NOT_SENT = ("tags", "fold", "group")


def answer_record(method, tags, **extra):
    """Return one record of a model server's answer: the method, its 0/1 focus tags, ``extra``."""
    return {"method": method} | {tag: int(tag in tags) for tag in FOCUS_TAGS} | extra


def scored(scores, thresholds):
    """Return the ``score: <tag>`` and ``threshold: <tag>`` fields of every focus tag."""
    return {f"score: {tag}": scores[tag] for tag in FOCUS_TAGS} | {
        f"threshold: {tag}": thresholds[tag] for tag in FOCUS_TAGS
    }


def test_server_url_takes_the_given_url_first(monkeypatch):
    monkeypatch.setenv(SERVING_URL_ENV_VAR, "http://127.0.0.1:18080")

    assert server_url("http://127.0.0.1:9000") == "http://127.0.0.1:9000"


@pytest.mark.parametrize("given", [None, ""])
def test_server_url_takes_the_variable_without_a_given_url(monkeypatch, given):
    monkeypatch.setenv(SERVING_URL_ENV_VAR, "http://127.0.0.1:18080")

    assert server_url(given) == "http://127.0.0.1:18080"


@pytest.mark.parametrize("variable", [None, ""])
def test_server_url_falls_back_on_the_default_when_the_variable_is_unset_or_empty(
    monkeypatch, variable
):
    monkeypatch.delenv(SERVING_URL_ENV_VAR, raising=False)
    if variable is not None:
        monkeypatch.setenv(SERVING_URL_ENV_VAR, variable)

    assert server_url() == SERVING_DEFAULT_URL == "http://127.0.0.1:8080"


def test_request_payload_sends_null_for_nan_none_and_blank_optional_fields(raw_problems):
    # Problem 30 has no notes and problem 20 no input specification (NaN, as read_samples gives
    # a field a sample lacks); a None, a blank and an empty text are the other missing texts.
    problems = raw_problems.assign(prob_desc_output_spec=["One integer.", " \r\n", None])
    problems.loc[10, "prob_desc_notes"] = ""

    payload = request_payload(problems)
    sent = json.dumps(payload, allow_nan=False)

    assert payload["dataframe_split"]["columns"] == list(RAW_INPUT_COLUMNS)
    assert payload["dataframe_split"]["data"] == [
        [
            "Count the leaves of a tree.",
            "A tree.",
            "One integer.",
            None,
            '["3\\n1 2\\n1 3"]',
            '["2"]',
            "n = int(input())\nprint(n - 1)\n",
        ],
        [
            "Reverse each of the strings.",
            "Some strings.",
            None,
            None,
            '["abc"]',
            '["cba"]',
            "s = input()\nprint(s[::-1])\n",
        ],
        [
            "Print the gcd, a number theory classic.",
            None,
            None,
            "NoteUse math.gcd.",
            '["4 6"]',
            '["2"]',
            "print(2)",
        ],
    ]
    assert sent.count("null") == 5
    assert "NaN" not in sent and '"nan"' not in sent.lower() and '""' not in sent
    assert not any(column in sent for column in NOT_SENT)


def test_request_payload_keeps_unicode_and_math_verbatim(stub_server):
    texts = {
        "prob_desc_description": "Найдите $$$\\sum_{i=1}^{n} a_i$$$ — ünïcödé 数学\r\nAgain.  ",
        "prob_desc_input_spec": "\tThe first line holds $$$n$$$ ($$$1 \\le n \\le 10^5$$$).\r\n",
        "prob_desc_output_spec": "Print «one» integer.",
        "prob_desc_notes": "Note: $$$$$$x^2$$$$$$\r\n",
        "source_code": "s = input()  # é\r\nprint(s)\r\n",
    }
    samples = {"prob_desc_sample_inputs": ["1\r\n4 é"], "prob_desc_sample_outputs": '["2"]'}
    problems = pd.DataFrame([texts | samples], index=pd.Index([7], name="sample_id"))
    stub_server.answer("POST", "/invocations", body={"predictions": [answer_record("tfidf", ())]})

    payload = request_payload(problems)
    predict(stub_server.url, problems)

    [row] = payload["dataframe_split"]["data"]
    sent = dict(zip(RAW_INPUT_COLUMNS, row, strict=True))
    assert {column: sent[column] for column in texts} == texts
    assert json.loads(sent["prob_desc_sample_inputs"]) == ["1\r\n4 é"]
    assert sent["prob_desc_sample_outputs"] == '["2"]'
    [request] = stub_server.requests
    assert request.body == json.dumps(payload).encode("ascii")
    assert json.loads(request.body) == payload


def test_request_payload_sends_a_column_the_problems_lack_as_null(raw_problems):
    problems = raw_problems.drop(columns=["prob_desc_notes"])

    payload = request_payload(problems)

    notes = RAW_INPUT_COLUMNS.index("prob_desc_notes")
    assert [row[notes] for row in payload["dataframe_split"]["data"]] == [None, None, None]


def test_request_payload_sends_the_columns_it_is_given_in_their_order(raw_problems):
    payload = request_payload(raw_problems, columns=("source_code", "prob_desc_description"))

    assert payload["dataframe_split"]["columns"] == ["source_code", "prob_desc_description"]
    assert payload["dataframe_split"]["data"][2] == [
        "print(2)",
        "Print the gcd, a number theory classic.",
    ]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("source_code", "  ", r"needs a text in prob_desc_description and source_code"),
        ("prob_desc_sample_inputs", "1 2", r"prob_desc_sample_inputs holds neither a list"),
        ("prob_desc_notes", 3.5, r"prob_desc_notes holds neither a text nor null"),
    ],
)
def test_request_payload_refuses_what_the_command_line_refuses(
    raw_problems, column, value, message
):
    problems = raw_problems.astype({column: object})
    problems.loc[20, column] = value

    with pytest.raises(CliError, match=message):
        request_payload(problems)


def test_request_payload_refuses_a_column_that_is_not_a_raw_input_column(raw_problems):
    with pytest.raises(ValueError, match=r"\['tags'\] are not raw input columns"):
        request_payload(raw_problems, columns=("source_code", "tags"))


def test_predict_reads_tags_scores_and_thresholds_with_an_infinite_threshold(
    stub_server, raw_problems
):
    scores = dict(zip(FOCUS_TAGS, np.linspace(-1.0, 0.75, len(FOCUS_TAGS)), strict=True))
    thresholds = dict.fromkeys(FOCUS_TAGS, 0.25) | {"games": math.inf}
    records = [
        answer_record("f2llm_late_fusion", ("math", "trees"), **scored(scores, thresholds)),
        answer_record("f2llm_late_fusion", (), **scored(scores, thresholds)),
        answer_record("f2llm_late_fusion", ("strings",), **scored(scores, thresholds)),
    ]
    stub_server.answer("POST", "/invocations", body={"predictions": records})

    predictions = predict(stub_server.url + "/", raw_problems)

    assert [prediction.tags for prediction in predictions] == [
        ("math", "trees"),
        (),
        ("strings",),
    ]
    first = predictions[0]
    assert first == ServedPrediction(
        "f2llm_late_fusion", ("math", "trees"), scores, thresholds, None, None, first.seconds
    )
    assert first.thresholds["games"] == math.inf
    assert first.seconds > 0
    assert {prediction.seconds for prediction in predictions} == {first.seconds}
    [request] = stub_server.requests
    assert (request.method, request.path, request.content_type) == (
        "POST",
        "/invocations",
        "application/json",
    )
    assert json.loads(request.body) == request_payload(raw_problems)


@pytest.mark.parametrize(
    ("tags", "failed", "cost_usd"),
    [((), True, 0.0), (("math", "number theory"), False, 0.000133)],
)
def test_predict_reads_the_failure_and_the_cost_of_a_method_that_calls_an_api(
    stub_server, raw_problems, tags, failed, cost_usd
):
    # A refused request failed and cost nothing; a completed answer was billed its tokens.
    record = answer_record("jev-statement-code", tags, failed=failed, cost_usd=cost_usd)
    stub_server.answer("POST", "/invocations", body={"predictions": [record]})

    [prediction] = predict(stub_server.url, raw_problems.loc[[10]])

    assert (prediction.tags, prediction.scores, prediction.thresholds) == (tags, {}, {})
    assert (prediction.failed, prediction.cost_usd) == (failed, cost_usd)


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (
            400,
            {
                "error_code": "BAD_REQUEST",
                "message": "Encountered an unexpected error while evaluating the model.",
                "stack_trace": "Traceback (most recent call last):\n  File ...",
            },
            "answered 400 Bad Request: Encountered an unexpected error while evaluating the model.",
        ),
        (
            500,
            {
                "error_code": "INTERNAL_ERROR",
                "message": "Failed to predict data '   prob_desc_description\n0  Count the "
                "leaves'. \nError: an API method takes one problem per request",
            },
            "answered 500 Internal Server Error: an API method takes one problem per request",
        ),
        (
            400,
            {"error_code": "BAD_REQUEST", "message": "Invalid input.\nInput: \nb'{'\n"},
            "answered 400 Bad Request: Invalid input.",
        ),
    ],
)
def test_predict_keeps_the_first_line_of_mlflows_json_message(
    stub_server, raw_problems, status, body, message
):
    stub_server.answer("POST", "/invocations", status, body)

    with pytest.raises(ServingError) as raised:
        predict(stub_server.url, raw_problems)

    assert raised.value.status == status
    assert raised.value.message == f"the model server {message}"
    assert str(raised.value) == raised.value.message
    assert "Traceback" not in raised.value.message and "\n" not in raised.value.message


def test_predict_keeps_the_methods_own_message_of_an_error_that_mlflow_wraps(
    stub_server, raw_problems
):
    # The locked MLflow writes the problems' frame into the message of an error the method raised,
    # as the entry file raises a CliError: the client keeps the method's own message only.
    def refuse(data, params=None):
        raise MlflowException("an API method takes one problem per request")

    model = SimpleNamespace(metadata=SimpleNamespace(), predict=refuse)
    with pytest.raises(MlflowException) as wrapped:
        invocations(json.dumps(request_payload(raw_problems)), "application/json", model, None)
    error = wrapped.value
    stub_server.answer(
        "POST", "/invocations", error.get_http_status_code(), error.serialize_as_json()
    )

    with pytest.raises(ServingError) as raised:
        predict(stub_server.url, raw_problems)

    assert error.message.startswith("Failed to predict data '")
    assert raised.value.message == (
        "the model server answered 500 Internal Server Error: "
        "an API method takes one problem per request"
    )


@pytest.mark.parametrize(
    ("answer", "status_line"),
    [
        (
            (415, "This predictor only supports the following content types", "text/plain"),
            "415 Unsupported Media Type",
        ),
        (
            (504, b"Request processing time exceeded limit", "application/json"),
            "504 Gateway Timeout",
        ),
        ((404, b"{}", "application/json"), "404 Not Found"),
        ((500, b'{"message": ""}', "application/json"), "500 Internal Server Error"),
    ],
)
def test_predict_falls_back_on_the_status_line_without_a_json_message(
    stub_server, raw_problems, answer, status_line
):
    # MLflow's 415 is plain text; its own 504 says application/json but is not JSON; nginx answers
    # {} to any other path; a blank message says nothing.
    stub_server.answer("POST", "/invocations", *answer)

    with pytest.raises(ServingError) as raised:
        predict(stub_server.url, raw_problems)

    assert raised.value.status == answer[0]
    assert raised.value.message == f"the model server answered {status_line}"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("html_502", (502, "the model server answered 502 Bad Gateway")),
        ("refused", (None, "Connection refused")),
        ("timeout", (None, "timed out")),
    ],
)
def test_predict_raises_serving_error_on_html_502_refused_connection_and_timeout(
    stub_server, refused_url, raw_problems, case, expected
):
    # While the method loads, nginx answers its HTML page; a stopped server refuses; a busy one
    # keeps the request past the timeout (the path under /slow hangs).
    stub_server.answer("POST", "/invocations", 502, NGINX_502, "text/html")
    stub_server.hang("POST", "/slow/invocations")
    url = {
        "html_502": stub_server.url,
        "refused": refused_url,
        "timeout": stub_server.url + "/slow",
    }

    with pytest.raises(ServingError) as raised:
        predict(url[case], raw_problems, timeout=1.0)

    status, words = expected
    assert raised.value.status == status
    assert words in raised.value.message
    assert "<" not in raised.value.message and "Traceback" not in raised.value.message


def test_predict_keeps_one_line_when_the_peer_does_not_answer_http(monkeypatch, raw_problems):
    # A port of another service answers its banner, which http.client quotes, line end included.
    def banner(request, timeout):
        raise http.client.BadStatusLine("SSH-2.0-OpenSSH_8.9\r\n")

    monkeypatch.setattr("codeforces_tags.serving.client._open", banner)

    with pytest.raises(ServingError) as raised:
        predict("http://127.0.0.1:22", raw_problems)

    assert raised.value.status is None
    assert raised.value.message == (
        "no answer from the model server at http://127.0.0.1:22/invocations: SSH-2.0-OpenSSH_8.9"
    )


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"<html>not JSON</html>", "JSONDecodeError"),
        (b'{"outputs": []}', "KeyError"),
        (b'{"predictions": [["tfidf", 1]]}', "AttributeError"),
        (b'{"predictions": [{"method": "tfidf", "math": 1}]}', "KeyError"),
    ],
)
def test_predict_refuses_a_200_whose_body_it_cannot_read(stub_server, raw_problems, body, reason):
    stub_server.answer("POST", "/invocations", 200, body)

    with pytest.raises(ServingError) as raised:
        predict(stub_server.url, raw_problems.loc[[10]])

    assert raised.value.status == 200
    assert f"cannot be read ({reason})" in raised.value.message


def test_predict_refuses_an_answer_with_another_count_of_predictions(stub_server, raw_problems):
    stub_server.answer("POST", "/invocations", body={"predictions": [answer_record("tfidf", ())]})

    with pytest.raises(ServingError) as raised:
        predict(stub_server.url, raw_problems)

    assert raised.value.status == 200
    assert "does not hold one prediction per problem: 1 for 3" in raised.value.message


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            "127.0.0.1:8080",
            "'127.0.0.1:8080' is not the URL of a model server: give http://<host>:<port>.",
        ),
        (
            "http://[::1:8080",
            "no answer from the model server at http://[::1:8080/invocations: Invalid IPv6 URL",
        ),
        (
            "http://127.0.0.1:80a",
            "no answer from the model server at http://127.0.0.1:80a/invocations: nonnumeric "
            "port: '80a'",
        ),
    ],
)
def test_predict_refuses_a_url_it_cannot_use_before_sending_anything(raw_problems, url, message):
    # The client refuses a URL without http://; urllib cannot parse the second (a ValueError),
    # and http.client cannot use the third (InvalidURL).
    with pytest.raises(ServingError) as raised:
        predict(url, raw_problems)

    assert raised.value.status is None
    assert raised.value.message == message


def test_predict_refuses_what_the_command_line_refuses_before_sending_anything(
    stub_server, raw_problems
):
    with pytest.raises(CliError):
        predict(stub_server.url, raw_problems.drop(columns=["source_code"]))

    assert stub_server.requests == []


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("ready", True),
        ("loading", False),
        ("refused", False),
        ("busy", False),
        ("no_url", False),
        ("bad_url", False),
    ],
)
def test_ping_is_true_only_once_the_model_server_answers_200(
    stub_server, refused_url, case, expected
):
    stub_server.answer("GET", "/ping", 200, b"\n")
    stub_server.answer("GET", "/loading/ping", 502, NGINX_502, "text/html")
    stub_server.hang("GET", "/busy/ping")
    url = {
        "ready": stub_server.url,
        "loading": stub_server.url + "/loading",
        "refused": refused_url,
        "busy": stub_server.url + "/busy",
        "no_url": "localhost:8080",
        "bad_url": "http://[::1:8080",
    }

    assert ping(url[case], timeout=1.0) is expected


def test_predict_and_ping_never_go_through_a_proxy(
    monkeypatch, stub_server, refused_url, raw_problems
):
    # Through the proxy of the environment the connection would be refused, and the statements
    # and solutions would reach it.
    for variable in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(variable, refused_url)
    for variable in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(variable, raising=False)
    stub_server.answer("GET", "/ping", 200, b"\n")
    stub_server.answer(
        "POST", "/invocations", body={"predictions": [answer_record("tfidf", ("math",))] * 3}
    )

    predictions = predict(stub_server.url, raw_problems)

    assert [prediction.tags for prediction in predictions] == [("math",)] * 3
    assert ping(stub_server.url) is True
    assert [request.path for request in stub_server.requests] == ["/invocations", "/ping"]


def test_importing_the_client_loads_no_heavy_library(heavy_after_import):
    assert heavy_after_import("codeforces_tags.serving.client") == []
