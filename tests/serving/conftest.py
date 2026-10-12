"""Fixtures for the serving tests: MLflow isolated in each test, and fake tagging methods.

Every test here runs with MLflow's environment cleared and restored, and a store under
``tmp_path`` until it chooses one (``mlflow_environment``, in ``tests/conftest.py``): the code it
runs writes MLflow's variables (``mlflow_env``). The fakes are the conforming methods of
``tests/methods/fakes/conformance_fakes.py``.

``stub_server`` stands in for a model server (or MLflow's UI) on ``127.0.0.1``: it answers each
method and path with what the test gives it, and records every request. ``refused_url`` is the
URL of a port of ``127.0.0.1`` where nothing listens. ``heavy_after_import`` tells which heavy
libraries importing a module loads, in a fresh interpreter.
"""

import importlib
import json
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

FAKES = Path(__file__).parents[1] / "methods" / "fakes"

# What a request the test gave no answer for gets: nginx's answer to a path of no location in
# MLflow's conf.
NOT_FOUND = (404, "application/json", b"{}")
# The most seconds a request that hangs waits, should its test never end it.
HANG_S = 30
# How often the server's thread checks for its shutdown: teardown waits up to this long.
POLL_S = 0.05
# The libraries that no module of the prediction path imports: HEAVY_MODULES of
# tests/methods/test_registry.py, which checks the core.
HEAVY_MODULES = ("mlflow", "optuna", "gepa", "openai", "torch", "typesafe_sdk")
# Imports the module argv[1], then prints, as JSON, which names of the JSON list argv[2] the
# interpreter has loaded.
HEAVY_AFTER_IMPORT = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(set(json.loads(sys.argv[2])) & set(sys.modules))))
"""


@pytest.fixture(autouse=True)
def isolate_mlflow(mlflow_environment):  # pylint: disable=unused-argument
    """Isolate MLflow's environment and default store in every serving test."""


@pytest.fixture(name="fakes")
def fakes_fixture(monkeypatch):
    """Return the module ``conformance_fakes``, importable by that name while the test runs."""
    monkeypatch.syspath_prepend(str(FAKES))
    return importlib.import_module("conformance_fakes")


@dataclass(frozen=True)
class StubRequest:
    """One request that a stub server received.

    Attributes:
        method: ``GET`` or ``POST``.
        path: The path, with its query string.
        content_type: The ``Content-Type`` header, or None.
        body: The body's bytes (empty without a body).
    """

    method: str
    path: str
    content_type: str | None
    body: bytes


@dataclass
class StubServer:
    """Answers each method and path of ``url`` as a test says, and records every request.

    Attributes:
        url: ``http://127.0.0.1:<port>``.
        answers: The status, content type and body of each (method, path); None for one that
            hangs.
        requests: Every request received, in order.
        released: Set when the test ends: a request that hangs then gets no answer.
    """

    url: str
    answers: dict[tuple[str, str], tuple[int, str, bytes] | None] = field(default_factory=dict)
    requests: list[StubRequest] = field(default_factory=list)
    released: threading.Event = field(default_factory=threading.Event)

    def answer(self, method, path, status=200, body=b"", content_type="application/json"):
        """Answer every ``method`` request to ``path`` the same way, until another answer.

        Args:
            method: ``GET`` or ``POST``.
            path: The path, with its query string, e.g. ``/invocations``.
            status: The HTTP status.
            body: Bytes, sent as they are; a text, sent as its UTF-8 bytes; or any other value,
                sent as its JSON text (an infinite float as ``Infinity``, as MLflow writes it).
            content_type: The ``Content-Type`` header.
        """
        if isinstance(body, str):
            body = body.encode("utf-8")
        elif not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        self.answers[method, path] = (status, content_type, body)

    def hang(self, method, path):
        """Answer no ``method`` request to ``path``: each waits until the test ends, unanswered."""
        self.answers[method, path] = None


class StubHandler(BaseHTTPRequestHandler):
    """Records each request, then answers it as its server's ``StubServer`` says."""

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        """Write no access-log line to stderr: the server records requests in ``requests``.

        Args:
            format: The log line's format string.
            *args: Its values.
        """

    def reply(self):
        """Record the request, then send the answer of its method and path (else nginx's 404)."""
        stub = self.server.stub
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        content_type = self.headers.get("Content-Type")
        stub.requests.append(StubRequest(self.command, self.path, content_type, body))
        answer = stub.answers.get((self.command, self.path), NOT_FOUND)
        if answer is None:
            stub.released.wait(HANG_S)
            return
        status, answer_type, payload = answer
        self.send_response(status)
        self.send_header("Content-Type", answer_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = reply


class StubHTTPServer(ThreadingHTTPServer):
    """A threading HTTP server on a free port of 127.0.0.1, answering as its ``stub`` says."""

    def __init__(self):
        """Bind a free port of 127.0.0.1, and make the ``StubServer`` of its URL."""
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.stub = StubServer(f"http://127.0.0.1:{self.server_address[1]}")


@pytest.fixture(name="stub_server")
def stub_server_fixture():
    """Serve a ``StubServer`` on 127.0.0.1 in a thread while the test runs, and return it.

    Teardown ends the requests that hang, then stops the server and waits for its thread.
    """
    server = StubHTTPServer()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": POLL_S}, daemon=True
    )
    thread.start()
    yield server.stub
    server.stub.released.set()
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture(name="refused_url")
def refused_url_fixture():
    """Return the URL of a port of 127.0.0.1 where nothing listens, while the test runs.

    The port stays bound, never listened on, so no other process takes it: a connection to it is
    refused at once. A port bound then closed could be taken again before the test connects.
    """
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        yield f"http://127.0.0.1:{bound.getsockname()[1]}"


@pytest.fixture(name="heavy_after_import")
def heavy_after_import_fixture(agent_python):
    """Return a function that tells which heavy libraries importing a module loads.

    It takes a module's name, imports the module in a fresh interpreter (``agent_python``) and
    returns the names of ``HEAVY_MODULES`` that the interpreter then holds, sorted: ``[]`` for a
    module that imports neither mlflow, nor torch, nor any other of them.
    """

    def heavy_after_import(module):
        output, _ = agent_python(HEAVY_AFTER_IMPORT, module, json.dumps(HEAVY_MODULES))
        return json.loads(output)

    return heavy_after_import
