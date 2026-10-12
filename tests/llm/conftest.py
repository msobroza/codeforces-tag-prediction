"""Fixtures shared by the LLM toolkit's tests: builders of the objects a request is made of."""

import asyncio
import sys
import types
from contextlib import asynccontextmanager

import pytest

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.llm import providers
from codeforces_tags.llm.prompts import (
    LlmMethodSpec,
    LlmRequest,
    PromptSnapshot,
    prompt_content_sha256,
)
from codeforces_tags.llm.settings import LlmSettings


@pytest.fixture(name="make_settings")
def make_settings_fixture():
    """Return a builder of the settings of a made-up model, with fields overridable by keyword."""

    def make(**changes):
        values = {
            "provider": "openai",
            "model": "model-1",
            "reasoning_effort": "none",
            "temperature": 0.3157894736842105,
            "top_p": 1.0,
            "max_output_tokens": 512,
            "output_format": "text",
        }
        return LlmSettings(**(values | changes))

    return make


@pytest.fixture(name="make_request")
def make_request_fixture(make_settings):
    """Return a builder of the request of problem 1, with fields overridable by keyword.

    ``make_request("json")`` gives a request of the ``json`` output format, with a schema.
    """

    def make(output_format="text", **changes):
        json_format = output_format == "json"
        values = {
            "sample_id": 1,
            "method": f"fake-{output_format}",
            "input": "Tag this.",
            "settings": make_settings(output_format=output_format),
            "schema": {"type": "object", "required": ["tags"]} if json_format else None,
            "schema_name": "problem_tags" if json_format else None,
        }
        return LlmRequest(**(values | changes))

    return make


@pytest.fixture(name="requests_for")
def requests_for_fixture(make_request):
    """Return a builder of the requests of problems 1 to n, each with an input of its own."""

    def build(count):
        return [make_request(sample_id=n, input=f"Tag problem {n}.") for n in range(1, count + 1)]

    return build


@pytest.fixture(name="fake_provider")
def fake_provider_fixture(monkeypatch):
    """Register the provider ``"fake"``, whose adapter is the module ``fake_provider``.

    Its ``request_body`` keeps the request's input, its ``ask`` returns what it was given, and
    its SDK is ``fake_sdk``. Both are forgotten after the test.
    """
    module = types.ModuleType("fake_provider")
    module.SDK_MODULE = "fake_sdk"
    module.request_body = lambda request: {"input": request.input}
    module.ask = lambda settings: ("the fake ask of", settings)
    monkeypatch.setitem(sys.modules, "fake_provider", module)
    monkeypatch.setitem(providers.PROVIDERS, "fake", "fake_provider")
    return module


class InFlight:  # pylint: disable=too-few-public-methods
    """Wraps a fake Ask to count its requests in flight: each one lets the loop run once.

    Attributes:
        most: The most requests that were in flight at once.
    """

    def __init__(self, ask):
        """Wrap a fake Ask, with no request in flight yet."""
        self.ask = ask
        self.now = 0
        self.most = 0

    async def __call__(self, request):
        """Count the request as in flight, let the other requests run, then answer it."""
        self.now += 1
        self.most = max(self.most, self.now)
        await asyncio.sleep(0)
        self.now -= 1
        return await self.ask(request)


@pytest.fixture(name="in_flight")
def in_flight_fixture():
    """Return ``InFlight``: ``in_flight(ask).most`` counts the requests ``ask`` had at once."""
    return InFlight


class OpenableAsk:
    """An adapter's Ask: ``open`` opens a client in the running loop, and ``send`` answers.

    Attributes:
        events: What happened, in order: ``"open"``, each request's sample id, ``"close"``.
    """

    def __init__(self, answer):
        """Answer each request with ``answer(request)``, through a client not opened yet."""
        self.answer = answer
        self.events = []

    @asynccontextmanager
    async def open(self):
        """Open the client, give the Ask that sends through it, then close it."""
        self.events.append("open")
        yield self.send
        self.events.append("close")

    async def send(self, request):
        """Answer a request through the open client."""
        self.events.append(request.sample_id)
        return self.answer(request)


@pytest.fixture(name="openable_ask")
def openable_ask_fixture():
    """Return ``OpenableAsk``: ``openable_ask(answer).events`` records its client and requests."""
    return OpenableAsk


@pytest.fixture(name="make_llm_spec")
def make_llm_spec_fixture(make_settings):
    """Return a builder of the spec of ``prompted-<output_format>``, fields overridable by keyword.

    The method reads a problem's description and solution, and chooses among every tag of
    ``ALL_TAGS``; ``make_llm_spec("json")`` gives the ``json`` method, with a schema. Its paths
    name no file: tests build its prompt with ``snapshot_of``.
    """

    def make(output_format="text", **changes):
        json_format = output_format == "json"
        values = {
            "name": f"prompted-{output_format}",
            "prompt_name": "prompted",
            "template_path": "tags.j2",
            "schema_path": "tags.schema.json" if json_format else None,
            "schema_name": "problem_tags" if json_format else None,
            "fields": ("prob_desc_description", "source_code"),
            "constants": {},
            "settings": make_settings(output_format=output_format),
            "all_tags": ALL_TAGS,
        }
        return LlmMethodSpec(**(values | changes))

    return make


@pytest.fixture(name="snapshot_of")
def snapshot_of_fixture():
    """Return a builder of a spec's prompt snapshot, from a template and a schema.

    The snapshot is the one ``prompt_snapshot(spec)`` would read from the spec's package: its
    name is the spec's prompt name, and its content hash covers the template, the schema, and the
    spec's settings and extra.
    """

    def build(spec, template, response_format=None):
        content = prompt_content_sha256(template, response_format, spec.settings, spec.extra)
        return PromptSnapshot(spec.prompt_name, None, None, content, template, response_format)

    return build
