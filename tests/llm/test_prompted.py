"""Tests for PromptedLlm, the tagging method that asks an LLM, and its method folder."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager

import pandas as pd
import pytest

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.llm import client, prompted
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.prompted import PROMPTED_FILE, PromptedLlm, load_prompted
from codeforces_tags.llm.prompts import build_requests
from codeforces_tags.methods import METHODS, MethodSpec, TaggingMethod
from codeforces_tags.preprocessing.pipeline import raw_to_problems

TEMPLATE = (
    "Tag this problem.\n"
    "Description: {{ prob_desc_description }}\n"
    "\n"
    "{{ source_code }}\n"
    "\n"
    'Choose from: {{ all_tags | join(", ") }}'
)
SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string", "enum": list(ALL_TAGS)}}},
    "required": ["tags"],
    "additionalProperties": False,
}
# A word of each problem of the raw_problems fixture, and the tag a model would answer for it.
KEYWORDS = {"leaves": "trees", "reverse": "strings", "gcd": "number theory"}
# The module of the made-up prompted package that the conformance check loads the method with.
MODULE = "prompted_fake"


@pytest.fixture(name="keyword_ask")
def keyword_ask_fixture(make_answer):
    """Return a fake Ask that answers the tags whose keyword the request's prompt holds."""

    async def ask(request):
        text = request.input.lower()
        answer = ", ".join(tag for word, tag in KEYWORDS.items() if word in text)
        return make_answer(
            sample_id=request.sample_id, request_sha256=request.sha256(), output_text=answer
        )

    return ask


@pytest.fixture(name="method")
def method_fixture(make_llm_spec, snapshot_of, keyword_ask):
    """Return the ``text`` prompted method, asking through ``keyword_ask``."""
    spec = make_llm_spec()
    return PromptedLlm(spec, snapshot_of(spec, TEMPLATE), keyword_ask)


def tagged(index, tags_by_problem):
    """Return the 0/1 frame over ALL_TAGS that gives each problem the tags listed for it."""
    frame = pd.DataFrame(0, index=index, columns=list(ALL_TAGS))
    for sample_id, tags in tags_by_problem.items():
        frame.loc[sample_id, list(tags)] = 1
    return frame


def test_a_prompted_method_is_a_tagging_method_over_the_tags_of_its_spec(method):
    assert isinstance(method, TaggingMethod)
    assert (method.name, method.tags) == ("prompted-text", ALL_TAGS)


def test_a_zero_shot_prompted_method_has_no_request_context(method, raw_problems):
    assert method.request_context(raw_problems) is None


def test_requests_render_the_preprocessed_problems_in_their_order(method, raw_problems):
    requests = method.requests(raw_problems)

    assert requests == build_requests(raw_to_problems(raw_problems), method.spec, method.prompt)
    assert [request.sample_id for request in requests] == [30, 10, 20]
    assert requests[0].input.startswith("Tag this problem.\nDescription: Count the leaves")


def test_requests_refuse_a_prompt_that_is_not_the_one_of_the_spec(
    make_llm_spec, snapshot_of, keyword_ask, raw_problems
):
    spec = make_llm_spec()
    method = PromptedLlm(
        spec, snapshot_of(make_llm_spec(prompt_name="other"), TEMPLATE), keyword_ask
    )

    with pytest.raises(ValueError, match="is not the prompt of prompted-text"):
        method.requests(raw_problems)


def test_predict_encodes_the_answers_as_0_1_tags_with_the_parser_of_the_output_format(
    method, raw_problems
):
    predictions = method.predict(raw_problems)

    expected = tagged(raw_problems.index, {30: ["trees"], 10: ["strings"], 20: ["number theory"]})
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_of_a_json_method_reads_its_json_answers(
    make_llm_spec, snapshot_of, make_answer, raw_problems
):
    spec = make_llm_spec("json")

    async def ask(request):
        tags = ["math", "math"] if request.sample_id == 20 else []
        return make_answer(sample_id=request.sample_id, output_text=json.dumps({"tags": tags}))

    method = PromptedLlm(spec, snapshot_of(spec, TEMPLATE, SCHEMA), ask)

    predictions = method.predict(raw_problems)

    pd.testing.assert_frame_equal(predictions, tagged(raw_problems.index, {20: ["math"]}))


def test_a_failed_answer_predicts_no_tag(method, make_answer, raw_problems):
    async def ask(request):
        return make_answer(sample_id=request.sample_id, status="incomplete", output_text="trees")

    method.ask = ask

    assert method.predict(raw_problems).to_numpy().sum() == 0


def test_apredict_with_answers_gives_the_answers_in_the_order_of_the_problems(method, raw_problems):
    predictions, answers = asyncio.run(method.apredict_with_answers(raw_problems))

    assert [answer.sample_id for answer in answers] == [30, 10, 20]
    assert [answer.output_text for answer in answers] == ["trees", "strings", "number theory"]
    pd.testing.assert_frame_equal(predictions, method.predict(raw_problems))


def test_apredict_sends_at_most_llm_concurrency_requests_at_once(
    method, keyword_ask, raw_problems, monkeypatch, in_flight
):
    monkeypatch.setattr(prompted, "LLM_CONCURRENCY", 2)
    method.ask = in_flight(keyword_ask)

    asyncio.run(method.apredict(raw_problems))

    assert method.ask.most == 2


def test_predict_lets_a_transient_error_through(method, keyword_ask, raw_problems):
    # A live prediction has no answers file to resume from: the error is the caller's.
    async def ask(request):
        if request.sample_id == 10:
            raise TransientApiError("The provider did not answer.")
        return await keyword_ask(request)

    method.ask = ask

    with pytest.raises(TransientApiError, match="did not answer"):
        method.predict(raw_problems)


def test_predict_without_an_ask_builds_a_fresh_adapter_in_each_event_loop(
    make_llm_spec, snapshot_of, keyword_ask, raw_problems, monkeypatch
):
    # Two predict calls in one process run in two event loops: each builds its adapter with
    # client.ask_for and opens its client in its own loop, as the OpenAI adapter must.
    spec = make_llm_spec()
    adapters = []

    class LoopBoundAsk:  # pylint: disable=too-few-public-methods
        """An adapter's Ask: its client opens in the running loop and works only there."""

        def __init__(self, settings):
            """Keep the settings; the loops it was opened and closed in are recorded."""
            self.settings = settings
            self.opened = []
            self.closed = []

        @asynccontextmanager
        async def open(self):
            """Open the client in the running loop, and close it there."""
            loop = asyncio.get_running_loop()
            self.opened.append(loop)

            async def send(request):
                assert asyncio.get_running_loop() is loop
                return await keyword_ask(request)

            yield send
            self.closed.append(loop)

    def ask_for(settings):
        adapters.append(LoopBoundAsk(settings))
        return adapters[-1]

    monkeypatch.setattr(client, "ask_for", ask_for)
    method = PromptedLlm(spec, snapshot_of(spec, TEMPLATE))

    first, second = method.predict(raw_problems), method.predict(raw_problems)

    pd.testing.assert_frame_equal(first, second)
    assert [adapter.settings for adapter in adapters] == [spec.settings, spec.settings]
    assert [len(adapter.opened) for adapter in adapters] == [1, 1]
    assert [adapter.closed for adapter in adapters] == [adapter.opened for adapter in adapters]
    assert adapters[0].opened[0] is not adapters[1].opened[0]


def test_predict_inside_a_running_loop_points_to_apredict(method, raw_problems):
    # A notebook (Jupyter) runs its cells inside an event loop, where predict cannot start one.
    async def notebook_cell():
        with pytest.raises(RuntimeError, match=r"await method\.apredict\(raw_problems\)"):
            method.predict(raw_problems)
        return await method.apredict(raw_problems)

    predictions = asyncio.run(notebook_cell())

    pd.testing.assert_frame_equal(predictions, method.predict(raw_problems))


def test_save_writes_the_spec_and_the_prompt_without_an_absolute_path(method, tmp_path):
    method.save(tmp_path)

    text = (tmp_path / PROMPTED_FILE).read_text(encoding="utf-8")
    assert json.loads(text) == {"spec": method.spec.to_dict(), "prompt": method.prompt.to_dict()}
    assert str(tmp_path) not in text
    assert [path.name for path in tmp_path.iterdir()] == [PROMPTED_FILE]


def test_load_prompted_gives_back_the_spec_and_the_prompt_with_no_ask(
    make_llm_spec, snapshot_of, tmp_path
):
    spec = make_llm_spec("json", extra={"note": "an extra value"})
    method = PromptedLlm(spec, snapshot_of(spec, TEMPLATE, SCHEMA))
    method.save(tmp_path)

    loaded = load_prompted(tmp_path)

    assert (loaded.spec, loaded.prompt, loaded.ask) == (spec, method.prompt, None)


@pytest.fixture(name="registry")
def registry_fixture(tmp_path, monkeypatch):
    """Write a made-up prompted package's module, and return a registry of its method.

    Its ``load`` is ``load_prompted``. The module is importable while the test runs, from the
    folder ``tmp_path / "package"``, which is also the isolation subprocess's ``sys_path``.
    """
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / f"{MODULE}.py").write_text(
        '"""A made-up prompted package."""\n\n'
        "from codeforces_tags.llm.prompted import load_prompted as load\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(folder))
    spec = MethodSpec(module=MODULE, kind="zero_shot", requirements=("pandas",), calls_api=True)
    yield METHODS | {"prompted-text": spec}
    sys.modules.pop(MODULE, None)


def test_a_prompted_method_passes_the_conformance_check(
    method, registry, raw_problems, tmp_path, monkeypatch
):
    # The reloaded method has no Ask: it asks client.ask_for, which gives the fake here. The
    # isolation subprocess only loads it, since it calls an API: no provider SDK is imported.
    monkeypatch.setattr(client, "ask_for", lambda settings: method.ask)

    check_tagging_method(
        method,
        raw_problems,
        tmp_path / "model",
        registry=registry,
        sys_path=(str(tmp_path / "package"),),
    )
