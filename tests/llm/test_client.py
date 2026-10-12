"""Tests for the seam between the toolkit and a provider: answers, Ask, sessions, ask_for."""

import asyncio
import dataclasses
import json

import pytest

from codeforces_tags.llm.client import (
    USAGE_KEYS,
    LlmAnswer,
    RejectedSetting,
    TransientApiError,
    ask_for,
    session,
)


def test_usage_counts_the_input_cached_output_and_reasoning_tokens():
    assert USAGE_KEYS == ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def test_an_answer_round_trips_through_one_json_line(make_answer):
    answer = make_answer(tags=("dp", "math"), unknown_pieces=1, git_commit="0" * 40)

    line = answer.to_json()

    assert "\n" not in line
    assert LlmAnswer.from_json(line) == answer
    assert LlmAnswer.from_json(line + "\n").tags == ("dp", "math")


def test_an_answer_line_has_sorted_keys_and_keeps_non_ascii_text(make_answer):
    line = make_answer(output_text="dp, mathématiques → 数学").to_json()

    assert list(json.loads(line)) == sorted(json.loads(line))
    assert "mathématiques → 数学" in line


def test_an_answer_line_escapes_what_splitlines_takes_for_a_line_break(make_answer):
    # json.dumps keeps U+0085, U+2028 and U+2029 as they are when ensure_ascii is off, and
    # str.splitlines() splits on them: escaped, an answer is one line whatever reads it.
    answer = make_answer(output_text="dp\x85math\u2028trees\u2029graphs\r\n")

    line = answer.to_json()

    assert line.splitlines() == [line]
    assert LlmAnswer.from_json(line) == answer


def test_an_answer_is_stamped_and_parsed_by_the_runner_later(make_answer):
    answer = make_answer()

    assert (answer.git_commit, answer.code_dirty) == ("", False)
    assert (answer.tags, answer.failed, answer.unknown_pieces) == ((), False, 0)


def test_from_json_refuses_a_line_with_an_unknown_key(make_answer):
    stored = json.loads(make_answer().to_json()) | {"prompt": "the problem's text"}

    with pytest.raises(TypeError, match="prompt"):
        LlmAnswer.from_json(json.dumps(stored))


@pytest.mark.parametrize("line", ["[]", "3", "null", '"tags"'])
def test_from_json_refuses_a_line_that_is_not_a_json_object(line):
    with pytest.raises(TypeError, match="JSON object"):
        LlmAnswer.from_json(line)


def test_from_json_refuses_a_line_that_lacks_a_field_of_the_answer(make_answer):
    stored = json.loads(make_answer().to_json())
    del stored["request_sha256"]

    with pytest.raises(TypeError, match="request_sha256"):
        LlmAnswer.from_json(json.dumps(stored))


def test_from_json_reads_a_line_that_lacks_the_fields_the_runner_stamps(make_answer):
    stored = json.loads(make_answer().to_json())
    for key in ("git_commit", "code_dirty", "tags", "failed", "unknown_pieces"):
        del stored[key]

    assert LlmAnswer.from_json(json.dumps(stored)) == make_answer()


def test_answers_are_frozen(make_answer):
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_answer().status = "failed"


@pytest.mark.parametrize("error", [TransientApiError, RejectedSetting])
def test_the_api_errors_that_stop_an_answer_are_runtime_errors(error):
    assert issubclass(error, RuntimeError)


def test_a_session_of_a_plain_ask_is_that_ask(make_answer, make_request):
    # A test's fake Ask is a coroutine function: there is no client to open.
    async def fake_ask(request):
        return make_answer(sample_id=request.sample_id)

    async def run():
        async with session(fake_ask) as send:
            return send, await send(make_request(sample_id=7))

    send, answer = asyncio.run(run())

    assert send is fake_ask
    assert answer.sample_id == 7


def test_a_session_opens_an_adapters_ask_for_the_batch_and_closes_it_after(
    make_answer, make_request, openable_ask
):
    ask = openable_ask(lambda request: make_answer(sample_id=request.sample_id))

    async def run():
        async with session(ask) as send:
            await send(make_request(sample_id=1))
            await send(make_request(sample_id=2))

    asyncio.run(run())

    assert ask.events == ["open", 1, 2, "close"]


@pytest.mark.usefixtures("fake_provider")
def test_ask_for_returns_the_ask_of_the_settings_provider(make_settings):
    settings = make_settings(provider="fake")

    assert ask_for(settings) == ("the fake ask of", settings)


def test_ask_for_refuses_a_provider_that_has_no_adapter(make_settings):
    with pytest.raises(ValueError, match="'other' is not a provider"):
        ask_for(make_settings(provider="other"))
