"""Tests for the runner: concurrent requests, each answer written as it arrives, the cost cap."""

import asyncio
from unittest.mock import Mock

import pytest

from codeforces_tags.llm import runner
from codeforces_tags.llm.client import USAGE_KEYS, RejectedSetting, TransientApiError
from codeforces_tags.llm.runner import answer_each
from codeforces_tags.llm.store import read_answers
from codeforces_tags.provenance import git_state

# The git state the tests stamp answers with, as provenance.git_state() gives it.
GIT = {"commit": "c" * 40, "code_dirty": True}
# The usage of an answer that costs $0.10 with gpt-6-luna: a million input tokens.
DIME = {"input_tokens": 1_000_000, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}


@pytest.fixture(name="reply")
def reply_fixture(make_answer):
    """Return a builder of the answer to a request: its problem, under its hash."""

    def build(request, **changes):
        return make_answer(sample_id=request.sample_id, request_sha256=request.sha256(), **changes)

    return build


def run(requests, ask, path, **options):
    """Send requests with answer_each in an event loop of its own, as one llm run does."""
    values = {"concurrency": 1, "answers_file": path, "output_format": "text", "git": GIT}
    return asyncio.run(answer_each(requests, ask, **(values | options)))


def line_of(answer):
    """Return the bytes that an answers file holds for an answer: its line and a newline."""
    return answer.to_json().encode("utf-8") + b"\n"


def test_answer_each_writes_each_answer_stamped_with_the_git_state_and_its_tags(
    reply, requests_for, tmp_path
):
    requests = requests_for(2)
    path = tmp_path / "answers_dev.jsonl"

    async def ask(request):
        return reply(request, output_text="dp, MATH, nonsense")

    written = run(requests, ask, path, all_tags=("dp", "math"))

    assert read_answers(path) == written
    assert [answer.sample_id for answer in written] == [1, 2]
    assert {(answer.git_commit, answer.code_dirty) for answer in written} == {("c" * 40, True)}
    assert {(answer.tags, answer.failed, answer.unknown_pieces) for answer in written} == {
        (("dp", "math"), False, 1)
    }


def test_answer_each_parses_each_answer_with_the_parser_of_its_output_format(
    make_request, reply, tmp_path
):
    request = make_request("json")

    async def ask(request):
        return reply(request, output_text='{"tags": ["dp", "dp"]}')

    [answer] = run([request], ask, tmp_path / "answers_dev.jsonl", output_format="json")

    assert (answer.tags, answer.failed, answer.unknown_pieces) == (("dp",), False, 0)


def test_answer_each_stamps_the_git_state_of_the_code_when_given_none(
    reply, requests_for, tmp_path
):
    async def ask(request):
        return reply(request)

    [answer] = run(requests_for(1), ask, tmp_path / "answers_dev.jsonl", git=None)

    state = git_state()
    assert (answer.git_commit, answer.code_dirty) == (state["commit"], state["code_dirty"])


def test_answer_each_sends_only_the_requests_that_have_no_line_yet(
    reply, requests_for, tmp_path, capsys
):
    requests = requests_for(3)
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(line_of(reply(requests[1], status="incomplete")))
    sent = []

    async def ask(request):
        sent.append(request.sample_id)
        return reply(request)

    written = run(requests, ask, path)

    assert sent == [1, 3]
    assert [answer.sample_id for answer in written] == [1, 3]
    assert [answer.sample_id for answer in read_answers(path)] == [2, 1, 3]
    assert "2 of 2 answers" in capsys.readouterr().out


def test_answer_each_with_nothing_to_send_opens_no_session(reply, requests_for, tmp_path, capsys):
    requests = requests_for(2)
    path = tmp_path / "answers_dev.jsonl"
    path.write_bytes(b"".join(line_of(reply(request)) for request in requests))

    # A Mock records a call of itself, or of its open(): an adapter would open a client, which
    # needs the provider's API key.
    ask = Mock()

    assert run(requests, ask, path) == []
    assert not ask.mock_calls
    assert capsys.readouterr().out == f"{path}: every request has an answer: nothing to send.\n"


def test_answer_each_sends_its_batch_through_one_session(
    reply, requests_for, tmp_path, openable_ask
):
    ask = openable_ask(reply)

    run(requests_for(2), ask, tmp_path / "answers_dev.jsonl")

    assert ask.events == ["open", 1, 2, "close"]


def test_answer_each_sends_at_most_concurrency_requests_at_once(
    reply, requests_for, tmp_path, in_flight
):
    async def ask(request):
        return reply(request)

    counted = in_flight(ask)

    written = run(requests_for(7), counted, tmp_path / "answers_dev.jsonl", concurrency=3)

    assert len(written) == 7
    assert counted.most == 3


def test_answer_each_writes_no_line_for_a_transient_error_and_goes_on(
    reply, requests_for, tmp_path, capsys
):
    path = tmp_path / "answers_dev.jsonl"
    sent = []

    async def ask(request):
        sent.append(request.sample_id)
        if request.sample_id == 2:
            raise TransientApiError("The provider did not answer.")
        return reply(request)

    written = run(requests_for(3), ask, path)

    assert sent == [1, 2, 3]
    assert [answer.sample_id for answer in written] == [1, 3]
    assert [answer.sample_id for answer in read_answers(path)] == [1, 3]
    assert capsys.readouterr().err == (
        "Problem 2: The provider did not answer. No line was written: the next run sends its "
        "request again.\n"
    )


def test_answer_each_stops_sending_when_the_api_rejects_a_setting(reply, requests_for, tmp_path):
    # Problem 1's request is rejected while problem 2's is in flight: problem 2's answer is
    # still written, then the run stops, and problems 3 and 4 are never sent.
    path = tmp_path / "answers_dev.jsonl"
    sent = []

    async def rejected_run():
        second_sent, first_rejected = asyncio.Event(), asyncio.Event()

        async def ask(request):
            sent.append(request.sample_id)
            if request.sample_id == 1:
                await second_sent.wait()
                first_rejected.set()
                raise RejectedSetting("OpenAI rejected a setting: temperature.")
            second_sent.set()
            await first_rejected.wait()
            return reply(request)

        options = {"concurrency": 2, "answers_file": path, "output_format": "text", "git": GIT}
        await answer_each(requests_for(4), ask, **options)

    with pytest.raises(RejectedSetting, match="temperature"):
        asyncio.run(rejected_run())

    assert sent == [1, 2]
    assert [answer.sample_id for answer in read_answers(path)] == [2]


def test_answer_each_writes_an_answer_before_it_prices_it(reply, requests_for, tmp_path):
    # A served model with no price stops the run, but the answer it paid for is written first.
    path = tmp_path / "answers_dev.jsonl"
    sent = []

    async def ask(request):
        sent.append(request.sample_id)
        return reply(request, served_model="gpt-6-luna-2026-09-01")

    with pytest.raises(ValueError, match="'gpt-6-luna-2026-09-01' has no price"):
        run(requests_for(3), ask, path)

    assert sent == [1]
    assert [answer.sample_id for answer in read_answers(path)] == [1]


def test_answer_each_stops_at_the_first_error_of_every_request(reply, requests_for, tmp_path):
    # An HTTP error of every request (a missing key, an unknown model) is the SDK's own error.
    sent = []

    class AuthenticationError(Exception):
        """Stands in for an SDK's error of a missing API key."""

    async def ask(request):
        sent.append(request.sample_id)
        if request.sample_id == 2:
            raise AuthenticationError("Error code: 401")
        return reply(request)

    with pytest.raises(AuthenticationError):
        run(requests_for(3), ask, tmp_path / "answers_dev.jsonl")

    assert sent == [1, 2]


def test_answer_each_raises_the_first_error_once_the_requests_in_flight_are_written(
    requests_for, tmp_path
):
    # Both requests in flight fail: the first error is raised, the second is dropped.
    async def failing_run():
        both_sent, first_failed = asyncio.Event(), asyncio.Event()

        async def ask(request):
            if request.sample_id == 1:
                await both_sent.wait()
                first_failed.set()
            else:
                both_sent.set()
                await first_failed.wait()
            raise RejectedSetting(f"rejected problem {request.sample_id}")

        options = {"concurrency": 2, "answers_file": tmp_path / "a.jsonl", "output_format": "text"}
        await answer_each(requests_for(2), ask, git=GIT, **options)

    with pytest.raises(RejectedSetting, match="rejected problem 1"):
        asyncio.run(failing_run())


def test_answer_each_stops_sending_at_the_cost_cap(reply, requests_for, tmp_path, capsys):
    # Each answer costs $0.10. The second reaches the $0.15 cap while the third is in flight:
    # the third is still written, nothing more is sent, and the next run sends the rest.
    requests = requests_for(6)
    path = tmp_path / "answers_dev.jsonl"
    sent = []

    async def capped_run():
        third_sent, second_answered = asyncio.Event(), asyncio.Event()

        async def ask(request):
            sent.append(request.sample_id)
            if request.sample_id == 2:
                await third_sent.wait()
                second_answered.set()
            if request.sample_id == 3:
                third_sent.set()
                await second_answered.wait()
            return reply(request, usage=DIME)

        options = {"concurrency": 2, "answers_file": path, "output_format": "text", "git": GIT}
        return await answer_each(requests, ask, max_cost_usd=0.15, **options)

    written = asyncio.run(capped_run())

    assert sent == [1, 2, 3]
    assert [answer.sample_id for answer in written] == [1, 2, 3]
    assert read_answers(path) == written
    output = capsys.readouterr()
    assert output.out.endswith(
        "3 of 6 answers (0 failed), 3,000,000 input tokens (0 cached), 0 output tokens, $0.3000\n"
    )
    assert output.err == (
        f"{path}: stopped at the cost cap of $0.1500, after spending $0.3000: 3 requests were "
        "not sent. Run again to send them.\n"
    )

    async def ask_the_rest(request):
        sent.append(request.sample_id)
        return reply(request, usage=DIME)

    run(requests, ask_the_rest, path)

    assert sent == [1, 2, 3, 4, 5, 6]
    assert sorted(answer.sample_id for answer in read_answers(path)) == [1, 2, 3, 4, 5, 6]


def test_answer_each_resumes_after_a_killed_run(reply, requests_for, tmp_path, capsys):
    # The first run answers problem 1, loses problem 2 to a transient error, has problem 3's
    # request refused (a final answer), and is killed while problems 4 and 5 are in flight and
    # problem 6 is waiting; the kill also cuts problem 4's line short. The next run sends 2, 4,
    # 5 and 6 once each, and replaces the partial line.
    requests = requests_for(6)
    path = tmp_path / "answers_dev.jsonl"
    sent = []

    async def killed_run():
        in_flight = asyncio.Event()

        async def ask(request):
            sent.append(request.sample_id)
            if request.sample_id == 2:
                raise TransientApiError("The provider did not answer.")
            if request.sample_id == 3:
                usage = dict.fromkeys(USAGE_KEYS, 0)
                return reply(request, output_text=None, status="api_error:400", usage=usage)
            if request.sample_id == 5:
                in_flight.set()
            if request.sample_id in (4, 5):
                await asyncio.Event().wait()
            return reply(request)

        options = {"concurrency": 2, "answers_file": path, "output_format": "text", "git": GIT}
        task = asyncio.create_task(answer_each(requests, ask, **options))
        await asyncio.wait_for(in_flight.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(killed_run())
    assert sent == [1, 2, 3, 4, 5]
    kept = path.read_bytes()
    with path.open("ab") as file:
        file.write(line_of(reply(requests[3]))[:40])
    capsys.readouterr()

    async def ask_again(request):
        sent.append(request.sample_id)
        return reply(request)

    run(requests, ask_again, path, concurrency=2)

    assert sorted(sent[5:]) == [2, 4, 5, 6]
    assert "ignored a last line of 40 bytes" in capsys.readouterr().err
    answers = read_answers(path)
    assert sorted(answer.sample_id for answer in answers) == [1, 2, 3, 4, 5, 6]
    assert [answer.status for answer in answers if answer.sample_id == 3] == ["api_error:400"]
    assert path.read_bytes().startswith(kept)
    assert path.read_bytes().count(b"\n") == 6
    assert capsys.readouterr().err == ""
    assert run(requests, ask_again, path) == []
    assert len(sent) == 9


def test_answer_each_prints_its_progress_every_few_answers_and_at_the_end(
    reply, requests_for, tmp_path, monkeypatch, capsys
):
    # Each answer costs $0.0016: 10,000 uncached input tokens at $0.10 per million, 10,000
    # cached ones at $0.01, and 1,000 output tokens at $0.50. Problem 3's answer is cut off.
    monkeypatch.setattr(runner, "LLM_PROGRESS_EVERY", 2)
    path = tmp_path / "answers_dev.jsonl"
    usage = {"input_tokens": 20_000, "cached_tokens": 10_000, "output_tokens": 1_000}

    async def ask(request):
        status = "incomplete" if request.sample_id == 3 else "completed"
        return reply(request, status=status, usage=usage | {"reasoning_tokens": 0})

    run(requests_for(5), ask, path)

    assert capsys.readouterr().out.splitlines() == [
        f"{path}: 2 of 5 answers (0 failed), 40,000 input tokens (20,000 cached), 2,000 output "
        "tokens, $0.0032",
        f"{path}: 4 of 5 answers (1 failed), 80,000 input tokens (40,000 cached), 4,000 output "
        "tokens, $0.0064",
        f"{path}: 5 of 5 answers (1 failed), 100,000 input tokens (50,000 cached), 5,000 output "
        "tokens, $0.0080",
    ]


def test_answer_each_prints_its_last_progress_line_once(
    reply, requests_for, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "LLM_PROGRESS_EVERY", 2)
    path = tmp_path / "answers_dev.jsonl"

    async def ask(request):
        return reply(request)

    run(requests_for(4), ask, path)

    lines = capsys.readouterr().out.splitlines()
    assert [line.partition(",")[0] for line in lines] == [
        f"{path}: 2 of 4 answers (0 failed)",
        f"{path}: 4 of 4 answers (0 failed)",
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"concurrency": 0}, "concurrency must be at least 1"),
        ({"output_format": "yaml"}, "'yaml' is not an output format"),
        ({"output_format": "json"}, "answer in text, not json"),
        ({"max_cost_usd": 0.0}, "cost cap must be above 0"),
        ({"max_cost_usd": float("nan")}, "cost cap must be above 0"),  # else no cap at all
    ],
)
def test_answer_each_refuses_bad_options_before_sending_anything(
    requests_for, tmp_path, options, message
):
    ask = Mock()

    with pytest.raises(ValueError, match=message):
        run(requests_for(2), ask, tmp_path / "answers_dev.jsonl", **options)

    assert not ask.mock_calls
