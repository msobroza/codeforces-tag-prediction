"""Tests for the Jev runner: concurrent requests, each answer appended as it arrives, the cap."""

import asyncio
import dataclasses
from unittest.mock import Mock

import pandas as pd
import pytest

from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.store import append_answer, read_answers
from codeforces_tags.methods.typesafe_jev import questions, runner

# What runner.git_state() returns in these tests: the commit and state that stamp every answer.
GIT = {"commit": "c" * 40, "code_dirty": True}
# The usage of an answer that costs $0.042 with jev-1.13.0: a million input tokens.
MILLION = {"input_tokens": 1_000_000, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}


@pytest.fixture(autouse=True)
def fixed_code_state(monkeypatch):
    """Stamp answers with GIT, so that no test asks git about the repository."""
    monkeypatch.setattr(runner, "git_state", lambda: GIT)


@pytest.fixture(name="requests_for")
def requests_for_fixture(make_raw_record):
    """Return a builder of the requests of problems 1 to ``count`` of a method, in that order.

    Each problem has a statement of its own, so no two requests share a hash.
    """

    def build(count, name="jev-statement"):
        ids = pd.Index(range(1, count + 1), name="sample_id")
        records = [make_raw_record(prob_desc_description=f"Add up {n} numbers.") for n in ids]
        return questions.requests(pd.DataFrame(records, index=ids), name)

    return build


@pytest.fixture(name="answers_path")
def answers_path_fixture(tmp_path):
    """Return the answers file of a method's dev part, in a folder that does not exist yet."""
    return tmp_path / "jev-statement" / "answers_dev.jsonl"


def run_send(requests, ask, path, max_cost_usd=None):
    """Run runner.send in an event loop of its own, as one run command does."""
    return asyncio.run(runner.send(requests, ask, path, max_cost_usd=max_cost_usd))


def run_scripted(call):
    """Run a scripted call in an event loop of its own: one that deadlocks fails within 10 s."""
    return asyncio.run(asyncio.wait_for(call, timeout=10))


def answering(asked, jev_line, **changes):
    """Return a fake Ask that records each problem it is sent, then answers it at once."""

    async def ask(request):
        asked.append(request.sample_id)
        return jev_line(request, **changes)

    return ask


def test_send_appends_each_answer_as_it_arrives_stamped_with_the_code_state(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # Problem 1's answer comes back last: problems 2 and 3 are on disk while it is in flight.
    monkeypatch.setattr(runner, "CONCURRENCY", 3)
    code_state = Mock(return_value=GIT)
    monkeypatch.setattr(runner, "git_state", code_state)
    requests = requests_for(3)
    seen_by_one = []

    async def scripted():
        three_answered = asyncio.Event()

        async def ask(request):
            if request.sample_id == 1:
                await three_answered.wait()
                seen_by_one.extend(line.sample_id for line in read_answers(answers_path))
            elif request.sample_id == 3:
                three_answered.set()
            return jev_line(request)

        return await runner.send(requests, ask, answers_path, max_cost_usd=None)

    assert run_scripted(scripted()) == 0

    assert seen_by_one == [2, 3]
    stamped = [
        dataclasses.replace(jev_line(requests[index]), git_commit="c" * 40, code_dirty=True)
        for index in (1, 2, 0)
    ]
    assert read_answers(answers_path) == stamped
    code_state.assert_called_once_with()
    assert capsys.readouterr().out == (
        f"{answers_path}: 3 of 3 answers (0 failed), 7,500 input tokens, 0 output tokens, $0.0003\n"
    )


def test_send_with_nothing_pending_says_so_and_never_asks(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # A refused request's line is final too: nothing is sent, and git is not even asked.
    requests = requests_for(2)
    append_answer(answers_path, jev_line(requests[0]))
    append_answer(answers_path, jev_line(requests[1], status="api_error:422"))
    asker, code_state = Mock(), Mock()
    monkeypatch.setattr(runner, "git_state", code_state)

    assert run_send(requests, asker, answers_path) == 0

    assert not asker.mock_calls
    assert not code_state.mock_calls
    assert capsys.readouterr().out == (
        f"{answers_path}: every request has an answer: nothing to send.\n"
    )


def test_send_prints_progress_every_hundred_answers(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # Here every 2 answers, then once more at the end. Problem 3's request is refused: a failed
    # line with no tokens, which costs nothing. Each other answer costs $0.000105.
    monkeypatch.setattr(runner, "LLM_PROGRESS_EVERY", 2)

    async def refuse_three(request):
        if request.sample_id == 3:
            return jev_line(request, status="api_error:400")
        return jev_line(request)

    assert run_send(requests_for(5), refuse_three, answers_path) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"{answers_path}: 2 of 5 answers (0 failed), 5,000 input tokens, 0 output tokens, $0.0002",
        f"{answers_path}: 4 of 5 answers (1 failed), 7,500 input tokens, 0 output tokens, $0.0003",
        f"{answers_path}: 5 of 5 answers (1 failed), 10,000 input tokens, 0 output tokens, $0.0004",
    ]


def test_send_prints_its_last_progress_line_once(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "LLM_PROGRESS_EVERY", 2)

    run_send(requests_for(4), answering([], jev_line), answers_path)

    counts = [line.split(" answers ")[0] for line in capsys.readouterr().out.splitlines()]
    assert counts == [f"{answers_path}: 2 of 4", f"{answers_path}: 4 of 4"]


def test_send_keeps_at_most_concurrency_requests_in_flight(
    requests_for, jev_line, answers_path, monkeypatch
):
    monkeypatch.setattr(runner, "CONCURRENCY", 2)
    flying, peaks = set(), []

    async def slow(request):
        # Each request lets the loop run once before it is answered.
        flying.add(request.sample_id)
        peaks.append(len(flying))
        await asyncio.sleep(0)
        flying.discard(request.sample_id)
        return jev_line(request)

    assert run_send(requests_for(7), slow, answers_path) == 0

    assert max(peaks) == 2
    assert len(read_answers(answers_path)) == 7


def test_send_stops_sending_at_the_cost_cap_and_sends_the_rest_next_time(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # Each answer costs $0.042. Problem 1's leaves the call under the $0.06 cap, so problem 3
    # is sent while problem 2 is in flight; problem 2's reaches the cap. Problem 3's answer is
    # still written, and problems 4 to 6 are not sent.
    monkeypatch.setattr(runner, "CONCURRENCY", 2)
    requests = requests_for(6)
    asked = []

    async def scripted():
        three_sent, two_answered = asyncio.Event(), asyncio.Event()

        async def ask(request):
            asked.append(request.sample_id)
            if request.sample_id == 2:
                await three_sent.wait()
            if request.sample_id == 3:
                three_sent.set()
                await two_answered.wait()
            answer = jev_line(request, usage=MILLION)
            if request.sample_id == 2:
                two_answered.set()
            return answer

        return await runner.send(requests, ask, answers_path, max_cost_usd=0.06)

    assert run_scripted(scripted()) == 1

    assert asked == [1, 2, 3]
    assert [line.sample_id for line in read_answers(answers_path)] == [1, 2, 3]
    output = capsys.readouterr()
    assert output.out == (
        f"{answers_path}: 3 of 6 answers (0 failed), 3,000,000 input tokens, 0 output tokens, "
        "$0.1260\n"
    )
    assert output.err.splitlines() == [
        f"{answers_path}: the cost cap of $0.0600 was reached after spending $0.1260; 3 requests "
        "were not sent: run again to send them.",
        f"{answers_path}: 3 of 6 requests have no answer yet: run again to send them.",
    ]

    # Each call counts only what it spends: the next one sends two more requests under the cap.
    resume = answering(asked, jev_line, usage=MILLION)
    assert run_send(requests, resume, answers_path, max_cost_usd=0.06) == 1
    assert run_send(requests, resume, answers_path, max_cost_usd=0.06) == 0
    assert asked == [1, 2, 3, 4, 5, 6]
    assert [line.sample_id for line in read_answers(answers_path)] == [1, 2, 3, 4, 5, 6]


def test_send_writes_a_final_line_for_each_refused_request_and_never_reaches_the_cap(
    requests_for, jev_line, answers_path, capsys
):
    # A request the API refuses (a 422 here) is a final failed line with no tokens: it costs
    # nothing, so no cap stops a call whose every request is refused; the run of failed answers
    # does, at runner.MAX_FAILED_IN_A_ROW (20), which four refusals do not reach.
    refuse = answering([], jev_line, status="api_error:422")

    assert run_send(requests_for(4), refuse, answers_path, max_cost_usd=0.0001) == 0

    assert [(line.sample_id, line.status, line.failed) for line in read_answers(answers_path)] == [
        (sample_id, "api_error:422", True) for sample_id in (1, 2, 3, 4)
    ]
    output = capsys.readouterr()
    assert output.out == (
        f"{answers_path}: 4 of 4 answers (4 failed), 0 input tokens, 0 output tokens, $0.0000\n"
    )
    assert not output.err


def test_send_stops_after_a_run_of_failed_answers_and_says_why(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # A fault that refuses every request would write a final line for the whole part within
    # seconds, at no cost: after MAX_FAILED_IN_A_ROW failed answers in a row, no new request is
    # sent. The lines written stay (they are final); the call returns 1 and says why.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    monkeypatch.setattr(runner, "MAX_FAILED_IN_A_ROW", 3)
    asked = []
    refuse = answering(asked, jev_line, status="api_error:422")

    assert run_send(requests_for(5), refuse, answers_path) == 1

    assert asked == [1, 2, 3]
    assert [line.status for line in read_answers(answers_path)] == ["api_error:422"] * 3
    err = capsys.readouterr().err
    assert f"{answers_path}: 3 answers in a row failed (last: api_error:422)" in err
    assert "2 requests were not sent" in err
    assert "cost cap" not in err


def test_a_completed_answer_resets_the_run_of_failed_answers(
    requests_for, jev_line, answers_path, monkeypatch
):
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    monkeypatch.setattr(runner, "MAX_FAILED_IN_A_ROW", 2)
    statuses = {1: "api_error:422", 2: "completed", 3: "api_error:422", 4: "completed"}

    async def ask(request):
        return jev_line(request, status=statuses[request.sample_id])

    assert run_send(requests_for(4), ask, answers_path) == 0

    assert [line.status for line in read_answers(answers_path)] == list(statuses.values())


def test_send_resumes_after_a_killed_run_replacing_a_partial_last_line(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # One request at a time: the first call answers problem 1, then is killed while problem 2
    # is in flight and problem 3 waits. A power cut in the middle of an append would also leave
    # a partial last line, written here by hand. The next call sends problems 2 and 3 once
    # each, and its first answer replaces the partial line.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    requests = requests_for(3)
    asked = []

    async def killed():
        two_sent = asyncio.Event()

        async def hang_on_two(request):
            asked.append(request.sample_id)
            if request.sample_id == 2:
                two_sent.set()
                await asyncio.Event().wait()
            return jev_line(request)

        call = asyncio.create_task(
            runner.send(requests, hang_on_two, answers_path, max_cost_usd=None)
        )
        await asyncio.wait_for(two_sent.wait(), timeout=10)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(killed())
    assert asked == [1, 2]
    whole = answers_path.read_bytes()
    with answers_path.open("ab") as file:
        file.write(jev_line(requests[1]).to_json().encode("utf-8")[:40])

    assert run_send(requests, answering(asked, jev_line), answers_path) == 0

    assert asked == [1, 2, 2, 3]
    assert [line.sample_id for line in read_answers(answers_path)] == [1, 2, 3]
    assert answers_path.read_bytes().startswith(whole)
    assert answers_path.read_bytes().count(b"\n") == 3
    assert capsys.readouterr().err.count("ignored a last line of 40 bytes") == 1


class KeyRefused(Exception):
    """Stands in for an SDK error that every request would get, such as a 401."""


def test_send_writes_no_line_for_a_transient_error_goes_on_and_returns_1(
    requests_for, jev_line, answers_path, capsys
):
    requests = requests_for(3)
    asked = []

    async def busy_on_two(request):
        asked.append(request.sample_id)
        if request.sample_id == 2:
            raise TransientApiError("TypeSafeRateLimitError 429")
        return jev_line(request)

    assert run_send(requests, busy_on_two, answers_path) == 1

    assert asked == [1, 2, 3]
    assert [line.sample_id for line in read_answers(answers_path)] == [1, 3]
    assert capsys.readouterr().err.splitlines() == [
        f"{answers_path}: problem 2 got no answer (TypeSafeRateLimitError 429); no line was "
        "written, so the next run sends its request again.",
        f"{answers_path}: 1 of 3 requests have no answer yet: run again to send them.",
    ]
    # The next call sends that request again, and every request then has its line.
    assert run_send(requests, answering(asked, jev_line), answers_path) == 0
    assert asked == [1, 2, 3, 2]


def test_send_raises_the_first_fatal_error_once_the_answers_in_flight_are_written(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # Problems 1, 2 and 3 are in flight together. Problem 1's request fails, then problem 2's
    # is answered and written, then problem 3's fails too: problem 1's error is raised, problem
    # 3's is dropped, and problems 4 and 5 are never sent.
    monkeypatch.setattr(runner, "CONCURRENCY", 3)
    asked = []

    async def scripted():
        three_sent, one_failed = asyncio.Event(), asyncio.Event()

        async def ask(request):
            asked.append(request.sample_id)
            if request.sample_id == 3:
                three_sent.set()
                await one_failed.wait()
                raise KeyRefused("problem 3")
            await three_sent.wait()
            if request.sample_id == 1:
                one_failed.set()
                raise KeyRefused("problem 1")
            return jev_line(request)

        await runner.send(requests_for(5), ask, answers_path, max_cost_usd=None)

    with pytest.raises(KeyRefused, match="problem 1"):
        run_scripted(scripted())

    assert asked == [1, 2, 3]
    assert [line.sample_id for line in read_answers(answers_path)] == [2]
    assert capsys.readouterr().out == (
        f"{answers_path}: 1 of 5 answers (0 failed), 2,500 input tokens, 0 output tokens, $0.0001\n"
    )


def test_send_reports_a_call_that_wrote_nothing_then_raises_its_error(
    requests_for, answers_path, monkeypatch, capsys
):
    # One request at a time, and the first fails as every one would: nothing more is sent.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    asked = []

    async def refuse_the_key(request):
        asked.append(request.sample_id)
        raise KeyRefused("401")

    with pytest.raises(KeyRefused):
        run_send(requests_for(3), refuse_the_key, answers_path)

    assert asked == [1]
    assert not answers_path.exists()
    assert capsys.readouterr().out == (
        f"{answers_path}: 0 of 3 answers (0 failed), 0 input tokens, 0 output tokens, $0.0000\n"
    )


def test_send_stops_after_writing_an_answer_from_another_model(
    requests_for, jev_line, answers_path, monkeypatch, capsys
):
    # The alias jev-latest answered problem 1: its paid line is written, nothing more is sent,
    # and the error names the model before any price is looked up for it.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    pricing = Mock()
    monkeypatch.setattr(runner, "answer_cost_usd", pricing)
    asked = []
    aliased = answering(asked, jev_line, served_model="jev-latest")

    with pytest.raises(ValueError, match="jev-latest"):
        run_send(requests_for(3), aliased, answers_path)

    assert not pricing.mock_calls
    assert asked == [1]
    assert [(line.sample_id, line.served_model) for line in read_answers(answers_path)] == [
        (1, "jev-latest")
    ]
    assert capsys.readouterr().out == (
        f"{answers_path}: 1 of 3 answers (0 failed), 2,500 input tokens, 0 output tokens, $0.0000\n"
    )


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan")])  # a NaN cap would be no cap at all
def test_send_refuses_a_cap_that_is_not_above_zero_before_sending(requests_for, answers_path, cap):
    # Refused before the answers file is read: its one line is no answer.
    answers_path.parent.mkdir()
    answers_path.write_bytes(b"not an answer\n")
    asker = Mock()

    with pytest.raises(ValueError, match="A cost cap must be above 0 USD"):
        run_send(requests_for(2), asker, answers_path, max_cost_usd=cap)

    assert not asker.mock_calls
    assert answers_path.read_bytes() == b"not an answer\n"
