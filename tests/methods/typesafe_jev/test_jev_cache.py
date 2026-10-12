"""Tests for the Jev answer cache of a search: the seed's file, one file per words, the cap."""

import dataclasses
from contextlib import asynccontextmanager

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.store import read_answers
from codeforces_tags.methods.typesafe_jev import questions, runner
from codeforces_tags.methods.typesafe_jev.jev_cache import (
    CostCapSpent,
    JevCache,
    SearchInterrupted,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    GEPA_SEED_NAME,
    MAX_FAILED_IN_A_ROW,
)
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS

# What runner.git_state() returns in these tests: no test asks git about the repository.
GIT = {"commit": "c" * 40, "code_dirty": False}
# What one answer of the fake Jev costs: 2,500 input tokens at jev-1.13.0's $0.042 per million.
ANSWER_USD = 2_500 * 0.042 / 1_000_000
# Words other than the seed's: another convention.
OTHER_WORDS = dataclasses.replace(SEED_WORDS, convention="Tags name the ideas of a solution.")
# What JevCache says when a call left requests unanswered under its cap: a rerun goes on.
TRANSIENT = r"unanswered \(a transient error or a run of failed answers\): run the search again"


@pytest.fixture(autouse=True)
def fixed_code_state(monkeypatch):
    """Stamp answers with GIT."""
    monkeypatch.setattr(runner, "git_state", lambda: GIT)


@pytest.fixture(name="committed")
def committed_fixture(search_raw_dev, jev_line):
    """Return committed seed answers: a jev-statement-code line for every dev problem."""
    seed_requests = questions.requests(search_raw_dev, GEPA_SEED_NAME, SEED_WORDS)
    return [jev_line(request, GEPA_SEED_NAME) for request in seed_requests]


def ids_of_fold(raw, fold):
    """Return the sample ids of one fold, in order."""
    return [int(sample_id) for sample_id in raw.index[raw["fold"] == fold]]


def test_the_seed_file_holds_the_committed_answers_of_folds_1_to_4_only(
    tmp_path, search_raw_dev, committed
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    added = cache.prefill_seed(committed)

    held = read_answers(cache.path(SEED_WORDS))
    searched = search_raw_dev.index[search_raw_dev["fold"] != 0]
    assert added == len(searched) == 80
    assert [answer.sample_id for answer in held] == list(searched)
    assert {answer.request_sha256 for answer in held} <= {a.request_sha256 for a in committed}
    assert cache.path(SEED_WORDS) == tmp_path / f"{SEED_WORDS.sha256()}.jsonl"


def test_prefilling_the_seed_again_adds_nothing_and_keeps_the_file(
    tmp_path, search_raw_dev, committed
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    cache.prefill_seed(committed)
    before = cache.path(SEED_WORDS).read_bytes()

    again = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0).prefill_seed(committed)

    assert again == 0
    assert cache.path(SEED_WORDS).read_bytes() == before


def test_prefilling_completes_a_seed_file_cut_short_by_an_interrupted_write(
    tmp_path, search_raw_dev, committed
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    cache.prefill_seed(committed)
    path = cache.path(SEED_WORDS)
    first_eight = b"".join(path.read_bytes().splitlines(keepends=True)[:8])
    path.write_bytes(first_eight + b'{"partial')

    added = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0).prefill_seed(committed)

    assert added == 72
    assert len(read_answers(path)) == 80


def test_a_committed_answer_to_another_request_is_refused_before_anything_is_written(
    tmp_path, search_raw_dev, committed
):
    changed = [
        dataclasses.replace(answer, request_sha256="f" * 64) if answer.sample_id == 1002 else answer
        for answer in committed
    ]
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    with pytest.raises(ValueError, match=r"problems \[1002\] are not to the seed requests"):
        cache.prefill_seed(changed)
    assert not cache.path(SEED_WORDS).exists()


def test_a_searched_problem_without_a_committed_answer_is_refused(
    tmp_path, search_raw_dev, committed
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    with pytest.raises(ValueError, match=r"lack the problems \[1001\]"):
        cache.prefill_seed([answer for answer in committed if answer.sample_id != 1001])


def test_the_seed_is_scored_from_its_file_without_a_request(
    tmp_path, search_raw_dev, committed, jev_api
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    cache.prefill_seed(committed)
    ids = ids_of_fold(search_raw_dev, 4)

    scores = cache.probabilities(SEED_WORDS, ids)

    assert not jev_api.sent and not jev_api.sessions
    assert list(scores.index) == ids and scores.index.name == "sample_id"
    assert list(scores.columns) == list(FOCUS_TAGS)
    assert (scores == 0.5).all().all()  # the committed lines answer 0.5 on every tag
    assert cache.used_usd() == 0.0


def test_new_words_are_asked_once_each_into_their_own_file(tmp_path, search_raw_dev, jev_api):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    first = cache.probabilities(OTHER_WORDS, [1001, 1002, 1003])
    second = cache.probabilities(OTHER_WORDS, [1003, 1004, 1001])

    assert jev_api.sent == [(GEPA_NAME, sample_id) for sample_id in (1001, 1002, 1003, 1004)]
    assert jev_api.sessions == [GEPA_NAME, GEPA_NAME]
    path = cache.path(OTHER_WORDS)
    assert path == tmp_path / f"{OTHER_WORDS.sha256()}.jsonl"
    asked = search_raw_dev.loc[[1001, 1002, 1003, 1004]]
    expected = questions.requests(asked, GEPA_NAME, OTHER_WORDS)
    assert [answer.request_sha256 for answer in read_answers(path)] == [
        request.sha256() for request in expected
    ]
    assert list(second.index) == [1003, 1004, 1001]
    assert second.loc[1003].equals(first.loc[1003])
    assert second.loc[1004].to_dict() == jev_api.probabilities(expected[3].state)


def test_spent_counts_every_file_but_the_seed_and_used_counts_each_answer_once(
    tmp_path, search_raw_dev, committed, jev_api
):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    cache.prefill_seed(committed)
    cache.probabilities(SEED_WORDS, [1001, 1002])

    cache.probabilities(OTHER_WORDS, [1001, 1002, 1003])
    cache.probabilities(OTHER_WORDS, [1001, 1002])

    assert len(jev_api.sent) == 3
    assert cache.spent_usd() == pytest.approx(3 * ANSWER_USD)
    assert cache.used_usd() == pytest.approx(3 * ANSWER_USD)
    replayed = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    replayed.probabilities(OTHER_WORDS, [1002])
    assert replayed.used_usd() == pytest.approx(ANSWER_USD)
    assert replayed.spent_usd() == pytest.approx(3 * ANSWER_USD)


def test_a_held_out_problem_is_refused(tmp_path, search_raw_dev, jev_api):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    with pytest.raises(ValueError, match=r"problems \[1000, 1005\] .* held-out fold"):
        cache.probabilities(OTHER_WORDS, [1001, 1000, 1005])
    assert not jev_api.sent


def test_the_cap_reached_inside_a_call_interrupts_the_search_and_a_rerun_resumes(
    tmp_path, search_raw_dev, jev_api
):
    ids = [1001, 1002, 1003, 1004, 1006]
    capped = JevCache(tmp_path, search_raw_dev, max_cost_usd=ANSWER_USD / 2)

    with pytest.raises(CostCapSpent, match="unanswered at the cost cap.* with a higher cap"):
        capped.probabilities(OTHER_WORDS, ids)

    written = len(read_answers(capped.path(OTHER_WORDS)))
    assert 1 <= written < len(ids)
    resumed = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    resumed.probabilities(OTHER_WORDS, ids)
    assert len(jev_api.sent) == len(ids)


def test_a_spent_cap_interrupts_before_any_session_opens(tmp_path, search_raw_dev, jev_api):
    JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0).probabilities(OTHER_WORDS, [1001])
    capped = JevCache(tmp_path, search_raw_dev, max_cost_usd=ANSWER_USD)

    with pytest.raises(CostCapSpent, match="this search cost .* with a higher cap"):
        capped.probabilities(OTHER_WORDS, [1002])
    assert jev_api.sessions == [GEPA_NAME]


def test_a_transient_error_interrupts_the_search_and_writes_no_line(
    tmp_path, search_raw_dev, jev_api
):
    jev_api.outcomes[1002] = TransientApiError("TypeSafe did not answer")
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    with pytest.raises(SearchInterrupted, match=TRANSIENT) as interrupted:
        cache.probabilities(OTHER_WORDS, [1001, 1002])

    assert interrupted.type is SearchInterrupted

    assert [answer.sample_id for answer in read_answers(cache.path(OTHER_WORDS))] == [1001]
    assert cache.probabilities(OTHER_WORDS, [1001, 1002]).index.tolist() == [1001, 1002]


def test_a_run_of_failed_answers_interrupts_the_search_and_a_rerun_sends_only_the_others(
    tmp_path, search_raw_dev, jev_api, monkeypatch, capsys
):
    # Jev refuses every request (a 422, as a state made too long by new words would get): each
    # refusal is a final line that costs nothing, so the cap never stops them, and runner.send
    # stops after MAX_FAILED_IN_A_ROW of them in a row. One request at a time, as Jev answers.
    monkeypatch.setattr(runner, "CONCURRENCY", 1)
    searched = search_raw_dev.index[search_raw_dev["fold"] != 0]
    ids = [int(sample_id) for sample_id in searched[: MAX_FAILED_IN_A_ROW + 5]]
    jev_api.outcomes.update({sample_id: {"status": "api_error:422"} for sample_id in ids})
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)

    with pytest.raises(SearchInterrupted, match=TRANSIENT) as interrupted:
        cache.probabilities(OTHER_WORDS, ids)

    assert interrupted.type is SearchInterrupted

    refused, left = ids[:MAX_FAILED_IN_A_ROW], ids[MAX_FAILED_IN_A_ROW:]
    lines = read_answers(cache.path(OTHER_WORDS))
    assert [(line.sample_id, line.status, line.failed) for line in lines] == [
        (sample_id, "api_error:422", True) for sample_id in refused
    ]
    assert f"{MAX_FAILED_IN_A_ROW} answers in a row failed" in capsys.readouterr().err
    # The failed lines are final: the next call sends only the requests left (which Jev now
    # answers), and scores each refused problem 0.0 on every tag, so a rerun makes progress.
    for sample_id in left:
        del jev_api.outcomes[sample_id]
    del jev_api.sent[:]

    scores = cache.probabilities(OTHER_WORDS, ids)

    assert jev_api.sent == [(GEPA_NAME, sample_id) for sample_id in left]
    assert (scores.loc[refused] == 0.0).all().all()
    assert (scores.loc[left] > 0.0).all().all()
    assert cache.spent_usd() == pytest.approx(len(left) * ANSWER_USD)


def test_an_explicit_session_is_used_instead_of_the_client(tmp_path, search_raw_dev, jev_api):
    opened = []

    @asynccontextmanager
    async def session(name):
        opened.append(name)
        async with jev_api.session(name) as ask:
            yield ask

    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0, jev_session=session)

    cache.probabilities(OTHER_WORDS, [1001])

    assert opened == [GEPA_NAME]
    assert jev_api.sent == [(GEPA_NAME, 1001)]


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan")])
def test_a_cap_that_is_not_above_zero_is_refused(tmp_path, search_raw_dev, cap):
    with pytest.raises(ValueError, match="above 0 USD"):
        JevCache(tmp_path, search_raw_dev, max_cost_usd=cap)


def test_the_folds_kept_are_the_learning_and_selection_folds(tmp_path, search_raw_dev, jev_api):
    cache = JevCache(tmp_path, search_raw_dev, max_cost_usd=1.0)
    fold_four = ids_of_fold(search_raw_dev, 4)

    scores = cache.probabilities(OTHER_WORDS, fold_four)

    assert isinstance(scores, pd.DataFrame) and len(scores) == len(fold_four) == 20
    assert len(jev_api.sent) == 20
