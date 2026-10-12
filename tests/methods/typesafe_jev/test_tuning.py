"""Tests for the search over Jev's words: GEPA's real engine, with a fake Jev and a fake Luna."""

import hashlib
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pandas as pd
import pytest

from codeforces_tags.config import EXPERIMENTS_DIR, FOCUS_TAGS
from codeforces_tags.conformance import starts_like_pickle
from codeforces_tags.llm.client import LlmAnswer, TransientApiError
from codeforces_tags.llm.store import read_answers
from codeforces_tags.methods.typesafe_jev import answers, questions, runner, tuning
from codeforces_tags.methods.typesafe_jev.jev_cache import (
    CostCapSpent,
    JevCache,
    SearchInterrupted,
)
from codeforces_tags.methods.typesafe_jev.objective import (
    balanced_brier,
    problem_scores,
    tag_weights,
)
from codeforces_tags.methods.typesafe_jev.questions import question_id
from codeforces_tags.methods.typesafe_jev.reflection import Reflector
from codeforces_tags.methods.typesafe_jev.search_record import HISTORY_COLUMNS
from codeforces_tags.methods.typesafe_jev.settings import GEPA_NAME, GEPA_SEED_NAME
from codeforces_tags.methods.typesafe_jev.tuning import (
    CostCap,
    IterationCap,
    JevSearchAdapter,
    SearchRecorder,
    run_search,
    search_data,
    search_folder,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    candidate_from_words,
)

# What runner.git_state() returns in these tests: no test asks git about the repository.
GIT = {"commit": "c" * 40, "code_dirty": False}
# The word of the fake reflection model's rewrites; the seed words never hold it.
DRAFT = "Draft"
# The fields of a reflection answer that failed: cut off once its reasoning used up
# max_output_tokens, as the OpenAI adapter gives it (no text, billed tokens), or refused.
INCOMPLETE = {
    "status": "incomplete",
    "output_text": "",
    "extra": {"incomplete_reason": "max_output_tokens"},
}
REFUSED = {"status": "refused", "output_text": None, "failed": True}
# The keys of a line of history.jsonl, in order: the columns of history.csv that export reads,
# and the minibatch after the piece. Then the keys of a line of candidates.jsonl, in order.
HISTORY_KEYS = [*HISTORY_COLUMNS[:3], "minibatch", *HISTORY_COLUMNS[3:]]
CANDIDATE_KEYS = [
    "index",
    "parents",
    "iteration",
    "piece",
    "words",
    "words_sha256",
    "selection_scores",
    "selection_mean",
    "selection_brier",
]
# The keys of outcome.json, in order: the caps this run used follow what it cost.
OUTCOME_KEYS = [
    "stop_reason",
    "iterations",
    "best_index",
    "cost_usd",
    "max_cost_usd",
    "max_iterations",
    "gepa_version",
]


@pytest.fixture(autouse=True)
def fixed_code_state(monkeypatch):
    """Stamp answers with GIT."""
    monkeypatch.setattr(runner, "git_state", lambda: GIT)


@dataclass
class DraftJev:
    """A fake Jev whose answer on a tag is right once the tag's question holds a draft.

    A question that holds ``DRAFT`` (in its tag's criteria, or in the convention, which every
    question holds) gets 0.9 for a problem with the tag and 0.1 for one without; any other gets
    a probability drawn from the request's hash. So a rewrite helps, a second one does not, and
    every answer is a function of its request.

    Attributes:
        line: The ``jev_line`` builder.
        tags: Each problem's tags, by sample id.
        sent: The sample id of every request, in the order they were asked.
        failing: Sample ids whose next request raises a ``TransientApiError``.
    """

    line: Callable[..., LlmAnswer]
    tags: pd.Series
    sent: list = field(default_factory=list)
    failing: set = field(default_factory=set)

    def probabilities(self, request):
        """Return the probability of each focus tag for a request."""
        digest = hashlib.sha256(request.sha256().encode()).digest()
        values = {}
        for position, tag in enumerate(FOCUS_TAGS):
            if DRAFT in json.dumps(request.questions[question_id(tag)]):
                values[tag] = 0.9 if tag in self.tags[request.sample_id] else 0.1
            else:
                values[tag] = digest[position] / 255
        return values

    def answer(self, name, request):
        """Return the line this fake Jev writes for a request."""
        return self.line(request, name, probabilities=self.probabilities(request))

    @asynccontextmanager
    async def session(self, name):
        """Open a session of method ``name``, and give the ask that answers through it."""

        async def ask(request):
            if request.sample_id in self.failing:
                self.failing.discard(request.sample_id)
                raise TransientApiError("TypeSafe did not answer")
            self.sent.append(request.sample_id)
            return self.answer(name, request)

        yield ask


@dataclass
class DraftLuna:
    """A fake reflection model: it rewrites a piece as a draft that names the iteration.

    Attributes:
        refuse: Iterations whose answer is not JSON, which ``read_proposal`` refuses.
        fail: Iterations whose answer failed: ``failure``'s fields replace the answer's.
        failure: The fields of a failed answer, ``INCOMPLETE`` by default.
        errors: An error to raise, once, at an iteration.
        input_tokens: The input tokens every answer reports (20,000: $0.002).
        output_tokens: The output tokens every answer reports (10,000: $0.005).
        sent: The iteration of every request it answered, in order.
    """

    refuse: set = field(default_factory=set)
    fail: set = field(default_factory=set)
    failure: dict = field(default_factory=lambda: dict(INCOMPLETE))
    errors: dict = field(default_factory=dict)
    input_tokens: int = 20_000
    output_tokens: int = 10_000
    sent: list = field(default_factory=list)

    async def __call__(self, request):
        """Answer one reflection request, or raise the error of its iteration."""
        iteration = request.sample_id
        if iteration in self.errors:
            raise self.errors.pop(iteration)
        self.sent.append(iteration)
        text = "not JSON" if iteration in self.refuse else draft(request.schema_name, iteration)
        answer = LlmAnswer(
            request_sha256=request.sha256(),
            sample_id=iteration,
            method=request.method,
            provider="openai",
            output_text=text,
            status="completed",
            served_model="gpt-6-luna",
            response_id=f"resp-{iteration}",
            request_id=f"req-{iteration}",
            usage={
                "input_tokens": self.input_tokens,
                "cached_tokens": 0,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.output_tokens * 3 // 5,
            },
            extra={},
            started_ns=1_791_000_000_000_000_000,
            latency_s=2.0,
            retries=0,
            created_at="2026-10-10T10:00:00+00:00",
            completed_at="2026-10-10T10:00:02+00:00",
        )
        return replace(answer, **self.failure) if iteration in self.fail else answer


def draft(schema_name, iteration):
    """Return the fake reflection model's rewrite of a piece, as JSON text."""
    if schema_name == "convention":
        return json.dumps({"convention": f"{DRAFT} {iteration}: tags name the method."})
    return json.dumps(
        {
            "true": {
                "definition": f"{DRAFT} {iteration}: the method relies on the idea.",
                "includes": [f"use {iteration} of the idea"],
            },
            "false": {
                "definition": f"{DRAFT} {iteration}: the idea plays no part.",
                "excludes": [f"other ideas, case {iteration}"],
            },
        }
    )


@dataclass
class ListLog:
    """A live log that keeps every line it is told."""

    iterations: list = field(default_factory=list)
    candidates: list = field(default_factory=list)

    def iteration(self, line):
        """Keep a history line."""
        self.iterations.append(line)

    def candidate(self, line):
        """Keep a candidate line."""
        self.candidates.append(line)


class FailingLog(ListLog):
    """A live log whose store is down."""

    def candidate(self, line):
        """Fail to log a candidate line."""
        raise RuntimeError("the store is down")


@pytest.fixture(name="data")
def data_fixture(search_raw_dev):
    """Return the search's data of the 100 synthetic dev problems."""
    return search_data(search_raw_dev)


@pytest.fixture(name="jev")
def jev_fixture(jev_line, search_raw_dev):
    """Return a fake Jev that answers about the 100 synthetic dev problems."""
    return DraftJev(jev_line, search_raw_dev["tags"])


@pytest.fixture(name="committed")
def committed_fixture(search_raw_dev, jev):
    """Return the committed seed answers: the fake Jev's line for every dev problem."""
    seed_requests = questions.requests(search_raw_dev, GEPA_SEED_NAME, SEED_WORDS)
    return [jev.answer(GEPA_SEED_NAME, request) for request in seed_requests]


@pytest.fixture(name="search")
def search_fixture(tmp_path, data, committed, jev):
    """Return a runner of the search in ``tmp_path``, with fakes given by keyword."""

    def run(*, luna=None, fake_jev=None, max_iterations=3, max_cost_usd=1.0, live=None):
        return run_search(
            data,
            tmp_path,
            max_cost_usd=max_cost_usd,
            max_iterations=max_iterations,
            committed_seed_answers=committed,
            jev_session=(fake_jev or jev).session,
            reflection_ask=luna or DraftLuna(),
            live=live,
        )

    return run


def read_lines(path):
    """Return the JSON lines of a file of the record."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_search_data_drops_the_held_out_fold_and_weighs_the_learning_folds(search_raw_dev):
    data = search_data(search_raw_dev)

    assert set(data.raw["fold"]) == {1, 2, 3, 4}
    assert data.learning_ids == sorted(search_raw_dev.index[search_raw_dev["fold"].isin([1, 2, 3])])
    assert data.selection_ids == sorted(search_raw_dev.index[search_raw_dev["fold"] == 4])
    assert list(data.problems.index) == list(data.raw.index)
    assert list(data.labels.columns) == list(FOCUS_TAGS)
    assert data.weights.equals(tag_weights(data.labels.loc[data.learning_ids]))


def test_search_data_needs_the_folds(search_raw_dev):
    with pytest.raises(ValueError, match="no fold column"):
        search_data(search_raw_dev.drop(columns="fold"))
    with pytest.raises(ValueError, match="the selection fold 4 holds 0"):
        search_data(search_raw_dev[search_raw_dev["fold"] != 4])


def test_the_search_folder_is_under_the_experiments_folder(tmp_path):
    assert search_folder(tmp_path) == tmp_path / GEPA_NAME / "search"
    assert search_folder() == EXPERIMENTS_DIR / GEPA_NAME / "search"


def test_the_iteration_cap_counts_the_iterations_done():
    cap = IterationCap(3)

    assert not cap(SimpleNamespace(i=-1)) and cap.iterations == 0
    assert not cap(SimpleNamespace(i=1)) and cap.iterations == 2
    assert cap(SimpleNamespace(i=2)) and cap.reached and cap.iterations == 3


def test_the_cost_cap_compares_what_was_spent_with_the_cap():
    spent = [0.5]
    cap = CostCap(lambda: spent[0], 1.0)

    assert not cap(SimpleNamespace(i=0)) and not cap.reached
    spent[0] = 1.0
    assert cap(SimpleNamespace(i=1)) and cap.reached


@pytest.mark.parametrize("cap", [0, -2])
def test_an_iteration_cap_below_one_is_refused(cap):
    with pytest.raises(ValueError, match="at least 1 iteration"):
        IterationCap(cap)


@pytest.mark.parametrize("cap", [0.0, float("nan")])
def test_a_cost_cap_that_is_not_above_zero_is_refused(cap):
    with pytest.raises(ValueError, match="above 0 USD"):
        CostCap(lambda: 0.0, cap)


def test_the_adapter_refuses_a_problem_of_the_held_out_fold(tmp_path, data, search_raw_dev, jev):
    cache = JevCache(tmp_path / "jev", data.raw, max_cost_usd=1.0, jev_session=jev.session)
    recorder = SearchRecorder(tmp_path, data, cache)
    adapter = JevSearchAdapter(data, cache, Reflector(tmp_path / "r.jsonl", DraftLuna()), recorder)
    held_out = int(search_raw_dev.index[search_raw_dev["fold"] == 0][0])

    with pytest.raises(ValueError, match=r"held-out fold \(fold 0\)"):
        adapter.evaluate([data.learning_ids[0], held_out], candidate_from_words(SEED_WORDS))
    assert not jev.sent


def test_the_iteration_cap_stops_the_search(tmp_path, search):
    outcome = search(max_iterations=3)

    assert outcome.stop_reason == "iterations"
    assert outcome.iterations == 3
    history = read_lines(tmp_path / "history.jsonl")
    assert [line["iteration"] for line in history] == [1, 2, 3]
    stored = json.loads((tmp_path / "outcome.json").read_text(encoding="utf-8"))
    assert list(stored) == OUTCOME_KEYS
    assert stored == {
        "stop_reason": "iterations",
        "iterations": 3,
        "best_index": outcome.best_index,
        "cost_usd": outcome.cost_usd,
        "max_cost_usd": 1.0,
        "max_iterations": 3,
        "gepa_version": "0.1.4",
    }


def test_the_cost_cap_stops_the_search_between_iterations(tmp_path, search):
    outcome = search(max_iterations=50, max_cost_usd=0.02)

    assert outcome.stop_reason == "cost"
    history = read_lines(tmp_path / "history.jsonl")
    assert outcome.iterations == len(history) < 50
    spent = [line["jev_cost_usd"] + line["luna_cost_usd"] for line in history]
    assert sum(spent[:-1]) < 0.02 <= sum(spent)
    assert outcome.cost_usd["total"] == pytest.approx(sum(spent))
    stored = json.loads((tmp_path / "outcome.json").read_text(encoding="utf-8"))
    assert (stored["max_cost_usd"], stored["max_iterations"]) == (0.02, 50)


def test_a_rerun_replays_the_search_from_its_caches_and_sends_nothing(
    tmp_path, search, jev, jev_line, search_raw_dev
):
    luna = DraftLuna()
    first = search(luna=luna, max_iterations=4)
    files = {name: (tmp_path / name).read_bytes() for name in ("history.jsonl", "candidates.jsonl")}
    assert jev.sent and luna.sent

    again_jev = DraftJev(jev_line, search_raw_dev["tags"])
    again_luna = DraftLuna()
    second = search(luna=again_luna, fake_jev=again_jev, max_iterations=4)

    assert not again_jev.sent and not again_luna.sent
    assert second == first
    for name, content in files.items():
        assert (tmp_path / name).read_bytes() == content


def test_a_rewrite_that_helps_is_accepted_and_scored_on_the_selection_fold(tmp_path, search):
    outcome = search(max_iterations=1)

    [line] = read_lines(tmp_path / "history.jsonl")
    assert line["piece"] == "convention" and line["parent"] == 0
    assert line["accepted"] and line["candidate"] == 1
    assert line["child_sum"] > line["parent_sum"]
    seed, child = read_lines(tmp_path / "candidates.jsonl")
    assert child["index"] == 1 and child["parents"] == [0] and child["iteration"] == 1
    assert child["piece"] == "convention"
    assert child["words"]["convention"].startswith(DRAFT)
    assert child["selection_mean"] > seed["selection_mean"]
    assert outcome.best_index == 1


def test_the_seed_is_candidate_0_scored_from_the_committed_answers(
    tmp_path, search, data, committed, jev
):
    search(luna=DraftLuna(refuse={1}), max_iterations=1)

    assert not jev.sent
    [seed] = read_lines(tmp_path / "candidates.jsonl")
    assert list(seed) == CANDIDATE_KEYS
    assert seed["index"] == 0 and seed["parents"] == [] and seed["iteration"] == 0
    assert seed["piece"] is None
    assert seed["words"] == SEED_WORDS.settings()
    assert seed["words_sha256"] == SEED_WORDS.sha256()
    selection = data.selection_ids
    lines = [answer for answer in committed if answer.sample_id in set(selection)]
    probabilities = answers.probabilities(lines, pd.Index(selection, name="sample_id"))
    labels = data.labels.loc[selection]
    expected = problem_scores(probabilities, labels, data.weights)
    assert seed["selection_scores"] == pytest.approx(
        {str(sample_id): score for sample_id, score in expected.items()}
    )
    assert seed["selection_mean"] == pytest.approx(expected.mean())
    brier = balanced_brier(probabilities, labels)
    assert seed["selection_brier"] == pytest.approx(brier.to_dict())


def test_a_refused_rewrite_is_recorded_and_its_child_is_skipped(tmp_path, search, data):
    search(luna=DraftLuna(refuse={1}), max_iterations=1)

    [line] = read_lines(tmp_path / "history.jsonl")
    assert list(line) == HISTORY_KEYS
    assert line["refusal"] == "invalid_json"
    assert line["child_sum"] is None and line["candidate"] is None and not line["accepted"]
    assert line["piece"] in PIECES and line["parent"] == 0
    assert isinstance(line["parent_sum"], float)
    assert len(line["minibatch"]) == 40 and set(line["minibatch"]) <= set(data.learning_ids)
    assert line["jev_cost_usd"] == 0.0
    assert line["luna_cost_usd"] == pytest.approx(0.007)
    assert line["luna_input_tokens"] == 20_000 and line["luna_output_tokens"] == 10_000


def test_a_live_log_is_told_every_line_of_the_record(tmp_path, search):
    log = ListLog()

    search(max_iterations=3, live=log)

    assert log.iterations == read_lines(tmp_path / "history.jsonl")
    assert log.candidates == read_lines(tmp_path / "candidates.jsonl")


def test_an_error_in_the_recorder_stops_the_search_instead_of_being_dropped(tmp_path, search):
    with pytest.raises(RuntimeError, match="the store is down"):
        search(max_iterations=3, live=FailingLog())

    assert not (tmp_path / "outcome.json").exists()


def test_a_reflection_model_without_an_answer_interrupts_and_a_rerun_resumes(tmp_path, search):
    with pytest.raises(SearchInterrupted, match="at iteration 2"):
        search(luna=DraftLuna(errors={2: TransientApiError("timeout")}), max_iterations=3)

    assert [line["iteration"] for line in read_lines(tmp_path / "history.jsonl")] == [1]
    assert not (tmp_path / "outcome.json").exists()
    luna = DraftLuna()
    outcome = search(luna=luna, max_iterations=3)
    assert luna.sent == [2, 3]
    assert outcome.iterations == 3


def test_any_other_error_of_a_proposal_is_raised_not_dropped_by_gepa(search):
    with pytest.raises(ValueError, match="broken reflection"):
        search(luna=DraftLuna(errors={1: ValueError("broken reflection")}))


def test_jev_without_an_answer_interrupts_the_search(search, data, jev):
    jev.failing.update(data.learning_ids)

    with pytest.raises(SearchInterrupted, match="run the search again"):
        search(max_iterations=2)


def test_the_search_writes_no_pickle(tmp_path, search):
    search(max_iterations=3)

    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written
    assert not [path for path in written if starts_like_pickle(path)]


def test_the_progress_line_says_what_became_of_each_child(search, capsys):
    search(luna=DraftLuna(refuse={2}), max_iterations=3)

    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("iter")]
    assert printed[0].startswith("iteration 1: convention of candidate 0, accepted as candidate 1")
    assert "refused (invalid_json)" in printed[1]
    assert printed[2].startswith("iteration 3: ") and "rejected" in printed[2]
    assert all(line.endswith(" spent") for line in printed)


def test_an_error_building_the_records_is_raised_not_dropped_by_gepa(search, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("broken records")

    monkeypatch.setattr(tuning, "convention_records", broken)

    with pytest.raises(ValueError, match="broken records"):
        search(max_iterations=1)


def test_the_recorder_keeps_the_first_error_of_its_callbacks(tmp_path, data, jev, capsys):
    cache = JevCache(tmp_path / "jev", data.raw, max_cost_usd=1.0, jev_session=jev.session)
    recorder = SearchRecorder(tmp_path, data, cache)
    recorder.raise_kept()

    recorder.keep(RuntimeError("first"))
    assert capsys.readouterr().err == ""
    recorder.keep(ValueError("second"))

    # The second error is not kept, but it is not dropped silently either.
    assert "ValueError: second" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="first"):
        recorder.raise_kept()


@pytest.mark.parametrize("failure", [INCOMPLETE, REFUSED], ids=["incomplete", "refused"])
def test_failed_reflections_are_recorded_and_the_search_still_stops_at_its_cap(
    tmp_path, search, data, jev, failure
):
    # The reflection model fails at every iteration but the last (its reasoning used up its
    # output tokens, or it refused): each failed rewrite leaves its piece unchanged, so GEPA
    # skips the child, which Jev never scores; the failed answer is still paid for.
    luna = DraftLuna(fail={1, 2, 3, 4}, failure=failure)

    outcome = search(luna=luna, max_iterations=5)

    history = read_lines(tmp_path / "history.jsonl")
    failed, last = history[:4], history[4]
    assert [line["refusal"] for line in failed] == ["failed_answer"] * 4
    assert all(line["child_sum"] is None and line["candidate"] is None for line in failed)
    assert all(not line["accepted"] and line["parent"] == 0 for line in failed)
    assert [line["jev_cost_usd"] for line in failed] == [0.0] * 4
    assert [line["luna_cost_usd"] for line in history] == pytest.approx([0.007] * 5)
    assert last["refusal"] is None and last["child_sum"] is not None
    assert luna.sent == [1, 2, 3, 4, 5]
    # Jev scored the last child alone: its minibatch, then the selection fold once accepted.
    assert len(jev.sent) == 40 + (len(data.selection_ids) if last["accepted"] else 0)
    assert outcome.stop_reason == "iterations" and outcome.iterations == 5
    stored = json.loads((tmp_path / "outcome.json").read_text(encoding="utf-8"))
    assert stored["iterations"] == len(history) == history[-1]["iteration"]
    assert stored["cost_usd"]["luna"] == pytest.approx(5 * 0.007)


def test_a_cap_reached_while_jev_scores_a_child_on_the_selection_fold_resumes_at_a_higher_cap(
    tmp_path, search, data, jev_line, search_raw_dev
):
    # A cheap reflection model ($0.00001 an answer), so that Jev's answers reach the cap of $0.02
    # inside iteration 4: iterations 1 to 3 cost $0.0143 of them, iteration 4's minibatches
    # $0.0046 more, and Jev stops past the cap while it scores the accepted child, candidate 3,
    # on the selection fold. Iteration 1's rewrite is refused, so that later ones are accepted.
    first_log = ListLog()
    with pytest.raises(CostCapSpent, match="unanswered at the cost cap"):
        search(
            luna=DraftLuna(refuse={1}, input_tokens=100, output_tokens=0),
            max_iterations=5,
            max_cost_usd=0.02,
            live=first_log,
        )
    files = {name: (tmp_path / name).read_bytes() for name in ("history.jsonl", "candidates.jsonl")}
    held = {path.name: read_answers(path) for path in (tmp_path / "jev").glob("*.jsonl")}
    assert [line["iteration"] for line in first_log.iterations] == [1, 2, 3]
    assert [line["index"] for line in first_log.candidates] == [0, 1, 2]
    assert not (tmp_path / "outcome.json").exists()

    second_log = ListLog()
    again_jev = DraftJev(jev_line, search_raw_dev["tags"])
    again_luna = DraftLuna(refuse={1}, input_tokens=100, output_tokens=0)
    outcome = search(
        luna=again_luna, fake_jev=again_jev, max_iterations=5, max_cost_usd=1.0, live=second_log
    )

    # The replay writes the first run's lines again, byte for byte, with the same costs.
    assert all((tmp_path / name).read_bytes().startswith(files[name]) for name in files)
    assert second_log.iterations[:3] == first_log.iterations
    assert second_log.candidates[:3] == first_log.candidates
    # The interrupted evaluation: candidate 3, accepted at iteration 4, had part of the
    # selection fold answered; the rerun asked Jev for the rest, and Luna for iteration 5 only.
    child = second_log.candidates[3]
    assert second_log.iterations[3]["accepted"] and second_log.iterations[3]["candidate"] == 3
    assert child["index"] == 3 and child["iteration"] == 4
    selection = set(data.selection_ids)
    answered = {a.sample_id for a in held[f"{child['words_sha256']}.jsonl"]} & selection
    assert 0 < len(answered) < len(selection)
    assert set(again_jev.sent[: len(selection) - len(answered)]) == selection - answered
    assert again_luna.sent == [5]
    # No answer was paid for twice, and none on disk is left out of the search's cost.
    assert JevCache(tmp_path / "jev", data.raw, max_cost_usd=1.0).spent_usd() == pytest.approx(
        outcome.cost_usd["jev"]
    )
    assert Reflector(tmp_path / "reflections.jsonl").total_cost == pytest.approx(
        outcome.cost_usd["luna"]
    )
    assert outcome.stop_reason == "iterations" and outcome.iterations == 5


def test_an_interrupted_rerun_of_a_finished_search_leaves_no_outcome(tmp_path, search):
    # export reads outcome.json as the sign that a cap stopped the search: a rerun that an
    # error interrupts must not leave the finished run's file beside its own partial lines.
    search(max_iterations=2)
    assert (tmp_path / "outcome.json").exists()

    with pytest.raises(SearchInterrupted, match="at iteration 3"):
        search(luna=DraftLuna(errors={3: TransientApiError("timeout")}), max_iterations=4)

    assert not (tmp_path / "outcome.json").exists()
    assert [line["iteration"] for line in read_lines(tmp_path / "history.jsonl")] == [1, 2]
