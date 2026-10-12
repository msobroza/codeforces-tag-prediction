"""Fixtures of the TypeSafe Jev tests: a fake ``typesafe_sdk``, answers-file lines and a fake Jev.

No request leaves the process. ``server`` puts a fake ``typesafe_sdk`` in ``sys.modules``, where
it wins over an installed SDK. ``noul_response`` builds what that SDK returns for a request, and
``jev_line`` what the toolkit's answers file holds for it. ``jev_api`` replaces
``client.jev_session`` as a whole, for the tests of what sits above the client.

``search_run`` writes a finished search's folder by hand, as the search's recorder writes it, for
the tests of its export and of its frozen record.

The fake mirrors typesafe-sdk 0.7.2: an error has a ``status`` (never ``status_code``), the
response's ``request_id`` raises when the header is absent, and a timeout is not an API error.
"""

import asyncio
import copy
import json
import sys
import types
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import USAGE_KEYS, LlmAnswer
from codeforces_tags.methods.typesafe_jev.objective import problem_scores, tag_weights
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_LINES_FILE,
    GEPA_NAME,
    HISTORY_LINES_FILE,
    MODEL,
    OUTCOME_FILE,
    PROVIDER,
    SEARCH_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS

# The word of each focus tag that the fake Jev looks for in a state, in FOCUS_TAGS order.
TAG_WORDS = {
    "math": "sum",
    "graphs": "vertices",
    "strings": "letters",
    "number theory": "gcd",
    "trees": "leaves",
    "geometry": "circle",
    "games": "wins",
    "probabilities": "dice",
}
# The fake Jev's probability of a tag: by (its word is in the statement, it is in the solution).
WORD_PROBABILITY = {(False, False): 0.1, (False, True): 0.2, (True, False): 0.8, (True, True): 0.9}


class TypeSafeError(Exception):
    """Stands in for ``typesafe_sdk.TypeSafeError``: the base of every SDK failure.

    A missing API key raises it, when the client is built.
    """


class TypeSafeAPIError(TypeSafeError):
    """An HTTP error response: its ``status``, the response's ``request_id`` and its ``body``.

    The message echoes the body, as the SDK's does: a test can prove that it never reaches a line.
    """

    def __init__(self, status, request_id=None, body=None):
        """Hold the status, the request id (None without the header) and the body."""
        super().__init__(f"POST https://api.typesafe.ai/v1/systemone: {status} {body}")
        self.status = status
        self.request_id = request_id
        self.body = body


class TypeSafeBadRequestError(TypeSafeAPIError):
    """A 400."""


class TypeSafeAuthenticationError(TypeSafeAPIError):
    """A 401."""


class TypeSafePermissionDeniedError(TypeSafeAPIError):
    """A 403."""


class TypeSafeNotFoundError(TypeSafeAPIError):
    """A 404."""


class TypeSafeUnprocessableEntityError(TypeSafeAPIError):
    """A 422."""


class TypeSafeRateLimitError(TypeSafeAPIError):
    """A 429."""


class TypeSafeInternalServerError(TypeSafeAPIError):
    """Every status from 500 up, 529 included: the SDK has no class of its own for 529."""


class TypeSafeAPIResponseValidationError(TypeSafeAPIError):
    """A 200 whose body fails the SDK's validation: an API error of status 200."""

    def __init__(self, request_id=None, field_path=""):
        """Hold the request id and the path of the field that failed."""
        super().__init__(200, request_id)
        self.field_path = field_path


class TypeSafeAPIConnectionError(TypeSafeError, ConnectionError):
    """No HTTP response at all. It is not an API error: it has no status."""


class TypeSafeAPITimeoutError(TypeSafeAPIConnectionError, TimeoutError):
    """A timeout, a connection error of its own kind (and not an API error either)."""


@dataclass(frozen=True)
class RetryPolicy:
    """The retry policy a client is built with (the real one has more fields)."""

    max_retries: int = 2
    timeout: float | None = 30.0


@dataclass(frozen=True)
class NoulAnswer:
    """The answer to a Noul question: a probability, which the SDK does not range-check."""

    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class Usage:
    """The tokens a response reports; either count may be None."""

    input_tokens: int | None
    output_tokens: int | None


class SystemOneResponse:
    """A 200 response: the model that answered, its usage and its answers."""

    def __init__(self, model, usage, answers, request_id):
        """Hold the parts of the response; a ``request_id`` of None stands for an absent header."""
        self.model = model
        self.usage = usage
        self.answers = answers
        self.header = request_id

    @property
    def nouls(self):
        """Return the answers that are probabilities, by question id: the SDK's own view."""
        return {name: a for name, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def request_id(self):
        """Return the request id, or raise ``TypeSafeError`` when the header was absent."""
        if self.header is None:
            raise TypeSafeError("The response did not include a request ID.")
        return self.header


# The names that the fake module ``typesafe_sdk`` takes from this file.
SDK_CLASSES = (
    TypeSafeError,
    TypeSafeAPIError,
    TypeSafeBadRequestError,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
    TypeSafeNotFoundError,
    TypeSafeUnprocessableEntityError,
    TypeSafeRateLimitError,
    TypeSafeInternalServerError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    RetryPolicy,
    NoulAnswer,
    Usage,
    SystemOneResponse,
)


@dataclass
class FakeServer:
    """What the fake TypeSafe API answers and what it was sent.

    Each ``system_one`` call takes the first outcome of ``outcomes``: a response, or an exception
    that the call raises.

    Attributes:
        sdk: The fake ``typesafe_sdk`` module, which is in ``sys.modules``.
        outcomes: What the calls return or raise, first in first out.
        calls: The arguments of every call: ``state``, ``model``, ``questions`` and the options.
        clients: The constructor options of every client that was built.
        closed: How many clients were closed.
        key_error: What building a client raises, as a missing key does; None for no error.
    """

    sdk: types.ModuleType
    outcomes: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    closed: int = 0
    key_error: BaseException | None = None


class FakeTypeSafeClient:
    """Stands in for ``typesafe_sdk.AsyncTypeSafeClient``, usable only in the loop it opened in."""

    def __init__(self, server, **options):
        """Record the options, or raise the server's key error: the SDK reads the key here."""
        if server.key_error is not None:
            raise server.key_error
        server.clients.append(options)
        self.server = server
        self.home_loop = None
        self.is_open = False

    async def __aenter__(self):
        """Open the client in the running loop."""
        self.home_loop = asyncio.get_running_loop()
        self.is_open = True
        return self

    async def __aexit__(self, *exc_info):
        """Close the client, in the loop that opened it."""
        assert asyncio.get_running_loop() is self.home_loop
        self.is_open = False
        self.server.closed += 1

    async def system_one(self, state, questions, *, model=None, **options):
        """Record the call, then return the next outcome or raise it.

        A client used in another loop fails every other call in httpx: a client is opened in the
        running loop and used only there, until it is closed.
        """
        assert self.is_open and asyncio.get_running_loop() is self.home_loop
        call = {"state": state, "model": model, "questions": questions, **options}
        self.server.calls.append(call)
        outcome = self.server.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(name="server")
def server_fixture(monkeypatch):
    """Put a fake ``typesafe_sdk`` in ``sys.modules`` and return the fake server behind it.

    It wins over an installed SDK, and is forgotten after the test.
    """
    sdk = types.ModuleType("typesafe_sdk")
    server = FakeServer(sdk)
    for sdk_class in SDK_CLASSES:
        setattr(sdk, sdk_class.__name__, sdk_class)
    sdk.AsyncTypeSafeClient = lambda **options: FakeTypeSafeClient(server, **options)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)
    return server


@pytest.fixture(name="noul_response")
def noul_response_fixture():
    """Return a builder of the response to one request, with the fields overridable by keyword.

    ``probabilities`` maps focus tags to floats, answered under their question ids (``"number
    theory"`` is ``"number_theory"``); a tag left out is not answered, and None answers 0.5 for
    every focus tag. ``others`` maps question ids to answers that are not probabilities, which
    ``nouls`` leaves out. ``request_id=None`` is a response without the header.
    """

    # pylint: disable-next=too-many-arguments
    def build(
        probabilities=None,
        *,
        model=MODEL,
        input_tokens=2500,
        output_tokens=0,
        request_id="req-1",
        others=None,
    ):
        probabilities = dict.fromkeys(FOCUS_TAGS, 0.5) if probabilities is None else probabilities
        answers = {tag.replace(" ", "_"): NoulAnswer(p) for tag, p in probabilities.items()}
        answers.update(others or {})
        return SystemOneResponse(model, Usage(input_tokens, output_tokens), answers, request_id)

    return build


@pytest.fixture(name="jev_line")
def jev_line_fixture():
    """Return a builder of one line of a Jev answers file, with the fields overridable by keyword.

    ``build(request, method, probabilities, **changes)``: ``request`` is anything with a
    ``sample_id`` and a ``sha256()`` (a ``JevRequest``), or None for problem 1 (or the
    ``sample_id`` of ``changes``) under a hash made of its number. ``probabilities`` maps focus
    tags to floats, 0.0 for a tag left out, and None gives 0.5 for every focus tag. A status
    other than ``completed`` gives a line with no text, no model and no tokens, as a failed
    request has, and ``failed`` follows the status. ``changes`` override every field.
    """

    def build(request=None, method="jev-statement", probabilities=None, **changes):
        sample_id = changes.get("sample_id", 1) if request is None else request.sample_id
        completed = changes.get("status", "completed") == "completed"
        probabilities = dict.fromkeys(FOCUS_TAGS, 0.5) if probabilities is None else probabilities
        scored = {tag: probabilities.get(tag, 0.0) for tag in FOCUS_TAGS}
        values = {
            "request_sha256": f"{sample_id:064x}" if request is None else request.sha256(),
            "sample_id": sample_id,
            "method": method,
            "provider": PROVIDER,
            "output_text": json.dumps(scored, separators=(",", ":")) if completed else None,
            "status": "completed",
            "served_model": MODEL if completed else None,
            "response_id": f"req-{sample_id}",
            "request_id": f"req-{sample_id}",
            "usage": dict.fromkeys(USAGE_KEYS, 0) | ({"input_tokens": 2500} if completed else {}),
            "extra": {},
            "started_ns": 1_791_000_000_000_000_000 + sample_id * 1_000_000_000,
            "latency_s": 0.4,
            "retries": 0,
            "created_at": "2026-10-08T10:00:00+00:00",
            "completed_at": "2026-10-08T10:00:00.400000+00:00",
            "failed": not completed,
        }
        return LlmAnswer(**(values | changes))

    return build


@dataclass
class FakeJev:
    """A fake Jev that stands in for ``client.jev_session``: it asks nothing and answers at once.

    A state's probability of a tag follows the words of ``TAG_WORDS`` it holds. A problem with an
    entry in ``outcomes`` gets it, once: an exception is raised by the ask, a mapping is the
    changes to its answer line (e.g. ``{"status": "api_error:422"}``).

    Attributes:
        line: The ``jev_line`` builder.
        sent: The ``(method, sample_id)`` of every request, in the order they were asked.
        sessions: The method of every session that was opened.
        outcomes: The scripted outcome of a problem, by sample id.
    """

    line: Callable[..., LlmAnswer]
    sent: list = field(default_factory=list)
    sessions: list = field(default_factory=list)
    outcomes: dict = field(default_factory=dict)

    def probabilities(self, state):
        """Return the probability of each focus tag for a state, by the words it holds."""
        statement = state["statement"].lower()
        solution = state.get("solution", "").lower()
        return {
            tag: WORD_PROBABILITY[(word in statement, word in solution)]
            for tag, word in TAG_WORDS.items()
        }

    def answer(self, name, request, **changes):
        """Return the line that this fake Jev writes for a request of method ``name``."""
        probabilities = self.probabilities(request.state)
        return self.line(request, name, probabilities=probabilities, **changes)

    @asynccontextmanager
    async def session(self, name):
        """Open a session of method ``name``, and give the ask that answers through it."""
        self.sessions.append(name)

        async def ask(request):
            self.sent.append((name, request.sample_id))
            outcome = self.outcomes.pop(request.sample_id, {})
            if isinstance(outcome, BaseException):
                raise outcome
            return self.answer(name, request, **outcome)

        yield ask


@pytest.fixture(name="jev_api")
def jev_api_fixture(monkeypatch, jev_line):
    """Replace ``client.jev_session`` by a fake Jev and return it.

    The path is resolved when the fixture runs, so this file does not import the client.
    """
    fake = FakeJev(jev_line)
    monkeypatch.setattr("codeforces_tags.methods.typesafe_jev.client.jev_session", fake.session)
    return fake


# The kinds of problem of search_raw_dev: a focus tag (or none), the start of the statement and
# the solution. Each problem's statement and solution end with its number, so no two problems
# share a request.
SEARCH_KINDS = (
    ("math", "Compute the sum of the digits of the number", "print(sum(map(int, input())))"),
    ("graphs", "Find the shortest path between two vertices", "print(bfs(read_graph()))"),
    ("strings", "Reverse the letters of the given word", "print(input()[::-1])"),
    ("number theory", "Print the gcd of the two numbers", "print(gcd(*read_ints()))"),
    ("trees", "Count the leaves of the rooted tree", "print(count_leaves(read_tree()))"),
    ("geometry", "Find the area of the circle with the radius", "print(area(read_float()))"),
    ("games", "Tell who wins the game of stones", "print('Alice' if read_int() % 2 else 'Bob')"),
    ("probabilities", "Find the probability that the dice shows six", "print(1 / 6)"),
    ("greedy", "Sort the array and print its largest element", "print(max(read_ints()))"),
)
SEARCH_PROBLEMS = 100  # 20 in each of the five dev folds


@pytest.fixture(name="search_raw_dev")
def search_raw_dev_fixture():
    """Return 100 raw dev problems, 20 in each fold 0 to 4, as ``select_part`` gives them.

    Sample ids 1000 to 1099; problem ``1000 + p`` is in fold ``p % 5`` and of the kind
    ``SEARCH_KINDS[p // 5 % 9]``, so every fold holds at least two positives of each focus tag
    and problems without one. Raw rows with every column of ``RAW_INPUT_COLUMNS``, the tags,
    the fold and a group of their own.
    """
    rows = []
    for position in range(SEARCH_PROBLEMS):
        tag, statement, code = SEARCH_KINDS[position // 5 % len(SEARCH_KINDS)]
        rows.append(
            {
                "prob_desc_description": f"{statement}, exercise {position}.",
                "prob_desc_input_spec": "The first line holds $$$n$$$.",
                "prob_desc_output_spec": "Print the answer.",
                "prob_desc_notes": float("nan"),
                "prob_desc_sample_inputs": json.dumps(["1"]),
                "prob_desc_sample_outputs": json.dumps(["1"]),
                "source_code": f"{code}\n# exercise {position}\n",
                "tags": [tag],
                "fold": position % 5,
                "group": position,
            }
        )
    index = pd.Index(range(1000, 1000 + SEARCH_PROBLEMS), name="sample_id")
    return pd.DataFrame(rows, index=index)


# A finished search's problems: 16 of the learning folds, where each focus tag has 2 positives
# (prevalence 1/8), 8 of the selection fold, where each has 1, and 2 of the held-out fold, which
# only the seed's frozen scores hold.
SEARCH_LEARNING = list(range(101, 117))
SEARCH_SELECTION = list(range(201, 209))
SEARCH_HELD_OUT = [301, 302]
# The conventions of candidates 1 to 3. Candidate 1 keeps the first 8 words of the seed's
# convention, which a dataset text holds too: the seed's own sequences are exempt. Candidate 2
# copies 3 sequences of 8 words from the robot's text. Candidate 3 ties with candidate 1.
SEARCH_CONVENTIONS = {
    1: "Codeforces tags name the ideas and techniques of the intended solution only.",
    2: "Codeforces tags follow how the robot walks along the grid and collects every coin.",
    3: "Codeforces tags name the techniques that the intended solution needs.",
}
# The texts the dataset-text check reads, as dataset_texts would give them.
SEARCH_DATASET_TEXTS = [
    "The robot walks along the grid and collects every coin it sees.",
    "Codeforces tags name the ideas and techniques of every round.",
    "print(sum(a))",
]
# What outcome.json holds once the search stopped at its iteration cap, after its 4 iterations,
# in the order of the search's recorder.
SEARCH_OUTCOME = {
    "stop_reason": "iterations",
    "iterations": 4,
    "best_index": 2,
    "cost_usd": {"jev": 0.04, "luna": 0.008, "total": 0.048},
    "max_cost_usd": 15.0,
    "max_iterations": 4,
    "gepa_version": "0.1.4",
}


def _search_labels():
    """Return the 0/1 focus tags of the learning then the selection problems."""
    tags = range(len(FOCUS_TAGS))
    learning = range(len(SEARCH_LEARNING))
    rows = [[int(row % len(FOCUS_TAGS) == tag) for tag in tags] for row in learning]
    rows += [[int(row == tag) for tag in tags] for row in range(len(SEARCH_SELECTION))]
    index = pd.Index([*SEARCH_LEARNING, *SEARCH_SELECTION], name="sample_id")
    return pd.DataFrame(rows, index=index, columns=list(FOCUS_TAGS))


def _seed_scores(labels):
    """Return the seed's dev probabilities: 0.7 on a positive, 0.2 on a negative, 0.5 held out."""
    held_out = pd.DataFrame(
        0.5, index=pd.Index(SEARCH_HELD_OUT, name="sample_id"), columns=list(FOCUS_TAGS)
    )
    return pd.concat([held_out, labels * 0.5 + 0.2])


@dataclass
class SearchRun:
    """A finished search's folder, written by hand as the search's recorder writes it.

    ``write_finished`` writes the seed (candidate 0, at iteration 0) and three candidates
    (``SEARCH_CONVENTIONS``), iterations 1 to 4, the third refused, and the outcome of a search
    stopped at its iteration cap. Each accepted candidate's parent is its iteration's parent.

    Attributes:
        folder: The search folder, ``<experiments>/jev-statement-code-gepa/search``.
        data: The ``SearchData`` of the learning and selection problems.
        seed_scores: The seed's frozen dev probabilities, held-out problems included.
        texts: The dataset texts that the dataset-text check reads.
        outcome: What ``outcome.json`` holds; ``write_outcome`` writes it.
        words: The words of each candidate, by index, once ``write_finished`` ran.
    """

    folder: Path
    data: Any  # a tuning.SearchData, imported by the fixture only
    seed_scores: pd.DataFrame
    texts: list = field(default_factory=lambda: list(SEARCH_DATASET_TEXTS))
    outcome: dict = field(default_factory=lambda: copy.deepcopy(SEARCH_OUTCOME))
    words: dict = field(default_factory=dict)

    def seed_line(self):
        """Return candidate 0's line: the seed's words, scored from its frozen probabilities."""
        ids = self.data.selection_ids
        scores = problem_scores(
            self.seed_scores.loc[ids], self.data.labels.loc[ids], self.data.weights
        )
        line = self.candidate_line(0, SEED_WORDS, float(scores.mean()), iteration=0, parents=())
        line["piece"] = None
        line["selection_scores"] = {
            str(sample_id): float(score) for sample_id, score in scores.items()
        }
        return line

    def candidate_line(self, index, words, mean, *, iteration, parents=(0,)):
        """Return the line of a candidate that changed the convention, of the given mean score."""
        return {
            "index": index,
            "parents": list(parents),
            "iteration": iteration,
            "piece": "convention",
            "words": words.settings(),
            "words_sha256": words.sha256(),
            "selection_scores": {str(sample_id): mean for sample_id in self.data.selection_ids},
            "selection_mean": mean,
            "selection_brier": dict.fromkeys(FOCUS_TAGS, 0.1),
        }

    def history_line(self, iteration, *, parent=0, candidate=None, refusal=None):
        """Return an iteration's line: accepted as ``candidate``, or refused for ``refusal``."""
        return {
            "iteration": iteration,
            "parent": parent,
            "piece": "convention",
            "minibatch": self.data.learning_ids[:4],
            "parent_sum": 3.5,
            "child_sum": None if refusal else 3.75,
            "accepted": candidate is not None,
            "refusal": refusal,
            "candidate": candidate,
            "jev_cost_usd": 0.01,
            "luna_cost_usd": 0.002,
            "luna_input_tokens": 1500,
            "luna_output_tokens": 800,
        }

    def write_lines(self, file_name, lines):
        """Write one JSON line per item into a file of the search folder, which it creates."""
        self.folder.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(line) + "\n" for line in lines)
        (self.folder / file_name).write_text(text, encoding="utf-8")

    def write_outcome(self):
        """Write ``outcome`` as the search folder's outcome.json."""
        (self.folder / OUTCOME_FILE).write_text(json.dumps(self.outcome), encoding="utf-8")

    def write_finished(self):
        """Write the seed, three candidates, four iterations and the outcome; return the lines."""
        self.words = {0: SEED_WORDS} | {
            index: replace(SEED_WORDS, convention=text)
            for index, text in SEARCH_CONVENTIONS.items()
        }
        candidates = [
            self.seed_line(),
            self.candidate_line(1, self.words[1], 0.95, iteration=1),
            self.candidate_line(2, self.words[2], 0.99, iteration=2),
            self.candidate_line(3, self.words[3], 0.95, iteration=4, parents=(1,)),
        ]
        history = [
            self.history_line(1, candidate=1),
            self.history_line(2, candidate=2),
            self.history_line(3, parent=2, refusal="too_long"),
            self.history_line(4, parent=1, candidate=3),
        ]
        self.write_lines(CANDIDATE_LINES_FILE, candidates)
        self.write_lines(HISTORY_LINES_FILE, history)
        self.write_outcome()
        return candidates


@pytest.fixture(name="search_run")
def search_run_fixture(tmp_path):
    """Return a ``SearchRun`` whose folder is ``tmp_path/experiments/<GEPA_NAME>/search``.

    Nothing is written yet. ``tuning`` imports gepa and the Jev client, so it is imported here,
    when a test of a search asks for the fixture, not when this file is.
    """
    # pylint: disable-next=import-outside-toplevel
    from codeforces_tags.methods.typesafe_jev.tuning import SearchData

    labels = _search_labels()
    data = SearchData(
        raw=pd.DataFrame(index=labels.index),
        problems=pd.DataFrame(index=labels.index),
        labels=labels,
        learning_ids=list(SEARCH_LEARNING),
        selection_ids=list(SEARCH_SELECTION),
        weights=tag_weights(labels.loc[SEARCH_LEARNING]),
    )
    folder = tmp_path / "experiments" / GEPA_NAME / SEARCH_FOLDER
    return SearchRun(folder=folder, data=data, seed_scores=_seed_scores(labels))
