"""The Jev answers of a search: one answers file per set of candidate words (docs/adr/0010).

A search asks Jev about the same problems with many sets of words. Each set has its own file,
``<folder>/<words sha256>.jsonl``, in the LLM toolkit's answers-file format (``llm.store``),
written by ``runner.send``: one line per request hash, never rewritten. A request that has a
line is never sent again, so running a search again replays its answers at no cost. One file
per set of words also keeps each append small: ``append_answer`` reads the whole file.

The seed's file is filled first from the committed dev answers of ``jev-statement-code``
(``prefill_seed``): the searched method's state fields are that method's, so the seed's requests
are its requests, byte for byte, and the seed costs nothing.

Only the learning folds and the selection fold are ever asked about: a problem of the held-out
fold is refused (``SEARCH_HELD_OUT_FOLD``).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pandas as pd

from codeforces_tags.llm.answers import answer_cost_usd, problem_list
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.llm.store import read_answers
from codeforces_tags.methods.typesafe_jev import client, runner
from codeforces_tags.methods.typesafe_jev.answers import probabilities as answer_probabilities
from codeforces_tags.methods.typesafe_jev.client import Ask
from codeforces_tags.methods.typesafe_jev.questions import JevRequest, noul_questions, states
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    MODEL,
    SEARCH_HELD_OUT_FOLD,
    SEARCH_LEARNING_FOLDS,
    SEARCH_SELECTION_FOLD,
)
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS, JevWords
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The folds a search reads: its learning folds, then its selection fold.
SEARCH_FOLDS: tuple[int, ...] = (*SEARCH_LEARNING_FOLDS, SEARCH_SELECTION_FOLD)
# Opens a Jev session for a method and yields its Ask: client.jev_session, or a test's fake.
JevSession = Callable[[str], AbstractAsyncContextManager[Ask]]
_NEWLINE = b"\n"


class SearchInterrupted(RuntimeError):
    """The search cannot go on now; running it again resumes it where it stopped.

    Jev or the reflection model left a request unanswered (a transient error, a run of failed
    answers), or Jev's answers reached the search's cost cap inside one call (``CostCapSpent``,
    after which only a higher cap goes on). Every answer paid for is on disk, so the rerun replays
    the search from its caches and sends only what is left.
    """


class CostCapSpent(SearchInterrupted):
    """Jev's answers spent the search's cost cap inside a call: only a higher cap goes on.

    The same cap stops a rerun at the same point, before it sends anything: run the search again
    with a higher cap, and it replays every answer paid for, then sends what is left.
    """


class JevCache:
    """Jev's probabilities for any set of words, each answer asked once and kept on disk.

    Attributes:
        folder: The folder of the answers files, ``<search folder>/jev``.
        max_cost_usd: The cap on what Jev's answers in the folder may cost, the seed's left out.
        name: The method whose state fields the requests hold, and whose name the lines carry.
        jev_session: Opens a Jev session (``client.jev_session``'s signature); None to look up
            ``client.jev_session`` when a request is sent, which tests replace.
    """

    def __init__(
        self,
        folder: Path,
        raw_problems: pd.DataFrame,
        *,
        max_cost_usd: float,
        name: str = GEPA_NAME,
        jev_session: JevSession | None = None,
    ) -> None:
        """Build the states of the problems of the learning and selection folds, once.

        Args:
            folder: The folder of the answers files; created when the first one is written.
            raw_problems: Raw dev problems indexed by ``sample_id``, with their ``fold`` column
                (``select_part``'s rows); those of the held-out fold are dropped.
            max_cost_usd: The cap, in USD, on what Jev's answers in ``folder`` may cost, the
                seed's file left out: a call that would go over it stops at it.
            name: The method whose requests are built, a key of ``settings.NAMES``.
            jev_session: The Jev session to send through; None for ``client.jev_session``.

        Raises:
            ValueError: If ``max_cost_usd`` is not above 0 (NaN included), or as
                ``raw_to_problems`` and ``questions.states``.
        """
        # `not max_cost_usd > 0` refuses NaN too, which compares false with everything.
        if not max_cost_usd > 0:
            raise ValueError(f"A cost cap must be above 0 USD, not {max_cost_usd}.")
        self.folder = Path(folder)
        self.max_cost_usd = max_cost_usd
        self.name = name
        self.jev_session = jev_session
        searched = raw_problems[raw_problems["fold"].isin(SEARCH_FOLDS)]
        self._states = {
            int(sample_id): state
            for sample_id, state in states(raw_to_problems(searched), name).items()
        }
        # Each answers file read so far, by request hash; dropped when a call writes to it.
        self._held: dict[Path, dict[str, LlmAnswer]] = {}
        # The cost of each answer this cache served outside the seed's file, by request hash.
        self._used: dict[str, float] = {}

    def path(self, words: JevWords) -> Path:
        """Return the answers file of a set of words.

        Args:
            words: The words the requests hold.

        Returns:
            ``folder / "<words.sha256()>.jsonl"``; the file may not exist yet.
        """
        return self.folder / f"{words.sha256()}.jsonl"

    def prefill_seed(self, committed: Sequence[LlmAnswer]) -> int:
        """Copy the committed seed answers of the searched folds into the seed's file, once.

        The seed's requests are those of ``jev-statement-code`` (``settings.GEPA_SEED_NAME``):
        the same state fields, the same model, the same questions. A committed answer of a
        held-out problem, or of a problem that is not dev, is left out.

        Args:
            committed: The committed lines of ``assets/methods/jev-statement-code/
                answers_dev.jsonl``, as ``store.read_answers`` reads them.

        Returns:
            The lines added; 0 when the file already holds every one of them.

        Raises:
            ValueError: Before anything is written, if a problem of the searched folds has no
                committed answer, or one whose ``request_sha256`` is not its seed request's.
        """
        expected = {
            request.sample_id: request.sha256()
            for request in self._requests(SEED_WORDS, list(self._states))
        }
        lines = [answer for answer in committed if answer.sample_id in expected]
        _check_seed_lines(expected, lines)
        path = self.path(SEED_WORDS)
        held = self._answers(path)
        new = [answer for answer in lines if answer.request_sha256 not in held]
        if new:
            self._held.pop(path, None)
            _append_lines(path, new)
        return len(new)

    def probabilities(self, words: JevWords, sample_ids: Sequence[int]) -> pd.DataFrame:
        """Return Jev's probabilities for some problems with some words, asking only once.

        The requests that the words' file answers are not sent; the others are sent through one
        Jev session (``runner.send``) under what is left of the cap, and appended to the file.

        Args:
            words: The words the requests hold.
            sample_ids: The problems, distinct, of the learning or selection folds.

        Returns:
            One row per problem, in the order of ``sample_ids`` (index ``sample_id``), one
            column per focus tag (``answers.probabilities``: 0.0 on every tag for a failed
            answer).

        Raises:
            ValueError: If a problem is not of the learning or selection folds.
            SearchInterrupted: If a request was left unanswered (a transient error, a run of
                failed answers): run the search again.
            CostCapSpent: If the cap is spent before the call or inside it: run the search
                again with a higher cap.
        """
        ids = [int(sample_id) for sample_id in sample_ids]
        self._check_searched(ids)
        path = self.path(words)
        requests = self._requests(words, ids)
        held = self._answers(path)
        pending = [request for request in requests if request.sha256() not in held]
        if pending:
            self._send(pending, path)
            held = self._answers(path)
        answers = [held[request.sha256()] for request in requests]
        if path != self.path(SEED_WORDS):
            for answer in answers:
                self._used[answer.request_sha256] = answer_cost_usd(answer)["total_cost"]
        return answer_probabilities(answers, pd.Index(ids, name="sample_id"))

    def spent_usd(self) -> float:
        """Return what every answer in the folder cost, the seed's file left out.

        Returns:
            The ``math.fsum`` of their ``answer_cost_usd`` totals, in USD; 0.0 for none.
        """
        seed = self.path(SEED_WORDS)
        return math.fsum(
            answer_cost_usd(answer)["total_cost"]
            for path in sorted(self.folder.glob("*.jsonl"))
            if path != seed
            for answer in read_answers(path)
        )

    def used_usd(self) -> float:
        """Return what the answers this cache served cost, the seed's left out, each once.

        A replayed search serves the same answers in the same order, so this grows as it did
        when they were first paid for: it is the search's cost so far, replay or not.

        Returns:
            The ``math.fsum`` of their ``answer_cost_usd`` totals, in USD.
        """
        return math.fsum(self._used.values())

    def _requests(self, words: JevWords, ids: Sequence[int]) -> list[JevRequest]:
        """Build the requests of some problems with some words, as ``questions.requests`` does."""
        questions = noul_questions(words)
        return [JevRequest(int(i), MODEL, self._states[int(i)], questions) for i in ids]

    def _answers(self, path: Path) -> dict[str, LlmAnswer]:
        """Return the answers of a file by request hash, read once until a call writes to it."""
        if path not in self._held:
            self._held[path] = {answer.request_sha256: answer for answer in read_answers(path)}
        return self._held[path]

    def _check_searched(self, ids: Sequence[int]) -> None:
        """Refuse a problem that is not of the learning or selection folds."""
        outside = [sample_id for sample_id in ids if sample_id not in self._states]
        if outside:
            raise ValueError(
                f"The problems {problem_list(outside)} are not of the folds {list(SEARCH_FOLDS)} "
                f"that the search reads: it never asks about the held-out fold "
                f"(fold {SEARCH_HELD_OUT_FOLD})."
            )

    def _send(self, pending: Sequence[JevRequest], path: Path) -> None:
        """Send the requests a file lacks under what is left of the cap, or stop the search."""
        spent = self.spent_usd()
        self._check_cap(spent, "Jev's answers of this search cost")
        self._held.pop(path, None)
        if asyncio.run(self._ask(pending, path, self.max_cost_usd - spent)):
            left = f"{path}: Jev left some of {len(pending)} requests unanswered"
            self._check_cap(self.spent_usd(), f"{left} at the cost cap: its answers cost")
            raise SearchInterrupted(
                f"{left} (a transient error or a run of failed answers): run the search again "
                "to send them."
            )

    def _check_cap(self, spent: float, what: str) -> None:
        """Raise ``CostCapSpent`` once ``spent``, what Jev's answers cost, reaches the cap."""
        if not spent < self.max_cost_usd:
            raise CostCapSpent(
                f"{what} ${spent:.4f}, the cap of ${self.max_cost_usd:.4f}: run the search "
                "again with a higher cap to go on."
            )

    async def _ask(self, pending: Sequence[JevRequest], path: Path, max_cost_usd: float) -> int:
        """Send requests through one Jev session of the running loop; ``runner.send``'s code."""
        session = self.jev_session or client.jev_session
        async with session(self.name) as ask:
            return await runner.send(pending, ask, path, max_cost_usd=max_cost_usd)


def _check_seed_lines(expected: dict[int, str], lines: Sequence[LlmAnswer]) -> None:
    """Refuse committed seed answers that miss a searched problem or answer another request."""
    answered = {answer.sample_id for answer in lines}
    missing = [sample_id for sample_id in expected if sample_id not in answered]
    if missing:
        raise ValueError(
            f"The committed answers of jev-statement-code lack the problems "
            f"{problem_list(missing)}: the seed is scored from them, at no cost."
        )
    changed = [
        answer.sample_id for answer in lines if answer.request_sha256 != expected[answer.sample_id]
    ]
    if changed:
        raise ValueError(
            f"The committed answers of the problems {problem_list(changed)} are not to the seed "
            "requests: the seed words are the frozen words of jev-statement-code, so its "
            "requests must be byte-identical to that method's (docs/adr/0003)."
        )


def _append_lines(path: Path, answers: Sequence[LlmAnswer]) -> None:
    """Append answers to an answers file in one write, after its last complete line.

    ``append_answer`` reads the whole file for each line, which thousands of seed lines make
    slow; the lines are those it would write. A partial last line, an interrupted write, is
    dropped, as ``append_answer`` drops it. The file is replaced at once, through a temporary
    file of the same folder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = path.read_bytes() if path.exists() else b""
    complete = stored[: stored.rfind(_NEWLINE) + 1]
    new = b"".join(answer.to_json().encode("utf-8") + _NEWLINE for answer in answers)
    staged = path.with_suffix(".tmp")
    staged.write_bytes(complete + new)
    staged.replace(path)
