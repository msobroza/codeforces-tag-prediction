"""Send a Jev method's requests concurrently, and append each answer as it arrives.

``send`` is how ``python -m codeforces_tags.methods.typesafe_jev run`` asks Jev
(docs/adr/0007). It keeps the semantics of the LLM toolkit's runner (``llm.runner``), whose
stamp parses a tag list and so cannot write a Jev answer. It reads the answers file first and
sends only the requests that have no line in it, so a killed run resumes where it stopped. Each
answer is stamped with the git state of the code that sent it, appended to the answers file at
once (``llm.store``), checked for the model that served it, then priced: an answer that was
paid for is on disk before anything can stop the run.

What stops sending (the requests already in flight are always awaited and written):

- nothing, for a ``TransientApiError``: Jev did not answer after the SDK's retries, no line is
  written, and the next run sends that request again;
- the cost cap, once the answers of this call cost ``max_cost_usd``. A request that the API
  refuses is a final ``api_error`` line with no tokens: it costs nothing, so it never reaches
  the cap;
- a run of ``settings.MAX_FAILED_IN_A_ROW`` failed answers in a row, which the cap cannot stop: a
  fault that refuses every request would otherwise write a final line for the whole part;
- any other error (an HTTP error that every request would get, such as a 401; an answer served
  by another model than ``settings.MODEL``; a model with no price): it is raised once the
  requests in flight are written.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from codeforces_tags.config import LLM_PROGRESS_EVERY
from codeforces_tags.llm.answers import answer_cost_usd
from codeforces_tags.llm.client import LlmAnswer, TransientApiError
from codeforces_tags.llm.store import append_answer, read_answers, requests_to_send
from codeforces_tags.methods.typesafe_jev.answers import check_served_model
from codeforces_tags.methods.typesafe_jev.client import Ask
from codeforces_tags.methods.typesafe_jev.questions import JevRequest
from codeforces_tags.methods.typesafe_jev.settings import CONCURRENCY, MAX_FAILED_IN_A_ROW
from codeforces_tags.provenance import git_state


async def send(
    requests: Sequence[JevRequest], ask: Ask, path: Path, *, max_cost_usd: float | None
) -> int:
    """Send the requests that have no line in an answers file yet, appending each answer.

    At most ``settings.CONCURRENCY`` requests are in flight, in the running event loop, which
    alone appends to the file (``append_answer`` takes no lock). Every
    ``config.LLM_PROGRESS_EVERY`` answers, and once at the end, a line on stdout gives the
    answers written so far, how many failed, their tokens and their cost (``answer_cost_usd``);
    every failed answer is final, so a run of them is worth stopping with Ctrl-C. A transient
    error, a stop at the cost cap and the requests left are reported on stderr.

    Args:
        requests: The requests of one part of a method, one per problem
            (``questions.requests``); those that have a line in ``path`` are not sent.
        ask: Sends one request: the ``Ask`` of ``client.jev_session``, or a test's fake.
        path: The part's answers file (``store.answers_file``), created when missing.
        max_cost_usd: Send no new request once the answers of this call cost this much, in
            USD; None for no cap.

    Returns:
        0 when every request has a line in ``path``; 1 when some are left, after a stop at the
        cost cap or after a run of failed answers, or a transient error: the next call sends
        them.

    Raises:
        ValueError: Before anything is read or sent, if ``max_cost_usd`` is not above 0 (NaN
            included); as ``store.read_answers`` and ``store.requests_to_send``; once the
            requests in flight are written, if an answer was served by another model than
            ``settings.MODEL`` (``answers.check_served_model``), already has a line, or has
            tokens from a model with no price.
        Exception: Once the requests in flight are written, the first error that ``ask``
            raised other than a ``TransientApiError`` (e.g. the SDK's error of a 401).
    """
    # `not max_cost_usd > 0` refuses NaN too, which compares false with everything: a NaN
    # cap would be no cap at all.
    if max_cost_usd is not None and not max_cost_usd > 0:
        raise ValueError(f"A cost cap must be above 0 USD, not {max_cost_usd}.")
    pending = requests_to_send(requests, read_answers(path))
    if not pending:
        _say(f"{path}: every request has an answer: nothing to send.", sys.stdout)
        return 0
    sending = _Sending(path, len(pending), git_state(), max_cost_usd)
    # Made inside the coroutine, so that it belongs to the running event loop.
    slots = asyncio.Semaphore(CONCURRENCY)
    sent = await asyncio.gather(*(sending.one(request, ask, slots) for request in pending))
    sending.close(skipped=sent.count(False))
    return _left(requests, path)


@dataclasses.dataclass
class _Sending:
    """One call of ``send``: where it writes, what it wrote and spent, and what stopped it."""

    path: Path
    # The requests that had no line when the call started.
    total: int
    # The state of the code that sends them, as provenance.git_state() gives it.
    git: Mapping[str, object]
    max_cost_usd: float | None
    # The answers this call appended, stamped, in the order they arrived, and their cost in USD.
    written: list[LlmAnswer] = dataclasses.field(default_factory=list)
    cost: float = 0.0
    # The first error other than a TransientApiError: no request is sent after it.
    error: Exception | None = None

    def stopped(self) -> bool:
        """Tell whether no new request may be sent: an error, the cap or a run of failures."""
        if self.error is not None or self.failed_in_a_row() >= MAX_FAILED_IN_A_ROW:
            return True
        return self.max_cost_usd is not None and self.cost >= self.max_cost_usd

    def failed_in_a_row(self) -> int:
        """Count the failed answers written since the last completed one."""
        return sum(1 for _ in itertools.takewhile(lambda a: a.failed, reversed(self.written)))

    async def one(self, request: JevRequest, ask: Ask, slots: asyncio.Semaphore) -> bool:
        """Send one request once a slot is free, unless sending stopped; write its answer.

        A ``TransientApiError`` is reported on stderr and writes nothing; any other error is
        kept (``keep``), and no request is sent after it.

        Returns:
            False when sending had stopped before the request's turn, else True.
        """
        async with slots:
            if self.stopped():
                return False
            try:
                self.add(await ask(request))
            except TransientApiError as error:
                _say(
                    f"{self.path}: problem {request.sample_id} got no answer ({error}); no line "
                    "was written, so the next run sends its request again.",
                    sys.stderr,
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                self.keep(error)
        return True

    def keep(self, error: Exception) -> None:
        """Keep the first error, which ``close`` raises; a later one is dropped."""
        if self.error is None:
            self.error = error

    def add(self, answer: LlmAnswer) -> None:
        """Stamp an answer with the code's git state, append it, check its model, then price it.

        Raises:
            ValueError: As ``append_answer``, ``answers.check_served_model`` and
                ``answer_cost_usd``.
        """
        stamped = dataclasses.replace(
            answer, git_commit=str(self.git["commit"]), code_dirty=bool(self.git["code_dirty"])
        )
        append_answer(self.path, stamped)
        self.written.append(stamped)
        try:
            check_served_model([stamped])
            self.cost += answer_cost_usd(stamped)["total_cost"]
        finally:
            # The line is on disk: the progress line counts it, whether or not it passed.
            if len(self.written) % LLM_PROGRESS_EVERY == 0:
                _say(self.progress(), sys.stdout)

    def progress(self) -> str:
        """Describe what this call wrote so far: answers, failed ones, tokens and cost."""
        input_tokens = sum(answer.usage["input_tokens"] for answer in self.written)
        output_tokens = sum(answer.usage["output_tokens"] for answer in self.written)
        failed = sum(answer.failed for answer in self.written)
        return (
            f"{self.path}: {len(self.written)} of {self.total} answers ({failed} failed), "
            f"{input_tokens:,} input tokens, {output_tokens:,} output tokens, ${self.cost:.4f}"
        )

    def close(self, skipped: int) -> None:
        """End the call: the last progress line, the first error, then why sending stopped.

        Args:
            skipped: The requests that were not sent, because sending had stopped.

        Raises:
            Exception: The first error that ``keep`` kept.
        """
        if not self.written or len(self.written) % LLM_PROGRESS_EVERY:
            _say(self.progress(), sys.stdout)
        if self.error is not None:
            raise self.error
        if skipped and self.failed_in_a_row() >= MAX_FAILED_IN_A_ROW:
            _say(
                f"{self.path}: {self.failed_in_a_row()} answers in a row failed (last: "
                f"{self.written[-1].status}), so sending stopped; {skipped} requests were not "
                "sent. Read the failed lines (they are final) before running again.",
                sys.stderr,
            )
        elif skipped:
            _say(
                f"{self.path}: the cost cap of ${self.max_cost_usd:.4f} was reached after "
                f"spending ${self.cost:.4f}; {skipped} requests were not sent: run again to "
                "send them.",
                sys.stderr,
            )


def _left(requests: Sequence[JevRequest], path: Path) -> int:
    """Return 1, after saying how many on stderr, when requests still have no line; else 0."""
    left = requests_to_send(requests, read_answers(path))
    if not left:
        return 0
    _say(
        f"{path}: {len(left)} of {len(requests)} requests have no answer yet: run again to send "
        "them.",
        sys.stderr,
    )
    return 1


def _say(line: str, stream: TextIO) -> None:
    """Print one line on stdout or stderr at once: a long run reports as it goes."""
    print(line, file=stream, flush=True)
