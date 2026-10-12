"""Send a prompted method's requests concurrently, and write each answer as it arrives.

``answer_each`` is how ``llm run`` sends requests (docs/adr/0003). It reads the answers file
first and sends only the requests that have no line in it, so a run that was killed resumes
where it stopped. Each answer is stamped with the git state of the code that sent it and with
its parsed tags (a convenience copy, ``llm.answers``), then appended to the answers file
(``llm.store``) at once: a killed run loses only the requests in flight.

What stops sending:

- a ``TransientApiError`` (the provider did not answer, after its SDK's retries) writes no
  line and stops nothing: the next run sends that request again;
- the cost cap: once this run's answers cost ``max_cost_usd``, no new request is sent. The
  requests in flight are still awaited and written, so the spend may pass the cap by their cost;
- any other error (``RejectedSetting``, an HTTP error of every request such as a missing key or
  no credit left, a served model with no price): no new request is sent, the requests in flight
  are awaited and written, then the first error is raised.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from codeforces_tags.config import ALL_TAGS, LLM_PROGRESS_EVERY
from codeforces_tags.llm.answers import answer_cost_usd, parse_answer
from codeforces_tags.llm.client import USAGE_KEYS, Ask, LlmAnswer, TransientApiError, session
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.llm.settings import OUTPUT_FORMATS
from codeforces_tags.llm.store import append_answer, read_answers, requests_to_send
from codeforces_tags.provenance import git_state


async def answer_each(  # pylint: disable=too-many-arguments
    requests: Sequence[LlmRequest],
    ask: Ask,
    *,
    concurrency: int,
    answers_file: Path,
    output_format: str,
    all_tags: Sequence[str] = ALL_TAGS,
    max_cost_usd: float | None = None,
    git: Mapping[str, object] | None = None,
) -> list[LlmAnswer]:
    """Send the requests that have no answer yet, and append each answer as it arrives.

    At most ``concurrency`` requests are in flight, all sent through one ``session(ask)``.
    Every ``config.LLM_PROGRESS_EVERY`` answers, and once at the end, a line on stdout gives the
    answers written so far, how many failed (a run of failures is worth stopping with Ctrl-C:
    every failed answer is final), their tokens and their cost (``answer_cost_usd``). A
    transient error and a stop at the cost cap are reported on stderr.

    Args:
        requests: The requests of one part, one per problem (``build_requests``); those that
            already have a line in ``answers_file`` are not sent.
        ask: Sends one request: a provider's adapter (``client.ask_for``) or a test's fake.
        concurrency: The most requests in flight at once, e.g. ``config.LLM_CONCURRENCY``.
        answers_file: The part's answers file (``store.answers_file``), created when missing.
        output_format: One of ``OUTPUT_FORMATS``: the parser of the stamped tags.
        all_tags: The tags the model chooses from.
        max_cost_usd: Stop sending once this run's answers cost this much, in USD; None for no
            cap.
        git: The state of the code that sends the requests, as ``provenance.git_state()``
            gives it (``commit`` and ``code_dirty``); read from git when None.

    Returns:
        The answers this call wrote, stamped, in the order they arrived.

    Raises:
        ValueError: Before anything is sent, if ``concurrency`` is below 1, if
            ``output_format`` is not one of ``OUTPUT_FORMATS`` or not the requests' own, if
            ``max_cost_usd`` is not above 0 (NaN included), or as ``store.read_answers`` and
            ``store.requests_to_send``; after the requests in flight are written, if an answer
            has a line already or tokens from a model that ``config.LLM_PRICES`` does not price.
        RejectedSetting: If the provider rejected a setting of the requests.
    """
    _check_options(requests, concurrency, output_format, max_cost_usd)
    pending = requests_to_send(requests, read_answers(answers_file))
    if not pending:
        print(f"{answers_file}: every request has an answer: nothing to send.", flush=True)
        return []
    stamp = _Stamp(output_format, tuple(all_tags), git_state() if git is None else git)
    batch = _Batch(answers_file, stamp, max_cost_usd, len(pending))
    semaphore = asyncio.Semaphore(concurrency)
    async with session(ask) as send:
        sent = await asyncio.gather(
            *(batch.answer(request, send, semaphore) for request in pending)
        )
    batch.finish(unsent=sent.count(False))
    return batch.written


@dataclasses.dataclass(frozen=True)
class _Stamp:
    """What the runner adds to an answer before writing it: the code's git state and the parse."""

    output_format: str
    all_tags: tuple[str, ...]
    git: Mapping[str, object]

    def __call__(self, answer: LlmAnswer) -> LlmAnswer:
        """Return the answer with the git state and the tags parsed from its text."""
        parsed = parse_answer(answer, self.output_format, self.all_tags)
        return dataclasses.replace(
            answer,
            git_commit=str(self.git["commit"]),
            code_dirty=bool(self.git["code_dirty"]),
            tags=parsed.tags,
            failed=parsed.failed,
            unknown_pieces=parsed.unknown_pieces,
        )


class _Batch:
    """One call of ``answer_each``: it sends and writes, until the cost cap or an error."""

    def __init__(
        self, answers_file: Path, stamp: _Stamp, max_cost_usd: float | None, total: int
    ) -> None:
        """Start with nothing sent or written."""
        self.answers_file = answers_file
        self.stamp = stamp
        self.max_cost_usd = max_cost_usd
        self.total = total
        self.written: list[LlmAnswer] = []
        self.cost = 0.0
        self.error: Exception | None = None

    @property
    def capped(self) -> bool:
        """Whether this run's answers cost the cap already."""
        return self.max_cost_usd is not None and self.cost >= self.max_cost_usd

    async def answer(self, request: LlmRequest, send: Ask, semaphore: asyncio.Semaphore) -> bool:
        """Send one request when a slot is free, unless the run stopped; write its answer.

        Returns:
            Whether the request was sent.
        """
        async with semaphore:
            if self.capped or self.error is not None:
                return False
            try:
                self.write(await send(request))
            except TransientApiError as error:
                print(
                    f"Problem {request.sample_id}: {error} No line was written: the next run "
                    "sends its request again.",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                # No request is sent after it; the requests in flight are still written.
                if self.error is None:
                    self.error = error
            return True

    def write(self, answer: LlmAnswer) -> None:
        """Stamp an answer, append it to the answers file, then price it."""
        answer = self.stamp(answer)
        append_answer(self.answers_file, answer)
        self.written.append(answer)
        self.cost += answer_cost_usd(answer)["total_cost"]
        if len(self.written) % LLM_PROGRESS_EVERY == 0:
            print(self.progress(), flush=True)

    def progress(self) -> str:
        """Describe the answers written so far: how many, how many failed, tokens and cost."""
        failed = sum(answer.failed for answer in self.written)
        usage = {key: sum(answer.usage[key] for answer in self.written) for key in USAGE_KEYS}
        return (
            f"{self.answers_file}: {len(self.written)} of {self.total} answers ({failed} failed), "
            f"{usage['input_tokens']:,} input tokens ({usage['cached_tokens']:,} cached), "
            f"{usage['output_tokens']:,} output tokens, ${self.cost:.4f}"
        )

    def finish(self, unsent: int) -> None:
        """Print the last progress line, report a stop at the cap, and raise the first error."""
        if len(self.written) % LLM_PROGRESS_EVERY or not self.written:
            print(self.progress(), flush=True)
        if self.error is not None:
            raise self.error
        if unsent:
            print(
                f"{self.answers_file}: stopped at the cost cap of ${self.max_cost_usd:.4f}, after "
                f"spending ${self.cost:.4f}: {unsent} requests were not sent. Run again to send "
                "them.",
                file=sys.stderr,
                flush=True,
            )


def _check_options(
    requests: Sequence[LlmRequest], concurrency: int, output_format: str, max_cost_usd: float | None
) -> None:
    """Refuse options that would fail only once requests are paid for."""
    if concurrency < 1:
        raise ValueError(f"The concurrency must be at least 1, not {concurrency}.")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"{output_format!r} is not an output format: expected one of {OUTPUT_FORMATS}."
        )
    formats = sorted({request.settings.output_format for request in requests} - {output_format})
    if formats:
        raise ValueError(
            f"Some requests answer in {', '.join(formats)}, not {output_format}: their answers "
            "would be parsed with the wrong parser."
        )
    # Written so that NaN, which compares false with everything, is refused: a NaN cap is no cap.
    if max_cost_usd is not None and not max_cost_usd > 0:
        raise ValueError(f"A cost cap must be above 0 USD, not {max_cost_usd}.")
