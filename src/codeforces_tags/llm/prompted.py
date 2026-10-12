"""``PromptedLlm``: the tagging method that asks an LLM for a problem's tags (docs/adr/0003).

A prompted method renders one request per problem (``build_requests``), sends them live through
its ``Ask``, and encodes the answers as 0/1 tags with the fixed parser of its output format
(``llm.answers``). It never reads or writes an answers file: those hold the frozen answers of
``llm run``, from which its dev and test predictions are encoded, while ``predict`` serves new
problems (e.g. the latency harness, or a logged model).

Its method folder holds ``prompted.json``: its spec and its prompt snapshot, with no absolute
path and no dataset text. Loading it imports no provider SDK: without an injected ``Ask``, the
provider's adapter is built when it predicts (``client.ask_for``), so the conformance check's
isolation subprocess never imports one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from codeforces_tags.config import LLM_CONCURRENCY
from codeforces_tags.llm import client
from codeforces_tags.llm.answers import encode_answers
from codeforces_tags.llm.prompts import LlmMethodSpec, LlmRequest, PromptSnapshot, build_requests
from codeforces_tags.preprocessing.pipeline import raw_to_problems

PROMPTED_FILE = "prompted.json"


class PromptedLlm:
    """A tagging method that asks an LLM for each problem's tags (CONTEXT.md, Prompted method).

    As it is, a zero-shot method (kind "zero_shot"): nothing is learned from dev. A fitted
    prompted method (kind "fitted", e.g. few-shot) subclasses it with ``fit(raw_problems,
    labels)`` and its own ``request_context``.

    Attributes:
        spec: The method's spec, from its package's ``llm_spec(name)``.
        prompt: Its prompt snapshot.
        ask: The ``Ask`` it sends requests with; None to build the provider's adapter afresh in
            each ``apredict`` call.
    """

    def __init__(
        self, spec: LlmMethodSpec, prompt: PromptSnapshot, ask: client.Ask | None = None
    ) -> None:
        """Bind the spec, the prompt and the ``Ask``.

        Args:
            spec: The method's spec.
            prompt: Its prompt: ``prompt_snapshot(spec)``, or the one its method folder holds.
                A prompt that is not the spec's is refused when requests are built.
            ask: Sends one request, e.g. a test's fake; None to build the adapter of the
                settings' provider (``client.ask_for``) in each ``apredict`` call, in its loop.
        """
        self.spec = spec
        self.prompt = prompt
        self.ask = ask

    @property
    def name(self) -> str:
        """Its key in ``METHODS``, e.g. ``"descode2tag-json"``."""
        return self.spec.name

    @property
    def tags(self) -> tuple[str, ...]:
        """The output columns: the tags the model chooses from, ``spec.all_tags``."""
        return self.spec.all_tags

    def request_context(  # pylint: disable=unused-argument
        self, raw_problems: pd.DataFrame
    ) -> Mapping[int, Mapping[str, Any]] | None:
        """Return each problem's own template variables: none for a zero-shot method.

        A fitted prompted method overrides it, e.g. with each problem's examples.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            None: the template reads the spec's fields, constants and ``all_tags`` only.
        """
        return None

    def requests(self, raw_problems: pd.DataFrame) -> list[LlmRequest]:
        """Render the request of each problem.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            One request per problem, in order: ``build_requests`` on ``raw_to_problems``'s
            rows, with ``request_context``'s variables.

        Raises:
            ValueError: As ``raw_to_problems`` and ``build_requests`` (e.g. a prompt that is not
                the spec's).
        """
        problems = raw_to_problems(raw_problems)
        return build_requests(problems, self.spec, self.prompt, self.request_context(raw_problems))

    async def apredict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of raw problems, in the running event loop.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Int 0/1, with the index of ``raw_problems``, one column per tag of ``tags``.
        """
        predictions, _ = await self.apredict_with_answers(raw_problems)
        return predictions

    async def apredict_with_answers(
        self, raw_problems: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[client.LlmAnswer]]:
        """Predict the tags of raw problems, and keep the answers they come from.

        At most ``config.LLM_CONCURRENCY`` requests are in flight, through one
        ``client.session``. The answers are not written anywhere.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            The predictions, as ``apredict`` returns them, and the answers in the order of the
            problems (for their tokens, cost, latency and retries, e.g. ``tracking latency``).

        Raises:
            ValueError: As ``requests``.
            TransientApiError: If the provider did not answer a request. Every request ends
                first; then the error of the first problem, in order, whose request failed is
                raised, whatever it is.
        """
        requests = self.requests(raw_problems)
        ask = self.ask if self.ask is not None else client.ask_for(self.spec.settings)
        answers = await _ask_all(requests, ask)
        output_format = self.spec.settings.output_format
        predictions = encode_answers(answers, raw_problems.index, output_format, self.tags)
        return predictions, answers

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of raw problems, in an event loop of its own.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Int 0/1, with the index of ``raw_problems``, one column per tag of ``tags``.

        Raises:
            RuntimeError: If an event loop is running in this thread (e.g. in Jupyter), where
                ``await method.apredict(raw_problems)`` is the way to predict.
        """
        if _loop_running():
            raise RuntimeError(
                f"{self.name}: predict() starts an event loop of its own, and one is already "
                "running here (e.g. in a Jupyter notebook): call "
                "`await method.apredict(raw_problems)` instead."
            )
        return asyncio.run(self.apredict(raw_problems))

    def save(self, folder: Path) -> None:
        """Write ``prompted.json``, the spec and the prompt snapshot, into ``folder``.

        Args:
            folder: An existing directory.
        """
        stored = {"spec": self.spec.to_dict(), "prompt": self.prompt.to_dict()}
        text = json.dumps(stored, indent=2) + "\n"
        (Path(folder) / PROMPTED_FILE).write_text(text, encoding="utf-8")


def load_prompted(folder: Path) -> PromptedLlm:
    """Load the prompted method saved in a method folder: the ``load`` of a zero-shot package.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        The method, with no ``Ask``: it builds its provider's adapter when it predicts.

    Raises:
        FileNotFoundError: If ``folder`` holds no ``prompted.json``.
        TypeError: If the spec or the prompt lacks a field or has an unknown one.
    """
    stored = json.loads((Path(folder) / PROMPTED_FILE).read_text(encoding="utf-8"))
    spec = LlmMethodSpec.from_dict(stored["spec"])
    return PromptedLlm(spec, PromptSnapshot.from_dict(stored["prompt"]))


async def _ask_all(requests: Sequence[LlmRequest], ask: client.Ask) -> list[client.LlmAnswer]:
    """Send every request, at most ``LLM_CONCURRENCY`` at once, and raise the first error."""
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    async with client.session(ask) as send:
        outcomes = await asyncio.gather(
            *(_ask_one(request, send, semaphore) for request in requests), return_exceptions=True
        )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes


async def _ask_one(
    request: LlmRequest, send: client.Ask, semaphore: asyncio.Semaphore
) -> client.LlmAnswer:
    """Send one request when a slot is free."""
    async with semaphore:
        return await send(request)


def _loop_running() -> bool:
    """Tell whether an event loop is running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
