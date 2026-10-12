"""Reflection in the search over Jev's words: what the reflection model reads, and what it answers.

GEPA's search (docs/adr/0010) rewrites one piece of Jev's words per iteration: the convention or
one focus tag's criteria. This module turns the errors of an evaluated minibatch into the records
that the reflection model reads (``tag_records``, ``convention_records``), renders them into one
toolkit request (``reflection_request``, with the templates and strict schemas of
``templates/gepa/``), and reads its answer back into a proposal (``read_proposal``), or into the
reason it is refused. A refused proposal leaves the piece as its parent had it.

``Reflector`` sends the requests to the reflection model, GPT-6 Luna, through the toolkit's
``llm.client.ask_for``, which imports the OpenAI SDK only when a request is sent. Its answers
are kept in an answers file keyed by request hash, so a rerun of the search replays every
reflection without a paid call.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm import client as llm_client
from codeforces_tags.llm.answers import answer_cost_usd
from codeforces_tags.llm.client import Ask, LlmAnswer
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.llm.store import append_answer, read_answers
from codeforces_tags.methods.typesafe_jev.objective import weighted_errors
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    REFLECTION_SETTINGS,
    SEARCH_EXAMPLES,
    SEARCH_SOLUTION_CHARS,
    SEARCH_STATEMENT_CHARS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    JevWords,
    candidate_from_words,
    copied_sequences,
    length_cap,
    piece_length,
    piece_problem,
    piece_text,
    words_from_candidate,
)
from codeforces_tags.templating import render_template

# Why a proposal was refused: a failed or refused answer; output that is not JSON, or JSON of
# another shape than the schema's; a blank text or an empty list; over the piece's length cap; an
# 8-word sequence shared with a problem shown in the prompt and absent from the seed words.
REFUSAL_REASONS: tuple[str, ...] = (
    "failed_answer",
    "invalid_json",
    "empty_field",
    "too_long",
    "copied_text",
)
# The reflection prompts and the schemas of their answers, in the package.
TEMPLATE_DIR = Path(__file__).parent / "templates" / "gepa"
# The piece whose text is the convention; every other piece is a focus tag's criteria.
_CONVENTION = "convention"
# The template, schema file and schema name of each kind of piece.
_TAG_PROMPT = ("reflect_tag", "tag_criteria.schema.json", "tag_criteria")
_CONVENTION_PROMPT = ("reflect_convention", "convention.schema.json", "convention")


@dataclass(frozen=True)
class Proposal:
    """The reflection model's rewrite of one piece, or why it was refused.

    Attributes:
        piece: The piece rewritten: ``"convention"`` or a focus tag.
        text: The new piece text, as ``words.piece_text`` writes it; None when refused.
        reason: One of ``REFUSAL_REASONS`` when refused; None when accepted.
    """

    piece: str
    text: str | None
    reason: str | None


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def tag_records(
    tag: str,
    problems: pd.DataFrame,
    probabilities: pd.DataFrame,
    labels: pd.DataFrame,
    weights: pd.DataFrame,
    k: int = SEARCH_EXAMPLES,
) -> list[dict[str, object]]:
    """Build what the reflection model reads to rewrite one tag's criteria.

    Args:
        tag: A focus tag.
        problems: The minibatch's preprocessed problems (columns ``statement`` and
            ``source_code``), indexed by ``sample_id``.
        probabilities: Jev's probability of each focus tag, indexed by ``sample_id``; its rows
            of the minibatch are read.
        labels: The 0/1 focus labels, indexed by ``sample_id``; its rows of the minibatch are
            read.
        weights: The tag weights of ``objective.tag_weights``: index ``FOCUS_TAGS``, columns
            ``[0, 1]``.
        k: The most examples to give.

    Returns:
        A summary record (``kind``, ``tag``, ``positives``, ``negatives``,
        ``mean_probability_positives``, ``mean_probability_negatives``; a mean is None when
        the minibatch has no problem of that class), then up to ``k`` example records
        (``kind``, ``sample_id``, ``statement``, ``solution``, ``gold``, ``probability``,
        ``error``), the largest weighted error on the tag first, ties by sample id. Every value
        is a JSON type.

    Raises:
        ValueError: If the tag is not a focus tag.
    """
    if tag not in FOCUS_TAGS:
        raise ValueError(f"{tag!r} is not a focus tag: expected one of {FOCUS_TAGS}.")
    ids = problems.index
    errors = weighted_errors(probabilities.loc[ids], labels.loc[ids], weights)[tag]
    gold = labels.loc[ids, tag]
    probability = probabilities.loc[ids, tag]
    summary = {
        "kind": "summary",
        "tag": tag,
        "positives": int((gold == 1).sum()),
        "negatives": int((gold == 0).sum()),
        "mean_probability_positives": _mean(probability[gold == 1]),
        "mean_probability_negatives": _mean(probability[gold == 0]),
    }
    examples = [
        _example(problems, sample_id)
        | {
            "gold": _yes_no(gold[sample_id]),
            "probability": float(probability[sample_id]),
            "error": float(errors[sample_id]),
        }
        for sample_id in _worst_first(errors, k)
    ]
    return [summary, *examples]


def convention_records(
    problems: pd.DataFrame,
    probabilities: pd.DataFrame,
    labels: pd.DataFrame,
    weights: pd.DataFrame,
    k: int = SEARCH_EXAMPLES,
) -> list[dict[str, object]]:
    """Build what the reflection model reads to rewrite the convention.

    Args:
        problems: The minibatch's preprocessed problems (columns ``statement`` and
            ``source_code``), indexed by ``sample_id``.
        probabilities: Jev's probability of each focus tag, indexed by ``sample_id``.
        labels: The 0/1 focus labels, indexed by ``sample_id``.
        weights: The tag weights of ``objective.tag_weights``.
        k: The most examples to give.

    Returns:
        Up to ``k`` example records (``kind``, ``sample_id``, ``statement``, ``solution``,
        ``tags``: each focus tag's ``gold`` and ``probability``, and ``error``: the weighted
        errors summed over the tags), the largest summed error first, ties by sample id.
    """
    ids = problems.index
    errors = weighted_errors(probabilities.loc[ids], labels.loc[ids], weights)
    totals = errors.sum(axis=1)
    return [
        _example(problems, sample_id)
        | {
            "tags": {
                tag: {
                    "gold": _yes_no(labels.loc[sample_id, tag]),
                    "probability": float(probabilities.loc[sample_id, tag]),
                }
                for tag in FOCUS_TAGS
            },
            "error": float(totals[sample_id]),
        }
        for sample_id in _worst_first(totals, k)
    ]


def texts_shown(records: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the problem texts that a reflection prompt shows: the read_proposal check's input.

    Args:
        records: The records of ``tag_records`` or ``convention_records``.

    Returns:
        The statement and the solution of each example record, as cut for the prompt.
    """
    return [
        str(record[field])
        for record in records
        if record["kind"] == "example"
        for field in ("statement", "solution")
    ]


def reflection_request(
    piece: str, words: JevWords, records: Sequence[Mapping[str, object]], iteration: int
) -> LlmRequest:
    """Render the request that asks the reflection model to rewrite one piece.

    Args:
        piece: ``"convention"`` or a focus tag (``words.PIECES``).
        words: The parent candidate's words.
        records: The piece's records: ``tag_records`` for a tag (its summary first),
            ``convention_records`` for the convention.
        iteration: GEPA's iteration, the request's ``sample_id``.

    Returns:
        The request of method ``GEPA_NAME`` with ``REFLECTION_SETTINGS``, its input the
        rendered prompt, and its schema the strict schema of the piece's answer
        (``tag_criteria`` or ``convention``).

    Raises:
        ValueError: If the piece is not one of ``PIECES``, or a tag's records do not start
            with its summary.
    """
    _check_piece(piece)
    template, schema_file, schema_name = _CONVENTION_PROMPT if piece == _CONVENTION else _TAG_PROMPT
    variables = (
        _convention_variables(words, records)
        if piece == _CONVENTION
        else _tag_variables(piece, words, records)
    )
    return LlmRequest(
        sample_id=iteration,
        method=GEPA_NAME,
        input=render_template(template, TEMPLATE_DIR, **variables),
        settings=REFLECTION_SETTINGS,
        schema=json.loads((TEMPLATE_DIR / schema_file).read_text(encoding="utf-8")),
        schema_name=schema_name,
    )


def read_proposal(piece: str, answer: LlmAnswer, shown_texts: Sequence[str]) -> Proposal:
    """Read the reflection model's answer into a new piece text, or the reason to refuse it.

    The checks run in the order of ``REFUSAL_REASONS``. Whether the answer's value has its
    piece's shape is ``words.piece_problem``'s check, the one ``words.parse_piece`` raises on. The
    copy check runs on this piece alone: the seed's words with this piece replaced, against the
    texts shown in the prompt, so only a sequence the new text added can fail it.

    Args:
        piece: ``"convention"`` or a focus tag.
        answer: The reflection model's answer.
        shown_texts: The problem texts its prompt showed (``texts_shown(records)``).

    Returns:
        The proposal: its text as ``words.piece_text`` writes it, or its refusal reason.

    Raises:
        ValueError: If the piece is not one of ``PIECES``.
    """
    _check_piece(piece)
    if answer.failed or answer.status != "completed" or answer.output_text is None:
        return Proposal(piece, None, "failed_answer")
    value = _piece_value(piece, answer.output_text)
    problem = piece_problem(piece, value)
    if problem is not None:
        return Proposal(piece, None, problem)
    if piece_length(piece, value) > length_cap(piece):
        return Proposal(piece, None, "too_long")
    words = _seed_with(piece, value)
    if copied_sequences(words, shown_texts):
        return Proposal(piece, None, "copied_text")
    return Proposal(piece, piece_text(words, piece), None)


class Reflector:
    """Asks the reflection model, through an answers file that caches each request's answer.

    Attributes:
        cache_path: The answers file (``reflections.jsonl`` of the search folder).
        sent: How many requests this reflector sent, cached answers left out.
    """

    def __init__(self, cache_path: Path, ask: Ask | None = None) -> None:
        """Bind the cache; the ``Ask`` is built when the first request is sent.

        Args:
            cache_path: The answers file; it may not exist yet.
            ask: The ``Ask`` to send with; None for ``llm.client.ask_for(REFLECTION_SETTINGS)``.
        """
        self.cache_path = Path(cache_path)
        self._ask = ask
        self.sent = 0

    @property
    def total_cost(self) -> float:
        """Return the cost in USD of every answer in the cache, sent by this run or an earlier one.

        Returns:
            The sum of ``llm.answers.answer_cost_usd(answer)["total_cost"]``.
        """
        return sum(
            (answer_cost_usd(answer)["total_cost"] for answer in read_answers(self.cache_path)),
            0.0,
        )

    def answer(self, request: LlmRequest) -> LlmAnswer:
        """Return the answer to a request: the cached one, or a new one, sent once and cached.

        A new request is sent in an event loop of its own, through ``llm.client.session``, and
        its answer is appended to the cache with ``llm.store.append_answer`` before it is
        returned.

        Args:
            request: The request, from ``reflection_request``.

        Returns:
            The answer; a failed one too, which ``read_proposal`` refuses.

        Raises:
            TransientApiError: If the provider did not answer; no line is written, and the
                next call sends the request again.
            ValueError: If the answer is not keyed by the request's hash; no line is written.
        """
        key = request.sha256()
        cached = {line.request_sha256: line for line in read_answers(self.cache_path)}
        if key in cached:
            return cached[key]
        answer = asyncio.run(self._send(request))
        if answer.request_sha256 != key:
            raise ValueError(
                f"The answer to reflection {request.sample_id} is keyed by "
                f"{answer.request_sha256[:12]}, not by its request's hash {key[:12]}."
            )
        append_answer(self.cache_path, answer)
        return answer

    async def _send(self, request: LlmRequest) -> LlmAnswer:
        """Send one request through a session opened in the running loop."""
        if self._ask is None:
            self._ask = llm_client.ask_for(REFLECTION_SETTINGS)
        self.sent += 1
        async with llm_client.session(self._ask) as send:
            return await send(request)


def _check_piece(piece: str) -> None:
    """Raise unless the piece is one of the search's pieces."""
    if piece not in PIECES:
        raise ValueError(f"{piece!r} is not a piece of Jev's words: expected one of {PIECES}.")


def _mean(values: pd.Series) -> float | None:
    """Return the mean of some probabilities, or None when there are none."""
    return float(values.mean()) if len(values) else None


def _yes_no(label: object) -> str:
    """Return a 0/1 label as the reflection prompt writes it."""
    return "yes" if int(label) == 1 else "no"


def _worst_first(errors: pd.Series, k: int) -> list[int]:
    """Return the k sample ids of the largest errors, largest first, ties by sample id."""
    order = sorted(errors.index, key=lambda sample_id: (-float(errors[sample_id]), int(sample_id)))
    return [int(sample_id) for sample_id in order[:k]]


def _example(problems: pd.DataFrame, sample_id: int) -> dict[str, object]:
    """Return the start of an example record: its problem, cut to the prompt's lengths."""
    return {
        "kind": "example",
        "sample_id": int(sample_id),
        "statement": str(problems.loc[sample_id, "statement"])[:SEARCH_STATEMENT_CHARS],
        "solution": str(problems.loc[sample_id, "source_code"])[:SEARCH_SOLUTION_CHARS],
    }


def _tag_variables(
    tag: str, words: JevWords, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Return the variables of a tag's reflection prompt."""
    summary, *examples = records
    if summary["kind"] != "summary" or summary["tag"] != tag:
        raise ValueError(f"The records of {tag!r} must start with its summary (tag_records).")
    return {
        "tag": tag,
        "convention": words.convention,
        "question": words.question.format(tag=tag),
        "current": piece_text(words, tag),
        "current_length": piece_length(tag, words.tag_definitions[tag]),
        "length_cap": length_cap(tag),
        "summary": summary,
        "examples": examples,
    }


def _convention_variables(
    words: JevWords, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Return the variables of the convention's reflection prompt."""
    return {
        "convention": words.convention,
        "question": words.question,
        "definitions": {tag: piece_text(words, tag) for tag in FOCUS_TAGS},
        "current_length": piece_length(_CONVENTION, words.convention),
        "length_cap": length_cap(_CONVENTION),
        "examples": list(records),
    }


def _piece_value(piece: str, output_text: str) -> object:
    """Return the piece's value in an answer's text, for ``words.piece_problem`` to check.

    A tag's value is the decoded JSON itself; the convention's is the text under the key
    ``"convention"`` of an object that holds no other key. It is None when the text is not JSON,
    or when a convention's answer is not such an object: ``piece_problem`` names None
    ``"invalid_json"`` for every piece.
    """
    try:
        value = json.loads(output_text)
    except json.JSONDecodeError:
        return None
    if piece != _CONVENTION:
        return value
    wrapped = isinstance(value, dict) and set(value) == {_CONVENTION}
    return value[_CONVENTION] if wrapped else None


def _seed_with(piece: str, value: str | Mapping[str, object]) -> JevWords:
    """Return the seed's words with one piece replaced by a proposal's checked value."""
    text = value if piece == _CONVENTION else json.dumps(value, ensure_ascii=False, indent=2)
    return words_from_candidate(candidate_from_words(SEED_WORDS) | {piece: text})
