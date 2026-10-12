"""Read the tags of an answer, encode answers as 0/1, and count their cost and time in flight.

A free-text answer is read by a fixed parser, ``parse_text_tags`` (``config.TEXT_PARSER_VERSION``).
It was written and tested before any paid request and is never adjusted after results are seen:
a new rule would be a new parser version. An answer **fails** when its status is not
``completed`` (cut off at the token cap, a refusal, a request the API refused) or, for the
``json`` output format, when its JSON cannot be read. A failed answer scores as the empty tag
set and is counted, never dropped. A piece of an answer that names no tag is dropped and
counted, and an answer that names none (empty text) is the empty tag set, not a failure.

``encode_answers`` re-parses each answer's ``output_text`` and ``status``. It never reads the
``tags``, ``failed`` and ``unknown_pieces`` that the runner stores in each answer line, a copy
for reading: the text and the status are the record, so another rule could be replayed on them.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from codeforces_tags.config import ALL_TAGS, LLM_PRICES
from codeforces_tags.evaluation.predictions import PER_PROBLEM_COLUMNS
from codeforces_tags.llm.client import USAGE_KEYS, LlmAnswer
from codeforces_tags.llm.settings import OUTPUT_FORMATS

# A list marker at the start of a piece: "- dp", "* dp", "• dp", "1. dp", "2) dp". A space must
# follow it, so the tags "*special" and "2-sat" are never stripped.
_LIST_MARKER = re.compile(r"^(?:[-•*]|\d+[.)])\s+")
# What surrounds a tag in an answer without belonging to it: spaces, quotes and backticks.
_SURROUNDING = " \t\"'`“”‘’"
# The costs of an answer in USD, under the keys of MLflow's CostKey.
_COST_KEYS = ("input_cost", "output_cost", "total_cost")
# LLM_PRICES holds USD per million tokens.
_PRICE_UNIT = 1_000_000
# How many problems an error message lists before it counts the rest.
_LISTED_PROBLEMS = 10
# An answer's started_ns is in nanoseconds.
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class ParsedTags:
    """The tags read from one answer.

    Attributes:
        tags: The answer's tags that are in the tag list, each once, in the order they appear.
        unknown_pieces: How many pieces of the answer name no tag of the list: a piece that
            repeats is counted each time.
        failed: Whether the answer failed. It scores as the empty tag set.
    """

    tags: tuple[str, ...]
    unknown_pieces: int
    failed: bool


# What a failed answer is read as: no tag, and no piece to count.
_FAILED = ParsedTags((), 0, failed=True)


def parse_text_tags(output_text: str, all_tags: Sequence[str] = ALL_TAGS) -> ParsedTags:
    """Read the tags of a free-text answer with the fixed parser.

    The answer is cut into pieces at commas and line breaks. Each piece loses its list marker,
    then its surrounding spaces, quotes and backticks and one final period; it is lower-cased
    and its inner spaces are collapsed. A piece that spells a tag of ``all_tags`` is kept, once;
    an empty piece is skipped; any other piece is counted.

    Args:
        output_text: The answer's text.
        all_tags: The tags the model chooses from.

    Returns:
        The tags in the order they first appear, and the number of pieces that name no tag.
        It never fails: an answer with no tag, even an empty one, is the empty tag set.
    """
    known = set(all_tags)
    tags: dict[str, None] = {}
    unknown_pieces = 0
    for line in output_text.splitlines():
        for piece in line.split(","):
            name = _tag_name(piece)
            if name in known:
                tags[name] = None
            elif name:
                unknown_pieces += 1
    return ParsedTags(tuple(tags), unknown_pieces, failed=False)


def parse_json_tags(output_text: str, all_tags: Sequence[str] = ALL_TAGS) -> ParsedTags:
    """Read the tags of a ``json`` answer, which is ``{"tags": [...]}``.

    A tag is kept when it is exactly one of ``all_tags`` (the schema's enum gives the exact
    names), once however often it repeats; any other item is counted.

    Args:
        output_text: The answer's text.
        all_tags: The tags the model chooses from.

    Returns:
        The tags in the order they first appear, and the number of items outside the list. The
        answer fails when its text is not JSON, or is not an object with a ``tags`` list of
        strings.
    """
    items = _json_items(output_text)
    if items is None:
        return _FAILED
    known = set(all_tags)
    tags = tuple(dict.fromkeys(item for item in items if item in known))
    return ParsedTags(tags, sum(item not in known for item in items), failed=False)


# The parser of each output format of llm.settings.OUTPUT_FORMATS.
_PARSERS: dict[str, Callable[[str, Sequence[str]], ParsedTags]] = {
    "text": parse_text_tags,
    "json": parse_json_tags,
}


def parse_answer(
    answer: LlmAnswer, output_format: str, all_tags: Sequence[str] = ALL_TAGS
) -> ParsedTags:
    """Read the tags of an answer, applying the failure rule.

    The answer fails, and scores as the empty tag set, when its status is not ``completed``
    (``incomplete``, ``refused``, or ``api_error:<HTTP status>`` when the API refused the
    request), when it has no text, or, for ``json``, when its JSON cannot be read. Its stored
    ``tags``, ``failed`` and ``unknown_pieces`` are not read.

    Args:
        answer: The answer.
        output_format: One of ``OUTPUT_FORMATS``: the parser to read its text with.
        all_tags: The tags the model chooses from.

    Returns:
        The answer's tags, its unknown pieces, and whether it failed.

    Raises:
        ValueError: If ``output_format`` is not one of ``OUTPUT_FORMATS``.
    """
    if output_format not in _PARSERS:
        raise ValueError(
            f"{output_format!r} is not an output format: expected one of {OUTPUT_FORMATS}."
        )
    if answer.status != "completed" or answer.output_text is None:
        return _FAILED
    return _PARSERS[output_format](answer.output_text, all_tags)


def encode_answers(
    answers: Sequence[LlmAnswer],
    index: pd.Index,
    output_format: str,
    all_tags: Sequence[str] = ALL_TAGS,
) -> pd.DataFrame:
    """Encode the answers of some problems as 0/1 tags: each answer's text and status re-parsed.

    Args:
        answers: One answer for each problem of ``index``, in any order.
        index: The ``sample_id`` of each row wanted, in the order wanted.
        output_format: One of ``OUTPUT_FORMATS``.
        all_tags: The tags the model chooses from: the columns.

    Returns:
        Int 0/1, with the index ``index`` and one column per tag of ``all_tags``. A failed
        answer is a row of zeros.

    Raises:
        ValueError: If a problem of ``index`` has no answer, if an answer is to a problem that
            is not in ``index``, if a problem has two answers, or if ``output_format`` is not
            one of ``OUTPUT_FORMATS``.
    """
    by_problem = _by_problem(answers)
    _check_same_problems(by_problem, index)
    column = {tag: position for position, tag in enumerate(all_tags)}
    encoded = np.zeros((len(index), len(column)), dtype=np.int64)
    for row, sample_id in enumerate(index):
        for tag in parse_answer(by_problem[sample_id], output_format, all_tags).tags:
            encoded[row, column[tag]] = 1
    return pd.DataFrame(encoded, index=index, columns=list(all_tags))


def answer_summary(
    answers: Sequence[LlmAnswer], output_format: str, all_tags: Sequence[str] = ALL_TAGS
) -> dict[str, object]:
    """Summarise a set of answers: how they went, what they used and what they cost.

    Every answer is re-parsed, as ``encode_answers`` does, and priced with ``answer_cost_usd``.

    Args:
        answers: The answers, e.g. an answers file's.
        output_format: One of ``OUTPUT_FORMATS``.
        all_tags: The tags the model chooses from.

    Returns:
        A JSON-ready dict: ``answers`` (their number), ``statuses`` (the count of each status),
        ``failed``, ``unknown_pieces``, ``empty_answers`` (the answers that completed and name
        no tag), ``served_models`` (the count of each), ``usage`` (the total of each of
        ``USAGE_KEYS``) and ``cost_usd`` (the total of each of ``cost_usd``'s keys).

    Raises:
        ValueError: If ``output_format`` is not one of ``OUTPUT_FORMATS``, or tokens were
            served by a model that ``config.LLM_PRICES`` does not price.
    """
    parsed = [parse_answer(answer, output_format, all_tags) for answer in answers]
    return {
        "answers": len(answers),
        "statuses": _counts(answer.status for answer in answers),
        **_health(parsed),
        "served_models": _counts(answer.served_model for answer in answers if answer.served_model),
        "usage": {key: sum(answer.usage[key] for answer in answers) for key in USAGE_KEYS},
        "cost_usd": _total_cost(answers),
    }


def cost_usd(usage: Mapping[str, int], model: str) -> dict[str, float]:
    """Price an answer's tokens with ``config.LLM_PRICES``.

    The input tokens that were cached are charged at the cached price and the others at the
    input price; the output tokens, the reasoning ones among them, at the output price.

    Args:
        usage: Token counts by ``USAGE_KEYS``.
        model: A key of ``config.LLM_PRICES``, exactly as the API names the served model.

    Returns:
        The cost in USD under the keys ``input_cost``, ``output_cost`` and ``total_cost``,
        which MLflow's ``CostKey`` shares.

    Raises:
        ValueError: If the model has no price, or the usage has more cached than input tokens.
    """
    if model not in LLM_PRICES:
        raise ValueError(
            f"{model!r} has no price in config.LLM_PRICES, which prices {sorted(LLM_PRICES)}: "
            "add a line for it."
        )
    prices = LLM_PRICES[model]
    cached = usage["cached_tokens"]
    uncached = usage["input_tokens"] - cached
    if uncached < 0:
        raise ValueError(
            f"The usage has more cached tokens ({cached}) than input tokens "
            f"({usage['input_tokens']})."
        )
    input_cost = (uncached * prices["input"] + cached * prices["cached_input"]) / _PRICE_UNIT
    output_cost = usage["output_tokens"] * prices["output"] / _PRICE_UNIT
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def answer_cost_usd(answer: LlmAnswer) -> dict[str, float]:
    """Price an answer: the one formula of the runner's cost cap and ``per_problem.csv``.

    An answer with no tokens (one that the API refused, or an incomplete one that reported no
    usage) costs nothing, whatever its model.

    Args:
        answer: The answer.

    Returns:
        ``cost_usd(answer.usage, answer.served_model)``, or zeros.

    Raises:
        ValueError: If the answer has tokens and its served model has no price.
    """
    if not any(answer.usage.values()):
        return dict.fromkeys(_COST_KEYS, 0.0)
    return cost_usd(answer.usage, answer.served_model)


def in_flight_seconds(answers: Sequence[LlmAnswer]) -> float:
    """Count the seconds during which at least one of the answers' requests was in flight.

    Each request was in flight from its ``started_ns`` for its ``latency_s``. Overlapping
    requests count once, and the time between two runs (when none was in flight) not at all,
    so the answers of a run resumed later describe the pace of the runs alone.

    Args:
        answers: Answers, in any order, e.g. those of a method's answers files.

    Returns:
        The seconds that the union of their requests' intervals covers; 0.0 for no answer.
    """
    intervals = sorted(
        (answer.started_ns, answer.started_ns + round(answer.latency_s * _NS_PER_S))
        for answer in answers
    )
    covered = end = 0
    for start, stop in intervals:
        # The part of [start, stop] after every earlier interval: none when it ends before them.
        covered += max(stop, end) - max(start, end)
        end = max(end, stop)
    return covered / _NS_PER_S


def per_problem_frame(answers: Sequence[LlmAnswer], part: str) -> pd.DataFrame:
    """Describe each answer's call, as the rows of ``per_problem.csv``.

    Args:
        answers: One answer per problem of the part, in any order.
        part: ``dev`` or ``test``.

    Returns:
        One row per problem, in ``sample_id`` order, with the columns of
        ``evaluation.predictions.PER_PROBLEM_COLUMNS``; the costs come from ``answer_cost_usd``.

    Raises:
        ValueError: If a problem has two answers, or tokens were served by a model that
            ``config.LLM_PRICES`` does not price.
    """
    by_problem = _by_problem(answers)
    rows = [_per_problem_row(by_problem[sample_id], part) for sample_id in sorted(by_problem)]
    return pd.DataFrame(rows, columns=list(PER_PROBLEM_COLUMNS))


def problem_list(sample_ids: Iterable[int]) -> str:
    """Write sample ids as a short list for an error message: the first ten, then a count.

    Args:
        sample_ids: Any integers, numpy ones included, repeated or not.

    Returns:
        The distinct ids in increasing order, e.g. ``[3, 7]`` or ``[0, 1, ..., 9, and 15 more]``.
    """
    ids = sorted({int(sample_id) for sample_id in sample_ids})
    listed = ", ".join(str(sample_id) for sample_id in ids[:_LISTED_PROBLEMS])
    if len(ids) > _LISTED_PROBLEMS:
        listed += f", and {len(ids) - _LISTED_PROBLEMS} more"
    return f"[{listed}]"


def _tag_name(piece: str) -> str:
    """Return the tag that a piece of an answer spells: plain lower-case words, or nothing."""
    piece = _LIST_MARKER.sub("", piece.strip()).strip(_SURROUNDING)
    if piece.endswith("."):
        piece = piece[:-1].strip(_SURROUNDING)
    return " ".join(piece.lower().split())


def _json_items(output_text: str) -> list[str] | None:
    """Return the strings of the ``tags`` list of a JSON answer, or None if it is not one."""
    try:
        answer = json.loads(output_text)
    except (ValueError, RecursionError):
        return None
    items = answer.get("tags") if isinstance(answer, dict) else None
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        return None
    return items


def _by_problem(answers: Sequence[LlmAnswer]) -> dict[int, LlmAnswer]:
    """Key the answers by problem, refusing two answers to one problem."""
    counts = Counter(answer.sample_id for answer in answers)
    repeated = [sample_id for sample_id, count in counts.items() if count > 1]
    if repeated:
        raise ValueError(
            f"There are several answers for the problems {problem_list(repeated)}: a problem "
            "has one answer, to its current request."
        )
    return {answer.sample_id: answer for answer in answers}


def _check_same_problems(by_problem: Mapping[int, LlmAnswer], index: pd.Index) -> None:
    """Raise unless the answers and the index are about the same problems."""
    wanted = {int(sample_id) for sample_id in index}
    if wanted - by_problem.keys():
        raise ValueError(
            f"There is no answer for the problems {problem_list(wanted - by_problem.keys())}."
        )
    if by_problem.keys() - wanted:
        raise ValueError(
            f"There are answers for the problems {problem_list(by_problem.keys() - wanted)}, "
            "which are not among the problems to encode."
        )


def _counts(values: Iterable[str]) -> dict[str, int]:
    """Count how often each value occurs, listing the values in order."""
    return dict(sorted(Counter(values).items()))


def _health(parsed: Sequence[ParsedTags]) -> dict[str, int]:
    """Count the answers that failed, their unknown pieces, and the empty completed answers."""
    return {
        "failed": sum(tags.failed for tags in parsed),
        "unknown_pieces": sum(tags.unknown_pieces for tags in parsed),
        "empty_answers": sum(not tags.failed and not tags.tags for tags in parsed),
    }


def _total_cost(answers: Sequence[LlmAnswer]) -> dict[str, float]:
    """Sum the cost of the answers, exactly rounded whatever their order."""
    costs = [answer_cost_usd(answer) for answer in answers]
    return {key: math.fsum(cost[key] for cost in costs) for key in _COST_KEYS}


def _per_problem_row(answer: LlmAnswer, part: str) -> dict[str, object]:
    """Return the row of ``per_problem.csv`` that describes an answer."""
    cost = answer_cost_usd(answer)
    return {
        "sample_id": answer.sample_id,
        "part": part,
        "provider": answer.provider,
        "input_tokens": answer.usage["input_tokens"],
        "cached_tokens": answer.usage["cached_tokens"],
        "output_tokens": answer.usage["output_tokens"],
        "input_cost_usd": cost["input_cost"],
        "output_cost_usd": cost["output_cost"],
        "total_cost_usd": cost["total_cost"],
        "latency_s": answer.latency_s,
        "served_model": answer.served_model,
        "response_id": answer.response_id,
        "status": answer.status,
    }
