"""Read Jev's answer lines: their probabilities, their summary, and the model that served them.

A Jev answer is the LLM toolkit's ``LlmAnswer`` line (docs/adr/0007). A ``completed`` line holds
the compact JSON of a probability per focus tag in ``output_text``; a line of any other status
(``malformed``, ``api_error:<HTTP status>``) failed, and scores 0.0 on every focus tag: it is
counted, never dropped. The toolkit's ``parse_answer``, ``encode_answers`` and ``answer_summary``
read a tag list from the text, so they never read a Jev line; its store, ``answer_cost_usd`` and
``per_problem_frame`` read Jev lines as they are.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.answers import answer_cost_usd, problem_list
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.methods.typesafe_jev.settings import MODEL


def probabilities(answers: Sequence[LlmAnswer], index: pd.Index) -> pd.DataFrame:
    """Read each problem's probabilities from its answer: the scores of a Jev method.

    A ``completed`` answer gives the probability its ``output_text`` holds for each focus tag,
    whatever the order of its keys; an answer of any other status failed, and scores 0.0 on
    every tag.

    Args:
        answers: One answer for each problem of ``index``, in any order.
        index: The ``sample_id`` of each row wanted, in the order wanted.

    Returns:
        Float64 probabilities, with the index ``index`` (its order and name) and one column per
        focus tag, in ``FOCUS_TAGS`` order; never NaN.

    Raises:
        ValueError: If a problem of ``index`` has no answer, an answer is to a problem that is
            not in ``index``, a problem has two answers, or a completed answer does not hold a
            probability (a finite number in [0, 1]) for every focus tag: an edited file.
    """
    by_problem = _answer_of_each_problem(answers, index)
    rows = [_probability_row(by_problem[int(sample_id)]) for sample_id in index]
    values = np.array(rows, dtype=np.float64).reshape(len(index), len(FOCUS_TAGS))
    return pd.DataFrame(values, index=index, columns=list(FOCUS_TAGS))


def answers_summary(answers: Sequence[LlmAnswer]) -> dict[str, object]:
    """Summarise answers for a run record: lines, statuses, failures, models, tokens and cost.

    Args:
        answers: The answers, e.g. an answers file's.

    Returns:
        A JSON-ready dict of plain Python values, keys in this order: ``lines``; ``statuses``,
        the count of each status, by status; ``failed``, the lines whose status is not
        ``completed``; ``served_models``, the count of each model, by model, over the lines
        that name one; ``input_tokens`` and ``output_tokens``, their totals; ``cost_usd``, the
        ``math.fsum`` of the lines' ``answer_cost_usd`` totals.

    Raises:
        ValueError: If a line has tokens from a model that ``config.LLM_PRICES`` does not price.
    """
    return {
        "lines": len(answers),
        "statuses": _tally([answer.status for answer in answers]),
        "failed": sum(1 for answer in answers if answer.status != "completed"),
        "served_models": _tally([answer.served_model for answer in answers if answer.served_model]),
        "input_tokens": sum(answer.usage["input_tokens"] for answer in answers),
        "output_tokens": sum(answer.usage["output_tokens"] for answer in answers),
        "cost_usd": math.fsum(answer_cost_usd(answer)["total_cost"] for answer in answers),
    }


def check_served_model(answers: Sequence[LlmAnswer]) -> None:
    """Refuse a completed answer that a model other than the pinned one served.

    The API may answer with another model than the one a request names (the alias
    ``jev-latest``, another version): such an answer is never priced as Jev's nor mixed with
    its answers. ``MODEL`` is read when the function is called. Failed lines, which name no
    model, pass.

    Args:
        answers: The answers, e.g. an answers file's.

    Raises:
        ValueError: If a completed answer's ``served_model`` is not ``MODEL``; the message
            names the problems and the models.
    """
    others = [
        answer
        for answer in answers
        if answer.status == "completed" and answer.served_model != MODEL
    ]
    if others:
        models = sorted({str(answer.served_model) for answer in others})
        raise ValueError(
            f"The answers to the problems {problem_list(answer.sample_id for answer in others)} "
            f"were served by {models}, not {MODEL}: a Jev method's answers all come from its "
            "pinned model, so these answers files can be neither completed nor frozen; another "
            "model is a new method under a new name (docs/adr/0003)."
        )


def _answer_of_each_problem(answers: Sequence[LlmAnswer], index: pd.Index) -> dict[int, LlmAnswer]:
    """Key the answers by problem: one for each problem of the index, and none for another."""
    by_problem = _answers_by_problem(answers)
    asked = [int(sample_id) for sample_id in index]
    unanswered = [sample_id for sample_id in asked if sample_id not in by_problem]
    if unanswered:
        raise ValueError(f"The problems {problem_list(unanswered)} have no answer yet.")
    unasked = set(by_problem) - set(asked)
    if unasked:
        raise ValueError(
            f"The problems {problem_list(unasked)} have an answer but are not among the "
            "problems to score."
        )
    return by_problem


def _answers_by_problem(answers: Sequence[LlmAnswer]) -> dict[int, LlmAnswer]:
    """Key the answers by problem, refusing a problem that has several."""
    by_problem: dict[int, LlmAnswer] = {}
    repeated = set()
    for answer in answers:
        if answer.sample_id in by_problem:
            repeated.add(answer.sample_id)
        by_problem[answer.sample_id] = answer
    if repeated:
        raise ValueError(
            f"The problems {problem_list(repeated)} have several answers: an answers file holds "
            "one line per problem, the answer to its current request."
        )
    return by_problem


def _probability_row(answer: LlmAnswer) -> list[float]:
    """Return an answer's probabilities in FOCUS_TAGS order: zeros when it failed."""
    if answer.status != "completed":
        return [0.0] * len(FOCUS_TAGS)
    held = _json_object(answer.output_text)
    row = [held.get(tag) for tag in FOCUS_TAGS]
    if not all(_is_probability(value) for value in row):
        raise ValueError(
            f"The completed answer to problem {answer.sample_id} does not hold a probability "
            f"for each focus tag: {answer.output_text!r}. An answers file is only ever appended "
            "to: restore it from git."
        )
    return [float(value) for value in row]


def _json_object(text: str | None) -> dict[str, object]:
    """Return the JSON object that a text holds, or an empty one if it holds none."""
    try:
        held = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def _is_probability(value: object) -> bool:
    """Tell whether a value read from JSON is a probability: a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0 <= value <= 1


def _tally(values: Sequence[str]) -> dict[str, int]:
    """Count each value, the values in sorted order."""
    counts: dict[str, int] = {}
    for value in sorted(values):
        counts[value] = counts.get(value, 0) + 1
    return counts
