"""The requests of the Jev methods: each problem's state, the eight Noul questions and the hash.

A request is one call of TypeSafe's ``system_one``: a state, the model, and one Noul question per
focus tag. Jev reads the state (the problem's statement, and for ``jev-statement-code`` its
solution) and answers each question with the probability that the answer is yes.

A request is hashed as the exact JSON text the SDK puts on the wire, so a line of an answers file
names the request it answers (docs/adr/0003, docs/adr/0007). This module imports no SDK.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.answers import problem_list
from codeforces_tags.methods.typesafe_jev import words as jev_words
from codeforces_tags.methods.typesafe_jev.settings import MODEL, NAMES, STATE_FIELDS
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS, JevWords
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The column of the preprocessed problems that each field of a state is read from.
STATE_COLUMNS: dict[str, str] = {"statement": "statement", "solution": "source_code"}


def question_id(tag: str) -> str:
    """Return the key of a tag's question, in a request and in the response.

    Ids are for code. The question's instructions name the tag, so nothing depends on the id.

    Args:
        tag: A focus tag, e.g. ``"number theory"``.

    Returns:
        The tag with its spaces replaced by underscores, e.g. ``"number_theory"``.
    """
    return tag.replace(" ", "_")


def noul_questions(words: JevWords = SEED_WORDS) -> dict[str, dict[str, object]]:
    """Build the eight questions of a request: one Noul per focus tag.

    Each question asks whether Codeforces would put the tag on the problem (``words.question``,
    filled with the tag), says what a tag is (``words.convention``) and gives the tag's definition
    as what counts as a yes and as a no (``words.tag_definitions``).

    Args:
        words: The words the questions hold; the frozen methods' by default, those of the prompt
            file ``templates/jev-tags.v1.yaml`` (``settings.PROMPT``), from which the questions
            are the bytes the frozen answers files answer.

    Returns:
        The questions by ``question_id``, in ``FOCUS_TAGS`` order. Each holds ``type``,
        ``instructions`` and ``criteria``, in the order the SDK sends them. The criteria are
        copies: changing a question never changes the words.
    """
    return {
        question_id(tag): {
            "type": "noul",
            "instructions": {
                "tag": tag,
                "convention": words.convention,
                "question": words.question.format(tag=tag),
            },
            "criteria": {
                "true": copy.deepcopy(words.tag_definitions[tag]["true"]),
                "false": copy.deepcopy(words.tag_definitions[tag]["false"]),
            },
        }
        for tag in FOCUS_TAGS
    }


def states(problems: pd.DataFrame, name: str) -> pd.Series:
    """Build the state that a Jev method reads for each problem.

    Args:
        problems: Preprocessed problems, as ``raw_to_problems`` returns them.
        name: A key of ``NAMES``: it says which fields the state has (``STATE_FIELDS``).

    Returns:
        One dict per problem, indexed like ``problems``: the text of each field of the method's
        state, in the order of ``STATE_FIELDS``.

    Raises:
        ValueError: If the name is not one of ``NAMES``, or a field is not a text, or is blank,
            for some problem (the message names the field and the problems).
    """
    if name not in NAMES:
        raise ValueError(f"Unknown Jev method {name!r}: expected one of {NAMES}.")
    fields = STATE_FIELDS[name]
    texts = {field: problems[STATE_COLUMNS[field]].tolist() for field in fields}
    _check_texts(problems.index, texts)
    values = [dict(zip(fields, row, strict=True)) for row in zip(*texts.values(), strict=True)]
    return pd.Series(values, index=problems.index, dtype=object)


@dataclass(frozen=True)
class JevRequest:
    """One call of ``system_one`` for one problem, hashed as the JSON text the SDK sends.

    Attributes:
        sample_id: The problem, a Python ``int``: a numpy integer is not JSON.
        model: The model that answers, e.g. ``"jev-1.13.0"``.
        state: The text Jev reads, by field.
        questions: The questions, as ``noul_questions`` builds them. The requests that one call
            of ``requests`` makes share this mapping, and nothing may change it.
    """

    sample_id: int
    model: str
    state: dict[str, str]
    questions: dict[str, dict[str, object]]

    def body(self) -> dict[str, object]:
        """Return the JSON body of the call, its keys in the order the SDK sends them.

        Returns:
            ``state``, ``model`` and ``questions``, in that order: the order is part of the hash.
        """
        return {"state": self.state, "model": self.model, "questions": self.questions}

    def sha256(self) -> str:
        """Hash the request as the JSON text the SDK puts on the wire.

        That text is the compact JSON of the body, in the body's key order, with non-ASCII text
        kept as UTF-8 (checked on every request of the dataset against the SDK's bytes). It is
        not ``provenance.sha256_json``, which sorts the keys and escapes non-ASCII text.

        Returns:
            The sha256 of the text, in hexadecimal.
        """
        text = json.dumps(self.body(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def requests(
    raw_problems: pd.DataFrame, name: str, words: JevWords | None = None
) -> list[JevRequest]:
    """Build the request of each problem for a Jev method.

    Args:
        raw_problems: Rows of ``read_samples``, indexed by ``sample_id``.
        name: A key of ``NAMES``: it says which fields the state has.
        words: The words of the questions; None for the method's own (``words.words_of``):
            the seed words for a frozen method, its search's for the searched method.

    Returns:
        One request per problem, in the order of ``raw_problems``. They share one mapping of
        questions, built by this call.

    Raises:
        ValueError: If the problems are not raw ones with every raw column (as
            ``raw_to_problems`` checks), or as ``states`` and ``words.words_of``.
        FileNotFoundError: As ``words.words_of``, for the searched method before its export.
    """
    chosen = jev_words.words_of(name) if words is None else words
    problems = raw_to_problems(raw_problems)
    questions = noul_questions(chosen)
    return [
        JevRequest(int(sample_id), MODEL, state, questions)
        for sample_id, state in states(problems, name).items()
    ]


def _is_text(value: object) -> bool:
    """Tell whether a value is a text with something in it: not missing, not blank."""
    return isinstance(value, str) and bool(value.strip())


def _check_texts(index: pd.Index, texts: Mapping[str, list[object]]) -> None:
    """Raise unless every field of every problem holds a text."""
    blank = {
        field: [
            sample_id for sample_id, text in zip(index, column, strict=True) if not _is_text(text)
        ]
        for field, column in texts.items()
    }
    missing = [
        f"the {field} of the problems {problem_list(sample_ids)}"
        for field, sample_ids in blank.items()
        if sample_ids
    ]
    if missing:
        raise ValueError(
            "The state of a Jev request needs a text in every field, but some are missing or "
            f"blank: {'; '.join(missing)}."
        )
