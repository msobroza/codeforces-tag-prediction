"""Jev's words as data: the convention, the question and each focus tag's definition.

The frozen Jev methods read the words of the prompt file ``templates/jev-tags.v1.yaml``, which
``settings`` reads (``SEED_WORDS``); a searched method reads its own from its frozen files
(docs/adr/0010). A search rewrites nine pieces of them, the convention and the eight tag
definitions, each a text: the convention as itself, a tag's definition as canonical JSON
(``piece_text``), which ``parse_piece`` reads back exactly. The question template is not a
piece. ``piece_problem`` names what keeps a value from its piece's shape: the one check that
``parse_piece`` raises on and that a search's reading of a rewrite uses.

The dataset-text check (``copied_sequences``) finds the word sequences that words share with a
problem's text and that the seed words, written by hand, do not hold (docs/adr/0004).

Which words a method sends is ``words_of``, what its recipe and method folder hash is
``method_settings``, and what its run record names of them is ``settings_entry``. A calibrated
twin sends its base's words (``base_of``), and hashes its base's settings with its threshold
rule (docs/adr/0011). This module imports no SDK and nothing of the search.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev.settings import (
    CONVENTION,
    FROZEN_NAMES,
    MODEL,
    NAMES,
    PARAMS_FILE,
    PLATT_BASES,
    PROMPT_PATH,
    QUESTION,
    SEARCH_LENGTH_FACTOR,
    SEARCH_NGRAM,
    SETTINGS,
    STATE_FIELDS,
    TAG_DEFINITIONS,
    TUNING_FOLDER,
)
from codeforces_tags.model.platt_lipton import RULE_SETTINGS
from codeforces_tags.provenance import sha256_json

# GEPA's component names: the convention, then the focus tags ("number theory" keeps its space).
PIECES: tuple[str, ...] = ("convention", *FOCUS_TAGS)
# The two answers of a tag's criteria, in order, and the key of what each includes or excludes.
LISTINGS: dict[str, str] = {"true": "includes", "false": "excludes"}
# The keys of the words, in the order JevWords.settings() writes them.
WORD_KEYS: tuple[str, ...] = ("convention", "question", "tag_definitions")
# A word of a text: a run of letters and digits (no underscore), once the text is lower-cased.
_WORD = re.compile(r"[^\W_]+")
# What keeps a value from its piece's shape, in the order piece_problem names it: a missing key,
# another key or a wrong type, before a blank text or an empty list.
_PROBLEMS: tuple[str, ...] = ("invalid_json", "empty_field")


@dataclass(frozen=True)
class JevWords:
    """Every word of a Jev request besides the state and the model: a value, compared and hashed.

    Attributes:
        convention: What every question tells Jev about Codeforces tags.
        question: The question asked for each tag: a ``str.format`` template with ``{tag}``.
        tag_definitions: Each focus tag's criteria, in ``FOCUS_TAGS`` order: ``{"true":
            {"definition", "includes"}, "false": {"definition", "excludes"}}``.
    """

    convention: str
    question: str
    tag_definitions: Mapping[str, Mapping[str, Mapping[str, object]]]

    def settings(self) -> dict[str, object]:
        """Return the words as plain JSON values that share nothing with them.

        Returns:
            ``convention``, ``question`` and ``tag_definitions``, in that order; the
            definitions in ``FOCUS_TAGS`` order, each in the key order a request sends.
        """
        return {
            "convention": self.convention,
            "question": self.question,
            "tag_definitions": {tag: _canonical(self.tag_definitions[tag]) for tag in FOCUS_TAGS},
        }

    def sha256(self) -> str:
        """Hash the words.

        Returns:
            ``sha256_json`` of ``settings()``: the name of the words in the search's caches and
            records.
        """
        return sha256_json(self.settings())


# The words of the frozen methods, written by hand in the prompt file jev-tags v1 and read by
# settings: the words are defined once. It holds TAG_DEFINITIONS itself, not a copy: the requests
# of the frozen methods are built from it.
SEED_WORDS = JevWords(CONVENTION, QUESTION, TAG_DEFINITIONS)


def piece_text(words: JevWords, piece: str) -> str:
    """Write one piece of the words as the text a search rewrites.

    Args:
        words: The words.
        piece: One of ``PIECES``.

    Returns:
        The convention itself, or a tag's criteria as JSON: ``{"true": {"definition",
        "includes"}, "false": {"definition", "excludes"}}`` in that key order, indented by 2,
        non-ASCII text kept.

    Raises:
        ValueError: If ``piece`` is not one of ``PIECES``.
    """
    _check_piece(piece)
    if piece == "convention":
        return words.convention
    return json.dumps(_canonical(words.tag_definitions[piece]), ensure_ascii=False, indent=2)


def parse_piece(piece: str, text: str) -> str | dict[str, dict[str, object]]:
    """Read one piece back from its text: the inverse of ``piece_text``.

    Args:
        piece: One of ``PIECES``.
        text: The piece's text.

    Returns:
        The convention, unchanged, or a tag's criteria in the key order a request sends.

    Raises:
        ValueError: If ``piece`` is not one of ``PIECES``, or the text is not one of its shape:
            a convention that is not a text or is blank; criteria that are not JSON, that lack a
            key or hold another one, whose definition is not a text or is blank, or whose
            includes or excludes are not a non-empty list of texts that are not blank. The
            message names the piece.
    """
    _check_piece(piece)
    if not isinstance(text, str):
        raise ValueError(_not_a_text(piece, text))
    if piece == "convention":
        return _checked(piece, text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The piece {piece!r} is not JSON: {error.msg}.") from error
    return _checked(piece, value)


def piece_problem(piece: str, value: object) -> str | None:
    """Name what keeps a piece's value from the shape a request needs, if anything.

    Args:
        piece: One of ``PIECES``.
        value: The convention's text, or a tag's criteria as decoded JSON (not their text).

    Returns:
        ``"invalid_json"`` if a key is missing or another one is there, or a value is of the
        wrong type: a convention that is not a text; criteria, or one of their two answers,
        that are not a mapping of exactly their keys; a definition that is not a text; includes
        or excludes that are not a list of texts. Else ``"empty_field"`` if the convention or a
        definition is blank, or the includes or excludes are empty or hold a blank item. Else
        None.

    Raises:
        ValueError: If ``piece`` is not one of ``PIECES``.
    """
    _check_piece(piece)
    problem = _first_problem(piece, value)
    return None if problem is None else problem[0]


def candidate_from_words(words: JevWords) -> dict[str, str]:
    """Write the words as a GEPA candidate.

    Args:
        words: The words.

    Returns:
        The text of each piece, by piece, in ``PIECES`` order.
    """
    return {piece: piece_text(words, piece) for piece in PIECES}


def words_from_candidate(candidate: Mapping[str, str], question: str = QUESTION) -> JevWords:
    """Read the words of a GEPA candidate: the inverse of ``candidate_from_words``.

    Args:
        candidate: The text of each piece, by piece: exactly the keys of ``PIECES``.
        question: The question template, which is not a piece.

    Returns:
        The words, their definitions in ``FOCUS_TAGS`` order.

    Raises:
        ValueError: If the candidate lacks a piece or holds another key, or as ``parse_piece``.
    """
    if set(candidate) != set(PIECES):
        raise ValueError(
            f"A candidate holds exactly the pieces {list(PIECES)}, not {sorted(candidate)}."
        )
    return JevWords(
        convention=parse_piece("convention", candidate["convention"]),
        question=question,
        tag_definitions={tag: parse_piece(tag, candidate[tag]) for tag in FOCUS_TAGS},
    )


def piece_length(piece: str, value: str | Mapping[str, object]) -> int:
    """Measure a piece as its length cap counts it.

    Args:
        piece: One of ``PIECES``.
        value: The convention's text; for a tag, its criteria, or their text (``piece_text``).

    Returns:
        The characters of the convention, or of the criteria's compact JSON (non-ASCII text
        kept), whatever the indentation of its text.

    Raises:
        ValueError: As ``parse_piece``.
    """
    _check_piece(piece)
    if piece == "convention":
        return len(parse_piece(piece, value))
    criteria = parse_piece(piece, value) if isinstance(value, str) else _checked(piece, value)
    return len(json.dumps(criteria, ensure_ascii=False, separators=(",", ":")))


def length_cap(piece: str) -> int:
    """Return the most characters a piece may hold: ``SEARCH_LENGTH_FACTOR`` times the seed's.

    Args:
        piece: One of ``PIECES``.

    Returns:
        ``floor(SEARCH_LENGTH_FACTOR * piece_length(piece, <the seed's value>))``.

    Raises:
        ValueError: If ``piece`` is not one of ``PIECES``.
    """
    _check_piece(piece)
    if piece == "convention":
        seed: str | Mapping[str, object] = SEED_WORDS.convention
    else:
        seed = SEED_WORDS.tag_definitions[piece]
    return math.floor(SEARCH_LENGTH_FACTOR * piece_length(piece, seed))


def words_from_json(values: Mapping[str, object]) -> JevWords:
    """Read words from JSON values: the inverse of ``JevWords.settings``.

    Args:
        values: ``convention``, ``question`` and ``tag_definitions``, and no other key.

    Returns:
        The words, their definitions in ``FOCUS_TAGS`` order and in the key order a request
        sends.

    Raises:
        ValueError: If a key is missing or another one is there, if the question is not a text
            that holds ``{tag}``, if the definitions are not those of exactly the focus tags, or
            as ``parse_piece``.
    """
    if not isinstance(values, Mapping) or set(values) != set(WORD_KEYS):
        raise ValueError(
            f"Jev's words hold exactly the keys {list(WORD_KEYS)}, not {_keys(values)}."
        )
    question = values["question"]
    if not isinstance(question, str) or "{tag}" not in question:
        raise ValueError("The question of Jev's words must be a text that holds '{tag}'.")
    definitions = values["tag_definitions"]
    if not isinstance(definitions, Mapping) or set(definitions) != set(FOCUS_TAGS):
        raise ValueError(
            f"Jev's tag definitions are those of exactly the focus tags {list(FOCUS_TAGS)}, not "
            f"{_keys(definitions)}."
        )
    return JevWords(
        convention=parse_piece("convention", values["convention"]),
        question=question,
        tag_definitions={tag: _checked(tag, definitions[tag]) for tag in FOCUS_TAGS},
    )


def word_strings(words: JevWords) -> list[str]:
    """List every free text of the words.

    Args:
        words: The words.

    Returns:
        The convention, the question, then for each tag in ``FOCUS_TAGS`` order its yes
        definition and includes, then its no definition and excludes.
    """
    strings = [words.convention, words.question]
    for tag in FOCUS_TAGS:
        for answer, listing in LISTINGS.items():
            side = words.tag_definitions[tag][answer]
            strings += [side["definition"], *side[listing]]
    return strings


def word_sequences(text: str, n: int = SEARCH_NGRAM) -> set[tuple[str, ...]]:
    """Return the sequences of ``n`` consecutive words of a text.

    Args:
        text: Any text.
        n: The number of words of a sequence.

    Returns:
        The sequences, each a tuple of ``n`` words: the text is lower-cased, then split on every
        run of characters that are not letters or digits. A text of fewer words has none.
    """
    tokens = _WORD.findall(text.lower())
    return {tuple(tokens[start : start + n]) for start in range(len(tokens) - n + 1)}


def copied_sequences(words: JevWords, texts: Iterable[str], n: int = SEARCH_NGRAM) -> list[str]:
    """Find the word sequences that words share with texts and that the seed words do not hold.

    Each free text of the words (``word_strings``) is cut into sequences on its own, so no
    sequence runs from one item into the next. A sequence of the seed words is exempt: those
    words were written by hand, not copied (docs/adr/0010).

    Args:
        words: The words to check.
        texts: The texts they must not copy from, e.g. every problem's statement, notes and
            solution; a missing one (not a str, e.g. NaN) holds no sequence.
        n: The number of words of a sequence.

    Returns:
        The shared sequences, each as its words joined by single spaces, sorted; empty when the
        words copy nothing.
    """
    own = _sequences_of(word_strings(words), n) - _sequences_of(word_strings(SEED_WORDS), n)
    shared: set[tuple[str, ...]] = set()
    if own:
        for text in texts:
            if isinstance(text, str):
                shared |= own & word_sequences(text, n)
    return sorted(" ".join(sequence) for sequence in shared)


def method_settings(name: str, words: JevWords) -> dict[str, object]:
    """Return what a Jev method's recipe and method folder hash, besides its name and tags.

    Args:
        name: One of ``NAMES``.
        words: The words the method sends.

    Returns:
        For a frozen method, ``settings.SETTINGS`` itself, the bytes it was frozen with; for the
        searched method, the model, its state fields and its words (``JevWords.settings``); for
        a calibrated twin, its base, the hash of its base's ``method_settings`` with these words
        and its threshold rule (``RULE_SETTINGS``, docs/adr/0011).

    Raises:
        ValueError: If ``name`` is not a Jev method, or a frozen method, or the twin of one, is
            given other words than the seed words: its words never change (docs/adr/0003).
    """
    _check_name(name)
    if name in PLATT_BASES:
        base = PLATT_BASES[name]
        return {
            "base": base,
            "base_settings_sha256": sha256_json(method_settings(base, words)),
            "threshold_rule": dict(RULE_SETTINGS),
        }
    if name in FROZEN_NAMES:
        if words != SEED_WORDS:
            raise ValueError(
                f"{name} sends the seed words, those of the prompt file {PROMPT_PATH}: a frozen "
                "method's words never change (docs/adr/0003)."
            )
        return SETTINGS
    return {"model": MODEL, "state_fields": {name: list(STATE_FIELDS[name])}, **words.settings()}


def read_words(path: Path) -> JevWords:
    """Read words from a JSON file of the searched method.

    Args:
        path: ``tuning/params.json``, whose ``words`` key holds them, or the method folder's
            ``words.json``, which holds them alone.

    Returns:
        The words, as ``words_from_json`` reads them.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the words are not of their shape (as ``words_from_json``); the message
            names the file.
    """
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    values = stored["words"] if isinstance(stored, Mapping) and "words" in stored else stored
    try:
        return words_from_json(values)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def base_of(name: str) -> str:
    """Return the Jev method whose requests and answers a method's own are.

    Args:
        name: One of ``NAMES``.

    Returns:
        A calibrated twin's base (``PLATT_BASES``); any other Jev method itself.

    Raises:
        ValueError: If ``name`` is not a Jev method.
    """
    _check_name(name)
    return PLATT_BASES.get(name, name)


def words_of(name: str) -> JevWords:
    """Return the words a Jev method sends.

    Args:
        name: One of ``NAMES``.

    Returns:
        ``SEED_WORDS`` for a frozen method; for the searched method, the words its search chose,
        read from its frozen ``tuning/params.json``; for a calibrated twin, its base's words.

    Raises:
        ValueError: If ``name`` is not a Jev method, or as ``read_words``.
        FileNotFoundError: If the searched method's ``tuning/params.json`` does not exist yet.
    """
    base = base_of(name)
    if base in FROZEN_NAMES:
        return SEED_WORDS
    path = method_dir(base) / TUNING_FOLDER / PARAMS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: {base} sends the words its search chose. Run the search, "
            f"then `python -m codeforces_tags.methods.typesafe_jev export --method {base}` "
            "(docs/adr/0010)."
        )
    return read_words(path)


def settings_entry(name: str) -> dict[str, object]:
    """Return the ``settings`` entry of a Jev method's run record, which its freeze writes.

    The recipe hashes every word of the settings; the entry names the model, the problem fields
    the requests send and the hash of ``method_settings``. A calibrated twin's verify and second
    freeze compare it with the frozen record's (``platt_commands.check_record``).

    Args:
        name: One of ``NAMES``.

    Returns:
        ``model`` (``settings.MODEL``), ``state_fields`` (the method's ``STATE_FIELDS``, as a
        list) and ``settings_sha256`` (``sha256_json`` of ``method_settings`` with the words of
        ``words_of``).

    Raises:
        ValueError: If ``name`` is not a Jev method, or as ``words_of``.
        FileNotFoundError: As ``words_of``.
    """
    settings_sha256 = sha256_json(method_settings(name, words_of(name)))  # checks the name first
    return {
        "model": MODEL,
        "state_fields": list(STATE_FIELDS[name]),
        "settings_sha256": settings_sha256,
    }


def _check_name(name: str) -> None:
    """Raise unless ``name`` is the name of a Jev method."""
    if name not in NAMES:
        raise ValueError(f"Unknown Jev method {name!r}: expected one of {NAMES}.")


def _sequences_of(strings: Iterable[str], n: int) -> set[tuple[str, ...]]:
    """Return the sequences of ``n`` words of each string, cut on its own, together."""
    return set().union(*(word_sequences(string, n) for string in strings))


def _check_piece(piece: str) -> None:
    """Raise unless ``piece`` is one of ``PIECES``."""
    if piece not in PIECES:
        raise ValueError(
            f"{piece!r} is not a piece of Jev's words: expected one of {list(PIECES)}."
        )


def _canonical(criteria: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Copy a tag's criteria as plain values, in the key order a request sends."""
    return {
        answer: {
            "definition": criteria[answer]["definition"],
            listing: list(criteria[answer][listing]),
        }
        for answer, listing in LISTINGS.items()
    }


def _checked(piece: str, value: object) -> str | dict[str, dict[str, object]]:
    """Return a piece's value as a request sends it, or raise on the problem it has."""
    problem = _first_problem(piece, value)
    if problem is not None:
        raise ValueError(problem[1])
    return value if piece == "convention" else _canonical(value)


def _first_problem(piece: str, value: object) -> tuple[str, str] | None:
    """Return the first problem of a piece's value, in ``_PROBLEMS`` order, with its message."""
    if piece == "convention":
        problems = _convention_problems(value)
    else:
        problems = _criteria_problems(piece, value)
    return min(problems, key=lambda problem: _PROBLEMS.index(problem[0]), default=None)


def _convention_problems(value: object) -> list[tuple[str, str]]:
    """List what keeps a value from being a convention: a text that is not blank."""
    if not isinstance(value, str):
        return [("invalid_json", _not_a_text("convention", value))]
    return [] if value.strip() else [("empty_field", "The piece 'convention' is blank.")]


def _criteria_problems(piece: str, value: object) -> list[tuple[str, str]]:
    """List what keeps a value from being a tag's criteria, each problem with its message."""
    if not isinstance(value, Mapping) or set(value) != set(LISTINGS):
        message = (
            f"The piece {piece!r} holds exactly the keys {list(LISTINGS)}, not {_keys(value)}."
        )
        return [("invalid_json", message)]
    return [
        problem for answer in LISTINGS for problem in _side_problems(piece, answer, value[answer])
    ]


def _side_problems(piece: str, answer: str, value: object) -> list[tuple[str, str]]:
    """List what keeps one answer of a tag's criteria from a definition and a list of texts."""
    listing = LISTINGS[answer]
    if not isinstance(value, Mapping) or set(value) != {"definition", listing}:
        message = (
            f"The piece {piece!r} holds exactly the keys ['definition', {listing!r}] under "
            f"{answer!r}, not {_keys(value)}."
        )
        return [("invalid_json", message)]
    found = [
        (
            _text_problem(value["definition"]),
            f"The piece {piece!r} needs a definition under {answer!r} that is a text, not blank.",
        ),
        (
            _listing_problem(value[listing]),
            f"The piece {piece!r} needs under {answer!r} a non-empty list of texts in "
            f"{listing!r}, none of them blank.",
        ),
    ]
    return [(reason, message) for reason, message in found if reason is not None]


def _text_problem(value: object) -> str | None:
    """Name what keeps a value from being a text that is not blank, if anything."""
    if not isinstance(value, str):
        return "invalid_json"
    return None if value.strip() else "empty_field"


def _listing_problem(value: object) -> str | None:
    """Name what keeps a value from being a non-empty list of texts, none blank, if anything."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return "invalid_json"
    return None if value and all(item.strip() for item in value) else "empty_field"


def _not_a_text(piece: str, value: object) -> str:
    """Write the message for a piece given as something other than a text."""
    return f"The piece {piece!r} must be a text, not a {type(value).__name__}."


def _keys(value: object) -> object:
    """Describe what a value holds in an error: its sorted keys, or its type."""
    return sorted(value) if isinstance(value, Mapping) else type(value).__name__
