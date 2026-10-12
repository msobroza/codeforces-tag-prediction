"""Find the self-identifying solutions: solutions that name or quote their own problem.

Every problem predates the knowledge cutoff of current LLMs and pre-trained encoders, and
Codeforces shows each problem's tags beside its statement, so a model may recall a problem's
tags instead of inferring them. A solution that names its problem (a Codeforces URL, a contest
id) or quotes its description makes that recall easier: these problems form the
contamination-sensitivity slice, and notebooks report every score also without them.

The rule was fixed before any API call: the regexes of ``config.SELF_IDENTIFYING_PATTERNS``,
searched in the solution, and ``quotes_description``, true when the solution's comments and
triple-quoted strings hold at least ``SELF_IDENTIFYING_MIN_SHARE`` of the description's
distinct word ``SELF_IDENTIFYING_NGRAM``-grams. ``python -m codeforces_tags.evaluation slices``
freezes the slice once in ``config.SELF_IDENTIFYING_FILE`` and later only checks it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from codeforces_tags.config import (
    SELF_IDENTIFYING_FILE,
    SELF_IDENTIFYING_MIN_SHARE,
    SELF_IDENTIFYING_NGRAM,
    SELF_IDENTIFYING_PATTERNS,
)

QUOTES_DESCRIPTION = "quotes_description"
# Every rule, in the order the rules column lists them: the patterns, then the quoted description.
SELF_IDENTIFYING_RULES: tuple[str, ...] = (*SELF_IDENTIFYING_PATTERNS, QUOTES_DESCRIPTION)
RULE_SEPARATOR = ";"
# The columns of the frozen slice, after its sample_id index.
SLICE_COLUMNS = ("part", "rules")
# A solution's comments: each "#" to the end of its line, then each triple-quoted string; two
# separate searches, so a "#" inside a docstring is read twice.
_COMMENT_LINE = re.compile(r"#[^\n]*")
_TRIPLE_QUOTED = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_WORD = re.compile(r"[a-z]+")


def rule_matches(problems: pd.DataFrame) -> pd.DataFrame:
    """Tell which rules each problem's solution matches.

    Args:
        problems: Problems indexed by ``sample_id``, with ``prob_desc_description`` and
            ``source_code``; a missing value reads as empty text.

    Returns:
        One bool column per rule of ``SELF_IDENTIFYING_RULES``, indexed like ``problems``.
    """
    code = problems["source_code"].fillna("")
    matches = pd.DataFrame(
        {name: code.str.contains(pattern) for name, pattern in SELF_IDENTIFYING_PATTERNS.items()},
        index=problems.index,
    )
    descriptions = problems["prob_desc_description"].fillna("")
    shares = [
        _quoted_share(text, solution) for text, solution in zip(descriptions, code, strict=True)
    ]
    matches[QUOTES_DESCRIPTION] = [share >= SELF_IDENTIFYING_MIN_SHARE for share in shares]
    return matches.astype(bool)


def self_identifying_solutions(problems: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """List the problems whose solution names or quotes its own problem.

    Args:
        problems: Every problem, as ``load_dataset()`` returns them (``read_samples()`` gives
            the same slice), indexed by ``sample_id``.
        split: The frozen split, from ``load_split``, over the same problems.

    Returns:
        One row per self-identifying problem, indexed by ``sample_id`` in increasing order:
        ``part`` (dev or test) and ``rules``, the names of the rules it matches, in rule order,
        joined by ``RULE_SEPARATOR``.

    Raises:
        ValueError: If ``problems`` and ``split`` do not hold the same sample ids.
    """
    not_in_split = problems.index.difference(split.index)
    not_in_problems = split.index.difference(problems.index)
    if len(not_in_split) or len(not_in_problems):
        raise ValueError(
            f"The slice is defined over every problem of the split: {len(not_in_split)} problems "
            f"are not in the split, and {len(not_in_problems)} split rows are not in the "
            "problems. Pass load_dataset() with load_split()."
        )
    matches = rule_matches(problems).sort_index()
    flagged = matches[matches.any(axis=1)]
    rules = [RULE_SEPARATOR.join(flagged.columns[row]) for row in flagged.to_numpy()]
    return pd.DataFrame(
        {
            "part": split["part"].reindex(flagged.index).astype("str"),
            "rules": pd.Series(rules, index=flagged.index, dtype="str"),
        }
    )


def freeze_slice(slice_: pd.DataFrame, path: Path = SELF_IDENTIFYING_FILE) -> str:
    """Save the slice once, then only check that it has not changed.

    The rule was fixed before any API call, so an existing file is compared with ``slice_``
    rather than overwritten.

    Args:
        slice_: The slice from ``self_identifying_solutions``.
        path: CSV file of the frozen slice.

    Returns:
        A line to print: the file was saved, or it matches ``slice_``.

    Raises:
        ValueError: If the file exists and holds another slice.
    """
    path = Path(path)
    slice_ = slice_[list(SLICE_COLUMNS)]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        slice_.to_csv(path, index_label="sample_id")
        return f"saved {len(slice_):,} self-identifying problems to {path.name}"
    if not load_slice(path).equals(slice_):
        raise ValueError(
            f"self_identifying_solutions no longer gives the frozen slice in {path}: the data, "
            "the split or the rule changed. The rule was fixed before any API call; delete the "
            "file only to define a new slice before any API method runs."
        )
    return f"{path.name} matches the slice ({len(slice_):,} problems)"


def load_slice(path: Path = SELF_IDENTIFYING_FILE) -> pd.DataFrame:
    """Read the frozen slice.

    Args:
        path: CSV written by ``freeze_slice``.

    Returns:
        Columns ``part`` and ``rules`` (str), indexed by ``sample_id`` (int64).
    """
    dtypes = {"sample_id": "int64", "part": "str", "rules": "str"}
    return pd.read_csv(path, index_col="sample_id", dtype=dtypes)


def _quoted_share(description: str, code: str) -> float:
    """The share of the description's distinct word n-grams found in the solution's comments."""
    grams = _word_grams(description)
    if not grams:
        return 0.0
    comments = " ".join(_COMMENT_LINE.findall(code) + _TRIPLE_QUOTED.findall(code))
    return len(grams & _word_grams(comments)) / len(grams)


def _word_grams(text: str) -> set[tuple[str, ...]]:
    """The distinct word n-grams of a text, lowercased; words are runs of the letters a to z."""
    words = _WORD.findall(text.lower())
    size = SELF_IDENTIFYING_NGRAM
    return {tuple(words[start : start + size]) for start in range(len(words) - size + 1)}
