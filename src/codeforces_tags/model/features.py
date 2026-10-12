"""The model's inputs: the statement block's text, the code block's tokens and the cues.

``feature_blocks`` assembles them into the feature blocks a configuration keeps, and
``block_feature_names`` names the features of a fitted block.

Frozen import path: skops stores ``statement_text``, ``normalise_statement``, ``code_tokens``
and ``sample_io_cues`` in a saved pipeline by their import path,
``codeforces_tags.model.features.<name>``. Renaming one of them, or moving it to another
module, makes the saved tag models unloadable. ``STORED_FUNCTIONS`` lists these paths; it
must equal what ``skops.io.get_untrusted_types`` reports for the dumped feature blocks with all
three blocks (tests/model/test_features.py pins it, and the classifier adds no untrusted type).
A saved tag model also stores ``raw_to_problems`` and ``CrossFittedTagThresholds``:
``tag_model.STORED_TYPES`` adds them, and lists the only types its ``load`` trusts.
Changing what these functions compute, or the ``CUE_*`` settings they read, changes a saved
model's features without any load error: refit after such a change.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from codeforces_tags.config import CUE_MAX_SAMPLE_LINES, CUE_PLAYER_WORDS, CUE_TREE_MIN_NODES
from codeforces_tags.model.params import ModelParams

STORED_FUNCTIONS = (
    "codeforces_tags.model.features.code_tokens",
    "codeforces_tags.model.features.normalise_statement",
    "codeforces_tags.model.features.sample_io_cues",
    "codeforces_tags.model.features.statement_text",
)  # sorted, as skops lists them
SAMPLE_IO_CUES = (
    "tree_edge_list",
    "graph_edge_list",
    "decimal_output",
    "letters_output",
    "player_output",
)

# Statement rules, applied in order. The moduli come before the powers of ten (10^9+7 holds
# 10^9), and the synonyms before the other commands (\leq would become "leq"). Synonyms and
# commands are padded with spaces: "1\le n" must give "1 le n", not "1le n".
_STATEMENT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\$\$\$"), " "),  # math delimiters
    (re.compile(r"[\u00a0\u2000-\u200b\u202f\u2060\ufeff]"), " "),  # no-break, thin, zero-width
    (re.compile(r"\\[,;!]"), ""),  # LaTeX thin spaces, also between digit groups: 998\,244\,353
    (re.compile(r"(?<!\d)998[ ,]?244[ ,]?353(?!\d)"), "mod998244353"),
    (
        re.compile(
            r"(?<!\d)(?:10\^\{?9\}?\s*\+\s*7"  # 10^9+7, 10^{9} + 7
            r"|1[ ,]?000[ ,]?000[ ,]?007"  # 1000000007, 1 000 000 007
            r"|109\s*\+\s*7)(?!\d)"  # 10<sup>9</sup> + 7, flattened by the scraper until
            # the switch to $$$ LaTeX in 2018
        ),
        "mod1e9p7",
    ),
    (re.compile(r"(?<!\d)10\^\{?(\d+)\}?"), r"1e\1"),  # 10^5 and 10^{18}
    (re.compile(r"\\le(?:q|qslant)?(?![a-zA-Z])|≤"), " le "),
    (re.compile(r"\\ge(?:q|qslant)?(?![a-zA-Z])|≥"), " ge "),
    (re.compile(r"\\neq?(?![a-zA-Z])|≠"), " ne "),
    (re.compile(r"\\[lc]?dots(?![a-zA-Z])|…|\.{3,}"), " dots "),
    (re.compile(r"\\[bp]?mod(?![a-zA-Z])"), " mod "),
    (re.compile(r"\\([a-zA-Z]+)"), r" \1 "),  # any other command keeps its name: \frac -> frac
)
# Code: the moduli in the spellings solutions use, then sklearn's default token pattern.
_CODE_MODULI: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b998244353\b"), "mod998244353"),
    (
        re.compile(
            r"\b(?:10\s*\*\*\s*9\s*\+\s*7"  # 10**9+7, 10 ** 9 + 7
            r"|int\(\s*1e9\s*\)\s*\+\s*7"  # int(1e9)+7
            r"|1e9\s*\+\s*7"  # 1e9+7, also inside int(1e9+7)
            r"|1000000007)\b"
        ),
        "mod1e9p7",
    ),
)
_CODE_TOKEN = re.compile(r"(?u)\b\w\w+\b")
# Sample I/O cues.
_INTEGER = re.compile(r"-?\d+")
_DECIMAL = re.compile(r"\d\.\d")
_LETTERS_LINE = re.compile(r"[A-Za-z ]+")
_YES_NO = re.compile(r"yes|no", re.IGNORECASE)
_PLAYER_WORD = re.compile(
    r"\b(?:" + "|".join(map(re.escape, CUE_PLAYER_WORDS)) + r")\b", re.IGNORECASE
)
# A sample input read line by line: the integers of each line, or None if it holds anything else.
_IntegerLines = list[list[int] | None]


def statement_text(problems: pd.DataFrame, include_notes: bool = False) -> list[str]:
    """Return the text the statement block reads, one per problem.

    Args:
        problems: Preprocessed problems, with ``statement`` and ``prob_desc_notes`` (missing
            notes are NaN or None).
        include_notes: Append each problem's notes after a blank line, when it has any.

    Returns:
        One text per row of ``problems``, in order.
    """
    if not include_notes:
        return problems["statement"].tolist()
    return [
        f"{statement}\n\n{notes}" if isinstance(notes, str) else statement
        for statement, notes in zip(problems["statement"], problems["prob_desc_notes"], strict=True)
    ]


def normalise_statement(text: str) -> str:
    r"""Rewrite a statement's LaTeX and HTML so that one idea is one word.

    The ``TfidfVectorizer`` preprocessor of the statement block. Rules, in order:

    - HTML entities are unescaped, and the ``$$$`` math delimiters dropped.
    - No-break, thin and zero-width spaces become spaces; LaTeX thin spaces (``\,``, ``\;``,
      ``\!``) are removed, which joins digit groups (``200\,000`` gives ``200000``).
    - The two usual moduli become one word in every spelling: ``mod998244353``, and
      ``mod1e9p7`` for ``10^9+7``, ``10^{9} + 7``, ``1000000007`` and the ``109 + 7`` that the
      scraper flattened until the switch to $$$ LaTeX in 2018.
    - Powers of ten ``10^k`` and ``10^{k}`` become ``1ek``. Flattened ones (``105``) are left
      as they are: they cannot be told apart from numbers.
    - Synonyms become one word: ``le`` (``\le``, ``\leq``, ``≤``), ``ge``, ``ne``, ``dots``
      (``\dots``, ``\ldots``, ``\cdots``, ``…``, ``...``) and ``mod`` (``\bmod``, ``\pmod``,
      ``\mod``). Any other command keeps its name: ``\frac`` gives ``frac``.
    - The text is lowercased, and each run of whitespace becomes one space.

    Args:
        text: A statement, as built by ``preprocess()``.

    Returns:
        The normalised text.
    """
    text = html.unescape(text)
    for pattern, replacement in _STATEMENT_RULES:
        text = pattern.sub(replacement, text)
    return " ".join(text.lower().split())


def code_tokens(source: str) -> list[str]:
    """Split a solution into the tokens of the code block.

    The ``TfidfVectorizer`` tokenizer of the code block. The code is lowercased; comments,
    string literals and the fast-I/O template are kept. The moduli become the statement's
    words: ``mod1e9p7`` for ``10**9+7``, ``int(1e9)+7``, ``1e9+7`` and ``1000000007`` (spaces
    allowed around the operators), ``mod998244353`` for ``998244353``.

    Args:
        source: A Python 2 or 3 solution.

    Returns:
        The runs of two or more word characters, in order (sklearn's default token pattern).
    """
    source = source.lower()
    for pattern, sentinel in _CODE_MODULI:
        source = pattern.sub(sentinel, source)
    return _CODE_TOKEN.findall(source)


def sample_io_cues(problems: pd.DataFrame) -> np.ndarray:
    """Read yes/no facts from each problem's sample inputs and outputs: the cue block.

    The samples are HTML-unescaped first. Cues, in ``SAMPLE_IO_CUES`` order:

    - ``tree_edge_list``: a sample input has a line whose first integer n is at least
      ``CUE_TREE_MIN_NODES``, followed by n - 1 consecutive lines of 2 or 3 integers whose
      first two are nodes 1 to n and join all n nodes without a cycle (union-find).
    - ``graph_edge_list``: a sample input has a line "n m" (m >= 1), followed by m
      consecutive lines of 2 or 3 integers whose first two are nodes 1 to n.
    - ``decimal_output``: a sample output has a decimal number (a digit, a point, a digit).
    - ``letters_output``: every sample output line is letters and spaces, and not every line
      is YES or NO in some case.
    - ``player_output``: a sample output holds one of ``CUE_PLAYER_WORDS``, as a whole word
      in any case.

    "Followed by" means anywhere further down the same sample input, since a line of node
    values often comes between n and the edges. The edge-list cues read the first
    ``CUE_MAX_SAMPLE_LINES`` lines of each sample input. A problem without samples, or whose
    sample field is missing, has no cue.

    Args:
        problems: Preprocessed problems, with ``prob_desc_sample_inputs`` and
            ``prob_desc_sample_outputs`` (lists of strings, possibly empty; NaN or None when
            the field is missing, as ``preprocess()`` leaves it).

    Returns:
        A float64 array of 0 and 1, one row per problem and one column per cue.
    """
    rows = [
        _cues(_unescape(inputs), _unescape(outputs))
        for inputs, outputs in zip(
            problems["prob_desc_sample_inputs"], problems["prob_desc_sample_outputs"], strict=True
        )
    ]
    return np.array(rows, dtype=np.float64).reshape(len(problems), len(SAMPLE_IO_CUES))


def _cues(inputs: list[str], outputs: list[str]) -> list[bool]:
    """Return the cues of one problem, in ``SAMPLE_IO_CUES`` order."""
    return [
        _found_in_inputs(inputs, _starts_tree_edge_list),
        _found_in_inputs(inputs, _starts_graph_edge_list),
        any(_DECIMAL.search(output) for output in outputs),
        _is_letters_only(outputs),
        any(_PLAYER_WORD.search(output) for output in outputs),
    ]


def _unescape(samples: list[str] | float | None) -> list[str]:
    """Turn the HTML entities of each sample into characters (``&lt;`` into ``<``).

    A field that is not a list (NaN or None when it is missing) gives no sample.
    """
    if not isinstance(samples, list):
        return []
    return [html.unescape(sample) for sample in samples]


def _found_in_inputs(inputs: list[str], starts_here: Callable[[_IntegerLines, int], bool]) -> bool:
    """Tell whether ``starts_here(lines, start)`` holds at some line of some sample input.

    Only the first ``CUE_MAX_SAMPLE_LINES`` lines of each input are read: the search is
    quadratic in the lines read.
    """
    for sample in inputs:
        lines = [_integers(line) for line in sample.splitlines()[:CUE_MAX_SAMPLE_LINES]]
        if any(starts_here(lines, start) for start in range(len(lines))):
            return True
    return False


def _integers(line: str) -> list[int] | None:
    """Return the integers of a line made of integers only, or None."""
    words = line.split()
    if not words or not all(_INTEGER.fullmatch(word) for word in words):
        return None
    return [int(word) for word in words]


def _starts_tree_edge_list(lines: _IntegerLines, start: int) -> bool:
    """Tell whether line ``start`` begins with n, and n - 1 later lines are a tree's edges."""
    if lines[start] is None:
        return False
    n = lines[start][0]
    return n >= CUE_TREE_MIN_NODES and any(
        _is_spanning_tree(n, edges) for edges in _edge_runs(lines, start, n, n - 1)
    )


def _starts_graph_edge_list(lines: _IntegerLines, start: int) -> bool:
    """Tell whether line ``start`` is "n m", and m later lines are edges."""
    header = lines[start]
    if header is None or len(header) != 2:
        return False
    n, m = header
    return m >= 1 and any(_edge_runs(lines, start, n, m))  # each run is a non-empty list


def _edge_runs(lines: _IntegerLines, start: int, n: int, size: int) -> Iterator[_IntegerLines]:
    """Yield every run of ``size`` consecutive edges on nodes 1 to n after line ``start``."""
    for first in range(start + 1, len(lines) - size + 1):
        run = lines[first : first + size]
        if all(_is_edge(line, n) for line in run):
            yield run


def _is_edge(line: list[int] | None, n: int) -> bool:
    """Tell whether a line is 2 or 3 integers whose first two are nodes 1 to n."""
    return line is not None and len(line) in (2, 3) and 1 <= line[0] <= n and 1 <= line[1] <= n


def _is_spanning_tree(n: int, edges: _IntegerLines) -> bool:
    """Tell whether n - 1 edges join nodes 1 to n without a cycle, by union-find."""
    parent = list(range(n + 1))
    for edge in edges:
        first, second = _root(parent, edge[0]), _root(parent, edge[1])
        if first == second:
            return False
        parent[first] = second
    return True


def _root(parent: list[int], node: int) -> int:
    """Return the root of a node's set, halving the path on the way."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _is_letters_only(outputs: list[str]) -> bool:
    """Tell whether every output line is letters and spaces, and not every line YES or NO."""
    lines = [line.strip() for output in outputs for line in output.splitlines() if line.strip()]
    if not lines or not all(_LETTERS_LINE.fullmatch(line) for line in lines):
        return False
    return not all(_YES_NO.fullmatch(line) for line in lines)


def feature_blocks(params: ModelParams) -> ColumnTransformer:
    """Build the feature blocks that ``params`` keeps, as one unfitted ``ColumnTransformer``.

    Each block is named after itself; they come in ``BLOCKS`` order:

    - ``statement``: ``statement_text`` (with the notes if ``params.include_notes``), then
      TF-IDF over its words, with ``normalise_statement`` as the preprocessor. Weight 1.
    - ``code``: TF-IDF over the ``code_tokens`` of ``source_code``. Weight
      ``params.code_weight``.
    - ``cues``: ``sample_io_cues``. Weight ``params.cue_weight``.

    The other columns are dropped. The output is a sparse matrix, or a dense array when the
    cue block is the only one.

    Args:
        params: The blocks kept, their TF-IDF settings and their weights.

    Returns:
        The transformer, to fit on preprocessed problems.
    """
    transformers = {
        "statement": (_statement_pipeline(params), ["statement", "prob_desc_notes"]),
        "code": (_code_vectorizer(params), "source_code"),
        "cues": (
            FunctionTransformer(sample_io_cues),
            ["prob_desc_sample_inputs", "prob_desc_sample_outputs"],
        ),
    }
    weights = {"statement": 1.0, "code": params.code_weight, "cues": params.cue_weight}
    return ColumnTransformer(
        [(block, *transformers[block]) for block in params.blocks],
        transformer_weights={block: weights[block] for block in params.blocks},
        remainder="drop",
        sparse_threshold=1.0,
    )


def block_feature_names(blocks: ColumnTransformer, block: str) -> list[str]:
    """Return the names of a fitted block's features, in the order of its output columns.

    Args:
        blocks: Feature blocks from ``feature_blocks``, fitted.
        block: One of the blocks they keep.

    Returns:
        The statement or code vocabulary (words and n-grams), or ``SAMPLE_IO_CUES`` for the
        cues.
    """
    if block == "cues":
        return list(SAMPLE_IO_CUES)
    vectorizer = blocks.named_transformers_[block]
    if block == "statement":
        vectorizer = vectorizer["tfidf"]
    return vectorizer.get_feature_names_out().tolist()


def _statement_pipeline(params: ModelParams) -> Pipeline:
    """Return the statement block: the statement text, then its TF-IDF."""
    text = FunctionTransformer(statement_text, kw_args={"include_notes": params.include_notes})
    tfidf = TfidfVectorizer(
        preprocessor=normalise_statement,
        ngram_range=(1, params.statement_ngram_max),
        min_df=params.statement_min_df,
        max_df=params.statement_max_df,
        sublinear_tf=params.statement_sublinear_tf,
        dtype=np.float64,
    )
    return Pipeline([("text", text), ("tfidf", tfidf)])


def _code_vectorizer(params: ModelParams) -> TfidfVectorizer:
    """Return the code block: the TF-IDF of a solution's tokens."""
    return TfidfVectorizer(
        tokenizer=code_tokens,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, params.code_ngram_max),
        min_df=params.code_min_df,
        sublinear_tf=params.code_sublinear_tf,
    )
