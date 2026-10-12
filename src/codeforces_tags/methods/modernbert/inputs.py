"""The input a ModernBERT method reads: ``[CLS] statement [SEP] solution [SEP]``, cut to a budget.

When a problem does not fit, the solution is cut first: the statement says what the problem is
about, while a long solution is often a fast-I/O template followed by the algorithm. The solution
still keeps at least ``code_min_tokens`` tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def pair_budgets(
    statement_tokens: int, code_tokens: int, max_length: int, code_min_tokens: int
) -> tuple[int, int]:
    """Share the token budget between the statement and the solution.

    The statement is kept whole when it fits; the solution gets the rest, and at least
    ``code_min_tokens`` (its whole length when shorter) even when the statement is long.

    Args:
        statement_tokens: Tokens of the statement.
        code_tokens: Tokens of the solution.
        max_length: Tokens per input, the three special tokens included.
        code_min_tokens: Solution tokens kept whatever the statement's length.

    Returns:
        The tokens kept from the statement and from the solution.
    """
    room = max_length - 3
    statement = max(0, min(statement_tokens, room - min(code_tokens, code_min_tokens)))
    return statement, min(code_tokens, room - statement)


def encode_problems(
    problems: pd.DataFrame, tokenizer, max_length: int, code_min_tokens: int
) -> list[list[int]]:
    """Turn each problem into the token ids of ``[CLS] statement [SEP] solution [SEP]``.

    Args:
        problems: Preprocessed problems, with ``statement`` and ``source_code``.
        tokenizer: The pretrained model's tokenizer (a Hugging Face fast tokenizer).
        max_length: Tokens per input, the three special tokens included.
        code_min_tokens: Solution tokens kept whatever the statement's length.

    Returns:
        One list of token ids per problem, in the order of ``problems``.
    """
    statements = tokenizer(problems["statement"].tolist(), add_special_tokens=False)["input_ids"]
    codes = tokenizer(problems["source_code"].tolist(), add_special_tokens=False)["input_ids"]
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    encoded = []
    for statement, solution in zip(statements, codes, strict=True):
        kept_statement, kept_code = pair_budgets(
            len(statement), len(solution), max_length, code_min_tokens
        )
        encoded.append([cls, *statement[:kept_statement], sep, *solution[:kept_code], sep])
    return encoded


def take(input_ids: Sequence[list[int]], rows: np.ndarray) -> list[list[int]]:
    """Keep the token ids of the rows a boolean mask selects.

    Args:
        input_ids: Token ids per problem.
        rows: One boolean per problem, True to keep it.

    Returns:
        The token ids of the kept problems, in order.
    """
    return [input_ids[row] for row in np.flatnonzero(rows)]
