"""Tests for the ModernBERT methods' input: ``[CLS] statement [SEP] solution [SEP]``, cut."""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.methods.modernbert.inputs import encode_problems, pair_budgets, take
from codeforces_tags.methods.modernbert.modeling import load_tokenizer
from codeforces_tags.methods.modernbert.settings import LORA, VARIANTS
from codeforces_tags.methods.modernbert.weights import snapshot_folder


@pytest.fixture(name="tokenizer")
def tokenizer_fixture(caches):  # pylint: disable=unused-argument
    """Return the tiny snapshot's word-level tokenizer."""
    variant = VARIANTS[LORA]
    return load_tokenizer(snapshot_folder(variant.model_repo, variant.model_revision))


def ids(tokenizer, text):
    """Return the token ids of ``text``, without special tokens."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def make_problems(statements, codes, index):
    """Return preprocessed problems with these statements and solutions."""
    return pd.DataFrame(
        {"statement": statements, "source_code": codes},
        index=pd.Index(index, name="sample_id"),
    )


@pytest.mark.parametrize(
    ("statement", "code", "expected"),
    [
        pytest.param(10, 20, (10, 20), id="both fit whole"),
        pytest.param(17, 20, (17, 20), id="both fit exactly beside the three special tokens"),
        pytest.param(18, 20, (18, 19), id="one token over: the solution is cut first"),
        pytest.param(10, 100, (10, 27), id="a long solution gets what the statement leaves"),
        pytest.param(100, 50, (29, 8), id="a long statement leaves code_min_tokens to the code"),
        pytest.param(100, 5, (32, 5), id="a solution under code_min_tokens is kept whole"),
        pytest.param(0, 0, (0, 0), id="empty texts"),
    ],
)
def test_pair_budgets_share_max_length_minus_three_special_tokens(statement, code, expected):
    budgets = pair_budgets(statement, code, max_length=40, code_min_tokens=8)

    assert budgets == expected
    assert sum(budgets) <= 40 - 3


def test_encode_problems_places_cls_statement_sep_solution_sep(tokenizer):
    problems = make_problems(["a tree of nodes"], ["for i in range n"], [7])

    encoded = encode_problems(problems, tokenizer, max_length=48, code_min_tokens=8)

    statement, code = ids(tokenizer, "a tree of nodes"), ids(tokenizer, "for i in range n")
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    assert encoded == [[cls, *statement, sep, *code, sep]]
    assert tokenizer.unk_token_id not in encoded[0]
    assert len({cls, sep, *statement, *code}) == 2 + len(statement) + len(code)


def test_encode_problems_cuts_the_solution_first_and_keeps_code_min_tokens(tokenizer):
    statement_text = "a graph of vertices and edges is a path"
    code_text = "for i in range n if i print i else return"
    problems = make_problems([statement_text], [code_text], [7])

    encoded = encode_problems(problems, tokenizer, max_length=11, code_min_tokens=3)

    statement, code = ids(tokenizer, statement_text), ids(tokenizer, code_text)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    assert encoded == [[cls, *statement[:5], sep, *code[:3], sep]]
    assert len(encoded[0]) == 11


def test_encode_problems_gives_one_row_per_problem_in_the_problems_order(tokenizer):
    statements = ["a tree", "a graph", "a string"]
    codes = ["print n", "print m", "print k"]
    problems = make_problems(statements, codes, [30, 10, 20])  # out of sample id order

    encoded = encode_problems(problems, tokenizer, max_length=48, code_min_tokens=8)

    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    assert encoded == [
        [cls, *ids(tokenizer, statement), sep, *ids(tokenizer, code), sep]
        for statement, code in zip(statements, codes, strict=True)
    ]


def test_take_keeps_the_masked_rows_in_order():
    input_ids = [[1, 2], [3], [4, 5, 6], [7]]

    assert take(input_ids, np.array([False, True, True, False])) == [[3], [4, 5, 6]]
    assert take(input_ids, np.array([True, False, False, True])) == [[1, 2], [7]]
    assert take(input_ids, np.zeros(4, dtype=bool)) == []
