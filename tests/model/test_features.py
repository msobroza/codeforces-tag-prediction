"""Tests for the model's input features."""

import time
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import skops.io

from codeforces_tags.config import CUE_MAX_SAMPLE_LINES
from codeforces_tags.model import features
from codeforces_tags.model.features import (
    SAMPLE_IO_CUES,
    STORED_FUNCTIONS,
    block_feature_names,
    code_tokens,
    feature_blocks,
    normalise_statement,
    sample_io_cues,
    statement_text,
)
from codeforces_tags.model.params import BLOCKS


def test_statement_text_is_the_statement_alone_by_default():
    problems = pd.DataFrame(
        {"statement": ["Find n.", "Sort a."], "prob_desc_notes": ["Hint.", None]}
    )

    assert statement_text(problems) == ["Find n.", "Sort a."]


def test_statement_text_appends_the_notes_after_a_blank_line_when_asked():
    problems = pd.DataFrame(
        {
            "statement": ["Find n.", "Sort a.", "Print k."],
            "prob_desc_notes": ["Hint.", None, np.nan],
        }
    )

    assert statement_text(problems, include_notes=True) == [
        "Find n.\n\nHint.",
        "Sort a.",
        "Print k.",
    ]


def test_normalise_statement_unescapes_html_entities():
    assert normalise_statement("a &lt; b &amp;&amp; c &gt; d") == "a < b && c > d"


def test_normalise_statement_drops_the_math_delimiters():
    assert normalise_statement("Given $$$n$$$ points and $$$k$$$ lines.") == (
        "given n points and k lines."
    )


def test_normalise_statement_turns_unicode_spaces_into_spaces():
    no_break, thin, zero_width, narrow_no_break = "\u00a0", "\u2009", "\u200b", "\u202f"
    text = f"a{no_break}b{thin}c{zero_width}d{narrow_no_break}e"

    assert normalise_statement(text) == "a b c d e"


def test_normalise_statement_removes_latex_thin_spaces_and_joins_digit_groups():
    assert normalise_statement(r"$$$a\,b\;c\!d$$$") == "abcd"
    assert normalise_statement(r"$$$1 \le n \le 200\,000$$$") == "1 le n le 200000"


@pytest.mark.parametrize(
    ("text", "word"),
    [
        (r"\le \leq \leqslant ≤", "le"),
        (r"\ge \geq \geqslant ≥", "ge"),
        (r"\ne \neq ≠", "ne"),
        (r"\dots \ldots \cdots … ... ....", "dots"),
        (r"\bmod \pmod \mod", "mod"),
    ],
)
def test_normalise_statement_merges_synonyms_into_one_word(text, word):
    assert set(normalise_statement(text).split()) == {word}


def test_normalise_statement_keeps_synonyms_apart_from_their_neighbours():
    assert normalise_statement(r"$$$1\le n\le m$$$") == "1 le n le m"


def test_normalise_statement_does_not_merge_longer_commands_with_a_synonym_prefix():
    assert normalise_statement(r"\left \neg \gets \cdot \models") == "left neg gets cdot models"


@pytest.mark.parametrize(
    "text",
    ["998244353", r"998\,244\,353", "998,244,353", "998\u2009244\u2009353"],
)
def test_normalise_statement_replaces_every_spelling_of_998244353(text):
    assert normalise_statement(f"modulo $$${text}$$$.") == "modulo mod998244353 ."


@pytest.mark.parametrize(
    "text",
    [
        "10^9+7",
        "10^9 + 7",
        "10^{9}+7",
        "10^{9} + 7",
        "1000000007",
        r"1\,000\,000\,007",
        "1\u2009000\u2009000\u2009007",
        "109\u2009+\u20097",  # flattened by the scraper until the switch to $$$ LaTeX in 2018
    ],
)
def test_normalise_statement_replaces_every_spelling_of_1e9_plus_7(text):
    assert normalise_statement(f"modulo {text}.") == "modulo mod1e9p7."


def test_normalise_statement_leaves_other_numbers_near_a_modulus_alone():
    assert normalise_statement("2109 + 7, 109 + 75, 10^9+9") == "2109 + 7, 109 + 75, 1e9+9"


def test_normalise_statement_writes_powers_of_ten_as_1ek():
    assert (
        normalise_statement(r"$$$n \le 10^5$$$, $$$a_i \le 10^{18}$$$") == "n le 1e5 , a_i le 1e18"
    )
    assert normalise_statement(r"$$$2 \cdot 10^5$$$") == "2 cdot 1e5"


def test_normalise_statement_does_not_rebuild_flattened_powers():
    assert normalise_statement("1 ≤ n ≤ 105") == "1 le n le 105"


def test_normalise_statement_keeps_the_name_of_other_commands():
    assert normalise_statement(r"$$$\frac{a}{b}$$$ \Delta \texttt{YES}") == (
        "frac {a}{b} delta texttt {yes}"
    )


def test_normalise_statement_lowercases_and_collapses_whitespace():
    assert normalise_statement("Alice  WINS\n\nthe Game") == "alice wins the game"


def test_code_tokens_lowercases_and_drops_one_character_tokens():
    assert code_tokens("def Solve(N, nums):\n    return SUM(nums) + 1") == [
        "def",
        "solve",
        "nums",
        "return",
        "sum",
        "nums",
    ]


def test_code_tokens_keeps_comments_and_string_literals():
    assert code_tokens("x = 'YES'  # greedy pick") == ["yes", "greedy", "pick"]


@pytest.mark.parametrize(
    ("source", "tokens"),
    [
        ("MOD = 10**9+7", ["mod", "mod1e9p7"]),
        ("MOD = 10 ** 9 + 7", ["mod", "mod1e9p7"]),
        ("m = 1000000007", ["mod1e9p7"]),
        ("m = int(1e9)+7", ["mod1e9p7"]),
        ("m = int(1e9 + 7)", ["int", "mod1e9p7"]),
        ("m = 1e9+7", ["mod1e9p7"]),
        ("p = 998244353", ["mod998244353"]),
    ],
)
def test_code_tokens_replaces_every_spelling_of_a_modulus(source, tokens):
    assert code_tokens(source) == tokens


def test_code_tokens_leaves_other_numbers_near_a_modulus_alone():
    assert code_tokens("m = 10**9+9; k = 10**9+77; q = 1000000009") == [
        "10",
        "10",
        "77",
        "1000000009",
    ]


def cues(inputs=(), outputs=()):
    """Return the sample I/O cues of one problem, as ``{name: 0.0 or 1.0}``."""
    problems = pd.DataFrame(
        {"prob_desc_sample_inputs": [list(inputs)], "prob_desc_sample_outputs": [list(outputs)]}
    )
    return dict(zip(SAMPLE_IO_CUES, sample_io_cues(problems)[0].tolist(), strict=True))


def test_sample_io_cues_returns_one_float_row_per_problem_in_cue_order():
    problems = pd.DataFrame(
        {
            "prob_desc_sample_inputs": [["4\n1 2\n1 3\n3 4"], ["1"], []],
            "prob_desc_sample_outputs": [["0.5"], ["Alice"], []],
        },
        index=pd.Index([7, 3, 5], name="sample_id"),
    )

    values = sample_io_cues(problems)

    assert values.dtype == np.float64
    assert values.tolist() == [[1, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]
    found = [
        [name for name, value in zip(SAMPLE_IO_CUES, row, strict=True) if value] for row in values
    ]
    assert found == [["tree_edge_list", "decimal_output"], ["letters_output", "player_output"], []]


def test_sample_io_cues_of_no_problem_is_an_empty_matrix():
    problems = pd.DataFrame({"prob_desc_sample_inputs": [], "prob_desc_sample_outputs": []})

    assert sample_io_cues(problems).shape == (0, len(SAMPLE_IO_CUES))


def test_a_problem_without_samples_has_no_cue():
    assert cues() == dict.fromkeys(SAMPLE_IO_CUES, 0.0)


def test_a_problem_whose_sample_fields_are_missing_has_no_cue():
    problems = pd.DataFrame(
        {
            "prob_desc_sample_inputs": [None, np.nan],  # missing, as preprocess() leaves them
            "prob_desc_sample_outputs": [np.nan, ["Alice"]],
        }
    )

    assert sample_io_cues(problems).tolist() == [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]


def test_sample_io_cues_read_the_samples_html_unescaped():
    found = cues(inputs=["4&#10;1 2&#10;1 3&#10;3 4"], outputs=["Alice&#32;wins", "0&#46;5"])

    assert found["tree_edge_list"] == 1
    assert found["decimal_output"] == 1
    assert cues(outputs=["Alice&#32;wins"])["letters_output"] == 1


@pytest.mark.parametrize(
    "sample_input",
    [
        "4\n1 2\n1 3\n3 4",
        "2\n4 7\n1 2\n2 3\n2 4\n1\n5",  # after other lines, with a second number on n's line
        "4\n5 1 3 2\n1 2\n1 3\n3 4",  # the nodes' values between n and the edges
        "4\n1 2 5\n2 3 100\n3 4 7",  # weighted: the third number is free
    ],
    ids=["plain", "inside a longer input", "after the node values", "weighted"],
)
def test_tree_edge_list_finds_the_edges_of_a_tree(sample_input):
    assert cues(inputs=["1", sample_input])["tree_edge_list"] == 1


@pytest.mark.parametrize(
    "sample_input",
    [
        "4\n1 2\n2 3\n3 1",  # a cycle, node 4 left out
        "4\n1 2\n2 3\n3 5",  # node 5 is out of range
        "4\n1 2\n2 3",  # one edge missing
        "4\n1 2\n2 3\n3 4 1 1",  # four numbers on a line
        "4\n1 2\n2 x\n3 4",  # not only numbers
    ],
    ids=["cycle", "out of range", "too few lines", "too many numbers", "text"],
)
def test_tree_edge_list_rejects_what_is_not_a_tree(sample_input):
    assert cues(inputs=[sample_input])["tree_edge_list"] == 0


def test_tree_edge_list_skips_trees_smaller_than_cue_tree_min_nodes(monkeypatch):
    multi_test_pairs = "3\n3 4\n3 2\n3 1"  # three test cases, one pair each

    assert cues(inputs=[multi_test_pairs])["tree_edge_list"] == 0
    monkeypatch.setattr(features, "CUE_TREE_MIN_NODES", 3)
    assert cues(inputs=[multi_test_pairs])["tree_edge_list"] == 1


@pytest.mark.parametrize(
    "sample_input",
    [
        "3 2\n1 2\n2 3",
        "3 2\n7 7 7\n1 2\n2 3",  # the nodes' values between "n m" and the edges
        "1\n3 3\n1 2 4\n2 3 1\n1 3 9\n2 3",  # weighted, inside a longer input
    ],
    ids=["plain", "after the node values", "weighted, inside a longer input"],
)
def test_graph_edge_list_finds_n_m_and_m_edges(sample_input):
    assert cues(inputs=[sample_input])["graph_edge_list"] == 1


@pytest.mark.parametrize(
    "sample_input",
    [
        "3 2\n1 2\n2 4",  # node 4 is out of range
        "3 3\n1 2\n2 3",  # one edge missing
        "3 0\n5",  # no edge
        "3 2 1\n1 2\n2 3",  # three numbers on the header line
        "3 2\n1 2\nx y",  # not only numbers
    ],
    ids=["out of range", "too few lines", "no edge", "not n m", "text"],
)
def test_graph_edge_list_rejects_what_is_not_an_edge_list(sample_input):
    assert cues(inputs=[sample_input])["graph_edge_list"] == 0


@pytest.mark.parametrize(
    ("cue", "edge_list"),
    [("tree_edge_list", ["4", "1 2", "1 3", "3 4"]), ("graph_edge_list", ["3 2", "1 2", "2 3"])],
)
def test_edge_list_cues_read_the_first_cue_max_sample_lines_lines_of_an_input(cue, edge_list):
    ending_on_the_last_line_read = ["x"] * (CUE_MAX_SAMPLE_LINES - len(edge_list)) + edge_list
    one_line_further = ["x", *ending_on_the_last_line_read]

    assert cues(inputs=["\n".join(ending_on_the_last_line_read)])[cue] == 1
    assert cues(inputs=["\n".join(one_line_further)])[cue] == 0


def test_sample_io_cues_of_a_long_sample_input_take_under_a_second():
    long_input = "\n".join(["5 3"] * 5000)  # each line could start an edge list

    start = time.perf_counter()
    cues(inputs=[long_input])

    assert time.perf_counter() - start < 1.0


def test_decimal_output_needs_a_digit_on_both_sides_of_a_point():
    assert cues(outputs=["1\n0.500000"])["decimal_output"] == 1
    assert cues(outputs=["35", "1. 5", "x.5"])["decimal_output"] == 0


def test_letters_output_needs_every_line_to_be_letters_and_spaces():
    assert cues(outputs=["Alice wins\n", "abc"])["letters_output"] == 1
    assert cues(outputs=["YES\nabc"])["letters_output"] == 1  # not only YES/NO
    assert cues(outputs=["abc", "3"])["letters_output"] == 0
    assert cues(outputs=["ab-c"])["letters_output"] == 0
    assert cues(outputs=["\n"])["letters_output"] == 0  # no line at all


def test_letters_output_ignores_outputs_that_are_only_yes_or_no():
    assert cues(outputs=["YES\nno", "Yes"])["letters_output"] == 0


def test_player_output_needs_a_whole_player_word_in_any_case():
    assert cues(outputs=["Alice"])["player_output"] == 1
    assert cues(outputs=["3", "SECOND"])["player_output"] == 1
    assert cues(outputs=["The winner is 3"])["player_output"] == 1
    assert cues(outputs=["Alicea", "Bobby 2", "firstly"])["player_output"] == 0


def test_feature_blocks_keep_the_chosen_blocks_named_after_them_in_blocks_order(make_params):
    transformer = feature_blocks(make_params(blocks=("cues", "statement"), cue_weight=0.3))

    assert [name for name, _, _ in transformer.transformers] == ["statement", "cues"]
    assert transformer.transformer_weights == {"statement": 1.0, "cues": 0.3}


def test_feature_blocks_read_the_columns_of_their_input(make_params):
    transformer = feature_blocks(make_params())

    assert [(name, columns) for name, _, columns in transformer.transformers] == [
        ("statement", ["statement", "prob_desc_notes"]),
        ("code", "source_code"),
        ("cues", ["prob_desc_sample_inputs", "prob_desc_sample_outputs"]),
    ]


def test_feature_blocks_pass_each_setting_to_its_block(make_params):
    params = make_params(
        include_notes=True,
        statement_ngram_max=2,
        statement_min_df=3,
        statement_max_df=0.8,
        statement_sublinear_tf=True,
        code_ngram_max=3,
        code_min_df=4,
        code_sublinear_tf=True,
    )
    expected = {
        "statement__text__func": statement_text,
        "statement__text__kw_args": {"include_notes": True},
        "statement__tfidf__preprocessor": normalise_statement,
        "statement__tfidf__ngram_range": (1, 2),
        "statement__tfidf__min_df": 3,
        "statement__tfidf__max_df": 0.8,
        "statement__tfidf__sublinear_tf": True,
        "code__tokenizer": code_tokens,
        "code__token_pattern": None,
        "code__lowercase": False,
        "code__ngram_range": (1, 3),
        "code__min_df": 4,
        "code__sublinear_tf": True,
        "cues__func": sample_io_cues,
        "remainder": "drop",
        "sparse_threshold": 1.0,
    }

    settings = feature_blocks(params).get_params()

    assert {name: settings[name] for name in expected} == expected


@pytest.mark.parametrize("include_notes", [False, True])
def test_feature_blocks_read_the_notes_only_with_include_notes(
    problems, make_params, include_notes
):
    transformer = feature_blocks(make_params(include_notes=include_notes)).fit(problems)

    vocabulary = transformer.named_transformers_["statement"]["tfidf"].vocabulary_
    assert ("explained" in vocabulary) == include_notes  # a word of the notes alone


@pytest.mark.parametrize(
    "blocks",
    [blocks for size in range(1, len(BLOCKS) + 1) for blocks in combinations(BLOCKS, size)],
    ids=" + ".join,
)
def test_feature_blocks_fit_any_non_empty_set_of_blocks(problems, make_params, blocks):
    transformer = feature_blocks(make_params(blocks=blocks))

    values = transformer.fit_transform(problems)

    columns = transformer.output_indices_
    assert [name for name in columns if name != "remainder"] == list(blocks)
    assert all(columns[block].stop > columns[block].start for block in blocks)
    assert values.shape == (len(problems), columns[blocks[-1]].stop)


def test_feature_blocks_weigh_the_code_and_cue_blocks_against_the_statement_block(
    problems, make_params
):
    unweighted = feature_blocks(make_params(code_weight=1.0, cue_weight=1.0))
    weighted = feature_blocks(make_params(code_weight=0.4, cue_weight=0.25))

    plain = unweighted.fit_transform(problems).toarray()
    values = weighted.fit_transform(problems).toarray()

    columns = weighted.output_indices_
    np.testing.assert_array_equal(values[:, columns["statement"]], plain[:, columns["statement"]])
    np.testing.assert_allclose(values[:, columns["code"]], 0.4 * plain[:, columns["code"]])
    np.testing.assert_array_equal(values[:, columns["cues"]], 0.25 * sample_io_cues(problems))


def test_feature_blocks_fitted_on_some_folds_transform_the_held_out_fold(
    problems, folds, make_params
):
    transformer = feature_blocks(make_params())

    fitted = transformer.fit_transform(problems[folds != 0])
    held_out = transformer.transform(problems[folds == 0])

    assert held_out.shape == ((folds == 0).sum(), fitted.shape[1])


def test_block_feature_names_name_each_output_column_of_its_block(problems, make_params):
    transformer = feature_blocks(make_params()).fit(problems)

    names = {block: block_feature_names(transformer, block) for block in BLOCKS}

    columns = transformer.output_indices_
    assert {block: len(names[block]) for block in BLOCKS} == {
        block: columns[block].stop - columns[block].start for block in BLOCKS
    }
    vocabulary = transformer.named_transformers_["statement"]["tfidf"].vocabulary_
    assert names["statement"] == sorted(vocabulary, key=vocabulary.get)
    assert "adj" in names["code"]  # a token of the graph solutions
    assert names["cues"] == list(SAMPLE_IO_CUES)


@pytest.fixture(name="saved_blocks")
def saved_blocks_fixture(problems, make_params):
    """Return the three feature blocks fitted on ``problems``, and their skops bytes."""
    fitted = feature_blocks(make_params(include_notes=True)).fit(problems)
    return fitted, skops.io.dumps(fitted)


def test_stored_functions_are_the_types_skops_asks_to_trust_in_saved_feature_blocks(
    saved_blocks,
):
    _, data = saved_blocks

    assert tuple(skops.io.get_untrusted_types(data=data)) == STORED_FUNCTIONS


def test_saved_feature_blocks_load_with_the_stored_functions_and_transform_alike(
    problems, saved_blocks
):
    fitted, data = saved_blocks

    loaded = skops.io.loads(data, trusted=list(STORED_FUNCTIONS))

    np.testing.assert_array_equal(
        loaded.transform(problems).toarray(), fitted.transform(problems).toarray()
    )
