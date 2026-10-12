"""Tests for the Streamlit app's data: the test set, frozen rows, a typed problem, the tag table."""

import json
import math

import pandas as pd
import pytest

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import ALL_TAGS, CLI_JSON_LIST_FIELDS, FOCUS_TAGS, RAW_INPUT_COLUMNS
from codeforces_tags.evaluation.predictions import PREDICTIONS_FILE, SCORES_FILE, stack_parts
from codeforces_tags.serving.client import ServedPrediction, request_payload
from codeforces_tags.serving.problems import (
    FORM_FIELDS,
    FrozenRow,
    display_text,
    frozen_prediction,
    live_differences,
    load_test_set,
    problem_from_form,
    tag_table,
)

# The frozen scores of test problem 2 of the method "scored", and the per-tag thresholds that its
# model server answers: graphs and trees reach theirs, and no score reaches probabilities' (inf).
TEST_2_SCORES = dict(
    zip(FOCUS_TAGS, (-0.5, 0.31, -1.25, -0.75, 0.42, -0.9, -1.1, -2.0), strict=True)
)
THRESHOLDS = dict.fromkeys(FOCUS_TAGS, 0.3) | {"probabilities": math.inf}
TEST_2_TAGS = ("graphs", "trees")
# The two texts a typed problem needs.
REQUIRED_TEXTS = {"prob_desc_description": "Find $$$n$$$.", "source_code": "print(1)\n"}


def served(tags, scores=None, thresholds=None):
    """Return a model server's answer about one problem, from the method ``scored``."""
    return ServedPrediction("scored", tuple(tags), scores or {}, thresholds or {}, None, None, 0.05)


@pytest.fixture(name="assets_dir")
def assets_dir_fixture(tmp_path):
    """Return a folder of two frozen methods, in the format that ``freeze`` writes.

    ``scored`` is a score method with dev problem 1 and test problems 2 and 3. ``all_tags``
    predicts the 37 tags of the dataset, without scores (as the zero-shot methods do): dp and math
    for test problem 2.
    """
    (tmp_path / "scored").mkdir()
    (tmp_path / "all_tags").mkdir()
    scores = pd.DataFrame(
        [dict.fromkeys(FOCUS_TAGS, 0.5), TEST_2_SCORES, dict.fromkeys(FOCUS_TAGS, -1.0)],
        index=pd.Index([1, 2, 3], name="sample_id"),
    )
    tags = (scores >= pd.Series(THRESHOLDS)).astype(int)
    stack_parts(dev=tags.loc[[1]], test=tags.loc[[2, 3]]).to_csv(
        tmp_path / "scored" / PREDICTIONS_FILE
    )
    stack_parts(dev=scores.loc[[1]], test=scores.loc[[2, 3]]).to_csv(
        tmp_path / "scored" / SCORES_FILE
    )
    every_tag = pd.DataFrame(0, index=pd.Index([1, 2], name="sample_id"), columns=list(ALL_TAGS))
    every_tag.loc[2, ["dp", "math"]] = 1
    stack_parts(dev=every_tag.loc[[1]], test=every_tag.loc[[2]]).to_csv(
        tmp_path / "all_tags" / PREDICTIONS_FILE
    )
    return tmp_path


def test_load_test_set_reads_the_test_problems_with_their_focus_tags(cli_challenge, monkeypatch):
    checked = []
    monkeypatch.setattr("codeforces_tags.serving.problems.check_dataset", checked.append)

    problems, labels = load_test_set(cli_challenge["data_dir"], cli_challenge["split_path"])

    assert checked == [cli_challenge["data_dir"]]
    assert problems.index.tolist() == cli_challenge["test"]
    assert {"tags", "fold", "group", *RAW_INPUT_COLUMNS} <= set(problems.columns)
    expected = cli_challenge["labels"].loc[cli_challenge["test"]]
    pd.testing.assert_frame_equal(labels, expected, check_names=False)


def test_load_test_set_refuses_a_dataset_that_is_not_the_challenge_dataset(cli_challenge):
    with pytest.raises(CliError, match=r"cannot be read: The dataset has 16 problem files"):
        load_test_set(cli_challenge["data_dir"], cli_challenge["split_path"])


def test_load_test_set_refuses_a_missing_directory(tmp_path):
    with pytest.raises(CliError, match=r"missing cannot be read: Dataset directory not found"):
        load_test_set(tmp_path / "missing")


def test_frozen_prediction_reads_the_focus_tags_and_scores_of_a_test_row(assets_dir):
    frozen = frozen_prediction("scored", 2, assets_dir)

    tags = {tag: int(tag in TEST_2_TAGS) for tag in FOCUS_TAGS}
    assert frozen == FrozenRow(tags=tags, scores=TEST_2_SCORES)


def test_frozen_prediction_keeps_only_the_focus_tags_of_a_method_without_scores(assets_dir):
    frozen = frozen_prediction("all_tags", 2, assets_dir)

    assert frozen == FrozenRow(tags={tag: int(tag == "math") for tag in FOCUS_TAGS}, scores=None)


@pytest.mark.parametrize(
    ("name", "sample_id"), [("scored", 1), ("scored", 99), ("never_frozen", 2)]
)
def test_frozen_prediction_is_none_without_a_frozen_test_row(assets_dir, name, sample_id):
    # Problem 1 is a dev problem; problem 99 is no frozen problem; never_frozen has no folder.
    assert frozen_prediction(name, sample_id, assets_dir) is None


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("Find $$$n$$$ such that $$$n^2 = m$$$.", "Find $n$ such that $n^2 = m$."),
        ("$$$$$$\\sum_{i} a_i$$$$$$", "$$\\sum_{i} a_i$$"),
        ("No math at all.", "No math at all."),
        (None, ""),
        (math.nan, ""),
    ],
)
def test_display_text_writes_codeforces_math_as_markdown_reads_it(text, shown):
    assert display_text(text) == shown


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        # Inside a formula: KaTeX would show an entity as an error.
        ("$$$a_i &lt; a_j$$$", "$a_i < a_j$"),
        ("Find $$$x &gt; y$$$ and $$$p &amp; q$$$.", "Find $x > y$ and $p & q$."),
        ("$$$$$$a &lt; b &amp;&amp; c &gt; d$$$$$$", "$$a < b && c > d$$"),
        # Outside a formula.
        ("print &quot;Yes&quot; if a &lt; b &amp;&amp; c &gt; d", 'print "Yes" if a < b && c > d'),
        ("It&#39;s &#x3C;ok&#x3E;.", "It's <ok>."),
        # Decoded once: a double-encoded entity shows as the entity it spells.
        ("&amp;lt;", "&lt;"),
        # Delimiters first: a dollar that an entity decodes to never joins a delimiter.
        ("&#36;&#36;&#36;x", "$$$x"),
        # No entity, nothing to decode: a bare ampersand or angle bracket stays.
        ("Tom & Jerry: $$$1 < 2$$$ and 3 > 2, a&b.", "Tom & Jerry: $1 < 2$ and 3 > 2, a&b."),
    ],
)
def test_display_text_decodes_the_html_entities_of_the_scraped_page(text, shown):
    assert display_text(text) == shown


def test_the_form_holds_every_text_field_and_the_sample_pairs_hold_the_others():
    assert list(FORM_FIELDS) == [
        column for column in RAW_INPUT_COLUMNS if column not in CLI_JSON_LIST_FIELDS
    ]
    assert list(FORM_FIELDS.values()) == [
        "description",
        "input specification",
        "output specification",
        "notes",
        "solution",
    ]


def test_problem_from_form_keeps_typed_text_verbatim():
    texts = {
        "prob_desc_description": "  Найдите $$$\\sum_{i=1}^{n} a_i$$$, ünïcödé 数学\r\nAgain.\t",
        "prob_desc_input_spec": "The first line holds $$$n$$$ ($$$1 \\le n \\le 10^5$$$).\r\n",
        "prob_desc_output_spec": "Print «one» integer.",
        "prob_desc_notes": "\r\nNote: $$$$$$x^2$$$$$$",
        "source_code": "s = input()  # é\r\nprint(s)\r\n",
    }
    pairs = [("1\r\n4 é", "2\r\n"), ("  3", " 7 ")]

    problem = problem_from_form(texts, pairs, sample_id=5)

    [row] = problem.to_dict("records")
    assert {field: row[field] for field in texts} == texts
    assert row["prob_desc_sample_inputs"] == ["1\r\n4 é", "  3"]
    assert row["prob_desc_sample_outputs"] == ["2\r\n", " 7 "]
    [sent] = request_payload(problem)["dataframe_split"]["data"]
    assert dict(zip(RAW_INPUT_COLUMNS, sent, strict=True)) == texts | {
        "prob_desc_sample_inputs": json.dumps(["1\r\n4 é", "  3"]),
        "prob_desc_sample_outputs": json.dumps(["2\r\n", " 7 "]),
    }
    assert "$$$" in sent[0] and display_text(sent[0]).count("$$$") == 0


def test_problem_from_form_keeps_html_entities_as_typed():
    # Only display_text decodes entities, for display: the text sent is the text typed.
    texts = REQUIRED_TEXTS | {"prob_desc_description": "$$$a &lt; b$$$ &amp; $$$c &gt; d$$$"}

    problem = problem_from_form(texts, [])

    assert problem.loc[0, "prob_desc_description"] == "$$$a &lt; b$$$ &amp; $$$c &gt; d$$$"
    [sent] = request_payload(problem)["dataframe_split"]["data"]
    assert sent[0] == "$$$a &lt; b$$$ &amp; $$$c &gt; d$$$"
    assert display_text(sent[0]) == "$a < b$ & $c > d$"


def test_problem_from_form_keeps_half_blank_sample_pairs_and_drops_blank_rows():
    # Rows of Streamlit's data editor: a new row holds None, a cleared cell None, NaN or "".
    pairs = [
        ("1 2\n", "3"),
        ("4", ""),
        (None, "5"),
        ("", ""),
        (None, None),
        (" \n", math.nan),
        ("6", math.nan),
    ]

    problem = problem_from_form(REQUIRED_TEXTS, pairs)

    assert problem.loc[0, "prob_desc_sample_inputs"] == ["1 2\n", "4", "", "6"]
    assert problem.loc[0, "prob_desc_sample_outputs"] == ["3", "", "5", ""]


def test_problem_from_form_sends_a_blank_optional_text_as_null_and_no_pair_as_an_empty_list():
    texts = REQUIRED_TEXTS | {"prob_desc_input_spec": " \r\n", "prob_desc_notes": ""}

    problem = problem_from_form(texts, [], sample_id=3)

    assert list(problem.columns) == list(RAW_INPUT_COLUMNS)
    assert (problem.index.tolist(), problem.index.name) == ([3], "sample_id")
    assert problem.loc[3].tolist() == ["Find $$$n$$$.", None, None, None, [], [], "print(1)\n"]
    assert request_payload(problem)["dataframe_split"]["data"] == [
        ["Find $$$n$$$.", None, None, None, "[]", "[]", "print(1)\n"]
    ]


@pytest.mark.parametrize(
    ("texts", "message"),
    [
        (
            {"prob_desc_description": " \r\n", "source_code": "print(1)"},
            r"^the description is empty: a problem needs a description and a solution\.$",
        ),
        ({"prob_desc_description": "Find n."}, r"^the solution is empty: "),
        ({}, r"^the description is empty and the solution is empty: "),
    ],
)
def test_problem_from_form_refuses_a_blank_description_or_solution(texts, message):
    with pytest.raises(ValueError, match=message):
        problem_from_form(texts, [("1", "2")])


def test_problem_from_form_refuses_a_field_that_the_form_does_not_have():
    texts = REQUIRED_TEXTS | {"prob_desc_sample_inputs": '["1"]'}

    with pytest.raises(ValueError, match=r"the form has no field \['prob_desc_sample_inputs'\]"):
        problem_from_form(texts, [])


def test_tag_table_lays_out_the_answer_beside_the_true_tags_and_the_frozen_row(assets_dir):
    frozen = frozen_prediction("scored", 2, assets_dir)
    truth = {tag: int(tag in ("graphs", "games")) for tag in FOCUS_TAGS}
    prediction = served(("graphs", "games"), TEST_2_SCORES, THRESHOLDS)

    table = tag_table(prediction, truth, frozen)

    assert (table.index.tolist(), table.index.name) == (list(FOCUS_TAGS), "tag")
    assert list(table.columns) == ["predicted", "score", "threshold", "true", "frozen", "agrees"]
    assert table["predicted"].tolist() == [0, 1, 0, 0, 0, 0, 1, 0]
    assert table["score"].tolist() == list(TEST_2_SCORES.values())
    assert table["threshold"].tolist() == list(THRESHOLDS.values())
    assert table["true"].tolist() == [0, 1, 0, 0, 0, 0, 1, 0]
    assert table["frozen"].tolist() == [0, 1, 0, 0, 1, 0, 0, 0]
    assert table["agrees"].tolist() == [True, True, True, True, False, True, False, True]


def test_tag_table_of_a_typed_problem_leaves_the_truth_the_frozen_row_and_agreement_empty():
    table = tag_table(served(("math",)))

    assert table["predicted"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert table[["score", "threshold", "true", "frozen", "agrees"]].isna().all().all()
    assert [str(dtype) for dtype in table.dtypes] == [
        "int64",
        "float64",
        "float64",
        "Int64",
        "Int64",
        "boolean",
    ]


def test_live_differences_is_empty_for_the_frozen_row_itself(assets_dir):
    frozen = frozen_prediction("scored", 2, assets_dir)

    differences = live_differences(
        served(TEST_2_TAGS, TEST_2_SCORES, THRESHOLDS), frozen, score_atol=0.0
    )

    assert differences == []


@pytest.mark.parametrize(
    ("graphs_score", "score_atol", "expected"),
    [
        (0.31 + 1e-6, 0.0, ["graphs: score 0.310001, frozen 0.31: further apart than 1e-09"]),
        (0.31 + 1e-6, 1e-5, []),
        (math.nan, 1.0, ["graphs: score nan, frozen 0.31: further apart than 1"]),
    ],
)
def test_live_differences_compares_scores_within_score_atol_or_the_export_tolerance(
    assets_dir, graphs_score, score_atol, expected
):
    frozen = frozen_prediction("scored", 2, assets_dir)
    prediction = served(TEST_2_TAGS, TEST_2_SCORES | {"graphs": graphs_score}, THRESHOLDS)

    assert live_differences(prediction, frozen, score_atol=score_atol) == expected


@pytest.mark.parametrize(
    ("score_atol", "expected"),
    [
        (0.02, []),
        (
            0.005,
            [
                "graphs: score 0.299, frozen 0.31: further apart than 0.005",
                "graphs: the model server predicts 0, frozen 1, and its frozen score 0.31 lies "
                "further than 0.005 from the per-tag threshold 0.3",
            ],
        ),
    ],
)
def test_live_differences_allows_a_flip_only_within_score_atol_of_the_per_tag_threshold(
    assets_dir, score_atol, expected
):
    # The frozen score of graphs lies 0.01 above its per-tag threshold: a score that drifted to
    # 0.299 flips the tag, which a method of score_atol 0.02 allows and one of 0.005 does not.
    frozen = frozen_prediction("scored", 2, assets_dir)
    prediction = served(("trees",), TEST_2_SCORES | {"graphs": 0.299}, THRESHOLDS)

    assert live_differences(prediction, frozen, score_atol=score_atol) == expected


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (("math",), []),
        (("math", "games"), ["games: the model server predicts 1, frozen 0"]),
        ((), ["math: the model server predicts 0, frozen 1"]),
    ],
)
def test_live_differences_compares_the_tags_of_a_method_without_scores_exactly(
    assets_dir, tags, expected
):
    frozen = frozen_prediction("all_tags", 2, assets_dir)

    assert live_differences(served(tags), frozen, score_atol=0.0) == expected


def test_live_differences_reports_an_answer_without_the_scores_of_a_score_method(assets_dir):
    frozen = frozen_prediction("scored", 2, assets_dir)

    differences = live_differences(served(TEST_2_TAGS, TEST_2_SCORES), frozen, score_atol=0.0)

    assert differences == [
        "the answer holds no score or per-tag threshold of math, graphs, strings, number theory, "
        "trees, geometry, games, probabilities"
    ]


def test_importing_the_app_data_loads_no_heavy_library(heavy_after_import):
    assert heavy_after_import("codeforces_tags.serving.problems") == []
