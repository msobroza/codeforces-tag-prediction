"""Tests for the tables of the zero-shot LLM methods' notebook."""

import subprocess
import sys

import pandas as pd
import pytest

from codeforces_tags.analysis.llm_methods import (
    ALL_TAG_COLUMNS,
    ANSWER_HEALTH_COLUMNS,
    COST_COLUMNS,
    TAGS_PER_ANSWER_COLUMNS,
    all_tag_scores,
    answer_health,
    cost_per_thousand,
    disagreements,
    tags_per_answer,
)
from codeforces_tags.config import ALL_TAGS
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.evaluation.predictions import FROZEN_INDEX
from codeforces_tags.llm.answers import answer_summary, per_problem_frame


def usage(input_tokens, output_tokens, cached_tokens=0):
    """Return the token counts of an answer, with no reasoning tokens."""
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
    }


@pytest.fixture(name="dev_answers")
def dev_answers_fixture(make_answer):
    """Return five dev answers of a text method.

    They name two tags, name a tag and two unknown pieces, name nothing, are cut off at the token
    cap, and are refused by the API (a 400 that costs nothing).
    """
    return [
        make_answer(sample_id=1, output_text="dp, math", usage=usage(1000, 100, cached_tokens=200)),
        make_answer(sample_id=2, output_text="dp, foo, bar", usage=usage(1000, 10)),
        make_answer(sample_id=3, output_text="", usage=usage(500, 0)),
        make_answer(sample_id=4, output_text="dp", status="incomplete", usage=usage(500, 512)),
        make_answer(
            sample_id=5,
            output_text=None,
            status="api_error:400",
            served_model=None,
            usage=usage(0, 0),
        ),
    ]


@pytest.fixture(name="records")
def records_fixture(placeholder_record, make_answer, dev_answers):
    """Return the run records of a text and a json method, each with its dev and test answers."""
    text_test = [make_answer(sample_id=6, output_text="dp")]
    json_dev = [
        make_answer(sample_id=1, output_text='{"tags": ["dp", "math"]}'),
        make_answer(sample_id=2, output_text='{"tags": ["dp"]}'),
    ]
    json_test = [make_answer(sample_id=6, output_text='{"tags": []}')]
    return {
        "code2tag-text": placeholder_record(
            "code2tag-text",
            method_entries={
                "answers": {
                    "dev": answer_summary(dev_answers, "text"),
                    "test": answer_summary(text_test, "text"),
                }
            },
        ),
        "code2tag-json": placeholder_record(
            "code2tag-json",
            method_entries={
                "answers": {
                    "dev": answer_summary(json_dev, "json"),
                    "test": answer_summary(json_test, "json"),
                }
            },
        ),
    }


def test_answer_health_has_a_row_per_method_and_part_in_the_order_given(records):
    health = answer_health(records)

    assert health.index.names == ["method", "part"]
    assert health.index.tolist() == [
        ("code2tag-text", "dev"),
        ("code2tag-text", "test"),
        ("code2tag-json", "dev"),
        ("code2tag-json", "test"),
    ]
    assert list(health.columns) == list(ANSWER_HEALTH_COLUMNS)
    assert ", ".join(health.columns) == (
        "answers, completed, failed, unknown_pieces, empty_answers, served_models, "
        "input_tokens, cached_tokens, output_tokens, reasoning_tokens, cost_usd"
    )


def test_answer_health_counts_the_statuses_failures_and_tokens_of_the_answers(records):
    dev = answer_health(records).loc[("code2tag-text", "dev")]

    assert dev["answers"] == 5
    assert dev["completed"] == 3  # the cut-off and the refused answer did not complete
    assert dev["failed"] == 2  # both score as the empty set
    assert dev["unknown_pieces"] == 2  # "foo" and "bar"
    assert dev["empty_answers"] == 1  # completed, naming no tag
    assert dev["served_models"] == {"gpt-6-luna": 4}  # the refused answer was served by none
    assert dev[["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"]].tolist() == [
        3000,
        200,
        622,
        0,
    ]


def test_answer_health_totals_the_cost_of_the_answers(records):
    # Per answer, at 0.10 / 0.01 / 0.50 USD per million uncached / cached input / output tokens:
    # 800 uncached + 200 cached + 100 output = 1.32e-4; 1,000 + 10 = 1.05e-4; 500 + 0 = 5e-5;
    # 500 + 512 = 3.06e-4; and nothing for the refused answer.
    health = answer_health(records)

    assert health.at[("code2tag-text", "dev"), "cost_usd"] == pytest.approx(5.93e-4)
    assert health.at[("code2tag-text", "test"), "cost_usd"] == pytest.approx(1.25e-5)


def test_answer_health_counts_no_completed_answer_in_a_part_whose_answers_all_failed(
    placeholder_record, dev_answers
):
    refused = dev_answers[-1:]
    records = {
        "code2tag-text": placeholder_record(
            "code2tag-text", method_entries={"answers": {"dev": answer_summary(refused, "text")}}
        )
    }

    dev = answer_health(records).loc[("code2tag-text", "dev")]

    assert (dev["answers"], dev["completed"], dev["failed"]) == (1, 0, 1)


def test_answer_health_refuses_the_run_record_of_a_method_that_sends_no_request(
    records, placeholder_record
):
    records["label_prior"] = placeholder_record("label_prior")

    with pytest.raises(ValueError, match="run record of label_prior has no answers"):
        answer_health(records)


def test_tags_per_answer_counts_the_tags_of_each_method_and_of_the_true_labels():
    index = pd.Index([10, 11, 12, 13], name="sample_id")
    columns = ["math", "dp", "greedy"]
    truth = pd.DataFrame([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]], index, columns)
    predictions = {
        "one each": pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], index, columns),
        "mostly none": pd.DataFrame([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]], index, columns),
    }

    counts = tags_per_answer(predictions, truth)

    assert counts.index.tolist() == ["one each", "mostly none", "true labels"]
    assert counts.index.name == "method"
    assert list(counts.columns) == list(TAGS_PER_ANSWER_COLUMNS)
    assert list(counts.columns) == ["mean", "median", "max", "share_empty"]
    assert counts.loc["one each"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert counts.loc["mostly none"].tolist() == [0.5, 0.0, 2.0, 0.75]
    assert counts.loc["true labels"].tolist() == [1.5, 1.5, 3.0, 0.25]


@pytest.mark.parametrize(
    "sample_ids", [[1, 2, 4], [1, 3, 2]], ids=["other problems", "another order"]
)
def test_tags_per_answer_refuses_predictions_on_other_problems_or_in_another_order(sample_ids):
    truth = pd.DataFrame({"math": [1, 0, 1]}, index=pd.Index([1, 2, 3], name="sample_id"))
    stray = pd.DataFrame({"math": [1, 0, 1]}, index=pd.Index(sample_ids, name="sample_id"))

    with pytest.raises(ValueError, match="stray must be on truth's problems, in the same order"):
        tags_per_answer({"stray": stray}, truth)


@pytest.fixture(name="all_tag_truth")
def all_tag_truth_fixture():
    """Return the true tags of four problems over all 37 tags.

    Four tags are present: math and dp twice, greedy and *special once. *special is not one of
    the paper's 31 tags.
    """
    tags = pd.Series(
        [["math", "dp"], ["math"], ["*special"], ["greedy", "dp"]],
        index=pd.Index([1, 2, 3, 4], name="sample_id"),
    )
    return multi_hot(tags, ALL_TAGS)


def test_all_tag_scores_gives_the_headline_its_variants_and_the_tags_averaged(all_tag_truth):
    missing_a_math = all_tag_truth.copy()
    missing_a_math.loc[2, "math"] = 0  # math: 1 of 2 found, F1 = 2/3
    missing_a_math.loc[1, "hashing"] = 1  # a tag absent from the truth
    predictions = {
        "perfect": all_tag_truth.copy(),
        "nothing": all_tag_truth * 0,
        "partial": missing_a_math,
    }

    scores = all_tag_scores(all_tag_truth, predictions)

    assert scores.index.tolist() == ["perfect", "nothing", "partial"]
    assert scores.index.name == "method"
    assert list(scores.columns) == list(ALL_TAG_COLUMNS)
    assert ", ".join(scores.columns) == (
        "all_tag_macro_f1, present_tags, all_tag_macro_f1_zero_absent, paper_tag_macro_f1"
    )
    assert scores.loc["perfect"].tolist() == pytest.approx([1.0, 4, 4 / 37, 1.0])
    assert scores.loc["nothing"].tolist() == [0.0, 4, 0.0, 0.0]
    # Over the 4 present tags: (2/3 + 1 + 1 + 1) / 4; over all 37, the absent hashing counting
    # 0; over the 3 present tags of the paper's 31 (not *special): (2/3 + 1 + 1) / 3.
    assert scores.loc["partial"].tolist() == pytest.approx([11 / 12, 4, (11 / 3) / 37, 8 / 9])


def test_all_tag_scores_refuses_predictions_that_lack_a_tag(all_tag_truth):
    with pytest.raises(ValueError, match="predicted has no column for the tags"):
        all_tag_scores(all_tag_truth, {"focus only": all_tag_truth[["math", "dp"]]})


def test_cost_per_thousand_scales_each_methods_calls_to_1000_problems(make_answer):
    dev = [
        make_answer(sample_id=sample_id, usage=usage(1000, 100), latency_s=seconds)
        for sample_id, seconds in enumerate([1.0, 2.0, 3.0, 10.0], start=1)
    ]
    refused = make_answer(
        sample_id=5,
        output_text=None,
        status="api_error:400",
        served_model=None,
        usage=usage(0, 0),
        latency_s=0.2,
    )
    test = [
        make_answer(sample_id=sample_id, usage=usage(2000, 200), latency_s=seconds)
        for sample_id, seconds in enumerate([4.0, 6.0], start=10)
    ]
    calls = pd.concat([per_problem_frame([*dev, refused], "dev"), per_problem_frame(test, "test")])
    per_problem = {"code2tag-text": calls.set_index(list(FROZEN_INDEX))}

    costs = cost_per_thousand(per_problem)

    assert costs.index.names == ["method", "part"]
    assert costs.index.tolist() == [("code2tag-text", "dev"), ("code2tag-text", "test")]
    assert list(costs.columns) == list(COST_COLUMNS)
    assert list(costs.columns) == [
        "input_tokens",
        "output_tokens",
        "total_cost_usd",
        "median_latency_s",
    ]
    # 1,000 uncached input and 100 output tokens cost 1e-4 + 5e-5 USD; 2,000 and 200 twice that.
    # The refused dev answer used no token and cost nothing, but its problem counts: the dev
    # means are 4/5 of those of the other four answers.
    assert costs.loc[("code2tag-text", "dev")].tolist() == pytest.approx(
        [800_000, 80_000, 0.12, 2.0]
    )
    assert costs.loc[("code2tag-text", "test")].tolist() == pytest.approx(
        [2_000_000, 200_000, 0.3, 5.0]
    )


@pytest.fixture(name="one_tag")
def one_tag_fixture():
    """Return the truth and two methods' predictions of math and dp over six problems.

    On math, ``a`` is right on problems 10, 13 and 15, and ``b`` on 10, 11 and 14: both are
    right on 10, both wrong on 12, and exactly one is right on 11, 13, 14 and 15. On dp, both
    predict the tag for every problem and the truth has it for none.
    """
    index = pd.Index([10, 11, 12, 13, 14, 15], name="sample_id")
    truth = pd.DataFrame({"math": [1, 0, 1, 0, 1, 0], "dp": [0] * 6}, index=index)
    a = pd.DataFrame({"math": [1, 1, 0, 0, 0, 0], "dp": [1] * 6}, index=index)
    b = pd.DataFrame({"math": [1, 0, 0, 1, 1, 1], "dp": [1] * 6}, index=index)
    return truth, a, b


def test_disagreements_lists_the_first_problems_where_exactly_one_method_is_right(one_tag):
    truth, a, b = one_tag

    shown = disagreements(truth, a, b, "math")

    assert shown.index.tolist() == [11, 13, 14]
    assert shown.index.name == "sample_id"
    assert shown.columns.tolist() == ["truth", "a", "b"]
    assert shown.to_dict("records") == [
        {"truth": 0, "a": 1, "b": 0},
        {"truth": 0, "a": 0, "b": 1},
        {"truth": 1, "a": 0, "b": 1},
    ]


def test_disagreements_returns_at_most_limit_problems_and_none_when_the_methods_agree(one_tag):
    truth, a, b = one_tag

    assert disagreements(truth, a, b, "math", limit=1).index.tolist() == [11]
    assert disagreements(truth, a, b, "math", limit=10).index.tolist() == [11, 13, 14, 15]
    agree = disagreements(truth, a, b, "dp")
    assert agree.empty
    assert agree.columns.tolist() == ["truth", "a", "b"]


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize(
    "rows", [[1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5]], ids=["other problems", "another order"]
)
def test_disagreements_refuses_predictions_on_other_problems_or_in_another_order(
    one_tag, side, rows
):
    truth, a, b = one_tag
    predictions = {"a": a, "b": b}
    predictions[side] = predictions[side].iloc[rows]

    with pytest.raises(ValueError, match=f"{side} must be on truth's problems, in the same order"):
        disagreements(truth, predictions["a"], predictions["b"], "math")


def test_the_module_imports_neither_mlflow_nor_a_provider_sdk():
    # A fresh interpreter, so that the imports of other tests do not count: the notebook that
    # reads these tables asserts that it loaded neither.
    script = (
        "import sys\n"
        "import codeforces_tags.analysis.llm_methods\n"
        "print(sorted(name for name in ('mlflow', 'openai') if name in sys.modules))\n"
    )

    imported = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout

    assert imported.strip() == "[]"
