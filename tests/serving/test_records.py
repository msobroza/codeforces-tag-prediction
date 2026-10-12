"""Tests for the served records: what the model server answers for raw problems.

A served record is what the command line predicts for a problem, as one row. The local methods
are fakes of ``tests/methods/fakes/conformance_fakes.py``. The methods that call an API are the
frozen ``jev-statement-code`` behind the command line's fake Jev session (``cli_jev``), and a
prompted fake whose answers ``make_answer`` builds: no request leaves the process and no key is
read.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import FOCUS_TAGS, LLM_PRICES
from codeforces_tags.methods import METHODS, MethodSpec, load_method, model_dir
from codeforces_tags.methods.typesafe_jev.settings import MODEL as JEV_MODEL
from codeforces_tags.serving import records
from codeforces_tags.serving.records import (
    COST_COLUMN,
    FAILED_COLUMN,
    METHOD_COLUMN,
    score_column,
    served_records,
    threshold_column,
)

# The method that calls an API and has scores: the command line's default, frozen, asking a
# fake Jev.
JEV = "jev-statement-code"
# A module that no test may import: a spec that names it proves that nothing was imported.
NEVER_IMPORTED = "codeforces_tags.no_such_module"
FOCUS = list(FOCUS_TAGS)
SCORES = [score_column(tag) for tag in FOCUS_TAGS]
THRESHOLDS = [threshold_column(tag) for tag in FOCUS_TAGS]


def local_spec(kind):
    """Return the spec of a local method of the given kind, served by no module."""
    return MethodSpec(module=NEVER_IMPORTED, kind=kind, requirements=())


class PromptedFake:  # pylint: disable=too-few-public-methods
    """Calls an API and has no scores, as a prompted method: math for each problem, answered."""

    name = "prompted_fake"
    tags = FOCUS_TAGS

    def __init__(self, build):
        """Bind the answer builder (``make_answer``)."""
        self.build = build

    async def apredict_with_answers(self, raw_problems):
        """Predict math for each problem, with one completed answer each."""
        tags = pd.DataFrame(0, index=raw_problems.index, columns=FOCUS).assign(math=1)
        return tags, [self.build(sample_id=int(sample_id)) for sample_id in raw_problems.index]


@pytest.fixture(name="jev")
def jev_fixture(cli_jev):  # pylint: disable=unused-argument
    """Return the frozen jev-statement-code, whose requests the command line's fake Jev answers."""
    return load_method(model_dir(JEV))


def test_the_columns_are_named_after_the_focus_tags():
    assert (score_column("number theory"), threshold_column("math")) == (
        "score: number theory",
        "threshold: math",
    )
    assert (METHOD_COLUMN, FAILED_COLUMN, COST_COLUMN) == ("method", "failed", "cost_usd")


def test_a_score_method_is_served_its_focus_tags_scores_and_per_tag_thresholds(fakes, raw_problems):
    # EdgeFake scores a problem asked alone 1.0, above its threshold of 0.5, and a batch 0.0.
    method = fakes.EdgeFake(offset=1.0, threshold=0.5)

    alone = served_records(method, local_spec("score"), raw_problems.loc[[10]])
    batch = served_records(method, local_spec("score"), raw_problems)

    assert list(alone.columns) == [METHOD_COLUMN, *FOCUS, *SCORES, *THRESHOLDS]
    assert alone[METHOD_COLUMN].tolist() == ["edge_fake"]
    assert alone.loc[10, FOCUS].tolist() == [1] * 8
    assert alone.loc[10, SCORES].tolist() == [1.0] * 8
    assert alone.loc[10, THRESHOLDS].tolist() == [0.5] * 8
    assert alone.dtypes.drop(METHOD_COLUMN).tolist() == ["int64"] * 8 + ["float64"] * 16
    assert batch.index.tolist() == [30, 10, 20]
    assert batch[FOCUS].eq(0).all().all()
    assert batch[SCORES].eq(0.0).all().all()
    assert batch[THRESHOLDS].eq(0.5).all().all()


def test_a_score_method_is_scored_once_and_its_tags_come_from_those_scores(fakes, raw_problems):
    scored = []

    class CountedFake(fakes.EdgeFake):  # pylint: disable=too-few-public-methods
        """An EdgeFake that records the problems of each ``scores`` call."""

        def scores(self, raw_problems):
            """Record the call, then score as EdgeFake does."""
            scored.append(raw_problems.index.tolist())
            return super().scores(raw_problems)

    served_records(CountedFake(offset=1.0, threshold=0.5), local_spec("score"), raw_problems)

    assert scored == [[30, 10, 20]]


def test_a_method_without_scores_is_served_its_focus_tags_only(fakes, raw_problems):
    # KeywordFake predicts all 37 tags from the words of the statement and the notes.
    method = fakes.KeywordFake()

    served = served_records(method, local_spec("zero_shot"), raw_problems)

    assert list(served.columns) == [METHOD_COLUMN, *FOCUS]
    assert served[METHOD_COLUMN].tolist() == ["keyword_fake"] * 3
    pd.testing.assert_frame_equal(served[FOCUS], method.predict(raw_problems)[FOCUS])
    assert served.loc[20, ["math", "number theory"]].tolist() == [1, 1]
    assert served["strings"].tolist() == [0, 1, 0]


def test_an_api_method_is_served_its_scores_thresholds_failed_answer_and_cost(
    jev, cli_jev, raw_problems
):
    # One request per problem; its scores come from the answer that gave its tags.
    cli_jev.refused = {20}

    answered = served_records(jev, METHODS[JEV], raw_problems.loc[[10]])
    refused = served_records(jev, METHODS[JEV], raw_problems.loc[[20]])

    cost = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    assert cli_jev.asked == [10, 20]
    assert list(answered.columns) == [
        METHOD_COLUMN,
        *FOCUS,
        *SCORES,
        *THRESHOLDS,
        FAILED_COLUMN,
        COST_COLUMN,
    ]
    assert answered.loc[10, METHOD_COLUMN] == JEV
    assert [tag for tag in FOCUS_TAGS if answered.loc[10, tag]] == ["math", "number theory"]
    assert answered.loc[10, SCORES].tolist() == list(cli_jev.probabilities.values())
    assert answered.loc[10, THRESHOLDS].tolist() == jev.thresholds.thresholds_.tolist()
    assert answered.loc[10, COST_COLUMN] == pytest.approx(cost)
    assert not answered.loc[10, FAILED_COLUMN]
    assert refused.loc[20, FOCUS].tolist() == [0] * 8
    assert refused.loc[20, SCORES].tolist() == [0.0] * 8
    assert (bool(refused.loc[20, FAILED_COLUMN]), refused.loc[20, COST_COLUMN]) == (True, 0.0)
    assert (answered[FAILED_COLUMN].dtype, answered[COST_COLUMN].dtype) == ("bool", "float64")


def test_an_api_method_without_scores_is_served_its_focus_tags_failed_answer_and_cost(
    make_answer, raw_problems
):
    spec = MethodSpec(module=NEVER_IMPORTED, kind="zero_shot", requirements=(), calls_api=True)

    served = served_records(PromptedFake(make_answer), spec, raw_problems.loc[[10]])

    prices = LLM_PRICES["gpt-6-luna"]  # make_answer's model: 100 input and 5 output tokens
    assert list(served.columns) == [METHOD_COLUMN, *FOCUS, FAILED_COLUMN, COST_COLUMN]
    assert served.loc[10, FOCUS].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert not served.loc[10, FAILED_COLUMN]
    assert served.loc[10, COST_COLUMN] == pytest.approx(
        (100 * prices["input"] + 5 * prices["output"]) / 1e6
    )


@pytest.mark.parametrize("count", [0, 2])
def test_an_api_method_takes_exactly_one_problem_per_request_and_is_not_asked_otherwise(
    jev, cli_jev, raw_problems, count
):
    with pytest.raises(CliError) as error:
        served_records(jev, METHODS[JEV], raw_problems.iloc[:count])

    assert cli_jev.asked == []
    assert str(error.value) == (
        "jev-statement-code sends a paid request per problem, and the model server asks a method "
        "that calls an API about exactly one problem per request: a request carries no cost cap, "
        "which more than one paid request needs (--max-cost-usd on the command line). Send each "
        f"problem in a request of its own: this request holds {count}."
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("prob_desc_description", [0, 2], " \n"),
            "every sample needs a text in prob_desc_description and source_code, but "
            "prob_desc_description holds none in request row 0, request row 2.",
        ),
        (
            ("prob_desc_sample_inputs", [1], "3 4"),
            "a sample holds a field that no method can read: prob_desc_sample_inputs holds "
            "neither a list of texts (or its JSON text) nor null in request row 1.",
        ),
    ],
    ids=["blank_statement", "unreadable_sample_inputs"],
)
def test_a_field_the_command_line_refuses_is_refused_by_its_request_row_before_any_prediction(
    fakes, raw_problems, monkeypatch, change, message
):
    # A raw HTTP client sends rows that MLflow indexes 0, 1, 2: they are named by that index, and
    # no field's text is quoted. change: the column, the rows and the value they get.
    column, rows, value = change
    request = raw_problems.reset_index(drop=True)
    request.loc[rows, column] = value
    run_method = Mock(name="run_method")
    monkeypatch.setattr(records, "run_method", run_method)

    with pytest.raises(CliError) as error:
        served_records(fakes.EdgeFake(offset=1.0, threshold=0.5), local_spec("score"), request)

    assert str(error.value) == message
    run_method.assert_not_called()


def test_an_api_method_sends_no_request_for_a_problem_without_a_solution(
    jev, cli_jev, raw_problems
):
    # The fields are checked before the one-problem rule: two problems, one of them blank, are
    # refused for the blank field, and no request is sent.
    request = raw_problems.reset_index(drop=True).iloc[:2].assign(source_code=["print(1)", None])

    with pytest.raises(CliError) as error:
        served_records(jev, METHODS[JEV], request)

    assert cli_jev.asked == []
    assert str(error.value) == (
        "every sample needs a text in prob_desc_description and source_code, but source_code "
        "holds none in request row 1."
    )
