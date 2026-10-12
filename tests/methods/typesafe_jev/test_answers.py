"""Tests for the Jev readings of answer lines: probabilities, summaries and the served model."""

import json
import math
import re

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.predictions import PER_PROBLEM_COLUMNS
from codeforces_tags.llm.answers import answer_cost_usd, per_problem_frame
from codeforces_tags.llm.client import USAGE_KEYS
from codeforces_tags.methods.typesafe_jev import answers
from codeforces_tags.methods.typesafe_jev.settings import MODEL

# The usage of a line that read 3,000 input tokens: TypeSafe's output tokens are free.
USAGE_3000 = {"input_tokens": 3000, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}


def with_tag(tag, value):
    """Return the output text of a completed line: 0.5 on every focus tag but ``tag``."""
    return json.dumps({**dict.fromkeys(FOCUS_TAGS, 0.5), tag: value})


def test_probabilities_read_each_completed_answer_in_focus_order_and_index_order(jev_line):
    # The keys of a line's text may come in any order: the columns are FOCUS_TAGS.
    reversed_text = json.dumps({tag: 0.25 for tag in reversed(FOCUS_TAGS)} | {"math": 0.75})
    lines = [
        jev_line(sample_id=10, probabilities={"graphs": 0.125, "probabilities": 1.0}),
        jev_line(sample_id=20, output_text=reversed_text),
        jev_line(sample_id=30),
    ]
    index = pd.Index([30, 10, 20], name="sample_id")

    scores = answers.probabilities(lines, index)

    expected = pd.DataFrame(
        [
            [0.5] * 8,
            [0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        ],
        index=index,
        columns=list(FOCUS_TAGS),
    )
    pd.testing.assert_frame_equal(scores, expected)
    assert scores.dtypes.eq("float64").all()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "api_error:422"},
        {"status": "api_error:400"},
        {"status": "malformed"},
        # The text of a failed line is never read, whatever it holds.
        {"status": "malformed", "output_text": with_tag("math", 0.9)},
    ],
)
def test_probabilities_score_a_failed_line_0_on_every_tag(jev_line, changes):
    lines = [jev_line(sample_id=1), jev_line(sample_id=2, **changes)]

    scores = answers.probabilities(lines, pd.Index([1, 2], name="sample_id"))

    assert scores.loc[2].tolist() == [0.0] * len(FOCUS_TAGS)
    assert scores.loc[1].tolist() == [0.5] * len(FOCUS_TAGS)


def test_probabilities_refuse_a_problem_without_an_answer(jev_line):
    with pytest.raises(ValueError, match=re.escape("The problems [2, 3] have no answer yet.")):
        answers.probabilities([jev_line(sample_id=1)], pd.Index([1, 2, 3], name="sample_id"))


def test_probabilities_refuse_an_answer_to_another_problem_or_two_to_one(jev_line):
    index = pd.Index([1], name="sample_id")

    with pytest.raises(ValueError, match=re.escape("The problems [7] have an answer but are not")):
        answers.probabilities([jev_line(sample_id=1), jev_line(sample_id=7)], index)
    twice = [jev_line(sample_id=1), jev_line(sample_id=1, request_sha256="f" * 64)]
    with pytest.raises(ValueError, match=re.escape("The problems [1] have several answers")):
        answers.probabilities(twice, index)


@pytest.mark.parametrize(
    "output_text",
    [
        json.dumps({tag: 0.5 for tag in FOCUS_TAGS if tag != "games"}),
        with_tag("games", "high"),
        with_tag("games", True),
        with_tag("games", None),
        with_tag("games", math.nan),
        with_tag("games", 1.5),
        "games, math",
        "[0.5, 0.5]",
        None,
    ],
)
def test_probabilities_refuse_a_completed_line_without_every_tag(jev_line, output_text):
    line = jev_line(sample_id=4, output_text=output_text)

    with pytest.raises(ValueError, match="The completed answer to problem 4 does not hold"):
        answers.probabilities([line], pd.Index([4], name="sample_id"))


def test_the_toolkit_prices_and_describes_a_jev_line(jev_line):
    # config.LLM_PRICES prices jev-1.13.0, so the toolkit's per_problem.csv rows serve Jev as
    # they are: 2,500 input tokens at 0.042 USD per million cost 0.000105 USD, output is free.
    lines = [jev_line(sample_id=14, status="api_error:422", latency_s=0.1), jev_line(sample_id=13)]

    frame = per_problem_frame(lines, "dev")

    assert answer_cost_usd(lines[1]) == {
        "input_cost": 0.000105,
        "output_cost": 0.0,
        "total_cost": 0.000105,
    }
    # A refused request read no token: it costs nothing, and its row names no model.
    assert lines[0].usage == dict.fromkeys(USAGE_KEYS, 0)
    assert answer_cost_usd(lines[0])["total_cost"] == 0.0
    assert list(frame.columns) == list(PER_PROBLEM_COLUMNS)
    assert frame.to_csv(index=False).splitlines() == [
        ",".join(PER_PROBLEM_COLUMNS),
        "13,dev,typesafe,2500,0,0,0.000105,0.0,0.000105,0.4,jev-1.13.0,req-13,completed",
        "14,dev,typesafe,0,0,0,0.0,0.0,0.0,0.1,,req-14,api_error:422",
    ]


def test_answers_summary_counts_lines_statuses_failures_models_tokens_and_cost(jev_line):
    lines = [
        jev_line(sample_id=1),
        jev_line(sample_id=2, usage=USAGE_3000),
        jev_line(sample_id=3, status="api_error:422"),
        # A malformed answer was answered, and billed: its model and tokens count.
        jev_line(sample_id=4, status="malformed", served_model=MODEL, usage=USAGE_3000),
    ]

    summary = answers.answers_summary(lines)

    assert summary == {
        "lines": 4,
        "statuses": {"api_error:422": 1, "completed": 2, "malformed": 1},
        "failed": 2,
        "served_models": {"jev-1.13.0": 3},
        "input_tokens": 8_500,
        "output_tokens": 0,
        # 0.042 USD per million input tokens: 0.000105 USD for 2,500, 0.000126 for 3,000.
        "cost_usd": pytest.approx(0.000357, rel=1e-12, abs=0),
    }
    assert list(summary) == [
        "lines",
        "statuses",
        "failed",
        "served_models",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    ]
    # Plain Python values: a run record refuses numpy scalars, which JSON does not encode.
    assert json.loads(json.dumps(summary)) == summary
    counts = [summary[key] for key in ("lines", "failed", "input_tokens", "output_tokens")]
    assert all(isinstance(count, int) for count in counts)


def test_answers_summary_of_no_answer():
    assert answers.answers_summary([]) == {
        "lines": 0,
        "statuses": {},
        "failed": 0,
        "served_models": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def test_answers_summary_refuses_tokens_from_an_unpriced_model(jev_line):
    with pytest.raises(ValueError, match="'jev-latest' has no price in config.LLM_PRICES"):
        answers.answers_summary([jev_line(served_model="jev-latest")])


def test_check_served_model_accepts_the_pinned_model_and_failed_lines(jev_line):
    lines = [
        jev_line(sample_id=1),
        jev_line(sample_id=2, status="api_error:400"),
        jev_line(sample_id=3, status="malformed"),
    ]

    assert answers.check_served_model(lines) is None
    assert answers.check_served_model([]) is None


@pytest.mark.parametrize("model", ["jev-latest", "jev-1.14.0"])
def test_check_served_model_refuses_another_model_naming_the_problems(jev_line, model):
    lines = [
        jev_line(sample_id=1),
        jev_line(sample_id=9, served_model=model),
        jev_line(sample_id=5, served_model=model),
    ]
    message = f"The answers to the problems [5, 9] were served by ['{model}'], not jev-1.13.0:"

    with pytest.raises(ValueError, match=re.escape(message)):
        answers.check_served_model(lines)


def test_check_served_model_reads_the_pinned_model_when_called(jev_line, monkeypatch):
    # freeze and verify tests patch answers.MODEL to stand for an answer from another model.
    monkeypatch.setattr(answers, "MODEL", "jev-2.0.0")

    with pytest.raises(ValueError, match=re.escape("served by ['jev-1.13.0'], not jev-2.0.0")):
        answers.check_served_model([jev_line()])
