"""Tests for reading the tags of an answer, encoding answers as 0/1, and counting their cost."""

import pandas as pd
import pytest

from codeforces_tags.config import ALL_TAGS, FOCUS_TAGS, LLM_PRICES
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_COLUMNS,
    PER_PROBLEM_FILE,
    load_per_problem,
)
from codeforces_tags.llm.answers import (
    ParsedTags,
    answer_cost_usd,
    answer_summary,
    cost_usd,
    encode_answers,
    in_flight_seconds,
    parse_answer,
    parse_json_tags,
    parse_text_tags,
    per_problem_frame,
    problem_list,
)
from codeforces_tags.llm.settings import OUTPUT_FORMATS

# What a model may answer in free text, and what the fixed parser makes of it: the known tags in
# the order they appear, and the number of pieces that name no tag.
ODD_TEXT_ANSWERS = [
    pytest.param("", (), 0, id="empty text"),
    pytest.param(" \n\t\n", (), 0, id="blank text"),
    pytest.param("None", (), 1, id="the word None names no tag"),
    pytest.param("dp, math", ("dp", "math"), 0, id="plain"),
    pytest.param("DP, Math,  Binary Search", ("dp", "math", "binary search"), 0, id="other case"),
    pytest.param("1. dp\n2. math\n3) greedy", ("dp", "math", "greedy"), 0, id="numbered list"),
    pytest.param("- dp\n* math\n\u2022 greedy", ("dp", "math", "greedy"), 0, id="bulleted list"),
    pytest.param("dp, math.", ("dp", "math"), 0, id="final period"),
    pytest.param(
        "\"dp\", `math`, 'greedy', \u201ctrees\u201d",
        ("dp", "math", "greedy", "trees"),
        0,
        id="quotes and backticks",
    ),
    pytest.param('"dp.", "math".', ("dp", "math"), 0, id="period inside or outside the quotes"),
    pytest.param("```\ndp, math\n```", ("dp", "math"), 0, id="code fence"),
    pytest.param("dp, DP, dp.", ("dp",), 0, id="duplicates"),
    pytest.param("dp, quantum computing, math", ("dp", "math"), 1, id="a tag outside the list"),
    pytest.param(
        "binary   search,dfs\tand  similar",
        ("binary search", "dfs and similar"),
        0,
        id="inner spaces",
    ),
    pytest.param("*special, 2-sat", ("*special", "2-sat"), 0, id="a tag that looks like a marker"),
    pytest.param("* *special\n1. 2-sat", ("*special", "2-sat"), 0, id="a marker before such a tag"),
    pytest.param("-dp", (), 1, id="a marker needs a space after it"),
    pytest.param("dp\r\nmath\r\n", ("dp", "math"), 0, id="Windows line breaks"),
    pytest.param("dp,, math,\n\n", ("dp", "math"), 0, id="empty pieces"),
    pytest.param("dp; math", (), 1, id="a semicolon does not separate pieces"),
    pytest.param("Sure! The tags are: dp, math.", ("math",), 1, id="a sentence around the tags"),
]
# Answers that are not {"tags": [str, ...]}: JSON that cannot be read, or has another shape.
UNREADABLE_JSON = [
    pytest.param('{"tags": ["dp", "ma', id="truncated"),
    pytest.param("", id="empty"),
    pytest.param("dp, math", id="free text"),
    pytest.param("[]", id="a list"),
    pytest.param("null", id="null"),
    pytest.param('"dp"', id="a string"),
    pytest.param('{"tag": ["dp"]}', id="another key"),
    pytest.param('{"tags": "dp"}', id="a string instead of a list"),
    pytest.param('{"tags": ["dp", 3]}', id="a number among the tags"),
    pytest.param("[" * 100_000, id="nested too deep"),
]
# How an answer may list every tag: the same tags, written in each style.
LIST_STYLES = {
    "bare": lambda _number, tag: tag,
    "hyphen bullets": lambda _number, tag: f"- {tag}",
    "star bullets": lambda _number, tag: f"* {tag}",
    "dot bullets": lambda _number, tag: f"\u2022 {tag}",
    "numbers with a period": lambda number, tag: f"{number}. {tag}",
    "numbers with a parenthesis": lambda number, tag: f"{number}) {tag}",
    "quoted": lambda _number, tag: f'"{tag}"',
    "backticked": lambda _number, tag: f"`{tag}`",
    "final period": lambda _number, tag: f"{tag}.",
    "upper case": lambda _number, tag: tag.upper(),
}


def tokens(input_tokens=0, cached=0, output=0, reasoning=0):
    """Return the usage of an answer, by the keys of ``USAGE_KEYS``."""
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
    }


@pytest.mark.parametrize(("output_text", "tags", "unknown"), ODD_TEXT_ANSWERS)
def test_parse_text_tags_handles_odd_answers(output_text, tags, unknown):
    parsed = parse_text_tags(output_text)

    assert parsed == ParsedTags(tags, unknown, failed=False)


@pytest.mark.parametrize("style", LIST_STYLES)
def test_parse_text_tags_reads_every_tag_back_in_any_list_style(style):
    # No tag may be mangled by the cleaning: "*special" and "2-sat" start like list markers.
    lines = [LIST_STYLES[style](number, tag) for number, tag in enumerate(ALL_TAGS, start=1)]

    parsed = parse_text_tags("\n".join(lines))

    assert parsed == ParsedTags(ALL_TAGS, 0, failed=False)


def test_parse_text_tags_reads_every_tag_back_from_one_line_of_commas():
    assert parse_text_tags(", ".join(ALL_TAGS)) == ParsedTags(ALL_TAGS, 0, failed=False)


def test_parse_text_tags_keeps_only_the_tags_it_is_given():
    parsed = parse_text_tags("dp, math, trees", all_tags=("math", "trees"))

    assert parsed == ParsedTags(("math", "trees"), 1, failed=False)


def test_parse_text_tags_counts_every_occurrence_of_a_piece_outside_the_list():
    assert parse_text_tags("quantum, quantum, dp").unknown_pieces == 2


def test_parse_json_tags_reads_the_tags_list():
    parsed = parse_json_tags('{"tags": ["math", "dp"]}')

    assert parsed == ParsedTags(("math", "dp"), 0, failed=False)


def test_parse_json_tags_removes_duplicates_and_counts_the_tags_outside_the_list():
    parsed = parse_json_tags('{"tags": ["dp", "dp", "quantum", "quantum", "math"]}')

    assert parsed == ParsedTags(("dp", "math"), 2, failed=False)


def test_parse_json_tags_reads_an_empty_list_as_an_answer_with_no_tag():
    assert parse_json_tags('{"tags": []}') == ParsedTags((), 0, failed=False)


def test_parse_json_tags_matches_the_tags_exactly():
    # The schema's enum gives the exact tags: no case folding, unlike free text.
    assert parse_json_tags('{"tags": ["DP"]}') == ParsedTags((), 1, failed=False)


@pytest.mark.parametrize("output_text", UNREADABLE_JSON)
def test_parse_json_tags_fails_when_the_json_cannot_be_read_or_has_another_shape(output_text):
    assert parse_json_tags(output_text) == ParsedTags((), 0, failed=True)


@pytest.mark.parametrize(
    "status", ["incomplete", "refused", "failed", "cancelled", "api_error:400", "api_error:422"]
)
@pytest.mark.parametrize("output_format", OUTPUT_FORMATS)
def test_parse_answer_fails_an_answer_that_did_not_complete(make_answer, status, output_format):
    answer = make_answer(status=status, output_text='{"tags": ["dp"]}')

    assert parse_answer(answer, output_format) == ParsedTags((), 0, failed=True)


@pytest.mark.parametrize("output_format", OUTPUT_FORMATS)
def test_parse_answer_fails_a_completed_answer_that_has_no_text(make_answer, output_format):
    assert parse_answer(make_answer(output_text=None), output_format).failed


@pytest.mark.parametrize(
    ("output_format", "output_text"),
    [("text", "dp, math"), ("json", '{"tags": ["dp", "math"]}')],
)
def test_parse_answer_reads_a_completed_answer_in_its_output_format(
    make_answer, output_format, output_text
):
    parsed = parse_answer(make_answer(output_text=output_text), output_format)

    assert parsed == ParsedTags(("dp", "math"), 0, failed=False)


def test_the_output_format_decides_which_parser_reads_an_answer(make_answer):
    answer = make_answer(output_text='{"tags": ["dp"]}')

    assert parse_answer(answer, "json") == ParsedTags(("dp",), 0, failed=False)
    assert parse_answer(answer, "text") == ParsedTags((), 1, failed=False)


def test_a_cut_off_json_answer_fails_but_the_same_text_is_a_list_of_pieces_in_free_text(
    make_answer,
):
    answer = make_answer(output_text='{"tags": ["dp", "ma')

    assert parse_answer(answer, "json").failed
    assert parse_answer(answer, "text") == ParsedTags((), 2, failed=False)


def test_parse_answer_uses_the_tags_it_is_given(make_answer):
    parsed = parse_answer(make_answer(), "text", all_tags=("dp",))

    assert parsed == ParsedTags(("dp",), 1, failed=False)


def test_parse_answer_refuses_an_unknown_output_format(make_answer):
    with pytest.raises(ValueError, match=r"'xml' is not an output format: expected one of \("):
        parse_answer(make_answer(status="incomplete"), "xml")


def test_encode_answers_gives_a_row_of_zeros_and_ones_per_problem_in_the_order_of_the_index(
    make_answer,
):
    # The answers come in the order they arrived, not the order of the problems.
    answers = [
        make_answer(sample_id=2, output_text="math"),
        make_answer(sample_id=3, output_text="trees, dp, math"),
        make_answer(sample_id=1, output_text="dp"),
    ]
    index = pd.Index([3, 1, 2], name="sample_id")

    encoded = encode_answers(answers, index, "text", all_tags=("dp", "math", "trees"))

    assert encoded.index.equals(index)
    assert list(encoded.columns) == ["dp", "math", "trees"]
    assert encoded.to_numpy().tolist() == [[1, 1, 1], [1, 0, 0], [0, 1, 0]]
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in encoded.dtypes)


def test_encode_answers_has_a_column_for_each_of_the_37_tags_by_default(make_answer):
    encoded = encode_answers([make_answer(output_text="math")], pd.Index([1]), "text")

    assert list(encoded.columns) == list(ALL_TAGS)
    assert encoded.loc[1, "math"] == 1
    assert encoded.to_numpy().sum() == 1


def test_encode_answers_reads_json_answers(make_answer):
    answers = [make_answer(output_text='{"tags": ["dp", "dp", "math"]}')]

    encoded = encode_answers(answers, pd.Index([1]), "json", all_tags=("dp", "math"))

    assert encoded.to_numpy().tolist() == [[1, 1]]


def test_encode_answers_counts_failures_and_unknown_pieces(make_answer):
    # Every odd answer is scored, none crashes the encoding, and none drops its problem: a failed
    # answer is the empty tag set, a piece outside the list is dropped and counted.
    answers = [
        make_answer(sample_id=1, output_text="dp, math"),
        make_answer(sample_id=2, output_text="dp, quantum, None"),
        make_answer(sample_id=3, output_text="dp, ma", status="incomplete"),
        make_answer(
            sample_id=4,
            output_text=None,
            status="api_error:400",
            served_model=None,
            usage=tokens(),
        ),
        make_answer(sample_id=5, output_text="", status="refused"),
        make_answer(sample_id=6, output_text=""),
        make_answer(sample_id=7, output_text="DP.\n- Math\n2. dp"),
    ]
    index = pd.Index(range(1, 8), name="sample_id")

    encoded = encode_answers(answers, index, "text", all_tags=("dp", "math"))
    summary = answer_summary(answers, "text", all_tags=("dp", "math"))

    assert encoded.to_numpy().tolist() == [[1, 1], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1]]
    assert summary["failed"] == 3
    assert summary["unknown_pieces"] == 2
    assert summary["empty_answers"] == 1


def test_encode_answers_re_parses_the_text_and_never_reads_the_stored_copy(make_answer):
    # The runner stores tags, failed and unknown_pieces beside the text, for reading: a line
    # edited to say otherwise changes nothing, because the answer's text and status are the record.
    lying = make_answer(
        sample_id=1, output_text="dp", tags=("math",), failed=True, unknown_pieces=9
    )
    failed = make_answer(
        sample_id=2, output_text="dp", status="incomplete", tags=("dp",), failed=False
    )

    encoded = encode_answers([lying, failed], pd.Index([1, 2]), "text", all_tags=("dp", "math"))

    assert encoded.to_numpy().tolist() == [[1, 0], [0, 0]]
    assert answer_summary([lying, failed], "text", all_tags=("dp", "math"))["failed"] == 1


def test_encode_answers_refuses_a_problem_that_has_no_answer(make_answer):
    answers = [make_answer(sample_id=1)]

    with pytest.raises(ValueError, match=r"no answer for the problems \[2, 5\]"):
        encode_answers(answers, pd.Index([1, 5, 2], name="sample_id"), "text")


def test_encode_answers_refuses_an_answer_to_a_problem_that_is_not_encoded(make_answer):
    answers = [make_answer(sample_id=1), make_answer(sample_id=4, request_sha256="b" * 64)]

    with pytest.raises(ValueError, match=r"answers for the problems \[4\]"):
        encode_answers(answers, pd.Index([1]), "text")


def test_encode_answers_refuses_two_answers_to_one_problem(make_answer):
    answers = [make_answer(sample_id=1), make_answer(sample_id=1, request_sha256="b" * 64)]

    with pytest.raises(ValueError, match=r"several answers for the problems \[1\]"):
        encode_answers(answers, pd.Index([1]), "text")


def test_encode_answers_refuses_an_unknown_output_format(make_answer):
    with pytest.raises(ValueError, match="'xml' is not an output format"):
        encode_answers([make_answer()], pd.Index([1]), "xml")


def test_problem_list_shows_the_first_ten_problems_and_counts_the_rest():
    assert problem_list([7, 3]) == "[3, 7]"
    assert problem_list(range(10)) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    assert problem_list(range(25)) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 15 more]"


def test_problem_list_prints_plain_integers_for_numpy_ones():
    assert problem_list(pd.Index([3, 1]).to_numpy()) == "[1, 3]"


def test_cost_usd_charges_the_cached_input_at_the_cached_price():
    # gpt-6-luna: 0.10, 0.01 and 0.50 USD per million tokens (input, cached input, output).
    usage = tokens(input_tokens=1_000_000, cached=400_000, output=200_000, reasoning=150_000)

    cost = cost_usd(usage, "gpt-6-luna")

    assert cost == pytest.approx(
        {"input_cost": 0.6 * 0.10 + 0.4 * 0.01, "output_cost": 0.2 * 0.50, "total_cost": 0.164}
    )
    assert list(cost) == ["input_cost", "output_cost", "total_cost"]


def test_cost_usd_does_not_charge_the_reasoning_tokens_twice():
    # The output tokens already count them.
    cost = cost_usd(tokens(output=1_000_000, reasoning=1_000_000), "gpt-6-luna")

    assert cost["output_cost"] == pytest.approx(0.50)


def test_cost_usd_is_zero_without_tokens():
    assert cost_usd(tokens(), "gpt-6-luna") == {
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
    }


def test_cost_usd_prices_any_model_that_the_config_lists(monkeypatch):
    monkeypatch.setitem(
        LLM_PRICES, "other-model", {"input": 2.0, "cached_input": 1.0, "output": 8.0}
    )

    cost = cost_usd(tokens(input_tokens=500_000, cached=100_000, output=250_000), "other-model")

    assert cost == pytest.approx({"input_cost": 0.9, "output_cost": 2.0, "total_cost": 2.9})


def test_cost_usd_refuses_a_model_that_has_no_price():
    with pytest.raises(
        ValueError, match=r"'gpt-6-luna-2026-09-01' has no price in config.LLM_PRICES"
    ):
        cost_usd(tokens(input_tokens=1), "gpt-6-luna-2026-09-01")


def test_cost_usd_refuses_more_cached_tokens_than_input_tokens():
    with pytest.raises(ValueError, match="cached tokens"):
        cost_usd(tokens(input_tokens=10, cached=11), "gpt-6-luna")


def test_answer_cost_usd_prices_the_tokens_of_the_served_model(make_answer):
    answer = make_answer(usage=tokens(input_tokens=1_000_000, output=2_000_000))

    assert answer_cost_usd(answer) == cost_usd(answer.usage, "gpt-6-luna")


def test_answer_cost_usd_is_zero_for_an_answer_the_api_refused(make_answer):
    # An api_error answer has no served model and no tokens: nothing was billed.
    answer = make_answer(
        status="api_error:400", output_text=None, served_model=None, usage=tokens()
    )

    assert answer_cost_usd(answer) == {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}


def test_answer_cost_usd_is_zero_for_an_incomplete_answer_that_reported_no_usage(make_answer):
    answer = make_answer(status="incomplete", usage=tokens())

    assert answer_cost_usd(answer) == {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}


def test_answer_cost_usd_refuses_tokens_that_no_price_covers(make_answer):
    answer = make_answer(served_model=None, usage=tokens(input_tokens=1))

    with pytest.raises(ValueError, match="None has no price"):
        answer_cost_usd(answer)


def test_in_flight_seconds_counts_the_time_any_request_was_in_flight(make_answer):
    first = 1_791_000_000_000_000_000
    answers = [
        make_answer(started_ns=first, latency_s=2.0),  # 0 to 2 s
        make_answer(started_ns=first + 1_000_000_000, latency_s=0.5),  # within the first
        make_answer(started_ns=first + 1_500_000_000, latency_s=1.0),  # on to 2.5 s
        make_answer(started_ns=first + 10_000_000_000, latency_s=0.25),  # after a gap
    ]

    assert in_flight_seconds(answers) == 2.75  # the gap between two runs is not counted
    assert in_flight_seconds(answers[::-1]) == 2.75


def test_in_flight_seconds_of_no_answer_is_zero():
    assert in_flight_seconds([]) == 0.0


def test_answer_summary_counts_statuses_failures_unknown_pieces_models_tokens_and_cost(
    make_answer,
):
    answers = [
        make_answer(sample_id=1, output_text="dp, math", usage=tokens(1000, 0, 10)),
        make_answer(sample_id=2, output_text="dp, quantum, None", usage=tokens(2000, 500, 20)),
        make_answer(
            sample_id=3, output_text="dp, ma", status="incomplete", usage=tokens(3000, 0, 512)
        ),
        make_answer(
            sample_id=4,
            output_text=None,
            status="api_error:400",
            served_model=None,
            usage=tokens(),
        ),
        make_answer(sample_id=5, output_text="", usage=tokens(4000, 0, 0)),
    ]

    summary = answer_summary(answers, "text")

    assert summary == {
        "answers": 5,
        "statuses": {"api_error:400": 1, "completed": 3, "incomplete": 1},
        "failed": 2,
        "unknown_pieces": 2,
        "empty_answers": 1,
        "served_models": {"gpt-6-luna": 4},
        "usage": tokens(input_tokens=10_000, cached=500, output=542),
        "cost_usd": pytest.approx(
            {
                "input_cost": (9_500 * 0.10 + 500 * 0.01) / 1e6,
                "output_cost": 542 * 0.50 / 1e6,
                "total_cost": (9_500 * 0.10 + 500 * 0.01 + 542 * 0.50) / 1e6,
            }
        ),
    }


def test_answer_summary_reads_json_answers_with_the_json_failure_rule(make_answer):
    answers = [
        make_answer(sample_id=1, output_text='{"tags": ["dp", "quantum"]}'),
        make_answer(sample_id=2, output_text='{"tags": ["dp"'),
    ]

    summary = answer_summary(answers, "json")

    assert (summary["failed"], summary["unknown_pieces"], summary["empty_answers"]) == (1, 1, 0)


def test_answer_summary_of_no_answers_is_all_zeros():
    assert answer_summary([], "text") == {
        "answers": 0,
        "statuses": {},
        "failed": 0,
        "unknown_pieces": 0,
        "empty_answers": 0,
        "served_models": {},
        "usage": tokens(),
        "cost_usd": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0},
    }


def test_answer_summary_uses_the_tags_it_is_given(make_answer):
    summary = answer_summary([make_answer(output_text="dp, math")], "text", all_tags=FOCUS_TAGS)

    assert summary["unknown_pieces"] == 1


def test_per_problem_frame_has_the_columns_of_per_problem_csv(make_answer):
    frame = per_problem_frame([make_answer()], "dev")

    assert tuple(frame.columns) == PER_PROBLEM_COLUMNS
    assert len(frame) == 1


def test_per_problem_frame_describes_each_answer_by_problem_in_sample_id_order(make_answer):
    answers = [
        make_answer(
            sample_id=9,
            response_id="resp_9",
            latency_s=1.5,
            usage=tokens(input_tokens=1_000_000, cached=400_000, output=200_000),
        ),
        make_answer(
            sample_id=4,
            status="incomplete",
            response_id="resp_4",
            latency_s=0.25,
            usage=tokens(input_tokens=2_000, cached=0, output=512),
        ),
    ]

    frame = per_problem_frame(answers, "test")

    assert frame["sample_id"].tolist() == [4, 9]
    assert frame["part"].tolist() == ["test", "test"]
    assert frame["provider"].tolist() == ["openai", "openai"]
    assert frame["input_tokens"].tolist() == [2_000, 1_000_000]
    assert frame["cached_tokens"].tolist() == [0, 400_000]
    assert frame["output_tokens"].tolist() == [512, 200_000]
    assert frame["input_cost_usd"].tolist() == pytest.approx([0.0002, 0.064])
    assert frame["output_cost_usd"].tolist() == pytest.approx([0.000256, 0.1])
    assert frame["total_cost_usd"].tolist() == pytest.approx([0.000456, 0.164])
    assert frame["latency_s"].tolist() == [0.25, 1.5]
    assert frame["served_model"].tolist() == ["gpt-6-luna", "gpt-6-luna"]
    assert frame["response_id"].tolist() == ["resp_4", "resp_9"]
    assert frame["status"].tolist() == ["incomplete", "completed"]


def test_per_problem_frame_costs_agree_with_the_cost_of_each_answer(make_answer):
    answers = [make_answer(sample_id=n, usage=tokens(100 * n, 10, 5 * n)) for n in (1, 2, 3)]

    frame = per_problem_frame(answers, "dev")

    expected = [answer_cost_usd(answer)["total_cost"] for answer in answers]
    assert frame["total_cost_usd"].tolist() == expected


def test_per_problem_frame_of_no_answers_is_empty_with_the_columns():
    frame = per_problem_frame([], "dev")

    assert tuple(frame.columns) == PER_PROBLEM_COLUMNS
    assert frame.empty


def test_per_problem_frame_refuses_two_answers_to_one_problem(make_answer):
    answers = [make_answer(sample_id=1), make_answer(sample_id=1, request_sha256="b" * 64)]

    with pytest.raises(ValueError, match=r"several answers for the problems \[1\]"):
        per_problem_frame(answers, "dev")


def test_per_problem_frame_is_read_back_by_load_per_problem(make_answer, tmp_path):
    answers = [
        make_answer(sample_id=7, usage=tokens(1234, 100, 56), latency_s=0.1 + 0.2),
        make_answer(
            sample_id=3,
            output_text=None,
            status="api_error:400",
            served_model=None,
            response_id=None,
            usage=tokens(),
        ),
    ]

    per_problem_frame(answers, "dev").to_csv(tmp_path / PER_PROBLEM_FILE, index=False)
    loaded = load_per_problem(tmp_path)

    assert loaded.index.tolist() == [("dev", 3), ("dev", 7)]
    assert loaded.loc[("dev", 7), "latency_s"] == 0.1 + 0.2
    assert loaded.loc[("dev", 7), "total_cost_usd"] == answer_cost_usd(answers[0])["total_cost"]
    assert loaded.loc[("dev", 3), "total_cost_usd"] == 0.0
    assert loaded.loc[("dev", 3), "status"] == "api_error:400"
    assert pd.isna(loaded.loc[("dev", 3), "served_model"])
    assert loaded["input_tokens"].dtype == "int64"
