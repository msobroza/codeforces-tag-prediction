"""Tests of the search's reflection step: records, the request, proposals and the cached Luna."""

import copy
import json
from contextlib import asynccontextmanager

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.llm.store import read_answers
from codeforces_tags.methods.typesafe_jev.reflection import (
    TEMPLATE_DIR,
    Proposal,
    Reflector,
    convention_records,
    read_proposal,
    reflection_request,
    tag_records,
    texts_shown,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    REFLECTION_SETTINGS,
    SEARCH_SOLUTION_CHARS,
    SEARCH_STATEMENT_CHARS,
    TAG_DEFINITIONS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    SEED_WORDS,
    length_cap,
    parse_piece,
    piece_text,
)

# The minibatch of the record tests, in the order GEPA evaluated it.
MINIBATCH = [12, 5, 9, 30]
# An include of the seed's graphs criteria, ten words long: its 8-word sequences are exempt.
SEED_ITEM = "states or cells joined by moves and searched as a graph"
# A shown statement's sentence that no seed word holds, and its first nine words.
STORY = "the frog jumps over every lily pad in the pond today"
STORY_ITEM = "the frog jumps over every lily pad in the pond"
# The cost of one make_answer answer: 100 input tokens and 5 output tokens of gpt-6-luna.
ANSWER_COST = (100 * 0.10 + 5 * 0.50) / 1_000_000


@pytest.fixture(name="minibatch")
def minibatch_fixture():
    """Return a hand-built minibatch: its problems, Jev's probabilities, labels and weights.

    Only graphs and math have errors. On graphs (weight 2 for a yes, 1 for a no), problem 12 is
    a missed yes (0.2: error 1.28), 5 and 30 are false yeses at 0.7 (0.49 each, a tie), and 9 a
    caught yes (0.9: 0.02). On math, 9 is a false yes at 0.9 (0.81).
    """
    problems = pd.DataFrame(
        {
            "statement": ["s" * 3_500, "Count the leaves.", "Find a path.", "Sort the array."],
            "source_code": ["c" * 2_500, "print(1)", "print(2)", "print(3)"],
        },
        index=pd.Index(MINIBATCH, name="sample_id"),
    )
    probabilities = pd.DataFrame(0.0, index=problems.index, columns=list(FOCUS_TAGS))
    probabilities["graphs"] = [0.2, 0.7, 0.9, 0.7]
    probabilities.loc[9, "math"] = 0.9
    labels = pd.DataFrame(0, index=problems.index, columns=list(FOCUS_TAGS))
    labels["graphs"] = [1, 0, 1, 0]
    weights = pd.DataFrame({0: [1.0] * 8, 1: [2.0] * 8}, index=list(FOCUS_TAGS))
    return problems, probabilities, labels, weights


def test_tag_records_start_with_the_summary_then_the_worst_examples(minibatch):
    records = tag_records("graphs", *minibatch, k=3)

    assert records[0] == {
        "kind": "summary",
        "tag": "graphs",
        "positives": 2,
        "negatives": 2,
        "mean_probability_positives": pytest.approx(0.55),
        "mean_probability_negatives": pytest.approx(0.7),
    }
    examples = records[1:]
    assert [record["sample_id"] for record in examples] == [12, 5, 30]
    assert [record["gold"] for record in examples] == ["yes", "no", "no"]
    assert [record["probability"] for record in examples] == [0.2, 0.7, 0.7]
    assert [record["error"] for record in examples] == pytest.approx([1.28, 0.49, 0.49])
    assert {record["kind"] for record in examples} == {"example"}
    assert json.loads(json.dumps(records)) == records


def test_tag_records_cut_the_texts_to_the_prompt_lengths(minibatch):
    worst = tag_records("graphs", *minibatch)[1]

    assert worst["statement"] == "s" * SEARCH_STATEMENT_CHARS
    assert worst["solution"] == "c" * SEARCH_SOLUTION_CHARS


def test_tag_records_give_every_problem_when_k_exceeds_the_minibatch(minibatch):
    records = tag_records("graphs", *minibatch)

    assert [record["sample_id"] for record in records[1:]] == [12, 5, 30, 9]


def test_a_tag_summary_has_no_mean_for_a_class_the_minibatch_lacks(minibatch):
    summary = tag_records("games", *minibatch)[0]

    assert summary["positives"] == 0
    assert summary["mean_probability_positives"] is None
    assert summary["mean_probability_negatives"] == 0.0


def test_records_read_only_the_minibatch_rows(minibatch):
    problems, probabilities, labels, weights = minibatch
    extra = pd.Index([99], name="sample_id")
    wider_probabilities = pd.concat(
        [probabilities, pd.DataFrame(0.0, index=extra, columns=list(FOCUS_TAGS))]
    )
    wider_labels = pd.concat([labels, pd.DataFrame(1, index=extra, columns=list(FOCUS_TAGS))])

    tag = tag_records("graphs", problems, wider_probabilities, wider_labels, weights)
    convention = convention_records(problems, wider_probabilities, wider_labels, weights)

    assert tag[0]["positives"] == 2
    assert [record["sample_id"] for record in tag[1:]] == [12, 5, 30, 9]
    assert [record["sample_id"] for record in convention] == [12, 9, 5, 30]


def test_tag_records_refuse_a_tag_outside_the_focus_tags(minibatch):
    with pytest.raises(ValueError, match="'dp' is not a focus tag"):
        tag_records("dp", *minibatch)


def test_convention_records_rank_problems_by_their_summed_error(minibatch):
    records = convention_records(*minibatch, k=2)

    assert [record["sample_id"] for record in records] == [12, 9]
    assert records[1]["error"] == pytest.approx(0.02 + 0.81)
    assert records[1]["tags"]["math"] == {"gold": "no", "probability": 0.9}
    assert records[1]["tags"]["graphs"] == {"gold": "yes", "probability": 0.9}
    assert list(records[1]["tags"]) == list(FOCUS_TAGS)
    assert records[0]["statement"] == "s" * SEARCH_STATEMENT_CHARS
    assert {record["kind"] for record in records} == {"example"}
    assert json.loads(json.dumps(records)) == records


def test_texts_shown_are_the_cut_texts_of_the_examples(minibatch):
    records = tag_records("graphs", *minibatch, k=2)

    assert texts_shown(records) == [
        "s" * SEARCH_STATEMENT_CHARS,
        "c" * SEARCH_SOLUTION_CHARS,
        "Count the leaves.",
        "print(1)",
    ]


def test_a_tag_request_asks_luna_for_the_tag_criteria(minibatch):
    records = tag_records("graphs", *minibatch, k=2)

    request = reflection_request("graphs", SEED_WORDS, records, iteration=17)

    assert request.sample_id == 17
    assert request.method == GEPA_NAME
    assert request.settings == REFLECTION_SETTINGS
    assert request.schema_name == "tag_criteria"
    assert request.schema["required"] == ["true", "false"]
    assert request.body()["text"]["format"]["strict"] is True
    assert piece_text(SEED_WORDS, "graphs") in request.input
    assert SEED_WORDS.convention in request.input
    assert "Would Codeforces put the tag `graphs` on this problem?" in request.input
    assert f"Stay within {length_cap('graphs')} characters" in request.input
    assert "(the current one holds 687)" in request.input
    assert "Codeforces put `graphs` on it: yes. Jev's probability of yes: 0.200." in request.input
    assert "a sample of 4 problems (2 with the tag, 2 without it)" in request.input
    assert "Count the leaves." in request.input
    assert "Find a path." not in request.input  # problem 9, in the minibatch but not shown
    assert "Sort the array." not in request.input  # problem 30, likewise


def test_a_convention_request_shows_every_tag_of_each_example(minibatch):
    records = convention_records(*minibatch, k=1)

    request = reflection_request("convention", SEED_WORDS, records, iteration=3)

    assert request.sample_id == 3
    assert request.schema_name == "convention"
    assert request.schema["required"] == ["convention"]
    assert SEED_WORDS.convention in request.input
    assert SEED_WORDS.question in request.input
    assert all(piece_text(SEED_WORDS, tag) in request.input for tag in FOCUS_TAGS)
    assert "- `graphs`: Codeforces yes, Jev 0.200\n- `strings`:" in request.input
    assert f"Stay within {length_cap('convention')} characters" in request.input
    assert "Count the leaves." not in request.input


def test_a_problem_text_reaches_the_prompt_unrendered(minibatch):
    problems, probabilities, labels, weights = minibatch
    texts = ["{{ tag }} and {% if %}", *problems["statement"].tolist()[1:]]
    records = tag_records(
        "graphs", problems.assign(statement=texts), probabilities, labels, weights
    )

    request = reflection_request("graphs", SEED_WORDS, records, 0)

    assert "{{ tag }} and {% if %}" in request.input


def test_the_request_hash_follows_the_records(minibatch):
    one = reflection_request("graphs", SEED_WORDS, tag_records("graphs", *minibatch, k=1), 0)
    two = reflection_request("graphs", SEED_WORDS, tag_records("graphs", *minibatch, k=2), 0)

    assert one.sha256() != two.sha256()


def test_a_request_refuses_an_unknown_piece_or_records_without_their_summary(minibatch):
    records = tag_records("graphs", *minibatch)

    with pytest.raises(ValueError, match="'dp' is not a piece"):
        reflection_request("dp", SEED_WORDS, records, 0)
    with pytest.raises(ValueError, match="must start with its summary"):
        reflection_request("trees", SEED_WORDS, records, 0)


@pytest.mark.parametrize("schema_name", ["tag_criteria", "convention"])
def test_the_schemas_are_strict(schema_name):
    path = TEMPLATE_DIR / f"{schema_name}.schema.json"
    nodes = [json.loads(path.read_text(encoding="utf-8"))]
    while nodes:
        node = nodes.pop()
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
        nodes.extend(child for child in node["properties"].values() if child["type"] == "object")


def graphs_with(item):
    """Return the seed's graphs criteria with one more include."""
    criteria = copy.deepcopy(TAG_DEFINITIONS["graphs"])
    criteria["true"]["includes"].append(item)
    return criteria


def luna(make_answer, value, **changes):
    """Return a completed Luna answer whose text is the JSON of ``value``."""
    return make_answer(**({"output_text": json.dumps(value)} | changes))


def test_a_well_formed_proposal_is_accepted_as_its_piece_text(make_answer):
    criteria = graphs_with("grids searched cell by cell")

    proposal = read_proposal("graphs", luna(make_answer, criteria), [STORY])

    assert proposal.reason is None
    assert proposal.text == json.dumps(criteria, ensure_ascii=False, indent=2)
    assert parse_piece("graphs", proposal.text) == criteria


def test_a_convention_proposal_is_its_text(make_answer):
    answer = luna(make_answer, {"convention": "Tags name the intended solution."})

    assert read_proposal("convention", answer, []) == Proposal(
        "convention", "Tags name the intended solution.", None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "incomplete"},
        {"status": "refused", "output_text": None},
        {"output_text": None},
        {"failed": True},
    ],
)
def test_a_failed_answer_is_refused(make_answer, changes):
    answer = luna(make_answer, TAG_DEFINITIONS["graphs"], **changes)

    assert read_proposal("graphs", answer, []) == Proposal("graphs", None, "failed_answer")


def without(side, key):
    """Return the seed's graphs criteria with one key of one side removed."""
    criteria = copy.deepcopy(TAG_DEFINITIONS["graphs"])
    del criteria[side][key]
    return criteria


@pytest.mark.parametrize(
    ("piece", "output_text"),
    [
        ("graphs", "{not json"),
        ("graphs", json.dumps(["a", "list"])),
        ("graphs", json.dumps(TAG_DEFINITIONS["graphs"] | {"notes": "x"})),
        ("graphs", json.dumps({"true": TAG_DEFINITIONS["graphs"]["true"]})),
        ("graphs", json.dumps(without("false", "excludes"))),
        ("graphs", json.dumps(graphs_with(3))),
        ("graphs", json.dumps({"true": "yes", "false": "no"})),
        (
            "graphs",
            json.dumps({**TAG_DEFINITIONS["graphs"], "true": {"definition": 1, "includes": ["a"]}}),
        ),
        (
            "graphs",
            json.dumps({**TAG_DEFINITIONS["graphs"], "true": {"definition": "a", "includes": "b"}}),
        ),
        ("convention", json.dumps({"text": "Tags name ideas."})),
        ("convention", json.dumps({"convention": 3})),
    ],
)
def test_an_answer_of_another_shape_is_refused_as_invalid_json(make_answer, piece, output_text):
    answer = make_answer(output_text=output_text)

    assert read_proposal(piece, answer, []) == Proposal(piece, None, "invalid_json")


@pytest.mark.parametrize(
    ("piece", "value"),
    [
        ("graphs", {**TAG_DEFINITIONS["graphs"], "false": {"definition": " ", "excludes": ["a"]}}),
        ("graphs", {**TAG_DEFINITIONS["graphs"], "true": {"definition": "A.", "includes": []}}),
        ("graphs", graphs_with("  ")),
        ("convention", {"convention": "\n"}),
    ],
)
def test_a_blank_text_or_an_empty_list_is_refused(make_answer, piece, value):
    answer = luna(make_answer, value)

    assert read_proposal(piece, answer, []) == Proposal(piece, None, "empty_field")


@pytest.mark.parametrize(
    ("piece", "value"),
    [
        ("graphs", graphs_with("a" * 400)),
        ("convention", {"convention": "x" * (length_cap("convention") + 1)}),
    ],
)
def test_a_proposal_over_its_length_cap_is_refused(make_answer, piece, value):
    assert read_proposal(piece, luna(make_answer, value), []) == Proposal(piece, None, "too_long")


def test_a_proposal_at_its_length_cap_is_accepted(make_answer):
    value = {"convention": "x" * length_cap("convention")}

    assert read_proposal("convention", luna(make_answer, value), []).reason is None


@pytest.mark.parametrize(
    ("piece", "value"),
    [
        ("graphs", graphs_with(STORY_ITEM)),
        ("convention", {"convention": f"Tags name ideas, as {STORY_ITEM}."}),
    ],
)
def test_a_proposal_that_copies_a_shown_text_is_refused(make_answer, piece, value):
    answer = luna(make_answer, value)

    assert read_proposal(piece, answer, ["Story.", STORY]) == Proposal(piece, None, "copied_text")
    assert read_proposal(piece, answer, ["Story."]).reason is None  # only shown texts count


def test_a_sequence_of_the_seed_words_is_no_copy(make_answer):
    answer = luna(make_answer, graphs_with("grids searched cell by cell"))
    shown = [f"Here, {SEED_ITEM}, as the notes say."]

    assert read_proposal("graphs", answer, shown).reason is None


def test_read_proposal_refuses_an_unknown_piece(make_answer):
    with pytest.raises(ValueError, match="'dp' is not a piece"):
        read_proposal("dp", make_answer(), [])


def reflection(sample_id):
    """Return a reflection request of the search, with an input of its own."""
    return LlmRequest(
        sample_id, GEPA_NAME, f"Rewrite {sample_id}.", REFLECTION_SETTINGS, None, None
    )


@pytest.fixture(name="fake_luna")
def fake_luna_fixture(make_answer):
    """Return a fake Luna ``Ask`` that answers each request, and the sample ids it was asked."""
    asked = []

    async def ask(request):
        asked.append(request.sample_id)
        return make_answer(
            request_sha256=request.sha256(), sample_id=request.sample_id, method=GEPA_NAME
        )

    return ask, asked


def test_a_new_request_is_sent_once_and_appended(tmp_path, fake_luna):
    ask, asked = fake_luna
    path = tmp_path / "reflections.jsonl"
    reflector = Reflector(path, ask=ask)

    answer = reflector.answer(reflection(4))

    assert asked == [4]
    assert reflector.sent == 1
    assert answer.request_sha256 == reflection(4).sha256()
    assert read_answers(path) == [answer]


def test_a_cached_request_sends_nothing(tmp_path, monkeypatch, fake_luna):
    ask, asked = fake_luna
    path = tmp_path / "reflections.jsonl"
    first = Reflector(path, ask=ask).answer(reflection(4))
    built = []
    monkeypatch.setattr("codeforces_tags.llm.client.ask_for", built.append)

    replay = Reflector(path)

    assert replay.answer(reflection(4)) == first
    assert replay.sent == 0
    assert not built  # Luna was never even opened
    assert asked == [4]
    assert len(read_answers(path)) == 1


def test_a_transient_error_writes_no_line_and_the_next_call_sends_again(tmp_path, fake_luna):
    ask, asked = fake_luna
    path = tmp_path / "reflections.jsonl"

    async def down(request):
        raise TransientApiError(f"no answer to {request.sample_id}")

    with pytest.raises(TransientApiError, match="no answer to 4"):
        Reflector(path, ask=down).answer(reflection(4))
    assert not path.exists()

    Reflector(path, ask=ask).answer(reflection(4))
    assert asked == [4]
    assert len(read_answers(path)) == 1


def test_an_answer_under_another_hash_writes_no_line(tmp_path, make_answer):
    path = tmp_path / "reflections.jsonl"

    async def confused(request):
        return make_answer(sample_id=request.sample_id, method=GEPA_NAME)

    with pytest.raises(ValueError, match="not by its request's hash"):
        Reflector(path, ask=confused).answer(reflection(4))
    assert not path.exists()


def test_total_cost_sums_every_cached_answer(tmp_path, fake_luna):
    ask, _ = fake_luna
    path = tmp_path / "reflections.jsonl"
    assert Reflector(path, ask=ask).total_cost == 0.0

    Reflector(path, ask=ask).answer(reflection(1))
    reflector = Reflector(path, ask=ask)
    reflector.answer(reflection(2))

    assert reflector.total_cost == pytest.approx(2 * ANSWER_COST)


def test_without_an_ask_luna_is_opened_when_the_first_request_is_sent(
    tmp_path, monkeypatch, make_answer
):
    built, events = [], []

    class OpenableLuna:  # pylint: disable=too-few-public-methods
        """An adapter's Ask: ``open`` opens a client in the running loop."""

        @asynccontextmanager
        async def open(self):
            """Open the client, give the Ask that sends through it, then close it."""
            events.append("open")

            async def send(request):
                events.append(request.sample_id)
                return make_answer(request_sha256=request.sha256(), sample_id=request.sample_id)

            yield send
            events.append("close")

    def ask_for(settings):
        built.append(settings)
        return OpenableLuna()

    monkeypatch.setattr("codeforces_tags.llm.client.ask_for", ask_for)
    reflector = Reflector(tmp_path / "reflections.jsonl")
    assert not built

    reflector.answer(reflection(7))
    reflector.answer(reflection(7))

    assert built == [REFLECTION_SETTINGS]
    assert events == ["open", 7, "close"]
