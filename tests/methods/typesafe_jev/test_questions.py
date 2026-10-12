"""Tests for the questions, states and hashed requests of the TypeSafe Jev methods."""

import copy
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from codeforces_tags import methods
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.llm.store import check_complete, check_same_requests, requests_to_send
from codeforces_tags.methods.typesafe_jev import settings
from codeforces_tags.methods.typesafe_jev.questions import (
    JevRequest,
    noul_questions,
    question_id,
    requests,
    states,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    CONVENTION,
    FROZEN_NAMES,
    GEPA_NAME,
    MODEL,
    QUESTION,
    STATE_FIELDS,
    TAG_DEFINITIONS,
)
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS, words_from_json
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The statements that raw_to_problems builds for the three problems of the raw_problems fixture:
# the description, the input specification and the output specification, apart from the notes.
STATEMENT_30 = "Count the leaves of a tree.\n\nA tree.\n\nOne integer."
STATEMENT_10 = "Reverse each of the strings.\n\nSome strings.\n\nThe strings."
STATEMENT_20 = "Print the gcd, a number theory classic.\n\nOne integer."
# The sha256 of the seed questions as the SDK sends them (the compact UTF-8 JSON of
# noul_questions()), and of the requests of the raw_problems fixture (problems 30, 10, 20):
# computed before Jev's words became data and before they moved to the prompt file jev-tags v1,
# and never changed, since the committed answers files answer requests of these bytes. They are
# in every request's hash, so a change is a new method under a new name (docs/adr/0003).
SEED_QUESTIONS_SHA256 = "e3471bb5aa2af5fbd4ee9ca547f79a541b62cda72baf81918a78a602e8cd5cc4"
SEED_REQUESTS_SHA256 = {
    "jev-statement": [
        "ce8192e307b958046bee4b20ef97d8f565756ca98e738bf0a606d796b64a0c38",
        "232254ba40b0987e09d980beaf34e7462aca5b47f405f319e244409faca84040",
        "aeadda75977d7292a6d2dc6bc5647dd38e49fb6073a34d5fdeafec328b4d7780",
    ],
    "jev-statement-code": [
        "012fe0de2847e5e70ef8fca5f987b943db25685dc0aba91492891e11d9819a8c",
        "7bb52d8e77fb99a72935e4ca71578db84932f306d76a4ea4cd3f63c1d07f074f",
        "cdc6ffc9ac6cb516731bea452230cd4984e5ff96d2e448df5a8cea619cdbe575",
    ],
}


def wire_sha256(value):
    """Hash a value as the SDK puts it on the wire: compact JSON in its key order, UTF-8 kept."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def other_words():
    """Return the seed words with another convention, question and math yes definition."""
    values = SEED_WORDS.settings()
    values["convention"] = "Tags name the ideas of a solution."
    values["question"] = "Is `{tag}` a tag of this problem?"
    values["tag_definitions"]["math"]["true"]["definition"] = "A formula does the work."
    return words_from_json(values)


def test_question_ids_replace_spaces_and_are_distinct():
    ids = [question_id(tag) for tag in FOCUS_TAGS]

    assert question_id("number theory") == "number_theory"
    assert ids == [
        "math",
        "graphs",
        "strings",
        "number_theory",
        "trees",
        "geometry",
        "games",
        "probabilities",
    ]
    assert len(set(ids)) == len(FOCUS_TAGS)


def test_the_questions_hold_every_focus_tag_once_in_order():
    asked = noul_questions()

    assert list(asked) == [question_id(tag) for tag in FOCUS_TAGS]
    assert [question["instructions"]["tag"] for question in asked.values()] == list(FOCUS_TAGS)


def test_each_question_asks_whether_codeforces_would_put_its_tag_with_its_definition():
    asked = noul_questions()

    assert asked["number_theory"] == {
        "type": "noul",
        "instructions": {
            "tag": "number theory",
            "convention": CONVENTION,
            "question": "Would Codeforces put the tag `number theory` on this problem?",
        },
        "criteria": TAG_DEFINITIONS["number theory"],
    }
    # A dict equal to another is not necessarily in its order, and the order is in the hash.
    for tag, question in zip(FOCUS_TAGS, asked.values(), strict=True):
        assert list(question) == ["type", "instructions", "criteria"], tag
        assert list(question["instructions"]) == ["tag", "convention", "question"], tag
        assert question["instructions"]["question"] == QUESTION.format(tag=tag), tag
        assert list(question["criteria"]) == ["true", "false"], tag
        assert question["criteria"] == TAG_DEFINITIONS[tag], tag
        assert list(question["criteria"]["true"]) == ["definition", "includes"], tag
        assert list(question["criteria"]["false"]) == ["definition", "excludes"], tag


def test_the_questions_are_the_ones_every_committed_request_holds():
    assert wire_sha256(noul_questions()) == SEED_QUESTIONS_SHA256


def test_noul_questions_are_fresh_copies():
    frozen = copy.deepcopy(TAG_DEFINITIONS)
    asked = noul_questions()

    asked["math"]["criteria"]["true"]["includes"].append("a phrase nobody wrote")
    asked["math"]["criteria"]["false"]["definition"] = "Another definition."
    asked["math"]["instructions"]["tag"] = "another tag"

    assert frozen == TAG_DEFINITIONS
    assert noul_questions()["math"]["criteria"] == frozen["math"]
    assert noul_questions()["math"]["instructions"]["tag"] == "math"
    assert noul_questions()["math"] is not noul_questions()["math"]


def test_states_hold_the_fields_of_each_method_in_order(raw_problems):
    problems = raw_to_problems(raw_problems)

    statements = states(problems, "jev-statement")
    with_code = states(problems, "jev-statement-code")

    assert statements.index.equals(problems.index) and statements.index.name == "sample_id"
    assert with_code.index.equals(problems.index)
    assert statements.dtype == object and with_code.dtype == object
    assert statements[30] == {"statement": STATEMENT_30}
    assert list(with_code[10]) == ["statement", "solution"]
    assert with_code[10] == {"statement": STATEMENT_10, "solution": "s = input()\nprint(s[::-1])\n"}
    assert with_code[20] == {"statement": STATEMENT_20, "solution": "print(2)"}


def test_states_of_no_problem_are_an_empty_series_of_objects(raw_problems):
    problems = raw_to_problems(raw_problems).iloc[:0]

    empty = states(problems, "jev-statement-code")

    assert len(empty) == 0 and empty.dtype == object and empty.index.equals(problems.index)


@pytest.mark.parametrize("solution", [np.nan, "", " \n"])
def test_states_refuse_a_problem_without_its_solution(raw_problems, solution):
    problems = raw_to_problems(raw_problems.assign(source_code=["print(1)", solution, "print(2)"]))

    with pytest.raises(ValueError, match=r"missing or blank: the solution of the problems \[10\]"):
        states(problems, "jev-statement-code")
    assert states(problems, "jev-statement")[10] == {"statement": STATEMENT_10}


def test_states_name_every_field_and_problem_that_lacks_its_text(raw_problems):
    blank = raw_problems.assign(
        prob_desc_description=["", "Reverse each of the strings.", "Print the gcd."],
        prob_desc_input_spec=["", "Some strings.", "One integer."],
        prob_desc_output_spec=["", "The strings.", "One integer."],
        source_code=["print(1)", "print(2)", np.nan],
    )

    with pytest.raises(ValueError) as refused:
        states(raw_to_problems(blank), "jev-statement-code")

    assert str(refused.value).endswith(
        "missing or blank: the statement of the problems [30]; the solution of the problems [20]."
    )


@pytest.mark.parametrize("name", ["jev-latest", "label_prior", ""])
def test_states_refuse_an_unknown_method(raw_problems, name):
    with pytest.raises(ValueError, match="Unknown Jev method"):
        states(raw_to_problems(raw_problems), name)


@pytest.mark.parametrize("name", FROZEN_NAMES)
def test_requests_keep_the_input_order_with_int_sample_ids(raw_problems, name):
    asked = requests(raw_problems, name)

    assert [request.sample_id for request in asked] == [30, 10, 20]
    assert {type(request.sample_id) for request in asked} == {int}
    assert {request.model for request in asked} == {MODEL}
    assert [list(request.state) for request in asked] == [list(STATE_FIELDS[name])] * 3
    # One mapping of questions per call: 9,964 copies of the definitions would cost 100 MB.
    assert {id(request.questions) for request in asked} == {id(asked[0].questions)}
    assert requests(raw_problems, name)[0].questions is not asked[0].questions
    assert requests(raw_problems.iloc[:0], name) == []


def test_the_two_methods_ask_the_same_questions(raw_problems):
    statement_requests = requests(raw_problems, "jev-statement")
    code_requests = requests(raw_problems, "jev-statement-code")

    pairs = list(zip(statement_requests, code_requests, strict=True))
    assert all(plain.questions == coded.questions for plain, coded in pairs)
    assert all(plain.state != coded.state for plain, coded in pairs)
    assert all(plain.sha256() != coded.sha256() for plain, coded in pairs)


def test_the_body_is_in_the_sdks_wire_order(raw_problems):
    request = requests(raw_problems, "jev-statement-code")[0]

    body = request.body()

    assert list(body) == ["state", "model", "questions"]
    assert body["state"] is request.state and body["questions"] is request.questions
    assert body["model"] == MODEL


def test_the_hash_is_the_sha256_of_the_compact_utf8_body():
    ascii_request = JevRequest(
        1, "jev-1.13.0", {"statement": "Print the gcd of a and b."}, {"math": {"type": "noul"}}
    )
    # Two accented letters (e acute, U+00E9) as escapes, so that this file stays ASCII.
    accented = "Caf\u00e9: print \u00e9."
    accent_request = dataclasses.replace(ascii_request, state={"statement": accented})
    ascii_text = (
        '{"state":{"statement":"Print the gcd of a and b."},"model":"jev-1.13.0",'
        '"questions":{"math":{"type":"noul"}}}'
    )
    accent_text = ascii_text.replace("Print the gcd of a and b.", accented)

    assert ascii_request.sha256() == hashlib.sha256(ascii_text.encode("utf-8")).hexdigest()
    assert accent_request.sha256() == hashlib.sha256(accent_text.encode("utf-8")).hexdigest()
    # The pinned digests: non-ASCII text is kept as UTF-8, as the SDK sends it.
    assert ascii_request.sha256() == (
        "e3e1fb06f7cf380ff60c8827d7f418908538ee7e0d171c22adaf0ac087e01539"
    )
    assert accent_request.sha256() == (
        "720be7b208511534b0a3541db59ecc50c56e516dc3911ec08575dc3754e6bec7"
    )


def test_the_hash_changes_with_the_model_the_state_a_word_of_a_definition_and_the_key_order(
    raw_problems, monkeypatch
):
    request = requests(raw_problems, "jev-statement")[0]
    math = request.questions["math"]
    swapped_math = {
        "type": "noul",
        "criteria": math["criteria"],
        "instructions": math["instructions"],
    }
    variants = {
        "model": dataclasses.replace(request, model="jev-1.14.0"),
        "state": dataclasses.replace(
            request, state={"statement": request.state["statement"] + "."}
        ),
        "order of the questions": dataclasses.replace(
            request, questions=dict(reversed(request.questions.items()))
        ),
        "order inside a question": dataclasses.replace(
            request, questions={**request.questions, "math": swapped_math}
        ),
    }
    monkeypatch.setitem(TAG_DEFINITIONS["math"]["true"], "definition", "Another definition.")
    variants["word of a definition"] = requests(raw_problems, "jev-statement")[0]

    hashes = {change: variant.sha256() for change, variant in variants.items()}

    assert request.sha256() not in hashes.values()
    assert len(set(hashes.values())) == len(variants), hashes
    # Equal dicts in another order are equal for ==, and still another request.
    assert variants["order of the questions"].questions == request.questions


def test_the_hash_ignores_the_transport_settings(raw_problems, monkeypatch):
    before = [request.sha256() for request in requests(raw_problems, "jev-statement-code")]

    for name, value in (
        ("TIMEOUT_S", 1.0),
        ("MAX_RETRIES", 0),
        ("CONCURRENCY", 1),
        ("MAX_COST_USD", 0.01),
    ):
        monkeypatch.setattr(settings, name, value)

    after = [request.sha256() for request in requests(raw_problems, "jev-statement-code")]
    assert after == before


def test_requests_refuse_preprocessed_problems(raw_problems):
    with pytest.raises(ValueError, match="already preprocessed"):
        requests(raw_to_problems(raw_problems), "jev-statement")


def test_requests_refuse_problems_without_a_raw_column(raw_problems):
    with pytest.raises(ValueError, match=r"lack the columns \['source_code'\]"):
        requests(raw_problems.drop(columns="source_code"), "jev-statement")


def test_the_answers_store_takes_jev_requests(raw_problems, jev_line):
    asked = requests(raw_problems, "jev-statement")
    answers = [jev_line(request) for request in asked[:2]]

    assert requests_to_send(asked, answers) == [asked[2]]
    with pytest.raises(ValueError, match=r"no answer yet for the problems \[20\]"):
        check_complete(asked, answers)
    check_complete(asked, [*answers, jev_line(asked[2])])
    # The answers to the statements are not to the requests that add the solutions.
    with pytest.raises(ValueError, match=r"problems \[10, 20, 30\] are not the ones that were"):
        check_same_requests(
            requests(raw_problems, "jev-statement-code"), [*answers, jev_line(asked[2])]
        )


def test_the_seed_words_ask_the_bytes_of_the_frozen_requests(raw_problems):
    assert wire_sha256(noul_questions(SEED_WORDS)) == SEED_QUESTIONS_SHA256
    for name in ("jev-statement", "jev-statement-code"):
        built = [request.sha256() for request in requests(raw_problems, name)]
        seeded = [request.sha256() for request in requests(raw_problems, name, SEED_WORDS)]
        assert built == seeded == SEED_REQUESTS_SHA256[name], name


def test_noul_questions_hold_the_words_they_are_given():
    words = other_words()

    asked = noul_questions(words)

    assert asked["math"]["instructions"] == {
        "tag": "math",
        "convention": "Tags name the ideas of a solution.",
        "question": "Is `math` a tag of this problem?",
    }
    assert asked["math"]["criteria"]["true"]["definition"] == "A formula does the work."
    assert asked["math"]["criteria"]["false"] == TAG_DEFINITIONS["math"]["false"]
    assert asked["graphs"]["criteria"] == TAG_DEFINITIONS["graphs"]
    # The questions are copies: changing one changes neither the words nor the next questions.
    asked["math"]["criteria"]["true"]["includes"].append("a phrase nobody wrote")
    assert noul_questions(words)["math"]["criteria"] == words.tag_definitions["math"]


def test_requests_with_other_words_are_other_requests_of_the_same_states(raw_problems):
    words = other_words()

    seeded = requests(raw_problems, "jev-statement-code")
    rewritten = requests(raw_problems, "jev-statement-code", words)

    assert [request.state for request in rewritten] == [request.state for request in seeded]
    assert rewritten[0].questions == noul_questions(words)
    assert not {request.sha256() for request in rewritten} & set(
        SEED_REQUESTS_SHA256["jev-statement-code"]
    )


def test_the_searched_method_reads_what_its_seed_reads(raw_problems):
    problems = raw_to_problems(raw_problems)

    assert states(problems, GEPA_NAME).equals(states(problems, "jev-statement-code"))
    # With the seed words, its requests are the seed's, byte for byte: the search's seed costs
    # nothing, its answers are the committed ones of jev-statement-code.
    seeded = [request.sha256() for request in requests(raw_problems, GEPA_NAME, SEED_WORDS)]
    assert seeded == SEED_REQUESTS_SHA256["jev-statement-code"]


def test_the_searched_methods_requests_hold_the_words_of_its_frozen_params(
    raw_problems, monkeypatch, tmp_path
):
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="tuning/params.json does not exist"):
        requests(raw_problems, GEPA_NAME)
    path = tmp_path / GEPA_NAME / "tuning" / "params.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"words": other_words().settings()}), encoding="utf-8")

    asked = requests(raw_problems, GEPA_NAME)

    expected = requests(raw_problems, GEPA_NAME, other_words())
    assert [request.sha256() for request in asked] == [request.sha256() for request in expected]
    assert (
        asked[0].questions["math"]["instructions"]["question"] == "Is `math` a tag of this problem?"
    )
