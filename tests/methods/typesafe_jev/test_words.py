"""Tests for Jev's words as data: the seed words, the pieces, their lengths and copied text."""

import copy
import json
import math

import pytest

from codeforces_tags import methods
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.settings import (
    CONVENTION,
    FROZEN_NAMES,
    GEPA_NAME,
    MODEL,
    PLATT_BASES,
    PLATT_NAMES,
    PROMPT,
    QUESTION,
    SEARCH_LENGTH_FACTOR,
    SEARCH_NGRAM,
    SETTINGS,
    TAG_DEFINITIONS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    JevWords,
    base_of,
    candidate_from_words,
    copied_sequences,
    length_cap,
    method_settings,
    parse_piece,
    piece_length,
    piece_problem,
    piece_text,
    read_words,
    settings_entry,
    word_sequences,
    word_strings,
    words_from_candidate,
    words_from_json,
    words_of,
)
from codeforces_tags.model.platt_lipton import RULE_SETTINGS
from codeforces_tags.provenance import sha256_json
from codeforces_tags.templating import render_string

# sha256_json of the seed words' settings(): the words of the frozen methods, never changed.
SEED_WORDS_SHA256 = "08a795c15d143cbaa78638f38900eacdb05daa8d20dc98e6ad11bdbbfe783f75"
# The most characters each piece may hold: 1.5 times the seed's compact JSON (or text), floored.
LENGTH_CAPS = {
    "convention": 225,
    "math": 1227,
    "graphs": 1030,
    "strings": 975,
    "number theory": 891,
    "trees": 987,
    "geometry": 783,
    "games": 970,
    "probabilities": 958,
}
# A statement, and an includes item that copies twelve of its words.
STATEMENT = "Given an array, count the number of pairs (i, j) such that their sum is even."
COPIED_ITEM = "count the number of pairs i j such that their sum is even"


def edited_words(edit):
    """Return the seed words after ``edit`` changed their JSON values in place."""
    values = SEED_WORDS.settings()
    edit(values)
    return words_from_json(values)


def changed_math(edit):
    """Return a copy of the math piece's criteria after ``edit`` changed it."""
    criteria = copy.deepcopy(TAG_DEFINITIONS["math"])
    edit(criteria)
    return criteria


def edited_math(edit):
    """Return the text of the math piece after ``edit`` changed a copy of its criteria."""
    return json.dumps(changed_math(edit))


def blank_and_mistyped(criteria):
    """Blank the yes definition of criteria and make their excludes a text, not a list."""
    criteria["true"]["definition"] = " "
    criteria["false"]["excludes"] = "a single text"


def test_the_seed_words_are_the_frozen_settings_and_share_their_definitions():
    # The frozen methods' requests are built from SEED_WORDS: it holds the settings themselves.
    assert (SEED_WORDS.convention, SEED_WORDS.question) == (CONVENTION, QUESTION)
    assert SEED_WORDS.tag_definitions is TAG_DEFINITIONS
    assert SEED_WORDS.settings() == {
        key: SETTINGS[key] for key in ("convention", "question", "tag_definitions")
    }
    assert SEED_WORDS.sha256() == SEED_WORDS_SHA256


def test_the_seed_words_are_read_from_the_prompt_file_jev_tags_version_1():
    # The words are defined once, in templates/jev-tags.v1.yaml: the seed holds what it read.
    assert SEED_WORDS.convention is PROMPT.data["convention"]
    assert SEED_WORDS.tag_definitions is PROMPT.data["tag_definitions"]
    for tag in FOCUS_TAGS:
        asked = render_string(PROMPT.data["question"], tag=tag)
        assert SEED_WORDS.question.format(tag=tag) == asked, tag


def test_settings_are_plain_copies_in_request_order():
    values = SEED_WORDS.settings()

    values["tag_definitions"]["math"]["true"]["includes"].append("a phrase nobody wrote")

    assert list(values) == ["convention", "question", "tag_definitions"]
    assert list(values["tag_definitions"]) == list(FOCUS_TAGS)
    assert "a phrase nobody wrote" not in TAG_DEFINITIONS["math"]["true"]["includes"]
    assert SEED_WORDS.sha256() == SEED_WORDS_SHA256


def test_the_pieces_are_the_convention_then_the_focus_tags():
    assert ("convention", *FOCUS_TAGS) == PIECES
    assert "number theory" in PIECES
    assert "question" not in PIECES


def test_a_tag_piece_is_its_criteria_as_indented_json_in_request_order():
    text = piece_text(SEED_WORDS, "probabilities")

    assert text == json.dumps(TAG_DEFINITIONS["probabilities"], ensure_ascii=False, indent=2)
    assert text.startswith('{\n  "true": {\n    "definition": ')
    assert f"p{chr(0xB7)}q^-1" in text  # a middle dot: non-ASCII text is kept, not escaped
    assert piece_text(SEED_WORDS, "convention") == CONVENTION


@pytest.mark.parametrize("piece", PIECES)
def test_each_piece_round_trips(piece):
    value = CONVENTION if piece == "convention" else TAG_DEFINITIONS[piece]

    parsed = parse_piece(piece, piece_text(SEED_WORDS, piece))

    assert parsed == value
    assert json.dumps(parsed) == json.dumps(value)  # the same key and item order


def test_the_seed_candidate_gives_back_the_seed_words():
    candidate = candidate_from_words(SEED_WORDS)

    assert list(candidate) == list(PIECES)
    assert words_from_candidate(candidate) == SEED_WORDS
    assert words_from_json(SEED_WORDS.settings()) == SEED_WORDS
    assert words_from_json(json.loads(json.dumps(SEED_WORDS.settings()))) == SEED_WORDS


def test_a_rewritten_candidate_gives_its_words_and_round_trips():
    candidate = candidate_from_words(SEED_WORDS) | {"convention": "Tags name the ideas."}
    criteria = copy.deepcopy(TAG_DEFINITIONS["games"])
    criteria["false"]["excludes"] = ["a puzzle for one player"]
    candidate["games"] = json.dumps(criteria, indent=4)  # any indentation reads the same

    words = words_from_candidate(candidate, question="Is `{tag}` a tag of it?")

    assert words.convention == "Tags name the ideas."
    assert words.question == "Is `{tag}` a tag of it?"
    assert words.tag_definitions["games"] == criteria
    assert words.tag_definitions["math"] == TAG_DEFINITIONS["math"]
    assert words_from_candidate(candidate_from_words(words), words.question) == words
    assert words.sha256() != SEED_WORDS_SHA256


def test_parse_piece_puts_the_keys_in_request_order():
    criteria = TAG_DEFINITIONS["trees"]
    shuffled = {
        "false": {"excludes": criteria["false"]["excludes"], "definition": "No tree."},
        "true": {"includes": criteria["true"]["includes"], "definition": "A tree."},
    }

    parsed = parse_piece("trees", json.dumps(shuffled))

    assert list(parsed) == ["true", "false"]
    assert list(parsed["true"]) == ["definition", "includes"]
    assert list(parsed["false"]) == ["definition", "excludes"]
    assert parsed["true"]["includes"] == criteria["true"]["includes"]


@pytest.mark.parametrize(
    ("piece", "text", "message"),
    [
        ("dp", "{}", r"'dp' is not a piece of Jev's words"),
        ("convention", 3, r"The piece 'convention' must be a text, not a int"),
        ("convention", " \n", r"The piece 'convention' is blank"),
        ("math", "the math tag", r"The piece 'math' is not JSON"),
        ("math", "[]", r"'math' holds exactly the keys \['true', 'false'\], not list"),
        (
            "math",
            edited_math(lambda criteria: criteria.pop("false")),
            r"'math' holds exactly the keys \['true', 'false'\], not \['true'\]",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria.update(maybe={})),
            r"not \['false', 'maybe', 'true'\]",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["false"].update(includes=["x"])),
            r"\['definition', 'excludes'\] under 'false', not \['definition', 'excludes', 'inc",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria.update(true="yes")),
            r"\['definition', 'includes'\] under 'true', not str",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["true"].update(definition=3)),
            r"'math' needs a definition under 'true' that is a text, not blank",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["false"].update(definition="  ")),
            r"'math' needs a definition under 'false' that is a text, not blank",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["true"].update(includes=[])),
            r"'math' needs under 'true' a non-empty list of texts in 'includes'",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["true"].update(includes="sums")),
            r"'math' needs under 'true' a non-empty list of texts in 'includes'",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["false"]["excludes"].append(" ")),
            r"'math' needs under 'false' a non-empty list of texts in 'excludes'",
        ),
        (
            "math",
            edited_math(lambda criteria: criteria["false"]["excludes"].append(7)),
            r"'math' needs under 'false' a non-empty list of texts in 'excludes'",
        ),
    ],
)
def test_parse_piece_refuses_any_other_shape_and_names_the_piece(piece, text, message):
    with pytest.raises(ValueError, match=message):
        parse_piece(piece, text)


@pytest.mark.parametrize(
    ("piece", "value", "reason"),
    [
        ("convention", 3, "invalid_json"),
        ("convention", None, "invalid_json"),
        ("math", [], "invalid_json"),
        ("math", "the math tag", "invalid_json"),  # a text, not the criteria it would decode to
        ("math", changed_math(lambda criteria: criteria.pop("false")), "invalid_json"),
        ("math", changed_math(lambda criteria: criteria.update(true="yes")), "invalid_json"),
        (
            "math",
            changed_math(lambda criteria: criteria["false"].update(includes=["x"])),
            "invalid_json",
        ),
        (
            "math",
            changed_math(lambda criteria: criteria["true"].update(definition=3)),
            "invalid_json",
        ),
        (
            "math",
            changed_math(lambda criteria: criteria["true"].update(includes="sums")),
            "invalid_json",
        ),
        (
            "math",
            changed_math(lambda criteria: criteria["false"]["excludes"].append(7)),
            "invalid_json",
        ),
        ("convention", " \n", "empty_field"),
        (
            "math",
            changed_math(lambda criteria: criteria["false"].update(definition="  ")),
            "empty_field",
        ),
        (
            "math",
            changed_math(lambda criteria: criteria["true"].update(includes=[])),
            "empty_field",
        ),
        (
            "math",
            changed_math(lambda criteria: criteria["false"]["excludes"].append(" ")),
            "empty_field",
        ),
        ("math", changed_math(blank_and_mistyped), "invalid_json"),  # a wrong type comes first
    ],
)
def test_piece_problem_names_the_reason_a_value_is_not_of_its_pieces_shape(piece, value, reason):
    assert piece_problem(piece, value) == reason


@pytest.mark.parametrize("piece", PIECES)
def test_piece_problem_finds_nothing_wrong_with_the_seed_pieces(piece):
    value = CONVENTION if piece == "convention" else TAG_DEFINITIONS[piece]

    assert piece_problem(piece, value) is None
    assert piece_problem(piece, parse_piece(piece, piece_text(SEED_WORDS, piece))) is None


def test_parse_piece_raises_on_the_problem_that_piece_problem_names():
    # Blank and mistyped at once: the message is the wrong type's, the reason invalid_json.
    with pytest.raises(ValueError, match=r"under 'false' a non-empty list of texts in 'excludes'"):
        parse_piece("math", edited_math(blank_and_mistyped))


def test_piece_problem_refuses_an_unknown_piece():
    with pytest.raises(ValueError, match="'question' is not a piece of Jev's words"):
        piece_problem("question", QUESTION)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (lambda candidate: candidate.pop("games"), r"not \['convention', 'geometry', 'graphs'"),
        (lambda candidate: candidate.update(question="Q"), r"holds exactly the pieces"),
    ],
)
def test_words_from_candidate_needs_exactly_the_pieces(changed, message):
    candidate = candidate_from_words(SEED_WORDS)
    changed(candidate)

    with pytest.raises(ValueError, match=message):
        words_from_candidate(candidate)


def test_a_piece_is_measured_as_compact_json_whatever_its_indentation():
    criteria = TAG_DEFINITIONS["probabilities"]
    compact = json.dumps(criteria, ensure_ascii=False, separators=(",", ":"))

    assert piece_length("probabilities", criteria) == len(compact) == 639
    assert piece_length("probabilities", piece_text(SEED_WORDS, "probabilities")) == 639
    assert piece_length("probabilities", json.dumps(criteria, indent=7)) == 639
    assert piece_length("convention", CONVENTION) == len(CONVENTION) == 150


def test_the_length_caps_are_one_and_a_half_times_the_seed_pieces():
    caps = {piece: length_cap(piece) for piece in PIECES}

    assert caps == LENGTH_CAPS
    assert SEARCH_LENGTH_FACTOR == 1.5
    for piece in FOCUS_TAGS:
        assert caps[piece] == math.floor(1.5 * piece_length(piece, TAG_DEFINITIONS[piece]))


@pytest.mark.parametrize("measure", [length_cap, lambda piece: piece_length(piece, "x")])
def test_length_cap_and_piece_length_refuse_an_unknown_piece(measure):
    with pytest.raises(ValueError, match="'question' is not a piece of Jev's words"):
        measure("question")


def test_piece_length_refuses_a_piece_of_another_shape():
    with pytest.raises(ValueError, match="'convention' is blank"):
        piece_length("convention", "")
    with pytest.raises(ValueError, match=r"'graphs' holds exactly the keys"):
        piece_length("graphs", {"true": {}})


def test_words_from_json_reads_the_definitions_in_focus_order():
    values = SEED_WORDS.settings()
    values["tag_definitions"] = dict(reversed(values["tag_definitions"].items()))

    words = words_from_json(values)

    assert list(words.tag_definitions) == list(FOCUS_TAGS)
    assert words == SEED_WORDS


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda values: values.pop("question"), r"hold exactly the keys \['convention', 'quest"),
        (lambda values: values.update(model="jev-1.13.0"), r"not \['convention', 'model', "),
        (lambda values: values.update(question=7), r"must be a text that holds '\{tag\}'"),
        (lambda values: values.update(question="Is it a tag?"), r"a text that holds '\{tag\}'"),
        (
            lambda values: values["tag_definitions"].pop("games"),
            r"those of exactly the focus tags \[.*\], not \['geometry', 'graphs', 'math'",
        ),
        (lambda values: values.update(tag_definitions=[]), r"the focus tags \[.*\], not list"),
        (lambda values: values.update(convention=""), r"The piece 'convention' is blank"),
        (
            lambda values: values["tag_definitions"]["trees"]["true"].pop("includes"),
            r"'trees' holds exactly the keys \['definition', 'includes'\] under 'true'",
        ),
    ],
)
def test_words_from_json_refuses_words_of_another_shape(edit, message):
    with pytest.raises(ValueError, match=message):
        edited_words(edit)


def test_words_from_json_refuses_what_is_not_a_mapping():
    with pytest.raises(ValueError, match=r"hold exactly the keys .*, not list"):
        words_from_json([])


def test_word_strings_list_every_free_text_of_the_words():
    strings = word_strings(SEED_WORDS)

    sizes = [
        2 + len(criteria["true"]["includes"]) + len(criteria["false"]["excludes"])
        for criteria in TAG_DEFINITIONS.values()
    ]
    assert strings[:2] == [CONVENTION, QUESTION]
    assert strings[2] == TAG_DEFINITIONS["math"]["true"]["definition"]
    assert strings[-1] == TAG_DEFINITIONS["probabilities"]["false"]["excludes"][-1]
    assert len(strings) == 2 + sum(sizes)
    assert "Nim, Grundy values, the Sprague-Grundy theorem" in strings


def test_word_sequences_lower_case_and_split_on_what_is_not_a_letter_or_digit():
    assert SEARCH_NGRAM == 8
    assert word_sequences("Print the GCD_of a, b: x2!", 3) == {
        ("print", "the", "gcd"),
        ("the", "gcd", "of"),
        ("gcd", "of", "a"),
        ("of", "a", "b"),
        ("a", "b", "x2"),
    }
    accented = "Caf" + chr(0xE9)  # an e with an acute accent, a letter
    assert word_sequences(f"{accented} au lait", 3) == {(accented.lower(), "au", "lait")}
    assert word_sequences("Too short to hold eight words.") == set()


def test_the_seed_words_copy_nothing_even_from_their_own_text():
    assert copied_sequences(SEED_WORDS, [CONVENTION, *word_strings(SEED_WORDS)]) == []


def test_copied_sequences_find_the_word_sequences_shared_with_a_text():
    words = edited_words(
        lambda values: values["tag_definitions"]["math"]["true"]["includes"].append(COPIED_ITEM)
    )

    found = copied_sequences(words, [float("nan"), "Print the sum.", STATEMENT])

    assert found == [
        "count the number of pairs i j such",
        "i j such that their sum is even",
        "number of pairs i j such that their",
        "of pairs i j such that their sum",
        "pairs i j such that their sum is",
        "the number of pairs i j such that",
    ]
    assert copied_sequences(words, ["Print the sum."]) == []


def test_copied_sequences_exempt_the_seed_words_sequences_wherever_they_are():
    # The seed's convention, moved into a definition, is still the seed's hand-written words.
    words = edited_words(
        lambda values: values["tag_definitions"]["games"]["true"]["includes"].append(CONVENTION)
    )

    assert copied_sequences(words, [f"Statement. {CONVENTION}"]) == []


def test_copied_sequences_cut_each_free_text_on_its_own():
    items = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    words = edited_words(
        lambda values: values["tag_definitions"]["trees"]["false"].update(excludes=items)
    )
    text = "alpha beta gamma delta epsilon zeta eta theta"

    assert copied_sequences(words, [text]) == []  # no 8-word sequence runs across two items
    assert copied_sequences(words, [text], n=4) == items


def test_words_are_values_compared_by_their_words():
    same = JevWords(CONVENTION, QUESTION, copy.deepcopy(TAG_DEFINITIONS))

    assert same == SEED_WORDS
    assert same.sha256() == SEED_WORDS_SHA256
    assert JevWords("Other.", QUESTION, TAG_DEFINITIONS) != SEED_WORDS


def searched_words():
    """Return the seed words with another convention: the words of a searched method."""
    return edited_words(lambda values: values.update(convention="Tags name the solution's ideas."))


@pytest.mark.parametrize("name", FROZEN_NAMES)
def test_a_frozen_method_hashes_the_settings_it_was_frozen_with(name):
    assert method_settings(name, SEED_WORDS) is SETTINGS
    equal = JevWords(CONVENTION, QUESTION, copy.deepcopy(TAG_DEFINITIONS))  # a copy, equal
    assert method_settings(name, equal) is SETTINGS


@pytest.mark.parametrize("name", FROZEN_NAMES)
def test_a_frozen_method_refuses_other_words(name):
    with pytest.raises(
        ValueError, match=f"{name} sends the seed words, those of the prompt file templates/"
    ):
        method_settings(name, searched_words())


def test_the_searched_method_hashes_its_model_state_fields_and_words():
    words = searched_words()

    found = method_settings(GEPA_NAME, words)

    assert found == {
        "model": MODEL,
        "state_fields": {GEPA_NAME: ["statement", "solution"]},
        "convention": "Tags name the solution's ideas.",
        "question": QUESTION,
        "tag_definitions": words.settings()["tag_definitions"],
    }
    assert list(found) == list(SETTINGS)
    # Even with the seed words, its settings are not the frozen methods': its state fields differ.
    assert sha256_json(method_settings(GEPA_NAME, SEED_WORDS)) != sha256_json(SETTINGS)


@pytest.mark.parametrize(
    "call", [method_settings, lambda name, _: words_of(name), lambda name, _: base_of(name)]
)
def test_the_jev_name_functions_refuse_an_unknown_method(call):
    with pytest.raises(ValueError, match="Unknown Jev method 'label_prior'"):
        call("label_prior", SEED_WORDS)


def write_json(path, value):
    """Write a JSON value to a file, creating its folder; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_read_words_reads_the_params_of_a_search_or_the_words_alone(tmp_path):
    words = searched_words()
    params = write_json(tmp_path / "params.json", {"words": words.settings(), "chosen": 3})
    alone = write_json(tmp_path / "words.json", words.settings())

    assert read_words(params) == words
    assert read_words(alone) == words


def test_read_words_names_the_file_of_words_of_another_shape(tmp_path):
    values = searched_words().settings() | {"question": "Is it a tag?"}
    path = write_json(tmp_path / "words.json", values)

    with pytest.raises(ValueError, match=r"words\.json: The question of Jev's words must be"):
        read_words(path)


@pytest.mark.parametrize("name", FROZEN_NAMES)
def test_the_frozen_methods_send_the_seed_words(name):
    assert words_of(name) is SEED_WORDS


def test_the_searched_method_sends_the_words_of_its_frozen_params(tmp_path, monkeypatch):
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path)
    path = tmp_path / GEPA_NAME / "tuning" / "params.json"

    with pytest.raises(
        FileNotFoundError, match="typesafe_jev export --method jev-statement-code-gep"
    ):
        words_of(GEPA_NAME)
    write_json(path, {"words": searched_words().settings()})

    assert words_of(GEPA_NAME) == searched_words()


def test_base_of_names_a_twins_base_and_any_other_jev_method_itself():
    assert {twin: base_of(twin) for twin in PLATT_NAMES} == PLATT_BASES
    for name in (*FROZEN_NAMES, GEPA_NAME):
        assert base_of(name) == name


def test_the_twin_of_a_frozen_method_sends_the_seed_words():
    assert words_of("jev-statement-code-platt") is SEED_WORDS


def test_the_twin_of_the_searched_method_sends_the_words_of_its_bases_params(tmp_path, monkeypatch):
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=f"{GEPA_NAME}/tuning/params.json does not exist"):
        words_of("jev-statement-code-gepa-platt")
    write_json(
        tmp_path / GEPA_NAME / "tuning" / "params.json", {"words": searched_words().settings()}
    )

    assert words_of("jev-statement-code-gepa-platt") == searched_words() == words_of(GEPA_NAME)


@pytest.mark.parametrize(
    ("twin", "words"),
    [("jev-statement-code-platt", SEED_WORDS), ("jev-statement-code-gepa-platt", searched_words())],
)
def test_a_twin_hashes_its_base_its_bases_settings_and_its_threshold_rule(twin, words):
    base = PLATT_BASES[twin]

    found = method_settings(twin, words)

    assert found == {
        "base": base,
        "base_settings_sha256": sha256_json(method_settings(base, words)),
        "threshold_rule": RULE_SETTINGS,
    }
    assert list(found) == ["base", "base_settings_sha256", "threshold_rule"]
    assert found["threshold_rule"] is not RULE_SETTINGS  # a copy: the rule never changes
    # MLflow flattens a recipe's mappings by their dots: a dotted key would be one value.
    assert not [key for key in [*found, *found["threshold_rule"]] if "." in key]
    assert sha256_json(found) != sha256_json(method_settings(base, words))


def test_the_twin_of_a_frozen_method_hashes_the_frozen_settings_and_refuses_other_words():
    found = method_settings("jev-statement-code-platt", SEED_WORDS)

    assert found["base_settings_sha256"] == sha256_json(SETTINGS)
    with pytest.raises(ValueError, match="jev-statement-code sends the seed words"):
        method_settings("jev-statement-code-platt", searched_words())


def test_the_searched_twins_settings_follow_its_bases_words():
    other = edited_words(lambda values: values.update(convention="Tags name the setters' ideas."))

    found = method_settings("jev-statement-code-gepa-platt", searched_words())
    changed = method_settings("jev-statement-code-gepa-platt", other)

    assert found["base_settings_sha256"] != changed["base_settings_sha256"]
    assert found["threshold_rule"] == changed["threshold_rule"]


def test_a_run_records_settings_entry_names_the_model_the_fields_and_the_settings_hash():
    frozen = settings_entry("jev-statement-code")
    twin = settings_entry("jev-statement-code-platt")

    assert frozen == {
        "model": MODEL,
        "state_fields": ["statement", "solution"],
        "settings_sha256": sha256_json(SETTINGS),
    }
    # A twin sends its base's fields to the same model, under settings that hash its rule too.
    assert twin == frozen | {
        "settings_sha256": sha256_json(method_settings("jev-statement-code-platt", SEED_WORDS))
    }
    assert settings_entry("jev-statement")["state_fields"] == ["statement"]
    with pytest.raises(ValueError, match="Unknown Jev method 'tfidf'"):
        settings_entry("tfidf")
