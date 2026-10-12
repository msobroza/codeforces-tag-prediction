"""Tests for the settings of the TypeSafe Jev methods: their names, model, words and search."""

import hashlib
import json
from importlib.resources import files

import yaml

from codeforces_tags.config import FOCUS_TAGS, LLM_PRICES
from codeforces_tags.methods.typesafe_jev import settings
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_LINES_FILE,
    CANDIDATE_WORDS_FOLDER,
    CANDIDATES_FILE,
    CONCURRENCY,
    CONVENTION,
    FROZEN_NAMES,
    GEPA_NAME,
    GEPA_SEED_NAME,
    HISTORY_FILE,
    HISTORY_LINES_FILE,
    JEV_CACHE_FOLDER,
    MAX_COST_USD,
    MAX_RETRIES,
    METHOD_FILE_NAME,
    MODEL,
    NAMES,
    OUTCOME_FILE,
    PARAMS_FILE,
    PLATT_BASES,
    PLATT_FILES,
    PLATT_NAMES,
    PROMPT,
    PROMPT_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    PROVIDER,
    QUESTION,
    QUESTION_TEMPLATE,
    REFLECTION_SETTINGS,
    REFLECTIONS_FILE,
    RUN_ID_FILE,
    SEARCH_EXPERIMENT,
    SEARCH_FOLDER,
    SEARCH_GEPA_OPTIONS,
    SEARCH_HELD_OUT_FOLD,
    SEARCH_LEARNING_FOLDS,
    SEARCH_MAX_COST_USD,
    SEARCH_PROMPT_PREFIX,
    SEARCH_RUN_NAME,
    SEARCH_SEED,
    SEARCH_SELECTION_FOLD,
    SEARCH_SETTINGS,
    SETTINGS,
    STATE_FIELDS,
    STUDIES,
    TAG_DEFINITIONS,
    TEST_INFORMED,
    TIMEOUT_S,
    TUNING_FOLDER,
    WORDS_FILE_NAME,
)
from codeforces_tags.provenance import sha256_json
from codeforces_tags.templating import render_string

# The digests of the words Jev reads, computed on the text frozen at the render review. A changed
# word after the first dev request is a new method under a new name (docs/adr/0003): change these
# only before it, and say so in the commit.
DEFINITIONS_SHA256 = "998c676f9e2531b9f4b352c9587c810b55f07aa51dc5c0e41a4218d85c53ffdd"
SETTINGS_SHA256 = "8eda74ec4c3f3f8e38c952ac82b334c8888b2e08b0fe4ad6f1df1ebae2a2c7b7"
# The sha256 of the bytes of templates/jev-tags.v1.yaml, comments included: the content_sha256
# tag of version 1 in every prompt registry, which run and freeze check (tracking.py).
PROMPT_FILE_SHA256 = "199ce163c7b7892bcfa05c8698b631adf0fd98297349eb45ac613d9a1862a04c"
# The question as the frozen recipe recorded it, a str.format form, before the words moved to the
# prompt file jev-tags v1.
LEGACY_QUESTION = "Would Codeforces put the tag `{tag}` on this problem?"


def nested_keys(value):
    """Return every key of the dicts nested in a value, at any depth."""
    if isinstance(value, dict):
        return [*value, *(key for item in value.values() for key in nested_keys(item))]
    if isinstance(value, list):
        return [key for item in value for key in nested_keys(item)]
    return []


def test_the_methods_their_state_fields_and_their_studies_are_fixed():
    assert FROZEN_NAMES == ("jev-statement", "jev-statement-code")
    assert (GEPA_NAME, GEPA_SEED_NAME) == ("jev-statement-code-gepa", "jev-statement-code")
    assert (*FROZEN_NAMES, GEPA_NAME, *PLATT_NAMES) == NAMES
    assert STATE_FIELDS == {
        "jev-statement": ("statement",),
        "jev-statement-code": ("statement", "solution"),
        "jev-statement-code-gepa": ("statement", "solution"),
        "jev-statement-code-platt": ("statement", "solution"),
        "jev-statement-code-gepa-platt": ("statement", "solution"),
    }
    assert tuple(STATE_FIELDS) == NAMES
    # The searched method reads what its seed reads: only its words differ.
    assert STATE_FIELDS[GEPA_NAME] == STATE_FIELDS[GEPA_SEED_NAME]
    # Each method that sends requests is in exactly one study; the searched method is alone in
    # its own, and the calibrated twins, which send none, are in no study.
    assert STUDIES == (FROZEN_NAMES, (GEPA_NAME,))
    assert sorted(name for study in STUDIES for name in study) == sorted(
        name for name in NAMES if name not in PLATT_NAMES
    )


def test_each_calibrated_twin_reads_the_state_of_the_jev_method_that_reads_the_solution():
    assert PLATT_BASES == {
        "jev-statement-code-platt": "jev-statement-code",
        "jev-statement-code-gepa-platt": "jev-statement-code-gepa",
    }
    assert PLATT_NAMES == ("jev-statement-code-platt", "jev-statement-code-gepa-platt")
    for twin, base in PLATT_BASES.items():
        assert STATE_FIELDS[twin] == STATE_FIELDS[base] == ("statement", "solution")
    # No twin is the base of another, and jev-statement has none (docs/adr/0011).
    assert not set(PLATT_BASES.values()) & set(PLATT_NAMES)
    assert "jev-statement" not in PLATT_BASES.values()


def test_a_calibrated_twin_discloses_that_its_rule_followed_a_test_score():
    assert TEST_INFORMED == {
        "notebook": "notebooks/10_jev_gepa.ipynb",
        "commits": ["8b9c456", "5a8366b"],
        "note": "the rule was proposed after the test drop of jev-statement-code-gepa was read",
    }
    assert json.loads(json.dumps(TEST_INFORMED)) == TEST_INFORMED
    assert PLATT_FILES == ("params.json", "rules.csv", "nested.csv", "stability.csv")
    assert PLATT_FILES[0] == PARAMS_FILE


def test_the_model_is_a_pinned_version_never_the_alias():
    # jev-latest is the SDK's default and can move: every request names the version it is for.
    assert MODEL == "jev-1.13.0"
    assert "latest" not in MODEL
    assert PROVIDER == "typesafe"


def test_every_focus_tag_has_a_definition_of_one_shape_in_focus_order():
    assert tuple(TAG_DEFINITIONS) == FOCUS_TAGS
    for tag, criteria in TAG_DEFINITIONS.items():
        assert list(criteria) == ["true", "false"], tag
        for answer, listing in (("true", "includes"), ("false", "excludes")):
            entry = criteria[answer]
            assert list(entry) == ["definition", listing], (tag, answer)
            assert entry["definition"].endswith("."), (tag, answer)
            assert entry[listing], (tag, answer)
            for phrase in entry[listing]:
                assert phrase and phrase == phrase.strip(), (tag, phrase)
                assert not phrase.endswith("."), (tag, phrase)


def test_the_definitions_and_settings_are_the_frozen_text():
    assert sha256_json(TAG_DEFINITIONS) == DEFINITIONS_SHA256
    assert sha256_json(SETTINGS) == SETTINGS_SHA256


def test_the_prompt_file_is_the_frozen_file_byte_for_byte():
    data = files(settings.__package__).joinpath(PROMPT_PATH).read_bytes()

    assert hashlib.sha256(data).hexdigest() == PROMPT_FILE_SHA256
    assert PROMPT.sha256 == PROMPT_FILE_SHA256


def test_settings_hold_what_shapes_a_request_and_no_transport():
    assert list(SETTINGS) == ["model", "state_fields", "convention", "question", "tag_definitions"]
    # The frozen methods' recipes keep their bytes: neither the searched method nor a calibrated
    # twin is in SETTINGS.
    assert SETTINGS["state_fields"] == {
        "jev-statement": ["statement"],
        "jev-statement-code": ["statement", "solution"],
    }
    # MLflow flattens a recipe's mappings by their dots: a dotted key would be one value.
    assert not [key for key in nested_keys(SETTINGS) if "." in key]
    assert (TIMEOUT_S, MAX_RETRIES, CONCURRENCY, MAX_COST_USD) == (60.0, 5, 16, 2.0)
    assert METHOD_FILE_NAME == "jev.json"


def test_the_words_are_read_from_the_prompt_file_jev_tags_version_1():
    assert (PROMPT_NAME, PROMPT_VERSION) == ("jev-tags", 1)
    assert PROMPT_PATH == "templates/jev-tags.v1.yaml"
    assert (PROMPT.name, PROMPT.version) == (PROMPT_NAME, PROMPT_VERSION)
    assert list(PROMPT.data) == ["convention", "question", "tag_definitions"]
    assert PROMPT.data["convention"] is CONVENTION
    assert PROMPT.data["question"] is QUESTION_TEMPLATE
    assert PROMPT.data["tag_definitions"] is TAG_DEFINITIONS
    # Tests change a definition in place (monkeypatch.setitem): it is a plain dict.
    assert isinstance(TAG_DEFINITIONS, dict)


def test_the_question_template_renders_the_frozen_question_for_every_tag():
    assert QUESTION_TEMPLATE == "Would Codeforces put the tag `{{ tag }}` on this problem?"
    # The recipe hashes the str.format form the first freeze recorded.
    assert QUESTION == LEGACY_QUESTION
    for tag in FOCUS_TAGS:
        assert render_string(QUESTION_TEMPLATE, tag=tag) == LEGACY_QUESTION.format(tag=tag), tag


def test_the_prompt_file_says_what_it_is_and_keeps_its_keys_strings():
    text = files(settings.__package__).joinpath(PROMPT_PATH).read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]

    assert text.startswith("#")
    assert any("jev-tags" in line and "version 1" in line for line in header)
    assert any("docs/adr/0003" in line for line in header)
    assert any("docs/adr/0007" in line for line in header)
    assert list(yaml.safe_load(text)) == [
        "name",
        "version",
        "convention",
        "question",
        "tag_definitions",
    ]
    # Unquoted, YAML reads true and false as booleans: every key of a definition is a str.
    assert all(isinstance(key, str) for key in nested_keys(PROMPT.data))
    # The one non-ASCII character of the words survives the file's encoding.
    assert "p\u00b7q^-1" in str(TAG_DEFINITIONS["probabilities"]["true"]["includes"])


def test_the_search_never_reads_the_held_out_fold_and_learns_from_the_others():
    folds = {SEARCH_HELD_OUT_FOLD, SEARCH_SELECTION_FOLD, *SEARCH_LEARNING_FOLDS}

    assert (SEARCH_HELD_OUT_FOLD, SEARCH_SELECTION_FOLD, SEARCH_LEARNING_FOLDS) == (0, 4, (1, 2, 3))
    assert folds == {0, 1, 2, 3, 4} and len(SEARCH_LEARNING_FOLDS) == 3


def test_the_search_settings_are_json_values_recorded_with_the_reflection_model():
    assert SEARCH_SETTINGS == {
        "held_out_fold": 0,
        "selection_fold": 4,
        "learning_folds": [1, 2, 3],
        "minibatch_size": 40,
        "positives_per_tag": 2,
        "examples": 8,
        "statement_chars": 3000,
        "solution_chars": 2000,
        "length_factor": 1.5,
        "ngram": 8,
        "max_iterations": 200,
        "seed": 0,
        "reflection": {
            "provider": "openai",
            "model": "gpt-6-luna",
            "reasoning_effort": "high",
            "temperature": None,
            "top_p": None,
            "max_output_tokens": 32000,
            "output_format": "json",
        },
        "gepa": SEARCH_GEPA_OPTIONS,
    }
    assert json.loads(json.dumps(SEARCH_SETTINGS)) == SEARCH_SETTINGS
    assert REFLECTION_SETTINGS.model in LLM_PRICES  # every cost of the search is priced
    # GEPA's options, in order, written out even where they are its defaults: every accepted
    # child is scored on the whole selection fold.
    assert list(SEARCH_GEPA_OPTIONS.items()) == [
        ("candidate_selection_strategy", "pareto"),
        ("module_selector", "round_robin"),
        ("acceptance_criterion", "strict_improvement"),
        ("val_evaluation_policy", "full_eval"),
        ("frontier_type", "instance"),
        ("use_merge", False),
        ("skip_perfect_score", True),
        ("perfect_score", 1.0),
        ("seed", 0),
        ("raise_on_exception", True),
    ]
    assert SEARCH_GEPA_OPTIONS["seed"] == SEARCH_SEED == 0
    assert SEARCH_GEPA_OPTIONS["use_merge"] is False
    assert SEARCH_MAX_COST_USD == 15.0


def test_the_search_names_its_mlflow_runs_and_its_files():
    assert (SEARCH_EXPERIMENT, SEARCH_PROMPT_PREFIX) == ("jev-gepa", "jev-gepa-")
    assert SEARCH_RUN_NAME == "search-jev-statement-code-gepa"
    assert (SEARCH_FOLDER, JEV_CACHE_FOLDER, REFLECTIONS_FILE) == (
        "search",
        "jev",
        "reflections.jsonl",
    )
    assert (HISTORY_LINES_FILE, CANDIDATE_LINES_FILE, OUTCOME_FILE, RUN_ID_FILE) == (
        "history.jsonl",
        "candidates.jsonl",
        "outcome.json",
        "run.json",
    )
    assert (TUNING_FOLDER, PARAMS_FILE, CANDIDATES_FILE, HISTORY_FILE) == (
        "tuning",
        "params.json",
        "candidates.csv",
        "history.csv",
    )
    assert (CANDIDATE_WORDS_FOLDER, WORDS_FILE_NAME) == ("candidates", "words.json")
