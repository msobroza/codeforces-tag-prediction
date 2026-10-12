"""Tests for the approved prompts of the zero-shot methods, their prompt files and schema."""

import hashlib
import json
import re
from importlib.resources import files

import jinja2.meta
import pytest

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.methods.zero_shot import settings
from codeforces_tags.prompt_files import read_prompt_file
from codeforces_tags.templating import render_string

PACKAGE_NAME = "codeforces_tags.methods.zero_shot"
PACKAGE = files(PACKAGE_NAME)
# The sha256 of each setting's template, the bytes of the .j2 file it was approved and frozen in
# (templates/code2tag.j2 and templates/descode2tag.j2, before they moved into the prompt files),
# and of the answer schema of templates/tags.schema.json, as its JSON text in its own key order
# (the text that every json request hashes). The prompt files hold them unchanged, so every
# committed request hash still rebuilds (docs/adr/0003).
TEMPLATE_SHA256 = {
    "code2tag": "7b9e0944ce7bc90f55662a2ffec6f344fb59abf18dbd331cc1e4614d4da8bfe1",
    "descode2tag": "73e7c19cbc7e46678b20fcc3dcd36a4201af2201066fcdb8fd63737dbb518634",
}
SCHEMA_SHA256 = "b2e432ecefa7f3fc04bd8de15e31e325f02a53eceb56cbc35ca8fdd249730cc7"

# The values the approved text is rendered with: one made-up problem, and the settings'
# language.
VALUES = {
    "lang_cluster": settings.LANGUAGE,
    "prob_desc_description": "DESCRIPTION",
    "prob_desc_input_spec": "INPUT",
    "prob_desc_output_spec": "OUTPUT",
    "prob_desc_notes": "NOTES",
    "source_code": "SOLUTION",
    "all_tags": ALL_TAGS,
}
# The lines of the approved prompts that are too long to write on one line of this file.
OPENING = (
    "Predict the tags of a programming problem given a program written in Python language "
    "to solve it."
)
TAG_LIST = (
    "*special, 2-sat, binary search, bitmasks, brute force, chinese remainder theorem, "
    "combinatorics, constructive algorithms, data structures, dfs and similar, "
    "divide and conquer, dp, dsu, expression parsing, fft, flows, games, geometry, "
    "graph matchings, graphs, greedy, hashing, implementation, interactive, math, matrices, "
    "meet-in-the-middle, number theory, probabilities, schedules, shortest paths, sortings, "
    "string suffix structures, strings, ternary search, trees, two pointers"
)
CLOSING = (
    "Provide the tags of the problem, separated by commas, without any description or extra tokens."
)
# The text that the zero-shot design fixed for the two prompts, rendered with VALUES, with no
# trailing newline: Jinja drops the template's final one.
APPROVED_PROMPTS = {
    "code2tag": "\n".join(
        [
            OPENING,
            "",
            "Here is the code:",
            "",
            "SOLUTION",
            "",
            f"Choose the tags from this list: {TAG_LIST}",
            "",
            CLOSING,
        ]
    ),
    "descode2tag": "\n".join(
        [
            OPENING,
            "Description: DESCRIPTION",
            "Input Specification: INPUT",
            "Output Specification: OUTPUT",
            "Notes: NOTES",
            "",
            "Here is the code:",
            "",
            "SOLUTION",
            "",
            f"Choose the tags from this list: {TAG_LIST}",
            "",
            CLOSING,
        ]
    ),
}
# Made-up problems for the lines of the prompts: a plain one; one whose input and output
# specifications and notes are missing (they print None), whose solution has Windows line
# breaks and whose description holds Jinja syntax, which must reach the prompt as it is.
PROBLEMS = {
    "plain": {
        "prob_desc_description": "Find the number of leaves of a tree.",
        "prob_desc_input_spec": "The first line holds $$$n$$$.",
        "prob_desc_output_spec": "Print one integer.",
        "prob_desc_notes": "In the first example, the answer is $$$2$$$.",
        "source_code": "n = int(input())\nprint(n - 1)\n",
    },
    "missing fields, CRLF and Jinja syntax": {
        "prob_desc_description": "Print {{ n }} and {% n %} as they are.",
        "prob_desc_input_spec": None,
        "prob_desc_output_spec": None,
        "prob_desc_notes": None,
        "source_code": "print(1)\r\nprint(2)\r\n",
    },
}


def prompt_file(setting, output_format):
    """Return the prompt file of a setting and output format, as the package ships it."""
    keys = {"output_format", "template"} | (
        {"response_format"} if output_format == "json" else set()
    )
    return read_prompt_file(PACKAGE_NAME, settings.PROMPT_PATHS[setting][output_format], keys)


def template_source(setting):
    """Return the template of a setting, as its version 1 prompt file holds it."""
    return prompt_file(setting, "text").data["template"]


def schema():
    """Return the answer schema, as the version 2 prompt file of Code2Tag holds it."""
    return prompt_file("code2tag", "json").data["response_format"]


def sha256(text):
    """Return the sha256 hex digest of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_a_settings_template_is_the_approved_j2_file_byte_for_byte(setting):
    assert sha256(template_source(setting)) == TEMPLATE_SHA256[setting]


def test_the_schema_is_the_approved_json_schema_in_its_own_key_order():
    assert sha256(json.dumps(schema(), separators=(",", ":"), allow_nan=False)) == SCHEMA_SHA256


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_a_settings_prompt_files_are_its_versions_with_one_template_and_one_schema(setting):
    text, json_file = (prompt_file(setting, output_format) for output_format in ("text", "json"))

    assert (text.name, text.version, text.data["output_format"]) == (setting, 1, "text")
    assert (json_file.name, json_file.version, json_file.data["output_format"]) == (
        setting,
        2,
        "json",
    )
    assert json_file.data["template"] == text.data["template"]
    assert json_file.data["response_format"] == schema()


@pytest.mark.parametrize(
    "path", [path for paths in settings.PROMPT_PATHS.values() for path in paths.values()]
)
def test_a_prompt_file_holds_its_template_as_a_literal_block(path):
    text = PACKAGE.joinpath(path).read_text(encoding="utf-8")

    assert "\ntemplate: |\n" in text
    assert "\r" not in text


def test_the_templates_folder_holds_the_four_prompt_files_only():
    found = sorted(item.name for item in PACKAGE.joinpath("templates").iterdir())

    assert found == [
        "code2tag.v1.yaml",
        "code2tag.v2.yaml",
        "descode2tag.v1.yaml",
        "descode2tag.v2.yaml",
    ]


def both_prompts(problem):
    """Return the rendered Code2Tag and DesCode2Tag prompts of a made-up problem."""
    values = {"lang_cluster": settings.LANGUAGE, "all_tags": ALL_TAGS, **problem}
    return tuple(render_string(template_source(setting), **values) for setting in settings.SETTINGS)


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_rendered_prompts_are_the_approved_text_of_spec_8_4(setting):
    rendered = render_string(template_source(setting), **VALUES)

    assert rendered == APPROVED_PROMPTS[setting]


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_a_template_file_is_one_comment_line_then_the_prompt_and_one_final_newline(setting):
    source = template_source(setting)
    comment, prompt = source.split("\n", 1)

    assert comment.startswith("{#- ")
    assert comment.endswith(" -#}")
    assert comment.count("{#") == comment.count("#}") == 1
    assert prompt.startswith("Predict the tags")
    assert prompt.endswith("extra tokens.\n")
    assert "{#" not in prompt
    assert "\r" not in source


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_header_comment_is_not_rendered(setting):
    source = template_source(setting)
    prompt = source.split("\n", 1)[1]

    rendered = render_string(source, **VALUES)

    assert rendered == render_string(prompt, **VALUES)
    assert rendered.startswith("Predict the tags")


@pytest.mark.parametrize("problem", PROBLEMS.values(), ids=PROBLEMS)
def test_deleting_lines_2_to_5_of_a_descode2tag_prompt_gives_the_code2tag_prompt(problem):
    code2tag, descode2tag = both_prompts(problem)
    lines = descode2tag.split("\n")

    assert "\n".join(lines[:1] + lines[5:]) == code2tag


def test_a_field_with_a_line_break_moves_the_code_out_of_line_5_of_the_prompt():
    # The premise of the test above, and the reason verify_extra checks the four fields.
    problem = PROBLEMS["plain"] | {"prob_desc_description": "First line.\nSecond line."}
    code2tag, descode2tag = both_prompts(problem)
    lines = descode2tag.split("\n")

    assert "\n".join(lines[:1] + lines[5:]) != code2tag


def test_a_missing_field_prints_none_on_its_own_labelled_line():
    descode2tag = both_prompts(PROBLEMS["missing fields, CRLF and Jinja syntax"])[1]

    assert descode2tag.split("\n")[2:5] == [
        "Input Specification: None",
        "Output Specification: None",
        "Notes: None",
    ]


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_a_template_reads_its_allow_listed_fields_the_language_and_the_tag_list_only(setting):
    tree = jinja2.Environment().parse(template_source(setting))

    variables = jinja2.meta.find_undeclared_variables(tree)

    assert variables == {*settings.FIELDS[setting], "lang_cluster", "all_tags"}


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_fields_of_a_setting_are_in_the_order_its_prompt_prints_them(setting):
    printed = re.findall(r"{{\s*(\w+)", template_source(setting))

    fields = [name for name in printed if name in settings.FIELDS[setting]]

    assert fields == list(settings.FIELDS[setting])


def test_the_schema_enum_is_all_tags_and_the_rendered_tag_list():
    enum = schema()["properties"]["tags"]["items"]["enum"]

    assert enum == list(ALL_TAGS)
    for setting in settings.SETTINGS:
        rendered = render_string(template_source(setting), **VALUES)
        assert f"Choose the tags from this list: {', '.join(enum)}" in rendered.split("\n")


def test_the_schema_is_strict():
    root = schema()
    tags = root["properties"]["tags"]

    assert root["type"] == "object"
    assert root["required"] == list(root["properties"]) == ["tags"]
    assert root["additionalProperties"] is False
    assert "uniqueItems" not in tags  # structured outputs do not take it
    assert (tags["type"], tags["minItems"], tags["maxItems"]) == ("array", 0, len(ALL_TAGS))
    assert type(tags["minItems"]) is type(tags["maxItems"]) is int  # not False, not a float
    assert tags["items"]["type"] == "string"


def test_the_schema_keys_are_in_the_order_the_request_hash_covers():
    # A schema is hashed as its JSON text in its own key order (llm.prompts.LlmRequest.sha256).
    root = schema()
    tags = root["properties"]["tags"]

    assert list(root) == ["type", "properties", "required", "additionalProperties"]
    assert list(tags) == ["type", "items", "minItems", "maxItems"]
    assert list(tags["items"]) == ["type", "enum"]
