"""Tests for the judges' prompt templates."""

# ruff: noqa: E501 (the recorded prompts are long lines)

import jinja2
import pytest

from codeforces_tags.near_duplicates.audit.prompts import PASSES, labels_schema, render_prompt

# The texts the recorded judges received (their task prompts, with the folder renamed /work).
# pylint: disable=line-too-long
DEFINITIONS = """You judge whether two Codeforces problem statements describe the same underlying problem, to prevent train/test leakage in a tag-prediction dataset (knowing one problem's tags or solution would give away the other's).
Labels:
- same_problem: the same task (a copy, a re-wording, a translation, or a mirror of the same problem in another round).
- versions: easy/hard versions or subtasks of ONE problem: same story and task, differing only in constraints, limits or a small extra requirement.
- related: shares a story, characters or setting (e.g. the same round or series) but asks a genuinely different task.
- different: unrelated tasks.
Judge from the statement text only. Short or generic statements can share boilerplate; similar wording alone is not enough, the TASK must match for same_problem/versions. Use confidence=low when you cannot decide."""
RECORDED = {
    "first_pass": """Read the JSON file /work/batch_0.json (a list of items with pair_id, A and B; each side has description, input_spec, output_spec; long texts are truncated). Label EVERY pair in the file. Read-only task: do not write or modify any file.
Completion criterion: one label per pair_id in the file, none skipped.""",
    "second_pass_round_1": """Read the pair texts from the files /work/batch_0.json ... /work/batch_9.json (each item has pair_id, A, B). Label ONLY these pair_ids: [8,28,62].
You are an independent second judge: decide from the texts alone. Read-only task: do not write or modify any file.
Completion criterion: one label for every listed pair_id.""",
    "second_pass_round_2": """Read the JSON file /work/second_batch_0.json (a list of items with pair_id, A and B; each side has description, input_spec, output_spec; long texts are truncated). Label EVERY pair in the file. You are an independent second judge: decide from the texts alone. Read only that file: do not open any other file, and do not write or modify any file.
Completion criterion: one label per pair_id in the file, none skipped.""",
}
# pylint: enable=line-too-long
VARIABLES = {
    "first_pass": {"batch_file": "/work/batch_0.json"},
    "second_pass_round_1": {
        "first_batch_file": "/work/batch_0.json",
        "last_batch_file": "/work/batch_9.json",
        "pair_ids": [8, 28, 62],
    },
    "second_pass_round_2": {"batch_file": "/work/second_batch_0.json"},
}


@pytest.mark.parametrize("name", PASSES)
def test_render_prompt_reproduces_the_recorded_prompt(name):
    assert render_prompt(name, **VARIABLES[name]) == f"{DEFINITIONS}\n\n{RECORDED[name]}"


def test_render_prompt_fails_on_a_missing_variable():
    with pytest.raises(jinja2.UndefinedError):
        render_prompt("first_pass")


def test_labels_schema_requires_one_label_per_pair():
    item = labels_schema()["properties"]["labels"]["items"]

    assert item["required"] == ["pair_id", "label", "confidence", "reason"]
    assert item["properties"]["label"]["enum"] == [
        "same_problem",
        "versions",
        "related",
        "different",
    ]
