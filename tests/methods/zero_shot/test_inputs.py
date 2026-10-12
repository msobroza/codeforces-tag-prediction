"""Tests for the inputs of the zero-shot methods and the bodies of their requests."""

import pandas as pd
import pytest

from codeforces_tags.llm.prompted import PromptedLlm
from codeforces_tags.llm.prompts import prompt_snapshot
from codeforces_tags.methods.zero_shot import methods, settings
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.prompt_files import read_prompt_file

# Every column of a sample file of the dataset, and the two that select_part appends.
DATASET_COLUMNS = {
    *(f"prob_desc_{name}" for name in ("description", "input_spec", "output_spec", "notes")),
    *(f"prob_desc_{name}" for name in ("sample_inputs", "sample_outputs", "created_at")),
    *(f"prob_desc_{name}" for name in ("time_limit", "memory_limit", "input_from", "output_to")),
    "source_code",
    "tags",
    "difficulty",
    "src_uid",
    "code_uid",
    "file_name",
    "lang",
    "lang_cluster",
    "exec_outcome",
    "hidden_unit_tests",
}
SPLIT_COLUMNS = {"fold", "group"}
# A new value of each column that a prompt may not read: the labels first. The four statement
# fields are outside Code2Tag's allow-list only.
PERTURBATIONS = {
    "tags": ["geometry", "probabilities"],
    "difficulty": 3500,
    "src_uid": "uid-perturbed",
    "code_uid": "code-perturbed",
    "file_name": "prog_perturbed.py",
    "lang": "PyPy 3",
    "lang_cluster": "C++",
    "exec_outcome": "WRONG_ANSWER",
    "hidden_unit_tests": '[{"input": "1", "output": ["2"]}]',
    "prob_desc_sample_inputs": '["7\\n1 2 3"]',
    "prob_desc_sample_outputs": '["geometry"]',
    "prob_desc_created_at": "1700000000",
    "prob_desc_time_limit": "5 seconds",
    "prob_desc_memory_limit": "1024 megabytes",
    "prob_desc_input_from": "input.txt",
    "prob_desc_output_to": "output.txt",
    "fold": 4,
    "group": 99,
    "prob_desc_description": "Find the area of a triangle.",
    "prob_desc_input_spec": "Three points.",
    "prob_desc_output_spec": "The area.",
    "prob_desc_notes": "Mind the precision.",
}
# The answer schema, as the version 2 prompt file of Code2Tag holds it.
SCHEMA = read_prompt_file(
    "codeforces_tags.methods.zero_shot",
    settings.PROMPT_PATHS["code2tag"]["json"],
    {"output_format", "template", "response_format"},
).data["response_format"]


def requests_of(name, raw_problems):
    """Return a method's requests for raw problems, as ``llm run`` builds them."""
    spec = methods.llm_spec(name)
    return PromptedLlm(spec, prompt_snapshot(spec)).requests(raw_problems)


def sent(name, raw_problems):
    """Return what a method sends for each problem: its sample id, input and request hash."""
    return [
        (request.sample_id, request.input, request.sha256())
        for request in requests_of(name, raw_problems)
    ]


def perturbed(raw_problems, column):
    """Return a copy of raw problems whose ``column`` holds its perturbation on every row."""
    copy = raw_problems.copy()
    value = PERTURBATIONS[column]
    copy[column] = pd.Series([value] * len(copy), index=copy.index, dtype=object)
    return copy


def lines(name, raw_problems, sample_id):
    """Return the lines of a method's input for one problem, split at line feeds."""
    [request] = requests_of(name, raw_problems.loc[[sample_id]])
    return request.input.split("\n")


def test_every_column_of_the_dataset_but_the_solution_has_a_perturbation():
    assert set(PERTURBATIONS) == (DATASET_COLUMNS | SPLIT_COLUMNS) - {"source_code"}


def test_the_raw_problems_hold_every_column_of_the_dataset(zero_shot_raw):
    assert set(zero_shot_raw.columns) == DATASET_COLUMNS | SPLIT_COLUMNS


@pytest.mark.parametrize("column", PERTURBATIONS)
def test_the_prompt_is_byte_identical_when_a_field_outside_the_allow_list_changes(
    zero_shot_raw, column
):
    names = [
        name
        for name in settings.METHOD_NAMES
        if column not in settings.FIELDS[name.rsplit("-", 1)[0]]
    ]
    changed = perturbed(zero_shot_raw, column)

    assert not changed[column].equals(zero_shot_raw[column])
    assert names
    assert {name: sent(name, changed) for name in names} == {
        name: sent(name, zero_shot_raw) for name in names
    }


@pytest.mark.parametrize("column", settings.FIELDS["descode2tag"])
def test_the_descode2tag_prompt_changes_when_an_allow_listed_field_changes(zero_shot_raw, column):
    changed = zero_shot_raw.copy()
    changed.loc[21, column] = "Another text."

    for name in ("descode2tag-text", "descode2tag-json"):
        before, after = (sent(name, problems) for problems in (zero_shot_raw, changed))
        assert [row[0] for row in before] == [row[0] for row in after] == [21, 7, 13]
        assert before[1:] == after[1:]
        assert "Another text." in after[0][1]
        assert before[0][2] != after[0][2]


def test_a_missing_field_prints_none_on_its_labelled_line(zero_shot_raw):
    # Problem 7 has no input specification (NaN), a blank output specification and no notes.
    descode2tag = lines("descode2tag-text", zero_shot_raw, 7)

    assert descode2tag[1] == "Description: Print the gcd of $$$a$$$ and $$$b$$$."
    assert descode2tag[2:5] == [
        f"{label}: None" for label in ("Input Specification", "Output Specification", "Notes")
    ]


def test_each_statement_field_is_the_cleaned_text_of_raw_to_problems_as_it_is(zero_shot_raw):
    problems = raw_to_problems(zero_shot_raw)
    labels = ("Description", "Input Specification", "Output Specification", "Notes")

    descode2tag = lines("descode2tag-json", zero_shot_raw, 21)

    assert descode2tag[1:5] == [
        f"{label}: {problems.loc[21, field]}"
        for label, field in zip(labels, methods.LINE_FIELDS, strict=True)
    ]
    # The glued sentence is fixed and the notes' heading dropped; LaTeX and the HTML entity stay,
    # since TF-IDF's normalise_statement is not applied.
    assert descode2tag[1:5] == [
        "Description: Count the leaves of the tree. Each leaf has degree $$$1$$$.",
        "Input Specification: The first line holds $$$n$$$ ($$$2 &lt; n \\le 10^5$$$).",
        "Output Specification: Print the number of leaves.",
        "Notes: In the first example, the tree has $$$2$$$ leaves.",
    ]


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_the_solution_is_sent_as_submitted_with_comments_windows_line_breaks_and_no_cut(
    name, zero_shot_raw
):
    requests = requests_of(name, zero_shot_raw)

    assert len(zero_shot_raw.loc[13, "source_code"]) > 2048
    assert "\r\n" in zero_shot_raw.loc[7, "source_code"]
    assert "# count the leaves" in zero_shot_raw.loc[21, "source_code"]
    for request in requests:
        solution = zero_shot_raw.loc[request.sample_id, "source_code"]
        assert f"Here is the code:\n\n{solution}\n\nChoose the tags from" in request.input


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_every_body_is_one_user_message_with_the_request_settings_of_spec_8_2(name, zero_shot_raw):
    for request in requests_of(name, zero_shot_raw):
        body = request.body()

        assert body["input"] == [{"role": "user", "content": request.input}]
        assert "instructions" not in body
        assert body["model"] == "gpt-6-luna"
        assert body["reasoning"] == {"effort": "none"}
        assert (body["temperature"], body["top_p"]) == (0.3157894736842105, 1.0)
        assert body["max_output_tokens"] == 512


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_only_the_json_body_carries_the_strict_schema_in_its_text_format(setting, zero_shot_raw):
    text = requests_of(f"{setting}-text", zero_shot_raw)
    json_requests = requests_of(f"{setting}-json", zero_shot_raw)

    strict = {"type": "json_schema", "name": "problem_tags", "schema": SCHEMA, "strict": True}

    assert all("text" not in request.body() for request in text)
    assert [request.body()["text"] for request in json_requests] == [{"format": strict}] * 3


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_text_and_json_methods_of_a_setting_send_the_same_input_under_other_hashes(
    setting, zero_shot_raw
):
    text, json_requests = (
        requests_of(f"{setting}-{form}", zero_shot_raw) for form in settings.OUTPUT_FORMATS
    )

    assert [request.input for request in text] == [request.input for request in json_requests]
    assert not {request.sha256() for request in text} & {r.sha256() for r in json_requests}


def test_the_four_methods_send_a_distinct_request_for_each_problem(zero_shot_raw):
    hashes = {
        request.sha256()
        for name in settings.METHOD_NAMES
        for request in requests_of(name, zero_shot_raw)
    }

    assert len(hashes) == len(settings.METHOD_NAMES) * len(zero_shot_raw)
