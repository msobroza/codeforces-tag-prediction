"""Tests for the package contract of the zero-shot methods, their registry lines and folders."""

import dataclasses
import sys
from collections import Counter

import mlflow
import pandas as pd
import pytest

from codeforces_tags.config import ALL_TAGS, RAW_INPUT_COLUMNS, TEXT_PARSER_VERSION
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    load_predictions,
    load_run_record,
    stack_parts,
)
from codeforces_tags.llm import __main__ as cli
from codeforces_tags.llm.answers import answer_summary, encode_answers, per_problem_frame
from codeforces_tags.llm.prompted import PromptedLlm, load_prompted
from codeforces_tags.llm.prompts import prompt_content_sha256, prompt_snapshot
from codeforces_tags.llm.store import answers_file, read_answers
from codeforces_tags.methods import METHODS, load_method, method_dir, model_dir, save_method
from codeforces_tags.methods.zero_shot import methods, settings
from codeforces_tags.prompt_files import read_prompt_file
from codeforces_tags.provenance import file_sha256
from codeforces_tags.splitting import PARTS, load_split
from codeforces_tags.tracking.prompts import CONTENT_TAG, register_prompts

MODULE = "codeforces_tags.methods.zero_shot.methods"
PACKAGE_NAME = "codeforces_tags.methods.zero_shot"
# The content hash of each method's prompt: its template file, its schema, its settings and its
# extra. Final from the first dev request (docs/adr/0003): an edit to a template, to the schema or
# to a setting fails here, before `llm run` refuses the answers or `register_prompts` the prompt.
PROMPT_CONTENT_SHA256 = {
    "code2tag-text": "c70232aeccd8269142306351764022811d2f47c1180fedb5448510dd0fdfca67",
    "code2tag-json": "13f9312c5cb8528b996049ea5e3e2604f80b6b2bd1b213166f9ac9eb5b65dfad",
    "descode2tag-text": "8cfc4afe6a9bfa2f3cbcbf2efc5ee4b378389e4accd61c6476fe0a9153f8d610",
    "descode2tag-json": "e14e0992173645d345153d3af8a9d8a98dec1f30d4cc3d9682a891e57811dcf4",
}
# The paths of each committed method folder's spec: the .j2 template and .json schema files the
# methods were frozen with, before the prompts moved into the versioned prompt files. They name
# where the prompt was read, not what it is (PROMPT_CONTENT_SHA256 pins that, and the prompt files
# hold the same template and schema): a frozen folder keeps them, so its spec is the package's
# spec but for these paths.
FROZEN_PATHS = {
    "code2tag-text": {"template_path": "templates/code2tag.j2", "schema_path": None},
    "code2tag-json": {
        "template_path": "templates/code2tag.j2",
        "schema_path": "templates/tags.schema.json",
    },
    "descode2tag-text": {"template_path": "templates/descode2tag.j2", "schema_path": None},
    "descode2tag-json": {
        "template_path": "templates/descode2tag.j2",
        "schema_path": "templates/tags.schema.json",
    },
}
# The run record's entries that `llm score` copies from the method's spec.
SPEC_ENTRIES = ("llm_settings", "fields", "constants", "extra")
# Every character at which str.splitlines() splits, written as escapes: most are invisible.
LINE_BREAKS = ("\n", "\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_llm_spec_gives_the_prompt_fields_and_language_of_its_setting_and_the_request_settings(
    name,
):
    setting, output_format = name.rsplit("-", 1)
    version = {"text": 1, "json": 2}[output_format]

    spec = methods.llm_spec(name)

    assert (spec.name, spec.prompt_name) == (name, setting)
    assert spec.template_path == f"templates/{setting}.v{version}.yaml"
    assert spec.fields == settings.FIELDS[setting]
    assert spec.constants == {"lang_cluster": "Python"}
    assert spec.all_tags == ALL_TAGS
    assert spec.extra == {}
    assert spec.settings.to_dict() == {
        "provider": "openai",
        "model": "gpt-6-luna",
        "reasoning_effort": "none",
        "temperature": settings.TEMPERATURE,
        "top_p": 1.0,
        "max_output_tokens": 512,
        "output_format": output_format,
    }


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_json_method_of_a_setting_differs_from_the_text_one_by_its_schema_alone(setting):
    text, json_method = (methods.llm_spec(f"{setting}-{form}") for form in ("text", "json"))

    assert (text.schema_path, text.schema_name) == (None, None)
    assert (json_method.schema_path, json_method.schema_name) == (
        f"templates/{setting}.v2.yaml",
        "problem_tags",
    )
    assert json_method.template_path == json_method.schema_path  # one prompt file, version 2
    without_schema = dataclasses.replace(
        json_method,
        name=text.name,
        template_path=text.template_path,
        schema_path=None,
        schema_name=None,
        settings=text.settings,
    )
    assert without_schema == text
    assert json_method.settings == dataclasses.replace(text.settings, output_format="json")


@pytest.mark.parametrize(
    "name", ["code2tag", "code2tag-xml", "Code2Tag-text", "tfidf", "fake-json"]
)
def test_llm_spec_refuses_a_name_that_is_not_a_zero_shot_method(name):
    with pytest.raises(ValueError, match="is not a zero-shot method"):
        methods.llm_spec(name)


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_prompt_family_gives_the_text_then_the_json_method_of_a_setting(setting):
    family = methods.prompt_family(setting)

    assert [spec.name for spec in family] == [f"{setting}-text", f"{setting}-json"]
    assert family == tuple(methods.llm_spec(spec.name) for spec in family)
    assert {spec.prompt_name for spec in family} == {setting}


@pytest.mark.parametrize("prompt_name", ["code2tag-text", "fake", "tags"])
def test_prompt_family_refuses_a_prompt_that_is_not_a_setting(prompt_name):
    with pytest.raises(ValueError, match="is not the prompt of a zero-shot setting"):
        methods.prompt_family(prompt_name)


def test_the_registry_names_that_the_package_serves_are_the_method_names():
    served = {name for name, spec in METHODS.items() if spec.module == MODULE}

    assert served == set(settings.METHOD_NAMES)


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_each_method_is_registered_as_a_zero_shot_api_method_of_the_llm_group(name):
    spec = METHODS[name]

    assert (spec.module, spec.kind, spec.calls_api) == (MODULE, "zero_shot", True)
    assert spec.groups == ("llm",)
    assert spec.requirements == ("numpy", "pandas", "jinja2", "openai", "pyyaml")
    assert (spec.score_atol, spec.depends_on, spec.input_columns) == (0.0, (), RAW_INPUT_COLUMNS)


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_the_llm_toolkit_serves_each_method_through_the_package(name):
    package = cli.prompted_package(name)
    spec = package.llm_spec(name)

    assert package is methods
    assert spec.name == name
    assert spec in package.prompt_family(spec.prompt_name)
    assert methods.load is load_prompted
    assert methods.__all__ == ["load", "llm_spec", "prompt_family", "verify_extra"]


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_prompt_snapshot_reads_the_template_and_the_schema_from_the_prompt_file(name):
    spec = methods.llm_spec(name)
    json_format = name.endswith("-json")
    keys = {"output_format", "template"} | ({"response_format"} if json_format else set())
    stored = read_prompt_file(PACKAGE_NAME, spec.template_path, keys).data

    prompt = prompt_snapshot(spec)

    assert (prompt.name, prompt.version, prompt.uri) == (spec.prompt_name, None, None)
    assert prompt.template == stored["template"]
    assert prompt.response_format == (stored["response_format"] if json_format else None)


def test_the_prompt_content_sha256_of_each_method_is_pinned():
    found = {
        name: prompt_snapshot(methods.llm_spec(name)).content_sha256
        for name in settings.METHOD_NAMES
    }

    assert found == PROMPT_CONTENT_SHA256


@pytest.mark.usefixtures("fake_ask")
@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_load_reads_a_saved_method_back_and_it_asks_through_the_clients_ask(
    name, zero_shot_raw, asked, tmp_path
):
    spec = methods.llm_spec(name)
    save_method(PromptedLlm(spec, prompt_snapshot(spec)), tmp_path / "model")

    loaded = load_method(tmp_path / "model")
    predictions = loaded.predict(zero_shot_raw)

    assert isinstance(loaded, PromptedLlm)
    assert (loaded.spec, loaded.prompt, loaded.ask) == (spec, prompt_snapshot(spec), None)
    assert predictions.equals(multi_hot(zero_shot_raw["tags"], ALL_TAGS))
    assert sorted((request.method, request.sample_id) for request in asked) == [
        (name, 7),
        (name, 13),
        (name, 21),
    ]


def test_the_line_breaks_are_every_character_at_which_splitlines_splits():
    found = [
        chr(code) for code in range(sys.maxunicode + 1) if len(f"a{chr(code)}b".splitlines()) == 2
    ]

    assert tuple(found) == LINE_BREAKS


def test_verify_extra_reads_raw_rows_of_both_parts_with_missing_fields(zero_shot_raw):
    assert methods.verify_extra(zero_shot_raw) is None
    assert [field.removeprefix("prob_desc_") for field in methods.LINE_FIELDS] == [
        "description",
        "input_spec",
        "output_spec",
        "notes",
    ]


def test_verify_extra_accepts_solutions_with_crlf_line_endings(zero_shot_raw):
    # 1,037 real solutions end their lines with "\r\n": str.splitlines() would split at each
    # "\r" too, while the lines of a prompt end at "\n" alone.
    solution = "a, b = map(int, input().split())\r\nprint(a + b)\r\n"
    zero_shot_raw["source_code"] = solution
    descode2tag = methods.llm_spec("descode2tag-text")

    methods.verify_extra(zero_shot_raw)

    requests = PromptedLlm(descode2tag, prompt_snapshot(descode2tag)).requests(zero_shot_raw)
    assert all(f"\n\n{solution}\n\n" in request.input for request in requests)


@pytest.mark.parametrize("field", ["description", "input_spec", "output_spec", "notes"])
@pytest.mark.parametrize("line_break", LINE_BREAKS, ids=[hex(ord(c)) for c in LINE_BREAKS])
def test_verify_extra_reads_raw_rows_and_names_the_problems_whose_text_field_breaks_a_line(
    zero_shot_raw, field, line_break
):
    column = f"prob_desc_{field}"
    zero_shot_raw.loc[[13, 21], column] = f"First part.{line_break}Second part."

    with pytest.raises(ValueError, match=rf"in {column} of the problems \[13, 21\] would move"):
        methods.verify_extra(zero_shot_raw)
    methods.verify_extra(zero_shot_raw.drop(index=[13, 21]))


def test_verify_extra_names_every_field_that_breaks_a_line_in_one_error(zero_shot_raw):
    zero_shot_raw.loc[13, "prob_desc_description"] = "First part.\nSecond part."
    zero_shot_raw.loc[[7, 21], "prob_desc_notes"] = "First part.\rSecond part."
    named = (
        r"in prob_desc_description of the problems \[13\]; "
        r"prob_desc_notes of the problems \[7, 21\] would move"
    )

    with pytest.raises(ValueError, match=named):
        methods.verify_extra(zero_shot_raw)


def test_verify_extra_checks_the_fields_once_preprocessed(zero_shot_raw):
    # raw_to_problems strips the heading of the notes, and the line break after it.
    zero_shot_raw.loc[21, "prob_desc_notes"] = "Note\nIn the first example, the tree has 2 leaves."

    methods.verify_extra(zero_shot_raw)


def test_verify_extra_names_the_problems_whose_prompts_differ_beyond_lines_2_to_5(
    zero_shot_raw, monkeypatch
):
    notes = "Notes: {{ prob_desc_notes }}\n"

    def with_a_sixth_line(spec):
        prompt = prompt_snapshot(spec)
        template = prompt.template.replace(notes, f"{notes}Time limit: 2 seconds\n")
        content = prompt_content_sha256(template, None, spec.settings, spec.extra)
        return dataclasses.replace(prompt, template=template, content_sha256=content)

    monkeypatch.setattr(methods, "prompt_snapshot", with_a_sixth_line)

    with pytest.raises(ValueError, match=r"problems \[7, 13, 21\] are not their Code2Tag inputs"):
        methods.verify_extra(zero_shot_raw)


@pytest.mark.usefixtures("fake_ask")
@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_the_zero_shot_methods_pass_the_conformance_check_with_a_fake_ask(
    name, zero_shot_raw, asked, monkeypatch, tmp_path
):
    # No SDK and no key: any `import openai` raises ImportError, so a request that reached the
    # real adapter would fail the check instead of being sent.
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spec = methods.llm_spec(name)

    check_tagging_method(PromptedLlm(spec, prompt_snapshot(spec)), zero_shot_raw, tmp_path / "out")

    # Each problem is asked four times: in the batch, with every raw column, alone, and after
    # save_method and load_method; the isolation subprocess only loads an API method.
    assert Counter(request.sample_id for request in asked) == {21: 4, 7: 4, 13: 4}
    assert {request.method for request in asked} == {name}


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_register_prompts_makes_the_text_method_version_1_and_the_json_method_version_2(
    setting, store
):
    text, json_method = f"{setting}-text", f"{setting}-json"
    family = [(spec, prompt_snapshot(spec)) for spec in methods.prompt_family(setting)]
    stored = read_prompt_file(
        PACKAGE_NAME,
        settings.PROMPT_PATHS[setting]["json"],
        {"output_format", "template", "response_format"},
    ).data

    versions = register_prompts(setting, family, **store)
    again = register_prompts(setting, family, **store)

    assert {name: version.uri for name, version in versions.items()} == {
        text: f"prompts:/{setting}/1",
        json_method: f"prompts:/{setting}/2",
    }
    assert {name: version.version for name, version in again.items()} == {text: 1, json_method: 2}
    found = mlflow.MlflowClient().search_prompt_versions(setting)
    assert sorted((version.version, version.tags[CONTENT_TAG]) for version in found) == [
        (1, PROMPT_CONTENT_SHA256[text]),
        (2, PROMPT_CONTENT_SHA256[json_method]),
    ]
    assert (versions[text].response_format, versions[json_method].response_format) == (
        None,
        stored["response_format"],
    )
    assert {version.template for version in versions.values()} == {stored["template"]}


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_each_committed_method_folder_holds_the_packages_prompt_and_recipe(name):
    # Needs no dataset, so it runs in CI: an edit to a template, to the schema or to a setting
    # after the paid answers fails here (docs/adr/0003), as does a method folder whose spec or
    # prompt, or a run record whose recipe, prompt or spec entries, are not the package's.
    # Version 1 is text, version 2 json: the version of the method's prompt file.
    spec = methods.llm_spec(name)
    prompt = prompt_snapshot(spec)
    version = settings.PROMPT_VERSIONS[spec.settings.output_format]
    uri = f"prompts:/{spec.prompt_name}/{version}"

    loaded = load_method(model_dir(name))
    record = load_run_record(method_dir(name))

    assert spec.template_path == f"templates/{spec.prompt_name}.v{version}.yaml"
    assert loaded.spec == dataclasses.replace(spec, **FROZEN_PATHS[name])
    assert loaded.prompt == dataclasses.replace(prompt, version=version, uri=uri)
    assert loaded.prompt.content_sha256 == PROMPT_CONTENT_SHA256[name]
    assert record.recipe_sha256 == cli.recipe_sha256(spec, prompt)
    assert record.prompts == (uri,)
    assert record.method_entries["prompt"] == {
        "name": spec.prompt_name,
        "version": version,
        "uri": uri,
        "content_sha256": PROMPT_CONTENT_SHA256[name],
    }
    assert {key: record.method_entries[key] for key in SPEC_ENTRIES} == {
        "llm_settings": spec.settings.to_dict(),
        "fields": list(spec.fields),
        "constants": dict(spec.constants),
        "extra": dict(spec.extra),
    }
    assert record.method_entries["dev_in_sample"] is False


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_each_committed_method_predicts_the_37_tags_from_its_committed_answers(name):
    # Needs no dataset, so it runs in CI: predictions.csv, per_problem.csv and the run record's
    # answers entries are functions of the answers files and the frozen split alone, derived
    # here as `llm score` derives them (`llm verify` does too, with the dataset). A hand edit to
    # one of them, or a parser change that moves a committed answer's tags, fails here.
    output_format = methods.llm_spec(name).settings.output_format
    split = load_split()
    answers = {part: read_answers(answers_file(name, part)) for part in PARTS}
    encoded = {
        part: encode_answers(
            answers[part],
            pd.Index(sorted(split.index[split["part"] == part]), name="sample_id"),
            output_format,
        )
        for part in PARTS
    }
    calls = pd.concat([per_problem_frame(answers[part], part) for part in PARTS], ignore_index=True)

    predictions = load_predictions(method_dir(name))
    record = load_run_record(method_dir(name))

    assert tuple(predictions.columns) == ALL_TAGS
    pd.testing.assert_frame_equal(predictions, stack_parts(**encoded))
    per_problem = (method_dir(name) / PER_PROBLEM_FILE).read_text(encoding="utf-8")
    assert per_problem == calls.to_csv(index=False)
    assert record.method_entries["answers_sha256"] == {
        part: file_sha256(answers_file(name, part)) for part in PARTS
    }
    assert record.method_entries["answers"] == {
        part: answer_summary(answers[part], output_format) for part in PARTS
    }
    assert record.method_entries["text_parser_version"] == TEXT_PARSER_VERSION
