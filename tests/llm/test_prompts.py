"""Tests for prompt snapshots, prompted-method specs, and the requests built from them."""

import dataclasses
import importlib
import json
import sys
import textwrap
from importlib import resources

import jinja2
import numpy as np
import pandas as pd
import pytest

from codeforces_tags.llm import prompts
from codeforces_tags.llm.prompts import (
    LlmMethodSpec,
    LlmRequest,
    PromptSnapshot,
    build_requests,
    field_sha256s,
    field_values,
    prompt_content_sha256,
    prompt_snapshot,
)
from codeforces_tags.methods import METHODS, MethodSpec
from codeforces_tags.provenance import sha256_json

# A made-up prompted package, written to tmp_path: a module, a template and a JSON schema, and
# the same prompt as two versioned prompt files, version 1 for text and version 2 for json.
PACKAGE = "prompt_fakes"
TAGS = ("dp", "math", "trees")
TEMPLATE = (
    "{#- A made-up prompt: the problem, then its solution. -#}\n"
    "Tag this {{ lang_cluster }} solution.\n"
    "Description: {{ prob_desc_description }}\n"
    "Notes: {{ prob_desc_notes }}\n"
    "\n"
    "{{ source_code }}\n"
    "\n"
    'Choose from: {{ all_tags | join(", ") }}\n'
)
SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string", "enum": list(TAGS)}}},
    "required": ["tags"],
    "additionalProperties": False,
}
FIELDS = ("prob_desc_description", "prob_desc_notes", "source_code")


def prompt_file_text(version, output_format, *, name="fake", schema=None):
    """Return a prompt file of the fake prompt: its template as a literal block, then a schema."""
    text = (
        f"# The fake prompt, version {version}.\n"
        f"name: {name}\n"
        f"version: {version}\n"
        f"output_format: {output_format}\n"
        "template: |\n" + textwrap.indent(TEMPLATE, "  ")
    )
    return text if schema is None else text + f"response_format: {json.dumps(schema)}\n"


def write_prompt_file(registry_package, file_name, text):
    """Write a prompt file under the fake package's ``templates/``; return its relative path."""
    path = f"templates/{file_name}"
    (registry_package / path).write_text(text, encoding="utf-8")
    return path


@pytest.fixture(name="registry")
def registry_fixture(tmp_path, monkeypatch):
    """Write the prompted package, make it importable, and return a registry of its methods.

    The package is forgotten after the test: the next test's package lives in another folder.
    """
    package = tmp_path / PACKAGE
    (package / "templates").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "methods.py").write_text('"""Made-up prompted methods."""\n', encoding="utf-8")
    (package / "templates" / "tags.j2").write_text(TEMPLATE, encoding="utf-8")
    (package / "templates" / "tags.schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    write_prompt_file(package, "fake.v1.yaml", prompt_file_text(1, "text"))
    write_prompt_file(package, "fake.v2.yaml", prompt_file_text(2, "json", schema=SCHEMA))
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = MethodSpec(
        module=f"{PACKAGE}.methods", kind="zero_shot", requirements=(), calls_api=True
    )
    yield METHODS | {"fake-text": spec, "fake-json": spec}
    for name in [name for name in sys.modules if name.partition(".")[0] == PACKAGE]:
        del sys.modules[name]


@pytest.fixture(name="make_spec")
def make_spec_fixture(make_settings):
    """Return a builder of the spec of ``fake-<output_format>``, fields overridable by keyword."""

    def make(output_format="text", **changes):
        json_format = output_format == "json"
        values = {
            "name": f"fake-{output_format}",
            "prompt_name": "fake",
            "template_path": "templates/tags.j2",
            "schema_path": "templates/tags.schema.json" if json_format else None,
            "schema_name": "problem_tags" if json_format else None,
            "fields": FIELDS,
            "constants": {"lang_cluster": "Python"},
            "settings": make_settings(output_format=output_format),
            "all_tags": TAGS,
        }
        return LlmMethodSpec(**(values | changes))

    return make


def make_problems(**columns):
    """Return two problems with every field of ``FIELDS``, columns overridable by keyword."""
    values = {
        "prob_desc_description": ["Count the leaves.", "Sum the digits."],
        "prob_desc_notes": ["Leaves have one edge.", "Digits are 0 to 9."],
        "source_code": ["print(n - 1)", "print(sum(map(int, s)))"],
    }
    return pd.DataFrame(values | columns, index=pd.Index([20, 10], name="sample_id"))


def test_a_prompt_snapshot_round_trips_through_json(registry, make_spec):
    prompt = prompt_snapshot(make_spec("json"), registry=registry)

    stored = json.loads(json.dumps(prompt.to_dict()))

    assert PromptSnapshot.from_dict(stored) == prompt


def test_a_method_spec_round_trips_through_json(make_spec):
    spec = make_spec("json", extra={"k": 3, "fields": ["source_code"]})

    stored = json.loads(json.dumps(spec.to_dict()))

    assert LlmMethodSpec.from_dict(stored) == spec
    assert stored["settings"]["output_format"] == "json"


def test_a_method_spec_stores_its_sequences_as_tuples_and_its_mappings_as_copies(make_spec):
    constants = {"lang_cluster": "Python"}
    spec = make_spec(fields=list(FIELDS), all_tags=list(TAGS), constants=constants)
    constants["lang_cluster"] = "C++"

    assert (spec.fields, spec.all_tags) == (FIELDS, TAGS)
    assert spec.constants == {"lang_cluster": "Python"}
    assert spec.extra == {}


@pytest.mark.parametrize("field", ["fields", "all_tags"])
def test_a_method_spec_refuses_a_str_for_a_sequence(make_spec, field):
    with pytest.raises(TypeError, match=f"LlmMethodSpec.{field} must be a tuple of str"):
        make_spec(**{field: "source_code"})


@pytest.mark.parametrize(
    ("output_format", "changes", "match"),
    [
        ("json", {"schema_path": None}, "fake-json answers in JSON: it needs a schema_path"),
        ("json", {"schema_name": None}, "fake-json answers in JSON: it needs a schema_path"),
        ("text", {"schema_path": "tags.schema.json"}, "fake-text answers in text: it sends no"),
        ("text", {"schema_name": "problem_tags"}, "fake-text answers in text: it sends no"),
    ],
)
def test_a_method_spec_has_a_schema_if_and_only_if_it_answers_in_json(
    make_spec, output_format, changes, match
):
    with pytest.raises(ValueError, match=match):
        make_spec(output_format, **changes)


@pytest.mark.parametrize(
    ("field", "path"),
    [
        ("template_path", "/abs/tags.j2"),
        ("template_path", "../other/tags.j2"),
        ("schema_path", "templates/../../tags.schema.json"),
    ],
)
def test_a_method_spec_refuses_a_path_outside_its_package(make_spec, field, path):
    # The paths are stored in the method folder: an absolute path would name this machine.
    with pytest.raises(ValueError, match=f"{field} .* relative to the method's package"):
        make_spec("json", **{field: path})


def test_prompt_content_sha256_hashes_the_template_schema_settings_and_extra(make_settings):
    settings = make_settings(output_format="json")
    extra = {"k": 3}

    digest = prompt_content_sha256(TEMPLATE, SCHEMA, settings, extra)

    assert digest == sha256_json(
        {
            "template": TEMPLATE,
            "response_format": json.dumps(SCHEMA, separators=(",", ":")),
            "settings": settings.to_dict(),
            "extra": extra,
        }
    )
    assert prompt_content_sha256(TEMPLATE, None, make_settings(), {}) != digest


def test_prompt_snapshot_reads_the_template_of_a_text_method_from_its_package(registry, make_spec):
    spec = make_spec()

    prompt = prompt_snapshot(spec, registry=registry)

    assert prompt == PromptSnapshot(
        name="fake",
        version=None,  # filled in by llm score, from the prompt registry
        uri=None,
        content_sha256=prompt_content_sha256(TEMPLATE, None, spec.settings, {}),
        template=TEMPLATE,
        response_format=None,
    )


def test_prompt_snapshot_reads_the_schema_of_a_json_method_from_its_package(registry, make_spec):
    spec = make_spec("json")

    prompt = prompt_snapshot(spec, registry=registry)

    assert prompt.response_format == SCHEMA
    assert prompt.content_sha256 == prompt_content_sha256(TEMPLATE, SCHEMA, spec.settings, {})


@pytest.mark.parametrize("module", [f"{PACKAGE}.methods", PACKAGE])
def test_prompt_snapshot_reads_the_package_of_the_registered_module(
    registry, make_spec, monkeypatch, module
):
    # Python 3.11's importlib.resources.files takes a package only, where 3.12's also takes a
    # plain module: a method registered under a plain module reads its package's files.
    read = []

    def files(name):
        read.append((name, hasattr(importlib.import_module(name), "__path__")))
        return resources.files(name)

    monkeypatch.setattr(prompts, "files", files)
    registry = registry | {"fake-text": dataclasses.replace(registry["fake-text"], module=module)}

    prompt = prompt_snapshot(make_spec(), registry=registry)

    assert (prompt.template, read) == (TEMPLATE, [(PACKAGE, True)])  # a package


def test_prompt_snapshot_reads_the_template_of_a_text_methods_prompt_file(registry, make_spec):
    spec = make_spec(template_path="templates/fake.v1.yaml")

    prompt = prompt_snapshot(spec, registry=registry)

    assert prompt == prompt_snapshot(make_spec(), registry=registry)  # the .j2 file's prompt
    assert (prompt.template, prompt.response_format) == (TEMPLATE, None)


def test_prompt_snapshot_reads_the_template_and_schema_of_a_json_methods_prompt_file(
    registry, make_spec
):
    path = "templates/fake.v2.yaml"
    spec = make_spec("json", template_path=path, schema_path=path)

    prompt = prompt_snapshot(spec, registry=registry)

    assert prompt == prompt_snapshot(make_spec("json"), registry=registry)
    assert (prompt.template, prompt.response_format) == (TEMPLATE, SCHEMA)
    assert list(prompt.response_format) == list(SCHEMA)  # in the schema's own key order


# Prompt files that are not the spec's, for a text or a json spec, by file name: another prompt's,
# another output format's, a json spec's without a schema, and a text spec's with one.
WRONG_PROMPT_FILES = {
    "another name": (
        "text",
        "other.v1.yaml",
        prompt_file_text(1, "text", name="other"),
        "holds the prompt 'other', not 'fake'",
    ),
    "another output format": (
        "text",
        "fake.v1.yaml",
        prompt_file_text(1, "json"),
        "answers in json, but fake-text answers in text",
    ),
    "no schema": (
        "json",
        "fake.v2.yaml",
        prompt_file_text(2, "json"),
        r"keys missing \['response_format'\]",
    ),
    "a schema": (
        "text",
        "fake.v1.yaml",
        prompt_file_text(1, "text", schema=SCHEMA),
        r"unexpected \['response_format'\]",
    ),
}


@pytest.mark.parametrize("case", WRONG_PROMPT_FILES.values(), ids=WRONG_PROMPT_FILES)
def test_prompt_snapshot_refuses_a_prompt_file_that_is_not_the_specs(registry, make_spec, case):
    output_format, file_name, text, error = case
    path = write_prompt_file(resources.files(PACKAGE), file_name, text)
    paths = {"template_path": path} | ({"schema_path": path} if output_format == "json" else {})

    with pytest.raises(ValueError, match=error):
        prompt_snapshot(make_spec(output_format, **paths), registry=registry)


def test_prompt_snapshot_refuses_a_schema_outside_a_json_methods_prompt_file(registry, make_spec):
    spec = make_spec("json", template_path="templates/fake.v2.yaml")

    with pytest.raises(ValueError, match="its schema_path must be its prompt file"):
        prompt_snapshot(spec, registry=registry)


def test_prompt_snapshot_refuses_an_unregistered_method(make_spec):
    with pytest.raises(KeyError, match="fake-text"):
        prompt_snapshot(make_spec())


def test_build_requests_renders_one_request_per_problem_in_their_order(registry, make_spec):
    spec = make_spec()

    requests = build_requests(make_problems(), spec, prompt_snapshot(spec, registry=registry))

    assert [request.sample_id for request in requests] == [20, 10]
    assert requests[0] == LlmRequest(
        sample_id=20,
        method="fake-text",
        input=(
            "Tag this Python solution.\n"
            "Description: Count the leaves.\n"
            "Notes: Leaves have one edge.\n"
            "\n"
            "print(n - 1)\n"
            "\n"
            "Choose from: dp, math, trees"
        ),
        settings=spec.settings,
        schema=None,
        schema_name=None,
    )


def test_build_requests_renders_missing_fields_as_none_and_code_verbatim(registry, make_spec):
    # A missing field (NaN, None, empty or blank) prints None, so the block of
    # labelled lines keeps its shape; code that looks like Jinja, non-ASCII text and Windows line
    # endings reach the prompt as submitted; the hash is the same on every run and machine.
    code = "s = '{{ x }} {% if y %} {# z #}'\r\nprint('héllo → 日本')  # Ünïcode\n"
    index = pd.Index([7, 3, 5, 9], name="sample_id")
    problems = pd.DataFrame(
        {
            "prob_desc_description": ["Count the leaves.", "  \n\t", "", "Sum."],
            "prob_desc_notes": pd.Series([np.nan, None, pd.NA, " "], index, dtype=object),
            "source_code": [code, "print(1)", "print(2)", "print(3)"],
        },
        index=index,
    )
    spec = make_spec()
    prompt = prompt_snapshot(spec, registry=registry)

    requests = build_requests(problems, spec, prompt)

    assert requests[0].input == (
        "Tag this Python solution.\n"
        "Description: Count the leaves.\n"
        "Notes: None\n"
        "\n"
        f"{code}"
        "\n"
        "\n"
        "Choose from: dp, math, trees"
    )
    assert [request.input.splitlines()[1:3] for request in requests[1:]] == [
        ["Description: None", "Notes: None"],
        ["Description: None", "Notes: None"],
        ["Description: Sum.", "Notes: None"],
    ]
    assert requests[0].sha256() == (
        "26203790ecadc75500f9c68ac372441dbd8ba482e75ae713e56138efb00d60da"
    )
    assert [request.sha256() for request in build_requests(problems, spec, prompt)] == [
        request.sha256() for request in requests
    ]


def test_build_requests_renders_constants_the_tag_list_and_each_problems_context(
    registry, make_spec
):
    spec = make_spec(fields=("source_code",), constants={"lang_cluster": "Python", "k": "2"})
    template = "{{ k }} {{ lang_cluster }} {{ all_tags[0] }} {{ examples }}: {{ source_code }}"
    prompt = dataclasses.replace(
        prompt_snapshot(spec, registry=registry),
        template=template,
        content_sha256=prompt_content_sha256(template, None, spec.settings, {}),
    )
    context = {20: {"examples": "e20"}, 10: {"examples": "e10"}}

    requests = build_requests(make_problems(), spec, prompt, context)

    assert [request.input for request in requests] == [
        "2 Python dp e20: print(n - 1)",
        "2 Python dp e10: print(sum(map(int, s)))",
    ]


def test_build_requests_gives_a_json_method_its_schema_and_its_name(registry, make_spec):
    spec = make_spec("json")

    (request, _) = build_requests(make_problems(), spec, prompt_snapshot(spec, registry=registry))

    assert (request.schema, request.schema_name) == (SCHEMA, "problem_tags")
    assert request.settings.output_format == "json"


@pytest.mark.parametrize(
    ("changes", "context", "match"),
    [
        ({"constants": {"source_code": "x"}}, None, "'source_code' is both a field and a constant"),
        ({"constants": {"all_tags": "x"}}, None, "'all_tags' is both a constant and the tag list"),
        ({"fields": ("all_tags",)}, None, "'all_tags' is both a field and the tag list"),
        ({}, {20: {"source_code": "x"}, 10: {}}, "'source_code' is both a field and a context key"),
        (
            {},
            {20: {}, 10: {"lang_cluster": "x"}},
            "'lang_cluster' is both a constant and a context",
        ),
    ],
)
def test_build_requests_refuses_two_variables_of_one_name(
    registry, make_spec, changes, context, match
):
    spec = make_spec()
    prompt = prompt_snapshot(spec, registry=registry)

    with pytest.raises(ValueError, match=match):
        build_requests(make_problems(), dataclasses.replace(spec, **changes), prompt, context)


def test_build_requests_refuses_a_context_that_lacks_a_problem(registry, make_spec):
    spec = make_spec()

    with pytest.raises(ValueError, match=r"The context lacks the problems \[10\]"):
        build_requests(
            make_problems(), spec, prompt_snapshot(spec, registry=registry), {20: {"k": 1}}
        )


def test_build_requests_refuses_problems_without_a_field(registry, make_spec):
    spec = make_spec()
    problems = make_problems().drop(columns="prob_desc_notes")

    with pytest.raises(ValueError, match=r"lack the fields \['prob_desc_notes'\]"):
        build_requests(problems, spec, prompt_snapshot(spec, registry=registry))


def test_build_requests_refuses_a_template_variable_that_was_not_given(registry, make_spec):
    spec = make_spec(fields=("source_code",))

    with pytest.raises(jinja2.UndefinedError, match="'prob_desc_description' is undefined"):
        build_requests(make_problems(), spec, prompt_snapshot(spec, registry=registry))


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other"},
        {"template": TEMPLATE + "Answer now.\n"},
        {"response_format": SCHEMA},
        {"content_sha256": "0" * 64},
    ],
)
def test_build_requests_refuses_the_prompt_of_another_spec(registry, make_spec, changes):
    # The snapshot must be the spec's own: its content hash covers the template, the schema,
    # the settings and the extra values.
    spec = make_spec()
    prompt = dataclasses.replace(prompt_snapshot(spec, registry=registry), **changes)

    with pytest.raises(ValueError, match="is not the prompt of fake-text"):
        build_requests(make_problems(), spec, prompt)


def test_build_requests_refuses_a_spec_whose_settings_or_extra_changed(registry, make_spec):
    spec = make_spec()
    prompt = prompt_snapshot(spec, registry=registry)

    with pytest.raises(ValueError, match="is not the prompt of fake-text"):
        build_requests(make_problems(), dataclasses.replace(spec, extra={"k": 1}), prompt)


def test_a_request_hash_covers_only_what_shapes_the_answer(make_settings):
    request = LlmRequest(
        sample_id=1,
        method="fake-json",
        input="Tag this.",
        settings=make_settings(output_format="json"),
        schema=SCHEMA,
        schema_name="problem_tags",
    )

    assert request.sha256() == sha256_json(
        {
            "provider": "openai",
            "model": "model-1",
            "input": "Tag this.",
            "schema": json.dumps(SCHEMA, separators=(",", ":")),
            "schema_name": "problem_tags",
            "temperature": 0.3157894736842105,
            "top_p": 1.0,
            "reasoning_effort": "none",
            "max_output_tokens": 512,
        }
    )
    # Which problem and which method a request is for does not shape its answer.
    assert dataclasses.replace(request, sample_id=2, method="other").sha256() == request.sha256()


@pytest.mark.parametrize(
    "changes",
    [
        {"input": "Tag that."},
        {"schema": None},
        {"schema_name": "tags"},
    ],
)
def test_a_request_hash_changes_with_the_input_and_the_schema(make_settings, changes):
    request = LlmRequest(1, "fake-json", "Tag this.", make_settings(), SCHEMA, "problem_tags")

    assert dataclasses.replace(request, **changes).sha256() != request.sha256()


def test_the_request_and_content_hashes_keep_the_order_of_the_schemas_properties(make_settings):
    # Structured outputs write the answer's keys in the order of the schema's properties: a
    # model that reasons before it tags and one that tags first are asked two different things.
    reasoning_first = {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}, "tags": SCHEMA["properties"]["tags"]},
        "required": ["reasoning", "tags"],
        "additionalProperties": False,
    }
    properties = reasoning_first["properties"]
    tags_first = reasoning_first | {"properties": dict(reversed(properties.items()))}
    settings = make_settings(output_format="json")
    request = LlmRequest(1, "fake-json", "Tag this.", settings, reasoning_first, "problem_tags")

    assert tags_first == reasoning_first  # equal as dicts, which ignore the order
    assert dataclasses.replace(request, schema=tags_first).sha256() != request.sha256()
    assert prompt_content_sha256(TEMPLATE, tags_first, settings, {}) != prompt_content_sha256(
        TEMPLATE, reasoning_first, settings, {}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "other"},
        {"model": "model-2"},
        {"reasoning_effort": None},
        {"temperature": 0.0},
        {"top_p": 0.9},
        {"max_output_tokens": 256},
    ],
)
def test_a_request_hash_changes_with_each_answer_shaping_setting(make_settings, changes):
    request = LlmRequest(1, "fake-text", "Tag this.", make_settings(), None, None)
    changed = dataclasses.replace(request, settings=make_settings(**changes))

    assert changed.sha256() != request.sha256()


@pytest.mark.usefixtures("fake_provider")
def test_a_request_body_is_the_one_its_provider_builds(make_settings):
    request = LlmRequest(1, "fake-text", "Tag this.", make_settings(provider="fake"), None, None)

    assert request.body() == {"input": "Tag this."}


def test_field_values_and_their_hashes_show_a_missing_field_as_none():
    problems = pd.DataFrame(
        {"prob_desc_description": ["Sum.", " ", None], "prob_desc_notes": [np.nan, "", "Note."]},
        index=pd.Index([4, 2, 8], name="sample_id"),
    )
    fields = ("prob_desc_description", "prob_desc_notes")

    values = field_values(problems, fields)
    hashes = field_sha256s(problems, fields)

    assert values == {
        4: {"prob_desc_description": "Sum.", "prob_desc_notes": None},
        2: {"prob_desc_description": None, "prob_desc_notes": None},
        8: {"prob_desc_description": None, "prob_desc_notes": "Note."},
    }
    assert list(hashes) == [4, 2, 8]
    assert hashes == {sample_id: sha256_json(shown) for sample_id, shown in values.items()}
    with pytest.raises(ValueError, match=r"lack the fields \['source_code'\]"):
        field_sha256s(problems, ("source_code",))
