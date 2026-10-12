"""Tests for the prompt registry: each prompt's versions, registered in a fixed order."""

import dataclasses
import hashlib
import json
import threading

import mlflow
import pytest

from codeforces_tags.config import ALL_TAGS
from codeforces_tags.llm.prompts import LlmMethodSpec, PromptSnapshot, prompt_content_sha256
from codeforces_tags.llm.settings import OUTPUT_FORMATS, LlmSettings
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.prompts import (
    CONTENT_TAG,
    LINKED_PROMPTS_TAG,
    PromptVersionContent,
    link_prompt_version,
    register_prompt_versions,
    register_prompts,
    set_prompt_aliases,
)

PROMPT = "tags"
TEMPLATE = 'Tag this solution.\n\n{{ source_code }}\n\nChoose from: {{ all_tags | join(", ") }}'
# The schema of the json method's answer: a list of tags of ALL_TAGS.
SCHEMA = {
    "type": "object",
    "required": ["tags"],
    "properties": {"tags": {"type": "array", "items": {"enum": list(ALL_TAGS)}}},
}


def member(output_format, template=TEMPLATE):
    """Return the spec of the method ``tags-<output_format>`` and the snapshot of its prompt."""
    json_format = output_format == "json"
    settings = LlmSettings(
        provider="openai",
        model="gpt-6-luna",
        reasoning_effort="none",
        temperature=0.5,
        top_p=1.0,
        max_output_tokens=256,
        output_format=output_format,
    )
    spec = LlmMethodSpec(
        name=f"tags-{output_format}",
        prompt_name=PROMPT,
        template_path="tags.j2",
        schema_path="tags.schema.json" if json_format else None,
        schema_name="problem_tags" if json_format else None,
        fields=("source_code",),
        constants={},
        settings=settings,
        all_tags=ALL_TAGS,
    )
    schema = SCHEMA if json_format else None
    content = prompt_content_sha256(template, schema, settings, spec.extra)
    return spec, PromptSnapshot(PROMPT, None, None, content, template, schema)


def family(template=TEMPLATE):
    """Return the prompt's family in registration order: the text method, then the json one."""
    return [member(output_format, template) for output_format in OUTPUT_FORMATS]


def stored_versions():
    """Return the number and content tag of each version of the prompt in the current store."""
    found = mlflow.MlflowClient().search_prompt_versions(PROMPT)
    return sorted((version.version, version.tags[CONTENT_TAG]) for version in found)


def test_register_prompts_registers_the_family_in_its_order_with_no_api_key(store, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    [(text, text_prompt), (json_spec, json_prompt)] = family()
    started = []

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda thread: started.append(thread.name))
        versions = register_prompts(PROMPT, family(), **store)

    assert list(versions) == ["tags-text", "tags-json"]
    assert [version.uri for version in versions.values()] == ["prompts:/tags/1", "prompts:/tags/2"]
    assert stored_versions() == [(1, text_prompt.content_sha256), (2, json_prompt.content_sha256)]
    assert {version.template for version in versions.values()} == {TEMPLATE}
    assert versions["tags-text"].response_format is None
    assert versions["tags-json"].response_format == SCHEMA
    # A flat dict: MLflow's PromptModelConfig would drop the reasoning effort.
    assert versions["tags-text"].model_config == text.settings.to_dict()
    assert versions["tags-json"].model_config["reasoning_effort"] == "none"
    assert versions["tags-json"].model_config == json_spec.settings.to_dict()
    assert versions["tags-json"].commit_message == "tags-json: the json output format"
    # Neither a thread (register_prompt links the prompt to an experiment in one) nor a version's
    # tags on the prompt itself (register_prompt copies them there).
    assert not started
    assert mlflow.MlflowClient().get_prompt(PROMPT).tags == {}


def test_register_prompts_a_second_time_registers_nothing_and_gives_the_same_versions(store):
    first = register_prompts(PROMPT, family(), **store)

    second = register_prompts(PROMPT, family(), **store)

    assert {name: version.uri for name, version in second.items()} == {
        name: version.uri for name, version in first.items()
    }
    assert second["tags-json"].response_format == SCHEMA
    assert [number for number, _ in stored_versions()] == [1, 2]


def test_register_prompts_completes_a_family_whose_registration_stopped_early(store):
    register_prompts(PROMPT, family()[:1], **store)

    versions = register_prompts(PROMPT, family(), **store)

    assert [version.version for version in versions.values()] == [1, 2]
    assert [number for number, _ in stored_versions()] == [1, 2]


def test_register_prompts_refuses_a_changed_prompt(store):
    register_prompts(PROMPT, family(), **store)

    with pytest.raises(ValueError, match="a new wording is a new version or a new prompt"):
        register_prompts(PROMPT, family(TEMPLATE + "\nBe brief."), **store)
    assert [number for number, _ in stored_versions()] == [1, 2]


def test_register_prompts_refuses_a_version_that_is_not_in_the_family(store):
    register_prompts(PROMPT, family(), **store)
    mlflow.MlflowClient().create_prompt_version(PROMPT, "Another prompt, with no content tag.")

    with pytest.raises(ValueError, match=r"holds the versions \[1, 2, 3\]"):
        register_prompts(PROMPT, family(), **store)


def test_register_prompts_refuses_versions_registered_in_another_order(store):
    register_prompts(PROMPT, family()[::-1], **store)

    with pytest.raises(ValueError, match="whose content hashes are not"):
        register_prompts(PROMPT, family(), **store)


def test_register_prompts_refuses_an_empty_family(store):
    with pytest.raises(ValueError, match="no method"):
        register_prompts(PROMPT, [], **store)


def test_register_prompts_refuses_a_method_named_twice(store):
    with pytest.raises(ValueError, match="names a method twice"):
        register_prompts(PROMPT, family()[:1] * 2, **store)


def test_register_prompts_refuses_a_snapshot_that_is_not_the_prompt_of_its_spec(store):
    spec, prompt = member("text")
    changed = dataclasses.replace(prompt, template=TEMPLATE + "\nBe brief.")

    with pytest.raises(ValueError, match="is not the prompt of tags-text"):
        register_prompts(PROMPT, [(spec, changed)], **store)


def test_register_prompts_refuses_a_method_of_another_prompt(store):
    with pytest.raises(ValueError, match="belongs to the prompt 'tags', not 'other'"):
        register_prompts("other", family(), **store)


def test_set_prompt_aliases_names_each_version_after_its_output_format(store):
    versions = register_prompts(PROMPT, family(), **store)
    specs = {spec.name: spec for spec, _ in family()}

    set_prompt_aliases(PROMPT, versions, specs, **store)
    set_prompt_aliases(PROMPT, versions, specs, **store)

    client = mlflow.MlflowClient()
    assert client.get_prompt_version_by_alias(PROMPT, "text").version == 1
    assert client.get_prompt_version_by_alias(PROMPT, "json").version == 2


def test_set_prompt_aliases_refuses_a_version_without_its_spec(store):
    versions = register_prompts(PROMPT, family(), **store)
    [(text, _), _] = family()

    with pytest.raises(ValueError, match=r"\['tags-json'\] have no spec"):
        set_prompt_aliases(PROMPT, versions, {text.name: text}, **store)
    with pytest.raises(mlflow.MlflowException):
        mlflow.MlflowClient().get_prompt_version_by_alias(PROMPT, "text")


def test_set_prompt_aliases_refuses_a_version_of_another_prompt(store):
    versions = register_prompts(PROMPT, family(), **store)
    specs = {spec.name: spec for spec, _ in family()}

    with pytest.raises(ValueError, match="are not versions of 'other'"):
        set_prompt_aliases("other", versions, specs, **store)


def plain_version(template="Is it {{ tag }}?", **changes):
    """Return one version of a prompt as plain data: a free-text answer, with a settings tag."""
    fields = {
        "template": template,
        "response_format": None,
        "model_config": {"provider": "typesafe", "model": "jev-1.13.0"},
        "content_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "description": "the words of the test",
        "tags": {"settings_sha256": "f" * 64},
    } | changes
    return PromptVersionContent(**fields)


def test_register_prompt_versions_registers_plain_data_versions_in_order(store):
    first, second = plain_version(), plain_version("Is it {{ tag }}? Say yes or no.")

    versions = register_prompt_versions(PROMPT, [first, second], **store)

    assert [version.uri for version in versions] == ["prompts:/tags/1", "prompts:/tags/2"]
    assert stored_versions() == [(1, first.content_sha256), (2, second.content_sha256)]
    assert versions[0].template == first.template
    assert versions[0].response_format is None
    assert versions[0].model_config == {"provider": "typesafe", "model": "jev-1.13.0"}
    assert versions[0].commit_message == "the words of the test"
    assert versions[1].tags == {CONTENT_TAG: second.content_sha256, "settings_sha256": "f" * 64}
    assert mlflow.MlflowClient().get_prompt(PROMPT).tags == {}


def test_register_prompt_versions_a_second_time_only_checks_them(store):
    first = register_prompt_versions(PROMPT, [plain_version()], **store)

    second = register_prompt_versions(PROMPT, [plain_version()], **store)

    assert [version.uri for version in second] == [version.uri for version in first]
    assert second[0].tags["settings_sha256"] == "f" * 64
    assert [number for number, _ in stored_versions()] == [1]


def test_register_prompt_versions_refuses_a_changed_version(store):
    register_prompt_versions(PROMPT, [plain_version()], **store)

    with pytest.raises(ValueError, match="a new wording is a new version or a new prompt"):
        register_prompt_versions(PROMPT, [plain_version("Is it {{ tag }}, really?")], **store)
    assert [number for number, _ in stored_versions()] == [1]


def test_register_prompt_versions_refuses_no_version(store):
    with pytest.raises(ValueError, match="no version to register"):
        register_prompt_versions(PROMPT, [], **store)


def test_register_prompt_versions_refuses_a_tag_that_would_replace_the_content_hash(store):
    clashing = plain_version(tags={CONTENT_TAG: "0" * 64})

    with pytest.raises(ValueError, match="content_sha256 is the version's content hash"):
        register_prompt_versions(PROMPT, [clashing], **store)
    assert mlflow.MlflowClient().get_prompt(PROMPT) is None


def linked_prompts(tags):
    """Return the prompt versions that a run's or a model's tags say are linked to it."""
    return json.loads(tags.get(LINKED_PROMPTS_TAG, "[]"))


def test_link_prompt_version_links_the_model_and_its_run_once(store, logged_model):
    [version] = register_prompt_versions(PROMPT, [plain_version()], **store)

    link_prompt_version(version, logged_model.model_id, **store)
    link_prompt_version(version, logged_model.model_id, **store)

    client = mlflow.MlflowClient()
    expected = [{"name": PROMPT, "version": "1"}]
    assert linked_prompts(client.get_logged_model(logged_model.model_id).tags) == expected
    assert linked_prompts(client.get_run(logged_model.source_run_id).data.tags) == expected


def test_link_prompt_version_refuses_a_model_logged_outside_a_run(store):
    [version] = register_prompt_versions(PROMPT, [plain_version()], **store)
    model = mlflow.initialize_logged_model(name="tagging_method", experiment_id=use_store(**store))

    with pytest.raises(ValueError, match="was logged outside a run"):
        link_prompt_version(version, model.model_id, **store)
    assert linked_prompts(mlflow.MlflowClient().get_logged_model(model.model_id).tags) == []
