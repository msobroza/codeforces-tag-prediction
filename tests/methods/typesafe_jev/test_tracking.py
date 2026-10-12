"""Tests for the TypeSafe Jev methods' prompt in MLflow's prompt registry, on a temporary store.

Nothing is sent: the registry is the test's MLflow store, and registering needs no key.
"""

import hashlib
import json
import re
from importlib.resources import files

import mlflow
import pytest

from codeforces_tags.methods.typesafe_jev import tracking
from codeforces_tags.methods.typesafe_jev.settings import (
    MODEL,
    PROMPT,
    PROMPT_NAME,
    PROMPT_VERSION,
    SETTINGS,
)
from codeforces_tags.provenance import sha256_json
from codeforces_tags.tracking.prompts import (
    CONTENT_TAG,
    LINKED_PROMPTS_TAG,
    PromptVersionContent,
    register_prompt_versions,
)

# The hash of SETTINGS that both frozen recipes and the frozen methods' jev.json hold
# (test_settings.py pins it).
SETTINGS_SHA256 = "8eda74ec4c3f3f8e38c952ac82b334c8888b2e08b0fe4ad6f1df1ebae2a2c7b7"
# The name of a version file of the prompt, which gives its version.
VERSION_FILE = re.compile(rf"{re.escape(PROMPT_NAME)}\.v(\d+)\.yaml")


def prompt_files():
    """Return the bytes of each version file of the prompt in the package, by version."""
    folder = files("codeforces_tags.methods.typesafe_jev").joinpath("templates")
    matches = [VERSION_FILE.fullmatch(item.name) for item in folder.iterdir()]
    found = {int(match.group(1)): folder.joinpath(match.group(0)) for match in matches if match}
    return {version: found[version].read_bytes() for version in sorted(found)}


def linked(tags):
    """Return the prompt versions that a run's or a model's tags say are linked to it."""
    return json.loads(tags.get(LINKED_PROMPTS_TAG, "[]"))


def test_the_prompt_has_one_version_per_prompt_file_in_version_order():
    contents = tracking.prompt_versions()

    texts = prompt_files()
    assert list(texts) == list(range(1, len(contents) + 1))
    assert [content.content_sha256 for content in contents] == [
        hashlib.sha256(text).hexdigest() for text in texts.values()
    ]
    assert [content.template for content in contents] == [
        text.decode("utf-8") for text in texts.values()
    ]


def test_version_1_is_the_file_the_methods_read_with_their_model_and_settings():
    [version_1] = tracking.prompt_versions()

    assert PROMPT_VERSION == 1
    assert version_1 == PromptVersionContent(
        template=PROMPT.text,
        response_format=None,
        model_config={"provider": "typesafe", "model": MODEL},
        content_sha256=PROMPT.sha256,
        description=(
            "jev-tags v1: the words of jev-statement and jev-statement-code, one Noul question "
            "per focus tag"
        ),
        tags={"settings_sha256": SETTINGS_SHA256},
    )
    assert sha256_json(SETTINGS) == SETTINGS_SHA256


def test_register_jev_prompt_registers_version_1_once_with_no_key(store, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    first = tracking.register_jev_prompt(**store)
    second = tracking.register_jev_prompt(**store)

    assert first.uri == second.uri == "prompts:/jev-tags/1"
    assert first.template == PROMPT.text
    assert first.response_format is None
    assert first.model_config == {"provider": "typesafe", "model": MODEL}
    assert first.tags == {CONTENT_TAG: PROMPT.sha256, "settings_sha256": SETTINGS_SHA256}
    assert len(mlflow.MlflowClient().search_prompt_versions(PROMPT_NAME)) == 1


def test_register_jev_prompt_refuses_a_registry_that_holds_other_words(store):
    other = PromptVersionContent(
        template="Other words.",
        response_format=None,
        model_config={},
        content_sha256="0" * 64,
        description="other words",
    )
    register_prompt_versions(PROMPT_NAME, [other], **store)

    with pytest.raises(ValueError, match="a new wording is a new version or a new prompt"):
        tracking.register_jev_prompt(**store)


def test_link_jev_prompt_registers_the_prompt_and_links_the_model_and_its_run_once(
    store, logged_model
):
    version = tracking.link_jev_prompt(logged_model.model_id, **store)
    again = tracking.link_jev_prompt(logged_model.model_id, **store)

    assert version.uri == again.uri == "prompts:/jev-tags/1"
    client = mlflow.MlflowClient()
    expected = [{"name": PROMPT_NAME, "version": "1"}]
    assert linked(client.get_logged_model(logged_model.model_id).tags) == expected
    assert linked(client.get_run(logged_model.source_run_id).data.tags) == expected
    assert len(client.search_prompt_versions(PROMPT_NAME)) == 1
