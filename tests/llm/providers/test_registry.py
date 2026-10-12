"""Tests for the registry of LLM providers: one adapter module per provider."""

import json
import subprocess
import sys
import tomllib

import pytest

from codeforces_tags.config import PYPROJECT_FILE
from codeforces_tags.llm.providers import PROVIDERS, provider_module, sdk_modules


def test_openai_is_a_provider():
    assert PROVIDERS["openai"] == "codeforces_tags.llm.providers.openai"


def test_provider_module_imports_the_adapter_of_a_provider():
    adapter = provider_module("openai")

    assert adapter.__name__ == "codeforces_tags.llm.providers.openai"
    assert adapter.SDK_MODULE == "openai"


def test_provider_module_refuses_a_provider_that_has_no_adapter():
    with pytest.raises(ValueError, match=r"'anthropic' is not a provider: expected one of \["):
        provider_module("anthropic")


@pytest.mark.usefixtures("fake_provider")
def test_sdk_modules_lists_the_sdk_of_every_provider():
    names = sdk_modules()

    assert len(names) == len(PROVIDERS)
    assert {"openai", "fake_sdk"} <= set(names)


def test_sdk_modules_imports_no_sdk():
    # The conformance check calls it to know which modules a prediction path must not import:
    # it reads each adapter's SDK_MODULE, and no adapter imports its SDK at module level.
    code = (
        "import json, sys; from codeforces_tags.llm.providers import sdk_modules; "
        "names = sdk_modules(); print(json.dumps([names, sorted(set(names) & set(sys.modules))]))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    names, imported = json.loads(result.stdout)
    assert "openai" in names
    assert imported == []


def test_every_provider_sdk_fails_lint_when_imported_at_module_level():
    # ruff's TID253 keeps a provider's SDK off the prediction path: only its adapter's ask
    # imports it, so a method loads and predicts without the SDK installed.
    pyproject = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))
    banned = pyproject["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-module-level-imports"]

    assert set(sdk_modules()) <= set(banned)
