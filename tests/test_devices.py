"""Tests for the device a method that runs torch runs on: ``$CODEFORCES_DEVICE``, at call time."""

import re

import pytest

from codeforces_tags.config import (
    DEVICE_ENV_VAR,
    DEVICES,
    LEGACY_DEVICE_ENV_VAR,
    RECORDED_SETTINGS,
)
from codeforces_tags.devices import requested_device

# The refusal of the variable the F2LLM methods read before: it names the new one.
LEGACY_REFUSAL = (
    "$CODEFORCES_F2LLM_DEVICE is no longer read: unset it, and name the device in "
    "$CODEFORCES_DEVICE instead, one of ('auto', 'cpu', 'cuda')."
)


@pytest.fixture(name="no_device_variable", autouse=True)
def no_device_variable_fixture(monkeypatch):
    """Start every test with neither device variable set, whatever the shell exports."""
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)


def test_the_settings_name_the_variable_its_three_values_and_the_former_variable():
    assert (DEVICE_ENV_VAR, LEGACY_DEVICE_ENV_VAR) == (
        "CODEFORCES_DEVICE",
        "CODEFORCES_F2LLM_DEVICE",
    )
    assert DEVICES == ("auto", "cpu", "cuda")
    # The device shapes no recipe: a run record holds the device, never the variable.
    assert not {"DEVICE_ENV_VAR", "DEVICES", "LEGACY_DEVICE_ENV_VAR"} & set(RECORDED_SETTINGS)


def test_unset_the_device_is_auto_or_the_callers_default():
    assert requested_device() == "auto"
    assert requested_device("cpu") == "cpu"
    assert requested_device(default="cuda") == "cuda"


@pytest.mark.parametrize("device", DEVICES)
def test_the_variable_names_the_device_whatever_the_default(monkeypatch, device):
    monkeypatch.setenv(DEVICE_ENV_VAR, device)

    assert requested_device() == device
    assert requested_device(default="cpu") == device


def test_the_variable_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    before = requested_device()

    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")

    assert (before, requested_device()) == ("cpu", "cuda")


def test_an_empty_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "")

    assert requested_device(default="cpu") == "cpu"


@pytest.mark.parametrize("value", ["CUDA", " cuda", "cuda ", "gpu", "cuda:0"])
def test_any_other_value_of_the_variable_is_refused_as_written(monkeypatch, value):
    monkeypatch.setenv(DEVICE_ENV_VAR, value)

    with pytest.raises(ValueError) as refused:
        requested_device()

    assert str(refused.value) == (
        f"$CODEFORCES_DEVICE is {value!r}, not a device setting: expected one of "
        "('auto', 'cpu', 'cuda')."
    )


def test_a_default_that_is_no_device_setting_is_refused_even_beside_a_valid_variable(
    monkeypatch,
):
    message = re.escape(
        "The default device is 'gpu', not a device setting: expected one of "
        "('auto', 'cpu', 'cuda')."
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        requested_device(default="gpu")

    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")

    with pytest.raises(ValueError, match=f"^{message}$"):
        requested_device(default="gpu")


def test_the_former_f2llm_variable_is_refused_naming_the_new_one_even_beside_it(monkeypatch):
    monkeypatch.setenv(LEGACY_DEVICE_ENV_VAR, "cpu")
    with pytest.raises(ValueError, match=f"^{re.escape(LEGACY_REFUSAL)}$"):
        requested_device()

    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")  # the same device under the new name: still refused

    with pytest.raises(ValueError, match=f"^{re.escape(LEGACY_REFUSAL)}$"):
        requested_device(default="cpu")


def test_an_empty_former_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv(LEGACY_DEVICE_ENV_VAR, "")
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")

    assert requested_device() == "cuda"


def test_the_module_imports_only_the_standard_library_and_the_config(agent_python):
    code = (
        "import sys; before = set(sys.modules); import codeforces_tags.devices; "
        "new = set(sys.modules) - before; "
        "print(sorted(m for m in new if m.startswith('codeforces_tags'))); "
        "print(sorted({m.partition('.')[0] for m in new} - sys.stdlib_module_names "
        "- {'codeforces_tags'}))"
    )

    output, _ = agent_python(code)

    assert output.splitlines() == [
        "['codeforces_tags', 'codeforces_tags.config', 'codeforces_tags.devices']",
        "[]",
    ]
