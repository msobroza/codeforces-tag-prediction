"""Tests for the settings that shape an LLM's answer."""

import dataclasses
import json

import pytest

from codeforces_tags.llm.settings import OUTPUT_FORMATS, LlmSettings

FIELDS = [
    "provider",
    "model",
    "reasoning_effort",
    "temperature",
    "top_p",
    "max_output_tokens",
    "output_format",
]


def test_llm_settings_hold_only_the_fields_that_shape_an_answer():
    # A request's hash covers these fields, so a transport setting added here would change every
    # recorded hash: those stay in config.LLM_*, outside the settings.
    assert [field.name for field in dataclasses.fields(LlmSettings)] == FIELDS


def test_the_output_formats_are_text_then_json():
    # A prompt's versions are registered in this order: version 1 text, version 2 json.
    assert OUTPUT_FORMATS == ("text", "json")


def test_llm_settings_are_frozen(make_settings):
    settings = make_settings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model = "model-2"


def test_to_dict_lists_every_field_in_order(make_settings):
    assert list(make_settings().to_dict()) == FIELDS


def test_llm_settings_round_trip_through_json(make_settings):
    settings = make_settings(output_format="json")

    stored = json.loads(json.dumps(settings.to_dict()))

    assert LlmSettings.from_dict(stored) == settings


def test_llm_settings_may_leave_out_the_effort_the_temperature_and_the_top_p(make_settings):
    # None means "send nothing": a model without reasoning takes no effort, and one with it takes
    # no temperature.
    settings = make_settings(reasoning_effort=None, temperature=None, top_p=None)

    stored = json.loads(json.dumps(settings.to_dict()))

    assert stored["temperature"] is None
    assert LlmSettings.from_dict(stored) == settings


@pytest.mark.parametrize("output_format", OUTPUT_FORMATS)
def test_llm_settings_accept_every_output_format(make_settings, output_format):
    assert make_settings(output_format=output_format).output_format == output_format


def test_llm_settings_refuse_an_output_format_outside_the_known_ones(make_settings):
    with pytest.raises(ValueError, match="'xml' is not an output format"):
        make_settings(output_format="xml")


def test_from_dict_refuses_an_output_format_outside_the_known_ones(make_settings):
    stored = make_settings().to_dict() | {"output_format": "yaml"}

    with pytest.raises(ValueError, match="'yaml' is not an output format"):
        LlmSettings.from_dict(stored)


def test_from_dict_refuses_a_missing_field_and_an_unknown_one(make_settings):
    # A settings dict that does not match the fields would hash differently: never fill in or
    # drop a field silently.
    stored = make_settings().to_dict()
    without_tokens = {name: value for name, value in stored.items() if name != "max_output_tokens"}

    with pytest.raises(TypeError, match="max_output_tokens"):
        LlmSettings.from_dict(without_tokens)
    with pytest.raises(TypeError, match="seed"):
        LlmSettings.from_dict(stored | {"seed": 0})
