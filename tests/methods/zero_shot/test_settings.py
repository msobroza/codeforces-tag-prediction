"""Tests for the settings of the zero-shot methods."""

import numpy as np
import pytest

from codeforces_tags.config import (
    LABEL_AND_SPLIT_COLUMNS,
    LLM_PRICES,
    PROJECT_ROOT,
    RAW_INPUT_COLUMNS,
)
from codeforces_tags.llm import settings as llm_settings
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.zero_shot import settings
from codeforces_tags.provenance import sha256_json

PAPER_NOTES = PROJECT_ROOT / "docs" / "xcodeeval-paper-notes.md"


def table_3_row(label):
    """Return the cells of the row of the paper notes' Table 3 that starts with ``label``."""
    lines = PAPER_NOTES.read_text(encoding="utf-8").splitlines()
    row = next(line for line in lines if line.startswith(f"| {label} |"))
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_the_request_settings_are_the_ones_every_request_hashes():
    assert settings.PROVIDER == "openai"
    assert settings.MODEL == "gpt-6-luna"
    assert settings.REASONING_EFFORT == "none"
    assert settings.MAX_OUTPUT_TOKENS == 512


def test_the_model_is_priced_in_the_project_config():
    assert settings.MODEL in LLM_PRICES


def test_the_temperature_is_the_authors_point_of_a_linspace_and_not_the_papers_rounding():
    # The authors' scripts draw their temperatures from np.linspace(0, 2, 20); the paper's
    # "0.325" misreports the point they used.
    assert settings.TEMPERATURE == 0.3157894736842105
    assert float(np.linspace(0, 2, 20)[3]) == settings.TEMPERATURE


def test_top_p_is_a_float_because_one_and_one_point_zero_hash_differently():
    assert settings.TOP_P == 1.0
    assert isinstance(settings.TOP_P, float)
    assert sha256_json({"top_p": 1}) != sha256_json({"top_p": settings.TOP_P})


def test_the_method_names_are_the_two_settings_in_both_output_formats():
    assert settings.SETTINGS == ("code2tag", "descode2tag")
    assert settings.OUTPUT_FORMATS is llm_settings.OUTPUT_FORMATS
    assert settings.OUTPUT_FORMATS == ("text", "json")
    assert settings.METHOD_NAMES == (
        "code2tag-text",
        "code2tag-json",
        "descode2tag-text",
        "descode2tag-json",
    )


@pytest.mark.parametrize("name", settings.METHOD_NAMES)
def test_a_method_name_is_a_registry_key_that_splits_into_its_setting_and_output_format(name):
    setting, output_format = name.rsplit("-", 1)

    assert setting in settings.SETTINGS
    assert output_format in settings.OUTPUT_FORMATS
    assert method_dir(name).name == name


def test_each_setting_reads_only_its_allow_listed_fields():
    statement_parts = ("description", "input_spec", "output_spec", "notes")

    assert tuple(settings.FIELDS) == settings.SETTINGS
    assert settings.FIELDS["code2tag"] == ("source_code",)
    assert settings.FIELDS["descode2tag"] == (
        *(f"prob_desc_{part}" for part in statement_parts),
        "source_code",
    )


@pytest.mark.parametrize("setting", settings.SETTINGS)
def test_the_fields_of_a_setting_are_raw_input_columns_and_never_the_labels_or_the_split(setting):
    fields = set(settings.FIELDS[setting])

    assert not fields & set(LABEL_AND_SPLIT_COLUMNS)
    assert fields <= set(RAW_INPUT_COLUMNS)


def test_the_prompt_versions_follow_the_output_formats_text_first():
    # The order register_prompts registers them in: prompts:/code2tag/1 is code2tag-text.
    assert settings.PROMPT_VERSIONS == {"text": 1, "json": 2}
    assert tuple(settings.PROMPT_VERSIONS) == settings.OUTPUT_FORMATS


def test_each_setting_has_one_prompt_file_per_version_relative_to_the_package():
    assert settings.PROMPT_PATHS == {
        "code2tag": {"text": "templates/code2tag.v1.yaml", "json": "templates/code2tag.v2.yaml"},
        "descode2tag": {
            "text": "templates/descode2tag.v1.yaml",
            "json": "templates/descode2tag.v2.yaml",
        },
    }


def test_the_language_and_the_schema_name_are_the_ones_the_requests_carry():
    assert settings.LANGUAGE == "Python"
    assert settings.SCHEMA_NAME == "problem_tags"


def test_the_paper_scores_are_the_python_column_of_table_3_in_the_paper_notes():
    python = table_3_row("Setting").index("**Python**")
    code2tag, descode2tag, gain = (
        float(table_3_row(label)[python].strip("*"))
        for label in ("Code2Tag", "DesCode2Tag", "Gain (derived)")
    )

    assert {"code2tag": code2tag, "descode2tag": descode2tag} == settings.PAPER_ALL_TAG_MACRO_F1
    assert settings.PAPER_ALL_TAG_MACRO_F1 == {"code2tag": 29.45, "descode2tag": 33.73}
    assert round(descode2tag - code2tag, 2) == gain == 4.28
