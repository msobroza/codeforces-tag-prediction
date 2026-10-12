"""Tests for the tag model's configuration."""

import dataclasses
import json
import math

import pytest

from codeforces_tags.config import ABLATIONS
from codeforces_tags.model.params import BLOCKS, CLASS_WEIGHTS, CLASSIFIERS, ModelParams


def test_model_params_store_the_blocks_as_a_tuple_in_blocks_order(make_params):
    assert make_params(blocks=["cues", "statement"]).blocks == ("statement", "cues")
    assert make_params(blocks=("code", "cues", "statement")).blocks == BLOCKS


@pytest.mark.parametrize("blocks", ABLATIONS.values(), ids=ABLATIONS.keys())
def test_model_params_accept_the_blocks_of_every_ablation(make_params, blocks):
    assert make_params(blocks=blocks).blocks == blocks


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize("class_weight", CLASS_WEIGHTS)
def test_model_params_accept_every_classifier_and_class_weight(
    make_params, classifier, class_weight
):
    params = make_params(classifier=classifier, class_weight=class_weight)

    assert (params.classifier, params.class_weight) == (classifier, class_weight)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("classifier", "random_forest", "classifier"),
        ("class_weight", "auto", "class_weight"),
        ("blocks", (), "at least one block"),
        ("blocks", ("code", "code"), "more than once"),
        ("blocks", ("statement", "images"), "'images'"),
        ("c", 0.0, "c must be above 0"),
        ("c", -1.0, "c must be above 0"),
        ("c", math.nan, "c must be above 0"),
        ("code_weight", 0.0, "code_weight must be above 0"),
        ("code_weight", math.nan, "code_weight must be above 0"),
        ("cue_weight", -0.5, "cue_weight must be above 0"),
        ("cue_weight", math.nan, "cue_weight must be above 0"),
    ],
)
def test_model_params_reject_invalid_values(make_params, field, value, message):
    with pytest.raises(ValueError, match=message):
        make_params(**{field: value})


def test_model_params_ignore_the_weight_of_a_block_they_do_not_keep(make_params):
    params = make_params(blocks=("statement",), code_weight=0.0, cue_weight=-1.0)

    assert params.blocks == ("statement",)


def test_model_params_cannot_be_changed(make_params):
    params = make_params()

    with pytest.raises(dataclasses.FrozenInstanceError):
        params.c = 2.0


def test_to_dict_gives_every_field_as_json_with_the_blocks_as_a_list(make_params):
    params = make_params(blocks=("statement", "cues"))

    values = params.to_dict()

    assert list(values) == [field.name for field in dataclasses.fields(ModelParams)]
    assert values["blocks"] == ["statement", "cues"]
    assert json.loads(json.dumps(values)) == values


def test_from_dict_reads_back_what_to_dict_wrote_through_json(make_params):
    params = make_params(classifier="linear_svm", c=0.05, include_notes=True)

    assert ModelParams.from_dict(json.loads(json.dumps(params.to_dict()))) == params


def test_from_dict_checks_the_values(make_params):
    values = make_params().to_dict() | {"c": -2.0}

    with pytest.raises(ValueError, match="c must be above 0"):
        ModelParams.from_dict(values)


def test_from_dict_names_the_unknown_and_missing_keys(make_params):
    values = make_params().to_dict()
    values["C"] = values.pop("c")
    del values["blocks"]

    with pytest.raises(ValueError, match=r"unknown \['C'\], missing \['blocks', 'c'\]"):
        ModelParams.from_dict(values)
