"""Tests for the ModernBERT methods' settings: their JSON form and the recipe their studies hash."""

import json
from dataclasses import asdict, fields, replace

import pytest

from codeforces_tags.methods.modernbert.settings import (
    CONFIG_FILES,
    ENCODER_FILE,
    FULL,
    LORA,
    VARIANTS,
    Variant,
    settings,
    study_recipe,
)

# The recipes that the former notebooks 05 and 06 stored in their Optuna journals
# (experiments/<name>/optuna.jsonl): the package resumes and exports those studies only while its
# recipe is exactly this.
LORA_JOURNAL_RECIPE = {
    "model_repo": "answerdotai/ModernBERT-large",
    "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    "max_length": 1024,
    "code_min_tokens": 128,
    "micro_batch": 4,
    "lora_targets": ["Wqkv", "Wo", "Wi"],
    "trained_modules": ["head", "classifier"],
    "lora_alpha_per_rank": 2,
    "weight_decay": 0.01,
    "seed": 0,
    "tuning_fold": 0,
    "subfolds": 5,
    "search_space": {
        "learning_rate": [5e-05, 0.001],
        "lora_rank": [8, 16, 32],
        "lora_dropout": [0.0, 0.2],
        "epochs": [2, 4],
        "batch_size": [16, 32],
        "warmup_ratio": [0.0, 0.2],
        "pos_weight_power": [0.0, 1.0],
    },
    "smoke": False,
}
FULL_JOURNAL_RECIPE = {
    "model_repo": "answerdotai/ModernBERT-base",
    "model_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    "max_length": 2048,
    "code_min_tokens": 128,
    "micro_batch": 4,
    "seed": 0,
    "tuning_fold": 0,
    "subfolds": 5,
    "search_space": {
        "learning_rate": [1e-05, 0.0002],
        "weight_decay": [1e-05, 0.1],
        "classifier_dropout": [0.0, 0.2],
        "epochs": [2, 5],
        "batch_size": [16, 32],
        "warmup_ratio": [0.0, 0.2],
        "pos_weight_power": [0.0, 1.0],
    },
    "smoke": False,
}
LORA_ONLY_KEYS = ["lora_targets", "trained_modules", "lora_alpha_per_rank", "weight_decay"]
LAST_KEYS = ["seed", "tuning_fold", "subfolds", "search_space", "smoke"]


def test_the_two_variants_are_registered_under_their_own_names():
    assert (LORA, FULL) == ("modernbert_lora", "modernbert_base_full")
    assert {name: variant.name for name, variant in VARIANTS.items()} == {LORA: LORA, FULL: FULL}
    assert (VARIANTS[LORA].lora, VARIANTS[FULL].lora) == (True, False)


@pytest.mark.parametrize("name", [LORA, FULL])
def test_settings_hold_every_field_of_the_variant(name):
    held = settings(VARIANTS[name])

    assert list(held) == [field.name for field in fields(Variant)]


@pytest.mark.parametrize("name", [LORA, FULL])
def test_settings_are_plain_json_values_that_survive_a_json_round_trip(name):
    held = settings(VARIANTS[name])

    # A tuple never equals a list, so both equalities fail on any tuple left inside.
    assert json.loads(json.dumps(held)) == held
    assert held == json.loads(json.dumps(asdict(VARIANTS[name])))


def test_settings_turn_the_lora_tuples_into_lists():
    held = settings(VARIANTS[LORA])

    assert held["lora_targets"] == ["Wqkv", "Wo", "Wi"]
    assert held["trained_modules"] == ["head", "classifier"]
    assert held["search_space"]["lora_rank"] == [8, 16, 32]


def test_settings_turn_tuples_nested_in_lists_into_lists():
    variant = replace(VARIANTS[FULL], search_space={"epochs": [(2, 3), (4, 5)]})

    assert settings(variant)["search_space"] == {"epochs": [[2, 3], [4, 5]]}


def test_the_lora_study_recipe_is_the_one_notebook_05_stored_in_its_journal():
    recipe = study_recipe(VARIANTS[LORA])

    assert recipe == LORA_JOURNAL_RECIPE
    assert list(recipe) == list(LORA_JOURNAL_RECIPE)


def test_the_full_study_recipe_is_the_one_notebook_06_stored_in_its_journal():
    recipe = study_recipe(VARIANTS[FULL])

    assert recipe == FULL_JOURNAL_RECIPE
    assert list(recipe) == list(FULL_JOURNAL_RECIPE)


def test_only_the_lora_recipe_holds_the_lora_keys_and_both_end_with_the_same_keys():
    lora, full = study_recipe(VARIANTS[LORA]), study_recipe(VARIANTS[FULL])

    assert list(lora)[5:9] == LORA_ONLY_KEYS
    assert not set(LORA_ONLY_KEYS) & set(full)
    assert list(lora)[-5:] == list(full)[-5:] == LAST_KEYS
    assert lora["smoke"] is full["smoke"] is False


def test_the_study_recipe_reads_the_variants_values_as_plain_json():
    variant = replace(VARIANTS[LORA], max_length=48, code_min_tokens=8, lora_targets=("Wo",))

    recipe = study_recipe(variant)

    assert (recipe["max_length"], recipe["code_min_tokens"]) == (48, 8)
    assert recipe["lora_targets"] == ["Wo"]
    assert json.loads(json.dumps(recipe)) == recipe


def test_the_study_recipe_holds_none_of_the_settings_a_study_may_change():
    recipe = study_recipe(VARIANTS[LORA])

    assert not {"tuning_trials", "score_batch", "collapse_ap", "device", "name"} & set(recipe)


def test_changing_a_returned_recipe_leaves_the_variant_unchanged():
    recipe = study_recipe(VARIANTS[LORA])
    recipe["search_space"]["lora_rank"].append(64)

    assert VARIANTS[LORA].search_space["lora_rank"] == (8, 16, 32)


def test_only_lora_pins_the_encoder_weights_of_the_pretrained_snapshot():
    config_files = (
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )

    lora, full = VARIANTS[LORA], VARIANTS[FULL]

    assert config_files == CONFIG_FILES
    assert ENCODER_FILE == "model.safetensors"
    assert lora.pretrained_files == (*config_files, "model.safetensors")
    assert full.pretrained_files == config_files


@pytest.mark.parametrize("name", [LORA, FULL])
def test_the_retrain_bounds_are_declared_before_any_retraining(name):
    variant = VARIANTS[name]

    assert variant.retrain_min_correlation == 0.9
    assert variant.retrain_max_median_gap == 0.02
    assert variant.retrain_max_flips == 0.025
    assert variant.retrain_f1_margin == 0.03
