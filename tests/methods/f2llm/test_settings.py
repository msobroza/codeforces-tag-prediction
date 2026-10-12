"""Tests for the settings of the F2LLM embedding methods."""

import json
import re
from fnmatch import fnmatchcase
from itertools import pairwise
from pathlib import PurePosixPath

import pytest
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

from codeforces_tags import config
from codeforces_tags.config import FOCUS_TAGS, METHODS_ASSETS_DIR, PROJECT_ROOT
from codeforces_tags.methods.f2llm.settings import (
    AUTO_DEVICE,
    CACHE_NAME,
    CHUNK_TEXTS,
    CPU_BATCH_TOKENS,
    CPU_THREADS,
    EARLY_FUSION,
    EARLY_FUSION_GRID,
    ENCODER_FILE_PATTERNS,
    ENCODER_FILES,
    ENCODER_REPO_ID,
    ENCODER_REVISION,
    EOS_TOKEN_ID,
    FUSION_INNER_FOLDS,
    GPU_BATCH_TOKENS,
    GPU_MAX_BATCH_SEQUENCES,
    GRIDS,
    LATE_FUSION,
    LATE_FUSION_GRID,
    LBFGS_MAX_ITER,
    LORA_TARGETS,
    MAX_LENGTH,
    NAMES,
    PICKLE_SUFFIXES,
    POOLING,
    PROBE,
    PROBE_GRID,
    SETFIT,
    SETFIT_METHOD,
    SETFIT_SMOKE_PAIRS,
    SETTINGS,
    SHARED_ASSETS_DIR,
    SOLUTION_INSTRUCTION,
    STATEMENT_INSTRUCTION,
    TFIDF_TEST_MACRO_F1,
    TUNED,
    experiments_dir,
)
from codeforces_tags.provenance import sha256_json
from codeforces_tags.tracking.methods import flat_record

# The 11 files of the repository at the pinned revision that match the patterns, sorted: the Hub
# lists 9 small files and the two stored in git LFS (the weights and the tokenizer).
ENCODER_PATHS = [
    "1_Pooling/config.json",
    "added_tokens.json",
    "config.json",
    "config_sentence_transformers.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
WEIGHTS_SHA256 = "d950e8a1e04cbe3df4acb46a15263c5eb79e3131b8084fb4c3075efc32b3e8fc"
TOKENIZER_SHA256 = "38360d5a512a43641b36d6fba2df87b8a3f5464c6b5c76f03e82d6d795175566"


def test_the_four_methods_are_named_in_registry_order_and_the_tuned_ones_in_tuning_order():
    assert NAMES == ("f2llm_early_fusion", "f2llm_late_fusion", "f2llm_probe", "f2llm_setfit")
    assert list(NAMES) == sorted(NAMES)
    assert TUNED == (PROBE, LATE_FUSION, EARLY_FUSION)
    assert set(TUNED) | {SETFIT_METHOD} == set(NAMES)


def test_the_encoder_is_pinned_to_a_revision_of_f2llm_0_6b_v1():
    assert ENCODER_REPO_ID == "codefuse-ai/F2LLM-0.6B"
    assert ENCODER_REVISION == "eb2778d335980b42b986c75d7fe3a78be19ef3a6"
    assert re.fullmatch(r"[0-9a-f]{40}", ENCODER_REVISION)
    assert ENCODER_FILE_PATTERNS == ("*.json", "*.safetensors", "*.txt")


def test_every_file_the_encoder_needs_is_pinned_by_its_sha256():
    assert list(ENCODER_FILES) == ENCODER_PATHS
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in ENCODER_FILES.values())
    # The two files stored in git LFS have the sha256 that the Hub lists for the revision.
    assert ENCODER_FILES["model.safetensors"] == WEIGHTS_SHA256
    assert ENCODER_FILES["tokenizer.json"] == TOKENIZER_SHA256


def test_the_pinned_files_are_what_fetch_downloads_and_none_of_them_can_hold_a_pickle():
    for path in ENCODER_FILES:
        parts = PurePosixPath(path).parts
        assert not PurePosixPath(path).is_absolute()
        assert ".cache" not in parts
        assert ".." not in parts
        assert any(fnmatchcase(path, pattern) for pattern in ENCODER_FILE_PATTERNS)
        assert not path.endswith(PICKLE_SUFFIXES)
    assert set(PICKLE_SUFFIXES) == {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib"}


def test_the_stored_embedding_is_the_normalised_last_token_of_a_1024_token_text():
    assert (EOS_TOKEN_ID, POOLING, MAX_LENGTH) == (151645, "last_token_l2_fp32", 1024)


def test_the_batches_threads_cache_and_solver_settings_are_the_specs():
    batches = (CPU_THREADS, CPU_BATCH_TOKENS, GPU_BATCH_TOKENS, GPU_MAX_BATCH_SEQUENCES)

    assert batches == (6, 8192, 32768, 16)
    assert (CACHE_NAME, CHUNK_TEXTS) == ("f2llm", 256)
    assert (LBFGS_MAX_ITER, FUSION_INNER_FOLDS) == (5000, 5)
    # The automatic device is one of the project-wide device settings (config.DEVICES).
    assert AUTO_DEVICE == "auto"
    assert AUTO_DEVICE in config.DEVICES


def test_each_input_has_one_fixed_instruction_that_ends_where_its_text_starts():
    assert STATEMENT_INSTRUCTION == (
        "Instruct: Classify the given competitive programming problem by the algorithmic topics "
        "it involves, such as math, graphs, strings, number theory, trees, geometry, games and "
        "probabilities.\nQuery: "
    )
    assert SOLUTION_INSTRUCTION == (
        "Instruct: Classify the given Python solution of a competitive programming problem by "
        "the algorithmic topics it involves, such as math, graphs, strings, number theory, "
        "trees, geometry, games and probabilities.\nQuery: "
    )
    for instruction in (STATEMENT_INSTRUCTION, SOLUTION_INSTRUCTION):
        assert all(tag in instruction for tag in FOCUS_TAGS)


def test_the_probe_grid_crosses_head_c_class_weight_and_solution_weight_simplest_first():
    assert len(PROBE_GRID) == 168
    assert PROBE_GRID[0] == {
        "head": "logistic_regression",
        "c": 0.01,
        "class_weight": "none",
        "solution_weight": 0.0,
    }
    assert PROBE_GRID[-1] == {
        "head": "ridge",
        "c": 100.0,
        "class_weight": "balanced",
        "solution_weight": 1.0,
    }
    # The last axis varies fastest, then class weight, then C, then the head.
    assert [PROBE_GRID[index]["solution_weight"] for index in range(4)] == [0.0, 0.25, 0.5, 1.0]
    assert PROBE_GRID[4]["class_weight"] == "balanced"
    assert PROBE_GRID[8]["c"] == 0.0464
    assert PROBE_GRID[56]["head"] == "linear_svm"
    assert {cell["c"] for cell in PROBE_GRID} == {0.01, 0.0464, 0.215, 1.0, 4.64, 21.5, 100.0}


def test_the_early_fusion_grid_crosses_c_and_the_two_embedding_weights_simplest_first():
    assert len(EARLY_FUSION_GRID) == 80
    assert EARLY_FUSION_GRID[0] == {"c": 0.2, "statement_weight": 0.1, "solution_weight": 0.0}
    assert EARLY_FUSION_GRID[-1] == {"c": 1.6, "statement_weight": 1.0, "solution_weight": 0.5}
    assert {cell["c"] for cell in EARLY_FUSION_GRID} == {0.2, 0.4, 0.62, 1.0, 1.6}
    assert {cell["statement_weight"] for cell in EARLY_FUSION_GRID} == {0.1, 0.25, 0.5, 1.0}
    assert {cell["solution_weight"] for cell in EARLY_FUSION_GRID} == {0.0, 0.1, 0.25, 0.5}


def test_the_late_fusion_grid_runs_from_0_05_to_0_95_in_steps_of_0_05():
    weights = [cell["weight"] for cell in LATE_FUSION_GRID]

    assert len(weights) == 19
    assert weights[:3] == [0.05, 0.1, 0.15]  # round decimals, not 0.15000000000000002
    assert weights[-1] == 0.95
    assert all(after - before == pytest.approx(0.05) for before, after in pairwise(weights))


def test_each_tuned_method_has_a_grid_of_distinct_cells_in_tuning_order():
    assert list(GRIDS) == list(TUNED)
    for grid in GRIDS.values():
        assert len({json.dumps(cell, sort_keys=True) for cell in grid}) == len(grid)


def test_setfit_is_fixed_before_any_result_and_draws_six_thousand_pairs():
    fixed = [
        ("pairs_per_tag_positive", 375),
        ("batch_pairs", 16),
        ("micro_batch_pairs", 8),
        ("learning_rate", 1e-4),
        ("epochs", 1),
        ("warmup", 0.1),
        ("lora_r", 8),
        ("lora_alpha", 16),
        ("lora_dropout", 0.0),
        ("seed", 0),
    ]

    assert dict(fixed) == SETFIT
    assert SETFIT["batch_pairs"] % SETFIT["micro_batch_pairs"] == 0
    assert 2 * SETFIT["pairs_per_tag_positive"] * len(FOCUS_TAGS) == 6000
    assert SETFIT_SMOKE_PAIRS == 100


def test_lora_adapts_the_four_attention_and_the_three_mlp_projections_of_every_layer():
    attention = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}

    assert set(LORA_TARGETS) == attention | mlp
    assert list(LORA_TARGETS) == sorted(LORA_TARGETS)


def test_the_settings_are_json_values_that_summarise_the_grids():
    assert json.loads(json.dumps(SETTINGS, allow_nan=False)) == SETTINGS
    assert SETTINGS["probe_grid"] == {"size": 168, "sha256": sha256_json(list(PROBE_GRID))}
    assert SETTINGS["late_fusion_grid"]["size"] == 19
    assert SETTINGS["early_fusion_grid"]["size"] == 80
    assert SETTINGS["encoder_files"] == ENCODER_FILES
    assert SETTINGS["setfit"] == SETFIT


def test_the_hardware_settings_stay_out_of_the_recipe():
    hardware = {"cpu_threads", "cpu_batch_tokens", "gpu_batch_tokens", "gpu_max_batch_sequences"}

    assert not hardware & set(SETTINGS)


def test_the_settings_flatten_to_run_record_parameters_within_mlflows_limit(placeholder_record):
    record = placeholder_record(PROBE, method_entries={"settings": SETTINGS, "supersedes": []})

    flat = flat_record(record)  # raises when a value is longer than MLflow keeps

    assert flat["method_entries.settings.setfit.learning_rate"] == "0.0001"
    assert json.loads(flat["method_entries.settings.encoder_files"]) == ENCODER_FILES
    kept = [value for key, value in flat.items() if key.startswith("method_entries.settings.")]
    assert max(len(value) for value in kept) < MAX_PARAM_VAL_LENGTH


def test_experiments_dir_is_read_when_called(monkeypatch, tmp_path):
    assert experiments_dir(PROBE) == config.EXPERIMENTS_DIR / "f2llm_probe"

    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)

    assert experiments_dir(SETFIT_METHOD) == tmp_path / "f2llm_setfit"


def test_the_shared_analysis_files_live_beside_the_frozen_methods_not_among_them():
    # CI checks every folder of assets/methods as a frozen method.
    assert SHARED_ASSETS_DIR == PROJECT_ROOT / "assets" / "f2llm"
    assert METHODS_ASSETS_DIR not in SHARED_ASSETS_DIR.parents


def test_the_tfidf_reference_score_is_the_frozen_models_single_test_score():
    assert TFIDF_TEST_MACRO_F1 == 0.6887545239153905
