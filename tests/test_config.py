"""Tests for the project settings."""

import ast
import math
import re
import subprocess
import tomllib
from pathlib import Path

from codeforces_tags.config import (
    ALL_TAGS,
    API_KEY_ENV_VARS,
    API_SDK_ERRORS,
    CALIBRATION_BINS,
    CALIBRATION_METHODS,
    CALIBRATION_STRATEGY,
    CLI_API_BATCH,
    CLI_COST_DECIMALS,
    CLI_DEFAULT_METHOD,
    CLI_JSON_LIST_FIELDS,
    CLI_OFFLINE_METHOD,
    CLI_REQUIRED_FIELDS,
    CLI_UNCAPPED_REQUESTS,
    EVALUATION_COST_REL_TOL,
    EVALUATION_INPUT_FIELDS,
    EXPERIMENTS_DIR,
    FOCUS_TAGS,
    LATENCY_BUDGET_S,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    LLM_PRICES,
    LLM_PROGRESS_EVERY,
    LLM_PROMPT_CACHE_MODE,
    LLM_SERVICE_TIER,
    LLM_STORE,
    LLM_TIMEOUT_S,
    METHOD_FILE_MAX_BYTES,
    METHOD_FOLDER_MAX_BYTES,
    MLFLOW_DATASETS,
    MLFLOW_SCREEN_COMPARE,
    MLFLOW_SCREEN_GEPA_FOLDER,
    MLFLOW_SCREEN_GEPA_TAG,
    MLFLOW_SCREEN_SAMPLE_ID,
    MLFLOW_SCREEN_TAG,
    MLFLOW_SCREENS_DIR,
    MLFLOW_UI_PORT,
    PAPER_TAGS,
    PLATT_CLIP,
    PLATT_SOLVER,
    PLATT_TOL,
    PRETRAINED_CACHE_DIR,
    PROJECT_ROOT,
    PYPROJECT_FILE,
    RAW_INPUT_COLUMNS,
    RECORDED_SETTINGS,
    SCORE_GRID,
    SPLIT_FILE,
    TEXT_PARSER_VERSION,
    TRAINED_CACHE_DIR,
    TRAINED_DIR_ENV_VAR,
)
from codeforces_tags.methods import METHODS
from codeforces_tags.splitting import load_split

TASK_STATEMENT = Path(__file__).resolve().parents[1] / "PROBLEM.md"


def test_focus_tags_match_the_task_statement():
    listed = re.search(r"\['math'.*?\]", TASK_STATEMENT.read_text(encoding="utf-8")).group(0)

    assert tuple(ast.literal_eval(listed)) == FOCUS_TAGS


def test_all_tags_are_the_37_distinct_tags_in_sorted_order_with_every_focus_tag():
    assert len(ALL_TAGS) == 37
    assert list(ALL_TAGS) == sorted(set(ALL_TAGS))
    assert set(FOCUS_TAGS) <= set(ALL_TAGS)


def test_paper_tags_are_all_tags_but_the_six_the_paper_leaves_out():
    left_out = {
        "*special",
        "chinese remainder theorem",
        "hashing",
        "schedules",
        "string suffix structures",
        "ternary search",
    }

    assert tuple(tag for tag in ALL_TAGS if tag not in left_out) == PAPER_TAGS
    assert len(PAPER_TAGS) == 31


def test_the_calibration_compares_four_probabilities_and_records_no_setting_in_the_recipe():
    # The ids are parts of MLflow metric keys (brier/<id>/<tag>): word characters only. A recorded
    # setting would change the tfidf recipe, and orphan its frozen files and its MLflow run.
    assert list(CALIBRATION_METHODS) == [
        "logistic_regression",
        "svm_minmax",
        "svm_platt",
        "svm_isotonic",
    ]
    assert all(re.fullmatch(r"\w+", method) for method in CALIBRATION_METHODS)
    assert len(set(CALIBRATION_METHODS.values())) == len(CALIBRATION_METHODS)
    assert CALIBRATION_BINS == 10
    assert CALIBRATION_STRATEGY == "quantile"
    assert not [name for name in RECORDED_SETTINGS if name.startswith("CALIBRATION_")]


def test_the_threshold_rule_clips_before_its_logit_and_snaps_to_jevs_grid():
    # A calibrated twin's recipe holds both through platt_lipton.RULE_SETTINGS; in
    # RECORDED_SETTINGS they would change every frozen method's recipe.
    assert (PLATT_CLIP, SCORE_GRID) == (1e-4, 0.01)
    assert not {"PLATT_CLIP", "SCORE_GRID"} & set(RECORDED_SETTINGS)


def test_the_threshold_rules_platt_map_is_fitted_to_convergence():
    # The maximum-likelihood map, whatever iteration a looser solver would stop at; like the clip,
    # the twins' recipes hold them through platt_lipton.RULE_SETTINGS, never RECORDED_SETTINGS.
    assert (PLATT_SOLVER, PLATT_TOL) == ("newton-cholesky", 1e-12)
    assert not {"PLATT_SOLVER", "PLATT_TOL"} & set(RECORDED_SETTINGS)


def test_the_latency_budget_is_the_task_statements_ten_seconds():
    assert "under ten seconds" in TASK_STATEMENT.read_text(encoding="utf-8")
    assert LATENCY_BUDGET_S == 10


def test_the_cli_runs_a_registered_method_by_default_and_names_a_local_free_one_on_failure():
    # Every failure of the CLI names the offline method: it must run with no group and no key.
    offline = METHODS[CLI_OFFLINE_METHOD]

    assert CLI_DEFAULT_METHOD in METHODS
    assert CLI_DEFAULT_METHOD != CLI_OFFLINE_METHOD
    assert (offline.groups, offline.calls_api) == ((), False)


def test_every_group_of_a_method_that_calls_an_api_names_the_key_its_sdk_reads():
    groups = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))["dependency-groups"]
    api_groups = {group for spec in METHODS.values() if spec.calls_api for group in spec.groups}

    assert api_groups <= set(API_KEY_ENV_VARS)
    assert set(API_KEY_ENV_VARS) <= set(groups)
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in API_KEY_ENV_VARS.values())


def test_the_cli_reads_two_fields_of_a_sample_and_caps_what_an_api_method_spends():
    # A hand-written sample may hold only the description and the solution: every other raw
    # field reads as null. One paid request goes without a cap; more need --max-cost-usd.
    assert CLI_REQUIRED_FIELDS == ("prob_desc_description", "source_code")
    assert set(CLI_REQUIRED_FIELDS) < set(RAW_INPUT_COLUMNS)
    # The sample inputs and outputs, which clean_columns decodes from their JSON text.
    assert CLI_JSON_LIST_FIELDS == ("prob_desc_sample_inputs", "prob_desc_sample_outputs")
    assert set(CLI_JSON_LIST_FIELDS) < set(RAW_INPUT_COLUMNS) - set(CLI_REQUIRED_FIELDS)
    assert CLI_UNCAPPED_REQUESTS == 1
    assert CLI_API_BATCH >= 1
    # A request's printed cost keeps the price of one token of every kind, of every model.
    per_token = [price / 1e6 for prices in LLM_PRICES.values() for price in prices.values()]
    assert all(math.isclose(round(price, CLI_COST_DECIMALS), price) for price in per_token)
    assert set(API_SDK_ERRORS) == set(API_KEY_ENV_VARS)
    assert all(re.fullmatch(r"\w+(\.\w+)*\.[A-Z]\w*", path) for path in API_SDK_ERRORS.values())


def test_method_folders_are_capped_and_pretrained_weights_cached_in_the_ignored_experiments():
    weights = PRETRAINED_CACHE_DIR.relative_to(PROJECT_ROOT) / "encoder" / "model.safetensors"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(weights)], cwd=PROJECT_ROOT, check=False
    )

    assert 0 < METHOD_FILE_MAX_BYTES <= METHOD_FOLDER_MAX_BYTES
    assert PRETRAINED_CACHE_DIR.is_relative_to(EXPERIMENTS_DIR)
    assert ignored.returncode == 0


def test_trained_weights_too_large_for_a_method_folder_are_cached_in_the_ignored_experiments():
    weights = TRAINED_CACHE_DIR.relative_to(PROJECT_ROOT) / "method" / "fold_0" / "weights.bin"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(weights)], cwd=PROJECT_ROOT, check=False
    )

    assert TRAINED_CACHE_DIR.is_relative_to(EXPERIMENTS_DIR)
    assert TRAINED_DIR_ENV_VAR == "CODEFORCES_TRAINED_DIR"
    assert ignored.returncode == 0


def test_the_evaluation_datasets_show_every_raw_input_of_a_problem_one_dataset_per_part():
    assert MLFLOW_DATASETS == {"dev": "codeforces-dev", "test": "codeforces-test"}
    assert EVALUATION_INPUT_FIELDS == RAW_INPUT_COLUMNS
    assert EVALUATION_COST_REL_TOL == 1e-9


def test_the_toolkit_keeps_no_response_at_the_provider_and_sets_no_cache_breakpoint():
    # OpenAI keeps stored responses for 30 days or more, and an automatic cache breakpoint writes
    # a prompt of 1,024 tokens or more to its cache at 1.25 times the input price, with no reuse.
    assert LLM_STORE is False
    assert LLM_PROMPT_CACHE_MODE == "explicit"
    assert LLM_SERVICE_TIER == "default"


def test_the_transport_settings_leave_the_runner_able_to_send_retry_and_report():
    for count in (LLM_CONCURRENCY, LLM_MAX_RETRIES, LLM_PROGRESS_EVERY):
        assert isinstance(count, int)
    assert LLM_CONCURRENCY >= 1  # a semaphore of 0 lets no request through
    assert LLM_MAX_RETRIES >= 0
    assert LLM_TIMEOUT_S > 0
    assert LLM_PROGRESS_EVERY >= 1  # the progress line comes every N-th answer: N % 0 fails


def test_every_priced_model_has_an_input_a_cached_input_and_an_output_price():
    assert LLM_PRICES
    for model, prices in LLM_PRICES.items():
        assert set(prices) == {"input", "cached_input", "output"}, model
        assert 0 < prices["cached_input"] <= prices["input"], model
        # Zero is a price: Jev's output tokens are free (TypeSafe bills its input tokens only).
        assert prices["output"] >= 0, model


def test_the_prices_of_gpt_6_luna_are_those_the_frozen_costs_are_computed_with():
    # USD per million tokens. A frozen per_problem.csv holds costs computed from them, and
    # llm verify recomputes those: a price changed after the paid runs would fail it.
    assert LLM_PRICES["gpt-6-luna"] == {"input": 0.10, "cached_input": 0.01, "output": 0.50}


def test_the_prices_of_jev_1_13_0_are_those_the_frozen_costs_are_computed_with():
    # TypeSafe's models page of 2026-10-08: USD per million tokens of input, none for output, and
    # no cached price because its SDK reports no cached tokens. The TypeSafe verify recomputes the
    # costs of per_problem.csv from them, as llm verify does.
    assert LLM_PRICES["jev-1.13.0"] == {"input": 0.042, "cached_input": 0.042, "output": 0.0}


def test_the_text_parser_is_at_version_1():
    # The parser was written and tested before any paid request, and is never adjusted after
    # results are seen: a new rule would be version 2, recorded in the run record.
    assert TEXT_PARSER_VERSION == 1


def test_the_mlflow_screens_show_a_dev_problem_and_a_zero_shot_method_on_a_focus_tag():
    # The screenshots are dev only (ADR 0001), and they choose no prediction: a setting in the
    # recipe would change every method's recipe_sha256, and orphan its frozen files and runs.
    assert load_split(SPLIT_FILE).loc[MLFLOW_SCREEN_SAMPLE_ID, "part"] == "dev"
    assert METHODS[MLFLOW_SCREEN_COMPARE].kind == "zero_shot"
    assert MLFLOW_SCREEN_TAG in FOCUS_TAGS
    # Not MLflow's default port 5000, where a user's own `mlflow ui` would answer.
    assert 1024 < MLFLOW_UI_PORT < 65536
    assert MLFLOW_UI_PORT != 5000
    assert MLFLOW_SCREENS_DIR == PROJECT_ROOT / "notebooks" / "figures" / "mlflow"
    assert not [name for name in RECORDED_SETTINGS if name.startswith("MLFLOW_")]
    # The search's views (mlflow_ui.py --set gepa) compare a focus tag's criteria, and have a
    # folder of their own inside notebook 05's.
    assert MLFLOW_SCREEN_GEPA_TAG in FOCUS_TAGS
    assert MLFLOW_SCREEN_GEPA_FOLDER == MLFLOW_SCREENS_DIR / "gepa"
