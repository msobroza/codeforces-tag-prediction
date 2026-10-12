"""The committed frozen files of the TypeSafe Jev methods, read without the dataset or the SDK."""

import json

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS, METHODS_ASSETS_DIR
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    load_predictions,
    load_run_record,
    load_scores,
)
from codeforces_tags.evaluation.timing import LATENCY_FILE
from codeforces_tags.llm.store import answers_file
from codeforces_tags.methods import load_method, model_dir
from codeforces_tags.methods.typesafe_jev import __main__ as cli
from codeforces_tags.methods.typesafe_jev import platt
from codeforces_tags.methods.typesafe_jev.search_record import read_search_record
from codeforces_tags.methods.typesafe_jev.settings import (
    FROZEN_NAMES,
    GEPA_NAME,
    MODEL,
    PARAMS_FILE,
    PLATT_NAMES,
    TEST_INFORMED,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.words import base_of, words_of
from codeforces_tags.model.platt_lipton import RULE_SETTINGS, apply_fold_thresholds
from codeforces_tags.provenance import file_sha256
from codeforces_tags.splitting import PARTS, load_split


def assert_test_rows_follow_the_thresholds(name):
    """Check that a method's frozen test rows are its scores at or above its folder's thresholds."""
    directory = METHODS_ASSETS_DIR / name
    thresholds = load_method(model_dir(name)).thresholds.thresholds_
    expected = (load_scores(directory).loc["test"] >= thresholds).astype("int64")
    assert load_predictions(directory).loc["test"].equals(expected)


# The Jev methods that sent their own requests; the calibrated twins read their bases' answers.
@pytest.mark.parametrize("name", [*FROZEN_NAMES, GEPA_NAME])
def test_the_committed_jev_methods_are_frozen_with_the_recipe_of_the_code(name):
    # Needs no dataset: a changed word of a tag definition makes the committed recipe stale,
    # and such a change is a new method under a new name (docs/adr/0003).
    directory = METHODS_ASSETS_DIR / name
    check_frozen_method(name, assets_dir=METHODS_ASSETS_DIR)
    stored = load_run_record(directory)
    parts = load_scores(directory).index.get_level_values("part")

    assert (stored.method, stored.recipe_sha256) == (name, cli.recipe_sha256(name))
    assert stored.method_entries["served_models"] == [MODEL]
    # The searched method's words were chosen on dev answers (docs/adr/0010).
    assert stored.method_entries["dev_in_sample"] is (name == GEPA_NAME)
    assert parts.value_counts().to_dict() == {"dev": 3988, "test": 994}
    # The answers files are those the record hashed, and the test rows are the frozen scores
    # against the method folder's thresholds: both read without the dataset or the SDK.
    hashes = {part: file_sha256(answers_file(name, part)) for part in PARTS}
    assert hashes == stored.method_entries["answers_sha256"]
    assert_test_rows_follow_the_thresholds(name)


def test_the_committed_searched_method_sends_the_words_of_its_search_record():
    # Needs no dataset: the run record pins the search record it was frozen from, the method
    # folder holds the chosen words, and the record reads back with its words checked.
    tuning = METHODS_ASSETS_DIR / GEPA_NAME / TUNING_FOLDER
    search = load_run_record(METHODS_ASSETS_DIR / GEPA_NAME).method_entries["search"]
    record = read_search_record(tuning)

    assert file_sha256(tuning / PARAMS_FILE) == search["params_sha256"]
    assert load_method(model_dir(GEPA_NAME)).words == words_of(GEPA_NAME)
    assert record.params["chosen"] == search["chosen"]


@pytest.mark.parametrize("twin", PLATT_NAMES)
def test_the_committed_calibrated_twins_are_frozen_on_their_bases_committed_files(twin):
    # Needs no dataset: a changed rule or base setting makes the recipe stale, and a base file
    # changed after the freeze no longer has the sha256 the run record pins (docs/adr/0011).
    base = base_of(twin)
    directory, base_directory = METHODS_ASSETS_DIR / twin, METHODS_ASSETS_DIR / base
    check_frozen_method(twin, assets_dir=METHODS_ASSETS_DIR)
    stored = load_run_record(directory)
    entries = stored.method_entries
    recorded = {name: file["sha256"] for name, file in entries["base"]["files"].items()}

    assert (stored.method, stored.recipe_sha256) == (twin, cli.recipe_sha256(twin))
    assert entries["base"]["name"] == base
    assert {answers_file(base, part).name for part in PARTS} <= set(recorded)
    assert recorded == {name: file_sha256(base_directory / name) for name in recorded}
    # Proposed after a test score was read, so never eligible; dev read in sample as its base.
    assert entries["test_informed"] == TEST_INFORMED
    base_entries = load_run_record(base_directory).method_entries
    assert entries["dev_in_sample"] is base_entries["dev_in_sample"]
    # Its scores are its base's probabilities, and its test rows follow its own thresholds.
    assert load_scores(directory).equals(load_scores(base_directory))
    assert_test_rows_follow_the_thresholds(twin)


@pytest.mark.parametrize("twin", PLATT_NAMES)
def test_the_committed_calibrated_twins_hold_their_records_thresholds_and_their_bases_calls(twin):
    # Needs no dataset: the method folder holds the thresholds of the dev-only record, and the
    # twin's requests are its base's, so are its timings and each problem's recorded call.
    directory, base_directory = METHODS_ASSETS_DIR / twin, METHODS_ASSETS_DIR / base_of(twin)
    params = platt.read_platt_params(directory / TUNING_FOLDER)
    stored = json.loads((model_dir(twin) / "jev.json").read_text(encoding="utf-8"))

    assert (params["rule"], params["threshold_rule"]) == (platt.TWIN_RULE, RULE_SETTINGS)
    assert params["base"] == base_of(twin)
    assert params["base_files"] == load_run_record(directory).method_entries["base"]["files"]
    assert stored["thresholds"] == [params["thresholds"][tag] for tag in FOCUS_TAGS]
    for file_name in (LATENCY_FILE, PER_PROBLEM_FILE):
        assert (directory / file_name).read_bytes() == (base_directory / file_name).read_bytes()


@pytest.mark.parametrize("twin", PLATT_NAMES)
def test_the_committed_calibrated_twins_dev_rows_follow_their_records_r2_fold_thresholds(twin):
    # Needs no dataset: the committed split names each dev problem's fold, and stability.csv
    # the R2 thresholds that fold is predicted with, chosen on the other four (docs/adr/0011).
    directory = METHODS_ASSETS_DIR / twin
    tuning = directory / TUNING_FOLDER
    stability = pd.read_csv(
        tuning / platt.STABILITY_FILE, index_col=["tag", "rule"], float_precision="round_trip"
    )
    rows = stability.xs(platt.TWIN_RULE, level="rule").loc[list(FOCUS_TAGS)]
    scores = load_scores(directory).loc["dev"]
    folds = load_split()["fold"].loc[scores.index]
    fold_ids = sorted(folds.unique())
    fold_thresholds = rows[[f"fold_{fold}" for fold in fold_ids]].T.set_axis(fold_ids)
    params = platt.read_platt_params(tuning)

    assert fold_ids == list(range(5))
    expected = apply_fold_thresholds(scores, fold_thresholds, folds)
    assert load_predictions(directory).loc["dev"].equals(expected.astype("int64"))
    # The final thresholds of the record are those its params.json names, an inf as "inf".
    assert rows["final"].tolist() == [float(params["thresholds"][tag]) for tag in FOCUS_TAGS]
