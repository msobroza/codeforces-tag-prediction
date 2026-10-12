"""Tests for a calibrated twin's dev-only record: the three threshold rules, its tables, export.

Every test runs on synthetic dev scores (``synthetic_dev``): 100 problems on Jev's 0.01 grid,
20 in each of five folds, each focus tag positive for about 30% of them and scored higher when
positive, with noise, so that every rule's Platt slope is above 0 in every fold. Nothing reads
the dataset, and no test row is ever built but the one a test refuses.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit  # pylint: disable=no-name-in-module

from codeforces_tags import methods
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.predictions import PER_PROBLEM_FILE
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.evaluation.timing import LATENCY_FILE, Latency, load_latency, save_latency
from codeforces_tags.methods.typesafe_jev import platt
from codeforces_tags.methods.typesafe_jev.method import JevProbabilities
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    PARAMS_FILE,
    PLATT_BASES,
    PLATT_FILES,
    TEST_INFORMED,
)
from codeforces_tags.model.platt_lipton import (
    RULE_SETTINGS,
    PlattLiptonThresholds,
    cross_fitted_lipton_thresholds,
    lipton_thresholds,
    platt_feature,
    platt_fit,
)
from codeforces_tags.model.thresholds import (
    apply_thresholds,
    cross_fitted_predictions,
    cross_fitted_thresholds,
    per_tag_thresholds,
)
from codeforces_tags.provenance import json_safe

TAGS = list(FOCUS_TAGS)
TWIN = "jev-statement-code-platt"
# What params.json records of the base's files in these tests.
BASE_FILES = {
    "answers_dev.jsonl": {"sha256": "a" * 64, "commit": "1" * 40},
    "answers_test.jsonl": {"sha256": "b" * 64, "commit": "2" * 40},
    "scores.csv": {"sha256": "c" * 64, "commit": "3" * 40},
    "run_record.json": {"sha256": "d" * 64, "commit": "3" * 40},
}


def synthetic_dev(problems=100, seed=0):
    """Return synthetic dev scores, labels, folds and groups, as ``platt.DevScores``.

    Problem p (sample id p + 1) is in fold p % 5 and in group p % 50, so no group spans two
    folds. A positive's score is expit(1 + noise), a negative's expit(-1 + noise), rounded to
    Jev's 0.01 grid.
    """
    rng = np.random.default_rng(seed)
    index = pd.Index(range(1, problems + 1), name="sample_id")
    labels = pd.DataFrame(
        (rng.random((problems, len(TAGS))) < 0.3).astype(int), index=index, columns=TAGS
    )
    noise = rng.normal(0.0, 1.0, labels.shape)
    scores = pd.DataFrame(
        expit(2.0 * labels.to_numpy() - 1.0 + noise).round(2), index=index, columns=TAGS
    )
    positions = np.arange(problems)
    folds = pd.Series(positions % 5, index=index, name="fold")
    groups = pd.Series(positions % 50, index=index, name="group")
    return platt.DevScores(scores, labels, folds, groups)


@pytest.fixture(name="dev")
def dev_fixture():
    """Return the synthetic dev part."""
    return synthetic_dev()


def test_r0_and_r2_are_the_f1_argmax_and_the_platt_lipton_thresholds(dev):
    pd.testing.assert_series_equal(
        platt.rule_thresholds("R0", dev.scores, dev.labels),
        per_tag_thresholds(dev.scores, dev.labels),
    )
    pd.testing.assert_series_equal(
        platt.rule_thresholds("R2", dev.scores, dev.labels),
        lipton_thresholds(dev.scores, dev.labels),
    )


def test_r1_predicts_where_platts_calibrated_probability_reaches_one_half(dev):
    found = platt.rule_thresholds("R1", dev.scores, dev.labels)

    for tag in TAGS:
        slope, intercept = platt_fit(dev.scores[tag].to_numpy(), dev.labels[tag].to_numpy())
        threshold = found[tag]
        # A midpoint of Jev's grid: the grid value above it is mapped to at least 0.5, the one
        # below it to less, by the map on the feature it was fitted on.
        assert round(threshold * 1000) % 10 == 5
        above, below = platt_feature(np.array([threshold + 0.005, threshold - 0.005]))
        assert expit(slope * above + intercept) >= 0.5
        assert expit(slope * below + intercept) < 0.5


def test_a_tag_without_a_positive_has_an_infinite_threshold_under_every_rule(dev):
    labels = dev.labels.assign(probabilities=0)

    for rule in platt.RULES:
        assert platt.rule_thresholds(rule, dev.scores, labels)["probabilities"] == math.inf


def test_rule_thresholds_refuses_an_unknown_rule_and_labels_of_other_problems(dev):
    with pytest.raises(ValueError, match=re.escape("'R3' is not a threshold rule of the record")):
        platt.rule_thresholds("R3", dev.scores, dev.labels)
    with pytest.raises(ValueError, match="labels must have the index and columns of scores"):
        platt.rule_thresholds("R0", dev.scores, dev.labels.iloc[::-1])


def test_each_folds_thresholds_are_chosen_on_the_other_folds(dev):
    r0 = platt.cross_fitted_rule_thresholds("R0", dev.scores, dev.labels, dev.folds)
    r1 = platt.cross_fitted_rule_thresholds("R1", dev.scores, dev.labels, dev.folds)
    r2 = platt.cross_fitted_rule_thresholds("R2", dev.scores, dev.labels, dev.folds)

    assert r0.index.tolist() == [0, 1, 2, 3, 4]
    assert r0.index.name == "fold"
    expected_r0 = cross_fitted_thresholds(dev.scores, dev.labels, dev.folds)
    np.testing.assert_array_equal(r0.to_numpy(), expected_r0.to_numpy())
    expected_r2 = cross_fitted_lipton_thresholds(dev.scores, dev.labels, dev.folds)
    np.testing.assert_array_equal(r2.to_numpy(), expected_r2.to_numpy())
    others = dev.folds != 3
    expected_r1 = platt.rule_thresholds("R1", dev.scores[others], dev.labels[others])
    np.testing.assert_array_equal(r1.loc[3].to_numpy(), expected_r1.to_numpy())


def test_the_cross_fitted_predictions_of_r0_and_r2_are_those_of_the_threshold_classes(dev):
    r0 = platt.cross_fitted_rule_predictions("R0", dev.scores, dev.labels, dev.folds)
    r2 = platt.cross_fitted_rule_predictions("R2", dev.scores, dev.labels, dev.folds)

    _, expected_r0 = cross_fitted_predictions(dev.scores, dev.labels, dev.folds)
    pd.testing.assert_frame_equal(r0, expected_r0)
    wrapper = PlattLiptonThresholds(JevProbabilities()).fit(
        dev.scores, dev.labels, groups=dev.folds
    )
    np.testing.assert_array_equal(r2.to_numpy(), wrapper.oof_predictions_)
    assert r2.index.equals(dev.scores.index)


def test_cross_fitting_refuses_a_test_problem(dev):
    folds = dev.folds.copy()
    folds.iloc[0] = -1  # the test part's fold in the split

    for function in (platt.cross_fitted_rule_thresholds, platt.cross_fitted_rule_predictions):
        with pytest.raises(ValueError, match="negative fold"):
            function("R2", dev.scores, dev.labels, folds)
    with pytest.raises(ValueError, match="negative fold"):
        platt.nested_table(dev.scores, dev.labels, folds)


def test_rules_table_scores_each_rule_and_its_paired_difference_from_r0(dev):
    table = platt.rules_table(dev.scores, dev.labels, dev.folds, dev.groups)

    assert table.index.names == ["rule", "metric"]
    metrics = ["macro_f1", "micro_f1", *TAGS]
    assert table.index.tolist() == [(rule, metric) for rule in platt.RULES for metric in metrics]
    assert list(table.columns) == ["f1", "f1_low", "f1_high", "delta", "delta_low", "delta_high"]
    assert (table.loc["R0", "delta"] == 0).all()
    for rule in platt.RULES:
        predicted = platt.cross_fitted_rule_predictions(rule, dev.scores, dev.labels, dev.folds)
        macro_f1 = evaluate(dev.labels, predicted).macro_f1
        assert table.loc[(rule, "macro_f1"), "f1"] == pytest.approx(macro_f1)
        r0 = table.loc[("R0", "macro_f1"), "f1"]
        assert table.loc[(rule, "macro_f1"), "delta"] == pytest.approx(macro_f1 - r0)
        assert (table.loc[rule, "f1_low"] <= table.loc[rule, "f1_high"]).all()


def test_nested_table_chooses_inside_the_other_folds_and_scores_the_outer_fold(dev):
    table = platt.nested_table(dev.scores, dev.labels, dev.folds)

    assert table.index.tolist() == [0, 1, 2, 3, 4]
    assert table.index.name == "fold"
    assert list(table.columns) == list(platt.NESTED_COLUMNS)
    inner = dev.folds != 2
    inner_f1 = {}
    for rule in platt.RULES:
        predicted = platt.cross_fitted_rule_predictions(
            rule, dev.scores[inner], dev.labels[inner], dev.folds[inner]
        )
        inner_f1[rule] = evaluate(dev.labels[inner], predicted).macro_f1
        assert table.loc[2, f"inner_macro_f1_{rule}"] == pytest.approx(inner_f1[rule])
    chosen = table.loc[2, "chosen"]
    assert inner_f1[chosen] == max(inner_f1.values())
    thresholds = platt.rule_thresholds(chosen, dev.scores[inner], dev.labels[inner])
    outer = apply_thresholds(dev.scores[~inner], thresholds)
    assert table.loc[2, "fold_macro_f1"] == pytest.approx(
        evaluate(dev.labels[~inner], outer).macro_f1
    )
    # The nested macro F1 scores every dev problem once, with its outer fold's chosen rule.
    nested = pd.concat(
        apply_thresholds(
            dev.scores[dev.folds == fold],
            platt.rule_thresholds(
                table.loc[fold, "chosen"],
                dev.scores[dev.folds != fold],
                dev.labels[dev.folds != fold],
            ),
        )
        for fold in table.index
    ).loc[dev.scores.index]
    expected = evaluate(dev.labels, nested).macro_f1
    assert table["nested_macro_f1"].tolist() == [pytest.approx(expected)] * 5


def test_nested_table_keeps_the_first_rule_among_equal_scores():
    # Positives at 0.9 and negatives at 0.1: every rule's thresholds fall between the two, so
    # every rule predicts every fold alike, and R0, the first, is chosen everywhere.
    dev = synthetic_dev()
    scores = (dev.labels * 0.8 + 0.1).astype(float)

    table = platt.nested_table(scores, dev.labels, dev.folds)

    assert table["chosen"].tolist() == ["R0"] * 5
    assert table["nested_macro_f1"].tolist() == [1.0] * 5


def test_stability_table_counts_the_problems_whose_prediction_depends_on_the_folds(dev):
    table = platt.stability_table(dev.scores, dev.labels, dev.folds)

    assert table.index.names == ["tag", "rule"]
    assert table.index.tolist() == [(tag, rule) for tag in TAGS for rule in ("R0", "R2")]
    folds = [f"fold_{fold}" for fold in range(5)]
    assert list(table.columns) == [*folds, "spread", "final", "flips"]
    for rule in platt.STABILITY_RULES:
        fold_thresholds = platt.cross_fitted_rule_thresholds(
            rule, dev.scores, dev.labels, dev.folds
        )
        final = platt.rule_thresholds(rule, dev.scores, dev.labels)
        for tag in TAGS:
            row = table.loc[(tag, rule)]
            values = fold_thresholds[tag].to_numpy()
            assert row[folds].tolist() == values.tolist()
            assert row["final"] == final[tag]
            assert row["spread"] == pytest.approx(values.max() - values.min())
            # Reached by the lowest fold threshold, missed by the highest.
            score = dev.scores[tag]
            assert row["flips"] == ((score >= values.min()) & (score < values.max())).sum()


def test_stability_spread_is_zero_for_equal_thresholds_and_inf_for_some_infinite_ones(dev):
    # games keeps its positives of fold 2 only: the thresholds of fold 2, chosen on the other
    # folds, are inf, the others finite. No problem has probabilities.
    labels = dev.labels.assign(games=dev.labels["games"].where(dev.folds == 2, 0), probabilities=0)

    table = platt.stability_table(dev.scores, labels, dev.folds)

    for rule in platt.STABILITY_RULES:
        games = table.loc[("games", rule)]
        assert games["fold_2"] == math.inf
        assert games["spread"] == math.inf
        never = table.loc[("probabilities", rule)]
        assert never[[f"fold_{fold}" for fold in range(5)]].tolist() == [math.inf] * 5
        assert (never["spread"], never["final"], never["flips"]) == (0.0, math.inf, 0)


def written(out):
    """Return the bytes and the modification time of each file of a record, by its name."""
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in sorted(out.iterdir())
    }


def test_export_platt_writes_the_record_of_the_twins_rule_once(dev, tmp_path):
    out = tmp_path / TWIN / "tuning"

    assert platt.export_platt(TWIN, out, dev, BASE_FILES) == out

    assert sorted(path.name for path in out.iterdir()) == sorted(PLATT_FILES)
    params = json.loads((out / PARAMS_FILE).read_text(encoding="utf-8"))
    assert list(params) == [
        "rule",
        "threshold_rule",
        "base",
        "base_files",
        "thresholds",
        "nested_choice",
        "nested_macro_f1",
    ]
    nested = platt.nested_table(dev.scores, dev.labels, dev.folds)
    assert params["rule"] == "R2"
    assert params["threshold_rule"] == json_safe(RULE_SETTINGS)
    assert (params["base"], params["base_files"]) == (PLATT_BASES[TWIN], BASE_FILES)
    assert params["thresholds"] == json_safe(lipton_thresholds(dev.scores, dev.labels).to_dict())
    assert params["nested_choice"] == nested["chosen"].tolist()
    assert params["nested_macro_f1"] == nested["nested_macro_f1"].iloc[0]
    files = platt.platt_files(
        params,
        platt.rules_table(dev.scores, dev.labels, dev.folds, dev.groups),
        nested,
        platt.stability_table(dev.scores, dev.labels, dev.folds),
    )
    assert {name: (out / name).read_bytes() for name in files} == files
    assert not list(tmp_path.glob(f"{TWIN}/.tuning-*"))  # no staging folder is left behind


def test_export_platt_a_second_time_only_compares_and_refuses_an_edited_record(dev, tmp_path):
    out = tmp_path / "tuning"
    platt.export_platt(TWIN, out, dev, BASE_FILES)
    before = written(out)

    platt.export_platt(TWIN, out, dev, BASE_FILES)

    assert written(out) == before
    path = out / platt.RULES_FILE
    path.write_bytes(path.read_bytes().replace(b"R1", b"R9", 1))
    message = (
        f"holds another record of the threshold rule of {TWIN} than the one computed now: "
        "['rules.csv'] differ"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        platt.export_platt(TWIN, out, dev, BASE_FILES)


def test_export_platt_reads_the_focus_tags_of_the_dev_problems_only(dev, tmp_path):
    # The scores of other problems, such as test rows full of NaN, and of other
    # columns are never read; the dev rows may come in any order.
    test_rows = pd.DataFrame(np.nan, index=pd.Index([1001, 1002], name="sample_id"), columns=TAGS)
    scores = pd.concat([dev.scores.iloc[::-1], test_rows]).assign(extra=np.nan)
    platt.export_platt(TWIN, tmp_path / "clean", dev, BASE_FILES)

    platt.export_platt(TWIN, tmp_path / "mixed", dev._replace(scores=scores), BASE_FILES)

    assert written(tmp_path / "mixed").keys() == written(tmp_path / "clean").keys()
    for name in PLATT_FILES:
        assert (tmp_path / "mixed" / name).read_bytes() == (tmp_path / "clean" / name).read_bytes()
        assert b"nan" not in (tmp_path / "mixed" / name).read_bytes().lower()


def drop_a_problem(scores):
    """Leave out dev problem 4."""
    return scores.drop(index=4)


def lose_a_score(scores):
    """Make dev problem 4's probability of games NaN."""
    changed = scores.copy()
    changed.loc[4, "games"] = np.nan
    return changed


def drop_a_tag(scores):
    """Leave out the probabilities of games."""
    return scores.drop(columns="games")


@pytest.mark.parametrize("change", [drop_a_problem, lose_a_score, drop_a_tag])
def test_export_platt_refuses_dev_scores_without_every_dev_problem_and_tag(dev, tmp_path, change):
    scores = change(dev.scores)

    with pytest.raises(ValueError, match="lack a dev problem or a focus tag, or hold NaN"):
        platt.export_platt(TWIN, tmp_path / "tuning", dev._replace(scores=scores), BASE_FILES)

    assert not (tmp_path / "tuning").exists()


def test_export_platt_refuses_a_method_that_is_not_a_calibrated_twin(dev, tmp_path):
    with pytest.raises(ValueError, match="'jev-statement-code' is not a calibrated twin"):
        platt.export_platt("jev-statement-code", tmp_path / "tuning", dev, BASE_FILES)


def reverse_the_folds(dev):
    """Pass the folds of the dev problems in the reverse order."""
    return dev._replace(folds=dev.folds.iloc[::-1])


def drop_a_fold(dev):
    """Leave dev problem 4 out of the folds."""
    return dev._replace(folds=dev.folds.drop(index=4))


def put_a_test_problem(dev):
    """Give dev problem 4 the test part's fold, -1."""
    return dev._replace(folds=dev.folds.where(dev.folds.index != 4, -1))


def reverse_the_groups(dev):
    """Pass the groups of the dev problems in the reverse order."""
    return dev._replace(groups=dev.groups.iloc[::-1])


SAME_PROBLEMS = "must have the same index as problems: the same problems, in order."


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (reverse_the_folds, f"folds {SAME_PROBLEMS}"),
        (drop_a_fold, f"folds {SAME_PROBLEMS}"),
        (
            put_a_test_problem,
            "1 problems have a negative fold (the test set's -1): the test set is never "
            "cross-fitted. Pass the dev set, select_part(problems, split, 'dev').",
        ),
        (reverse_the_groups, f"groups {SAME_PROBLEMS}"),
    ],
)
def test_export_platt_refuses_folds_or_groups_of_other_problems_as_such(
    dev, tmp_path, change, message
):
    # Regression: a fold of the wrong problem is the caller's error, never a Platt fit that failed.
    with pytest.raises(ValueError) as caught:
        platt.export_platt(TWIN, tmp_path / "tuning", change(dev), BASE_FILES)

    assert str(caught.value) == message
    assert not (tmp_path / "tuning").exists()


def reverse_games(dev):
    """Score games lowest where it is true: every Platt slope of games is below 0."""
    return dev._replace(scores=dev.scores.assign(games=(1 - dev.scores["games"]).round(2)))


def games_on_every_problem_out_of_fold_3(dev):
    """Put games, scored 0.9, on every problem out of fold 3: without fold 3, it is one class."""
    out = dev.folds != 3
    labels = dev.labels.assign(games=dev.labels["games"].where(~out, 1))
    return dev._replace(
        scores=dev.scores.assign(games=dev.scores["games"].where(~out, 0.9)), labels=labels
    )


@pytest.mark.parametrize(
    ("change", "fold", "cause"),
    [
        (reverse_games, 1, "The Platt map's slope is -"),
        (games_on_every_problem_out_of_fold_3, 3, "these labels hold one class"),
    ],
)
def test_export_platt_names_the_twin_the_tag_and_the_fold_of_a_platt_fit_that_fails(
    dev, tmp_path, change, fold, cause
):
    # R1 and R2 fit a Platt map on each subset of folds, with no fallback. The
    # first to fail is R1's, inside outer fold 0, on the folds other than ``fold``.
    where = (
        f"The record of {TWIN} cannot be computed on the dev scores of jev-statement-code: "
        f"Inside the folds other than outer fold 0: On the folds other than fold {fold}: "
        "R1 cannot choose the threshold of 'games': "
    )

    with pytest.raises(ValueError, match=re.escape(where)) as caught:
        platt.export_platt(TWIN, tmp_path / "tuning", change(dev), BASE_FILES)

    assert cause in str(caught.value)
    assert "nothing falls back to another one" in str(caught.value)
    assert not (tmp_path / "tuning").exists()


def test_base_paths_name_the_answers_scores_and_run_record_of_the_base(tmp_path, monkeypatch):
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", tmp_path)
    base = tmp_path / "jev-statement-code"

    assert platt.base_paths("jev-statement-code") == {
        "answers_dev.jsonl": base / "answers_dev.jsonl",
        "answers_test.jsonl": base / "answers_test.jsonl",
        "scores.csv": base / "scores.csv",
        "run_record.json": base / "run_record.json",
    }


def test_read_platt_params_reads_an_infinite_threshold_as_text(dev, tmp_path):
    labels = dev.labels.assign(probabilities=0)
    platt.export_platt(TWIN, tmp_path / "tuning", dev._replace(labels=labels), BASE_FILES)

    params = platt.read_platt_params(tmp_path / "tuning")

    assert params["thresholds"]["probabilities"] == "inf"
    assert params["base_files"] == BASE_FILES


@pytest.fixture(name="exported")
def exported_fixture(dev, tmp_path):
    """Export the twin's record, with probabilities that no problem has; return tuning/."""
    out = tmp_path / "tuning"
    labels = dev.labels.assign(probabilities=0)
    platt.export_platt(TWIN, out, dev._replace(labels=labels), BASE_FILES)
    return out


def test_check_base_files_refuses_a_file_changed_or_missing_since_the_export(exported):
    changed = BASE_FILES | {"scores.csv": {"sha256": "e" * 64, "commit": "4" * 40}}
    without = {name: value for name, value in BASE_FILES.items() if name != "run_record.json"}

    platt.check_base_files(exported, BASE_FILES)  # the files params.json names pass

    with pytest.raises(ValueError, match=re.escape("now: ['scores.csv'] differ")):
        platt.check_base_files(exported, changed)
    with pytest.raises(ValueError, match=re.escape("now: ['run_record.json'] differ")):
        platt.check_base_files(exported, without)
    with pytest.raises(FileNotFoundError):
        platt.check_base_files(exported.parent / "not-exported", BASE_FILES)


def test_check_thresholds_compares_the_snapped_thresholds_exactly(exported):
    stated = platt.read_platt_params(exported)["thresholds"]
    thresholds = [float(stated[tag]) for tag in TAGS]  # "inf" for probabilities
    assert thresholds[-1] == math.inf

    platt.check_thresholds(exported, thresholds)

    with pytest.raises(ValueError, match="but the twin's rule gives"):
        platt.check_thresholds(exported, [thresholds[0] + 0.01, *thresholds[1:]])
    with pytest.raises(ValueError, match="but the twin's rule gives"):
        platt.check_thresholds(exported, [*thresholds[:-1], 0.995])


@pytest.mark.parametrize(("twin", "in_sample"), [(TWIN, False), (f"{GEPA_NAME}-platt", True)])
def test_twin_entries_name_the_base_the_rule_and_the_disclosure(twin, in_sample):
    base_entries = {"served_models": ["jev-1.13.0"], "dev_in_sample": in_sample, "answers": {}}

    entries = platt.twin_entries(twin, BASE_FILES, base_entries)

    assert entries == {
        "base": {"name": PLATT_BASES[twin], "files": BASE_FILES},
        "served_models": ["jev-1.13.0"],
        "threshold_rule": RULE_SETTINGS,
        "dev_in_sample": in_sample,
        "test_informed": TEST_INFORMED,
        "latency_from": PLATT_BASES[twin],
    }


def write_base_file(directory: Path, name: str, text: str) -> Path:
    """Write one file of a base into a new directory; return the directory."""
    directory.mkdir()
    (directory / name).write_text(text, encoding="utf-8")
    return directory


def test_a_twins_copies_are_its_bases_timings_and_calls_and_none_of_its_other_files(tmp_path):
    # The twin sends its base's requests: its latency and each problem's call are its base's,
    # while the files the record names by hash (answers, scores, run record) stay the base's.
    calls = "sample_id,part,latency_s\n1,dev,0.4\n2,dev,0.6\n"
    base = write_base_file(tmp_path / "base", PER_PROBLEM_FILE, calls)
    seconds = pd.Series([0.25, 0.5], index=pd.Index([1, 2], name="sample_id"), name="seconds")
    save_latency(base, Latency(seconds, bulk_s=1.0, load_s=0.125))
    for name in BASE_FILES:
        (base / name).write_text(name, encoding="utf-8")
    twin = tmp_path / "twin"
    twin.mkdir()

    for name in platt.BASE_COPIES:
        platt.copy_from_base(base, twin, name)

    timings = load_latency(twin)
    pd.testing.assert_series_equal(timings.seconds, seconds)
    assert (timings.bulk_s, timings.load_s, timings.calls) == (1.0, 0.125, None)
    assert (twin / PER_PROBLEM_FILE).read_text(encoding="utf-8") == calls
    assert sorted(path.name for path in twin.iterdir()) == sorted([LATENCY_FILE, PER_PROBLEM_FILE])


@pytest.mark.parametrize("name", [LATENCY_FILE, PER_PROBLEM_FILE])
def test_copy_from_base_copies_the_bases_file_once_then_only_compares(tmp_path, name):
    base = write_base_file(tmp_path / "base", name, "sample_id,seconds\n1,0.25\nbulk,1.0\n")
    twin = tmp_path / "twin"
    twin.mkdir()

    first = platt.copy_from_base(base, twin, name)
    mtime = (twin / name).stat().st_mtime_ns
    second = platt.copy_from_base(base, twin, name)

    assert first == f"copied {base / name} to {twin / name}: the twin sends its base's requests"
    assert second == f"{twin / name} is {base / name}, byte for byte"
    assert (twin / name).read_bytes() == (base / name).read_bytes()
    assert (twin / name).stat().st_mtime_ns == mtime
    assert sorted(path.name for path in twin.iterdir()) == [name]  # no partial file


def test_check_copy_refuses_a_missing_or_other_copy(tmp_path):
    base = write_base_file(tmp_path / "base", PER_PROBLEM_FILE, "sample_id,part\n1,dev\n")
    twin = tmp_path / "twin"
    twin.mkdir()

    with pytest.raises(ValueError, match="byte for byte: a calibrated twin sends its base's"):
        platt.check_copy(base, twin, PER_PROBLEM_FILE)
    (twin / PER_PROBLEM_FILE).write_text("sample_id,part\n1,test\n", encoding="utf-8")
    for check in (platt.check_copy, platt.copy_from_base):
        with pytest.raises(ValueError, match="is not .* byte for byte"):
            check(base, twin, PER_PROBLEM_FILE)
