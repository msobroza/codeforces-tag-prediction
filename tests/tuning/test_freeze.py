"""Tests for tuning freeze and verify: the tag model frozen as the tagging method tfidf.

They run on a tiny dataset written to a temporary folder. The ``study`` fixture writes the frozen
study of that dataset, as ``export`` does, and sets the test score that freeze must reproduce.
"""

import dataclasses
import json
import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import skops.io

from codeforces_tags import provenance
from codeforces_tags.config import (
    FOCUS_TAGS,
    MODEL_MAX_ITER,
    RAW_INPUT_COLUMNS,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.conformance import ConformanceError, check_frozen_method
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    load_predictions,
    load_run_record,
    load_scores,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    load_method,
    method_dir,
    model_dir,
    save_method,
)
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.tag_model import (
    MODEL_FILE,
    NAME,
    STORED_TYPES,
    TAG_MODEL_FILE,
    TagModel,
)
from codeforces_tags.model.training import OutOfFold, build_pipeline, cross_fit, fit_tag_model
from codeforces_tags.provenance import settings_snapshot, sha256_json
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import (
    load_logged_method,
    log_method,
    method_run_id,
    standard_metrics,
)
from codeforces_tags.tuning import __main__ as cli
from codeforces_tags.tuning import freeze as freezing
from codeforces_tags.tuning.results import (
    TuningResults,
    load_tuning_results,
    save_tuning_results,
)

# The kinds of problem of the dataset, in turn: tags, statement, solution. The eight focus tags
# are on the first four kinds, two each, so every focus tag is positive in every fold; the last
# kind has none.
KINDS = (
    (
        ["graphs", "trees"],
        "Count the leaves of the tree on these vertices and edges.",
        "adj = [[] for _ in range(n)]\nprint(len(adj))",
    ),
    (
        ["strings", "games"],
        "Alice and Bob take turns to erase a letter of the word.",
        "word = input()\nprint(word[::-1])",
    ),
    (
        ["math", "number theory"],
        "Print the greatest common divisor of the two numbers.",
        "from math import gcd\nprint(gcd(a, b))",
    ),
    (
        ["geometry", "probabilities"],
        "Find the probability that a random point falls in the circle.",
        "import math\nprint(math.pi / 4)",
    ),
    (["greedy"], "Sort the array, then print its largest element.", "a.sort()\nprint(a[-1])"),
)
DEV, TEST, FOLDS = 45, 5, 3  # dev: three folds of three problems of each kind
# The configuration of the study: a linear SVM loose enough (C = 3) to learn each problem's own
# number, in its statement. Every seventh problem has the tags of the next kind, so the refit,
# which learned them, predicts some dev problems otherwise than the models that never saw them.
PARAMS = ModelParams(
    classifier="linear_svm",
    c=3.0,
    class_weight="none",
    include_notes=False,
    statement_ngram_max=1,
    statement_min_df=1,
    statement_max_df=1.0,
    statement_sublinear_tf=False,
    code_weight=0.5,
    code_ngram_max=1,
    code_min_df=1,
    code_sublinear_tf=False,
    cue_weight=0.0,
    blocks=("statement", "code"),
)


@dataclasses.dataclass(frozen=True)
class Study:
    """The frozen study of the dataset, and what its configuration predicts.

    Attributes:
        results: The folder of the study's results, as ``export`` writes it.
        dev: The raw dev problems, with their fold and group.
        test: The raw test problems.
        out_of_fold: ``cross_fit`` of the configuration on dev, as ``export`` computes it.
        tag_model: The configuration fitted on dev, with ``fit_tag_model``.
    """

    results: Path
    dev: pd.DataFrame
    test: pd.DataFrame
    out_of_fold: OutOfFold
    tag_model: TagModel


def raw_record(make_raw_record, position):
    """Return the raw record of the problem at a position: its kind's, with its own number."""
    tags, statement, source_code = KINDS[position % len(KINDS)]
    if position % 7 == 3:
        tags = KINDS[(position + 1) % len(KINDS)][0]
    return make_raw_record(
        prob_desc_description=f"{statement} Problem {100 + position}.",
        source_code=source_code,
        tags=tags,
    )


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record):
    """Write the problems as sample files, ids 100 on, and return the dataset folder."""
    positions = range(DEV + TEST)
    return write_samples({100 + at: raw_record(make_raw_record, at) for at in positions})


@pytest.fixture(name="split_file")
def split_file_fixture(tmp_path):
    """Write the split: the first DEV problems dev, dealt to folds by kind, then TEST test ones."""
    positions = np.arange(DEV + TEST)
    is_dev = positions < DEV
    split = pd.DataFrame(
        {
            "sample_id": 100 + positions,
            "part": np.where(is_dev, "dev", "test"),
            "fold": np.where(is_dev, positions // len(KINDS) % FOLDS, -1),
            "group": 100 + positions,
        }
    )
    split.to_csv(tmp_path / "split.csv", index=False)
    return tmp_path / "split.csv"


@pytest.fixture(name="study")
def study_fixture(dataset, split_file, tmp_path, monkeypatch):
    """Write the frozen study of PARAMS, as export does, and set TFIDF_TEST_MACRO_F1 to its score.

    The test score is that of PARAMS fitted on dev, as notebook 04 scored the tuned model.
    """
    raw, split = read_samples(dataset), load_split(split_file)
    dev, test = select_part(raw, split, "dev"), select_part(raw, split, "test")
    labels = focus_labels(dev)
    out_of_fold = cross_fit(build_pipeline(PARAMS), dev, labels, dev["fold"])
    scores = evaluate(labels, out_of_fold.predictions).overall_f1()
    results = TuningResults(
        params=PARAMS,
        thresholds=out_of_fold.thresholds,
        trials=pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [scores["macro_f1"]]}),
        importances=pd.Series(name="importance", dtype=float).rename_axis("parameter"),
        out_of_fold=scores.to_frame("f1"),
    )
    save_tuning_results(results, tmp_path / "tfidf_baseline")
    tag_model = fit_tag_model(PARAMS, dev, labels, dev["fold"])
    test_score = evaluate(focus_labels(test), tag_model.predict(test)).macro_f1
    monkeypatch.setattr(freezing, "TFIDF_TEST_MACRO_F1", test_score)
    return Study(tmp_path / "tfidf_baseline", dev, test, out_of_fold, tag_model)


@pytest.fixture(name="assets")
def assets_fixture(tmp_path):
    """Return the folder of the frozen methods; nothing is frozen in it yet."""
    return tmp_path / "assets"


@pytest.fixture(name="common")
def common_fixture(dataset, split_file, study, assets):
    """Return the options both commands take: the dataset, split, study and assets."""
    return [
        "--data",
        str(dataset),
        "--split",
        str(split_file),
        "--results",
        str(study.results),
        "--assets",
        str(assets),
    ]


@pytest.fixture(name="store_options")
def store_options_fixture(store):
    """Return the options that point freeze to the temporary MLflow store."""
    return ["--tracking-uri", store["tracking_uri"], "--artifact-dir", str(store["artifact_dir"])]


@pytest.fixture(name="freeze_args")
def freeze_args_fixture(common, store_options):
    """Return the freeze command's arguments; the tree may be dirty while the tests run."""
    return ["freeze", "--allow-dirty", *common, *store_options]


@pytest.fixture(name="calls", autouse=True)
def calls_fixture(monkeypatch):
    """Replace the dataset check, which only the challenge dataset passes, and the MLflow log.

    Returns the arguments each was called with; one test puts the real ``log_method`` back.
    """
    calls = {"check_dataset": [], "log_method": []}

    def log(folder, record, metrics_by_dataset, **arguments):
        logged = {"folder": folder, "record": record, "metrics": metrics_by_dataset}
        calls["log_method"].append(logged | arguments)
        return "model-id"

    monkeypatch.setattr(freezing, "check_dataset", calls["check_dataset"].append)
    monkeypatch.setattr(freezing, "log_method", log)
    return calls


def files_of(directory):
    """Map each file under ``directory`` to its bytes and modification time."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}


def same_cells(frame, other):
    """Tell whether two frames have the same rows, columns and values, exactly."""
    same_axes = frame.index.equals(other.index) and list(frame.columns) == list(other.columns)
    return same_axes and np.array_equal(frame.to_numpy(), other.to_numpy())


def change_study(results, **changes):
    """Rewrite the frozen study's results with some of its fields changed."""
    save_tuning_results(dataclasses.replace(load_tuning_results(results), **changes), results)


def predict_fewer_tags(directory):
    """Save the frozen method folder again with every threshold raised by 10."""
    folder = directory / MODEL_FOLDER
    tag_model = load_method(folder)
    tag_model.estimator.thresholds_ = tag_model.estimator.thresholds_ + 10
    save_method(tag_model, folder)


def predict_a_dev_problem_otherwise(directory, study):
    """Save the frozen method folder again with one threshold raised past dev scores only.

    The tag is the first whose threshold has a dev score of the refit above it and below every
    test score above it: raised just past that dev score, the threshold predicts that dev problem
    otherwise, and every test problem as before.
    """
    folder = directory / MODEL_FOLDER
    tag_model = load_method(folder)
    thresholds = tag_model.thresholds
    dev, test = tag_model.scores(study.dev), tag_model.scores(study.test)
    lowest_test = test.where(test >= thresholds).min().fillna(np.inf)  # per tag
    between = dev.where((dev >= thresholds) & (dev < lowest_test))
    tag = between.count().idxmax()
    assert between[tag].count() > 0
    thresholds[tag] = np.nextafter(between[tag].max(), np.inf)
    tag_model.estimator.thresholds_ = thresholds.to_numpy()
    save_method(tag_model, folder)


def score_one_millionth_higher(directory):
    """Save the frozen method folder again with every tag's intercept raised by 1e-6.

    As a fit on another machine or code would, it scores every problem otherwise, though by too
    little to predict any other tag.
    """
    folder = directory / MODEL_FOLDER
    tag_model = load_method(folder)
    for model in tag_model.pipeline.named_steps["classifier"].estimators_:
        model.intercept_ = model.intercept_ + 1e-6
    save_method(tag_model, folder)


def test_freeze_saves_the_predictions_scores_run_record_and_method_folder(
    freeze_args, study, assets, calls, capsys
):
    directory = assets / NAME
    folder = directory / MODEL_FOLDER

    assert cli.main(freeze_args) == 0

    frozen = {MODEL_FOLDER, PREDICTIONS_FILE, RUN_RECORD_FILE, SCORES_FILE}
    assert {path.name for path in directory.iterdir()} == frozen
    assert {path.name for path in folder.iterdir()} == {METHOD_FILE, MODEL_FILE, TAG_MODEL_FILE}
    assert calls["check_dataset"] == [freeze_args[3]]
    macro_f1 = float(load_tuning_results(study.results).out_of_fold.at["macro_f1", "f1"])
    assert capsys.readouterr().out.splitlines() == [
        f"the frozen study in {study.results} is reproduced by the fitted tag model: every row "
        f"of out_of_fold.csv (out-of-fold macro F1 {macro_f1!r}), the thresholds within 1e-09 of "
        f"thresholds.json, and the test macro F1 (TFIDF_TEST_MACRO_F1 = "
        f"{freezing.TFIDF_TEST_MACRO_F1!r})",
        f"saved 50 predictions and scores to {directory}",
        f"saved the method folder {folder}: it predicts every dev and test problem as the fitted "
        "tag model does, and scores them within 1e-09 of it",
        "logged tfidf to MLflow: model model-id",
    ]


def test_freeze_writes_the_out_of_fold_dev_rows_not_the_refits(freeze_args, study, assets):
    # The refit predicts dev in sample, otherwise than out of fold.
    in_sample = study.tag_model.predict(study.dev)
    assert not same_cells(in_sample, study.out_of_fold.predictions)

    cli.main(freeze_args)

    predictions, scores = load_predictions(assets / NAME), load_scores(assets / NAME)
    assert same_cells(predictions.loc["dev"], study.out_of_fold.predictions)
    assert same_cells(scores.loc["dev"], study.out_of_fold.scores)
    assert same_cells(predictions.loc["test"], study.tag_model.predict(study.test))
    assert same_cells(scores.loc["test"], study.tag_model.scores(study.test))


def raise_an_out_of_fold_row_by_one_ulp(results, metric):
    """Write out_of_fold.csv with one row one float above what the study scored."""
    out_of_fold = load_tuning_results(results).out_of_fold.copy()
    out_of_fold.loc[metric, "f1"] = np.nextafter(out_of_fold.loc[metric, "f1"], 2)
    change_study(results, out_of_fold=out_of_fold)


def move_a_threshold_beyond_the_tolerance(results, tag):
    """Write thresholds.json with one tag's threshold moved by twice the tolerance."""
    thresholds = load_tuning_results(results).thresholds.copy()
    thresholds[tag] += 2 * TUNING_EXPORT_TOLERANCE
    change_study(results, thresholds=thresholds)


def lower_the_test_score_by_one_ulp(monkeypatch):
    """Set TFIDF_TEST_MACRO_F1 one float below what the refit scores on test."""
    test_score = float(np.nextafter(freezing.TFIDF_TEST_MACRO_F1, 0))
    monkeypatch.setattr(freezing, "TFIDF_TEST_MACRO_F1", test_score)


# What differs from the study -> how to make it differ, and the start of what freeze names: each
# row of out_of_fold.csv in turn (macro F1, micro F1, the F1 of each focus tag), then each tag's
# threshold.
NOT_REPRODUCED = {
    **{
        f"the {metric} row": (raise_an_out_of_fold_row_by_one_ulp, metric, f"{metric} is ")
        for metric in ("macro_f1", "micro_f1", *FOCUS_TAGS)
    },
    **{
        f"the {tag} threshold": (move_a_threshold_beyond_the_tolerance, tag, f"of {tag} is ")
        for tag in FOCUS_TAGS
    },
}


@pytest.fixture(name="not_reproduced", params=list(NOT_REPRODUCED))
def not_reproduced_fixture(request, study, monkeypatch):
    """Make the study differ as the case says, and the test score too; return what freeze names."""
    change, element, named = NOT_REPRODUCED[request.param]
    change(study.results, element)
    lower_the_test_score_by_one_ulp(monkeypatch)
    return named


def test_freeze_writes_nothing_when_the_study_is_not_reproduced(
    not_reproduced, freeze_args, assets, calls
):
    # The gates run before anything is written or logged. The message names the
    # one difference, and no other: no "; " joins a second one.
    pattern = f"not reproduced by the fitted tag model: [^;]*{not_reproduced}[^;]*\\. The code"
    with pytest.raises(ValueError, match=pattern) as refused:
        cli.main(freeze_args)

    # The test score is compared only once the study is reproduced.
    assert "test macro F1" not in str(refused.value)
    assert not assets.exists()  # no predictions.csv, no model/
    assert calls["log_method"] == []  # no MLflow run


def test_freeze_refuses_another_test_macro_f1_without_printing_it(
    freeze_args, study, assets, calls, monkeypatch
):
    test_score = evaluate(focus_labels(study.test), study.tag_model.predict(study.test)).macro_f1
    lower_the_test_score_by_one_ulp(monkeypatch)
    expected = (
        "is reproduced by the fitted tag model, whose test predictions do not score "
        f"TFIDF_TEST_MACRO_F1 = {freezing.TFIDF_TEST_MACRO_F1!r}"
    )

    with pytest.raises(ValueError, match=re.escape(expected)) as refused:
        cli.main(freeze_args)

    # Not the code, data or split: the study is reproduced. Nor the score, which is not frozen.
    assert "The code, data, split" not in str(refused.value)
    assert repr(test_score) not in str(refused.value)
    assert not assets.exists()
    assert calls["log_method"] == []


def test_freeze_records_the_recipe_and_the_methods_entries(freeze_args, study, assets, split_file):
    cli.main(freeze_args)

    frozen = load_run_record(assets / NAME)
    assert (frozen.method, frozen.allow_dirty) == (NAME, True)
    assert frozen.recipe_sha256 == freezing.recipe_sha256(study.results)
    assert frozen.split_sha256 == provenance.file_sha256(split_file)  # --split, not SPLIT_FILE
    assert set(METHODS[NAME].requirements) <= set(frozen.environment["packages"])
    classifier = study.tag_model.pipeline.named_steps["classifier"]
    assert frozen.method_entries == {
        "params": PARAMS.to_dict(),
        "thresholds": study.out_of_fold.thresholds.to_dict(),
        "max_n_iter": max(model.n_iter_ for model in classifier.estimators_),
        "dev_in_sample": False,
        "supersedes": [],
    }
    assert 0 < frozen.method_entries["max_n_iter"] < MODEL_MAX_ITER  # every model converged


def test_the_recipe_hashes_the_params_thresholds_settings_and_focus_tags(study):
    def read(name):
        return json.loads((study.results / name).read_text(encoding="utf-8"))

    recipe = freezing.recipe_sha256(study.results)

    assert recipe == sha256_json(
        {
            "params": read("params.json"),
            "settings": settings_snapshot(),
            "thresholds": read("thresholds.json"),
            "tags": list(FOCUS_TAGS),
        }
    )
    thresholds = load_tuning_results(study.results).thresholds
    change_study(study.results, thresholds=thresholds + TUNING_EXPORT_TOLERANCE / 2)
    assert freezing.recipe_sha256(study.results) != recipe


def test_the_frozen_files_pass_check_frozen_method(freeze_args, assets, split_file):
    cli.main(freeze_args)

    check_frozen_method(NAME, assets_dir=assets, split=load_split(split_file))


def test_freeze_logs_the_frozen_record_and_split_to_the_store(
    freeze_args, assets, split_file, store, calls
):
    cli.main(freeze_args)

    [logged] = calls["log_method"]
    assert logged["folder"] == assets / NAME / MODEL_FOLDER
    assert logged["record"] == load_run_record(assets / NAME)
    assert (logged["tracking_uri"], logged["artifact_dir"]) == tuple(store.values())
    pd.testing.assert_frame_equal(logged["split"], load_split(split_file))


def test_freeze_logs_each_parts_metrics_dev_out_of_fold(freeze_args, study, calls):
    cli.main(freeze_args)

    [logged] = calls["log_method"]
    parts = {
        "dev-oof": (study.dev, study.out_of_fold.predictions),
        "test": (study.test, study.tag_model.predict(study.test)),
    }
    assert list(logged["metrics"]) == list(parts)
    for dataset, (problems, predicted) in parts.items():
        expected = standard_metrics(focus_labels(problems), predicted, FOCUS_TAGS)
        assert logged["metrics"][dataset] == pytest.approx(expected, nan_ok=True)
    out_of_fold = load_tuning_results(study.results).out_of_fold["f1"]
    assert logged["metrics"]["dev-oof"]["macro_f1"] == out_of_fold["macro_f1"]
    assert logged["metrics"]["test"]["macro_f1"] == freezing.TFIDF_TEST_MACRO_F1


def test_a_second_freeze_rewrites_nothing_and_logs_the_frozen_record(
    freeze_args, assets, calls, capsys
):
    cli.main(freeze_args)
    frozen = files_of(assets)
    capsys.readouterr()

    assert cli.main(freeze_args) == 0

    assert files_of(assets) == frozen
    records = [logged["record"] for logged in calls["log_method"]]
    assert records == [load_run_record(assets / NAME)] * 2
    assert capsys.readouterr().out.splitlines()[1:3] == [
        f"50 predictions match the frozen files in {assets / NAME}",
        f"checked the method folder {assets / NAME / MODEL_FOLDER}: it predicts every dev and "
        "test problem as the fitted tag model does, and scores them within 1e-09 of it",
    ]


def test_freeze_refuses_a_method_folder_that_predicts_other_tags(freeze_args, assets, calls):
    cli.main(freeze_args)
    predict_fewer_tags(assets / NAME)

    with pytest.raises(ValueError, match="predicts other tags for the dev problems"):
        cli.main(freeze_args)

    assert len(calls["log_method"]) == 1


def test_freeze_refuses_a_method_folder_that_scores_otherwise(freeze_args, study, assets, calls):
    cli.main(freeze_args)
    score_one_millionth_higher(assets / NAME)
    folder = load_method(assets / NAME / MODEL_FOLDER)
    assert same_cells(folder.predict(study.test), study.tag_model.predict(study.test))

    with pytest.raises(ValueError, match="scores the dev problems further than 1e-09"):
        cli.main(freeze_args)

    assert len(calls["log_method"]) == 1


def test_freeze_refuses_a_method_folder_left_without_the_frozen_files(freeze_args, assets, calls):
    # Deleting the frozen files to freeze again leaves model/, which may come from another fit:
    # freeze would keep it, and verify would then refuse its scores.
    cli.main(freeze_args)
    directory = assets / NAME
    score_one_millionth_higher(directory)
    for name in (PREDICTIONS_FILE, SCORES_FILE, RUN_RECORD_FILE):
        (directory / name).unlink()

    with pytest.raises(ValueError, match="holds the method folder model/ but no predictions.csv"):
        cli.main(freeze_args)

    assert {path.name for path in directory.iterdir()} == {MODEL_FOLDER}  # nothing written
    assert len(calls["log_method"]) == 1


def another_configuration(results):
    """Write params.json with another C, as a new study that chose otherwise would."""
    change_study(results, params=dataclasses.replace(PARAMS, c=1.0))


def thresholds_moved_within_the_tolerance(results):
    """Write thresholds.json moved within the tolerance: the gates pass, but the recipe moved."""
    thresholds = load_tuning_results(results).thresholds
    change_study(results, thresholds=thresholds + TUNING_EXPORT_TOLERANCE / 2)


@pytest.fixture(
    name="another_recipe", params=[another_configuration, thresholds_moved_within_the_tolerance]
)
def another_recipe_fixture(request, frozen, study):
    """Change the study's recipe after the freeze, as the case says; return the frozen files."""
    request.param(study.results)
    return frozen


def test_freeze_refuses_another_recipe_before_it_fits_anything(
    another_recipe, freeze_args, calls, monkeypatch
):
    before = files_of(another_recipe)
    monkeypatch.setattr(freezing, "fit_tag_model", None)  # a fit would raise TypeError

    with pytest.raises(ValueError, match=f"{another_recipe} was frozen with another recipe"):
        cli.main(freeze_args)

    assert files_of(another_recipe) == before
    assert len(calls["log_method"]) == 1


def test_freeze_checks_the_dataset_then_refuses_a_dirty_tree(
    common, store_options, assets, calls, monkeypatch
):
    monkeypatch.setattr(provenance, "git_state", lambda root: {"commit": "0", "code_dirty": True})

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        cli.main(["freeze", *common, *store_options])

    assert calls["check_dataset"] == [common[1]]  # the dataset of --data
    assert (assets.exists(), calls["log_method"]) == (False, [])


def test_freeze_refuses_a_store_that_is_not_sqlite_before_any_work(
    common, assets, calls, tmp_path, capsys
):
    store = (tmp_path / "mlruns").as_uri()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["freeze", "--allow-dirty", *common, "--tracking-uri", store])

    assert exit_info.value.code == 2
    assert f"{store} is not a sqlite:/// store" in capsys.readouterr().err
    assert calls["check_dataset"] == []  # nothing read, fitted or written
    assert not assets.exists()


def test_freeze_logs_the_method_to_mlflow_once(freeze_args, study, assets, store, monkeypatch):
    monkeypatch.setattr(freezing, "log_method", log_method)

    cli.main(freeze_args)
    cli.main(freeze_args)

    recipe = load_run_record(assets / NAME).recipe_sha256
    assert method_run_id(recipe, **store) is not None
    [model] = mlflow.search_logged_models([use_store(**store)], output_format="list")
    logged = mlflow.get_logged_model(model.model_id).metrics
    assert sorted(metric.dataset_name for metric in logged if metric.key == "macro_f1") == [
        "dev-oof",
        "test",
    ]
    tag_model = load_logged_method(f"models:/{model.model_id}", **store)
    predicted = tag_model.predict(study.test[list(RAW_INPUT_COLUMNS)])
    frozen = load_predictions(assets / NAME).loc["test"]
    assert np.array_equal(predicted.to_numpy(), frozen.to_numpy())


@pytest.fixture(name="frozen")
def frozen_fixture(freeze_args, assets, capsys):
    """Freeze the method, and return its directory."""
    cli.main(freeze_args)
    capsys.readouterr()
    return assets / NAME


@pytest.fixture(name="verify_args")
def verify_args_fixture(common):
    """Return the verify command's arguments."""
    return ["verify", *common]


def test_verify_re_derives_the_frozen_files_at_both_levels_and_reports_the_scores(
    frozen, verify_args, study, calls, capsys
):
    assert cli.main(verify_args) == 0

    assert calls["check_dataset"] == [verify_args[2]] * 2
    folder = frozen / MODEL_FOLDER
    out_of_fold = load_tuning_results(study.results).out_of_fold["f1"]
    macro_f1, micro_f1 = float(out_of_fold["macro_f1"]), float(out_of_fold["micro_f1"])
    reproduced = (
        f"every row of out_of_fold.csv (out-of-fold macro F1 {macro_f1!r}), the "
        "thresholds within 1e-09 of thresholds.json, and the test macro F1 (TFIDF_TEST_MACRO_F1 "
        f"= {freezing.TFIDF_TEST_MACRO_F1!r})"
    )
    test = evaluate(focus_labels(study.test), study.tag_model.predict(study.test))
    assert capsys.readouterr().out.splitlines() == [
        "the frozen files of tfidf are complete and consistent with the split",
        f"they hold the recipe of {study.results}",
        "the thresholds cross-fitted on the dev rows of scores.csv give the dev rows of "
        "predictions.csv",
        f"the frozen study in {study.results} is reproduced by scores.csv and predictions.csv: "
        f"{reproduced}",
        f"the method folder {folder} scores the 5 test problems within 1e-09 of scores.csv and "
        "predicts them as predictions.csv",
        f"the frozen study in {study.results} is reproduced by the fitted tag model: {reproduced}",
        "the fitted tag model scores the 50 dev and test problems within 1e-09 of scores.csv and "
        "predicts them as predictions.csv",
        f"the method folder {folder} predicts every dev and test problem as the fitted tag model "
        "does, and scores them within 1e-09 of it",
        f"dev-oof: macro F1 {macro_f1!r}, micro F1 {micro_f1!r}",
        f"test: macro F1 {test.macro_f1!r}, micro F1 {test.micro_f1!r}",
    ]


def test_verify_writes_and_logs_nothing(frozen, verify_args, calls):
    before = files_of(frozen)

    cli.main(verify_args)

    assert files_of(frozen) == before
    assert len(calls["log_method"]) == 1  # the freeze's


def test_verify_refuses_frozen_files_that_are_missing(verify_args):
    with pytest.raises(ConformanceError, match="holds no method.json"):
        cli.main(verify_args)


def test_verify_refuses_another_recipe(frozen, verify_args, study):
    thresholds = load_tuning_results(study.results).thresholds
    change_study(study.results, thresholds=thresholds + TUNING_EXPORT_TOLERANCE / 2)

    with pytest.raises(ValueError, match=f"{frozen} was frozen with another recipe"):
        cli.main(verify_args)


def test_verify_refuses_dev_predictions_that_the_frozen_scores_do_not_give(frozen, verify_args):
    predictions = load_predictions(frozen)
    predictions.iloc[0] = 1 - predictions.iloc[0]
    predictions.to_csv(frozen / PREDICTIONS_FILE)

    with pytest.raises(
        ValueError, match="cross-fitted on the dev rows of scores.csv predict other"
    ):
        cli.main(verify_args)


@pytest.mark.usefixtures("frozen")
def test_verify_refuses_predictions_that_no_longer_score_the_study(verify_args, study):
    raise_an_out_of_fold_row_by_one_ulp(study.results, "micro_f1")

    with pytest.raises(ValueError, match="not reproduced by scores.csv and predictions.csv: micro"):
        cli.main(verify_args)


def test_verify_refuses_test_scores_that_the_method_folder_does_not_give(frozen, verify_args):
    scores = load_scores(frozen)
    scores.loc["test"] = scores.loc["test"].to_numpy() + 1e-6
    scores.to_csv(frozen / SCORES_FILE)

    with pytest.raises(
        ValueError, match="On the test problems, the method folder .* scores further"
    ):
        cli.main(verify_args)


def test_verify_refuses_a_method_folder_that_predicts_other_tags(frozen, verify_args):
    predict_fewer_tags(frozen)

    with pytest.raises(
        ValueError, match="On the test problems, the method folder .* predicts other"
    ):
        cli.main(verify_args)


def test_verify_refuses_a_method_folder_that_predicts_other_tags_for_dev_problems_only(
    frozen, verify_args, study
):
    # Level (a) reads the method folder on the test problems only: level (b) checks dev too.
    predict_a_dev_problem_otherwise(frozen, study)

    with pytest.raises(ValueError, match="predicts other tags for the dev problems"):
        cli.main(verify_args)


@pytest.mark.parametrize("tag", FOCUS_TAGS)
def test_verify_re_derivation_refuses_dev_scores_that_the_refit_does_not_give(
    frozen, verify_args, tag
):
    # Level (a) cannot tell: lowering the lowest dev score of a tag changes no prediction and
    # no threshold. Level (b) fits the tag model again, and finds the score it gives.
    scores = load_scores(frozen)
    lowest = scores.loc["dev", tag].idxmin()
    scores.loc[("dev", lowest), tag] -= 1
    scores.to_csv(frozen / SCORES_FILE)

    with pytest.raises(ValueError, match="the fitted tag model scores further than 1e-09"):
        cli.main(verify_args)


def test_the_committed_tfidf_is_frozen_with_the_recipe_of_the_study_and_the_code():
    # Needs no dataset: a changed params.json, thresholds.json or recorded setting makes the
    # committed recipe stale until tfidf is frozen again.
    check_frozen_method(NAME)

    record = load_run_record(method_dir(NAME))
    assert record.recipe_sha256 == freezing.recipe_sha256()
    assert (record.allow_dirty, record.git["code_dirty"]) == (False, False)
    entries = record.method_entries
    assert entries["params"] == load_tuning_results().params.to_dict()
    assert (entries["dev_in_sample"], entries["supersedes"]) == (False, [])
    assert 0 < entries["max_n_iter"] < MODEL_MAX_ITER


def test_the_committed_method_folder_holds_the_fit_of_the_frozen_scores():
    # Needs no dataset: the model keeps the out-of-fold scores its thresholds were chosen on,
    # which are the dev rows of scores.csv, bit for bit.
    tag_model = load_method(model_dir(NAME))

    assert same_cells(tag_model.out_of_fold_scores(), load_scores(method_dir(NAME)).loc["dev"])
    study = load_tuning_results()
    assert tag_model.params == study.params
    assert tag_model.thresholds.index.equals(study.thresholds.index)
    assert np.allclose(tag_model.thresholds, study.thresholds, rtol=0, atol=TUNING_EXPORT_TOLERANCE)


def test_the_committed_model_stores_only_trusted_types_and_predicts_raw_problems(raw_problems):
    # Runs in CI: a skops or scikit-learn upgrade that breaks the frozen folder fails here.
    stored = skops.io.get_untrusted_types(file=model_dir(NAME) / MODEL_FILE)
    assert set(stored) <= set(STORED_TYPES)

    tag_model = load_method(model_dir(NAME))

    # The tree problem also reads as math. Every score is at least 0.05 from its threshold.
    assert tag_model.predicted_tags(raw_problems).to_dict() == {
        30: ["math", "trees"],
        10: ["strings"],
        20: ["math", "number theory"],
    }
