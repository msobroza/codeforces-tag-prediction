"""Tests for logging tagging methods to MLflow and loading them back."""

import io
import json
import math
import os
import py_compile
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.data.pandas_dataset import from_pandas
from mlflow.exceptions import MlflowException

from codeforces_tags.config import (
    ALL_TAGS,
    FOCUS_TAGS,
    MLFLOW_METRIC_DATASETS,
    PAPER_TAGS,
)
from codeforces_tags.conformance import PICKLE_SUFFIXES, ConformanceError, starts_like_pickle
from codeforces_tags.evaluation.report import all_tag_macro_f1, evaluate
from codeforces_tags.methods import METHOD_FILE, MethodSpec
from codeforces_tags.provenance import sha256_json
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import (
    CODE_MODEL_LOADER,
    LOGGED_MODEL_NAME,
    PYFUNC_ENTRY,
    RUN_KIND_TAG,
    flat_record,
    load_logged_method,
    log_method,
    log_run_extras,
    logged_model_id,
    method_requirements,
    method_run_id,
    part_metrics,
    raw_input_signature,
    standard_metrics,
)

# Dataset-level metrics of a method, by dataset.
METRICS = {
    "dev-oof": {"macro_f1": 0.5, "micro_f1": 0.625},
    "test": {"macro_f1": 0.25, "micro_f1": 0.375},
}
# The split the metrics were computed on: two dev problems and one test problem.
SPLIT = pd.DataFrame(
    {"part": ["dev", "dev", "test"], "fold": [0, 1, -1], "group": [1, 2, 3]},
    index=pd.Index([10, 20, 30], name="sample_id"),
)


def lock_file(tmp_path, versions):
    """Write a ``uv.lock`` that pins each package at the given versions, and return it."""
    packages = [
        f'[[package]]\nname = "{name}"\nversion = "{each}"\n'
        for name, pinned in versions.items()
        for each in pinned
    ]
    path = tmp_path / "uv.lock"
    path.write_text("version = 1\n\n" + "\n".join(packages), encoding="utf-8")
    return path


def needs(*requirements):
    """Return a registry of one method, ``fake``, that needs ``requirements``."""
    return {"fake": MethodSpec(module="fake", kind="other", requirements=requirements)}


def log_fake(folder, registry, store):
    """Log a saved fake from code in a run of its own, and return the logged model's URI."""
    stored = json.loads((folder / METHOD_FILE).read_text(encoding="utf-8"))
    name = stored["method"]
    with mlflow.start_run(experiment_id=use_store(**store)):
        info = mlflow.pyfunc.log_model(
            name="fake",
            python_model=str(PYFUNC_ENTRY),
            artifacts={"method": str(folder)},
            signature=raw_input_signature(stored["tags"], registry[name].input_columns),
            pip_requirements=method_requirements(name, registry=registry),
        )
    return info.model_uri


@pytest.fixture(name="logged_uri")
def logged_uri_fixture(folders, registry, store):
    """Return the URI of the logged ``keyword_fake``, which reads every raw column."""
    return log_fake(folders["keyword_fake"], registry, store)


def model_root(store):
    """Return the folder of the only model logged in a store."""
    return next(store["artifact_dir"].rglob("MLmodel")).parent


def test_raw_input_signature_takes_required_strings_and_gives_a_long_per_tag():
    signature = raw_input_signature(("math", "trees"), ("prob_desc_description", "source_code"))

    assert signature.inputs.to_dict() == [
        {"type": "string", "name": "prob_desc_description", "required": True},
        {"type": "string", "name": "source_code", "required": True},
    ]
    assert signature.outputs.to_dict() == [
        {"type": "long", "name": "math", "required": True},
        {"type": "long", "name": "trees", "required": True},
    ]


def test_method_requirements_pin_each_requirement_to_its_installed_locked_version(tmp_path):
    versions = {"pandas": [version("pandas")], "numpy": ["0.0.1", version("numpy")]}
    lock = lock_file(tmp_path, versions)

    pinned = method_requirements("fake", registry=needs("pandas", "numpy"), lock=lock)

    assert pinned == [f"pandas=={version('pandas')}", f"numpy=={version('numpy')}"]


def test_method_requirements_read_the_project_lock_by_canonical_name():
    # uv.lock pins scipy once per Python fork; the installed version is the one kept.
    pinned = method_requirements("fake", registry=needs("scikit_learn", "scipy"))

    assert pinned == [f"scikit_learn=={version('scikit-learn')}", f"scipy=={version('scipy')}"]


def test_method_requirements_refuse_an_installed_version_the_lock_does_not_pin(tmp_path):
    lock = lock_file(tmp_path, {"pandas": ["0.0.1"]})

    with pytest.raises(ValueError, match=r"pandas .* is installed, but uv.lock pins \['0.0.1'\]"):
        method_requirements("fake", registry=needs("pandas"), lock=lock)


def test_method_requirements_refuse_a_requirement_that_is_not_installed():
    with pytest.raises(ValueError, match="surely-not-an-installed-distribution is not installed"):
        method_requirements("fake", registry=needs("surely-not-an-installed-distribution"))


def test_method_requirements_refuse_a_requirement_that_is_not_a_bare_name():
    with pytest.raises(ValueError, match="'pandas>=2' is not a distribution name"):
        method_requirements("fake", registry=needs("pandas>=2"))


def test_method_requirements_refuse_an_unregistered_method():
    with pytest.raises(ValueError, match="'missing' is not registered"):
        method_requirements("missing", registry=needs())


def test_a_logged_method_predicts_raw_problems_as_the_method_does(
    fakes, logged_uri, store, raw_problems
):
    # Safeguard 2: raw string columns, a frame indexed by sample_id with a list column (tags).
    predicted = load_logged_method(logged_uri, **store).predict(raw_problems)

    pd.testing.assert_frame_equal(predicted, fakes.KeywordFake().predict(raw_problems))
    assert predicted.index.tolist() == [30, 10, 20]


def test_a_logged_method_requires_exactly_its_input_columns(
    fakes, folders, registry, store, raw_problems
):
    uri = log_fake(folders["code_rule_fake"], registry, store)
    model = load_logged_method(uri, **store)

    signature = mlflow.models.get_model_info(uri).signature
    assert signature.inputs.input_names() == ["prob_desc_description", "source_code"]
    assert signature.outputs.input_names() == list(fakes.CodeRuleFake.tags)
    pd.testing.assert_frame_equal(
        model.predict(raw_problems[["prob_desc_description", "source_code"]]),
        fakes.CodeRuleFake().predict(raw_problems),
    )
    with pytest.raises(MlflowException, match="source_code"):
        model.predict(raw_problems.drop(columns="source_code"))


def test_a_logged_method_takes_a_missing_field_in_a_str_column_not_a_float_one(
    fakes, logged_uri, store, raw_problems
):
    # Safeguard 2, as load_logged_method's docstring states it: problem 30 has no notes, NaN in a
    # str column as read_samples() gives it. Typed by hand, np.nan alone makes a float64 column,
    # which MLflow's schema check refuses until the column is cast to str.
    model = load_logged_method(logged_uri, **store)
    as_read = raw_problems.loc[[30]]
    by_hand = as_read.assign(prob_desc_notes=np.nan)

    with pytest.raises(MlflowException, match="Can not safely convert float64"):
        model.predict(by_hand)
    pd.testing.assert_frame_equal(
        model.predict(by_hand.astype({"prob_desc_notes": "str"})),
        fakes.KeywordFake().predict(as_read),
    )


def test_load_logged_method_turns_pickle_loading_off(logged_uri, store, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")

    load_logged_method(logged_uri, **store)

    assert os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] == "false"


def test_load_logged_method_refuses_a_model_not_logged_from_code(logged_uri, store):
    mlmodel = model_root(store) / "MLmodel"
    text = mlmodel.read_text(encoding="utf-8")
    mlmodel.write_text(text.replace(CODE_MODEL_LOADER, "mlflow.pyfunc.model"), encoding="utf-8")

    with pytest.raises(ValueError, match="loads with mlflow.pyfunc.model"):
        load_logged_method(logged_uri, **store)


def test_load_logged_method_never_runs_bytecode_found_in_the_store(
    fakes, logged_uri, store, raw_problems, tmp_path
):
    # Safeguard 1: an unchecked .pyc next to the entry file runs instead of the reviewed .py.
    planted = tmp_path / "planted.py"
    planted.write_text('raise RuntimeError("planted bytecode ran")\n', encoding="utf-8")
    cache = model_root(store) / "__pycache__" / f"pyfunc_model.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(
        str(planted),
        cfile=str(cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    with pytest.raises(MlflowException, match="planted bytecode ran"):
        mlflow.pyfunc.load_model(logged_uri)

    predicted = load_logged_method(logged_uri, **store).predict(raw_problems)

    pd.testing.assert_frame_equal(predicted, fakes.KeywordFake().predict(raw_problems))
    assert list(cache.parent.iterdir()) == [cache]


def test_a_logged_method_holds_its_folder_and_requirements_but_no_pickle_lock_or_bytecode(
    logged_uri, store, raw_problems
):
    # Safeguard 3: explicit requirements, no uv.lock, nothing a load could unpickle.
    load_logged_method(logged_uri, **store).predict(raw_problems)

    root = model_root(store)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert [path.relative_to(root).as_posix() for path in files] == [
        "MLmodel",
        "artifacts/keyword_fake/fake.json",
        "artifacts/keyword_fake/method.json",
        "conda.yaml",
        "pyfunc_model.py",
        "python_env.yaml",
        "registered_model_meta",
        "requirements.txt",
    ]
    pickles = [path for path in files if path.suffix in PICKLE_SUFFIXES or starts_like_pickle(path)]
    assert pickles == []
    assert (root / "requirements.txt").read_text(encoding="utf-8").split() == [
        f"mlflow=={version('mlflow')}",
        f"numpy=={version('numpy')}",
        f"pandas=={version('pandas')}",
    ]


def method_runs(store):
    """Return every run of the store's tag-methods experiment."""
    return mlflow.search_runs([use_store(**store)], output_format="list")


def logged_metrics(model_id):
    """Return the metrics of a logged model, by dataset name, as (key, value) pairs."""
    by_dataset = {}
    for metric in mlflow.get_logged_model(model_id).metrics:
        assert metric.model_id == model_id
        by_dataset.setdefault(metric.dataset_name, []).append((metric.key, metric.value))
    return {name: sorted(pairs) for name, pairs in by_dataset.items()}


@pytest.fixture(name="logged")
def logged_fixture(folders, registry, store, placeholder_record):
    """Log ``code_rule_fake`` with ``log_method``; return its record and its logged model id."""
    record = placeholder_record("code_rule_fake")
    return record, log_method(
        folders["code_rule_fake"], record, METRICS, split=SPLIT, registry=registry, **store
    )


def test_flat_record_gives_dotted_keys_and_json_values(placeholder_record):
    record = placeholder_record(
        "label_prior",
        environment={"packages": {"numpy": "2.5.3"}, "threadpools": [{"user_api": "blas"}]},
        method_entries={"settings": {"k": 3}, "supersedes": [], "dev_in_sample": True},
    )

    flat = flat_record(record)

    assert flat["method"] == '"label_prior"'
    assert flat["git.commit"] == json.dumps("0" * 40)
    assert flat["git.code_dirty"] == "false"
    assert flat["environment.packages.numpy"] == '"2.5.3"'
    assert flat["environment.threadpools"] == '[{"user_api": "blas"}]'
    assert flat["settings.FOCUS_TAGS"] == json.dumps(list(FOCUS_TAGS))
    assert flat["prompts"] == "[]"
    assert flat["method_entries.settings.k"] == "3"
    assert flat["method_entries.supersedes"] == "[]"
    assert flat["method_entries.dev_in_sample"] == "true"


def test_flat_record_keeps_whole_a_mapping_whose_keys_are_not_mlflow_names(placeholder_record):
    thresholds = {"*special": math.inf, "math": 0.5}
    record = placeholder_record("all_tags", method_entries={"thresholds": thresholds}, git={})

    flat = flat_record(record)

    assert flat["method_entries.thresholds"] == '{"*special": "inf", "math": 0.5}'
    assert flat["git"] == "{}"
    assert json.loads(flat["environment.packages.pandas"]) == version("pandas")


def test_flat_record_refuses_a_value_longer_than_mlflow_keeps(placeholder_record):
    record = placeholder_record("few_shot", method_entries={"examples": "x" * 6000})

    with pytest.raises(ValueError, match="method_entries.examples"):
        flat_record(record)


def test_standard_metrics_score_the_focus_tags_of_a_focus_tag_method():
    rng = np.random.default_rng(0)
    index = pd.Index(range(40), name="sample_id")
    truth = pd.DataFrame(rng.integers(0, 2, (40, 37)), index=index, columns=ALL_TAGS)
    predictions = pd.DataFrame(rng.integers(0, 2, (40, 8)), index=index, columns=FOCUS_TAGS)

    metrics = standard_metrics(truth, predictions, FOCUS_TAGS)

    scores = evaluate(truth[list(FOCUS_TAGS)], predictions)
    per_tag = scores.per_tag.loc["all problems"]
    assert list(metrics) == [
        "macro_f1",
        "micro_f1",
        *(f"{metric}/{tag}" for metric in ("f1", "precision", "recall") for tag in FOCUS_TAGS),
    ]
    assert (metrics["macro_f1"], metrics["micro_f1"]) == (scores.macro_f1, scores.micro_f1)
    assert metrics["recall/number theory"] == per_tag.at["number theory", "recall"]
    assert {type(value) for value in metrics.values()} == {float}


def test_part_metrics_score_each_part_under_the_name_of_its_mlflow_dataset():
    rng = np.random.default_rng(2)
    index = pd.Index(range(30), name="sample_id")
    truth = pd.DataFrame(rng.integers(0, 2, (30, 8)), index=index, columns=FOCUS_TAGS)
    frozen = pd.DataFrame(rng.integers(0, 2, (30, 8)), index=index, columns=FOCUS_TAGS)
    predictions = pd.concat({"dev": frozen.iloc[:20], "test": frozen.iloc[20:]}, names=["part"])

    metrics = part_metrics(truth, predictions, FOCUS_TAGS)

    assert list(metrics) == ["dev-oof", "test"]
    assert metrics["dev-oof"] == standard_metrics(truth.iloc[:20], frozen.iloc[:20], FOCUS_TAGS)
    assert metrics["test"] == standard_metrics(truth.iloc[20:], frozen.iloc[20:], FOCUS_TAGS)


def test_standard_metrics_add_the_all_tag_scores_of_an_all_tag_method():
    rng = np.random.default_rng(1)
    index = pd.Index(range(40), name="sample_id")
    truth = pd.DataFrame(rng.integers(0, 2, (40, 37)), index=index, columns=ALL_TAGS)
    truth["schedules"] = 0
    predictions = pd.DataFrame(rng.integers(0, 2, (40, 37)), index=index, columns=ALL_TAGS)

    metrics = standard_metrics(truth, predictions, ALL_TAGS)

    assert list(metrics)[-3:] == [
        "all_tag_macro_f1",
        "all_tag_macro_f1_zero_absent",
        "paper_tag_macro_f1",
    ]
    assert metrics["all_tag_macro_f1"] == all_tag_macro_f1(truth, predictions, ALL_TAGS)[0]
    assert (
        metrics["all_tag_macro_f1_zero_absent"]
        == all_tag_macro_f1(truth, predictions, ALL_TAGS, absent="zero")[0]
    )
    assert metrics["paper_tag_macro_f1"] == all_tag_macro_f1(truth, predictions, PAPER_TAGS)[0]
    assert metrics["all_tag_macro_f1"] != metrics["all_tag_macro_f1_zero_absent"]


@pytest.mark.parametrize(
    ("columns", "tags"),
    [
        (FOCUS_TAGS[::-1], FOCUS_TAGS),
        (FOCUS_TAGS[:7], FOCUS_TAGS),
        (FOCUS_TAGS[:7], FOCUS_TAGS[:7]),  # the columns are the tags, which lack a focus tag
    ],
)
def test_standard_metrics_refuse_predictions_without_their_tags_or_a_focus_tag(columns, tags):
    index = pd.Index(range(4), name="sample_id")
    truth = pd.DataFrame(0, index=index, columns=FOCUS_TAGS)

    with pytest.raises(ValueError, match="every focus tag"):
        standard_metrics(truth, truth[list(columns)], tags)


def test_method_run_id_is_none_until_the_method_is_logged(logged, store):
    record, model_id = logged

    assert method_run_id(sha256_json("another recipe"), **store) is None
    run_id = method_run_id(record.recipe_sha256, **store)
    assert mlflow.get_logged_model(model_id).source_run_id == run_id


def test_method_run_id_skips_other_runs_of_the_recipe_and_refuses_two_method_runs(store):
    recipe = sha256_json("recipe")
    tags = {"recipe_sha256": recipe, RUN_KIND_TAG: "method"}
    experiment_id = use_store(**store)
    mlflow.MlflowClient().create_run(experiment_id, tags={"recipe_sha256": recipe, "part": "dev"})
    assert method_run_id(recipe, **store) is None

    first = mlflow.MlflowClient().create_run(experiment_id, tags=tags).info.run_id
    assert method_run_id(recipe, **store) == first
    mlflow.MlflowClient().create_run(experiment_id, tags=tags)

    with pytest.raises(ValueError, match="2 method runs"):
        method_run_id(recipe, **store)


def test_method_run_id_refuses_a_recipe_that_is_not_a_sha256_digest(store):
    with pytest.raises(ValueError, match="not a sha256"):
        method_run_id("x' OR tags.method = 'tfidf", **store)


def test_log_method_logs_one_run_and_one_model_from_code(logged, registry):
    record, model_id = logged

    model = mlflow.get_logged_model(model_id)
    run = mlflow.get_run(model.source_run_id)
    assert run.info.run_name == "code_rule_fake"
    assert {key: run.data.tags[key] for key in ("method", "recipe_sha256", RUN_KIND_TAG)} == {
        "method": "code_rule_fake",
        "recipe_sha256": record.recipe_sha256,
        RUN_KIND_TAG: "method",
    }
    assert "dev_in_sample" not in run.data.tags
    assert "test_informed" not in run.data.tags
    assert (model.name, model.params) == (LOGGED_MODEL_NAME, flat_record(record))
    info = mlflow.models.get_model_info(f"models:/{model_id}")
    assert info.flavors["python_function"]["loader_module"] == CODE_MODEL_LOADER
    assert info.signature.inputs.input_names() == list(registry["code_rule_fake"].input_columns)


def test_log_method_logs_each_datasets_metrics_on_the_model_with_the_parts_problems(logged):
    _, model_id = logged

    assert logged_metrics(model_id) == {
        name: sorted(metrics.items()) for name, metrics in METRICS.items()
    }
    run = mlflow.get_run(mlflow.get_logged_model(model_id).source_run_id)
    inputs = {each.dataset.name: each.dataset for each in run.inputs.dataset_inputs}
    for part, name in MLFLOW_METRIC_DATASETS.items():
        sample_ids = pd.DataFrame({"sample_id": SPLIT.index[SPLIT["part"] == part]})
        assert inputs[name].digest == from_pandas(sample_ids, name=name).digest


def test_log_method_twice_logs_once(logged, folders, registry, store, placeholder_record):
    # A second freeze mirrors to MLflow without duplicating anything.
    record, model_id = logged
    again = placeholder_record("code_rule_fake", created_at="2026-10-08T00:00:00+00:00")

    assert log_method(
        folders["code_rule_fake"], again, METRICS, split=SPLIT, registry=registry, **store
    ) == (model_id)
    assert [run.info.run_id for run in method_runs(store)] == [
        method_run_id(record.recipe_sha256, **store)
    ]
    assert len(mlflow.search_logged_models([use_store(**store)], output_format="list")) == 1
    assert mlflow.get_logged_model(model_id).params == flat_record(record)
    assert logged_metrics(model_id) == {
        name: sorted(metrics.items()) for name, metrics in METRICS.items()
    }


def test_log_method_logs_a_dataset_it_had_no_metrics_of(
    folders, registry, store, placeholder_record
):
    record = placeholder_record("code_rule_fake")
    folder = folders["code_rule_fake"]
    model_id = log_method(
        folder, record, {"dev-oof": METRICS["dev-oof"]}, split=SPLIT, registry=registry, **store
    )

    log_method(
        folder,
        record,
        {"dev-oof": {"macro_f1": 0.0}} | METRICS,
        split=SPLIT,
        registry=registry,
        **store,
    )

    assert logged_metrics(model_id) == {
        name: sorted(metrics.items()) for name, metrics in METRICS.items()
    }


@pytest.mark.parametrize("first", [["dev-oof", "test"], ["dev-oof"], ["test"]])
def test_log_method_shows_the_out_of_fold_dev_scores_on_the_method_run(
    folders, registry, store, first, placeholder_record
):
    # mlflow.log_metrics also writes the run's metrics, whose view keeps each key's latest
    # value: it must show the selection score (ADR 0001), never the test score, whatever
    # dataset was logged first.
    metrics = {"dev-oof": {"macro_f1": 0.125}, "test": {"macro_f1": 0.875}}
    record = placeholder_record("code_rule_fake")
    folder = folders["code_rule_fake"]
    some = {name: metrics[name] for name in first}
    log_method(folder, record, some, split=SPLIT, registry=registry, **store)

    model_id = log_method(folder, record, metrics, split=SPLIT, registry=registry, **store)

    run = mlflow.get_run(mlflow.get_logged_model(model_id).source_run_id)
    assert run.data.metrics["macro_f1"] == 0.125
    assert run.data.metrics == search_run_metrics(store, run.info.run_id)
    assert logged_metrics(model_id) == {
        name: sorted(values.items()) for name, values in metrics.items()
    }


def search_run_metrics(store, run_id):
    """Return a run's metrics as search_runs, which the runs table sorts by, returns them."""
    [run] = mlflow.search_runs(
        [use_store(**store)], filter_string=f"attributes.run_id = '{run_id}'", output_format="list"
    )
    return run.data.metrics


def test_log_method_refuses_a_logged_model_whose_logging_failed(
    folders, registry, store, monkeypatch, placeholder_record
):
    # MLflow links the model to the run before it saves the artifacts: a failure in between
    # leaves a FAILED model without artifacts, which no later call may take for the method.
    record = placeholder_record("code_rule_fake")
    folder = folders["code_rule_fake"]

    def full_disk(*args, **kwargs):
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(mlflow.MlflowClient, "log_model_artifacts", full_disk)
        with pytest.raises(OSError, match="No space left"):
            log_method(folder, record, METRICS, split=SPLIT, registry=registry, **store)

    with pytest.raises(ValueError, match=r"is FAILED, not READY: .* freeze again"):
        log_method(folder, record, METRICS, split=SPLIT, registry=registry, **store)
    with pytest.raises(ValueError, match="FAILED, not READY"):
        logged_model_id(method_run_id(record.recipe_sha256, **store))
    [model] = mlflow.search_logged_models([use_store(**store)], output_format="list")
    assert not mlflow.get_logged_model(model.model_id).metrics


def test_load_logged_method_reads_the_store_it_is_given(
    logged, store, raw_problems, tmp_path, monkeypatch
):
    # A fresh process, such as a notebook kernel, has no store yet: MLflow's default is
    # ./mlflow.db in the working directory, which load_logged_method must never create.
    _, model_id = logged
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(None)

    predicted = load_logged_method(f"models:/{model_id}", **store).predict(raw_problems)

    assert predicted.index.equals(raw_problems.index)
    assert not list((tmp_path / "cwd").iterdir())


def test_log_method_logs_the_model_of_a_run_left_without_one(
    folders, registry, store, placeholder_record
):
    record = placeholder_record("code_rule_fake")
    tags = {"method": record.method, "recipe_sha256": record.recipe_sha256, RUN_KIND_TAG: "method"}
    experiment_id = use_store(**store)
    run = mlflow.MlflowClient().create_run(experiment_id, tags=tags)

    model_id = log_method(
        folders["code_rule_fake"], record, {}, split=SPLIT, registry=registry, **store
    )

    assert mlflow.get_logged_model(model_id).source_run_id == run.info.run_id
    assert len(method_runs(store)) == 1


def test_log_method_refuses_a_method_run_holding_two_models(
    folders, registry, store, placeholder_record
):
    record = placeholder_record("code_rule_fake")
    tags = {"method": record.method, "recipe_sha256": record.recipe_sha256, RUN_KIND_TAG: "method"}
    with mlflow.start_run(experiment_id=use_store(**store), tags=tags):
        for name in ("first", "second"):
            mlflow.pyfunc.log_model(name=name, python_model=str(PYFUNC_ENTRY), pip_requirements=[])

    with pytest.raises(ValueError, match="holds 2 models, not one"):
        log_method(
            folders["code_rule_fake"], record, METRICS, split=SPLIT, registry=registry, **store
        )


def test_log_method_tags_a_method_whose_dev_score_is_in_sample(
    folders, registry, store, placeholder_record
):
    record = placeholder_record(
        "code_rule_fake", method_entries={"supersedes": [], "dev_in_sample": True}
    )

    model_id = log_method(
        folders["code_rule_fake"], record, METRICS, split=SPLIT, registry=registry, **store
    )

    model = mlflow.get_logged_model(model_id)
    assert model.tags["dev_in_sample"] == "true"
    assert mlflow.get_run(model.source_run_id).data.tags["dev_in_sample"] == "true"


def test_log_method_tags_a_method_proposed_after_a_test_score_was_read(
    folders, registry, store, placeholder_record
):
    # docs/adr/0011: a calibrated twin's freeze records test_informed, and its run says so.
    entries = {"supersedes": [], "test_informed": {"note": "proposed after a test score"}}
    record = placeholder_record("code_rule_fake", method_entries=entries)

    model_id = log_method(
        folders["code_rule_fake"], record, METRICS, split=SPLIT, registry=registry, **store
    )

    model = mlflow.get_logged_model(model_id)
    run_tags = mlflow.get_run(model.source_run_id).data.tags
    assert (model.tags["test_informed"], run_tags["test_informed"]) == ("true", "true")
    assert "dev_in_sample" not in run_tags


def test_log_method_links_the_prompts_of_the_run_record(
    folders, registry, store, placeholder_record
):
    use_store(**store)
    mlflow.genai.register_prompt(name="code2tag", template="Tag this solution: {{ code }}")
    record = placeholder_record("keyword_fake", prompts=("prompts:/code2tag/1",))

    model_id = log_method(
        folders["keyword_fake"], record, METRICS, split=SPLIT, registry=registry, **store
    )

    linked = json.loads(mlflow.get_logged_model(model_id).tags["mlflow.linkedPrompts"])
    assert linked == [{"name": "code2tag", "version": "1"}]


def test_a_method_logged_by_log_method_loads_and_predicts_as_saved(
    fakes, logged, store, raw_problems
):
    _, model_id = logged

    predicted = load_logged_method(f"models:/{model_id}", **store).predict(raw_problems)

    pd.testing.assert_frame_equal(predicted, fakes.CodeRuleFake().predict(raw_problems))
    root = model_root(store)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert [path.relative_to(root).as_posix() for path in files] == [
        "MLmodel",
        "artifacts/code_rule_fake/fake.json",
        "artifacts/code_rule_fake/method.json",
        "conda.yaml",
        "pyfunc_model.py",
        "python_env.yaml",
        "registered_model_meta",
        "requirements.txt",
    ]
    stored = [path for path in store["artifact_dir"].rglob("*") if path.is_file()]
    assert {path.name for path in stored} & {"uv.lock", "python_model.pkl"} == set()
    assert not any(path.suffix in PICKLE_SUFFIXES or starts_like_pickle(path) for path in stored)
    assert not any("__pycache__" in path.parts for path in stored)


def test_log_method_refuses_a_dataset_it_does_not_know(
    folders, registry, store, placeholder_record
):
    record = placeholder_record("code_rule_fake")

    with pytest.raises(ValueError, match=r"\['dev'\]"):
        log_method(
            folders["code_rule_fake"], record, {"dev": {}}, split=SPLIT, registry=registry, **store
        )

    assert method_runs(store) == []


def test_log_method_refuses_the_folder_of_another_method(
    folders, registry, store, placeholder_record
):
    record = placeholder_record("keyword_fake")

    with pytest.raises(ValueError, match="holds the method 'code_rule_fake'"):
        log_method(
            folders["code_rule_fake"], record, METRICS, split=SPLIT, registry=registry, **store
        )

    assert method_runs(store) == []


def test_log_method_refuses_an_unregistered_method(folders, store, placeholder_record):
    record = placeholder_record("code_rule_fake")

    with pytest.raises(ValueError, match="'code_rule_fake' is not registered"):
        log_method(folders["code_rule_fake"], record, METRICS, split=SPLIT, **store)

    assert method_runs(store) == []


def test_log_method_refuses_a_folder_that_holds_a_pickle(
    folders, registry, store, placeholder_record
):
    (folders["code_rule_fake"] / "model.pkl").write_bytes(b"\x80\x04.")

    with pytest.raises(ConformanceError, match="model.pkl"):
        log_method(
            folders["code_rule_fake"],
            placeholder_record("code_rule_fake"),
            METRICS,
            split=SPLIT,
            registry=registry,
            **store,
        )

    assert method_runs(store) == []


def test_log_run_extras_adds_metrics_figures_and_tables_once(logged, store):
    record, model_id = logged
    figure, axes = plt.subplots()
    axes.plot([0, 1], [1, 0])
    table = pd.DataFrame({"seconds": [0.5, 0.25]}, index=pd.Index([10, 20], name="sample_id"))
    extras = {
        "metrics": {"latency/median_s": 0.375},
        "figures": {"figures/latency.png": figure},
        "tables": {"latency.csv": table},
    }

    log_run_extras(record.recipe_sha256, **extras, **store)
    log_run_extras(
        record.recipe_sha256,
        metrics={"latency/median_s": 9.0},
        figures={"figures/latency.png": plt.figure()},
        tables={"latency.csv": table * 2},
        **store,
    )
    plt.close("all")

    run_id = method_run_id(record.recipe_sha256, **store)
    history = mlflow.MlflowClient().get_metric_history(run_id, "latency/median_s")
    assert [point.value for point in history] == [0.375]
    assert {metric.key for metric in mlflow.get_logged_model(model_id).metrics} == {
        "macro_f1",
        "micro_f1",
    }
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/latency.csv")
    pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(text), index_col="sample_id"), table)
    image = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/figures/latency.png")
    assert Path(image).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_log_run_extras_refuses_a_recipe_with_no_method_run(store):
    with pytest.raises(ValueError, match="run the method's freeze first"):
        log_run_extras(sha256_json("never frozen"), metrics={"latency/max_s": 1.0}, **store)
