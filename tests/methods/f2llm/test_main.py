"""Tests for the F2LLM commands, on synthetic problems and the tiny encoder: nothing downloads."""

import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info

from codeforces_tags import provenance
from codeforces_tags.config import (
    DEVICE_ENV_VAR,
    FOCUS_TAGS,
    LATENCY_SEED,
    METHODS_ASSETS_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
    TUNING_BLAS_THREADS,
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
from codeforces_tags.methods import METHOD_FILE, METHODS, MODEL_FOLDER, MethodSpec
from codeforces_tags.methods.f2llm import __main__ as cli
from codeforces_tags.methods.f2llm.analysis import (
    INPUT_LENGTHS_FILE,
    PAIR_COUNTS_FILE,
    PAIRS_FILE,
    TFIDF_REFERENCE_FILE,
)
from codeforces_tags.methods.f2llm.drift import suggest_score_atol
from codeforces_tags.methods.f2llm.embedding import SOLUTION, STATEMENT, encoder_inputs, text_key
from codeforces_tags.methods.f2llm.encoder import (
    ENCODER_FILE,
    PRETRAINED_FILE,
    PretrainedRecord,
    frozen_encoder,
    loaded_device,
)
from codeforces_tags.methods.f2llm.estimators import ScoreFusion, tfidf_params
from codeforces_tags.methods.f2llm.method import MODEL_FILE, THRESHOLDS_FILE
from codeforces_tags.methods.f2llm.setfit import SetFitEstimator
from codeforces_tags.methods.f2llm.settings import (
    AUTO_DEVICE,
    EARLY_FUSION,
    LATE_FUSION,
    LATE_FUSION_GRID,
    PROBE,
    PROBE_GRID,
    SETFIT,
    SETFIT_METHOD,
    SETFIT_SMOKE_PAIRS,
    SETTINGS,
    TFIDF_TEST_MACRO_F1,
)
from codeforces_tags.methods.f2llm.tuning import (
    INDEX_PARAM,
    JOURNAL_FILE,
    OUT_OF_FOLD_FILE,
    PARAMS_FILE,
    TRIALS_FILE,
    TUNING_FOLDER,
    open_study,
)
from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    cross_fitted_thresholds,
    per_tag_thresholds,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.splitting import load_split
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import log_method, method_run_id

# The probe's chosen configuration in these tests: a logistic regression on both inputs.
PROBE_PARAMS = next(
    params
    for params in PROBE_GRID
    if params["c"] == 1 and params["class_weight"] == "none" and params["solution_weight"] == 0.5
)
# The tiny encoder's embedding width (its hidden size).
TINY_WIDTH = 32
# The F2LLM lines' score_atol in the registry the commands read here, not the measured one.
SCORE_ATOL = 1e-4


def snapshot(directory):
    """Map each file under ``directory`` to its bytes and modification time."""
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def split_part(common, part):
    """Return one part's rows of the split ``common`` names, by sample id as read_samples sorts."""
    split = load_split(common[3])
    return split[split["part"] == part].sort_index()


@pytest.fixture(name="tiny_commands", autouse=True)
def tiny_commands_fixture(calls):
    """Point every command of these tests at the tiny encoder and at tmp_path, first (calls)."""
    return calls


def test_parse_args_gives_each_command_its_options_and_defaults():
    embed = cli.parse_args(["embed"])
    atol = cli.parse_args(["atol"])
    tune = cli.parse_args(["tune", "--method", LATE_FUSION])
    freeze = cli.parse_args(["freeze", "--method", PROBE])
    verify = cli.parse_args(["verify", "--method", PROBE, SETFIT_METHOD])

    assert vars(cli.parse_args(["fetch"])) == {"command": "fetch", "method": None}
    # The command line's fetch hint names the method; the four share one encoder.
    assert vars(cli.parse_args(["fetch", "--method", PROBE])) == {
        "command": "fetch",
        "method": PROBE,
    }
    # verify re-predicts on the CPU unless asked: --device auto or cuda is opt-in.
    assert (embed.device, atol.device, freeze.device, verify.device) == ("cuda",) * 3 + ("cpu",)
    assert cli.parse_args(["verify", "--method", PROBE, "--device", AUTO_DEVICE]).device == (
        AUTO_DEVICE
    )
    assert (embed.data, embed.split, tune.method) == (None, SPLIT_FILE, LATE_FUSION)
    assert not hasattr(tune, "assets")
    assert (freeze.method, freeze.assets, freeze.allow_dirty) == (PROBE, METHODS_ASSETS_DIR, False)
    assert (freeze.tracking_uri, freeze.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)
    assert (verify.method, verify.assets, verify.retrain, verify.single) == (
        [PROBE, SETFIT_METHOD],
        METHODS_ASSETS_DIR,
        False,
        False,  # the test rows in one call, unless --single
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["tune", "--method", SETFIT_METHOD],  # SetFit tunes nothing
        ["fetch", "--method", "modernbert_lora"],
        ["export", "--method", "label_prior"],
        ["freeze", "--method", "label_prior"],
        ["embed", "--device", "tpu"],
        ["verify"],
    ],
)
def test_parse_args_refuses_other_methods_and_devices(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


# The command line's fetch hint names the method: every name fetches the one shared encoder.
@pytest.mark.parametrize("argv", [["fetch"], ["fetch", "--method", LATE_FUSION]])
def test_fetch_downloads_the_pinned_weights_and_names_their_folder(argv, monkeypatch, capsys):
    monkeypatch.undo()  # the pinned F2LLM weights, not the tiny encoder of the other tests
    fetched = []

    def fetch(record):
        fetched.append(record)
        return Path("weights")

    monkeypatch.setattr(cli, "fetch", fetch)

    assert cli.main(argv) == 0

    pinned = PretrainedRecord.from_settings()
    assert fetched == [pinned]
    assert capsys.readouterr().out == (
        f"the {len(pinned.files)} files of {pinned.repo_id} at {pinned.revision} are in weights, "
        "each with its pinned sha256\n"
    )


def test_embed_fills_the_cache_once_then_writes_the_shared_files_once(
    common, tmp_path, tiny_config, calls, capsys
):
    texts = encoder_inputs(raw_to_problems(read_samples(common[1])), tiny_config)
    distinct = len(set(texts[STATEMENT] + texts[SOLUTION]))  # test problems repeat dev texts

    assert cli.main(["embed", "--device", "cpu", *common]) == 0
    first = capsys.readouterr().out.splitlines()
    shared = snapshot(tmp_path / "shared")
    assert cli.main(["embed", "--device", "cpu", *common]) == 0
    second = capsys.readouterr().out.splitlines()

    assert len(list((tmp_path / "cache").glob("*.npy"))) == distinct
    assert f"embedded {distinct} encoder inputs on cpu; the others were cached" in first
    assert "embedded 0 encoder inputs on cpu; the others were cached" in second
    assert any(line.endswith(" per s") and line.startswith("encoded ") for line in first)
    assert not any(line.startswith("encoded ") for line in second)  # nothing encoded again
    assert sorted(shared) == sorted([INPUT_LENGTHS_FILE, PAIRS_FILE, PAIR_COUNTS_FILE])
    assert snapshot(tmp_path / "shared") == shared  # checked by the second run, not rewritten
    assert calls["check_dataset"] == [common[1]] * 2


def test_embed_on_the_gpu_prints_its_peak_memory(common, monkeypatch, capsys):
    monkeypatch.setattr(cli, "resolve_device", lambda device: device)
    monkeypatch.setattr(
        cli,
        "frozen_encoder",
        lambda record, config, _device: frozen_encoder(record, config, "cpu"),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 2**30)

    cli.main(["embed", *common])  # --device cuda by default

    printed = capsys.readouterr().out.splitlines()
    assert "peak GPU memory: 3.00 GiB" in printed
    assert any(line.endswith("encoder inputs on cuda; the others were cached") for line in printed)


def test_reference_writes_the_tfidf_reference_rows_once(common, tmp_path, monkeypatch, capsys):
    received = []
    rows = pd.DataFrame({"part": ["dev"], "sample_id": [1], "kind": ["score"], "math": [0.5]})

    def reference(dev, test, labels, params, **expected):
        received.append((list(dev.index), list(test.index), labels.shape, params, expected))
        return rows

    monkeypatch.setattr(cli, "tfidf_reference", reference)

    assert cli.main(["reference", *common]) == 0
    written = snapshot(tmp_path / "shared")
    assert cli.main(["reference", *common]) == 0

    assert list(written) == [TFIDF_REFERENCE_FILE]
    assert snapshot(tmp_path / "shared") == written
    dev_ids = list(split_part(common, "dev").index)
    test_ids = list(split_part(common, "test").index)
    shape = (len(dev_ids) + len(test_ids), len(FOCUS_TAGS))
    expected = {
        # The frozen TF-IDF's out-of-fold macro F1 (assets/tfidf_baseline/out_of_fold.csv).
        "expected_dev_macro_f1": 0.6629367124205685,
        "expected_test_macro_f1": TFIDF_TEST_MACRO_F1,
    }
    assert received == [(dev_ids, test_ids, shape, tfidf_params(), expected)] * 2
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.usefixtures("embedded", "probe_tuned")
@pytest.mark.parametrize("name", [PROBE, LATE_FUSION, EARLY_FUSION])
def test_tune_runs_the_grid_study_on_the_cached_dev_embeddings(
    name, common, tmp_path, monkeypatch, capsys
):
    received = []

    def run_study(*rows, **options):
        received.append((rows, options))
        study = optuna.create_study(direction="maximize")
        index = optuna.distributions.CategoricalDistribution([0, 1])
        study.add_trial(
            optuna.trial.create_trial(params={"index": 0}, distributions={"index": index}, value=1)
        )
        return study

    monkeypatch.setattr(cli, "run_study", run_study)
    capsys.readouterr()

    assert cli.main(["tune", "--method", name, *common]) == 0

    assert len(received) == 1
    (called, frame, labels, folds), options = received[0]
    dev = split_part(common, "dev")
    journal = tmp_path / "experiments" / name / JOURNAL_FILE
    assert called == name
    assert list(frame.index) == list(dev.index)  # the dev problems only, in order
    expected = focus_labels(raw_to_problems(read_samples(common[1]))).loc[dev.index]
    assert labels.equals(expected)
    assert folds.tolist() == dev["fold"].tolist()
    assert options == {
        "width": TINY_WIDTH,
        "probe_params": PROBE_PARAMS if name == LATE_FUSION else None,  # late fusion fuses it
        "tfidf": None if name == PROBE else tfidf_params(),  # the fusion methods use TF-IDF
        "journal": journal,
    }
    assert capsys.readouterr().out == (
        f"{name}: 1 of {len(cli.GRIDS[name])} configurations scored, in {journal}\n"
    )


@pytest.mark.usefixtures("embedded", "probe_tuned")
def test_export_loads_the_study_of_the_journal_and_writes_to_experiments(
    common, tmp_path, monkeypatch, capsys
):
    journal = tmp_path / "experiments" / LATE_FUSION / JOURNAL_FILE
    grid = list(range(len(LATE_FUSION_GRID)))
    tuned = open_study(LATE_FUSION, journal)
    tuned.optimize(lambda trial: trial.suggest_categorical(INDEX_PARAM, grid) / 100, n_trials=3)
    received = []

    def export(name, study, *rows, directory, **inputs):
        received.append((name, len(study.trials), [len(row) for row in rows], inputs))
        return directory

    monkeypatch.setattr(cli, "export", export)
    capsys.readouterr()

    assert cli.main(["export", "--method", LATE_FUSION, *common]) == 0

    inputs = {"width": TINY_WIDTH, "probe_params": PROBE_PARAMS, "tfidf": tfidf_params()}
    assert received == [(LATE_FUSION, 3, [len(split_part(common, "dev"))] * 3, inputs)]
    folder = tmp_path / "experiments" / LATE_FUSION / TUNING_FOLDER
    assert capsys.readouterr().out == (
        f"exported the configuration {LATE_FUSION}'s study chose to {folder}\n"
    )


@pytest.mark.usefixtures("embedded")
def test_tune_of_late_fusion_refuses_before_the_probe_is_exported(common):
    with pytest.raises(FileNotFoundError, match="then export --method f2llm_probe"):
        cli.main(["tune", "--method", LATE_FUSION, *common])


def test_export_refuses_a_method_without_a_study(common):
    with pytest.raises(FileNotFoundError, match="run `tune --method f2llm_probe` first"):
        cli.main(["export", "--method", PROBE, *common])


@pytest.fixture(name="measured")
def measured_fixture(monkeypatch):
    """Replace both gap measures by one fake, and draw 4 problems; return what it was given.

    Each call records the estimator ``build()`` returns, the rows, and the device variable; it
    returns 1e-6 times the number of calls so far.
    """
    measured = []

    def gap(build, *rows):
        measured.append((build(), *rows, os.environ.get(DEVICE_ENV_VAR)))
        return len(measured) * 1e-6

    monkeypatch.setattr(cli, "score_gaps", gap)
    monkeypatch.setattr(cli, "setfit_score_gap", gap)
    monkeypatch.setattr(cli, "LATENCY_PROBLEMS", 4)
    return measured


@pytest.mark.usefixtures("embedded", "probe_tuned", "fusions_tuned")
def test_atol_measures_each_methods_score_gap_and_suggests_score_atol(common, measured, capsys):
    capsys.readouterr()

    assert cli.main(["atol", "--device", "cpu", *common]) == 0

    dev = split_part(common, "dev")
    drawn = list(dev.sample(n=4, random_state=LATENCY_SEED).index.sort_values())
    assert isinstance(measured[1][0], ScoreFusion)  # late fusion, built with the probe's params
    for _, fit_rows, fit_labels, reference, live, device in measured[:3]:
        assert list(fit_rows.index) == [i for i in dev.index if i not in drawn]
        assert fit_labels.dtype == np.int64
        assert list(reference.index) == drawn == list(live.index)
        columns = [column for column in reference.columns if "_emb_" in column]
        np.testing.assert_allclose(live[columns], reference[columns], rtol=0, atol=1e-5)
        assert device is None  # the variable is left unset: a loaded method encodes on CPU
    gaps = {name: (number + 1) * 1e-6 for number, name in enumerate(measured_names())}
    assert capsys.readouterr().out.splitlines() == [
        *(
            f"{name}: the scores move by up to {gap:.3g} from cached to live embeddings"
            for name, gap in gaps.items()
        ),
        f"score_atol {suggest_score_atol(gaps.values()):g}: 10 times the largest gap, rounded "
        "up; write it into _F2LLM_SCORE_ATOL in codeforces_tags/methods/__init__.py",
    ]


def measured_names():
    """Return the methods in the order atol measures them: the tuned ones, then SetFit."""
    return [PROBE, LATE_FUSION, EARLY_FUSION, SETFIT_METHOD]


@pytest.mark.usefixtures("embedded", "probe_tuned", "fusions_tuned")
def test_atol_trains_setfit_briefly_and_scores_live_with_an_adapter_merged_on_cpu(
    common, measured, tiny_pretrained, tiny_config, monkeypatch
):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")  # a loaded method would merge on the GPU

    cli.main(["atol", "--device", "cpu", *common])

    estimator, fit_rows, fit_labels, reference, raw, device = measured[3]
    assert estimator.get_params(deep=False) == {
        "head_params": PROBE_PARAMS,
        "settings": SETFIT | {"pairs_per_tag_positive": SETFIT_SMOKE_PAIRS // 16},
        "device": "cpu",  # --device: where the adapter trains and is first merged
        "pretrained": tiny_pretrained.to_dict(),
        "encoder_config": tiny_config.to_dict(),
        "resume": False,
    }
    assert list(fit_rows.index) == list(measured[0][1].index)  # the tuned methods' rows
    assert fit_labels.shape == (len(fit_rows), len(FOCUS_TAGS))
    assert list(raw.index) == list(reference.index)
    assert list(raw.columns) == list(RAW_INPUT_COLUMNS)  # raw problems, as a loaded method reads
    assert device == "cpu"  # the live side merges its adapter on CPU all the same
    assert os.environ[DEVICE_ENV_VAR] == "cuda"  # and the variable is restored


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.methods.f2llm", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.methods.f2llm.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.methods.f2llm", run_name="__main__")

    assert exit_info.value.code == 0


def test_main_sets_the_cublas_workspace_before_any_cuda_work(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setattr(cli, "fetch", lambda record: Path("weights"))

    cli.main(["fetch"])

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def crash_before_the_predictions(monkeypatch, freeze_args):
    """Run a freeze that stops where it would write the predictions, as a crash would."""

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:  # undoes this patch only, on leaving
        patch.setattr(cli, "freeze_predictions", crash)
        with pytest.raises(KeyboardInterrupt):
            cli.main(freeze_args)


def names(folder):
    """Return the sorted names of a folder's entries."""
    return sorted(path.name for path in folder.iterdir())


def rewrite(directory, file_name, change):
    """Read a frozen CSV indexed by (part, sample_id), change it in place, and write it back."""
    path = directory / file_name
    frame = pd.read_csv(path, index_col=["part", "sample_id"], float_precision="round_trip")
    change(frame)
    frame.to_csv(path)


def frozen_thresholds(directory, common):
    """Return the per-tag thresholds the frozen dev scores give: those of the test rows."""
    scores = load_scores(directory).loc["dev"]
    labels = focus_labels(raw_to_problems(read_samples(common[1]))).loc[scores.index]
    return per_tag_thresholds(scores, labels)


@pytest.mark.usefixtures("embedded")
def test_freeze_saves_the_folder_and_copies_tuning_before_the_predictions(
    freeze_args, assets, probe_tuned, monkeypatch, capsys
):
    directory = assets / PROBE
    seen = []
    write = cli.freeze_predictions

    def freeze_predictions(*args, **kwargs):
        seen.append(sorted(path.name for path in directory.iterdir()))
        return write(*args, **kwargs)

    monkeypatch.setattr(cli, "freeze_predictions", freeze_predictions)
    capsys.readouterr()

    assert cli.main(freeze_args) == 0

    assert seen == [[MODEL_FOLDER, TUNING_FOLDER]]  # the folder and tuning/, no prediction yet
    frozen = [MODEL_FOLDER, PREDICTIONS_FILE, RUN_RECORD_FILE, SCORES_FILE, TUNING_FOLDER]
    assert names(directory) == sorted(frozen)
    folder = [ENCODER_FILE, METHOD_FILE, MODEL_FILE, PRETRAINED_FILE, THRESHOLDS_FILE]
    assert names(directory / MODEL_FOLDER) == sorted(folder)
    assert names(directory / TUNING_FOLDER) == sorted([OUT_OF_FOLD_FILE, PARAMS_FILE, TRIALS_FILE])
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith(f"{PROBE}: the refit scores dev out of fold at macro F1 ")
    assert printed[1].startswith(f"saved the method folder {directory / MODEL_FOLDER}: it scores")
    assert printed[2:] == [
        f"copied {probe_tuned} to {directory / TUNING_FOLDER}",
        f"saved {len(load_predictions(directory)):,} predictions and scores to {directory}",
        "logged f2llm_probe to MLflow: model model-id",
    ]


@pytest.mark.usefixtures("embedded", "probe_tuned")
def test_freeze_checks_its_folder_on_the_cpu_when_torch_sees_a_gpu(freeze_args, monkeypatch):
    # The folder check is compared with the refit within score_atol, measured on the CPU.
    # The device variable is unset (calls): a loaded method would take cuda by itself.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    devices = []
    load = cli.load_method

    def load_method(folder, **registry):
        devices.append(loaded_device())
        assert devices[-1] == "cpu", "the folder check would encode on the GPU"
        return load(folder, **registry)

    monkeypatch.setattr(cli, "load_method", load_method)

    assert cli.main(freeze_args) == 0

    assert devices == ["cpu"]
    assert DEVICE_ENV_VAR not in os.environ  # restored once the folder has scored


def test_freeze_predicts_dev_out_of_fold_and_test_once(frozen, common):
    check_frozen_method(PROBE, assets_dir=frozen.parent, split=load_split(common[3]))
    predictions, scores = load_predictions(frozen), load_scores(frozen)

    assert list(predictions.columns) == list(FOCUS_TAGS)
    assert list(predictions.loc["dev"].index) == list(split_part(common, "dev").index)
    assert list(predictions.loc["test"].index) == list(split_part(common, "test").index)
    # The test rows apply the thresholds chosen on every out-of-fold dev score.
    expected = (scores.loc["test"] >= frozen_thresholds(frozen, common)).astype(int)
    assert predictions.loc["test"].equals(expected)


def test_freeze_records_the_recipe_and_the_method_entries(frozen, common, tiny_config):
    record = load_run_record(frozen)

    assert record.method == PROBE
    assert record.split_sha256 == provenance.file_sha256(common[3])  # --split, not SPLIT_FILE
    assert record.recipe_sha256 == cli.recipe_sha256(PROBE, PROBE_PARAMS)
    assert record.allow_dirty is True
    assert record.method_entries == {
        "settings": json.loads(json.dumps(SETTINGS)),  # as JSON stores it
        "params": PROBE_PARAMS,
        "dev_in_sample": False,
        "supersedes": [],
        "device": "cpu",
        "dtype": "float32",
        "batch_tokens": tiny_config.cpu_batch_tokens,
        "threads": cli.CPU_THREADS,
    }


def test_supersedes_names_the_committed_f2llm_methods_and_no_method_of_another_package(
    assets, monkeypatch
):
    committed = {LATE_FUSION, "label_prior"}  # label_prior is committed, but of another package
    checked = []

    def committed_unchanged(path):
        checked.append(path)
        return path.parent.name in committed

    monkeypatch.setattr(cli, "committed_unchanged", committed_unchanged)

    assert cli.supersedes(assets, PROBE) == [LATE_FUSION]
    siblings = (EARLY_FUSION, LATE_FUSION, SETFIT_METHOD)
    assert sorted(checked) == sorted(assets / name / PREDICTIONS_FILE for name in siblings)
    assert cli.supersedes(assets, LATE_FUSION) == []


@pytest.mark.parametrize("name", [PROBE, LATE_FUSION, EARLY_FUSION, SETFIT_METHOD])
def test_the_recipe_hashes_the_name_the_settings_the_params_and_the_weights(
    name, tiny_pretrained, monkeypatch
):
    params = {"probe": PROBE_PARAMS}
    recipe = {
        "pretrained": tiny_pretrained.to_dict(),
        "params": params,
        "settings": SETTINGS,
        "tags": list(FOCUS_TAGS),
        "method": name,
    }
    if name in (LATE_FUSION, EARLY_FUSION):  # the fusion methods use the frozen TF-IDF's params
        recipe["tfidf"] = tfidf_params().to_dict()

    assert cli.recipe_sha256(name, params) == provenance.sha256_json(recipe)
    assert cli.recipe_sha256(name, {"probe": PROBE_GRID[0]}) != cli.recipe_sha256(name, params)
    monkeypatch.setattr(cli, "SETTINGS", SETTINGS | {"max_length": 512})
    assert cli.recipe_sha256(name, params) != provenance.sha256_json(recipe)


def test_freeze_twice_only_checks_the_frozen_files_and_logs_them_again(
    freeze_args, frozen, calls, capsys
):
    files = snapshot(frozen.parent)
    capsys.readouterr()

    assert cli.main(freeze_args) == 0

    assert snapshot(frozen.parent) == files
    assert capsys.readouterr().out.splitlines() == [
        f"{frozen} is frozen: its files are complete and hold the recipe of the code "
        f"(score_atol {SCORE_ATOL:g}, so nothing is re-derived)",
        "logged f2llm_probe to MLflow: model model-id",
    ]
    first, second = calls["log_method"]
    assert first[1] == second[1] == load_run_record(frozen)  # the frozen record, both times
    assert first[2] == second[2]


def test_freeze_refuses_frozen_files_of_another_recipe(freeze_args, frozen, calls, monkeypatch):
    files = snapshot(frozen.parent)
    monkeypatch.setattr(cli, "SETTINGS", SETTINGS | {"max_length": 512})

    with pytest.raises(ValueError, match="frozen with another recipe"):
        cli.main(freeze_args)

    assert snapshot(frozen.parent) == files
    assert len(calls["log_method"]) == 1


def test_freeze_refuses_predictions_without_their_method_folder(freeze_args, frozen, calls):
    shutil.rmtree(frozen / MODEL_FOLDER)

    with pytest.raises(ValueError, match="no method folder model/") as error:
        cli.main(freeze_args)

    for name in (PREDICTIONS_FILE, SCORES_FILE, RUN_RECORD_FILE):
        assert name in str(error.value)  # the files to delete if they were never committed
    assert len(calls["log_method"]) == 1


@pytest.mark.usefixtures("embedded")
def test_freeze_after_a_crash_keeps_the_saved_folder_and_writes_the_predictions(
    freeze_args, assets, probe_tuned, monkeypatch, capsys
):
    crash_before_the_predictions(monkeypatch, freeze_args)
    folder = snapshot(assets / PROBE / MODEL_FOLDER)
    capsys.readouterr()

    assert cli.main(freeze_args) == 0

    assert snapshot(assets / PROBE / MODEL_FOLDER) == folder
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].endswith(f", as {probe_tuned / OUT_OF_FOLD_FILE} records")  # checked again
    assert printed[1].startswith(f"the method folder {assets / PROBE / MODEL_FOLDER}: it scores")
    assert printed[2] == f"{assets / PROBE / TUNING_FOLDER} holds the results of {probe_tuned}"
    assert printed[3].startswith("saved ")


@pytest.mark.usefixtures("probe_tuned")
def test_freeze_refuses_a_saved_folder_from_another_fit(
    freeze_args, assets, embedded, write_tuning, monkeypatch
):
    crash_before_the_predictions(monkeypatch, freeze_args)
    paths = sorted(embedded.glob("*.npy"))
    vectors = [np.load(path) for path in paths]
    for path, vector in zip(paths, vectors[1:] + vectors[:1], strict=True):
        np.save(path, vector)  # each text now has another's embedding: the refit changes
    write_tuning(PROBE, PROBE_PARAMS)  # exported again: only the saved folder is of the old fit

    with pytest.raises(ValueError, match="the folder comes from another fit"):
        cli.main(freeze_args)

    assert not (assets / PROBE / PREDICTIONS_FILE).exists()


@pytest.mark.usefixtures("embedded")
def test_freeze_refuses_a_tuning_copy_that_the_study_no_longer_holds(
    freeze_args, assets, probe_tuned, monkeypatch
):
    crash_before_the_predictions(monkeypatch, freeze_args)
    (probe_tuned / TRIALS_FILE).write_text("number,value\n0,0.6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="the study was exported again"):
        cli.main(freeze_args)

    assert not (assets / PROBE / PREDICTIONS_FILE).exists()


def test_freeze_refuses_a_method_registered_as_bit_reproducible(freeze_args, monkeypatch, calls):
    exact = MethodSpec(module=METHODS[PROBE].module, kind="score", requirements=())
    monkeypatch.setattr(cli, "METHODS", cli.METHODS | {PROBE: exact})

    with pytest.raises(ValueError, match="registered with score_atol 0.0"):
        cli.main(freeze_args)

    assert calls["check_dataset"] == []


@pytest.mark.usefixtures("embedded", "probe_tuned", "fusions_tuned")  # tuned, not frozen
def test_freeze_of_late_fusion_needs_the_probes_frozen_configuration(freeze_args):
    argv = [LATE_FUSION if argument == PROBE else argument for argument in freeze_args]

    with pytest.raises(FileNotFoundError, match=f"{PROBE}/{TUNING_FOLDER}/{PARAMS_FILE} does not"):
        cli.main(argv)


class Built(Exception):
    """Stops a freeze once it has built its estimator."""


def stop_at_the_fit(estimator, *, refit):
    """Stand in for CrossFittedTagThresholds: stop, with its arguments and the BLAS threads."""
    blas = {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}
    raise Built(estimator, refit, blas)


@pytest.mark.usefixtures("frozen", "fusions_tuned")
@pytest.mark.parametrize(
    "case",
    [
        (LATE_FUSION, ScoreFusion, {"weight": LATE_FUSION_GRID[0]["weight"]}),
        (EARLY_FUSION, Pipeline, {"memory": None}),  # never a joblib cache
        # The probe's head; a rerun of the same fit reloads its adapter from experiments/.
        (SETFIT_METHOD, SetFitEstimator, {"head_params": PROBE_PARAMS, "settings": SETFIT}),
        (SETFIT_METHOD, SetFitEstimator, {"device": "cpu", "resume": True}),
    ],
)
def test_freeze_builds_each_methods_estimator_from_its_configuration(
    case, freeze_args, monkeypatch
):
    name, kind, expected = case
    monkeypatch.setattr(cli, "CrossFittedTagThresholds", stop_at_the_fit)

    with pytest.raises(Built) as built:
        cli.main([name if argument == PROBE else argument for argument in freeze_args])

    estimator, refit, blas = built.value.args
    assert isinstance(estimator, kind)
    assert refit is True
    assert blas == {TUNING_BLAS_THREADS}  # every fit runs on the tuning's BLAS threads
    assert estimator.get_params(deep=False).items() >= expected.items()


def test_freeze_mirrors_the_method_to_mlflow_once(freeze_args, frozen, store, monkeypatch):
    monkeypatch.setattr(cli, "log_method", log_method)

    cli.main(freeze_args)
    cli.main(freeze_args)

    record = load_run_record(frozen)
    assert method_run_id(record.recipe_sha256, **store) is not None
    [model] = mlflow.search_logged_models([use_store(**store)], output_format="list")
    metrics = mlflow.get_logged_model(model.model_id).metrics
    assert {metric.dataset_name for metric in metrics} == {"dev-oof", "test"}
    assert len({(metric.dataset_name, metric.key) for metric in metrics}) == len(metrics)


def test_verify_re_derives_the_frozen_rows_and_re_predicts_the_test_rows_live(
    verify_args, frozen, common, calls, capsys
):
    capsys.readouterr()

    assert cli.main(verify_args) == 0

    printed = capsys.readouterr().out.splitlines()
    assert printed[:2] == [
        f"{PROBE}: the frozen files are complete and hold the recipe of the code",
        f"{PROBE}: the dev rows and the thresholds re-derive from {SCORES_FILE}, and the test rows "
        "apply them",
    ]
    assert printed[2].startswith(
        f"{PROBE} test, re-predicted live by {frozen / MODEL_FOLDER}: the scores are within "
        f"{SCORE_ATOL:g} of the frozen ones"
    )
    assert printed[2].endswith("0 tags flip, each within score_atol of its threshold")
    assert [line.split(":")[0] for line in printed[3:]] == [f"{PROBE} dev-oof", f"{PROBE} test"]
    assert calls["check_dataset"] == [common[1]] * 3  # embed, freeze, then verify
    assert DEVICE_ENV_VAR not in os.environ  # set while the method loaded, then removed


def test_verify_refuses_a_method_that_is_not_frozen(verify_args):
    with pytest.raises(ConformanceError, match="holds no method.json"):
        cli.main(verify_args)


@pytest.mark.parametrize("part", ["dev", "test"])
def test_verify_refuses_predictions_that_the_frozen_scores_do_not_give(
    part, frozen, verify_args, common
):
    sample_id = split_part(common, part).index[0]

    def flip(frame):
        frame.loc[(part, sample_id), "math"] = 1 - frame.loc[(part, sample_id), "math"]

    rewrite(frozen, PREDICTIONS_FILE, flip)

    with pytest.raises(ValueError, match=f"The {part} rows of f2llm_probe's predictions.csv"):
        cli.main(verify_args)


@pytest.fixture(name="live")
def live_fixture(monkeypatch):
    """Replace the loaded method folder by one whose live scores the test sets; return them.

    ``live["scores"]`` and ``live["thresholds"]`` are set by the test; ``live["seen"]`` gets the
    columns each call was given and the device the method was loaded for, and ``live["rows"]``
    the sample ids of each call. The device variable is ``cuda`` before verify runs.
    """
    live = {"scores": None, "thresholds": None, "seen": [], "rows": []}

    class Loaded:  # pylint: disable=too-few-public-methods
        """Stands in for a loaded method folder."""

        def __init__(self):
            self.thresholds = live["thresholds"]

        def scores(self, problems):
            """Return the scores the test set, for the problems given."""
            live["seen"].append((list(problems.columns), os.environ[DEVICE_ENV_VAR]))
            live["rows"].append(list(problems.index))
            return live["scores"].loc[problems.index]

    monkeypatch.setattr(cli, "load_method", lambda _folder, **_registry: Loaded())
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")  # verify loads for --device, then restores it
    return live


def on_the_threshold(frozen, common, live):
    """Move one frozen test score just above its tag's threshold; ``live`` scores as frozen.

    Returns:
        Where: the sample id, the tag and the frozen score, which the frozen rows predict.
    """
    thresholds = frozen_thresholds(frozen, common)
    tag = next(tag for tag in FOCUS_TAGS if np.isfinite(thresholds[tag]))
    row = ("test", split_part(common, "test").index[0])
    score = thresholds[tag] + SCORE_ATOL / 4

    def put_on_the_threshold(frame):
        frame.loc[row, tag] = score

    def predict_the_tag(frame):
        frame.loc[row, tag] = 1

    rewrite(frozen, SCORES_FILE, put_on_the_threshold)
    rewrite(frozen, PREDICTIONS_FILE, predict_the_tag)
    live["scores"] = load_scores(frozen).loc["test"].copy()
    live["thresholds"] = thresholds.to_numpy()
    return row[1], tag, score


def test_verify_counts_a_test_flip_within_score_atol_of_its_threshold(
    frozen, common, verify_args, live, capsys
):
    sample_id, tag, score = on_the_threshold(frozen, common, live)
    live["scores"].loc[sample_id, tag] = score - SCORE_ATOL / 2  # below the threshold now
    capsys.readouterr()

    assert cli.main(verify_args) == 0

    assert "1 tags flip, each within score_atol of its threshold" in capsys.readouterr().out
    assert live["seen"] == [(list(RAW_INPUT_COLUMNS), "cpu")]  # input columns, loaded for CPU
    assert os.environ[DEVICE_ENV_VAR] == "cuda"  # restored once the method has scored


@pytest.fixture(name="gpu_visible")
def gpu_visible_fixture(monkeypatch):
    """Make torch see a GPU, which the test machine may not have; nothing runs on it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)


@pytest.mark.usefixtures("gpu_visible")
@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_verify_on_the_gpu_scores_each_test_problem_in_its_own_call(
    frozen, common, verify_args, live, device
):
    # Never one batch of every test problem on the GPU: its sustained full power rebooted the
    # machine once. Without --single, the GPU still gets one problem per call.
    on_the_threshold(frozen, common, live)

    assert cli.main([*verify_args, "--device", device]) == 0

    test_ids = list(load_scores(frozen).loc["test"].index)
    assert live["rows"] == [[sample_id] for sample_id in test_ids]
    assert live["seen"] == [(list(RAW_INPUT_COLUMNS), "cuda")] * len(test_ids)


@pytest.mark.usefixtures("gpu_visible")
def test_verify_stays_on_the_cpu_by_default_when_torch_sees_a_gpu(
    frozen, common, verify_args, live
):
    on_the_threshold(frozen, common, live)

    assert cli.main(verify_args) == 0

    assert live["seen"] == [(list(RAW_INPUT_COLUMNS), "cpu")]  # one call, on the CPU


def test_verify_single_scores_each_test_problem_in_its_own_call(
    frozen, common, verify_args, live, capsys
):
    # On the GPU, one problem per call, as the command line predicts one exercise: never the
    # sustained full power of a batch of every test problem.
    sample_id, tag, score = on_the_threshold(frozen, common, live)
    live["scores"].loc[sample_id, tag] = score - SCORE_ATOL / 2  # below the threshold now
    capsys.readouterr()

    assert cli.main([*verify_args, "--single"]) == 0

    test_ids = list(load_scores(frozen).loc["test"].index)
    assert live["rows"] == [[sample_id] for sample_id in test_ids]  # in order, one each
    assert len(live["seen"]) == len(test_ids)
    printed = capsys.readouterr().out
    assert f"{PROBE} test, re-predicted live by {frozen / MODEL_FOLDER}, one problem per call" in (
        printed
    )
    assert "1 tags flip, each within score_atol of its threshold" in printed


def test_verify_single_refuses_live_scores_beyond_score_atol(frozen, common, verify_args, live):
    sample_id, tag, score = on_the_threshold(frozen, common, live)
    live["scores"].loc[sample_id, tag] = score + 2 * SCORE_ATOL

    with pytest.raises(ValueError, match="one problem per call: the scores differ from the frozen"):
        cli.main([*verify_args, "--single"])


def test_verify_refuses_live_scores_beyond_score_atol(frozen, common, verify_args, live):
    sample_id, tag, score = on_the_threshold(frozen, common, live)
    live["scores"].loc[sample_id, tag] = score + 2 * SCORE_ATOL

    with pytest.raises(ValueError, match="the scores differ from the frozen scores.csv"):
        cli.main(verify_args)


def test_verify_refuses_a_folder_whose_thresholds_are_not_the_frozen_ones(
    frozen, common, verify_args, live
):
    on_the_threshold(frozen, common, live)
    live["thresholds"] = live["thresholds"] + SCORE_ATOL / 10

    with pytest.raises(ValueError, match="holds other thresholds than the frozen dev scores give"):
        cli.main(verify_args)


def test_verify_retrain_refits_from_the_cache_and_writes_nothing(
    verify_args, frozen, calls, tmp_path, capsys
):
    files = snapshot(frozen.parent)
    shutil.rmtree(tmp_path / "experiments")  # gitignored: verify reads the frozen tuning/ copies
    capsys.readouterr()

    assert cli.main([*verify_args, "--retrain"]) == 0

    printed = capsys.readouterr().out.splitlines()
    for part in ("dev", "test"):
        [line] = [line for line in printed if line.startswith(f"{PROBE} {part}, refit")]
        assert line.endswith("0 tags flip, each within score_atol of its threshold")
    assert snapshot(frozen.parent) == files
    assert len(calls["log_method"]) == 1  # freeze's only


@pytest.mark.usefixtures("frozen")
def test_verify_retrain_refuses_a_refit_whose_scores_moved(
    verify_args, common, embedded, tiny_pretrained, tiny_config
):
    first = read_samples(common[1]).loc[split_part(common, "dev").index[:1]]
    statement = encoder_inputs(raw_to_problems(first), tiny_config)[STATEMENT][0]
    path = embedded / f"{text_key(statement, tiny_pretrained, tiny_config)}.npy"
    np.save(path, -np.load(path))  # one dev statement's cached embedding changes

    with pytest.raises(ValueError, match="dev, refit from the cache: the scores differ"):
        cli.main([*verify_args, "--retrain"])


def test_verify_retrain_refuses_a_dev_flip_far_from_the_frozen_fold_threshold(
    frozen, common, verify_args, monkeypatch
):
    scores = load_scores(frozen).loc["dev"]
    folds = split_part(common, "dev")["fold"]
    labels = focus_labels(raw_to_problems(read_samples(common[1]))).loc[scores.index]
    first = cross_fitted_thresholds(scores, labels, folds).loc[folds.iloc[0]]
    column = int(np.argmax((np.abs(scores.iloc[0] - first) > SCORE_ATOL).to_numpy()))

    class FlippingFit(CrossFittedTagThresholds):
        """Flips the first dev problem's out-of-fold prediction of one tag after fitting."""

        def fit(self, X, y, *, groups):
            """Fit, then flip one tag far from its fold threshold."""
            super().fit(X, y, groups=groups)
            self.oof_predictions_ = self.oof_predictions_.copy()
            self.oof_predictions_[0, column] = 1 - self.oof_predictions_[0, column]
            return self

    monkeypatch.setattr(cli, "CrossFittedTagThresholds", FlippingFit)

    with pytest.raises(ValueError, match="1 tags flip where the frozen score lies beyond"):
        cli.main([*verify_args, "--retrain"])
