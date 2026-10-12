"""Tests for the ModernBERT methods' commands, on a tiny dataset and tiny models, on the CPU.

The fold models that ``adopt`` copies are written as the former notebooks 05 and 06 saved theirs:
random tiny models, with the exact scores of the fold each leaves out, so that ``freeze`` and
``verify`` check them as they check the real ones.
"""

# A test takes one argument per fixture it uses.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import itertools
import json
import os
import runpy
import subprocess
import sys
from dataclasses import asdict, replace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from codeforces_tags.config import (
    EXPERIMENTS_DIR,
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    SPLIT_FILE,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    SCORES_FILE,
    load_predictions,
    load_run_record,
    load_scores,
)
from codeforces_tags.evaluation.timing import LATENCY_FILE, load_latency
from codeforces_tags.methods import METHODS, MODEL_FOLDER, MethodSpec
from codeforces_tags.methods.modernbert import __main__ as cli
from codeforces_tags.methods.modernbert import tuning
from codeforces_tags.methods.modernbert.folds import NOTEBOOK_WEIGHTS, fold_name, read_dev
from codeforces_tags.methods.modernbert.inputs import take
from codeforces_tags.methods.modernbert.method import FineTunedBag
from codeforces_tags.methods.modernbert.modeling import load_tokenizer, predict_scores
from codeforces_tags.methods.modernbert.settings import (
    ANALYSIS_FOLDER,
    BAG_FILE,
    CONFIG_FILES,
    CPU_LATENCY_FOLDER,
    ENCODER_FILE,
    FOLD_TEST_SCORES_FILE,
    FULL,
    HELD_OUT_FILE,
    LORA,
    PRETRAINED_FILE,
    TRAINED_FILE,
    TRAINING_FILE,
    TUNING_FOLDER,
    VARIANTS,
    study_recipe,
)
from codeforces_tags.methods.modernbert.training import build_training_model, fold_state
from codeforces_tags.methods.modernbert.weights import snapshot_folder, trained_root
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import log_method, log_run_extras, method_run_id

# The tuning fold of four problems lacks most tags, whose average precision then has no positive.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found in y_true:UserWarning")
# Four dev problems per fold, then four test problems; problem i has focus tag i mod 8.
DEV_PROBLEMS = 20
TEST_PROBLEMS = 4


def problem_tag(sample_id):
    """Return the focus tag of a synthetic problem."""
    return FOCUS_TAGS[sample_id % len(FOCUS_TAGS)]


def perturbed_model(name, variant, params, seed):
    """Build a fold model as training would, its trainable weights moved at random."""
    torch.manual_seed(seed)
    model = build_training_model(
        variant, params[name], snapshot_folder(variant.model_repo, variant.model_revision)
    )
    with torch.no_grad():
        for weight in model.parameters():
            if weight.requires_grad:
                weight.add_(0.5 * torch.randn_like(weight))
    return model.eval()


def snapshot(directory):
    """Map each file under ``directory`` to its bytes."""
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record, statements):
    """Write the problems as sample files, and return the dataset folder."""
    records = {
        sid: make_raw_record(
            prob_desc_description=statements[problem_tag(sid)],
            source_code=f"n = int ( input ( ) )\nprint ( n + {sid} )",
            tags=[problem_tag(sid)],
        )
        for sid in range(DEV_PROBLEMS + TEST_PROBLEMS)
    }
    return write_samples(records)


@pytest.fixture(name="split_file")
def split_file_fixture(tmp_path):
    """Write the frozen split, four dev problems per fold, and return its path."""
    ids = range(DEV_PROBLEMS + TEST_PROBLEMS)
    split = pd.DataFrame(
        {
            "part": ["dev" if sid < DEV_PROBLEMS else "test" for sid in ids],
            "fold": [sid % 5 if sid < DEV_PROBLEMS else -1 for sid in ids],
            "group": list(ids),
        },
        index=pd.Index(list(ids), name="sample_id"),
    )
    path = tmp_path / "split.csv"
    split.to_csv(path)
    return path


@pytest.fixture(name="assets")
def assets_fixture(tmp_path, tiny_variants, params):
    """Return the folder of the frozen methods, holding each method's exported tuning choice."""
    assets = tmp_path / "assets"
    for name, variant in tiny_variants.items():
        folder = assets / name / TUNING_FOLDER
        folder.mkdir(parents=True)
        chosen = {
            "params": asdict(params[name]),
            "trial": 3,
            "tuning_macro_f1": 0.5,
            "recipe": study_recipe(variant),
        }
        (folder / "params.json").write_text(json.dumps(chosen), encoding="utf-8")
    return assets


@pytest.fixture(name="common")
def common_fixture(dataset, split_file, assets):
    """Return the options every command takes."""
    return ["--data", str(dataset), "--split", str(split_file), "--assets", str(assets)]


@pytest.fixture(name="store_options")
def store_options_fixture(store):
    """Return freeze's MLflow options."""
    return ["--tracking-uri", store["tracking_uri"], "--artifact-dir", str(store["artifact_dir"])]


@pytest.fixture(name="calls", autouse=True)
def calls_fixture(monkeypatch):
    """Replace the dataset check, which only the challenge dataset passes, and the MLflow logs.

    Returns the arguments each was called with. The MLflow test puts the real functions back.
    """
    calls = {"check_dataset": [], "log_method": [], "log_run_extras": []}

    def log(folder, record, metrics, *, split, **store):
        calls["log_method"].append((folder, record, metrics, split, store))
        return "model-id"

    def extras(recipe, *, metrics, **store):
        calls["log_run_extras"].append((recipe, metrics, store))

    monkeypatch.setattr(cli, "check_dataset", calls["check_dataset"].append)
    monkeypatch.setattr(cli, "log_method", log)
    monkeypatch.setattr(cli, "log_run_extras", extras)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return calls


@pytest.fixture(name="notebook_run")
def notebook_run_fixture(caches, tiny_variants, params, dataset, split_file, tmp_path):
    """Return a writer of a notebook's fold models, as ``adopt`` reads them.

    ``notebook_run(name)`` writes ``fold_k/`` for each dev fold under a new folder, with random
    weights, the exact scores of the fold each leaves out, and the study's recipe and the chosen
    parameters in ``training.json``, and returns that folder.
    """
    del caches  # the snapshot it builds from

    def write(name):
        variant = tiny_variants[name]
        tokenizer = load_tokenizer(snapshot_folder(variant.model_repo, variant.model_revision))
        raw_dev = select_part(read_samples(dataset), load_split(split_file), "dev")
        dev = read_dev(raw_dev, tokenizer, variant)
        source = tmp_path / "notebook" / name
        for fold in range(5):
            model = perturbed_model(name, variant, params, seed=fold)
            folder = source / fold_name(fold)
            folder.mkdir(parents=True)
            save_file(fold_state(model, variant), str(folder / NOTEBOOK_WEIGHTS[variant.lora]))
            held_out = predict_scores(
                model,
                take(dev.input_ids, (dev.folds == fold).to_numpy()),
                tokenizer,
                batch_size=variant.score_batch,
                pad_to_multiple_of=variant.pad_to_multiple_of,
            )
            np.save(folder / HELD_OUT_FILE, held_out)
            record = {
                "recipe": study_recipe(variant),
                "params": asdict(params[name]),
                "seed": variant.seed + (fold == 1),  # fold 1 collapsed once
                "train_macro_ap": 0.5,
                "minutes": 0.1,
            }
            (folder / TRAINING_FILE).write_text(json.dumps(record), encoding="utf-8")
        return source

    return write


@pytest.fixture(name="adopted")
def adopted_fixture(notebook_run, common):
    """Adopt a notebook's fold models of each method into the trained cache."""
    for name in (LORA, FULL):
        cli.main(["adopt", "--method", name, "--source", str(notebook_run(name)), *common])


@pytest.fixture(name="frozen")
def frozen_fixture(adopted, common, store_options, assets, capsys):
    """Freeze modernbert_lora from its adopted fold models, and return its frozen directory."""
    del adopted
    cli.main(["freeze", "--method", LORA, "--allow-dirty", *common, *store_options])
    capsys.readouterr()
    return assets / LORA


def test_parse_args_defaults_to_the_committed_files_and_the_experiments_store():
    freeze = cli.parse_args(["freeze", "--method", LORA])

    assert vars(freeze) == {
        "command": "freeze",
        "allow_dirty": False,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "artifact_dir": MLFLOW_ARTIFACT_DIR,
        "method": LORA,
        "data": None,
        "split": SPLIT_FILE,
        "assets": METHODS_ASSETS_DIR,
    }
    assert cli.parse_args(["tune", "--method", FULL]).journal is None
    assert cli.parse_args(["train", "--method", LORA]).root is None
    assert cli.parse_args(["adopt", "--method", LORA]).source is None
    assert cli.parse_args(["verify", "--method", LORA]).retrain is False


def test_every_command_needs_a_registered_method(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["verify", "--method", "tfidf"])

    assert "invalid choice: 'tfidf'" in capsys.readouterr().err


def test_fetch_downloads_the_pinned_files_into_the_pretrained_cache(caches, monkeypatch, capsys):
    calls = []

    def download(repo_id, *, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision, local_dir, allow_patterns))

    monkeypatch.setattr(cli, "snapshot_download", download)

    assert cli.main(["fetch", "--method", LORA]) == 0

    variant = VARIANTS[LORA]
    folder = caches["pretrained"] / "answerdotai--ModernBERT-large" / variant.model_revision
    assert calls == [
        (variant.model_repo, variant.model_revision, folder, [*CONFIG_FILES, ENCODER_FILE])
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"fetched {variant.model_repo}@{variant.model_revision} into {folder}; {LORA} pins "
        f"{sorted([*CONFIG_FILES, ENCODER_FILE])}"
    ]


def test_tune_runs_the_study_then_export_freezes_its_results(
    caches, common, tmp_path, assets, monkeypatch, capsys
):
    del caches
    # The tiny models score 0 on four problems; fANOVA needs trials that differ.
    scores = itertools.count(1)
    monkeypatch.setattr(
        tuning,
        "held_out_metrics",
        lambda *args: {"macro_f1": next(scores) / 100, "macro_ap": 0.5},
    )
    journal = tmp_path / "optuna.jsonl"

    assert cli.main(["tune", "--method", LORA, "--journal", str(journal), *common]) == 0
    assert cli.main(["export", "--method", LORA, "--journal", str(journal), *common]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines[:2]] == ["trial 0", "trial 1"]
    assert lines[2].startswith("best: trial ") and lines[2].endswith("run export next")
    exported = assets / LORA / TUNING_FOLDER
    assert lines[3].startswith("exported trial ") and lines[3].endswith(
        f"to {exported}: params.json, trials.csv, history.csv, importances.csv, out_of_fold.csv"
    )
    chosen = json.loads((exported / "params.json").read_text(encoding="utf-8"))
    assert chosen["recipe"] == study_recipe(VARIANTS[LORA])
    assert len(pd.read_csv(exported / "trials.csv")) == 2


def test_train_trains_one_fold_model_per_dev_fold_and_resumes(
    caches, common, params, tmp_path, capsys
):
    del caches
    root = tmp_path / "trained_elsewhere"
    arguments = ["train", "--method", FULL, "--root", str(root), *common]

    assert cli.main(arguments) == 0
    first = capsys.readouterr().out.splitlines()
    cli.main(arguments)

    assert [line.split(" in ")[0] for line in first] == [
        f"trained {root / fold_name(fold)}" for fold in range(5)
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"{root / fold_name(fold)} holds a complete fold model: kept" for fold in range(5)
    ]
    record = json.loads((root / "fold_2" / TRAINING_FILE).read_text(encoding="utf-8"))
    assert (record["recipe"], record["params"]) == (
        study_recipe(VARIANTS[FULL]),
        asdict(params[FULL]),
    )
    assert np.load(root / "fold_2" / HELD_OUT_FILE).shape == (4, len(FOCUS_TAGS))


def test_train_refuses_the_cache_of_a_frozen_method(caches, common, assets):
    (assets / FULL / MODEL_FOLDER).mkdir(parents=True)
    (assets / FULL / MODEL_FOLDER / TRAINED_FILE).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="is frozen: its fold models are the weights"):
        cli.main(["train", "--method", FULL, *common])

    assert not caches["trained"].exists()


def test_a_command_reading_the_chosen_configuration_needs_the_export(common, assets):
    (assets / FULL / TUNING_FOLDER / "params.json").unlink()

    with pytest.raises(FileNotFoundError, match="run export --method modernbert_base_full first"):
        cli.main(["adopt", "--method", FULL, *common])


def test_adopt_copies_a_notebooks_fold_models_into_the_trained_cache(
    caches, notebook_run, common, capsys
):
    source = notebook_run(LORA)

    assert cli.main(["adopt", "--method", LORA, "--source", str(source), *common]) == 0

    root = caches["trained"] / LORA
    assert capsys.readouterr().out.splitlines() == [
        f"adopted {source / fold_name(fold)} as {root / fold_name(fold)}" for fold in range(5)
    ]
    assert sorted(path.name for path in (root / "fold_0").iterdir()) == [
        HELD_OUT_FILE,
        TRAINING_FILE,
        "weights.safetensors",
    ]


@pytest.mark.parametrize("name", [LORA, FULL])
def test_freeze_saves_the_frozen_files_the_method_folder_and_the_fold_test_scores(
    name, adopted, common, store_options, assets, calls, capsys
):
    del adopted

    assert cli.main(["freeze", "--method", name, "--allow-dirty", *common, *store_options]) == 0

    directory = assets / name
    assert sorted(path.name for path in directory.iterdir()) == [
        ANALYSIS_FOLDER,
        MODEL_FOLDER,
        PREDICTIONS_FILE,
        "run_record.json",
        SCORES_FILE,
        TUNING_FOLDER,
    ]
    assert sorted(path.name for path in (directory / MODEL_FOLDER).iterdir()) == [
        BAG_FILE,
        "method.json",
        PRETRAINED_FILE,
        TRAINED_FILE,
    ]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "each fold model gives back its out-of-fold scores: largest gap 0.00e+00",
        f"saved {DEV_PROBLEMS + TEST_PROBLEMS} predictions and scores to {directory}",
        "saved each fold model's test scores to "
        f"{directory / ANALYSIS_FOLDER / FOLD_TEST_SCORES_FILE}",
        f"logged {name} to MLflow: model model-id",
    ]
    assert calls["check_dataset"] == [common[1]]
    [(folder, record, metrics, _, _)] = calls["log_method"]
    assert (folder, record) == (directory / MODEL_FOLDER, load_run_record(directory))
    assert list(metrics) == ["dev-oof", "test"]
    [(recipe, extras, _)] = calls["log_run_extras"]
    assert recipe == record.recipe_sha256
    assert extras == {
        "outside_tuning_fold/macro_f1": record.method_entries["outside_tuning_fold_macro_f1"]
    }


def test_freeze_derives_dev_from_the_fold_models_and_scores_test_with_the_bag(frozen, split_file):
    scores, predictions = load_scores(frozen), load_predictions(frozen)
    bag = json.loads((frozen / MODEL_FOLDER / BAG_FILE).read_text(encoding="utf-8"))

    # Dev: each fold model's scores of the fold it left out, cast exactly to float64.
    held_out = np.load(trained_root() / LORA / "fold_3" / HELD_OUT_FILE)
    fold_3 = [sid for sid in range(DEV_PROBLEMS) if sid % 5 == 3]
    assert np.array_equal(scores.loc["dev"].loc[fold_3].to_numpy(), held_out.astype(np.float64))
    # Test: the mean of the fold models' scores, thresholded with the bag's thresholds.
    fold_test = pd.read_csv(
        frozen / ANALYSIS_FOLDER / FOLD_TEST_SCORES_FILE, index_col=["fold_model", "sample_id"]
    )
    mean = fold_test.groupby("sample_id").mean().astype(np.float32).astype(np.float64)
    np.testing.assert_allclose(scores.loc["test"].to_numpy(), mean.to_numpy(), atol=1e-7)
    assert predictions.loc["test"].equals(
        (scores.loc["test"] >= np.array(bag["thresholds"])).astype(int)
    )
    check_frozen_method(LORA, assets_dir=frozen.parent, split=load_split(split_file))


def test_freeze_records_the_recipe_the_training_and_the_dev_estimate(frozen):
    record = load_run_record(frozen)
    chosen = json.loads((frozen / TUNING_FOLDER / "params.json").read_text(encoding="utf-8"))
    pretrained = json.loads((frozen / MODEL_FOLDER / PRETRAINED_FILE).read_text(encoding="utf-8"))
    entries = record.method_entries

    assert record.recipe_sha256 == cli.recipe_sha256(VARIANTS[LORA], chosen, pretrained)
    assert record.allow_dirty is True
    assert sorted(entries) == [
        "dev_estimate",
        "dev_in_sample",
        "device",
        "dtype",
        "fold_models",
        "outside_tuning_fold_macro_f1",
        "params",
        "settings",
        "supersedes",
        "tuning",
    ]
    assert (entries["device"], entries["dtype"], entries["supersedes"]) == ("cpu", "float32", [])
    assert entries["tuning"] == {"trial": 3, "tuning_macro_f1": 0.5}
    assert entries["fold_models"]["fold_1"] == {
        "seed": 1,
        "collapsed_seeds": [0],
        "train_macro_ap": 0.5,
        "minutes": 0.1,
        "adopted_from": entries["fold_models"]["fold_1"]["adopted_from"],
    }
    assert entries["fold_models"]["fold_1"]["adopted_from"].endswith(
        "notebook/modernbert_lora/fold_1"
    )
    assert entries["settings"]["max_length"] == VARIANTS[LORA].max_length


def test_freeze_records_the_gpu_it_scored_on(adopted, common, store_options, assets, monkeypatch):
    del adopted
    monkeypatch.setattr(cli, "resolve_device", lambda setting: torch.device("cuda"))
    monkeypatch.setattr(cli.torch.cuda, "get_device_name", lambda device: "Tiny GPU")
    emptied = []
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "empty_cache", lambda: emptied.append(True))

    cli.main(["freeze", "--method", FULL, "--allow-dirty", *common, *store_options])

    entries = load_run_record(assets / FULL).method_entries
    assert (entries["device"], entries["dtype"]) == ("cuda (Tiny GPU)", "fp16-autocast")
    assert emptied == [True]  # the built bag is dropped before the folder is loaded


def test_freeze_twice_rewrites_nothing_and_logs_the_frozen_record(
    frozen, common, store_options, calls, capsys
):
    files = snapshot(frozen)

    assert cli.main(["freeze", "--method", LORA, "--allow-dirty", *common, *store_options]) == 0

    assert snapshot(frozen) == files
    assert capsys.readouterr().out.splitlines() == [
        f"{frozen} is frozen under the code's recipe: nothing re-derived (verify re-scores it)",
        f"logged {LORA} to MLflow: model model-id",
    ]
    assert [record for _, record, _, _, _ in calls["log_method"]] == [load_run_record(frozen)] * 2


def test_freeze_and_verify_refuse_another_recipe(
    frozen, common, store_options, tiny_variants, monkeypatch
):
    monkeypatch.setitem(VARIANTS, LORA, replace(tiny_variants[LORA], score_batch=8))

    for command in (["freeze", "--allow-dirty", *store_options], ["verify"]):
        with pytest.raises(ValueError, match="was frozen with another recipe"):
            cli.main([*command, "--method", LORA, *common])

    del frozen


def test_freeze_refuses_a_fold_model_that_does_not_give_back_its_scores(
    adopted, common, store_options, caches
):
    del adopted
    path = caches["trained"] / LORA / "fold_2" / HELD_OUT_FILE
    np.save(path, np.load(path) + 0.01)

    with pytest.raises(ValueError, match="moves an out-of-fold score by 1.00e-02"):
        cli.main(["freeze", "--method", LORA, "--allow-dirty", *common, *store_options])


def test_freeze_names_the_committed_siblings_it_supersedes(
    adopted, common, store_options, assets, monkeypatch
):
    del adopted
    monkeypatch.setattr(
        cli,
        "METHODS",
        METHODS
        | {
            "other_package": MethodSpec(
                module="codeforces_tags.methods.other.method", kind="score", requirements=()
            )
        },
    )
    monkeypatch.setattr(cli, "committed_unchanged", lambda path: True)

    cli.main(["freeze", "--method", LORA, "--allow-dirty", *common, *store_options])

    assert load_run_record(assets / LORA).method_entries["supersedes"] == [FULL]


def test_freeze_refuses_a_store_that_is_not_sqlite_before_any_work(
    adopted, common, assets, calls, tmp_path, capsys
):
    del adopted  # the fold models a freeze would read
    store = (tmp_path / "mlruns").as_uri()
    before = snapshot(assets)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["freeze", "--method", LORA, "--allow-dirty", *common, "--tracking-uri", store])

    assert exit_info.value.code == 2
    assert f"{store} is not a sqlite:/// store" in capsys.readouterr().err
    assert calls["check_dataset"] == []  # nothing read, scored or written
    assert snapshot(assets) == before


def test_freeze_logs_the_method_and_its_outside_fold_score_to_mlflow_once(
    adopted, common, store_options, store, assets, monkeypatch
):
    del adopted
    monkeypatch.setattr(cli, "log_method", log_method)
    monkeypatch.setattr(cli, "log_run_extras", log_run_extras)
    arguments = ["freeze", "--method", LORA, "--allow-dirty", *common, *store_options]

    cli.main(arguments)
    cli.main(arguments)

    record = load_run_record(assets / LORA)
    run_id = method_run_id(record.recipe_sha256, **store)
    [model] = mlflow.search_logged_models([use_store(**store)], output_format="list")
    assert model.source_run_id == run_id
    metrics = mlflow.get_run(run_id).data.metrics
    assert metrics["outside_tuning_fold/macro_f1"] == pytest.approx(
        record.method_entries["outside_tuning_fold_macro_f1"]
    )


def test_verify_re_derives_the_frozen_files_and_reports_the_scores(frozen, common, calls, capsys):
    files = snapshot(frozen)

    assert cli.main(["verify", "--method", LORA, *common]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        f"the frozen files of {LORA} are complete, consistent, under the code's recipe",
        "re-derived the dev predictions and the thresholds from scores.csv: they match",
        "each fold model gives back its out-of-fold scores: largest gap 0.00e+00",
        f"re-scored {TEST_PROBLEMS} test problems with the bag: largest gap 0.00e+00 "
        "(score_atol 0.01), 0 predictions flipped",
    ]
    assert [line.split(":")[0] for line in lines[4:]] == ["dev-oof", "test"]
    assert snapshot(frozen) == files
    assert len(calls["log_method"]) == 1


@pytest.mark.usefixtures("frozen")
def test_verify_re_scores_on_the_cpu_even_where_torch_sees_a_gpu(common, monkeypatch):
    # verify keeps the GPU off a sustained load (a batched run rebooted the machine once): it
    # re-scores dev and test on the CPU only, within score_atol of the frozen GPU batches.
    devices, load = [], cli.load

    def recording_load(folder, *, device=None):
        devices.append(device)
        return load(folder, device=device)

    monkeypatch.setattr(cli, "load", recording_load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert cli.main(["verify", "--method", LORA, *common]) == 0
    assert devices == ["cpu"]


def test_verify_refuses_dev_predictions_that_are_not_the_cross_fitted_ones(frozen, common):
    predictions = pd.read_csv(frozen / PREDICTIONS_FILE, index_col=["part", "sample_id"])
    predictions.loc[("dev", 0), "math"] = 1 - predictions.loc[("dev", 0), "math"]
    predictions.to_csv(frozen / PREDICTIONS_FILE)

    with pytest.raises(ValueError, match="The dev predictions re-derived from the frozen scores"):
        cli.main(["verify", "--method", LORA, *common])


def test_verify_refuses_a_method_folder_with_other_thresholds(frozen, common):
    path = frozen / MODEL_FOLDER / BAG_FILE
    bag = json.loads(path.read_text(encoding="utf-8"))
    bag["thresholds"][0] += 1e-3
    path.write_text(json.dumps(bag), encoding="utf-8")

    with pytest.raises(ValueError, match="thresholds differ from those of the frozen scores"):
        cli.main(["verify", "--method", LORA, *common])


def test_verify_refuses_test_predictions_that_are_not_the_scores_thresholded(frozen, common):
    predictions = pd.read_csv(frozen / PREDICTIONS_FILE, index_col=["part", "sample_id"])
    test_id = DEV_PROBLEMS
    predictions.loc[("test", test_id), "math"] = 1 - predictions.loc[("test", test_id), "math"]
    predictions.to_csv(frozen / PREDICTIONS_FILE)

    with pytest.raises(ValueError, match="are not the frozen test scores thresholded"):
        cli.main(["verify", "--method", LORA, *common])


def test_verify_refuses_a_bag_that_moves_a_test_score_beyond_score_atol(
    frozen, common, monkeypatch
):
    scores = FineTunedBag.scores
    monkeypatch.setattr(FineTunedBag, "scores", lambda self, raw: scores(self, raw) + 0.01)

    with pytest.raises(ValueError, match="Re-scoring test moves a score by 1.00e-02"):
        cli.main(["verify", "--method", LORA, *common])

    del frozen


def test_verify_retrain_trains_each_fold_from_its_frozen_seed_and_checks_the_bounds(
    frozen, common, tiny_variants, monkeypatch, capsys
):
    lenient = replace(
        tiny_variants[LORA],
        retrain_min_correlation=-1.0,
        retrain_max_median_gap=1.0,
        retrain_max_flips=1.0,
        retrain_f1_margin=1.0,
    )
    monkeypatch.setitem(VARIANTS, LORA, lenient)
    monkeypatch.setattr(cli, "recipe_sha256", lambda *args: load_run_record(frozen).recipe_sha256)

    assert cli.main(["verify", "--method", LORA, "--retrain", *common]) == 0

    lines = capsys.readouterr().out.splitlines()
    retrained = trained_root() / "retrain" / LORA
    assert [line.split(" in ")[0] for line in lines[6:11]] == [
        f"trained {retrained / fold_name(fold)}" for fold in range(5)
    ]
    assert lines[11].startswith("tuning-fold macro F1 of fold_0: retrained ")
    assert lines[11].endswith(
        "the study's trial 3: 0.5000 (the same configuration, seed and problems; GPU training "
        "is not bit-reproducible)"
    )
    assert [line.split(":")[0] for line in lines[12:]] == ["pass"] * 4
    record = json.loads((retrained / "fold_1" / TRAINING_FILE).read_text(encoding="utf-8"))
    assert record["seed"] == 1  # the seed that trained the frozen fold 1

    monkeypatch.setitem(VARIANTS, LORA, replace(lenient, retrain_max_median_gap=0.0))
    with pytest.raises(ValueError, match="fail a declared bound:\n(.*\n)FAIL: median score gap"):
        cli.main(["verify", "--method", LORA, "--retrain", *common])


def test_latency_cpu_times_the_method_folder_on_the_cpu_once(frozen, common, monkeypatch, capsys):
    monkeypatch.setattr(cli, "LATENCY_PROBLEMS", 3)
    loaded = []
    load = cli.load
    monkeypatch.setattr(
        cli, "load", lambda folder, *, device: loaded.append(device) or load(folder, device=device)
    )

    assert cli.main(["latency-cpu", "--method", LORA, *common]) == 0

    directory = frozen / ANALYSIS_FOLDER / CPU_LATENCY_FOLDER
    timings = load_latency(directory)
    assert (len(timings.seconds), loaded) == (3, ["cpu"])
    assert capsys.readouterr().out.startswith(
        f"saved {directory / LATENCY_FILE}: 3 problems on the CPU"
    )
    with pytest.raises(FileExistsError, match="the committed timings stay the record"):
        cli.main(["latency-cpu", "--method", LORA, *common])


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.methods.modernbert", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.methods.modernbert.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.methods.modernbert", run_name="__main__")

    assert exit_info.value.code == 0


def test_importing_the_command_starts_neither_mlflow_telemetry_nor_its_hint(tmp_path):
    # tests/tracking/test_environment.py checks every command the default job can import, which
    # this one is not without the finetuning group. The shell is a coding agent's, with nothing
    # that turns telemetry off, where a bare `import mlflow` starts both.
    switches = {"DO_NOT_TRACK", "PYTEST_CURRENT_TEST", "CI", "GITHUB_ACTIONS"}
    shell = {key: value for key, value in os.environ.items() if key not in switches}
    shell = {key: value for key, value in shell.items() if not key.startswith("MLFLOW_")}
    probe = (
        "import sys\n"
        "import codeforces_tags.methods.modernbert.__main__\n"
        "telemetry = sys.modules.get('mlflow.telemetry.client')\n"
        "print(telemetry is not None and telemetry.get_telemetry_client() is not None)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=shell | {"CLAUDECODE": "1", "XDG_CONFIG_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
    assert "mlflow.agent.hint" not in result.stderr


def test_the_notebook_folders_are_the_default_adopt_sources():
    assert cli.NOTEBOOK_FOLDERS == {True: "adapters", False: "models"}
    assert (EXPERIMENTS_DIR / LORA / cli.NOTEBOOK_FOLDERS[True]).name == "adapters"


@pytest.mark.parametrize("name", [LORA, FULL])
def test_the_committed_method_is_frozen_with_the_recipe_of_the_code(name):
    # Needs no weight: a changed setting makes the committed recipe stale until frozen anew.
    directory = METHODS_ASSETS_DIR / name

    check_frozen_method(name)

    record = load_run_record(directory)
    chosen = json.loads((directory / TUNING_FOLDER / "params.json").read_text(encoding="utf-8"))
    pretrained = json.loads(
        (directory / MODEL_FOLDER / PRETRAINED_FILE).read_text(encoding="utf-8")
    )
    assert record.recipe_sha256 == cli.recipe_sha256(VARIANTS[name], chosen, pretrained)
    assert (record.allow_dirty, record.method_entries["supersedes"]) == (False, [])
    parts = load_predictions(directory).index.get_level_values("part").value_counts()
    assert parts.to_dict() == {"dev": 3988, "test": 994}
