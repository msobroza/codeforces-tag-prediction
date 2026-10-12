"""Tests for the ModernBERT fold models in the trained cache: trained, adopted and read back."""

import json
import os
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.conformance import starts_like_pickle
from codeforces_tags.methods.modernbert import folds as folds_module
from codeforces_tags.methods.modernbert.folds import (
    adopt_folds,
    check_fold_record,
    fold_name,
    read_dev,
    read_fold_models,
    train_folds,
)
from codeforces_tags.methods.modernbert.inputs import encode_problems
from codeforces_tags.methods.modernbert.method import FineTunedBag
from codeforces_tags.methods.modernbert.modeling import load_tokenizer
from codeforces_tags.methods.modernbert.settings import (
    FOLD_WEIGHTS_FILE,
    FULL,
    HELD_OUT_FILE,
    LORA,
    TRAINING_FILE,
    VARIANTS,
    study_recipe,
)
from codeforces_tags.methods.modernbert.training import TrainingSetup, params_from
from codeforces_tags.methods.modernbert.weights import (
    MissingWeightsError,
    pretrained_weight_list,
    snapshot_folder,
    trained_weight_list,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# What tuning/params.json would hold: a configuration that trains in seconds on the tiny settings.
CHOSEN = {
    LORA: {
        "params": {
            "learning_rate": 1e-3,
            "lora_rank": 4,
            "lora_dropout": 0.0,
            "epochs": 1,
            "batch_size": 4,
            "warmup_ratio": 0.0,
            "pos_weight_power": 0.5,
        },
        "trial": 0,
        "tuning_macro_f1": 0.5,
    },
    FULL: {
        "params": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "classifier_dropout": 0.0,
            "epochs": 1,
            "batch_size": 4,
            "warmup_ratio": 0.0,
            "pos_weight_power": 0.5,
        },
        "trial": 0,
        "tuning_macro_f1": 0.5,
    },
}
# The weights file of each fold model that the former notebooks 05 (LoRA) and 06 (full) saved.
NOTEBOOK_FILES = {LORA: "adapter_model.safetensors", FULL: "model.safetensors"}
# The files of a complete fold model in the trained cache.
FOLD_FILES = sorted([FOLD_WEIGHTS_FILE, HELD_OUT_FILE, TRAINING_FILE])


def read_json(path):
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def held_out_scores(rows, seed=0):
    """Return random float32 probabilities, as a fold model's held-out scores."""
    return np.random.default_rng(seed).random((rows, len(FOCUS_TAGS)), dtype=np.float32)


def fold_record(variant, **changes):
    """Return the training.json of a fold model of the chosen configuration, fields overridable."""
    record = {
        "recipe": study_recipe(variant),
        "params": CHOSEN[variant.name]["params"],
        "seed": variant.seed,
        "train_macro_ap": 0.9,
        "minutes": 1.5,
        "collapsed_seeds": [],
    }
    return record | changes


def write_fold_model(folder, record, held_out):
    """Write a fold model's held-out scores, then its training.json, as training leaves them."""
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / HELD_OUT_FILE, held_out)
    (folder / TRAINING_FILE).write_text(json.dumps(record), encoding="utf-8")


def write_notebook_folds(source, variant, folds=(0, 1)):
    """Write a notebook's fold models (three held-out problems each) in its layout."""
    for fold in folds:
        folder = source / fold_name(fold)
        write_fold_model(folder, fold_record(variant, seed=fold), held_out_scores(3, seed=fold))
        (folder / NOTEBOOK_FILES[variant.name]).write_bytes(f"weights {fold}".encode())


def make_setup(variant, tokenizer, output_dir):
    """Return the setup that trains a method's chosen configuration on the CPU."""
    return TrainingSetup(
        variant=variant,
        params=params_from(variant, CHOSEN[variant.name]["params"]),
        snapshot=snapshot_folder(variant.model_repo, variant.model_revision),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        output_dir=output_dir,
    )


@pytest.fixture(name="raw_dev")
def raw_dev_fixture(tiny_problems):
    """Return 16 raw dev problems in two folds, each fold holding every focus tag once.

    The dev order alternates folds 1, 0, 1, 0, ...: it is not the folds' order.
    """
    copies = tiny_problems.set_axis(tiny_problems.index + 100)  # sample ids 201 to 208
    both = pd.concat([copies.assign(fold=1), tiny_problems.assign(fold=0)])
    order = np.ravel(np.column_stack([np.arange(8), np.arange(8, 16)]))
    return both.iloc[order].assign(group=np.arange(16) + 50)


@pytest.fixture(name="tokenizer")
def tokenizer_fixture(caches):  # pylint: disable=unused-argument
    """Return the tiny snapshot's tokenizer, the same at both methods' pinned revisions."""
    variant = VARIANTS[LORA]
    return load_tokenizer(snapshot_folder(variant.model_repo, variant.model_revision))


@pytest.fixture(name="dev_folds")
def dev_folds_fixture():
    """Return the dev fold of five problems, in a dev order that is not the folds' order."""
    index = pd.Index([201, 101, 202, 102, 203], name="sample_id")
    return pd.Series([1, 0, 1, 0, 1], index=index, name="fold")


def test_fold_name_names_the_fold_model_that_leaves_out_a_dev_fold():
    assert [fold_name(fold) for fold in (0, 4)] == ["fold_0", "fold_4"]


def test_read_dev_gives_folds_groups_labels_and_token_ids_aligned_with_the_raw_rows(
    raw_dev, tokenizer, tiny_variants
):
    variant = tiny_variants[LORA]

    dev = read_dev(raw_dev, tokenizer, variant)

    assert dev.folds.index.equals(raw_dev.index)
    assert dev.folds.tolist() == [1, 0] * 8
    assert dev.groups.index.equals(raw_dev.index)
    assert dev.groups.tolist() == list(range(50, 66))
    assert dev.labels.index.equals(raw_dev.index)
    assert list(dev.labels.columns) == list(FOCUS_TAGS)
    assert dev.labels.idxmax(axis=1).tolist() == [tags[0] for tags in raw_dev["tags"]]
    assert dev.labels.sum(axis=1).tolist() == [1] * 16
    assert len(dev.input_ids) == 16
    for position, ids in enumerate(dev.input_ids):
        one = raw_to_problems(raw_dev.iloc[[position]])
        encoded = encode_problems(one, tokenizer, variant.max_length, variant.code_min_tokens)
        assert ids == encoded[0]


def test_check_fold_record_accepts_the_studys_recipe_and_the_chosen_params(tmp_path):
    variant = VARIANTS[LORA]

    check_fold_record(fold_record(variant), variant, CHOSEN[LORA], tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"recipe": study_recipe(replace(VARIANTS[LORA], max_length=512))},
        {"recipe": study_recipe(VARIANTS[FULL])},
        {"params": CHOSEN[LORA]["params"] | {"lora_rank": 8}},
        {"recipe": None},
    ],
    ids=["another max_length", "the other method's recipe", "other params", "no recipe"],
)
def test_check_fold_record_refuses_another_recipe_or_other_params(changes, tmp_path):
    variant = VARIANTS[LORA]

    with pytest.raises(ValueError, match=re.escape(f"{tmp_path} was trained under another recipe")):
        check_fold_record(fold_record(variant, **changes), variant, CHOSEN[LORA], tmp_path)


@pytest.fixture(name="training")
def training_fixture(raw_dev, tokenizer, tiny_variants, caches, tmp_path):
    """Return a preparer of what ``train_folds`` reads for a method, on the tiny settings.

    ``training(name)`` returns ``{"variant", "dev", "setup", "root"}``: the method's tiny variant,
    its dev data from ``raw_dev``, its setup (Trainer folder ``tmp_path / "trainer"``) and its
    folder in the trained cache.
    """

    def prepare(name):
        variant = tiny_variants[name]
        return {
            "variant": variant,
            "dev": read_dev(raw_dev, tokenizer, variant),
            "setup": make_setup(variant, tokenizer, tmp_path / "trainer"),
            "root": caches["trained"] / name,
        }

    return prepare


@pytest.fixture(name="trained")
def trained_fixture(request, training):
    """Train both fold models of the method that parametrizes the test (``indirect``).

    Returns:
        ``training(name)``, with the ``lines`` that ``train_folds`` yielded.
    """
    prepared = training(request.param)
    lines = train_folds(prepared["setup"], CHOSEN[request.param], prepared["dev"], prepared["root"])
    return prepared | {"lines": list(lines)}


@pytest.mark.parametrize("trained", [LORA, FULL], indirect=True)
def test_train_folds_trains_one_fold_model_per_dev_fold_into_the_root(trained):
    variant, root = trained["variant"], trained["root"]

    assert len(trained["lines"]) == 2
    for fold, line in enumerate(trained["lines"]):
        folder = root / fold_name(fold)
        pattern = rf"trained {re.escape(str(folder))} in \d+\.\d min with seed 0 "
        assert re.fullmatch(pattern + r"\(training macro AP \d\.\d{3}\)", line)
        assert sorted(path.name for path in folder.iterdir()) == FOLD_FILES
        held_out = np.load(folder / HELD_OUT_FILE)
        assert (held_out.dtype, held_out.shape) == (np.float32, (8, len(FOCUS_TAGS)))
        record = read_json(folder / TRAINING_FILE)
        assert list(record) == [
            "recipe",
            "params",
            "seed",
            "train_macro_ap",
            "minutes",
            "collapsed_seeds",
        ]
        assert record["recipe"] == study_recipe(variant)
        assert record["params"] == CHOSEN[variant.name]["params"]
        assert (record["seed"], record["collapsed_seeds"]) == (variant.seed, [])
        assert 0 <= record["train_macro_ap"] <= 1
        assert record["minutes"] > 0


@pytest.mark.parametrize("trained", [LORA, FULL], indirect=True)
def test_train_folds_writes_no_pickle_and_leaves_the_trainers_folder_empty(trained):
    written = [path for path in trained["root"].rglob("*") if path.is_file()]

    assert len(written) == 6
    assert not any(starts_like_pickle(path) for path in written)
    assert not [path for path in trained["setup"].output_dir.rglob("*") if path.is_file()]


@pytest.mark.parametrize("trained", [LORA, FULL], indirect=True)
def test_the_held_out_scores_are_the_saved_fold_models_scores_of_its_fold_in_dev_order(
    trained, raw_dev
):
    variant, dev = trained["variant"], trained["dev"]
    names = [fold_name(fold) for fold in (0, 1)]
    pretrained = (variant.model_repo, variant.model_revision, variant.pretrained_files)
    bag = FineTunedBag.build(
        variant,
        CHOSEN[variant.name]["params"],
        np.full(len(FOCUS_TAGS), 0.5),
        pretrained_weight_list(*pretrained),
        trained_weight_list(variant.name, {name: f"{name}/{FOLD_WEIGHTS_FILE}" for name in names}),
    )

    for fold, name in enumerate(names):
        held_out = np.load(trained["root"] / name / HELD_OUT_FILE)
        rescored = bag.fold_scores(raw_dev[(dev.folds == fold).to_numpy()], names=[name])[name]
        np.testing.assert_allclose(rescored, held_out, rtol=0, atol=1e-5)


def test_train_folds_keeps_complete_fold_models_without_training_them(training):
    prepared = training(LORA)
    variant, root = prepared["variant"], prepared["root"]
    for fold in (0, 1):
        write_fold_model(root / fold_name(fold), fold_record(variant), held_out_scores(8, fold))

    lines = list(train_folds(prepared["setup"], CHOSEN[LORA], prepared["dev"], root))

    assert lines == [
        f"{root / f'fold_{fold}'} holds a complete fold model: kept" for fold in (0, 1)
    ]
    assert sorted(path.name for path in (root / "fold_0").iterdir()) == [
        HELD_OUT_FILE,
        TRAINING_FILE,
    ]
    np.testing.assert_array_equal(np.load(root / "fold_1" / HELD_OUT_FILE), held_out_scores(8, 1))


def test_train_folds_refuses_a_complete_fold_model_of_another_recipe(training):
    prepared = training(LORA)
    variant, root = prepared["variant"], prepared["root"]
    other = study_recipe(replace(variant, max_length=512))
    write_fold_model(root / "fold_0", fold_record(variant, recipe=other), held_out_scores(8))

    with pytest.raises(ValueError, match=re.escape(f"{root / 'fold_0'} was trained under")):
        list(train_folds(prepared["setup"], CHOSEN[LORA], prepared["dev"], root))


def test_train_folds_starts_from_the_given_seed_and_trains_on_the_other_folds(
    monkeypatch, training
):
    prepared = training(LORA)
    dev, root = prepared["dev"], prepared["root"]
    write_fold_model(root / "fold_1", fold_record(prepared["variant"]), held_out_scores(8))
    calls, train_fold_model = [], folds_module.train_fold_model

    def spy(setup, train_ids, train_labels, *, first_seed):
        calls.append({"ids": train_ids, "labels": train_labels, "first_seed": first_seed})
        return train_fold_model(setup, train_ids, train_labels, first_seed=first_seed)

    monkeypatch.setattr(folds_module, "train_fold_model", spy)
    seeds = {"fold_0": 1, "fold_7": 2}

    lines = list(train_folds(prepared["setup"], CHOSEN[LORA], dev, root, seeds))

    assert "with seed 1 " in lines[0]
    assert lines[1] == f"{root / 'fold_1'} holds a complete fold model: kept"
    assert read_json(root / "fold_0" / TRAINING_FILE)["seed"] == 1
    assert [call["first_seed"] for call in calls] == [1]
    other_folds = (dev.folds != 0).to_numpy()
    assert calls[0]["ids"] == [
        ids for ids, kept in zip(dev.input_ids, other_folds, strict=True) if kept
    ]
    np.testing.assert_array_equal(calls[0]["labels"], dev.labels[other_folds].to_numpy())


@pytest.mark.parametrize("name", [LORA, FULL])
def test_adopt_folds_hard_links_a_notebooks_fold_models_into_the_trained_cache(name, tmp_path):
    variant = VARIANTS[name]
    source, root = tmp_path / "runs" / name, tmp_path / "trained" / name
    write_notebook_folds(source, variant)

    lines = list(adopt_folds(variant, CHOSEN[name], source, root, range(2)))

    assert lines == [f"adopted {source / f'fold_{f}'} as {root / f'fold_{f}'}" for f in (0, 1)]
    for fold in (0, 1):
        origin, folder = source / fold_name(fold), root / fold_name(fold)
        assert sorted(path.name for path in folder.iterdir()) == FOLD_FILES
        assert os.path.samefile(origin / NOTEBOOK_FILES[name], folder / FOLD_WEIGHTS_FILE)
        assert os.path.samefile(origin / HELD_OUT_FILE, folder / HELD_OUT_FILE)
        expected = fold_record(variant, seed=fold) | {"adopted_from": origin.resolve().as_posix()}
        assert read_json(folder / TRAINING_FILE) == expected


def test_adopt_folds_records_a_source_inside_the_repository_relative_to_it(monkeypatch, tmp_path):
    monkeypatch.setattr(folds_module, "PROJECT_ROOT", tmp_path.resolve())
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))

    list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))

    assert read_json(root / "fold_0" / TRAINING_FILE)["adopted_from"] == f"runs/{LORA}/fold_0"


def test_adopt_folds_adopts_again_a_fold_model_left_incomplete_by_an_interrupted_run(tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))
    other = tmp_path / "other_run.safetensors"
    other.write_bytes(b"another run's weights")
    (root / "fold_0").mkdir(parents=True)
    os.link(other, root / "fold_0" / FOLD_WEIGHTS_FILE)  # linked, then stopped before training.json

    lines = list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))

    assert lines == [f"adopted {source / 'fold_0'} as {root / 'fold_0'}"]
    assert os.path.samefile(
        source / "fold_0" / NOTEBOOK_FILES[LORA], root / "fold_0" / FOLD_WEIGHTS_FILE
    )
    assert other.read_bytes() == b"another run's weights"  # the other path is left alone


def test_adopt_folds_copies_the_files_when_a_hard_link_fails(monkeypatch, tmp_path):
    def refuse(*_):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))

    list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))

    origin, folder = source / "fold_0", root / "fold_0"
    for original, copy in (
        (origin / NOTEBOOK_FILES[LORA], folder / FOLD_WEIGHTS_FILE),
        (origin / HELD_OUT_FILE, folder / HELD_OUT_FILE),
    ):
        assert not os.path.samefile(original, copy)
        assert copy.read_bytes() == original.read_bytes()


@pytest.mark.parametrize("missing", [NOTEBOOK_FILES[LORA], TRAINING_FILE, HELD_OUT_FILE])
def test_adopt_folds_raises_missing_weights_when_a_source_file_is_missing(missing, tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))
    (source / "fold_0" / missing).unlink()

    with pytest.raises(MissingWeightsError, match=re.escape(str(source / "fold_0" / missing))):
        list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))
    assert not root.exists()


def test_adopt_folds_reads_the_lora_adapter_file_not_a_full_models_file(tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))
    (source / "fold_0" / NOTEBOOK_FILES[LORA]).rename(source / "fold_0" / NOTEBOOK_FILES[FULL])

    with pytest.raises(MissingWeightsError, match=NOTEBOOK_FILES[LORA]):
        list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))


def test_adopt_folds_refuses_a_source_fold_model_of_another_configuration(tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))
    params = CHOSEN[LORA]["params"] | {"epochs": 9}
    write_fold_model(source / "fold_0", fold_record(variant, params=params), held_out_scores(3))

    with pytest.raises(ValueError, match=re.escape(f"{source / 'fold_0'} was trained under")):
        list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))
    assert not root.exists()


def test_adopt_folds_keeps_a_complete_fold_model_of_the_trained_cache(tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant)
    write_fold_model(root / "fold_0", fold_record(variant), held_out_scores(3, seed=7))

    lines = list(adopt_folds(variant, CHOSEN[LORA], source, root, range(2)))

    assert lines == [
        f"{root / 'fold_0'} holds a complete fold model: kept",
        f"adopted {source / 'fold_1'} as {root / 'fold_1'}",
    ]
    np.testing.assert_array_equal(np.load(root / "fold_0" / HELD_OUT_FILE), held_out_scores(3, 7))
    assert not (root / "fold_0" / FOLD_WEIGHTS_FILE).exists()


def test_adopt_folds_refuses_a_complete_fold_model_of_another_recipe_in_the_cache(tmp_path):
    variant = VARIANTS[LORA]
    source, root = tmp_path / "runs" / LORA, tmp_path / "trained" / LORA
    write_notebook_folds(source, variant, folds=(0,))
    other = study_recipe(replace(variant, max_length=512))
    write_fold_model(root / "fold_0", fold_record(variant, recipe=other), held_out_scores(3))

    with pytest.raises(ValueError, match=re.escape(f"{root / 'fold_0'} was trained under")):
        list(adopt_folds(variant, CHOSEN[LORA], source, root, [0]))


def test_read_fold_models_gives_records_in_fold_order_and_scores_in_dev_order(dev_folds, tmp_path):
    variant, root = VARIANTS[LORA], tmp_path / LORA
    held_out = {1: held_out_scores(3, seed=1), 0: held_out_scores(2, seed=0)}
    for fold, scores in held_out.items():
        write_fold_model(root / fold_name(fold), fold_record(variant, seed=fold), scores)

    records, out_of_fold = read_fold_models(variant, CHOSEN[LORA], dev_folds, root)

    assert records == {fold_name(fold): fold_record(variant, seed=fold) for fold in (0, 1)}
    assert list(records) == ["fold_0", "fold_1"]
    assert out_of_fold.index.equals(dev_folds.index)
    assert list(out_of_fold.columns) == list(FOCUS_TAGS)
    assert (out_of_fold.dtypes == "float64").all()
    for fold, sample_ids in ((0, [101, 102]), (1, [201, 202, 203])):
        exact = held_out[fold].astype(np.float64)  # the exact cast of the float32 file
        np.testing.assert_array_equal(out_of_fold.loc[sample_ids].to_numpy(), exact)


def test_a_fold_model_adopted_from_a_notebook_reads_back_with_its_scores(tmp_path):
    variant = VARIANTS[FULL]
    source, root = tmp_path / "runs" / FULL, tmp_path / "trained" / FULL
    write_notebook_folds(source, variant)
    list(adopt_folds(variant, CHOSEN[FULL], source, root, range(2)))
    folds = pd.Series([0, 1, 0, 1, 0, 1], index=pd.Index(range(6), name="sample_id"))

    records, out_of_fold = read_fold_models(variant, CHOSEN[FULL], folds, root)

    assert records["fold_1"]["adopted_from"] == (source / "fold_1").resolve().as_posix()
    np.testing.assert_array_equal(out_of_fold.loc[[1, 3, 5]].to_numpy(), held_out_scores(3, 1))


def test_read_fold_models_raises_missing_weights_for_an_incomplete_fold_model(dev_folds, tmp_path):
    variant, root = VARIANTS[LORA], tmp_path / LORA
    write_fold_model(root / "fold_0", fold_record(variant), held_out_scores(2))
    (root / "fold_1").mkdir()
    np.save(root / "fold_1" / HELD_OUT_FILE, held_out_scores(3))  # no training.json yet

    with pytest.raises(MissingWeightsError, match=re.escape(f"{root / 'fold_1'} holds no")):
        read_fold_models(variant, CHOSEN[LORA], dev_folds, root)


def test_read_fold_models_refuses_a_fold_model_of_other_params(dev_folds, tmp_path):
    variant, root = VARIANTS[LORA], tmp_path / LORA
    write_fold_model(root / "fold_0", fold_record(variant), held_out_scores(2))
    params = CHOSEN[LORA]["params"] | {"learning_rate": 1e-4}
    write_fold_model(root / "fold_1", fold_record(variant, params=params), held_out_scores(3))

    with pytest.raises(ValueError, match=re.escape(f"{root / 'fold_1'} was trained under")):
        read_fold_models(variant, CHOSEN[LORA], dev_folds, root)


def test_read_fold_models_refuses_held_out_scores_of_another_shape(dev_folds, tmp_path):
    variant, root = VARIANTS[LORA], tmp_path / LORA
    write_fold_model(root / "fold_0", fold_record(variant), held_out_scores(2))
    write_fold_model(root / "fold_1", fold_record(variant), held_out_scores(2))

    with pytest.raises(ValueError, match=r"has the shape \(2, 8\); fold 1 has 3 problems"):
        read_fold_models(variant, CHOSEN[LORA], dev_folds, root)
