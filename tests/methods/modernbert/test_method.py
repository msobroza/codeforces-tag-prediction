"""Tests for the ModernBERT tagging methods: a bag of fold models, and a ScoreMethod."""

import json
import math
import re
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from codeforces_tags.config import DEVICE_ENV_VAR, DEVICES, FOCUS_TAGS, PRETRAINED_DIR_ENV_VAR
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.methods import ScoreMethod, TaggingMethod, save_method
from codeforces_tags.methods.modernbert.method import FineTunedBag, load
from codeforces_tags.methods.modernbert.modeling import FullBag, LoraBag, bag_mean
from codeforces_tags.methods.modernbert.settings import (
    BAG_FILE,
    FOLD_WEIGHTS_FILE,
    FULL,
    LORA,
    PRETRAINED_FILE,
    TRAINED_FILE,
    VARIANTS,
)
from codeforces_tags.methods.modernbert.weights import ChangedWeightsError, MissingWeightsError

# The weights of the fold models that make_bag saves, relative to the trained cache.
FOLD_FILES = {fold: f"{fold}/{FOLD_WEIGHTS_FILE}" for fold in ("fold_0", "fold_1")}


def read_json(path):
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def with_thresholds(bag, thresholds):
    """Return the method of the same fold models, under other thresholds."""
    return FineTunedBag(
        bag.variant,
        bag.params,
        thresholds,
        pretrained=bag.pretrained,
        trained=bag.trained,
        bag=bag.bag,
    )


@pytest.fixture(name="lora_bag")
def lora_bag_fixture(make_bag):
    """Return modernbert_lora over two random fold models, with every threshold at 0.5."""
    return make_bag(LORA)


@pytest.mark.parametrize("name", [LORA, FULL])
def test_both_methods_follow_the_tagging_method_contract(name, make_bag, tiny_problems, tmp_path):
    # Saves, loads through the registry, and loads and predicts in a fresh interpreter, which
    # reads the caches through the environment that the caches fixture sets.
    check_tagging_method(make_bag(name), tiny_problems, tmp_path / "model")


@pytest.mark.parametrize(("name", "kind"), [(LORA, LoraBag), (FULL, FullBag)])
def test_a_bag_is_a_score_method_named_after_its_variant_over_the_focus_tags(name, kind, make_bag):
    method = make_bag(name)

    assert isinstance(method, TaggingMethod)
    assert isinstance(method, ScoreMethod)
    assert (method.name, method.tags) == (name, FOCUS_TAGS)
    assert isinstance(method.bag, kind)
    assert method.bag.names == ("fold_0", "fold_1")


def test_scores_are_float64_probabilities_indexed_like_the_input(lora_bag, tiny_problems):
    shuffled = tiny_problems.iloc[[3, 0, 7, 1, 6, 2, 5, 4]]

    scores = lora_bag.scores(shuffled)

    assert scores.index.equals(shuffled.index)
    assert list(scores.columns) == list(FOCUS_TAGS)
    assert (scores.dtypes == "float64").all()
    assert ((scores > 0) & (scores < 1)).all().all()
    in_order = lora_bag.scores(tiny_problems)
    pd.testing.assert_frame_equal(scores, in_order.loc[shuffled.index], rtol=0, atol=1e-6)


def test_scores_are_the_bag_mean_of_every_fold_models_scores(lora_bag, tiny_problems):
    fold_scores = lora_bag.fold_scores(tiny_problems)

    assert list(fold_scores) == ["fold_0", "fold_1"]
    for scores in fold_scores.values():
        assert (scores.dtype, scores.shape) == (np.float32, (8, len(FOCUS_TAGS)))
    assert not np.allclose(fold_scores["fold_0"], fold_scores["fold_1"])
    np.testing.assert_array_equal(lora_bag.scores(tiny_problems).to_numpy(), bag_mean(fold_scores))


def test_fold_scores_scores_only_the_named_fold_models(lora_bag, tiny_problems):
    every = lora_bag.fold_scores(tiny_problems)

    named = lora_bag.fold_scores(tiny_problems, names=["fold_1"])

    assert list(named) == ["fold_1"]
    np.testing.assert_array_equal(named["fold_1"], every["fold_1"])


def test_predict_gives_1_where_the_score_reaches_its_tags_threshold(lora_bag, tiny_problems):
    scores = lora_bag.scores(tiny_problems)
    # The first problem's scores as thresholds: each of its tags reaches its threshold exactly.
    method = with_thresholds(lora_bag, scores.iloc[0].to_numpy())

    predictions = method.predict(tiny_problems)

    assert predictions.index.equals(tiny_problems.index)
    assert list(predictions.columns) == list(FOCUS_TAGS)
    assert (predictions.dtypes == "int64").all()
    assert predictions.iloc[0].tolist() == [1] * len(FOCUS_TAGS)
    expected = (scores.to_numpy() >= scores.iloc[0].to_numpy()).astype(int)
    np.testing.assert_array_equal(predictions.to_numpy(), expected)
    assert 0 < expected.sum() < expected.size


def test_an_infinite_threshold_predicts_its_tag_for_no_problem(lora_bag, tiny_problems):
    # per_tag_thresholds gives math.inf to a tag that no out-of-fold problem has.
    thresholds = np.zeros(len(FOCUS_TAGS))
    thresholds[FOCUS_TAGS.index("games")] = math.inf

    predictions = with_thresholds(lora_bag, thresholds).predict(tiny_problems)

    assert predictions.sum().to_dict() == {tag: 0 if tag == "games" else 8 for tag in FOCUS_TAGS}


@pytest.mark.parametrize(
    "thresholds",
    [[0.5] * (len(FOCUS_TAGS) - 1), [[0.5] * len(FOCUS_TAGS)], [math.nan] + [0.5] * 7],
    ids=["one short", "two-dimensional", "a NaN"],
)
def test_the_constructor_refuses_anything_but_one_threshold_per_focus_tag(thresholds):
    with pytest.raises(ValueError, match=f"Expected {len(FOCUS_TAGS)} thresholds"):
        FineTunedBag(
            VARIANTS[LORA], {}, np.asarray(thresholds), pretrained={}, trained={}, bag=None
        )


def test_the_constructor_keeps_float64_thresholds_and_a_copy_of_the_params():
    params = {"lora_rank": 4}

    method = FineTunedBag(VARIANTS[LORA], params, [0, 1] * 4, pretrained={}, trained={}, bag=None)
    params["lora_rank"] = 8

    assert method.thresholds_.dtype == np.float64
    assert method.thresholds_.tolist() == [0.0, 1.0] * 4
    assert method.params == {"lora_rank": 4}


def test_scores_refuse_problems_already_preprocessed(lora_bag, tiny_problems):
    with pytest.raises(ValueError, match="already preprocessed"):
        lora_bag.scores(tiny_problems.assign(statement="a tree"))


@pytest.mark.parametrize("name", [LORA, FULL])
def test_save_writes_the_bag_and_both_weight_lists_without_an_absolute_path(
    name, make_bag, caches, tmp_path
):
    method = make_bag(name)
    folder = tmp_path / "model"
    folder.mkdir()

    method.save(folder)

    assert sorted(path.name for path in folder.iterdir()) == [
        BAG_FILE,
        PRETRAINED_FILE,
        TRAINED_FILE,
    ]
    assert read_json(folder / BAG_FILE) == {
        "method": name,
        "format_version": 1,
        "params": method.params,
        "thresholds": [0.5] * len(FOCUS_TAGS),
        "folds": ["fold_0", "fold_1"],
    }
    assert read_json(folder / PRETRAINED_FILE) == method.pretrained
    trained = read_json(folder / TRAINED_FILE)
    assert trained == method.trained
    assert trained["root"] == name
    assert {fold: entry["file"] for fold, entry in trained["folds"].items()} == FOLD_FILES
    for path in folder.iterdir():
        text = path.read_text(encoding="utf-8")
        for absolute in (tmp_path, caches["pretrained"], caches["trained"]):
            assert str(absolute) not in text


@pytest.mark.parametrize("name", [LORA, FULL])
def test_load_reads_a_moved_method_folder_back_with_its_weights_from_the_caches(
    name, make_bag, tiny_problems, tmp_path
):
    method = make_bag(name)
    save_method(method, tmp_path / "here")
    moved = shutil.move(tmp_path / "here", tmp_path / "there")

    loaded = load(moved)

    assert (loaded.variant, loaded.params) == (method.variant, method.params)
    np.testing.assert_array_equal(loaded.thresholds_, method.thresholds_)
    assert (loaded.pretrained, loaded.trained) == (method.pretrained, method.trained)
    pd.testing.assert_frame_equal(
        loaded.scores(tiny_problems), method.scores(tiny_problems), rtol=0, atol=1e-6
    )


def test_an_infinite_threshold_survives_save_and_load(lora_bag, tmp_path):
    thresholds = np.full(len(FOCUS_TAGS), 0.5)
    thresholds[0] = math.inf
    save_method(with_thresholds(lora_bag, thresholds), tmp_path / "model")

    loaded = load(tmp_path / "model")

    np.testing.assert_array_equal(loaded.thresholds_, thresholds)


def test_load_on_a_given_device_overrides_the_methods_setting_and_the_device_variable(
    monkeypatch, lora_bag, tmp_path
):
    save_method(lora_bag, tmp_path / "model")
    monkeypatch.setitem(VARIANTS, LORA, replace(VARIANTS[LORA], device="auto"))
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")  # latency-cpu stays on the CPU all the same

    loaded = load(tmp_path / "model", device="cpu")

    assert loaded.variant == replace(VARIANTS[LORA], device="cpu")
    assert {weight.device.type for weight in loaded.bag.model.parameters()} == {"cpu"}


def test_load_without_a_device_takes_the_device_variable_over_the_methods_setting(
    monkeypatch, lora_bag, tmp_path
):
    save_method(lora_bag, tmp_path / "model")
    monkeypatch.setitem(VARIANTS, LORA, replace(VARIANTS[LORA], device="cuda"))
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")  # the deployment's device, as an image sets it

    loaded = load(tmp_path / "model")

    assert loaded.variant == replace(VARIANTS[LORA], device="cpu")
    assert {weight.device.type for weight in loaded.bag.model.parameters()} == {"cpu"}


def test_load_refuses_cuda_when_torch_sees_no_gpu(monkeypatch, lora_bag, tmp_path):
    save_method(lora_bag, tmp_path / "model")
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="^The device 'cuda' was asked for, but torch sees no"):
        load(tmp_path / "model")


def test_every_registered_methods_device_setting_is_one_of_the_device_settings():
    # load passes it to requested_device as the default, which refuses any other name.
    assert {variant.device for variant in VARIANTS.values()} <= set(DEVICES)


@pytest.mark.parametrize(
    ("stored", "named"),
    [
        ({"method": "label_prior", "format_version": 1}, "'label_prior' (format 1)"),
        ({"method": LORA, "format_version": 2}, f"'{LORA}' (format 2)"),
        ({}, "None (format None)"),
    ],
    ids=["another method", "another format version", "nothing"],
)
def test_load_refuses_a_bag_file_of_another_method_or_format_version(stored, named, tmp_path):
    (tmp_path / BAG_FILE).write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / BAG_FILE} names {named}")):
        load(tmp_path)


def test_load_raises_missing_weights_when_a_fold_model_left_the_trained_cache(
    lora_bag, caches, tmp_path
):
    save_method(lora_bag, tmp_path / "model")
    (caches["trained"] / LORA / FOLD_FILES["fold_1"]).unlink()

    with pytest.raises(MissingWeightsError, match=re.escape(f"lacks ['{FOLD_FILES['fold_1']}']")):
        load(tmp_path / "model")


def test_load_raises_missing_weights_naming_fetch_without_the_pretrained_snapshot(
    monkeypatch, lora_bag, tmp_path
):
    save_method(lora_bag, tmp_path / "model")
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "empty"))

    with pytest.raises(MissingWeightsError, match="fetch --method"):
        load(tmp_path / "model")


def test_load_raises_changed_weights_when_a_fold_models_bytes_changed(lora_bag, caches, tmp_path):
    save_method(lora_bag, tmp_path / "model")
    weights = caches["trained"] / LORA / FOLD_FILES["fold_0"]
    content = bytearray(weights.read_bytes())
    content[-1] ^= 0xFF
    weights.write_bytes(bytes(content))

    with pytest.raises(ChangedWeightsError, match=re.escape(f"['{FOLD_FILES['fold_0']}']")):
        load(tmp_path / "model")
