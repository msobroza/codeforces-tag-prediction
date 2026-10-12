"""Tests for training one ModernBERT fold model with the Hugging Face ``Trainer``."""

import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import set_seed

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.conformance import starts_like_pickle
from codeforces_tags.methods.modernbert import training
from codeforces_tags.methods.modernbert.modeling import load_tokenizer
from codeforces_tags.methods.modernbert.settings import FOLD_WEIGHTS_FILE, FULL, LORA, VARIANTS
from codeforces_tags.methods.modernbert.training import (
    EpochHook,
    FullParams,
    LoraParams,
    TagDataset,
    TrainingSetup,
    WeightedBce,
    build_training_model,
    fold_state,
    params_from,
    positive_weights,
    save_fold,
    train_fold_model,
    train_model,
    training_arguments,
    training_macro_ap,
)
from codeforces_tags.methods.modernbert.weights import snapshot_folder

CPU = torch.device("cpu")


@pytest.fixture(name="snapshot")
def snapshot_fixture(caches):  # pylint: disable=unused-argument
    """Return the getter of a method's tiny pretrained snapshot folder: ``snapshot(name)``."""

    def folder(name):
        return snapshot_folder(VARIANTS[name].model_repo, VARIANTS[name].model_revision)

    return folder


@pytest.fixture(name="tokenizer")
def tokenizer_fixture(snapshot):
    """Return the tiny snapshot's tokenizer."""
    return load_tokenizer(snapshot(LORA))


@pytest.fixture(name="setup")
def setup_fixture(tiny_variants, snapshot, tokenizer, tmp_path):
    """Return a builder of a method's training setup on the CPU: ``setup(name, params)``."""

    def build(name, params):
        variant = tiny_variants[name]
        return TrainingSetup(variant, params, snapshot(name), tokenizer, CPU, tmp_path / "trainer")

    return build


@pytest.fixture(name="problems")
def problems_fixture(tokenizer, statements):
    """Return the token ids of one statement per focus tag, and their 0/1 labels."""
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    input_ids = [[cls, *tokenizer(statements[tag])["input_ids"], sep] for tag in FOCUS_TAGS]
    return input_ids, np.eye(len(FOCUS_TAGS), dtype=np.int64)


def fake_fold_training(monkeypatch, train_aps):
    """Replace training and its collapse check by fakes that record what they were given.

    Returns:
        ``{"seeds": [...], "models": {seed: model}, "samples": [(ids, labels), ...]}``.
    """
    calls = {"seeds": [], "models": {}, "samples": []}
    aps = iter(train_aps)

    def train(setup, train_ids, train_labels, *, seed):  # pylint: disable=unused-argument
        calls["seeds"].append(seed)
        calls["models"][seed] = SimpleNamespace(seed=seed)
        return calls["models"][seed]

    def macro_ap(model, variant, sample_ids, sample_labels, tokenizer):  # pylint: disable=unused-argument
        calls["samples"].append((sample_ids, sample_labels))
        return next(aps)

    monkeypatch.setattr(training, "train_model", train)
    monkeypatch.setattr(training, "training_macro_ap", macro_ap)
    return calls


def fake_setup(variant, lora_params, tmp_path):
    """Return a training setup that the faked training never reads."""
    return TrainingSetup(variant, lora_params, Path("unused"), None, CPU, tmp_path / "trainer")


@pytest.mark.parametrize("name", [LORA, FULL])
def test_params_from_reads_a_configuration_of_the_method_s_search_space(params, name):
    read = params_from(VARIANTS[name], asdict(params[name]))

    assert read == params[name]
    assert isinstance(read, LoraParams if name == LORA else FullParams)


def test_params_from_refuses_a_parameter_outside_the_method_s_search_space(params):
    with pytest.raises(TypeError, match="momentum"):
        params_from(VARIANTS[LORA], {**asdict(params[LORA]), "momentum": 0.9})
    with pytest.raises(TypeError, match="weight_decay"):
        params_from(VARIANTS[LORA], asdict(params[FULL]))


def test_positive_weights_raise_each_tag_s_negative_to_positive_ratio_to_the_power():
    labels = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])  # 3, 2 and 0 positives of 4

    weights = positive_weights(labels, 0.5)

    assert weights.dtype == torch.float32
    # A tag with no positive counts one: (4 / 1) ** 0.5.
    torch.testing.assert_close(weights, torch.tensor([math.sqrt(1 / 3), 1.0, 2.0]))
    torch.testing.assert_close(positive_weights(labels, 0.0), torch.ones(3))


def hand_weighted_bce(logits, labels, pos_weight):
    """Return the sum of each label's weighted binary cross-entropy, computed by hand."""
    total = 0.0
    for logit_row, label_row in zip(logits, labels, strict=True):
        for logit, label, weight in zip(logit_row, label_row, pos_weight, strict=True):
            probability = 1 / (1 + math.exp(-logit))
            positive = weight * label * math.log(probability)
            negative = (1 - label) * math.log(1 - probability)
            total -= positive + negative
    return total


@pytest.mark.parametrize(
    ("num_items_in_batch", "count"),
    [pytest.param(12, 12, id="divided by the step's labels"), pytest.param(None, 6, id="mean")],
)
def test_weighted_bce_is_the_summed_weighted_loss_over_the_labels_of_the_step(
    num_items_in_batch, count
):
    logits = [[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]]
    labels = [[1, 0, 1], [0, 1, 0]]
    pos_weight = [2.0, 0.5, 1.0]
    outputs = SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float16))  # as under fp16

    loss = WeightedBce(torch.tensor(pos_weight))(
        outputs, torch.tensor(labels), num_items_in_batch=num_items_in_batch
    )

    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(hand_weighted_bce(logits, labels, pos_weight) / count)


def test_a_tag_dataset_gives_each_problem_s_ids_attention_mask_and_float_labels():
    dataset = TagDataset([[1, 5, 2], [1, 2]], np.array([[1, 0], [0, 1]]))

    assert len(dataset) == 2
    assert dataset[1] == {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0.0, 1.0]}
    assert all(isinstance(label, float) for label in dataset[0]["labels"])


def test_the_epoch_hook_calls_the_function_with_the_finished_epoch_and_the_model():
    calls = []
    model = object()

    EpochHook(lambda epoch, hooked: calls.append((epoch, hooked))).on_epoch_end(
        None, SimpleNamespace(epoch=1.9999999), None, model=model
    )

    assert calls == [(2, model)]
    assert isinstance(calls[0][0], int)


def test_the_training_arguments_of_a_lora_configuration_on_the_cpu(params, tmp_path):
    tuned = replace(params[LORA], learning_rate=5e-4, epochs=3, batch_size=32, warmup_ratio=0.15)

    args = training_arguments(VARIANTS[LORA], tuned, output_dir=tmp_path, seed=7, device=CPU)

    assert args.output_dir == str(tmp_path)
    assert (args.per_device_train_batch_size, args.gradient_accumulation_steps) == (4, 8)
    assert (args.learning_rate, args.num_train_epochs) == (5e-4, 3)
    assert args.warmup_steps == 0.15  # a share of the steps, as the float warmup ratio
    assert args.lr_scheduler_type == "linear"
    assert args.weight_decay == VARIANTS[LORA].weight_decay == 0.01
    assert (args.fp16, args.use_cpu) == (False, True)
    assert args.train_sampling_strategy == "group_by_length"
    assert args.eval_strategy == args.save_strategy == args.logging_strategy == "no"
    assert args.report_to in ([], "none", ["none"])
    assert args.label_names == ["labels"]
    assert (args.seed, args.data_seed) == (7, 7)


def test_full_fine_tuning_takes_the_weight_decay_of_its_configuration(params, tmp_path):
    tuned = replace(params[FULL], weight_decay=0.05, batch_size=16)

    args = training_arguments(VARIANTS[FULL], tuned, output_dir=tmp_path, seed=0, device=CPU)

    assert args.weight_decay == 0.05
    assert args.gradient_accumulation_steps == 4


def test_a_batch_smaller_than_the_micro_batch_takes_one_micro_batch_per_step(params, tmp_path):
    tuned = replace(params[LORA], batch_size=2)

    args = training_arguments(VARIANTS[LORA], tuned, output_dir=tmp_path, seed=0, device=CPU)

    assert args.gradient_accumulation_steps == 1


def test_training_on_cuda_asks_for_fp16_and_not_for_the_cpu(monkeypatch, params, tmp_path):
    # Only the settings are read: TrainingArguments refuses fp16 on a machine without CUDA.
    monkeypatch.setattr(training, "TrainingArguments", dict)

    settings = training_arguments(
        VARIANTS[LORA], params[LORA], output_dir=tmp_path, seed=0, device=torch.device("cuda")
    )

    assert (settings["fp16"], settings["use_cpu"]) == (True, False)


def test_a_lora_training_model_trains_only_the_updates_the_head_and_the_classifier(
    snapshot, params
):
    model = build_training_model(VARIANTS[LORA], params[LORA], snapshot(LORA))

    trainable = {name for name, weight in model.named_parameters() if weight.requires_grad}
    lora = {name for name, _ in model.named_parameters() if "lora_" in name}
    copies = trainable - lora
    assert lora and lora <= trainable
    assert {name.split(".modules_to_save.")[0] for name in copies} == {
        "base_model.model.head",
        "base_model.model.classifier",
    }
    assert model.peft_config["default"].r == params[LORA].lora_rank
    assert model.peft_config["default"].lora_dropout == params[LORA].lora_dropout


def test_a_full_training_model_trains_every_weight_with_its_classifier_dropout(snapshot, params):
    tuned = replace(params[FULL], classifier_dropout=0.15)

    model = build_training_model(VARIANTS[FULL], tuned, snapshot(FULL))

    assert all(weight.requires_grad for weight in model.parameters())
    assert model.config.classifier_dropout == model.drop.p == 0.15


@pytest.mark.parametrize("name", [LORA, FULL])
def test_train_model_trains_the_model_runs_the_hook_after_each_epoch_and_writes_nothing(
    setup, params, problems, tmp_path, name
):
    trainer_setup = setup(name, replace(params[name], epochs=2))
    hooked = []

    model = train_model(
        trainer_setup, *problems, seed=3, epoch_hook=lambda epoch, at: hooked.append((epoch, at))
    )

    assert hooked == [(1, model), (2, model)]
    set_seed(3)
    initial = build_training_model(
        trainer_setup.variant, trainer_setup.params, trainer_setup.snapshot
    )
    trained, initial = dict(model.named_parameters()), dict(initial.named_parameters())
    changed = {key for key, weight in trained.items() if not torch.equal(weight, initial[key])}
    assert changed  # and only weights that train:
    assert changed <= {key for key, weight in trained.items() if weight.requires_grad}
    assert not any(trainer_setup.output_dir.rglob("*"))  # no checkpoint, log or report
    assert not [path for path in tmp_path.rglob("*") if path.is_file() and starts_like_pickle(path)]


def test_train_model_starts_from_the_initial_model_of_its_seed(setup, params, problems):
    # With a learning rate of 0, the trained model is the model the seed built.
    trainer_setup = setup(LORA, replace(params[LORA], learning_rate=0.0))

    model = train_model(trainer_setup, *problems, seed=3)

    set_seed(3)
    initial = build_training_model(
        trainer_setup.variant, trainer_setup.params, trainer_setup.snapshot
    ).state_dict()
    assert all(torch.equal(tensor, initial[key]) for key, tensor in model.state_dict().items())


def test_training_macro_ap_is_the_macro_average_precision_of_the_sample(monkeypatch, tiny_variants):
    variant = tiny_variants[LORA]
    asked = {}

    def scores(model, input_ids, tokenizer, **kwargs):  # pylint: disable=unused-argument
        asked.update(kwargs)
        return np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.6]], dtype=np.float32)

    monkeypatch.setattr(training, "predict_scores", scores)
    labels = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])

    train_ap = training_macro_ap(None, variant, [[1]] * 4, labels, None)

    # First tag: positives ranked 1st and 3rd, (1 + 2/3) / 2; second tag: ranked first, 1.
    assert math.isclose(train_ap, ((1 + 2 / 3) / 2 + 1) / 2)
    assert asked == {
        "batch_size": variant.score_batch,
        "pad_to_multiple_of": variant.pad_to_multiple_of,
    }


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_training_macro_ap_is_nan_when_a_score_is_not_finite(monkeypatch, tiny_variants, bad):
    monkeypatch.setattr(
        training, "predict_scores", lambda *args, **kwargs: np.array([[0.5], [bad]], np.float32)
    )

    assert math.isnan(training_macro_ap(None, tiny_variants[LORA], [[1], [2]], [[1], [0]], None))


def test_training_macro_ap_scores_a_model_on_its_training_sample(
    tiny_variants, snapshot, params, tokenizer, problems
):
    model = build_training_model(tiny_variants[LORA], params[LORA], snapshot(LORA))

    train_ap = training_macro_ap(model, tiny_variants[LORA], *problems, tokenizer)

    assert 0 <= train_ap <= 1


def test_a_collapsed_fold_model_is_trained_again_with_the_next_seed(
    monkeypatch, tiny_variants, params, tmp_path
):
    variant = replace(tiny_variants[LORA], seed=5, collapse_ap=0.3, collapse_sample=3)
    calls = fake_fold_training(monkeypatch, [0.0, 0.9])
    train_ids = [[100 + row] for row in range(8)]
    train_labels = np.arange(8 * 2).reshape(8, 2)

    model, record = train_fold_model(
        fake_setup(variant, params[LORA], tmp_path), train_ids, train_labels
    )

    assert calls["seeds"] == [5, 6]
    assert model is calls["models"][6]
    assert record == {
        "params": asdict(params[LORA]),
        "seed": 6,
        "train_macro_ap": 0.9,
        "minutes": record["minutes"],
        "collapsed_seeds": [5],
    }
    assert 0 <= record["minutes"] < 1
    assert len(calls["samples"]) == 2
    first_ids, first_labels = calls["samples"][0]
    second_ids, second_labels = calls["samples"][1]
    assert first_ids == second_ids  # the same training sample measures every seed
    assert len(first_ids) == len({ids[0] for ids in first_ids}) == 3
    rows = [ids[0] - 100 for ids in first_ids]
    np.testing.assert_array_equal(first_labels, train_labels[rows])
    np.testing.assert_array_equal(second_labels, train_labels[rows])


@pytest.mark.parametrize("collapse", [0.0, math.nan], ids=["uninformative", "diverged"])
def test_fold_training_fails_when_every_seed_collapses(
    monkeypatch, tiny_variants, params, tmp_path, collapse
):
    variant = replace(tiny_variants[LORA], collapse_ap=0.3)
    calls = fake_fold_training(monkeypatch, [collapse] * variant.fold_seeds)

    with pytest.raises(RuntimeError, match=r"Every seed tried collapsed: \[0, 1, 2\]"):
        train_fold_model(fake_setup(variant, params[LORA], tmp_path), [[1], [2]], np.eye(2))
    assert calls["seeds"] == [0, 1, 2]


def test_fold_training_starts_from_the_first_seed_given(
    monkeypatch, tiny_variants, params, tmp_path
):
    variant = replace(tiny_variants[LORA], collapse_ap=0.3)
    calls = fake_fold_training(monkeypatch, [0.0, 0.9])

    _, record = train_fold_model(
        fake_setup(variant, params[LORA], tmp_path), [[1], [2]], np.eye(2), first_seed=1
    )

    assert calls["seeds"] == [1, 2]
    assert (record["seed"], record["collapsed_seeds"]) == (2, [1])


def test_the_fold_state_of_a_lora_model_holds_only_what_lora_trained(snapshot, params):
    model = build_training_model(VARIANTS[LORA], params[LORA], snapshot(LORA))

    state = fold_state(model, VARIANTS[LORA])

    assert set(state) == set(get_peft_model_state_dict(model))
    assert sum("lora_" in key for key in state) == 2 * 4 * 2  # A and B, four layers, two blocks
    assert {key for key in state if "lora_" not in key} == {
        "base_model.model.head.dense.weight",
        "base_model.model.head.norm.weight",
        "base_model.model.classifier.weight",
        "base_model.model.classifier.bias",
    }
    assert all(tensor.device == CPU and tensor.is_contiguous() for tensor in state.values())


def test_the_fold_state_of_a_full_model_is_its_whole_state_dict_made_contiguous(snapshot, params):
    model = build_training_model(VARIANTS[FULL], params[FULL], snapshot(FULL))
    transposed = torch.randn(model.config.hidden_size, len(FOCUS_TAGS)).t()
    model.classifier.weight = torch.nn.Parameter(transposed)  # not contiguous

    state = fold_state(model, VARIANTS[FULL])

    assert set(state) == set(model.state_dict())
    assert state["classifier.weight"].is_contiguous()
    assert torch.equal(state["classifier.weight"], transposed)
    assert all(not tensor.requires_grad for tensor in state.values())


@pytest.mark.parametrize("name", [LORA, FULL])
def test_save_fold_writes_the_fold_state_as_safetensors_not_as_a_pickle(
    snapshot, params, tmp_path, name
):
    model = build_training_model(VARIANTS[name], params[name], snapshot(name))
    folder = tmp_path / "trained" / name / "fold_0"

    path = save_fold(model, VARIANTS[name], folder)

    assert path == folder / FOLD_WEIGHTS_FILE
    assert not starts_like_pickle(path)
    saved, state = load_file(path), fold_state(model, VARIANTS[name])
    assert set(saved) == set(state)
    assert all(torch.equal(saved[key], tensor) for key, tensor in state.items())
    assert save_fold(model, VARIANTS[name], folder) == path  # the folder may exist already
