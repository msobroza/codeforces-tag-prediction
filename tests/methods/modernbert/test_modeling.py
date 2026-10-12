"""Tests for the ModernBERT methods' models: built, scored, and gathered into a bag."""

import math

import numpy as np
import pytest
import torch
from peft import TaskType
from safetensors.torch import load_file, save_file

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.modernbert import modeling
from codeforces_tags.methods.modernbert.modeling import (
    FullBag,
    LoraBag,
    _Bag,
    bag_mean,
    classifier_model,
    load_tokenizer,
    lora_config,
    predict_scores,
    resolve_device,
)
from codeforces_tags.methods.modernbert.settings import ENCODER_FILE, FULL, LORA, VARIANTS
from codeforces_tags.methods.modernbert.training import build_training_model, save_fold
from codeforces_tags.methods.modernbert.weights import snapshot_folder

CPU = torch.device("cpu")
# Inputs of different lengths, out of length order, in words the tiny tokenizer knows.
TEXTS = (
    "a tree of nodes and edges of the graph is a path",
    "sum",
    "print a string",
    "the probability to win a game is the sum of the probability of each path in the graph",
    "gcd",
    "count the vertices",
)
FOLDS = ("fold_0", "fold_1")


@pytest.fixture(name="snapshots")
def snapshots_fixture(caches, tiny_variants):  # pylint: disable=unused-argument
    """Return the tiny pretrained snapshot's folder of each method."""
    return {
        name: snapshot_folder(variant.model_repo, variant.model_revision)
        for name, variant in tiny_variants.items()
    }


@pytest.fixture(name="tokenizer")
def tokenizer_fixture(snapshots):
    """Return the tiny snapshot's tokenizer."""
    return load_tokenizer(snapshots[LORA])


@pytest.fixture(name="input_ids")
def input_ids_fixture(tokenizer):
    """Return the token ids of ``[CLS] text [SEP]`` for each of ``TEXTS``."""
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    return [[cls, *tokenizer(text)["input_ids"], sep] for text in TEXTS]


@pytest.fixture(name="fold_models")
def fold_models_fixture(tiny_variants, snapshots, params, tmp_path):
    """Return a writer of two random fold models of a method: ``fold_models(name)``.

    It builds each fold model as training does, moves every weight that trains at random (LoRA's
    B matrices start at zero), saves it with ``save_fold``, and returns the fold models, in
    memory, and their weights files, by fold name.
    """

    def save(name):
        variant, models, files = tiny_variants[name], {}, {}
        for seed, fold in enumerate(FOLDS):
            torch.manual_seed(seed)
            models[fold] = build_training_model(variant, params[name], snapshots[name])
            trained = [weight for weight in models[fold].parameters() if weight.requires_grad]
            with torch.no_grad():
                for weight in trained:
                    weight.add_(torch.randn_like(weight) / 2)
            files[fold] = save_fold(models[fold], variant, tmp_path / name / fold)
        return models, files

    return save


@pytest.fixture(name="load_bag")
def load_bag_fixture(tiny_variants, snapshots, params):
    """Return a loader of a method's bag on the CPU: ``load_bag(name, files)``.

    A LoRA bag takes the rank and dropout of the fold models that ``fold_models`` writes.
    """

    def load(name, files):
        variant = tiny_variants[name]
        if name == LORA:
            rank, dropout = params[LORA].lora_rank, params[LORA].lora_dropout
            return LoraBag.load(
                variant, snapshots[name], files, rank=rank, dropout=dropout, device=CPU
            )
        return FullBag.load(variant, snapshots[name], files, device=CPU)

    return load


def bag_batches(model, input_ids, bag):
    """Score the problems with one model, in the batches of a bag."""
    return predict_scores(
        model,
        input_ids,
        bag.tokenizer,
        batch_size=bag.variant.score_batch,
        pad_to_multiple_of=bag.variant.pad_to_multiple_of,
    )


def test_auto_resolves_to_cuda_when_it_is_available_and_to_the_cpu_otherwise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == CPU


def test_any_other_device_setting_is_read_as_a_torch_device_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cpu") == CPU
    assert resolve_device("cuda:1") == torch.device("cuda", 1)


@pytest.mark.parametrize("setting", ["cuda", "cuda:1"])
def test_a_cuda_device_is_refused_when_torch_sees_no_gpu(monkeypatch, setting):
    # The project's own error, before any weight moves: torch's would come at the bag's load.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as refused:
        resolve_device(setting)

    assert str(refused.value) == f"The device {setting!r} was asked for, but torch sees no GPU."


def test_the_tokenizer_has_no_length_limit_of_its_own(tokenizer):
    assert tokenizer.model_max_length == math.inf
    assert len(tokenizer("sum " * 5000, add_special_tokens=False)["input_ids"]) == 5000


def test_a_classifier_with_weights_takes_every_weight_but_the_classifier_from_the_snapshot(
    snapshots,
):
    encoder = load_file(snapshots[LORA] / ENCODER_FILE)

    state = classifier_model(snapshots[LORA], weights=True).state_dict()

    from_snapshot = [key for key in state if not key.startswith("classifier.")]
    assert from_snapshot and set(from_snapshot) <= set(encoder)
    assert all(torch.equal(state[key], encoder[key]) for key in from_snapshot)


def test_a_classifier_without_weights_has_one_output_per_focus_tag_and_random_weights(snapshots):
    encoder = load_file(snapshots[FULL] / ENCODER_FILE)

    model = classifier_model(snapshots[FULL], weights=False, classifier_dropout=0.15)

    assert model.config.num_labels == model.classifier.out_features == len(FOCUS_TAGS)
    assert model.config.id2label == dict(enumerate(FOCUS_TAGS))
    assert model.config.label2id == {tag: index for index, tag in enumerate(FOCUS_TAGS)}
    assert model.config.problem_type == "multi_label_classification"
    assert model.drop.p == 0.15
    embeddings = "model.embeddings.tok_embeddings.weight"
    assert not torch.equal(model.state_dict()[embeddings], encoder[embeddings])


def test_the_lora_config_matches_the_adapter_config_of_notebook_05():
    # The values of experiments/modernbert_lora/adapters/fold_<k>/adapter_config.json, rank 32.
    config = lora_config(VARIANTS[LORA], 32, 0.014207211639577388)

    assert config.task_type == TaskType.SEQ_CLS
    assert (config.r, config.lora_alpha, config.lora_dropout) == (32, 64, 0.014207211639577388)
    assert set(config.target_modules) == {"Wqkv", "Wo", "Wi"}
    assert set(config.modules_to_save) == {"head", "classifier"}
    assert (config.bias, config.use_rslora, config.use_dora) == ("none", False, False)
    assert config.init_lora_weights is True


def test_lora_alpha_is_the_variant_s_alpha_per_rank_times_the_rank():
    assert lora_config(VARIANTS[LORA], 8, 0.0).lora_alpha == 2 * 8


def test_predict_scores_gives_float32_probabilities_in_the_order_of_the_inputs(
    snapshots, tokenizer, input_ids
):
    torch.manual_seed(0)
    model = classifier_model(snapshots[LORA], weights=True)

    scores = predict_scores(model, input_ids, tokenizer, batch_size=4, pad_to_multiple_of=8)

    assert scores.dtype == np.float32
    assert scores.shape == (len(TEXTS), len(FOCUS_TAGS))
    assert ((scores > 0) & (scores < 1)).all()
    assert len(np.unique(scores[:, 0])) == len(TEXTS)  # a row in the wrong place would show
    reversed_scores = predict_scores(
        model, input_ids[::-1], tokenizer, batch_size=4, pad_to_multiple_of=8
    )
    np.testing.assert_allclose(reversed_scores[::-1], scores, rtol=0, atol=1e-5)


def test_predict_scores_equals_the_model_s_own_probabilities_of_each_problem_alone(
    snapshots, tokenizer, input_ids
):
    torch.manual_seed(0)
    model = classifier_model(snapshots[LORA], weights=True).eval()

    scores = predict_scores(model, input_ids, tokenizer, batch_size=4, pad_to_multiple_of=8)

    with torch.no_grad():
        alone = [torch.sigmoid(model(input_ids=torch.tensor([ids])).logits)[0] for ids in input_ids]
    np.testing.assert_allclose(scores, torch.stack(alone).numpy(), rtol=0, atol=1e-5)


@pytest.mark.parametrize("training", [True, False], ids=["train mode", "eval mode"])
def test_predict_scores_runs_in_eval_mode_and_restores_the_model_s_mode(
    snapshots, tokenizer, input_ids, training
):
    model = classifier_model(snapshots[LORA], weights=True).train(training)
    modes = []
    model.register_forward_pre_hook(lambda module, args: modes.append(module.training))

    predict_scores(model, input_ids, tokenizer, batch_size=4, pad_to_multiple_of=8)

    assert modes == [False, False]  # two batches of at most four problems, without dropout
    assert model.training is training


def test_bag_mean_averages_in_float32_then_casts_exactly_to_float64():
    values = (0.1, 0.7, 0.3)
    fold_scores = {
        f"fold_{k}": np.full((2, 3), value, np.float32) for k, value in enumerate(values)
    }

    mean = bag_mean(fold_scores)

    float32_mean = (np.float32(0.1) + np.float32(0.7) + np.float32(0.3)) / np.float32(3)
    float64_mean = sum(float(np.float32(value)) for value in values) / 3
    assert mean.dtype == np.float64
    assert (mean == np.float64(float32_mean)).all()
    assert np.float64(float32_mean) != float64_mean  # so the test tells the two averages apart


def test_a_lora_bag_holds_one_adapter_per_fold_model_the_first_one_active(fold_models, load_bag):
    _, files = fold_models(LORA)

    bag = load_bag(LORA, files)

    assert bag.names == FOLDS
    assert set(bag.model.peft_config) == set(FOLDS)
    assert bag.model.active_adapter == "fold_0"
    assert not bag.model.training
    assert all(weight.device == CPU for weight in bag.model.parameters())


def test_a_full_bag_holds_each_fold_model_s_weights_the_first_one_loaded(fold_models, load_bag):
    _, files = fold_models(FULL)

    bag = load_bag(FULL, files)

    assert bag.names == FOLDS
    assert set(bag.weights) == set(FOLDS)
    assert not bag.model.training
    state = bag.model.state_dict()
    assert all(torch.equal(state[key], tensor) for key, tensor in bag.weights["fold_0"].items())


@pytest.mark.parametrize("name", [LORA, FULL])
def test_a_loaded_bag_reproduces_each_saved_fold_model(fold_models, load_bag, input_ids, name):
    models, files = fold_models(name)
    bag = load_bag(name, files)

    fold_scores = bag.fold_scores(input_ids)

    assert list(fold_scores) == list(FOLDS)
    for fold, model in models.items():
        np.testing.assert_array_equal(fold_scores[fold], bag_batches(model, input_ids, bag))


@pytest.mark.parametrize("name", [LORA, FULL])
def test_switching_fold_models_restores_each_one_exactly(fold_models, load_bag, input_ids, name):
    bag = load_bag(name, fold_models(name)[1])

    first = bag.fold_scores(input_ids, ["fold_0"])["fold_0"]
    other = bag.fold_scores(input_ids, ["fold_1"])["fold_1"]
    again = bag.fold_scores(input_ids, ["fold_0"])["fold_0"]

    np.testing.assert_array_equal(again, first)
    assert not np.allclose(first, other)  # the two fold models differ


@pytest.mark.parametrize("name", [LORA, FULL])
def test_a_bag_scores_the_float32_mean_of_its_fold_models_as_float64(
    fold_models, load_bag, input_ids, name
):
    bag = load_bag(name, fold_models(name)[1])

    scores = bag.scores(input_ids)

    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, bag_mean(bag.fold_scores(input_ids)))


@pytest.fixture(name="bag", params=[LORA, FULL])
def bag_fixture(request, fold_models, load_bag):
    """Return a bag of two random fold models of each method, on the CPU."""
    return load_bag(request.param, fold_models(request.param)[1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_bag_on_the_gpu_scores_one_problem_per_forward_pass(bag, input_ids, monkeypatch, device):
    # One batch of many problems held the GPU at full power and rebooted the machine once.
    batched = bag.fold_scores(input_ids)
    sizes = []

    def recording_predict_scores(model, ids, tokenizer, **batching):
        sizes.append(batching["batch_size"])
        return predict_scores(model, ids, tokenizer, **batching)

    monkeypatch.setattr(modeling, "predict_scores", recording_predict_scores)
    monkeypatch.setattr(bag, "device", torch.device(device))  # the tiny model stays on the CPU

    fold_scores = bag.fold_scores(input_ids)

    expected = 1 if device == "cuda" else bag.variant.score_batch
    assert sizes == [expected] * len(FOLDS)  # every fold model, at that batch size
    for fold in FOLDS:
        np.testing.assert_allclose(fold_scores[fold], batched[fold], rtol=0, atol=1e-5)


def test_a_bag_is_on_the_device_of_its_model(bag):
    assert bag.device == CPU


def test_a_lora_file_holding_a_tensor_the_adapter_lacks_is_refused(fold_models, load_bag):
    _, files = fold_models(LORA)
    tensors = load_file(files["fold_1"])
    tensors["base_model.model.classifier.extra"] = torch.zeros(3)
    save_file(tensors, str(files["fold_1"]))

    with pytest.raises(
        ValueError, match=r"does not hold the adapter's tensors: unexpected .*classifier\.extra"
    ):
        load_bag(LORA, files)


def test_a_lora_file_missing_one_of_the_adapters_tensors_is_refused(fold_models, load_bag):
    _, files = fold_models(LORA)
    tensors = load_file(files["fold_1"])
    del tensors[next(key for key in tensors if "lora_A" in key)]
    save_file(tensors, str(files["fold_1"]))

    with pytest.raises(
        ValueError, match=r"does not hold the adapter's tensors: .*missing .*lora_A\.fold_1\.weight"
    ):
        load_bag(LORA, files)


def test_a_lora_file_of_another_rank_fails_to_load(fold_models, tiny_variants, snapshots):
    _, files = fold_models(LORA)  # rank 4

    with pytest.raises(RuntimeError, match="size mismatch"):
        LoraBag.load(tiny_variants[LORA], snapshots[LORA], files, rank=8, dropout=0.0, device=CPU)


def test_a_full_bag_refuses_weights_that_are_not_a_whole_model(fold_models, load_bag):
    _, files = fold_models(LORA)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_bag(FULL, files)


def test_a_bag_that_cannot_switch_fold_models_cannot_score(tiny_variants, tokenizer, input_ids):
    bag = _Bag(FOLDS, tokenizer, tiny_variants[LORA])

    with pytest.raises(NotImplementedError):
        bag.scores(input_ids)
