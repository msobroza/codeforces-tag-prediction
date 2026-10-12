"""Tests for SetFit: the stratified pairs, the LoRA adapter, its merge and files, the estimator."""

import dataclasses
import json
import math
import os

import numpy as np
import pandas as pd
import peft
import pytest
import torch
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.conformance import starts_like_pickle
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.methods.f2llm import setfit
from codeforces_tags.methods.f2llm.embedding import (
    SOLUTION,
    STATEMENT,
    embedding_columns,
    encoder_inputs,
    live_feature_frame,
)
from codeforces_tags.methods.f2llm.encoder import (
    Encoder,
    frozen_encoder,
    load_model,
    load_tokenizer,
)
from codeforces_tags.methods.f2llm.setfit import (
    ADAPTER_FILE,
    ADAPTER_RECORD_FILE,
    ADAPTER_RECORD_KEYS,
    CUBLAS_WORKSPACE_CONFIG,
    SetFitEstimator,
    adapter_cache_path,
    adapter_record,
    load_adapter,
    lora_config,
    merged_model,
    sample_pairs,
    save_adapter,
    train_adapter,
)
from codeforces_tags.methods.f2llm.settings import LORA_TARGETS, PROBE_GRID, SETFIT, SETFIT_METHOD
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.pretrained import pretrained_root

# SETFIT scaled down to the tiny model: 2 + 2 pairs per tag, 8 steps of 4 pairs in micro-batches
# of 2, and a learning rate at which a few steps visibly lower the loss.
TINY_SETFIT = SETFIT | {
    "pairs_per_tag_positive": 2,
    "batch_pairs": 4,
    "micro_batch_pairs": 2,
    "learning_rate": 5e-3,
}
# Three tags held by disjoint problems, and two problems without a focus tag: each positive pair
# shares exactly one tag, and each negative pair's first problem holds exactly one.
DISJOINT = np.array(
    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
)


@pytest.fixture(name="problems")
def problems_fixture(synthetic_raw):
    """Return the synthetic problems, preprocessed."""
    return raw_to_problems(synthetic_raw)


@pytest.fixture(name="labels")
def labels_fixture(problems):
    """Return the synthetic problems' 0/1 focus labels, an int64 array."""
    return focus_labels(problems).to_numpy(dtype=np.int64)


@pytest.fixture(name="texts")
def texts_fixture(problems, tiny_config):
    """Return the synthetic problems' statement encoder inputs."""
    return encoder_inputs(problems, tiny_config)[STATEMENT]


@pytest.fixture(name="experiments")
def experiments_fixture(tmp_path, monkeypatch):
    """Point SetFit's experiments folder into ``tmp_path``: no test writes into experiments/."""
    root = tmp_path / "experiments"
    monkeypatch.setattr(setfit, "experiments_dir", lambda name: root / name)
    return root


def random_adapter(record):
    """Return a LoRA adapter state for the tiny model with small random tensors (no training)."""
    base = load_model(record.directory(), "cpu")
    model = get_peft_model(base, lora_config(adapter_record(SETFIT)))
    generator = torch.Generator().manual_seed(0)
    return {
        name: 0.1 * torch.randn(tensor.shape, generator=generator)
        for name, tensor in get_peft_model_state_dict(model).items()
    }


def pair_loss(encoder, texts, pairs):
    """Return the mean squared error between each pair's cosine and its target."""
    vectors = encoder.encode(texts)
    cosine = np.sum(vectors[pairs[:, 0]] * vectors[pairs[:, 1]], axis=1)
    return float(np.mean((cosine - pairs[:, 2]) ** 2))


def test_sample_pairs_draws_the_same_number_of_pairs_of_each_kind_for_each_tag():
    pairs = sample_pairs(DISJOINT, 2, seed=0)

    assert pairs.dtype == np.int64
    assert pairs.shape == (12, 3)  # 3 tags x (2 positive + 2 negative)
    assert pairs[:, :2].min() >= 0 and pairs[:, :2].max() < len(DISJOINT)  # rows of labels only
    first, second, target = DISJOINT[pairs[:, 0]], DISJOINT[pairs[:, 1]], pairs[:, 2]
    for tag in range(3):
        assert np.sum((target == 1) & (first[:, tag] == 1) & (second[:, tag] == 1)) == 2
        assert np.sum((target == 0) & (first[:, tag] == 1)) == 2


def test_positive_pairs_share_a_tag_and_negative_pairs_share_none():
    labels = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])

    pairs = sample_pairs(labels, 50, seed=0)

    first, second, target = labels[pairs[:, 0]], labels[pairs[:, 1]], pairs[:, 2]
    shared = np.sum(first * second, axis=1)
    assert np.all(shared[target == 1] > 0)
    assert np.all(shared[target == 0] == 0)
    assert np.all(first.sum(axis=1) > 0)  # never two problems without a focus tag
    assert np.all(pairs[:, 0] != pairs[:, 1])
    # Problem 0 (tags 0 and 1) only meets problems that hold neither: 3 and the two untagged.
    partners = {int(row[1]) for row in pairs if row[0] == 0 and row[2] == 0}
    assert partners == {3, 4, 5}


def test_sample_pairs_draws_without_replacement_until_every_pair_is_drawn():
    pairs = sample_pairs(DISJOINT, 2, seed=0)

    positives = [tuple(row[:2]) for row in pairs if row[2] == 1]
    tag_0 = [pair for pair in positives if DISJOINT[pair[0], 0]]
    assert len(set(tag_0)) == 2  # 6 possible pairs: 2 distinct ones
    tag_2 = [pair for pair in positives if DISJOINT[pair[0], 2]]
    assert tag_2 == [(7, 8), (7, 8)]  # 1 possible pair: drawn again once exhausted


def test_sample_pairs_skips_a_kind_with_no_possible_pair():
    lone = sample_pairs(np.array([[1, 0], [0, 0]]), 3, seed=0)
    assert lone[:, 2].tolist() == [0, 0, 0]  # one problem with tag 0: no positive pair
    assert np.all(lone[:, :2] == [0, 1])

    everyone = sample_pairs(np.array([[1], [1], [1]]), 2, seed=0)
    assert everyone[:, 2].tolist() == [1, 1]  # every problem shares the tag: no negative pair


def test_sample_pairs_is_reproducible_with_its_seed():
    np.testing.assert_array_equal(sample_pairs(DISJOINT, 2, 0), sample_pairs(DISJOINT, 2, 0))
    assert not np.array_equal(sample_pairs(DISJOINT, 2, 0), sample_pairs(DISJOINT, 2, 1))


def test_sample_pairs_refuses_labels_that_are_not_a_0_1_matrix_and_no_pair_per_tag():
    with pytest.raises(ValueError, match="2-D 0/1 array"):
        sample_pairs(np.array([1, 0, 1]), 2, 0)
    with pytest.raises(ValueError, match="2-D 0/1 array"):
        sample_pairs(np.array([[2, 0], [1, 0]]), 2, 0)
    with pytest.raises(ValueError, match="at least 1"):
        sample_pairs(DISJOINT, 0, 0)


def test_adapter_record_holds_explicit_fields_and_builds_the_lora_config():
    record = adapter_record(SETFIT)

    assert record == {
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "peft_version": peft.__version__,
        "r": 8,
        "target_modules": sorted(LORA_TARGETS),
    }
    assert tuple(sorted(record)) == ADAPTER_RECORD_KEYS
    config = lora_config(record)
    assert (config.r, config.lora_alpha, config.lora_dropout, config.bias) == (8, 16, 0.0, "none")
    assert set(config.target_modules) == set(LORA_TARGETS)


def test_lora_config_refuses_peft_s_own_configuration():
    peft_config = lora_config(adapter_record(SETFIT)).to_dict()

    with pytest.raises(ValueError, match="build it with adapter_record"):
        lora_config(peft_config)


def test_train_adapter_lowers_the_loss_of_its_pairs(tiny_pretrained, tiny_config, texts, labels):
    messages = []

    state = train_adapter(
        tiny_pretrained, tiny_config, texts, labels, TINY_SETFIT, device="cpu", log=messages.append
    )

    pairs = sample_pairs(labels, 2, TINY_SETFIT["seed"])  # the pairs it trained on
    frozen = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    body = merged_model(tiny_pretrained, state, adapter_record(TINY_SETFIT), "cpu")
    tuned = Encoder(tokenizer=frozen.tokenizer, model=body, config=tiny_config, device="cpu")
    assert pair_loss(tuned, texts, pairs) < pair_loss(frozen, texts, pairs)
    steps = math.ceil(len(pairs) / TINY_SETFIT["batch_pairs"])
    assert messages[0] == f"SetFit: {len(pairs)} pairs, {steps} steps on cpu."
    assert len(messages) == 1 + steps  # under 20 steps, every step logs a line
    assert messages[-1].startswith(f"SetFit: step {steps}/{steps}, loss ")
    assert all("lora_" in name for name in state)
    assert {name.split(".")[-3] for name in state} == set(LORA_TARGETS)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in state.values())


def test_train_adapter_is_reproducible_with_its_seed(tiny_pretrained, tiny_config, texts, labels):
    def train(settings):
        return train_adapter(
            tiny_pretrained, tiny_config, texts, labels, settings, device="cpu", log=lambda _: None
        )

    first, again = train(TINY_SETFIT), train(TINY_SETFIT)
    other = train(TINY_SETFIT | {"seed": 1})

    assert all(torch.equal(first[name], again[name]) for name in first)
    assert not all(torch.equal(first[name], other[name]) for name in first)


def test_train_adapter_trains_with_deterministic_algorithms_only(
    tiny_pretrained, tiny_config, texts, labels, monkeypatch
):
    # setenv records the value to restore, even an absent one, which delenv alone would not.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    one_pair_steps = TINY_SETFIT | {"batch_pairs": 1, "micro_batch_pairs": 1}
    steps = len(sample_pairs(labels, 2, TINY_SETFIT["seed"]))
    modes = []

    train_adapter(
        tiny_pretrained,
        tiny_config,
        texts,
        labels,
        one_pair_steps,
        device="cpu",
        log=lambda _: modes.append(torch.are_deterministic_algorithms_enabled()),
    )

    assert all(modes)
    assert 1 < len(modes) <= 12 < steps  # the first line, then one per tenth of the steps
    assert not torch.are_deterministic_algorithms_enabled()  # the previous mode, restored
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_train_adapter_trains_again_with_eager_attention_when_an_op_refuses_determinism(
    tiny_pretrained, tiny_config, texts, labels, monkeypatch
):
    attentions, messages = [], []
    train = setfit._train  # pylint: disable=protected-access

    def refusing_train(loss, *args):
        config = loss.model.config
        attentions.append(config._attn_implementation)  # pylint: disable=protected-access
        if len(attentions) == 1:  # as torch refuses an op without a deterministic mode
            raise RuntimeError("an op does not have a deterministic implementation, but you set")
        return train(loss, *args)

    monkeypatch.setattr(setfit, "_train", refusing_train)

    state = train_adapter(
        tiny_pretrained, tiny_config, texts, labels, TINY_SETFIT, device="cpu", log=messages.append
    )

    assert attentions == ["sdpa", "eager"]
    assert messages[0].startswith("SetFit: training again with eager attention: an op")
    assert {name.split(".")[-3] for name in state} == set(LORA_TARGETS)
    assert not torch.are_deterministic_algorithms_enabled()  # restored after both attempts


def test_train_adapter_lets_any_other_error_through(
    tiny_pretrained, tiny_config, texts, labels, monkeypatch
):
    def failing_train(*_):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(setfit, "_train", failing_train)

    with pytest.raises(RuntimeError, match="out of memory"):
        train_adapter(tiny_pretrained, tiny_config, texts, labels, TINY_SETFIT, device="cpu")


def test_train_adapter_refuses_inputs_that_do_not_end_with_the_end_of_sequence_token(
    tiny_pretrained, tiny_config, texts, labels, monkeypatch
):
    # A configuration whose end-of-sequence id is not the token the tokenizer appends, past the
    # check of load_tokenizer.
    monkeypatch.setattr(
        setfit,
        "load_tokenizer",
        lambda directory, config: AutoTokenizer.from_pretrained(directory, local_files_only=True),
    )
    wrong_end = dataclasses.replace(tiny_config, eos_token_id=tiny_config.eos_token_id + 1)

    with pytest.raises(ValueError, match="end-of-sequence token"):
        train_adapter(tiny_pretrained, wrong_end, texts, labels, TINY_SETFIT, device="cpu")
    assert not torch.are_deterministic_algorithms_enabled()


def test_train_adapter_refuses_texts_and_labels_of_different_lengths(
    tiny_pretrained, tiny_config, texts, labels
):
    with pytest.raises(ValueError, match="texts but"):
        train_adapter(tiny_pretrained, tiny_config, texts[:-1], labels, TINY_SETFIT, device="cpu")


def test_merged_model_encodes_like_the_adapter_it_merged(tiny_pretrained, tiny_config, texts):
    state = random_adapter(tiny_pretrained)
    directory = tiny_pretrained.directory()
    tokenizer = load_tokenizer(directory, tiny_config)
    unmerged = get_peft_model(load_model(directory, "cpu"), lora_config(adapter_record(SETFIT)))
    set_peft_model_state_dict(unmerged, state)
    unmerged.eval()

    merged = merged_model(tiny_pretrained, state, adapter_record(SETFIT), "cpu")

    def encodings(model):
        return Encoder(tokenizer=tokenizer, model=model, config=tiny_config, device="cpu").encode(
            texts
        )

    np.testing.assert_allclose(encodings(merged), encodings(unmerged), rtol=0, atol=1e-5)
    base = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    assert type(merged) is type(base.model)
    assert not any("lora" in name for name, _ in merged.named_modules())
    assert not merged.training
    assert not np.allclose(encodings(merged), base.encode(texts), atol=1e-4)


def test_merged_model_refuses_an_adapter_that_does_not_fit(tiny_pretrained):
    state = random_adapter(tiny_pretrained)
    lacking = dict(list(state.items())[1:])
    extra = state | {"base_model.model.layers.9.mlp.up_proj.lora_A.weight": torch.zeros(8, 32)}

    for wrong in (lacking, extra):
        with pytest.raises(ValueError, match="does not fit the base model"):
            merged_model(tiny_pretrained, wrong, adapter_record(SETFIT), "cpu")


def test_save_adapter_then_load_adapter_round_trips_without_a_path(tmp_path, tiny_pretrained):
    state = random_adapter(tiny_pretrained)
    folder = tmp_path / "model"
    folder.mkdir()

    save_adapter(folder, state, adapter_record(SETFIT))

    assert sorted(path.name for path in folder.iterdir()) == [ADAPTER_RECORD_FILE, ADAPTER_FILE]
    text = (folder / ADAPTER_RECORD_FILE).read_text(encoding="utf-8")
    assert json.loads(text) == adapter_record(SETFIT)
    assert str(tmp_path) not in text
    assert not any(starts_like_pickle(path) for path in folder.iterdir())
    loaded, record = load_adapter(folder)
    assert record == adapter_record(SETFIT)
    assert sorted(loaded) == sorted(state)
    assert all(torch.equal(loaded[name], state[name]) for name in state)


def test_save_and_load_adapter_refuse_a_record_with_other_keys(tmp_path, tiny_pretrained):
    state = random_adapter(tiny_pretrained)
    peft_config = {"base_model_name_or_path": str(tmp_path)} | adapter_record(SETFIT)

    with pytest.raises(ValueError, match="build it with adapter_record"):
        save_adapter(tmp_path, state, peft_config)
    save_adapter(tmp_path, state, adapter_record(SETFIT))
    (tmp_path / ADAPTER_RECORD_FILE).write_text(json.dumps(peft_config), encoding="utf-8")
    with pytest.raises(ValueError, match="build it with adapter_record"):
        load_adapter(tmp_path)


def test_adapter_cache_path_hashes_the_sample_ids_in_order_and_the_settings(experiments):
    path = adapter_cache_path([3, 1, 2], SETFIT)

    assert path.parent == experiments / SETFIT_METHOD / "adapters"
    assert path.suffix == ".safetensors"
    assert path == adapter_cache_path([3, 1, 2], dict(SETFIT))
    assert path != adapter_cache_path([1, 2, 3], SETFIT)  # the pairs are drawn by position
    assert path != adapter_cache_path([3, 1, 2], SETFIT | {"seed": 1})


def test_the_pair_loss_pools_each_input_as_the_encoder_does(tiny_pretrained, tiny_config, texts):
    # Every synthetic input is cut at max_length tokens. Three are shortened (6, 8 and 10 words:
    # the instruction, then the statement's first words), so that a batch pads them on the right;
    # each pair holds a long input and a short one.
    short = [
        " ".join(text.split()[:words]) for text, words in zip(texts[3:6], (6, 8, 10), strict=True)
    ]
    inputs = texts[:3] + short
    pairs = np.array([[0, 3, 1], [4, 1, 0], [5, 2, 0]])
    frozen = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    pair_loss_of = setfit._PairLoss  # pylint: disable=protected-access
    loss = pair_loss_of(frozen.model, frozen.tokenizer, inputs, tiny_config, "cpu")

    assert float(loss(pairs).detach()) == pytest.approx(pair_loss(frozen, inputs, pairs), abs=1e-6)


# The probe's configuration that SetFit's head takes: the grid's first head at C = 1, with the
# solution embedding at weight 0.5 (HEAD_PARAMS) or left out (STATEMENT_ONLY).
HEAD_PARAMS = PROBE_GRID[0] | {"c": 1.0, "solution_weight": 0.5}
STATEMENT_ONLY = HEAD_PARAMS | {"solution_weight": 0.0}


@pytest.fixture(name="frame")
def frame_fixture(synthetic_raw, tiny_pretrained, tiny_config):
    """Return the synthetic problems' feature frame, embedded by the frozen tiny encoder."""
    encoder = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    return live_feature_frame(synthetic_raw, encoder, encoder)


@pytest.fixture(name="make_setfit")
def make_setfit_fixture(tiny_pretrained, tiny_config):
    """Return a builder of unfitted SetFitEstimators of the tiny model on CPU.

    Its keyword arguments override the parameters: the head of ``HEAD_PARAMS``, ``TINY_SETFIT``,
    and no resume file unless ``resume=True``.
    """

    def make(**overrides):
        params = {
            "head_params": HEAD_PARAMS,
            "settings": TINY_SETFIT,
            "device": "cpu",
            "pretrained": tiny_pretrained.to_dict(),
            "encoder_config": tiny_config.to_dict(),
            "resume": False,
        }
        return SetFitEstimator(**(params | overrides))

    return make


def test_setfit_estimator_defaults_to_the_fixed_settings_on_the_gpu():
    assert SetFitEstimator().get_params() == {
        "device": "cuda",
        "encoder_config": None,
        "head_params": None,
        "pretrained": None,
        "resume": True,
        "settings": None,
    }


def test_fit_needs_the_probe_s_head_params(frame, labels, make_setfit):
    with pytest.raises(ValueError, match="needs head_params"):
        make_setfit(head_params=None).fit(frame, labels)


def test_a_fitted_setfit_estimator_follows_the_score_estimator_contract(
    frame, labels, make_setfit, experiments
):
    fitted = make_setfit().fit(frame, labels)

    scores = fitted.decision_function(frame)
    assert fitted.classes_.tolist() == list(range(8))
    assert scores.dtype == np.float64
    assert scores.shape == (len(frame), 8)
    np.testing.assert_array_equal(fitted.predict(frame), (scores >= 0).astype(int))
    assert all(tensor.device.type == "cpu" for tensor in fitted.adapter_state_.values())
    assert not experiments.exists()  # resume=False neither reads nor writes adapters


def test_fit_merges_the_adapter_before_it_re_encodes_the_statements(
    frame, labels, texts, tiny_pretrained, make_setfit
):
    fitted = make_setfit().fit(frame, labels)

    encoder = fitted.statement_encoder_
    assert not any("lora" in name for name, _ in encoder.model.named_modules())
    merged = merged_model(
        tiny_pretrained, fitted.adapter_state_, adapter_record(TINY_SETFIT), "cpu"
    )
    expected = Encoder(
        tokenizer=encoder.tokenizer, model=merged, config=encoder.config, device="cpu"
    ).encode(texts)
    tuned = encoder.encode(texts)
    np.testing.assert_allclose(tuned, expected, rtol=0, atol=1e-6)
    statement = embedding_columns(STATEMENT, encoder.width)
    assert not np.allclose(tuned, frame[statement].to_numpy(), atol=1e-4)  # not the frozen body
    head_rows = pd.DataFrame(tuned.astype(np.float64), index=frame.index, columns=statement)
    head_rows = head_rows.join(frame[embedding_columns(SOLUTION, encoder.width)])
    np.testing.assert_allclose(
        fitted.decision_function(frame), fitted.head_.decision_function(head_rows), atol=1e-9
    )


@pytest.mark.usefixtures("experiments")
def test_the_head_reads_the_frozen_solution_embedding_only_when_the_probe_kept_it(
    frame, labels, make_setfit
):
    with_solution = make_setfit(resume=True).fit(frame, labels)
    statement_only = make_setfit(head_params=STATEMENT_ONLY, resume=True).fit(frame, labels)

    width = with_solution.statement_encoder_.width
    statement, solution = embedding_columns(STATEMENT, width), embedding_columns(SOLUTION, width)
    assert list(with_solution.head_.feature_names_in_) == statement + solution
    assert list(statement_only.head_.feature_names_in_) == statement
    without_solution = frame.drop(columns=solution)
    assert statement_only.decision_function(without_solution).shape == (len(frame), 8)
    with pytest.raises(ValueError, match=f"lack {width} embedding columns"):
        with_solution.decision_function(without_solution)


def test_a_second_fit_of_the_same_rows_reloads_the_adapter_instead_of_training(
    frame, labels, make_setfit, experiments, monkeypatch
):
    calls = []

    def counting_train_adapter(*args, **kwargs):
        calls.append(len(args[2]))  # the texts it trains on
        return train_adapter(*args, **kwargs)

    monkeypatch.setattr(setfit, "train_adapter", counting_train_adapter)

    first = make_setfit(resume=True).fit(frame, labels)
    again = make_setfit(resume=True).fit(frame, labels)

    assert calls == [len(frame)]
    path = adapter_cache_path(frame.index, TINY_SETFIT)
    assert path.parent == experiments / SETFIT_METHOD / "adapters"
    assert sorted(file.name for file in path.parent.iterdir()) == [path.name]
    names = first.adapter_state_
    assert all(torch.equal(first.adapter_state_[n], again.adapter_state_[n]) for n in names)
    np.testing.assert_array_equal(first.decision_function(frame), again.decision_function(frame))
    make_setfit(resume=True).fit(frame.iloc[1:], labels[1:])
    assert calls == [len(frame), len(frame) - 1]  # other rows: another adapter


@pytest.mark.usefixtures("experiments")
def test_a_resumed_fit_refuses_an_adapter_trained_with_other_encoder_settings(
    frame, labels, tiny_config, make_setfit
):
    make_setfit(resume=True).fit(frame, labels)
    other = tiny_config.to_dict() | {"statement_instruction": "query "}

    with pytest.raises(ValueError, match="delete it"):
        make_setfit(encoder_config=other, resume=True).fit(frame, labels)


def test_a_setfit_fit_leaves_the_memoised_frozen_encoder_unchanged(
    frame, labels, tiny_pretrained, tiny_config, make_setfit
):
    frozen = frozen_encoder(tiny_pretrained, tiny_config, "cpu")
    before = {name: tensor.detach().clone() for name, tensor in frozen.model.named_parameters()}

    fitted = make_setfit().fit(frame, labels)

    assert frozen_encoder(tiny_pretrained, tiny_config, "cpu").model is frozen.model
    assert fitted.statement_encoder_.model is not frozen.model
    assert not type(frozen.model).__module__.startswith("peft")
    assert not any("lora" in name for name, _ in frozen.model.named_modules())
    after = dict(frozen.model.named_parameters())
    assert sorted(after) == sorted(before)
    assert all(torch.equal(after[name], before[name]) for name in before)


def test_a_fitted_adapter_saves_without_an_absolute_path_or_a_pickle(
    tmp_path, frame, labels, make_setfit, experiments
):
    fitted = make_setfit(resume=True).fit(frame, labels)
    folder = tmp_path / "model"
    folder.mkdir()

    save_adapter(folder, fitted.adapter_state_, adapter_record(TINY_SETFIT))

    text = (folder / ADAPTER_RECORD_FILE).read_text(encoding="utf-8")
    assert str(tmp_path) not in text  # the method folder's own path
    assert str(pretrained_root()) not in text  # the base weights' root
    assert tuple(json.loads(text)) == ADAPTER_RECORD_KEYS
    written = [path for path in [*folder.rglob("*"), *experiments.rglob("*")] if path.is_file()]
    assert sorted(path.suffix for path in written) == [".json", ".safetensors", ".safetensors"]
    assert not any(starts_like_pickle(path) for path in written)


def test_setfit_estimator_runs_inside_cross_fitted_tag_thresholds(
    synthetic_raw, frame, labels, make_setfit, experiments
):
    tags = pd.DataFrame(labels, index=frame.index, columns=list(FOCUS_TAGS))

    fitted = CrossFittedTagThresholds(make_setfit(resume=True), refit=True).fit(
        frame, tags, groups=synthetic_raw["fold"]
    )

    assert fitted.oof_scores_.shape == (len(frame), 8)
    assert fitted.decision_function(frame).shape == (len(frame), 8)
    adapters = list((experiments / SETFIT_METHOD / "adapters").iterdir())
    assert len(adapters) == 6  # one per training fold, and the refit's
