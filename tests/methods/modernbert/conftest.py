"""Fixtures of the ModernBERT methods' tests: a tiny random ModernBERT, offline, on the CPU.

The tiny snapshot (a word-level tokenizer and a 2-layer ModernBERT of hidden size 32) is saved
once per session at both methods' pinned repository and revision, inside a temporary pretrained
cache. The ``caches`` fixture points ``$CODEFORCES_PRETRAINED_DIR`` at it and
``$CODEFORCES_TRAINED_DIR`` at an empty folder, hides CUDA and unsets both device variables, so
that the isolation subprocess of ``check_tagging_method``, which loads the method through the
real registry and settings, finds the same snapshot and runs on the CPU too. In process,
``tiny_variants`` replaces both methods' settings with ones that train in seconds.
"""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast

from codeforces_tags.config import (
    DEVICE_ENV_VAR,
    FOCUS_TAGS,
    LEGACY_DEVICE_ENV_VAR,
    PRETRAINED_DIR_ENV_VAR,
    TRAINED_DIR_ENV_VAR,
)
from codeforces_tags.methods.modernbert.method import FineTunedBag
from codeforces_tags.methods.modernbert.settings import FOLD_WEIGHTS_FILE, FULL, LORA, VARIANTS
from codeforces_tags.methods.modernbert.training import (
    FullParams,
    LoraParams,
    build_training_model,
    save_fold,
)
from codeforces_tags.methods.modernbert.weights import (
    pretrained_weight_list,
    snapshot_folder,
    trained_weight_list,
)

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}
# The words the tiny tokenizer knows; any other word is [UNK].
WORDS = [
    "a",
    "tree",
    "graph",
    "string",
    "number",
    "prime",
    "sum",
    "count",
    "print",
    "input",
    "int",
    "n",
    "m",
    "k",
    "i",
    "for",
    "in",
    "range",
    "if",
    "else",
    "return",
    "def",
    "while",
    "gcd",
    "the",
    "of",
    "and",
    "is",
    "to",
    "edges",
    "vertices",
    "nodes",
    "path",
    "cycle",
    "game",
    "win",
    "lose",
    "probability",
    "circle",
    "point",
    "angle",
]
# Settings of the trials and fold models that train in the tests.
LORA_PARAMS = LoraParams(
    learning_rate=1e-3,
    lora_rank=4,
    lora_dropout=0.0,
    epochs=1,
    batch_size=4,
    warmup_ratio=0.0,
    pos_weight_power=0.5,
)
FULL_PARAMS = FullParams(
    learning_rate=1e-3,
    weight_decay=0.01,
    classifier_dropout=0.0,
    epochs=1,
    batch_size=4,
    warmup_ratio=0.0,
    pos_weight_power=0.5,
)
PARAMS = {LORA: LORA_PARAMS, FULL: FULL_PARAMS}


def save_tiny_snapshot(folder):
    """Save a tiny tokenizer and a tiny random ModernBERT as a pretrained snapshot.

    Args:
        folder: The snapshot's folder, created.
    """
    folder.mkdir(parents=True)
    vocab = {token: i for i, token in enumerate([*SPECIAL_TOKENS.values(), *WORDS])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **SPECIAL_TOKENS)
    fast.save_pretrained(folder)
    # transformers 5 keeps the special tokens in tokenizer_config.json; the Hub's snapshots also
    # hold this file, which pretrained.json pins.
    (folder / "special_tokens_map.json").write_text(json.dumps(SPECIAL_TOKENS), encoding="utf-8")
    config = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=2048,
        pad_token_id=vocab["[PAD]"],
        cls_token_id=vocab["[CLS]"],
        sep_token_id=vocab["[SEP]"],
        bos_token_id=vocab["[CLS]"],
        eos_token_id=vocab["[SEP]"],
    )
    torch.manual_seed(0)
    ModernBertForMaskedLM(config).save_pretrained(folder)


@pytest.fixture(name="params")
def params_fixture():
    """Return the configuration that trains in the tests, by method name (``PARAMS``)."""
    return PARAMS


@pytest.fixture(name="pretrained_cache", scope="session")
def pretrained_cache_fixture(tmp_path_factory):
    """Return a pretrained cache holding the tiny snapshot at both methods' pinned revisions."""
    cache = tmp_path_factory.mktemp("pretrained")
    for variant in VARIANTS.values():
        save_tiny_snapshot(cache / variant.model_repo.replace("/", "--") / variant.model_revision)
    return cache


@pytest.fixture(name="caches")
def caches_fixture(monkeypatch, tmp_path, pretrained_cache):
    """Point both weight caches at temporary folders, and hide CUDA from subprocesses.

    Neither device variable is set, whatever the shell exports: a method loaded without a device
    takes its own ``device`` setting, in process and in the subprocesses alike, and a test of the
    variable sets it itself.

    Returns:
        ``{"pretrained": <the tiny snapshots' cache>, "trained": <an empty trained cache>}``.
    """
    trained = tmp_path / "trained"
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(pretrained_cache))
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(trained))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_DEVICE_ENV_VAR, raising=False)
    return {"pretrained": pretrained_cache, "trained": trained}


@pytest.fixture(name="tiny_variants")
def tiny_variants_fixture(monkeypatch):
    """Replace both methods' settings with ones that train and tune in seconds, on the CPU.

    Returns:
        The tiny settings, by method name (also in ``VARIANTS`` for the test).
    """
    tiny = {}
    for name, variant in VARIANTS.items():
        tiny[name] = replace(
            variant,
            max_length=48,
            code_min_tokens=8,
            micro_batch=2,
            score_batch=4,
            subfolds=2,
            tuning_trials=2,
            startup_trials=1,
            pruner_min_peers=1,
            max_failed_trials=1,
            collapse_ap=0.0,
            collapse_sample=8,
            device="cpu",
        )
        monkeypatch.setitem(VARIANTS, name, tiny[name])
    return tiny


@pytest.fixture(name="statements")
def statements_fixture():
    """Return one statement per focus tag, made of words the tiny tokenizer knows."""
    return {
        "math": "count the sum of a number",
        "graphs": "a graph of vertices and edges",
        "strings": "print a string",
        "number theory": "the gcd of a prime number",
        "trees": "a tree of nodes",
        "geometry": "a circle and a point angle",
        "games": "a game to win or lose",
        "probabilities": "the probability to win",
    }


@pytest.fixture(name="tiny_problems")
def tiny_problems_fixture(statements):
    """Return eight raw problems, one per focus tag, as ``read_samples()`` gives them."""
    tags = list(FOCUS_TAGS)
    return pd.DataFrame(
        {
            "prob_desc_description": [statements[tag] for tag in tags],
            "prob_desc_input_spec": ["n m k"] * len(tags),
            "prob_desc_output_spec": ["print int"] * len(tags),
            "prob_desc_notes": [np.nan] * len(tags),
            "prob_desc_sample_inputs": ['["1"]'] * len(tags),
            "prob_desc_sample_outputs": ['["2"]'] * len(tags),
            "source_code": [f"n = int ( input ( ) )\nprint ( n ) # {tag}" for tag in tags],
            "tags": [[tag] for tag in tags],
        },
        index=pd.Index(range(101, 101 + len(tags)), name="sample_id"),
    )


@pytest.fixture(name="save_fold_models")
def save_fold_models_fixture(caches, tiny_variants):
    """Return a writer of random fold models into the trained cache, and their weight list.

    ``save_fold_models(name, folds=("fold_0", "fold_1"))`` builds each fold model as training
    would, moves every trainable weight at random (LoRA's B matrices start at zero), saves it with
    ``save_fold`` and returns ``trained_weight_list``.
    """

    def save(name, folds=("fold_0", "fold_1")):
        variant = tiny_variants[name]
        snapshot = snapshot_folder(variant.model_repo, variant.model_revision)
        files = {}
        for seed, fold in enumerate(folds):
            torch.manual_seed(seed)
            model = build_training_model(variant, PARAMS[name], snapshot)
            with torch.no_grad():
                for weight in model.parameters():
                    if weight.requires_grad:
                        weight.add_(0.5 * torch.randn_like(weight))
            save_fold(model, variant, caches["trained"] / name / fold)
            files[fold] = f"{fold}/{FOLD_WEIGHTS_FILE}"
        return trained_weight_list(name, files)

    return save


@pytest.fixture(name="make_bag")
def make_bag_fixture(save_fold_models, tiny_variants):
    """Return a builder of a method over random fold models: ``make_bag(name, thresholds=0.5)``."""

    def make(name, thresholds=0.5):
        variant = tiny_variants[name]
        return FineTunedBag.build(
            variant,
            asdict(PARAMS[name]),
            np.full(len(FOCUS_TAGS), thresholds, dtype=np.float64),
            pretrained_weight_list(
                variant.model_repo, variant.model_revision, variant.pretrained_files
            ),
            save_fold_models(name),
        )

    return make
