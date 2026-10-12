"""SetFit: contrastive LoRA tuning of the encoder on statement pairs, then a linear head.

SetFit's method (Tunstall et al., 2022), written here rather than taken from the ``setfit``
library, which saves its head with joblib (a pickle) and samples its pairs uniformly:

- ``sample_pairs``: statement pairs stratified by focus tag, as SetFit balances its pairs by
  class;
- ``train_adapter``: a LoRA adapter on a fresh copy of the base weights, trained so that the
  cosine of a pair's embeddings approaches its target (sentence-transformers'
  ``CosineSimilarityLoss``, reimplemented);
- ``merged_model``: the base weights with an adapter merged in, the body that encodes
  statements;
- ``save_adapter`` and ``load_adapter``: the adapter's tensors as safetensors, its settings as
  explicit JSON;
- ``adapter_cache_path``: where a fit saves its adapter under ``experiments/``, so that a rerun
  of the same fit reloads it instead of training (a fit-time option, like the array cache);
- ``SetFitEstimator``: the score estimator of ``f2llm_setfit``, which
  ``CrossFittedTagThresholds`` wraps.

It is on the prediction path of ``f2llm_setfit``: it imports torch, transformers and peft, never
mlflow or optuna.
"""

from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits
from transformers import get_linear_schedule_with_warmup

from codeforces_tags.config import TUNING_BLAS_THREADS
from codeforces_tags.methods.f2llm.embedding import (
    SOLUTION,
    STATEMENT,
    embedding_columns,
    encoder_inputs,
)
from codeforces_tags.methods.f2llm.encoder import (
    Encoder,
    EncoderConfig,
    PretrainedRecord,
    check_pretrained,
    load_model,
    load_tokenizer,
    read_json,
    resolve_device,
    write_json,
)
from codeforces_tags.methods.f2llm.estimators import make_probe
from codeforces_tags.methods.f2llm.settings import (
    LORA_TARGETS,
    SETFIT,
    SETFIT_METHOD,
    experiments_dir,
)
from codeforces_tags.provenance import sha256_json

ADAPTER_FILE = "adapter.safetensors"
ADAPTER_RECORD_FILE = "adapter.json"
# The keys of adapter.json, sorted: explicit fields only, never peft's own config, which holds
# the absolute path of the base weights and sets that JSON cannot encode.
ADAPTER_RECORD_KEYS = ("lora_alpha", "lora_dropout", "peft_version", "r", "target_modules")
# What torch says when an op has no deterministic implementation: training then runs again with
# eager attention instead of SDPA.
NONDETERMINISTIC_OP = "does not have a deterministic implementation"
# The cuBLAS workspace setting under which CUDA's matrix products are deterministic; it must be
# in the environment before the first one runs (PyTorch's reproducibility notes).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# The metadata key of a resume file: the hash of the base weights and encoder settings its
# adapter was trained with, checked before the file is reloaded.
TRAINED_ON_KEY = "trained_on"


def sample_pairs(labels: np.ndarray, per_tag_positive: int, seed: int) -> np.ndarray:
    """Draw statement pairs stratified by focus tag, as SetFit balances its pairs by class.

    For each tag (a column of ``labels``): ``per_tag_positive`` positive pairs, two problems
    with the tag (target 1), and as many negative pairs, one problem with the tag and one that
    shares no tag with it (target 0). Two problems without a focus tag are never paired. Each
    kind is drawn without replacement, and starts again only once every possible pair of the
    kind has been drawn (a rare tag). A tag with fewer than two problems has no positive pair;
    one whose problems share a tag with every other problem has no negative pair.

    Args:
        labels: 0/1, one row per problem being fitted, one column per focus tag.
        per_tag_positive: The positive pairs drawn per tag, and the negative ones.
        seed: The seed of the draws and of the pairs' order.

    Returns:
        int64 ``(n_pairs, 3)``: the row positions of the two problems in ``labels``, then the
        target; in a random order, the order of training.

    Raises:
        ValueError: If ``labels`` is not a 2-D 0/1 array, or ``per_tag_positive`` is below 1.
    """
    labels = _checked_labels(labels)
    if per_tag_positive < 1:
        raise ValueError(f"per_tag_positive must be at least 1, not {per_tag_positive}.")
    rng = np.random.default_rng(seed)
    blocks = []
    for tag in range(labels.shape[1]):
        holders = np.flatnonzero(labels[:, tag])
        positives = _draw(_positive_pairs(holders), per_tag_positive, rng)
        negatives = _draw(_negative_pairs(holders, labels), per_tag_positive, rng)
        blocks += [_with_target(positives, 1), _with_target(negatives, 0)]
    pairs = np.concatenate(blocks)
    return pairs[rng.permutation(len(pairs))]


def adapter_record(settings: dict) -> dict:
    """Describe the LoRA adapter of a SetFit fit: the content of ``adapter.json``.

    Args:
        settings: SetFit's settings (``SETFIT``), with ``lora_r``, ``lora_alpha`` and
            ``lora_dropout``.

    Returns:
        ``{"lora_alpha", "lora_dropout", "peft_version", "r", "target_modules"}``: the target
        modules are the sorted ``LORA_TARGETS``, the peft version the installed one.
    """
    return {
        "lora_alpha": int(settings["lora_alpha"]),
        "lora_dropout": float(settings["lora_dropout"]),
        "peft_version": peft.__version__,
        "r": int(settings["lora_r"]),
        "target_modules": sorted(LORA_TARGETS),
    }


def lora_config(record: dict) -> LoraConfig:
    """Build peft's configuration of an adapter from its record.

    Args:
        record: An ``adapter_record``, e.g. read from ``adapter.json``.

    Returns:
        The ``LoraConfig``: no bias, no task type (the encoder has no task head).

    Raises:
        ValueError: If ``record`` does not hold exactly ``ADAPTER_RECORD_KEYS``.
    """
    _check_adapter_record(record)
    return LoraConfig(
        r=record["r"],
        lora_alpha=record["lora_alpha"],
        lora_dropout=record["lora_dropout"],
        target_modules=list(record["target_modules"]),
        bias="none",
    )


def train_adapter(  # pylint: disable=too-many-arguments
    record: PretrainedRecord,
    config: EncoderConfig,
    texts: Sequence[str],
    labels: np.ndarray,
    settings: dict,
    *,
    device: str,
    log: Callable[[str], None] = print,
) -> dict[str, torch.Tensor]:
    """Train a LoRA adapter so that the cosine of each pair's embeddings approaches its target.

    It loads a fresh copy of the base weights (never the frozen-encoder memo's: peft injects
    its layers into the model it is given), adds LoRA layers on ``LORA_TARGETS``, and trains
    them on ``sample_pairs(labels, ...)``: ``settings["epochs"]`` passes over the pairs, in
    batches of ``batch_pairs`` pairs built from ``micro_batch_pairs``-pair micro-batches whose
    gradients accumulate (the loss is a mean of per-pair terms, so the update is the same),
    with AdamW and a linear warm-up over the first ``warmup`` share of the steps, then a linear
    decay to 0 (sentence-transformers' ``WarmupLinear``, which SetFit uses). The loss is the
    mean squared error between a pair's cosine and its target. Gradient checkpointing saves
    memory; torch is seeded with ``settings["seed"]`` and runs deterministic algorithms only.
    Training uses SDPA attention; if one of its ops has no deterministic implementation (torch
    raises a ``RuntimeError`` naming ``NONDETERMINISTIC_OP``), it runs again from the same seed
    with eager attention, and logs why.

    Args:
        record: The base weights, checked against their hashes before loading.
        config: The encoder settings: inputs are truncated at ``config.max_length`` tokens and
            pooled at their last token, which must be ``config.eos_token_id``.
        texts: The statement encoder inputs of the problems being fitted, one per row of
            ``labels``.
        labels: 0/1, one row per problem, one column per focus tag.
        settings: SetFit's settings (``SETFIT``).
        device: ``"cpu"`` or ``"cuda"``, as ``resolve_device`` returns it.
        log: Receives one progress line per tenth of the steps.

    Returns:
        The adapter's tensors (``get_peft_model_state_dict``), on CPU.

    Raises:
        ValueError: If ``texts`` and ``labels`` differ in length, as ``sample_pairs`` does, or
            if a truncated input does not end with the end-of-sequence token.
        RuntimeError: As torch raises it, but for an op without a deterministic implementation
            under SDPA attention.
    """
    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts but {len(labels)} rows of labels.")
    pairs = sample_pairs(labels, settings["pairs_per_tag_positive"], settings["seed"])
    directory = check_pretrained(record)
    inputs = {"config": config, "texts": texts, "settings": settings, "device": device, "log": log}
    refusal = ""
    try:
        return _trained_state(directory, pairs, attention="sdpa", **inputs)
    except RuntimeError as error:
        if NONDETERMINISTIC_OP not in str(error):
            raise
        refusal = str(error)
    # The refused attempt's model went with the exception: free it before the eager one loads.
    _free_memory()
    log(f"SetFit: training again with eager attention: {refusal}")
    return _trained_state(directory, pairs, attention="eager", **inputs)


def merged_model(
    record: PretrainedRecord, adapter_state: dict[str, torch.Tensor], adapter: dict, device: str
) -> torch.nn.Module:
    """Load the base weights with an adapter merged in: the body that encodes statements.

    Args:
        record: The base weights, checked against their hashes before loading.
        adapter_state: The adapter's tensors, as ``train_adapter`` or ``load_adapter`` return
            them.
        adapter: Its ``adapter_record``.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The base model class (no peft wrapper or layer left), in eval mode, on ``device``.

    Raises:
        ValueError: If ``adapter`` is not an adapter record, or ``adapter_state`` lacks a
            tensor of the adapter or holds one the model does not have.
    """
    model = get_peft_model(load_model(check_pretrained(record), device), lora_config(adapter))
    loaded = set_peft_model_state_dict(model, adapter_state)
    missing = [key for key in loaded.missing_keys if "lora_" in key]
    if missing or loaded.unexpected_keys:
        raise ValueError(
            f"The adapter does not fit the base model: it lacks {missing} and has the unexpected "
            f"tensors {loaded.unexpected_keys}."
        )
    merged = model.merge_and_unload()
    merged.eval()
    return merged


def save_adapter(folder: Path, adapter_state: dict[str, torch.Tensor], adapter: dict) -> None:
    """Write an adapter into a method folder: ``adapter.safetensors`` and ``adapter.json``.

    Args:
        folder: An existing directory.
        adapter_state: The adapter's tensors.
        adapter: Its ``adapter_record``: explicit fields, no path.

    Raises:
        ValueError: If ``adapter`` does not hold exactly ``ADAPTER_RECORD_KEYS``.
    """
    _check_adapter_record(adapter)
    save_file(_contiguous(adapter_state), Path(folder) / ADAPTER_FILE)
    write_json(Path(folder) / ADAPTER_RECORD_FILE, adapter)


def load_adapter(folder: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Read the adapter that ``save_adapter`` wrote into a method folder.

    Args:
        folder: The method folder.

    Returns:
        ``(adapter_state, adapter)``: its tensors, on CPU, and its record.

    Raises:
        ValueError: If ``adapter.json`` does not hold exactly ``ADAPTER_RECORD_KEYS``.
    """
    adapter = read_json(Path(folder) / ADAPTER_RECORD_FILE)
    _check_adapter_record(adapter)
    return load_file(Path(folder) / ADAPTER_FILE, device="cpu"), adapter


def adapter_cache_path(sample_ids: Sequence[int], settings: dict) -> Path:
    """Return where a fit saves its adapter, so that a rerun of the same fit reloads it.

    Args:
        sample_ids: The fitted rows' sample ids, in order (``sample_pairs`` draws row positions).
        settings: SetFit's settings, the seed among them.

    Returns:
        ``experiments_dir(SETFIT_METHOD) / "adapters" / "<sha256>.safetensors"``, the hash of the
        sample ids and the settings.
    """
    ids = [int(sample_id) for sample_id in sample_ids]
    key = sha256_json({"sample_ids": ids, "settings": settings})
    return experiments_dir(SETFIT_METHOD) / "adapters" / f"{key}.safetensors"


class SetFitEstimator(BaseEstimator):  # pylint: disable=too-many-instance-attributes
    """The score estimator of ``f2llm_setfit``: a LoRA-tuned encoder, then the probe's head.

    ``fit`` trains a LoRA adapter on the statements of the rows it is given (``train_adapter``),
    or reloads the one a previous fit of the same rows saved (``resume``); merges it into a
    fresh copy of the base weights; re-encodes those statements with the merged body; and fits
    the probe's head (``make_probe(head_params, width)``) on them, beside the frozen solution
    embedding when the probe kept it. It follows the score-estimator contract of
    ``CrossFittedTagThresholds``. The bodies and the adapter are fitted attributes, never
    parameters: ``clone`` deep-copies parameters, seven times per
    ``CrossFittedTagThresholds.fit``.

    Attributes:
        classes_: ``np.arange(n_tags)``.
        adapter_state_: The adapter's tensors, on CPU.
        statement_encoder_: An ``Encoder`` whose model is the base with the adapter merged in.
        head_: The fitted probe pipeline, over the columns of ``embedding_columns(STATEMENT,
            width)`` and, when the probe kept it, of ``embedding_columns(SOLUTION, width)``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        head_params: dict | None = None,
        settings: dict | None = None,
        device: str = "cuda",
        pretrained: dict | None = None,
        encoder_config: dict | None = None,
        resume: bool = True,
    ):
        """Set the parameters; ``fit`` reads them.

        Args:
            head_params: The probe's tuned configuration (a ``PROBE_GRID`` entry, from its
                ``tuning/params.json``): the head, C, class weight and solution weight.
                Required by ``fit``.
            settings: SetFit's settings; ``None`` means ``SETFIT``.
            device: Where the adapter trains and the merged body encodes: ``"cuda"`` or
                ``"cpu"``.
            pretrained: The base weights, a ``PretrainedRecord.to_dict()``; ``None`` means
                ``PretrainedRecord.from_settings()``.
            encoder_config: The encoder settings, an ``EncoderConfig.to_dict()``; ``None``
                means ``EncoderConfig.from_settings()``.
            resume: Whether ``fit`` saves its adapter at ``adapter_cache_path`` and reloads it
                there instead of training when a previous fit of the same rows saved it.
        """
        self.head_params = head_params
        self.settings = settings
        self.device = device
        self.pretrained = pretrained
        self.encoder_config = encoder_config
        self.resume = resume

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SetFitEstimator:
        """Train or reload the adapter, merge it, re-encode the statements, fit the head.

        Args:
            X: The rows of the feature frame (``feature_frame``), indexed by ``sample_id``:
                the preprocessed problems, whose statements are encoded, and the frozen
                solution embedding.
            y: 0/1 labels, an int array (n, n_tags).

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``head_params`` is ``None``, if ``X`` lacks the solution embedding
                the head needs, or if a resume file was trained with other base weights or
                encoder settings.
            RuntimeError: If ``device`` is ``"cuda"`` and no GPU is available.
        """
        if self.head_params is None:
            raise ValueError(
                "SetFitEstimator needs head_params: the probe's tuned configuration, from its "
                "tuning/params.json."
            )
        record, config, settings = self._resolved()
        device = resolve_device(self.device)
        labels = np.asarray(y, dtype=np.int64)
        texts = encoder_inputs(X, config)[STATEMENT]
        path = adapter_cache_path(X.index, settings) if self.resume else None
        self.adapter_state_ = _resumable_adapter(
            path,
            _trained_on(record, config),
            lambda: train_adapter(record, config, texts, labels, settings, device=device),
        )
        body = merged_model(record, self.adapter_state_, adapter_record(settings), device)
        self.statement_encoder_ = Encoder(
            tokenizer=load_tokenizer(record.directory(), config),
            model=body,
            config=config,
            device=device,
        )
        width = self.statement_encoder_.width
        solution = embedding_columns(SOLUTION, width) if self.head_params["solution_weight"] else []
        frame = _statement_frame(X.index, self.statement_encoder_.encode(texts))
        frame = frame.join(_columns(X, solution))
        with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
            self.head_ = make_probe(self.head_params, width).fit(frame, labels)
        self.classes_ = np.arange(labels.shape[1])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Re-encode the statements with the merged body, then score them with the head.

        Args:
            X: Rows of the feature frame: the solution embedding is read from it when the head
                uses it; its statement embedding columns are not read.

        Returns:
            float64 (n, n_tags).

        Raises:
            NotFittedError: If the estimator is not fitted.
            ValueError: If ``X`` lacks the solution embedding the head needs.
        """
        check_is_fitted(self, "head_")
        encoder = self.statement_encoder_
        texts = encoder_inputs(X, encoder.config)[STATEMENT]
        frame = _statement_frame(X.index, encoder.encode(texts))
        others = [column for column in self.head_.feature_names_in_ if column not in frame]
        frame = frame.join(_columns(X, others))
        return np.asarray(self.head_.decision_function(frame), dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict each tag whose score is at least 0, as the score-estimator contract asks.

        Args:
            X: Rows of the feature frame.

        Returns:
            0/1 ints (n, n_tags).
        """
        return (self.decision_function(X) >= 0).astype(int)

    def _resolved(self) -> tuple[PretrainedRecord, EncoderConfig, dict]:
        """The base weights, the encoder settings and SetFit's settings this fit uses."""
        record = (
            PretrainedRecord.from_settings()
            if self.pretrained is None
            else PretrainedRecord.from_dict(self.pretrained)
        )
        config = (
            EncoderConfig.from_settings()
            if self.encoder_config is None
            else EncoderConfig.from_dict(self.encoder_config)
        )
        return record, config, SETFIT if self.settings is None else self.settings


@dataclass
class _PairLoss:
    """The loss of a micro-batch of pairs, under the model being trained."""

    model: torch.nn.Module
    tokenizer: Callable[..., dict]
    texts: list[str]
    config: EncoderConfig
    device: str

    def __call__(self, pairs: np.ndarray) -> torch.Tensor:
        """Mean squared error between each pair's cosine and its target (CosineSimilarityLoss)."""
        texts = [self.texts[row] for row in pairs[:, 0]] + [self.texts[row] for row in pairs[:, 1]]
        pooled = self._pooled(texts)
        cosine = torch.nn.functional.cosine_similarity(
            pooled[: len(pairs)], pooled[len(pairs) :], dim=-1
        )
        target = torch.as_tensor(pairs[:, 2], dtype=torch.float32, device=self.device)
        return torch.nn.functional.mse_loss(cosine, target)

    def _pooled(self, texts: list[str]) -> torch.Tensor:
        """Embed texts at their last real token (right padding), keeping the gradients."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(len(texts), device=self.device)
        last = attention_mask.sum(dim=1) - 1
        if not torch.all(input_ids[rows, last] == self.config.eos_token_id):
            raise ValueError(
                "An encoder input does not end with the end-of-sequence token "
                f"{self.config.eos_token_id}: the tokenizer must append it, and truncation keep it."
            )
        return hidden[rows, last]


def _trained_state(  # pylint: disable=too-many-arguments
    directory: Path,
    pairs: np.ndarray,
    *,
    config: EncoderConfig,
    texts: Sequence[str],
    settings: dict,
    device: str,
    attention: str,
    log: Callable[[str], None],
) -> dict[str, torch.Tensor]:
    """Train a new adapter on the pairs with the given attention; return its tensors, on CPU."""
    with _deterministic(settings["seed"]):
        model = _lora_model(directory, adapter_record(settings), device, attention)
        pair_loss = _PairLoss(model, load_tokenizer(directory, config), list(texts), config, device)
        _train(pair_loss, pairs, settings, log)
        state = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in get_peft_model_state_dict(model).items()
        }
    del pair_loss, model
    _free_memory()
    return state


def _free_memory() -> None:
    """Free dropped models now: Python's reference cycles, then the GPU's cached blocks."""
    gc.collect()
    torch.cuda.empty_cache()


def _train(
    pair_loss: _PairLoss, pairs: np.ndarray, settings: dict, log: Callable[[str], None]
) -> None:
    """Run the optimisation: AdamW, warm-up then linear decay, one batch per step."""
    batches = _batches(pairs, settings["batch_pairs"], settings["epochs"])
    trainable = [parameter for parameter in pair_loss.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(settings["warmup"] * len(batches)), len(batches)
    )
    log(f"SetFit: {len(pairs)} pairs, {len(batches)} steps on {pair_loss.device}.")
    started = time.perf_counter()
    every = max(1, len(batches) // 10)
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = _accumulate(pair_loss, batch, settings["micro_batch_pairs"])
        optimizer.step()
        scheduler.step()
        if step % every == 0 or step == len(batches):
            elapsed = time.perf_counter() - started
            log(f"SetFit: step {step}/{len(batches)}, loss {loss:.4f}, {elapsed:.0f} s.")
    if pair_loss.device == "cuda":
        log(f"SetFit: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB.")


def _batches(pairs: np.ndarray, batch_pairs: int, epochs: int) -> list[np.ndarray]:
    """Cut the pairs into batches, in their order, once per epoch."""
    return [
        pairs[start : start + batch_pairs]
        for _ in range(epochs)
        for start in range(0, len(pairs), batch_pairs)
    ]


def _accumulate(pair_loss: _PairLoss, batch: np.ndarray, micro_batch_pairs: int) -> float:
    """Accumulate a batch's gradients over its micro-batches; return the batch's mean loss."""
    total = 0.0
    for start in range(0, len(batch), micro_batch_pairs):
        micro = batch[start : start + micro_batch_pairs]
        loss = pair_loss(micro) * (len(micro) / len(batch))
        loss.backward()
        total += float(loss.detach())
    return total


@contextmanager
def _deterministic(seed: int) -> Iterator[None]:
    """Seed torch and allow deterministic algorithms only, then restore the previous mode."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _lora_model(directory: Path, adapter: dict, device: str, attention: str) -> peft.PeftModel:
    """Load a fresh copy of the base weights, with new LoRA layers, ready to train.

    ``attention`` is ``"sdpa"``, which ``load_model`` loads, or ``"eager"``, the fallback of
    ``train_adapter``.
    """
    model = load_model(directory, device)
    model.set_attn_implementation(attention)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = get_peft_model(model, lora_config(adapter))
    lora.train()
    return lora


def _checked_labels(labels: np.ndarray) -> np.ndarray:
    """Return labels as an int64 array, after checking they are a 2-D 0/1 array."""
    array = np.asarray(labels)
    if array.ndim != 2 or not np.isin(array, (0, 1)).all():
        raise ValueError("labels must be a 2-D 0/1 array: one row per problem, one column per tag.")
    return array.astype(np.int64)


def _positive_pairs(holders: np.ndarray) -> np.ndarray:
    """Every pair of two problems with the tag, each once: ``(n, 2)`` row positions."""
    first, second = np.triu_indices(len(holders), k=1)
    return np.column_stack([holders[first], holders[second]])


def _negative_pairs(holders: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Every pair of a problem with the tag and a problem sharing no tag with it."""
    anchors, partners = np.nonzero(labels[holders] @ labels.T == 0)
    return np.column_stack([holders[anchors], partners])


def _draw(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``count`` rows of ``candidates``: every row once before any row twice."""
    if len(candidates) == 0:
        return candidates
    rounds = [
        rng.choice(len(candidates), size=min(len(candidates), count - start), replace=False)
        for start in range(0, count, len(candidates))
    ]
    return candidates[np.concatenate(rounds)]


def _with_target(pairs: np.ndarray, target: int) -> np.ndarray:
    """Append the target column to ``(n, 2)`` pairs."""
    return np.column_stack([pairs, np.full(len(pairs), target)]).astype(np.int64)


def _check_adapter_record(record: dict) -> None:
    """Fail unless ``record`` holds exactly the keys of an ``adapter_record``."""
    if sorted(record) != list(ADAPTER_RECORD_KEYS):
        raise ValueError(
            f"An adapter record holds the keys {list(ADAPTER_RECORD_KEYS)}, not {sorted(record)}: "
            "build it with adapter_record()."
        )


def _contiguous(adapter_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors as safetensors stores them: contiguous, on CPU."""
    return {name: tensor.detach().to("cpu").contiguous() for name, tensor in adapter_state.items()}


def _trained_on(record: PretrainedRecord, config: EncoderConfig) -> str:
    """Hash what an adapter is trained on besides its rows and settings: the base and encoder."""
    return sha256_json({"pretrained": record.to_dict(), "encoder": config.to_dict()})


def _resumable_adapter(
    path: Path | None, trained_on: str, train: Callable[[], dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Reload the adapter saved at ``path``, or train it and save it there (``None``: never)."""
    if path is not None and path.is_file():
        return _read_resume_file(path, trained_on)
    adapter_state = train()
    if path is not None:
        _write_resume_file(path, adapter_state, trained_on)
    return adapter_state


def _write_resume_file(path: Path, adapter_state: dict[str, torch.Tensor], trained_on: str) -> None:
    """Save an adapter with what it was trained on; a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    save_file(_contiguous(adapter_state), partial, metadata={TRAINED_ON_KEY: trained_on})
    os.replace(partial, path)


def _read_resume_file(path: Path, trained_on: str) -> dict[str, torch.Tensor]:
    """Read a saved adapter, after checking it was trained on the same base and encoder."""
    with safe_open(path, framework="pt", device="cpu") as file:
        stored = (file.metadata() or {}).get(TRAINED_ON_KEY)
    if stored != trained_on:
        raise ValueError(
            f"{path} holds an adapter trained with other base weights or encoder settings: "
            "delete it, and the fit trains the adapter again."
        )
    return load_file(path, device="cpu")


def _statement_frame(index: pd.Index, vectors: np.ndarray) -> pd.DataFrame:
    """The statement embedding block, as float64 columns indexed like the rows."""
    columns = embedding_columns(STATEMENT, vectors.shape[1])
    return pd.DataFrame(vectors.astype(np.float64), index=index, columns=columns)


def _columns(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The given embedding columns of the feature frame, as float64."""
    missing = [name for name in names if name not in X]
    if missing:
        raise ValueError(
            f"The rows lack {len(missing)} embedding columns, e.g. {missing[0]!r}: pass the "
            "feature frame (feature_frame), which holds the frozen solution embedding."
        )
    return X[names].astype(np.float64)
