"""The models of the ModernBERT methods: built for training, scored, and gathered into a bag.

A bag is the mean of its fold models' probabilities (docs/adr/0005), held with one copy of the
encoder in memory:

- ``LoraBag``: the pretrained encoder once, with each fold's LoRA updates, pooling head and
  classifier as a named PEFT adapter; ``set_adapter`` chooses which one runs;
- ``FullBag``: one model, and each fold's weights as a state dict (in pinned memory on CUDA),
  copied into it with ``load_state_dict`` before its forward pass.

Scoring runs in batches of similar length, with fp16 autocast on CUDA as in training and float32
on the CPU. A loaded bag on the GPU scores one problem per forward pass: one batch of many
problems held the GPU at full power and rebooted a machine once. In fp16 that moves a score from
the frozen batches by up to 6.0e-3 (modernbert_lora), within score_atol (docs/adr/0005).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.modernbert.settings import Variant

# Where a bag built without a model (the base class, in tests) says its fold models are.
_CPU = torch.device("cpu")

_LABELS = {
    "num_labels": len(FOCUS_TAGS),
    "problem_type": "multi_label_classification",
    "id2label": dict(enumerate(FOCUS_TAGS)),
    "label2id": {tag: index for index, tag in enumerate(FOCUS_TAGS)},
}


def resolve_device(setting: str) -> torch.device:
    """Turn a device setting into a device.

    Args:
        setting: ``"auto"`` (CUDA when available, else the CPU) or a torch device name.

    Returns:
        The device.

    Raises:
        RuntimeError: If ``setting`` names a CUDA device and torch sees no GPU: refused here,
            before any weight moves, rather than by torch when the bag moves to it.
    """
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"The device {setting!r} was asked for, but torch sees no GPU.")
    return device


def load_tokenizer(folder: Path):
    """Load the tokenizer of a pretrained snapshot, with no length limit of its own.

    Args:
        folder: The snapshot's folder.

    Returns:
        The tokenizer; ``encode_problems`` cuts the inputs, not the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    tokenizer.model_max_length = math.inf
    return tokenizer


def classifier_model(folder: Path, *, weights: bool, classifier_dropout: float = 0.0):
    """Build the encoder with a classification head of one output per focus tag.

    Args:
        folder: The pretrained snapshot's folder.
        weights: Load the pretrained weights (to train, or under LoRA updates); False builds the
            architecture only, for weights copied in afterwards.
        classifier_dropout: Dropout before the classifier (trained by full fine-tuning).

    Returns:
        The model, in float32 on the CPU (the Trainer and ``predict_scores`` set its mode).
    """
    config = AutoConfig.from_pretrained(
        folder, local_files_only=True, classifier_dropout=classifier_dropout, **_LABELS
    )
    if weights:
        return AutoModelForSequenceClassification.from_pretrained(
            folder,
            config=config,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
    return AutoModelForSequenceClassification.from_config(config, attn_implementation="sdpa")


def lora_config(variant: Variant, rank: int, dropout: float) -> LoraConfig:
    """Return the LoRA updates of a configuration: every linear layer of attention and MLP.

    Args:
        variant: The method's settings (targets, trained modules, alpha per rank).
        rank: The rank of each update.
        dropout: The dropout on each update's input.

    Returns:
        The PEFT configuration; the pooling head and the classifier are trained in full.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=variant.lora_alpha_per_rank * rank,
        lora_dropout=dropout,
        target_modules=list(variant.lora_targets),
        modules_to_save=list(variant.trained_modules),
    )


@torch.inference_mode()
def predict_scores(
    model: torch.nn.Module,
    input_ids: Sequence[list[int]],
    tokenizer,
    *,
    batch_size: int,
    pad_to_multiple_of: int,
) -> np.ndarray:
    """Score each problem with a model (its active adapter, for a PEFT model).

    Problems go through in batches of similar length, with fp16 autocast on CUDA. The model's
    train or eval mode is restored afterwards.

    Args:
        model: A model whose classifier gives one logit per focus tag.
        input_ids: Token ids per problem.
        tokenizer: The tokenizer, to pad the batches.
        batch_size: Problems per forward pass.
        pad_to_multiple_of: Batches are padded to a multiple of this many tokens.

    Returns:
        The probability of each tag, float32, one row per problem, in the order of ``input_ids``.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    order = np.argsort([len(ids) for ids in input_ids], kind="stable")
    scores = np.empty((len(input_ids), len(FOCUS_TAGS)), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        features = [
            {"input_ids": input_ids[row], "attention_mask": [1] * len(input_ids[row])}
            for row in rows
        ]
        batch = collator(features).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        scores[rows] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train(was_training)
    return scores


class _Bag:
    """What both bags share: averaging their fold models' probabilities."""

    def __init__(
        self, names: Sequence[str], tokenizer, variant: Variant, device: torch.device = _CPU
    ) -> None:
        """Keep the fold names and how to score.

        Args:
            names: The fold models' names, in order.
            tokenizer: The tokenizer, to pad the batches.
            variant: The method's settings (score batch size, padding).
            device: Where the fold models are.
        """
        self.names = tuple(names)
        self.tokenizer = tokenizer
        self.variant = variant
        self.device = device

    def _use(self, name: str) -> torch.nn.Module:
        """Make fold model ``name`` the one that runs, and return the module to call."""
        raise NotImplementedError

    def fold_scores(
        self, input_ids: Sequence[list[int]], names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Score the problems with each of several fold models, one after the other.

        Args:
            input_ids: Token ids per problem.
            names: The fold models to score with; all of them by default.

        Returns:
            Each fold model's float32 probabilities, by name.
        """
        # On the GPU, one problem per forward pass: never the full power of a batch of many.
        batch_size = 1 if self.device.type == "cuda" else self.variant.score_batch
        return {
            name: predict_scores(
                self._use(name),
                input_ids,
                self.tokenizer,
                batch_size=batch_size,
                pad_to_multiple_of=self.variant.pad_to_multiple_of,
            )
            for name in (self.names if names is None else names)
        }

    def scores(self, input_ids: Sequence[list[int]]) -> np.ndarray:
        """Score the problems with the bag: the mean of its fold models' probabilities.

        Args:
            input_ids: Token ids per problem.

        Returns:
            The mean probability of each tag, one row per problem, as float64.
        """
        return bag_mean(self.fold_scores(input_ids))


def bag_mean(fold_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Average fold models' probabilities, as the bag does.

    Args:
        fold_scores: Each fold model's float32 probabilities, by name.

    Returns:
        Their mean, averaged in float32 as the notebooks averaged them, then cast exactly to
        float64.
    """
    return np.mean(np.stack(list(fold_scores.values())), axis=0).astype(np.float64)


class LoraBag(_Bag):
    """One pretrained encoder, with each fold model's LoRA updates, head and classifier."""

    def __init__(self, model: PeftModel, names: Sequence[str], tokenizer, variant: Variant):
        """Keep the PEFT model holding one adapter per fold model.

        Args:
            model: The PEFT model, one adapter per name, in eval mode on its device.
            names: The adapters' names.
            tokenizer: The tokenizer.
            variant: The method's settings.
        """
        super().__init__(names, tokenizer, variant, next(model.parameters()).device)
        self.model = model

    @classmethod
    def load(  # pylint: disable=too-many-arguments
        cls,
        variant: Variant,
        snapshot: Path,
        folds: Mapping[str, Path],
        *,
        rank: int,
        dropout: float,
        device: torch.device,
    ) -> LoraBag:
        """Build the bag from a pretrained snapshot and the fold models' saved adapter tensors.

        Args:
            variant: The method's settings.
            snapshot: The pretrained snapshot's folder (encoder weights and tokenizer).
            folds: Each fold model's ``get_peft_model_state_dict`` file, by name.
            rank: The rank of the updates.
            dropout: The dropout of the updates (inactive when scoring).
            device: Where to put the model.

        Returns:
            The bag, its first adapter active.

        Raises:
            ValueError: If a file holds a tensor that the adapter lacks, or lacks one of its own.
        """
        config = lora_config(variant, rank, dropout)
        names = list(folds)
        model = get_peft_model(classifier_model(snapshot, weights=True), config, names[0])
        for name in names[1:]:
            model.add_adapter(name, config)
        for name, path in folds.items():
            loaded = set_peft_model_state_dict(model, load_file(path), adapter_name=name)
            # Both would load silently: a tensor of another architecture, or a missing one that
            # keeps its random starting values.
            missing = [key for key in loaded.missing_keys if f".{name}." in key]
            if loaded.unexpected_keys or missing:
                raise ValueError(
                    f"{path} does not hold the adapter's tensors: unexpected "
                    f"{loaded.unexpected_keys}, missing {missing}."
                )
        model.set_adapter(names[0])
        return cls(model.to(device).eval(), names, load_tokenizer(snapshot), variant)

    def _use(self, name: str) -> torch.nn.Module:
        self.model.set_adapter(name)
        return self.model


class FullBag(_Bag):
    """One model, into which each fold model's weights are copied before it scores."""

    def __init__(
        self,
        model: torch.nn.Module,
        weights: Mapping[str, Mapping[str, torch.Tensor]],
        tokenizer,
        variant: Variant,
    ) -> None:
        """Keep the model and every fold model's weights.

        Args:
            model: The model that receives each fold's weights, in eval mode on its device.
            weights: Each fold model's state dict, by name (pinned host memory on CUDA).
            tokenizer: The tokenizer.
            variant: The method's settings.
        """
        super().__init__(list(weights), tokenizer, variant, next(model.parameters()).device)
        self.model = model
        self.weights = dict(weights)

    @classmethod
    def load(
        cls, variant: Variant, snapshot: Path, folds: Mapping[str, Path], *, device: torch.device
    ) -> FullBag:
        """Build the bag from a pretrained snapshot's configuration and the fold models' weights.

        Args:
            variant: The method's settings.
            snapshot: The pretrained snapshot's folder (configuration and tokenizer only).
            folds: Each fold model's ``state_dict`` file, by name.
            device: Where to put the model.

        Returns:
            The bag, holding the first fold model's weights.
        """
        pin = device.type == "cuda"
        weights = {
            name: {
                key: (tensor.pin_memory() if pin else tensor)
                for key, tensor in load_file(path).items()
            }
            for name, path in folds.items()
        }
        model = classifier_model(snapshot, weights=False)
        model.load_state_dict(next(iter(weights.values())))
        return cls(model.to(device).eval(), weights, load_tokenizer(snapshot), variant)

    def _use(self, name: str) -> torch.nn.Module:
        self.model.load_state_dict(self.weights[name])
        return self.model
