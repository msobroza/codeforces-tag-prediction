"""Training one ModernBERT fold model with the Hugging Face ``Trainer``.

fp16 mixed precision on CUDA (float32 on the CPU), gradient accumulation (``micro_batch`` problems
per forward pass), length-grouped batches, linear warmup then decay, positives weighted by
``(negatives / positives) ** pos_weight_power``. The ``Trainer`` evaluates nothing and writes no
checkpoint (``save_strategy="no"``: its checkpoints hold pickle files) and no report.

A fold model's training can collapse to uninformative scores: ``train_fold_model`` checks its
macro average precision on a sample of its own training problems, and trains again with the next
seed when it stays under ``collapse_ap`` (docs/adr/0005). The check reads training labels only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from sklearn.metrics import average_precision_score
from transformers import (
    DataCollatorWithPadding,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from codeforces_tags.methods.modernbert.modeling import (
    classifier_model,
    lora_config,
    predict_scores,
)
from codeforces_tags.methods.modernbert.settings import FOLD_WEIGHTS_FILE, Variant


@dataclass(frozen=True)
class LoraParams:
    """One configuration of ``modernbert_lora``'s search space.

    Attributes:
        learning_rate: Peak learning rate of the LoRA updates and of the trained head.
        lora_rank: Rank of each update (alpha is ``lora_alpha_per_rank`` times it).
        lora_dropout: Dropout on each update's input.
        epochs: Passes over the training problems.
        batch_size: Problems per optimiser step.
        warmup_ratio: Share of the steps with a rising learning rate; it then falls to 0.
        pos_weight_power: Exponent of each tag's negative-to-positive ratio in the loss.
    """

    learning_rate: float
    lora_rank: int
    lora_dropout: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    pos_weight_power: float


@dataclass(frozen=True)
class FullParams:
    """One configuration of ``modernbert_base_full``'s search space.

    Attributes:
        learning_rate: Peak learning rate of every weight.
        weight_decay: AdamW's decoupled weight decay (not on biases and normalisation weights).
        classifier_dropout: Dropout before the classifier.
        epochs: Passes over the training problems.
        batch_size: Problems per optimiser step.
        warmup_ratio: Share of the steps with a rising learning rate; it then falls to 0.
        pos_weight_power: Exponent of each tag's negative-to-positive ratio in the loss.
    """

    learning_rate: float
    weight_decay: float
    classifier_dropout: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    pos_weight_power: float


Params = LoraParams | FullParams


class TrainingSetup(NamedTuple):
    """What a training reads beside its problems: a configuration, and where to train it."""

    variant: Variant
    params: Params | None  # None in a study, whose trials draw their own
    snapshot: Path  # the pretrained snapshot's folder
    tokenizer: object
    device: torch.device
    output_dir: Path  # the Trainer's working folder, which stays empty


def params_from(variant: Variant, values: Mapping) -> Params:
    """Read a configuration of a method's search space, e.g. ``tuning/params.json``'s.

    Args:
        variant: The method's settings.
        values: The parameters, by name (other keys are refused).

    Returns:
        ``LoraParams`` or ``FullParams``.
    """
    kind = LoraParams if variant.lora else FullParams
    return kind(**dict(values))


def positive_weights(labels: np.ndarray, power: float) -> torch.Tensor:
    """Weigh each tag's positives by its negative-to-positive ratio, raised to ``power``.

    Args:
        labels: 0/1 labels of the training problems, one column per tag.
        power: 0 for weights of 1 (plain cross-entropy), 1 for balanced classes.

    Returns:
        One weight per tag, as float32.
    """
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    return torch.tensor((negatives / np.maximum(positives, 1)) ** power, dtype=torch.float32)


class WeightedBce:  # pylint: disable=too-few-public-methods
    """Binary cross-entropy with weighted positives: the Trainer's ``compute_loss_func``."""

    def __init__(self, pos_weight: torch.Tensor) -> None:
        """Keep the weight of each tag's positives.

        Args:
            pos_weight: One weight per tag, from ``positive_weights``.
        """
        self.pos_weight = pos_weight

    def __call__(self, outputs, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
        """Return the batch's summed loss, divided by the number of labels of the whole step.

        Args:
            outputs: The model's outputs, with ``logits``.
            labels: 0/1 labels of the batch, one column per tag.
            num_items_in_batch: Labels in the optimiser step (every accumulated batch), which
                the Trainer passes when training; None to average over this batch alone.

        Returns:
            The loss; summed over the accumulated batches, it is the mean over the step.
        """
        logits = outputs.logits.float()
        losses = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=self.pos_weight.to(logits.device), reduction="none"
        )
        count = losses.numel() if num_items_in_batch is None else num_items_in_batch
        return losses.sum() / count


class TagDataset(torch.utils.data.Dataset):
    """The token ids of problems and their 0/1 focus tags."""

    def __init__(self, input_ids: Sequence[list[int]], labels: np.ndarray) -> None:
        """Keep the token ids and the labels.

        Args:
            input_ids: Token ids per problem.
            labels: 0/1 labels, one row per problem, one column per tag.
        """
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of problems."""
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, list]:
        """Return one problem's token ids, attention mask and labels."""
        ids = self.input_ids[index]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": self.labels[index].astype(np.float32).tolist(),
        }


class EpochHook(TrainerCallback):  # pylint: disable=too-few-public-methods
    """Call a function with the epoch number and the model after each epoch (e.g. to tune)."""

    def __init__(self, hook: Callable[[int, torch.nn.Module], None]) -> None:
        """Keep the function.

        Args:
            hook: Called as ``hook(epoch, model)``; may raise to stop the training.
        """
        self.hook = hook

    # The Trainer calls every callback with this signature.
    def on_epoch_end(self, args, state, control, model=None, **kwargs):  # pylint: disable=unused-argument
        """Call the hook with the finished epoch's number and the model."""
        self.hook(round(state.epoch), model)


def training_arguments(  # pylint: disable=too-many-arguments
    variant: Variant, params: Params, *, output_dir: Path, seed: int, device: torch.device
) -> TrainingArguments:
    """Build the Trainer settings of a configuration.

    Args:
        variant: The method's settings (micro-batch, fixed weight decay).
        params: The configuration.
        output_dir: The Trainer's working folder, which stays empty.
        seed: Seed of the batch order and of the dropout masks.
        device: Where to train: fp16 on CUDA, float32 on the CPU.

    Returns:
        The settings: gradient accumulation, length-grouped batches, linear warmup and decay, no
        evaluation, checkpoint or report of the Trainer's own.
    """
    weight_decay = params.weight_decay if isinstance(params, FullParams) else variant.weight_decay
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=variant.micro_batch,
        gradient_accumulation_steps=max(1, params.batch_size // variant.micro_batch),
        learning_rate=params.learning_rate,
        num_train_epochs=params.epochs,
        warmup_steps=params.warmup_ratio,  # a float below 1 is a share of the steps
        lr_scheduler_type="linear",
        weight_decay=weight_decay,
        fp16=device.type == "cuda",
        use_cpu=device.type == "cpu",
        train_sampling_strategy="group_by_length",
        eval_strategy="no",
        save_strategy="no",  # a Trainer checkpoint holds pickle files
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        label_names=["labels"],
        seed=seed,
        data_seed=seed,
    )


def build_training_model(variant: Variant, params: Params, snapshot: Path) -> torch.nn.Module:
    """Build a fresh model to train: LoRA updates on the encoder, or every weight trainable.

    Args:
        variant: The method's settings.
        params: The configuration.
        snapshot: The pretrained snapshot's folder.

    Returns:
        The model, on the CPU.
    """
    if isinstance(params, LoraParams):
        base = classifier_model(snapshot, weights=True)
        return get_peft_model(base, lora_config(variant, params.lora_rank, params.lora_dropout))
    return classifier_model(snapshot, weights=True, classifier_dropout=params.classifier_dropout)


def train_model(
    setup: TrainingSetup,
    train_ids: Sequence[list[int]],
    train_labels: np.ndarray,
    *,
    seed: int,
    epoch_hook: Callable[[int, torch.nn.Module], None] | None = None,
) -> torch.nn.Module:
    """Train a fresh model of a configuration on the given problems.

    Args:
        setup: The configuration, and where to train it.
        train_ids: Token ids of the training problems.
        train_labels: Their 0/1 labels, one column per focus tag.
        seed: Seed of the initial classifier, the batch order and the dropout masks.
        epoch_hook: Called as ``epoch_hook(epoch, model)`` after each epoch, or None.

    Returns:
        The trained model, on ``setup.device``.
    """
    variant, params = setup.variant, setup.params
    set_seed(seed)
    model = build_training_model(variant, params, setup.snapshot)
    trainer = Trainer(
        model=model,
        args=training_arguments(
            variant, params, output_dir=setup.output_dir, seed=seed, device=setup.device
        ),
        train_dataset=TagDataset(train_ids, train_labels),
        data_collator=DataCollatorWithPadding(
            setup.tokenizer, pad_to_multiple_of=variant.pad_to_multiple_of
        ),
        processing_class=setup.tokenizer,  # names the input the length-grouped sampler measures
        compute_loss_func=WeightedBce(positive_weights(train_labels, params.pos_weight_power)),
        callbacks=[] if epoch_hook is None else [EpochHook(epoch_hook)],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    return model


def training_macro_ap(
    model: torch.nn.Module,
    variant: Variant,
    sample_ids: Sequence[list[int]],
    sample_labels: np.ndarray,
    tokenizer,
) -> float:
    """Measure a trained model on a sample of its own training problems.

    Args:
        model: The trained model.
        variant: The method's settings (score batch, padding).
        sample_ids: Token ids of the sample.
        sample_labels: Its 0/1 labels.
        tokenizer: The tokenizer.

    Returns:
        The macro average precision; NaN when a score is not finite (a diverged training).
    """
    scores = predict_scores(
        model,
        sample_ids,
        tokenizer,
        batch_size=variant.score_batch,
        pad_to_multiple_of=variant.pad_to_multiple_of,
    )
    if not np.isfinite(scores).all():
        return math.nan
    return float(average_precision_score(sample_labels, scores, average="macro"))


def train_fold_model(
    setup: TrainingSetup,
    train_ids: Sequence[list[int]],
    train_labels: np.ndarray,
    *,
    first_seed: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Train one fold model, again with the next seed while its training collapses.

    Args:
        setup: The configuration, and where to train it; its variant gives the seed, the
            collapse threshold and sample, and the seeds to try (``seed`` to
            ``seed + fold_seeds - 1``).
        train_ids: Token ids of the fold model's training problems (the other folds).
        train_labels: Their 0/1 labels.
        first_seed: The first seed to try, within that range; ``seed`` by default. Training a
            frozen fold model again starts from the seed that trained it.

    Returns:
        The trained model, and its record: ``params``, the ``seed`` that trained it, the
        ``train_macro_ap`` of its training sample, the ``minutes`` of every attempt and the
        ``collapsed_seeds`` tried before.

    Raises:
        RuntimeError: If every one of the ``fold_seeds`` seeds collapses.
    """
    variant = setup.variant
    size = min(variant.collapse_sample, len(train_ids))
    sample = np.random.default_rng(variant.seed).choice(len(train_ids), size, replace=False)
    sample_ids = [train_ids[row] for row in sample]
    start, collapsed = time.perf_counter(), []
    start_seed = variant.seed if first_seed is None else first_seed
    for seed in range(start_seed, variant.seed + variant.fold_seeds):
        model = train_model(setup, train_ids, train_labels, seed=seed)
        train_ap = training_macro_ap(
            model, variant, sample_ids, train_labels[sample], setup.tokenizer
        )
        if train_ap >= variant.collapse_ap:  # False for NaN: a diverged training collapsed
            return model, {
                "params": asdict(setup.params),
                "seed": seed,
                "train_macro_ap": train_ap,
                "minutes": (time.perf_counter() - start) / 60,
                "collapsed_seeds": collapsed,
            }
        collapsed.append(seed)
        del model
    raise RuntimeError(f"Every seed tried collapsed: {collapsed}.")


def fold_state(model: torch.nn.Module, variant: Variant) -> dict[str, torch.Tensor]:
    """Return what a fold model trained, as tensors on the CPU.

    Args:
        model: The trained model.
        variant: The method's settings.

    Returns:
        ``get_peft_model_state_dict`` for LoRA (the updates, the head and the classifier), the
        whole ``state_dict`` for full fine-tuning.
    """
    state = get_peft_model_state_dict(model) if variant.lora else model.state_dict()
    return {key: tensor.detach().cpu().contiguous() for key, tensor in state.items()}


def save_fold(model: torch.nn.Module, variant: Variant, folder: Path) -> Path:
    """Save what a fold model trained, with ``safetensors.torch.save_file`` (no pickle, no path).

    Args:
        model: The trained model.
        variant: The method's settings.
        folder: The fold model's folder in the trained cache.

    Returns:
        The weights file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FOLD_WEIGHTS_FILE
    save_file(fold_state(model, variant), str(path))
    return path
