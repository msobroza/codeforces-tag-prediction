"""The fold models of a ModernBERT method in the trained cache: trained, adopted, read back.

Fold model ``fold_k`` trains on the dev folds other than k, with the configuration of
``tuning/params.json``, and scores fold k right after training: its out-of-fold scores
(docs/adr/0005). Its folder, ``<trained cache>/<method>/fold_k/``, holds:

- ``weights.safetensors``: what it trained (``save_fold``);
- ``training.json``: the study's recipe, the configuration, the seed that trained it (the seeds
  before it collapsed), the macro average precision of its training sample and its minutes;
- ``held_out_scores.npy``: its float32 probabilities on fold k, in the order of the dev problems.

``training.json`` is written last, so a folder that has one is complete: ``train_folds`` resumes
after the last complete fold model. ``read_fold_models`` reads them back for ``freeze``, refusing a
fold model trained under another recipe or configuration.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from codeforces_tags.config import FOCUS_TAGS, PROJECT_ROOT
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.methods.modernbert.inputs import encode_problems, take
from codeforces_tags.methods.modernbert.modeling import predict_scores
from codeforces_tags.methods.modernbert.settings import (
    FOLD_WEIGHTS_FILE,
    HELD_OUT_FILE,
    PARAMS_FILE,
    TRAINING_FILE,
    Variant,
    study_recipe,
)
from codeforces_tags.methods.modernbert.training import (
    TrainingSetup,
    save_fold,
    train_fold_model,
)
from codeforces_tags.methods.modernbert.weights import MissingWeightsError
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The weights file of a fold model that the former notebooks 05 and 06 trained, which ``adopt``
# copies: PEFT's adapter file for LoRA (the same tensors as ``get_peft_model_state_dict``), the
# model's ``state_dict`` for full fine-tuning.
NOTEBOOK_WEIGHTS = {True: "adapter_model.safetensors", False: "model.safetensors"}


class DevData(NamedTuple):
    """The dev problems as training reads them."""

    folds: pd.Series  # dev fold per problem, indexed by sample_id
    groups: pd.Series  # near-duplicate group per problem
    labels: pd.DataFrame  # 0/1 focus labels
    input_ids: list  # token ids per problem, in the same order


def fold_name(fold: int) -> str:
    """Return the name of the fold model that leaves out a dev fold.

    Args:
        fold: The dev fold.

    Returns:
        ``fold_<fold>``.
    """
    return f"fold_{fold}"


def read_dev(raw_dev: pd.DataFrame, tokenizer, variant: Variant) -> DevData:
    """Preprocess and tokenise the dev problems, with their labels, folds and groups.

    Args:
        raw_dev: Raw dev problems, with the ``fold`` and ``group`` of ``select_part``.
        tokenizer: The tokenizer of the method's snapshot.
        variant: The method's settings (input length).

    Returns:
        The dev data, in the order of ``raw_dev``.
    """
    problems = raw_to_problems(raw_dev)
    return DevData(
        folds=problems["fold"],
        groups=problems["group"],
        labels=focus_labels(problems),
        input_ids=encode_problems(problems, tokenizer, variant.max_length, variant.code_min_tokens),
    )


def check_fold_record(record: Mapping, variant: Variant, chosen: Mapping, where: Path) -> None:
    """Fail unless a fold model was trained under the study's recipe, with the chosen parameters.

    Args:
        record: The fold model's ``training.json``.
        variant: The method's settings.
        chosen: The content of ``tuning/params.json``.
        where: The fold model's folder, for the message.

    Raises:
        ValueError: If the recipe or the parameters differ.
    """
    if record.get("recipe") != study_recipe(variant) or record.get("params") != chosen["params"]:
        raise ValueError(
            f"{where} was trained under another recipe or configuration than {PARAMS_FILE}'s: "
            "train the fold models again (train), or export the study that chose them."
        )


def train_folds(
    setup: TrainingSetup,
    chosen: Mapping,
    dev: DevData,
    root: Path,
    first_seeds: Mapping[str, int] | None = None,
) -> Iterator[str]:
    """Train one fold model per dev fold, skipping those already complete in ``root``.

    Args:
        setup: The configuration and where to train it.
        chosen: The content of ``tuning/params.json``.
        dev: The dev data.
        root: The folder of the fold models in the trained cache.
        first_seeds: The first seed to try, by fold model name (e.g. the seeds that trained the
            frozen ones); the variant's seed for the others.

    Yields:
        One line per fold model, once it is trained or found complete.

    Raises:
        ValueError: If a complete fold model holds another recipe or configuration.
    """
    for fold in sorted(dev.folds.unique()):
        folder = root / fold_name(fold)
        if (folder / TRAINING_FILE).is_file():
            check_fold_record(_read_json(folder / TRAINING_FILE), setup.variant, chosen, folder)
            yield f"{folder} holds a complete fold model: kept"
            continue
        first_seed = (first_seeds or {}).get(fold_name(fold))
        record = train_fold(setup, dev, fold, folder, first_seed=first_seed)
        yield (
            f"trained {folder} in {record['minutes']:.1f} min with seed {record['seed']} "
            f"(training macro AP {record['train_macro_ap']:.3f})"
        )


def train_fold(
    setup: TrainingSetup, dev: DevData, fold: int, folder: Path, *, first_seed: int | None = None
) -> dict:
    """Train the fold model that leaves out one dev fold, and save it with its scores of that fold.

    Args:
        setup: The configuration and where to train it.
        dev: The dev data.
        fold: The dev fold it leaves out.
        folder: Its folder in the trained cache.
        first_seed: The first seed to try (``train_fold_model``).

    Returns:
        Its ``training.json``.
    """
    variant = setup.variant
    training = (dev.folds != fold).to_numpy()
    model, record = train_fold_model(
        setup,
        take(dev.input_ids, training),
        dev.labels[training].to_numpy(),
        first_seed=first_seed,
    )
    held_out = predict_scores(
        model,
        take(dev.input_ids, ~training),
        setup.tokenizer,
        batch_size=variant.score_batch,
        pad_to_multiple_of=variant.pad_to_multiple_of,
    )
    save_fold(model, variant, folder)
    np.save(folder / HELD_OUT_FILE, held_out)
    record = {"recipe": study_recipe(variant), **record}
    _write_json(folder / TRAINING_FILE, record)
    return record


def adopt_folds(
    variant: Variant, chosen: Mapping, source: Path, root: Path, folds: Iterable[int]
) -> Iterator[str]:
    """Copy the fold models a notebook trained into the trained cache, in its layout.

    Each source folder ``fold_k`` holds the notebook's weights file (``NOTEBOOK_WEIGHTS``),
    ``training.json`` and ``held_out_scores.npy``. The copied ``training.json`` gains
    ``adopted_from``, the source folder (relative to the repository when inside it). A file is
    hard-linked when the cache is on the same file system, else copied.

    Args:
        variant: The method's settings.
        chosen: The content of ``tuning/params.json``.
        source: The notebook's folder of fold models.
        root: The method's folder in the trained cache.
        folds: The dev folds.

    Yields:
        One line per fold model, once it is adopted or found complete.

    Raises:
        ValueError: If a source or complete fold model holds another recipe or configuration.
        MissingWeightsError: If a source file is missing.
    """
    for fold in folds:
        origin, folder = source / fold_name(fold), root / fold_name(fold)
        weights = origin / NOTEBOOK_WEIGHTS[variant.lora]
        files = (weights, origin / TRAINING_FILE, origin / HELD_OUT_FILE)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise MissingWeightsError(f"There is no fold model to adopt: {missing} are missing.")
        record = _read_json(origin / TRAINING_FILE)
        check_fold_record(record, variant, chosen, origin)
        if (folder / TRAINING_FILE).is_file():
            check_fold_record(_read_json(folder / TRAINING_FILE), variant, chosen, folder)
            yield f"{folder} holds a complete fold model: kept"
            continue
        folder.mkdir(parents=True, exist_ok=True)
        _link_or_copy(weights, folder / FOLD_WEIGHTS_FILE)
        _link_or_copy(origin / HELD_OUT_FILE, folder / HELD_OUT_FILE)
        _write_json(folder / TRAINING_FILE, record | {"adopted_from": _shown(origin)})
        yield f"adopted {origin} as {folder}"


def read_fold_models(
    variant: Variant, chosen: Mapping, folds: pd.Series, root: Path
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Read every fold model's record and out-of-fold scores back from the trained cache.

    Args:
        variant: The method's settings.
        chosen: The content of ``tuning/params.json``.
        folds: The dev fold per problem, indexed by ``sample_id``, in the dev order.
        root: The method's folder in the trained cache.

    Returns:
        The ``training.json`` of each fold model, by name, in fold order; and the out-of-fold
        scores of every dev problem, float64 (cast exactly from the float32 files), indexed like
        ``folds``, one column per focus tag.

    Raises:
        MissingWeightsError: If a fold model is not complete.
        ValueError: If one holds another recipe or configuration, or scores of another shape.
    """
    records, frames = {}, []
    for fold in sorted(folds.unique()):
        folder = root / fold_name(fold)
        if not (folder / TRAINING_FILE).is_file():
            raise MissingWeightsError(
                f"{folder} holds no complete fold model: run the train (or adopt) command first."
            )
        record = _read_json(folder / TRAINING_FILE)
        check_fold_record(record, variant, chosen, folder)
        index = folds.index[folds == fold]
        scores = np.load(folder / HELD_OUT_FILE)
        if scores.shape != (len(index), len(FOCUS_TAGS)):
            raise ValueError(
                f"{folder / HELD_OUT_FILE} has the shape {scores.shape}; fold {fold} has "
                f"{len(index)} problems and {len(FOCUS_TAGS)} focus tags."
            )
        records[fold_name(fold)] = record
        frames.append(
            pd.DataFrame(scores.astype(np.float64), index=index, columns=list(FOCUS_TAGS))
        )
    return records, pd.concat(frames).loc[folds.index]


def _link_or_copy(source: Path, target: Path) -> None:
    """Hard-link a file, or copy it across file systems, replacing what an interrupted run left.

    The target is removed first: writing into a leftover hard link would change its other path.
    """
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _shown(path: Path) -> str:
    """Name a path relative to the repository when it is inside it, so records hold no home."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, content: Mapping) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
