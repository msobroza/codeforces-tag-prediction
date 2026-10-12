"""The ModernBERT tagging methods: a bag of five fold models, a ``ScoreMethod`` (docs/adr/0005).

``scores`` is the mean of the fold models' probabilities; ``predict`` predicts each tag whose
score reaches its threshold, ``thresholds_``, chosen on the fold models' out-of-fold scores
(``per_tag_thresholds``). The method folder holds ``bag.json`` (the configuration, the thresholds,
the fold names), ``pretrained.json`` and ``trained.json`` (the weights, named with their hashes:
docs/adr/0004, 0006), and no weight.

This module is the methods' prediction path: it imports torch, transformers and peft, never
mlflow or optuna.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.devices import requested_device
from codeforces_tags.methods.modernbert.inputs import encode_problems
from codeforces_tags.methods.modernbert.modeling import FullBag, LoraBag, resolve_device
from codeforces_tags.methods.modernbert.settings import (
    BAG_FILE,
    PRETRAINED_FILE,
    TRAINED_FILE,
    VARIANTS,
    Variant,
)
from codeforces_tags.methods.modernbert.weights import check_pretrained, check_trained
from codeforces_tags.model.thresholds import apply_thresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems

BAG_FORMAT_VERSION = 1


class FineTunedBag:
    """A ``ScoreMethod``: the bag of a method's fold models, and its per-tag thresholds.

    Attributes:
        variant: The method's settings.
        params: The configuration its fold models were trained with, by parameter.
        thresholds_: The per-tag thresholds, float64, in the order of ``tags``.
        pretrained: The content of ``pretrained.json``.
        trained: The content of ``trained.json``.
        bag: The loaded fold models (``LoraBag`` or ``FullBag``).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        variant: Variant,
        params: Mapping,
        thresholds: np.ndarray,
        *,
        pretrained: Mapping,
        trained: Mapping,
        bag: LoraBag | FullBag,
    ) -> None:
        """Keep the fold models and what the method folder records.

        Args:
            variant: The method's settings.
            params: The configuration its fold models were trained with.
            thresholds: One threshold per focus tag, in ``FOCUS_TAGS`` order.
            pretrained: The content of ``pretrained.json``.
            trained: The content of ``trained.json``.
            bag: The loaded fold models.

        Raises:
            ValueError: If there is not one threshold per focus tag, or one is NaN (an infinite
                one, which ``per_tag_thresholds`` gives a tag with no positive, predicts nothing).
        """
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if thresholds.shape != (len(FOCUS_TAGS),) or np.isnan(thresholds).any():
            raise ValueError(f"Expected {len(FOCUS_TAGS)} thresholds, got {thresholds.tolist()}.")
        self.variant = variant
        self.params = dict(params)
        self.thresholds_ = thresholds
        self.pretrained = dict(pretrained)
        self.trained = dict(trained)
        self.bag = bag

    @classmethod
    def build(  # pylint: disable=too-many-arguments
        cls,
        variant: Variant,
        params: Mapping,
        thresholds: np.ndarray,
        pretrained: Mapping,
        trained: Mapping,
    ) -> FineTunedBag:
        """Load the fold models its two weight lists name, after checking every hash.

        Args:
            variant: The method's settings.
            params: The configuration the fold models were trained with.
            thresholds: One threshold per focus tag.
            pretrained: The content of ``pretrained.json``.
            trained: The content of ``trained.json``.

        Returns:
            The method, its fold models on ``variant.device``.

        Raises:
            MissingWeightsError: If a cache lacks a file (the message names the command).
            ChangedWeightsError: If a file's hash differs from its weight list's.
            RuntimeError: If ``variant.device`` names CUDA and torch sees no GPU.
        """
        snapshot = check_pretrained(pretrained)
        folds = check_trained(trained)
        device = resolve_device(variant.device)
        if variant.lora:
            bag = LoraBag.load(
                variant,
                snapshot,
                folds,
                rank=int(params["lora_rank"]),
                dropout=float(params["lora_dropout"]),
                device=device,
            )
        else:
            bag = FullBag.load(variant, snapshot, folds, device=device)
        return cls(variant, params, thresholds, pretrained=pretrained, trained=trained, bag=bag)

    @property
    def name(self) -> str:
        """Its key in ``METHODS``."""
        return self.variant.name

    @property
    def tags(self) -> tuple[str, ...]:
        """The output columns: the focus tags, in ``FOCUS_TAGS`` order."""
        return FOCUS_TAGS

    def scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Score raw problems with the bag: the mean of its fold models' probabilities.

        Args:
            raw_problems: Rows of ``read_samples``, indexed by ``sample_id``.

        Returns:
            One float64 probability per problem and focus tag, indexed like ``raw_problems``.
        """
        return self.frame(self.bag.scores(self._input_ids(raw_problems)), raw_problems)

    def fold_scores(
        self, raw_problems: pd.DataFrame, names: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Score raw problems with fold models of the bag, one after the other.

        ``bag_mean`` of every fold model's scores is what ``scores`` returns.

        Args:
            raw_problems: Rows of ``read_samples``, indexed by ``sample_id``.
            names: The fold models to score with; all of them by default.

        Returns:
            Each fold model's float32 probabilities, by name, one row per problem.
        """
        return self.bag.fold_scores(self._input_ids(raw_problems), names)

    def frame(self, scores: np.ndarray, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Label scores of raw problems as ``scores`` returns them.

        Args:
            scores: One row per problem of ``raw_problems``, one column per focus tag.
            raw_problems: The problems scored.

        Returns:
            The scores as float64, indexed like ``raw_problems``, one column per focus tag.
        """
        return pd.DataFrame(
            np.asarray(scores, dtype=np.float64), index=raw_problems.index, columns=list(self.tags)
        )

    def _input_ids(self, raw_problems: pd.DataFrame) -> list[list[int]]:
        """Preprocess raw problems (``raw_to_problems``, which checks them), then tokenise them."""
        problems = raw_to_problems(raw_problems)
        return encode_problems(
            problems, self.bag.tokenizer, self.variant.max_length, self.variant.code_min_tokens
        )

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict each focus tag whose score reaches its threshold (``>=``).

        Args:
            raw_problems: Rows of ``read_samples``, indexed by ``sample_id``.

        Returns:
            0/1 ints, indexed like ``raw_problems``, one column per focus tag.
        """
        return self.predict_from_scores(self.scores(raw_problems))

    def predict_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Predict each focus tag whose score, already computed, reaches its threshold (``>=``).

        Args:
            scores: What ``scores`` returned for some raw problems.

        Returns:
            0/1 ints, with the index and columns of ``scores``.
        """
        return apply_thresholds(scores, self.tag_thresholds())

    def tag_thresholds(self) -> pd.Series:
        """Return the per-tag thresholds that ``predict_from_scores`` applies (``>=``).

        Returns:
            ``thresholds_`` as floats indexed by the focus tags, in ``FOCUS_TAGS`` order: ``inf``
            for a tag that no dev problem has.
        """
        return pd.Series(self.thresholds_, index=list(self.tags), dtype=float)

    def save(self, folder: Path) -> None:
        """Write ``bag.json``, ``pretrained.json`` and ``trained.json`` into an existing folder.

        Args:
            folder: The method folder (``save_method`` writes method.json beside them).
        """
        bag = {
            "method": self.name,
            "format_version": BAG_FORMAT_VERSION,
            "params": self.params,
            "thresholds": self.thresholds_.tolist(),
            "folds": list(self.trained["folds"]),
        }
        for name, content in (
            (BAG_FILE, bag),
            (PRETRAINED_FILE, self.pretrained),
            (TRAINED_FILE, self.trained),
        ):
            (folder / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def load(folder: Path, *, device: str | None = None) -> FineTunedBag:
    """Load a method folder saved by ``FineTunedBag.save``, with its weights from the caches.

    Args:
        folder: The method folder.
        device: Where to load the fold models: when given, it wins whatever the environment
            says (``latency-cpu`` passes ``"cpu"``). When None (``load_method`` passes none),
            ``$CODEFORCES_DEVICE`` when it is set and not empty (``requested_device``), else
            the method's ``device`` setting (``"auto"``: CUDA when available); that setting
            must be one of ``config.DEVICES`` even when the variable is set.

    Returns:
        The method of the folder's ``bag.json``, its ``variant.device`` the device chosen.

    Raises:
        ValueError: If ``bag.json`` names no ModernBERT method or has another format version,
            or, when ``device`` is None, if the method's ``device`` setting or
            ``$CODEFORCES_DEVICE`` is not one of ``config.DEVICES``, or if the variable the
            F2LLM methods read before is set.
        RuntimeError: If the device chosen is CUDA and torch sees no GPU.
        MissingWeightsError: If a cache lacks a file (the message names the command).
        ChangedWeightsError: If a file's hash differs from its weight list's.
    """
    folder = Path(folder)
    bag = _read_json(folder / BAG_FILE)
    if bag.get("method") not in VARIANTS or bag.get("format_version") != BAG_FORMAT_VERSION:
        raise ValueError(
            f"{folder / BAG_FILE} names {bag.get('method')!r} (format "
            f"{bag.get('format_version')!r}): expected one of {sorted(VARIANTS)}, format "
            f"{BAG_FORMAT_VERSION}."
        )
    variant = VARIANTS[bag["method"]]
    chosen = requested_device(default=variant.device) if device is None else device
    return FineTunedBag.build(
        replace(variant, device=chosen),
        bag["params"],
        np.asarray(bag["thresholds"], dtype=np.float64),
        _read_json(folder / PRETRAINED_FILE),
        _read_json(folder / TRAINED_FILE),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
