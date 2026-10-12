"""The four F2LLM tagging methods: live embeddings, a fitted scorer and per-tag thresholds.

``F2LLMMethod`` is the ``ScoreMethod`` of the four registered methods (``f2llm_probe``,
``f2llm_late_fusion``, ``f2llm_early_fusion`` and ``f2llm_setfit``). It embeds raw problems
live, with the weights its ``pretrained.json`` names and the constants of its ``encoder.json``,
builds their feature frame, and scores it with its fitted scorer; a tag is predicted where its
score reaches the tag's threshold. ``from_fit`` builds one from a fitted
``CrossFittedTagThresholds``, and ``load``, the loader the registry names, reads a method folder
back. A method has no array cache: it encodes every problem it is given.

A method folder holds: ``method.json`` (``save_method``), ``model.skops`` (the scorer;
SetFit's linear head only), ``thresholds.json``, ``encoder.json``, ``pretrained.json`` (never the
weights) and, for SetFit, ``adapter.safetensors`` and ``adapter.json``. ``load`` trusts only
the skops types of ``TRUSTED_TYPES`` for the stored method.

This module is the methods' prediction path: it imports torch, transformers and peft, never
mlflow, optuna or openai.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio

from codeforces_tags.config import FOCUS_TAGS, SKOPS_COMPRESSION
from codeforces_tags.methods import read_method_file
from codeforces_tags.methods.f2llm.embedding import live_feature_frame
from codeforces_tags.methods.f2llm.encoder import (
    ENCODER_FILE,
    PRETRAINED_FILE,
    Encoder,
    EncoderConfig,
    PretrainedRecord,
    frozen_encoder,
    loaded_device,
    read_json,
    write_json,
)
from codeforces_tags.methods.f2llm.setfit import (
    adapter_record,
    load_adapter,
    merged_model,
    save_adapter,
)
from codeforces_tags.methods.f2llm.settings import (
    EARLY_FUSION,
    LATE_FUSION,
    NAMES,
    PROBE,
    SETFIT,
    SETFIT_METHOD,
)
from codeforces_tags.model.thresholds import CrossFittedTagThresholds, apply_thresholds

MODEL_FILE = "model.skops"
THRESHOLDS_FILE = "thresholds.json"

# The functions of the frozen TF-IDF's statement and code blocks, which skops stores by import
# path (model/features.py, STORED_FUNCTIONS); the frozen TF-IDF keeps no cue block.
_TFIDF_FUNCTIONS = [
    "codeforces_tags.model.features.code_tokens",
    "codeforces_tags.model.features.normalise_statement",
    "codeforces_tags.model.features.statement_text",
]
# The types of each method's model.skops that skops does not trust by default, sorted as
# get_untrusted_types lists them: load trusts these and no other (pinned by a test).
TRUSTED_TYPES: dict[str, list[str]] = {
    EARLY_FUSION: _TFIDF_FUNCTIONS,
    LATE_FUSION: ["codeforces_tags.methods.f2llm.estimators.ScoreFusion", *_TFIDF_FUNCTIONS],
    PROBE: [],
    SETFIT_METHOD: [],
}


class F2LLMMethod:  # pylint: disable=too-many-instance-attributes
    """A ``ScoreMethod`` over F2LLM embeddings, encoded live.

    Attributes:
        scorer: The fitted score estimator: the refit estimator of a
            ``CrossFittedTagThresholds`` (SetFit: its linear head), whose
            ``decision_function`` takes the feature frame.
        thresholds: The per-tag thresholds, (8,) floats in ``FOCUS_TAGS`` order.
        reads_solution: Whether the scorer keeps the solution embedding at a weight above 0;
            when it does not, ``scores`` encodes no solution.
        config: The encoder constants.
        pretrained: The pretrained weights, checked by their sha256 when loaded.
        adapter: SetFit's adapter, ``(state dict, adapter.json values)``; None otherwise.
        device: Where it encodes: ``$CODEFORCES_DEVICE`` when it was built, else "cuda" when
            torch sees a GPU, else "cpu" (``loaded_device``).
        solution_encoder: The frozen encoder, shared through the frozen-encoder memo.
        statement_encoder: The frozen encoder, or for SetFit the base weights with the adapter
            merged in.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        scorer,
        thresholds: np.ndarray,
        config: EncoderConfig,
        pretrained: PretrainedRecord,
        *,
        adapter: tuple[dict, dict] | None = None,
    ) -> None:
        """Hold a fitted scorer, then load its encoders on the device of the environment.

        Args:
            name: One of the four method names (``settings.NAMES``).
            scorer: The fitted scorer of the feature frame.
            thresholds: One threshold per focus tag, in ``FOCUS_TAGS`` order.
            config: The encoder constants.
            pretrained: The pretrained weights.
            adapter: SetFit's ``(adapter state dict, adapter.json values)``, for SetFit only.

        Raises:
            ValueError: If ``name`` is not an F2LLM method, if ``adapter`` is given for another
                method than SetFit or missing for SetFit, if there is not one threshold per
                focus tag, or if ``$CODEFORCES_DEVICE`` names no device setting or the variable
                the F2LLM methods read before is set (``loaded_device``).
            RuntimeError: If ``$CODEFORCES_DEVICE`` is "cuda" and no GPU is available.
        """
        _check_method(name, adapter)
        self.thresholds = np.asarray(thresholds, dtype=float)
        if self.thresholds.shape != (len(FOCUS_TAGS),):
            raise ValueError(
                f"{name} needs one threshold per focus tag, {len(FOCUS_TAGS)}, not an array of "
                f"shape {self.thresholds.shape}."
            )
        self.reads_solution = _reads_solution(scorer)
        self._name = name
        self.scorer = scorer
        self.config = config
        self.pretrained = pretrained
        self.adapter = adapter
        self.device = loaded_device()
        self.solution_encoder = frozen_encoder(pretrained, config, self.device)
        self.statement_encoder = self._statement_encoder()

    @property
    def name(self) -> str:
        """Its key in ``METHODS``."""
        return self._name

    @property
    def tags(self) -> tuple[str, ...]:
        """The output columns: the focus tags, in ``FOCUS_TAGS`` order."""
        return FOCUS_TAGS

    def scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Embed raw problems live, then score them.

        On the GPU each problem is encoded in its own call, as the command line predicts one
        exercise: one batch of many problems holds the GPU at full power, which rebooted a
        machine once.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Float64 scores with the index of ``raw_problems``, one column per focus tag.
        """
        if raw_problems.empty:
            return pd.DataFrame(index=raw_problems.index, columns=list(FOCUS_TAGS), dtype=float)
        if self.device == "cuda":
            rows = range(len(raw_problems))
            return pd.concat([self._scores(raw_problems.iloc[[row]]) for row in rows])
        return self._scores(raw_problems)

    def _scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Embed raw problems live, in one call per input, then score them."""
        # A scorer that drops the solution block never reads its columns: encoding the solutions
        # would only spend latency.
        solution_encoder = self.solution_encoder if self.reads_solution else None
        frame = live_feature_frame(raw_problems, self.statement_encoder, solution_encoder)
        scores = np.asarray(self.scorer.decision_function(frame), dtype=np.float64)
        return pd.DataFrame(scores, index=raw_problems.index, columns=list(FOCUS_TAGS))

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict each focus tag whose score reaches its threshold (``>=``).

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Int 0/1 with the index of ``raw_problems``, one column per focus tag.
        """
        return self.predict_from_scores(self.scores(raw_problems))

    def predict_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Predict each focus tag whose score, already computed, reaches its threshold (``>=``).

        Args:
            scores: What ``scores`` returned for some raw problems.

        Returns:
            Int 0/1, with the index and columns of ``scores``.
        """
        return apply_thresholds(scores, self.tag_thresholds())

    def tag_thresholds(self) -> pd.Series:
        """Return the per-tag thresholds that ``predict_from_scores`` applies (``>=``).

        Returns:
            ``thresholds`` as floats indexed by the focus tags, in ``FOCUS_TAGS`` order.
        """
        return pd.Series(self.thresholds, index=list(self.tags), dtype=float)

    def save(self, folder: Path) -> None:
        """Write the method's own files into a method folder (``save_method`` adds method.json).

        Args:
            folder: An existing directory.
        """
        folder = Path(folder)
        sio.dump(self.scorer, folder / MODEL_FILE, compression=SKOPS_COMPRESSION)
        thresholds = dict(zip(FOCUS_TAGS, self.thresholds.tolist(), strict=True))
        text = json.dumps(thresholds, indent=2)
        (folder / THRESHOLDS_FILE).write_text(text + "\n", encoding="utf-8")
        write_json(folder / ENCODER_FILE, self.config.to_dict())
        write_json(folder / PRETRAINED_FILE, self.pretrained.to_dict())
        if self.adapter is not None:
            save_adapter(folder, *self.adapter)

    def _statement_encoder(self) -> Encoder:
        """Return the frozen encoder, or for SetFit the base weights with its adapter merged."""
        if self.adapter is None:
            return self.solution_encoder
        state, record = self.adapter
        model = merged_model(self.pretrained, state, record, self.device)
        return Encoder(self.solution_encoder.tokenizer, model, self.config, self.device)


def from_fit(
    name: str,
    fitted: CrossFittedTagThresholds,
    config: EncoderConfig,
    pretrained: PretrainedRecord,
) -> F2LLMMethod:
    """Build a method from a fitted ``CrossFittedTagThresholds`` and its encoder.

    Args:
        name: One of the four method names.
        fitted: Fitted with ``refit=True`` on the feature frame and the focus labels; for
            SetFit, around a ``SetFitEstimator``.
        config: The encoder constants the fit used.
        pretrained: The pretrained weights the fit used.

    Returns:
        The method: the refit estimator (SetFit: its head and adapter) and ``thresholds_``.

    Raises:
        ValueError: If ``fitted`` was fitted with ``refit=False``, or on other labels than the
            focus tags in ``FOCUS_TAGS`` order; or as ``F2LLMMethod`` does.
    """
    if not hasattr(fitted, "estimator_"):
        raise ValueError(
            f"{name} needs a CrossFittedTagThresholds fitted with refit=True: the method "
            "scores with its refit estimator."
        )
    if tuple(fitted.tags_) != FOCUS_TAGS:
        raise ValueError(
            f"{name} predicts the focus tags in FOCUS_TAGS order, {list(FOCUS_TAGS)}; it was "
            f"fitted on {list(fitted.tags_)}."
        )
    estimator = fitted.estimator_
    if name != SETFIT_METHOD:
        return F2LLMMethod(name, estimator, fitted.thresholds_, config, pretrained)
    return F2LLMMethod(
        name,
        estimator.head_,
        fitted.thresholds_,
        config,
        pretrained,
        adapter=setfit_adapter(estimator),
    )


def setfit_adapter(estimator) -> tuple[dict, dict]:
    """Return the adapter of a fitted ``SetFitEstimator``, as ``F2LLMMethod`` takes it.

    Args:
        estimator: A fitted ``SetFitEstimator``.

    Returns:
        ``(adapter state dict, adapter_record(settings))``: its settings are ``SETFIT`` when it
        was built without any.
    """
    settings = SETFIT if estimator.settings is None else estimator.settings
    return estimator.adapter_state_, adapter_record(settings)


def load(folder: Path) -> F2LLMMethod:
    """Load the F2LLM method saved in a method folder: the ``load`` its registry entry names.

    The method's name comes from ``method.json``; it decides which skops types are trusted and
    whether an adapter is read.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        The method, its encoders loaded on ``$CODEFORCES_DEVICE`` (by default "cuda" when
        torch sees a GPU, else "cpu").

    Raises:
        ValueError: If the folder holds another method than an F2LLM one, if its thresholds
            are not the focus tags in order, or if a pretrained file is missing or changed.
        skops.io.exceptions.UntrustedTypesFoundException: If ``model.skops`` holds a type
            outside the method's ``TRUSTED_TYPES``.
    """
    folder = Path(folder)
    name = read_method_file(folder)["method"]
    if name not in TRUSTED_TYPES:
        raise ValueError(f"{folder} holds the method {name!r}, not one of the F2LLM methods.")
    return F2LLMMethod(
        name,
        sio.load(folder / MODEL_FILE, trusted=TRUSTED_TYPES[name]),
        _read_thresholds(folder / THRESHOLDS_FILE),
        EncoderConfig.from_dict(read_json(folder / ENCODER_FILE)),
        PretrainedRecord.from_dict(read_json(folder / PRETRAINED_FILE)),
        adapter=load_adapter(folder) if name == SETFIT_METHOD else None,
    )


def _reads_solution(scorer) -> bool:
    """Whether a fitted scorer reads the solution embedding: its block kept at a weight above 0.

    ``embedding_blocks`` leaves a block of weight 0 out of its ``ColumnTransformer``, so the
    fitted ``features`` step names ``solution_embedding`` exactly when the block is kept.
    ``ScoreFusion`` reads it when either of its scorers does.
    """
    if hasattr(scorer, "first_"):  # ScoreFusion
        return _reads_solution(scorer.first_) or _reads_solution(scorer.second_)
    features = scorer.named_steps["features"]
    return any(name == "solution_embedding" for name, _, _ in features.transformers_)


def _check_method(name: str, adapter: tuple[dict, dict] | None) -> None:
    """Fail unless the name is an F2LLM method's, with an adapter exactly for SetFit."""
    if name not in NAMES:
        raise ValueError(f"{name!r} is not an F2LLM method: expected one of {list(NAMES)}.")
    if (adapter is not None) != (name == SETFIT_METHOD):
        raise ValueError(
            f"{name} {'takes no' if adapter is not None else 'needs an'} adapter: only "
            f"{SETFIT_METHOD} merges a LoRA adapter into its statement encoder."
        )


def _read_thresholds(path: Path) -> np.ndarray:
    """Read thresholds.json, refusing other tags or another order than FOCUS_TAGS."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    if list(stored) != list(FOCUS_TAGS):
        raise ValueError(
            f"{path} holds thresholds for {list(stored)}, not the focus tags in order: "
            f"{list(FOCUS_TAGS)}."
        )
    return np.array(list(stored.values()), dtype=float)
