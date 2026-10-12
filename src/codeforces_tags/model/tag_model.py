"""The tag model: what turns a problem into its predicted focus tags, the method ``tfidf``.

A ``TagModel`` holds a fitted ``CrossFittedTagThresholds`` over ``build_pipeline(params)``: the
pipeline (preprocessing, the feature blocks, then one linear model per tag) refit on every dev
problem, and one threshold per tag, chosen on the out-of-fold scores. A tag is predicted when its
score reaches its threshold. It is built by ``training.fit_tag_model``, and reads raw problems.

It is the tagging method ``tfidf`` of ``METHODS``, of kind "score". ``save`` writes its method
files: ``model.skops``, the fitted estimator saved by skops (never a pickle) and compressed, and
``tag_model.json``, its tags and parameters. The module-level ``load`` reads them back, trusting
only ``STORED_TYPES``. Frozen import paths: ``model.skops`` names the types of ``STORED_TYPES``
by their import paths, and the registry names this module's ``load``; renaming or moving one of
them makes saved and logged tag models unloadable. This module is on the prediction path: it
never imports mlflow or optuna.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
from sklearn.pipeline import Pipeline

from codeforces_tags.config import SKOPS_COMPRESSION
from codeforces_tags.model.features import STORED_FUNCTIONS, block_feature_names
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds, apply_thresholds

NAME = "tfidf"  # the tag model's key in METHODS
MODEL_FILE = "model.skops"  # the fitted CrossFittedTagThresholds
TAG_MODEL_FILE = "tag_model.json"  # the tags and the ModelParams
# The types MODEL_FILE stores by their import paths, sorted as skops lists them: load trusts these
# and nothing else. A tag model without the cue block stores all but sample_io_cues.
STORED_TYPES: tuple[str, ...] = tuple(
    sorted(
        (
            *STORED_FUNCTIONS,
            "codeforces_tags.model.thresholds.CrossFittedTagThresholds",
            "codeforces_tags.preprocessing.pipeline.raw_to_problems",
        )
    )
)


@dataclass(frozen=True)
class TagModel:
    """A fitted pipeline and its per-tag thresholds, chosen out of fold.

    Attributes:
        estimator: The ``CrossFittedTagThresholds`` over ``build_pipeline(params)``, fitted with
            ``refit=True``: it holds the refit pipeline, the thresholds and the out-of-fold
            scores they were chosen on.
        params: The configuration the pipeline was built from.
    """

    estimator: CrossFittedTagThresholds
    params: ModelParams

    @property
    def name(self) -> str:
        """Its key in ``METHODS``: ``"tfidf"``."""
        return NAME

    @property
    def tags(self) -> tuple[str, ...]:
        """The tags the model predicts, in the label column order."""
        return self.estimator.tags_

    @property
    def pipeline(self) -> Pipeline:
        """The pipeline refit on every problem: ``preprocess``, ``features``, ``classifier``."""
        return self.estimator.estimator_

    @property
    def thresholds(self) -> pd.Series:
        """The per-tag threshold, named ``threshold``, indexed by the tags in the label order."""
        return pd.Series(self.estimator.thresholds_, index=self._tag_index(), name="threshold")

    def scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Score each tag of each problem: the higher, the likelier the tag.

        Args:
            raw_problems: Raw problems, one or more, as ``read_samples()`` gives them.

        Returns:
            The ``decision_function`` of each tag's linear model (logits for logistic
            regression), indexed like ``raw_problems``, one column per tag.

        Raises:
            ValueError: If the problems are already preprocessed, or lack a raw input column
                (``raw_to_problems``).
        """
        return pd.DataFrame(
            self.estimator.decision_function(raw_problems),
            index=raw_problems.index,
            columns=self._tag_index(),
        )

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict each tag whose score reaches the tag's threshold.

        Args:
            raw_problems: Raw problems, one or more, as ``read_samples()`` gives them.

        Returns:
            0/1 predictions, indexed like ``raw_problems``, one column per tag.
        """
        return self.predict_from_scores(self.scores(raw_problems))

    def predict_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Predict each tag whose score, already computed, reaches the tag's threshold.

        Args:
            scores: What ``scores`` returned for some raw problems.

        Returns:
            0/1 predictions, with the index and columns of ``scores``.
        """
        return apply_thresholds(scores, self.thresholds)

    def tag_thresholds(self) -> pd.Series:
        """Return the per-tag thresholds that ``predict_from_scores`` applies (``>=``).

        Returns:
            A copy of ``thresholds``: floats indexed by ``tags``, in order; ``inf`` for a tag that
            no dev problem has. A caller's change to it never reaches the model.
        """
        return self.thresholds.copy()

    def predicted_tags(self, raw_problems: pd.DataFrame) -> pd.Series:
        """List the predicted tags of each problem.

        Args:
            raw_problems: Raw problems, one or more, as ``read_samples()`` gives them.

        Returns:
            The list of predicted tags per problem, in the order of ``tags`` (empty when no
            tag is predicted), indexed like ``raw_problems``.
        """
        predicted = self.predict(raw_problems).to_numpy(dtype=bool)
        tags_per_problem = [
            [tag for tag, chosen in zip(self.tags, row, strict=True) if chosen] for row in predicted
        ]
        return pd.Series(
            tags_per_problem, index=raw_problems.index, name="predicted_tags", dtype=object
        )

    def out_of_fold_scores(self) -> pd.DataFrame:
        """Return the out-of-fold scores the thresholds were chosen on.

        Returns:
            Each fitted problem's score per tag, from the pipeline fitted on the other folds,
            indexed by ``sample_id`` in the order of the problems fitted, one column per tag.
        """
        return pd.DataFrame(
            self.estimator.oof_scores_,
            index=pd.Index(self.estimator.oof_index_, name="sample_id"),
            columns=self._tag_index(),
        )

    def feature_weights(self) -> pd.DataFrame:
        """Return what each feature adds to each tag's score, per unit of the feature.

        A block's features are multiplied by the block's weight before the linear models, so a
        feature's weight is its coefficient times its block's weight. The score of a tag is
        then its intercept plus the sum of each feature's value (as its block computes it, before
        the block weight) times its weight. For a sample I/O cue, the weight is what the cue
        adds to the score when it fires.

        Returns:
            One row per feature of the fitted blocks, indexed by (``block``, ``feature``): the
            words and n-grams of the statement and code vocabularies, ``SAMPLE_IO_CUES`` for
            the cues; one column per tag.
        """
        features = self.pipeline.named_steps["features"]
        index = pd.MultiIndex.from_tuples(
            [
                (block, name)
                for block in self.params.blocks
                for name in block_feature_names(features, block)
            ],
            names=["block", "feature"],
        )
        block_weights = index.get_level_values("block").map(features.transformer_weights)
        coefficients = np.column_stack(
            [model.coef_.ravel() for model in self.pipeline.named_steps["classifier"].estimators_]
        )
        return pd.DataFrame(
            coefficients * block_weights.to_numpy()[:, np.newaxis],
            index=index,
            columns=self._tag_index(),
        )

    def save(self, folder: Path) -> None:
        """Write the fitted estimator to ``model.skops``, its tags and parameters to JSON.

        ``tag_model.json`` holds the tags and ``ModelParams.to_dict()``. The estimator is
        compressed (``SKOPS_COMPRESSION``): the frozen one takes 53.85 MB uncompressed, above
        ``METHOD_FILE_MAX_BYTES``, and about 4 MB compressed.

        Args:
            folder: An existing directory.
        """
        folder = Path(folder)
        skops.io.dump(self.estimator, folder / MODEL_FILE, compression=SKOPS_COMPRESSION)
        stored = {"tags": list(self.tags), "params": self.params.to_dict()}
        (folder / TAG_MODEL_FILE).write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")

    def _tag_index(self) -> pd.Index:
        """Return the tags as the column index of the labels, named ``label``."""
        return pd.Index(self.tags, name="label")


def load(folder: Path) -> TagModel:
    """Load the tag model saved in a method folder: the ``load`` its registry entry names.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        The tag model.

    Raises:
        UntrustedTypesFoundException: If ``model.skops`` stores a type outside ``STORED_TYPES``
            (skops refuses to build it).
        ValueError: If ``tag_model.json`` stores other tags than the fitted estimator predicts,
            or invalid parameters (``ModelParams.from_dict``).
    """
    folder = Path(folder)
    stored = json.loads((folder / TAG_MODEL_FILE).read_text(encoding="utf-8"))
    estimator = skops.io.load(folder / MODEL_FILE, trusted=list(STORED_TYPES))
    if tuple(stored["tags"]) != estimator.tags_:
        raise ValueError(
            f"{folder / TAG_MODEL_FILE} stores the tags {stored['tags']}, but {MODEL_FILE} "
            f"predicts the tags {list(estimator.tags_)}."
        )
    return TagModel(estimator, ModelParams.from_dict(stored["params"]))
