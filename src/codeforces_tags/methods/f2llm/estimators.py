"""The score estimators of the three frozen-encoder methods: probe, early and late fusion.

Each follows the score-estimator contract of ``CrossFittedTagThresholds``: ``fit(X, y)`` takes
the feature frame (``embedding.feature_frame``: the preprocessed problems and one float64
column per embedding dimension) and an int64 ``(n, n_tags)`` array, sets
``classes_ = np.arange(n_tags)``, and ``decision_function(X)`` returns float64 ``(n, n_tags)``.

- ``make_probe``: linear heads on the embedding blocks (``f2llm_probe``).
- ``make_early_fusion``: one linear SVM on the frozen TF-IDF feature blocks and the embedding
  blocks (``f2llm_early_fusion``).
- ``make_late_fusion``: ``ScoreFusion`` of the frozen TF-IDF scorer and the probe
  (``f2llm_late_fusion``).

TF-IDF is built from ``feature_blocks`` and the frozen ``ModelParams`` only, never from
``model.training``, whose functions read raw problems and take the dev folds.

Frozen import path: skops stores ``ScoreFusion`` in a saved late-fusion model by its import
path, ``codeforces_tags.methods.f2llm.estimators.ScoreFusion``. Renaming it, or moving it to
another module, makes saved ``f2llm_late_fusion`` folders unloadable.

This module is on the prediction path: it never imports mlflow, optuna or openai.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from codeforces_tags.config import MODEL_MAX_ITER, MODEL_SEED
from codeforces_tags.methods.f2llm.embedding import INPUTS, SOLUTION, STATEMENT, embedding_columns
from codeforces_tags.methods.f2llm.settings import (
    EARLY_FUSION,
    FUSION_INNER_FOLDS,
    LATE_FUSION,
    LBFGS_MAX_ITER,
    PROBE,
)
from codeforces_tags.model.features import feature_blocks
from codeforces_tags.model.params import CLASS_WEIGHTS, ModelParams
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.tuning.results import load_tuning_results

# The probe's heads, the values of "head" in PROBE_GRID.
HEADS = ("logistic_regression", "linear_svm", "ridge")
# The column ScoreFusion.fit reads the near-duplicate groups from (select_part appends it).
GROUP_COLUMN = "group"


def embedding_blocks(weights: dict[str, float], width: int) -> ColumnTransformer:
    """Build the embedding blocks: one standardised, L2-normalised block per weighted input.

    Each block, named ``f"{input}_embedding"``, scales its input's ``width`` columns to zero
    mean and unit variance (``StandardScaler``), then each row to unit L2 norm
    (``Normalizer``), and is weighted by ``weights[input]``. An input of weight 0 is left out.

    Args:
        weights: Weight per input, keys among ``INPUTS``; at least one above 0.
        width: The embedding width, the number of columns of each input.

    Returns:
        The unfitted transformer; the other columns are dropped.

    Raises:
        ValueError: If a key is not an input, if a weight is negative, or if no weight is
            above 0.
    """
    kept = _kept_inputs(weights)
    return ColumnTransformer(
        [
            (
                f"{name}_embedding",
                Pipeline([("scale", StandardScaler()), ("normalise", Normalizer())]),
                embedding_columns(name, width),
            )
            for name in kept
        ],
        transformer_weights={f"{name}_embedding": float(weights[name]) for name in kept},
        remainder="drop",
    )


def _kept_inputs(weights: dict[str, float]) -> list[str]:
    """Return the inputs of weight above 0, in ``INPUTS`` order, after checking the weights."""
    unknown = sorted(set(weights) - set(INPUTS))
    if unknown:
        raise ValueError(f"Unknown inputs {unknown}: the inputs are {INPUTS}.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"Input weights must be 0 or above, not {weights}.")
    kept = [name for name in INPUTS if weights.get(name, 0.0) > 0]
    if not kept:
        raise ValueError(f"Give at least one input a weight above 0, not {weights}.")
    return kept


def make_head(
    head: str, c: float, class_weight: str
) -> LogisticRegression | LinearSVC | RidgeClassifier:
    """Build one tag's linear model, the probe's head.

    Args:
        head: One of ``HEADS``: ``logistic_regression`` (lbfgs, ``LBFGS_MAX_ITER``
            iterations), ``linear_svm`` (``MODEL_MAX_ITER`` iterations, seeded with
            ``MODEL_SEED``) or ``ridge`` (closed form, ``alpha = 1 / (2 c)``: the penalty of
            the other two heads, per unit of loss).
        c: Inverse regularisation strength, above 0.
        class_weight: ``"none"`` or ``"balanced"`` (``CLASS_WEIGHTS``).

    Returns:
        The unfitted binary model.

    Raises:
        ValueError: If the head or the class weight is unknown, or if ``c`` is not above 0.
    """
    if class_weight not in CLASS_WEIGHTS:
        raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}, not {class_weight!r}.")
    if not c > 0:
        raise ValueError(f"c must be above 0, not {c}.")
    weight = None if class_weight == "none" else "balanced"
    if head == "logistic_regression":
        return LogisticRegression(solver="lbfgs", C=c, class_weight=weight, max_iter=LBFGS_MAX_ITER)
    if head == "linear_svm":
        return LinearSVC(C=c, class_weight=weight, max_iter=MODEL_MAX_ITER, random_state=MODEL_SEED)
    if head == "ridge":
        return RidgeClassifier(alpha=1 / (2 * c), class_weight=weight)
    raise ValueError(f"head must be one of {HEADS}, not {head!r}.")


def make_probe(params: dict, width: int) -> Pipeline:
    """Build the probe: the embedding blocks, then one head per tag.

    The statement block has weight 1; the solution block, weight ``params["solution_weight"]``
    (0 leaves it out).

    Args:
        params: A configuration of ``PROBE_GRID``: ``head``, ``c``, ``class_weight`` and
            ``solution_weight``.
        width: The embedding width.

    Returns:
        The unfitted pipeline: steps ``features`` and ``classifier`` (a
        ``OneVsRestClassifier``).
    """
    head = make_head(params["head"], params["c"], params["class_weight"])
    weights = {STATEMENT: 1.0, SOLUTION: params["solution_weight"]}
    return Pipeline(
        [
            ("features", embedding_blocks(weights, width)),
            ("classifier", OneVsRestClassifier(head)),
        ]
    )


def tfidf_params() -> ModelParams:
    """Return the frozen TF-IDF configuration, chosen by the TF-IDF tuning study.

    Returns:
        ``load_tuning_results().params``, read from ``assets/tfidf_baseline/``.
    """
    return load_tuning_results().params


def make_tfidf(params: ModelParams) -> Pipeline:
    """Build the TF-IDF scorer of a configuration, on preprocessed problems.

    Args:
        params: A linear-SVM configuration, such as ``tfidf_params()``.

    Returns:
        The unfitted pipeline: steps ``features`` (``feature_blocks(params)``) and
        ``classifier`` (a ``OneVsRestClassifier`` of ``LinearSVC(C=params.c)``, with the
        configuration's class weight).

    Raises:
        ValueError: If the configuration's classifier is not ``linear_svm``.
    """
    return Pipeline(
        [
            ("features", feature_blocks(params)),
            ("classifier", OneVsRestClassifier(_tfidf_classifier(params, params.c))),
        ]
    )


def make_raw_tfidf(params: ModelParams) -> Pipeline:
    """Build the TF-IDF scorer of a configuration, on raw problems (the reference rows).

    Args:
        params: A linear-SVM configuration, such as ``tfidf_params()``.

    Returns:
        The unfitted pipeline: step ``problems`` (``raw_to_problems``), then the steps of
        ``make_tfidf(params)``.

    Raises:
        ValueError: If the configuration's classifier is not ``linear_svm``.
    """
    return Pipeline([("problems", FunctionTransformer(raw_to_problems)), *make_tfidf(params).steps])


def make_early_fusion(params: dict, width: int, tfidf: ModelParams) -> Pipeline:
    """Build early fusion: one linear SVM on the TF-IDF and the embedding blocks.

    The TF-IDF blocks and their weights are the frozen configuration's; so are the classifier
    and its class weight. Only C and the embedding weights come from ``params``.

    Args:
        params: A configuration of ``EARLY_FUSION_GRID``: ``c``, ``statement_weight`` and
            ``solution_weight`` (0 leaves the solution block out).
        width: The embedding width.
        tfidf: The frozen TF-IDF configuration, a linear SVM.

    Returns:
        The unfitted pipeline: steps ``features`` (one ``ColumnTransformer`` holding the TF-IDF
        blocks, then the embedding blocks; sparse output) and ``classifier``.

    Raises:
        ValueError: If ``tfidf`` is not a linear SVM, or as ``embedding_blocks`` does.
    """
    lexical = feature_blocks(tfidf)
    embedding = embedding_blocks(
        {STATEMENT: params["statement_weight"], SOLUTION: params["solution_weight"]}, width
    )
    features = ColumnTransformer(
        [*lexical.transformers, *embedding.transformers],
        transformer_weights={**lexical.transformer_weights, **embedding.transformer_weights},
        remainder="drop",
        sparse_threshold=1.0,
    )
    classifier = OneVsRestClassifier(_tfidf_classifier(tfidf, params["c"]))
    return Pipeline([("features", features), ("classifier", classifier)])


def _tfidf_classifier(tfidf: ModelParams, c: float) -> LinearSVC:
    """Return the frozen TF-IDF's binary model, a linear SVM, at the given C."""
    if tfidf.classifier != "linear_svm":
        raise ValueError(
            f"The TF-IDF configuration's classifier is {tfidf.classifier!r}: these estimators "
            "build the frozen one, a linear_svm."
        )
    return make_head("linear_svm", c, tfidf.class_weight)


class ScoreFusion(BaseEstimator):  # pylint: disable=too-many-instance-attributes
    """Fuse two scorers' standardised scores: ``weight * z_first + (1 - weight) * z_second``.

    ``fit`` standardises each scorer's scores per tag with the mean and SD of its inner
    out-of-fold scores, those of ``cross_val_predict`` over ``GroupKFold(inner_folds)`` on the
    rows' near-duplicate groups (the ``group`` column), then fits both scorers on every row.
    The statistics thus describe scores of rows a scorer did not train on, as held-out and
    test rows are; scores of training rows are more spread out. A tag whose inner scores are
    constant keeps scale 1. Its import path is frozen (see the module docstring).

    Attributes:
        classes_: ``np.arange(n_tags)``.
        first_: ``first`` fitted on every row.
        second_: ``second`` fitted on every row.
        means_: (2, n_tags) float64, the mean inner out-of-fold score of each scorer and tag.
        scales_: (2, n_tags) float64, their SD (ddof 0), or 1 where it is 0.
    """

    def __init__(
        self,
        first: BaseEstimator | None = None,
        second: BaseEstimator | None = None,
        weight: float = 0.5,
        inner_folds: int = FUSION_INNER_FOLDS,
    ):
        """Set the two scorers and how their scores are fused.

        Args:
            first: The unfitted first scorer (the TF-IDF scorer in ``f2llm_late_fusion``).
            second: The unfitted second scorer (the probe).
            weight: The weight of the first scorer's standardised scores, from 0 to 1.
            inner_folds: The group folds of the inner out-of-fold scores.
        """
        self.first = first
        self.second = second
        self.weight = weight
        self.inner_folds = inner_folds

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ScoreFusion:
        """Learn the standardisation statistics out of fold, then fit both scorers.

        Args:
            X: The feature frame of the rows to fit on, with their ``group`` column.
            y: 0/1 labels, an (n, n_tags) array (or frame) aligned on ``X``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``X`` has no ``group`` column, or if ``weight`` is outside [0, 1].
        """
        if GROUP_COLUMN not in X:
            raise ValueError(
                "ScoreFusion.fit standardises on inner out-of-fold scores over the "
                f"near-duplicate groups: X needs the {GROUP_COLUMN!r} column that select_part "
                "appends."
            )
        if not 0 <= self.weight <= 1:
            raise ValueError(f"weight must be from 0 to 1, not {self.weight}.")
        labels = np.asarray(y, dtype=np.int64)
        held_out = [
            _inner_out_of_fold(scorer, X, labels, self.inner_folds)
            for scorer in (self.first, self.second)
        ]
        self.means_ = np.vstack([scores.mean(axis=0) for scores in held_out])
        spread = np.vstack([scores.std(axis=0) for scores in held_out])
        self.scales_ = np.where(spread > 0, spread, 1.0)
        self.first_ = clone(self.first).fit(X, labels)
        self.second_ = clone(self.second).fit(X, labels)
        self.classes_ = np.arange(labels.shape[1])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Score rows: the weighted sum of both scorers' standardised scores.

        Args:
            X: The feature frame of the rows to score (no ``group`` column needed).

        Returns:
            float64, one row per row of ``X``, one column per tag.

        Raises:
            NotFittedError: If the estimator is not fitted.
        """
        check_is_fitted(self, "scales_")
        first = (self.first_.decision_function(X) - self.means_[0]) / self.scales_[0]
        second = (self.second_.decision_function(X) - self.means_[1]) / self.scales_[1]
        return (self.weight * first + (1 - self.weight) * second).astype(np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the tags whose fused score is at least 0.

        Args:
            X: The feature frame of the rows to score.

        Returns:
            0/1 ints, ``(decision_function(X) >= 0)``.
        """
        return (self.decision_function(X) >= 0).astype(int)


def _inner_out_of_fold(
    scorer: BaseEstimator, X: pd.DataFrame, labels: np.ndarray, folds: int
) -> np.ndarray:
    """Score every row with a clone of ``scorer`` fitted on the other groups' folds."""
    return cross_val_predict(
        clone(scorer),
        X,
        labels,
        cv=GroupKFold(n_splits=folds),
        groups=X[GROUP_COLUMN],
        method="decision_function",
    )


def make_late_fusion(
    params: dict, width: int, probe_params: dict, tfidf: ModelParams
) -> ScoreFusion:
    """Build late fusion: the frozen TF-IDF scorer's scores fused with the probe's.

    Args:
        params: A configuration of ``LATE_FUSION_GRID``: ``weight``, TF-IDF's share.
        width: The embedding width.
        probe_params: The probe's tuned configuration (its ``tuning/params.json``).
        tfidf: The frozen TF-IDF configuration, a linear SVM.

    Returns:
        The unfitted ``ScoreFusion(make_tfidf(tfidf), make_probe(probe_params, width),
        params["weight"], FUSION_INNER_FOLDS)``.
    """
    return ScoreFusion(
        first=make_tfidf(tfidf),
        second=make_probe(probe_params, width),
        weight=params["weight"],
        inner_folds=FUSION_INNER_FOLDS,
    )


def make_estimator(
    name: str,
    params: dict,
    *,
    width: int,
    probe_params: dict | None,
    tfidf: ModelParams | None,
) -> Pipeline | ScoreFusion:
    """Build the unfitted score estimator of a frozen-encoder method (SetFit's is in setfit.py).

    Args:
        name: ``PROBE``, ``LATE_FUSION`` or ``EARLY_FUSION``.
        params: A configuration of the method's grid.
        width: The embedding width.
        probe_params: The probe's tuned configuration: needed by late fusion only.
        tfidf: The frozen TF-IDF configuration: needed by both fusions.

    Returns:
        ``make_probe``, ``make_late_fusion`` or ``make_early_fusion`` of the configuration.

    Raises:
        ValueError: If ``name`` is not one of the three, or if a configuration it needs is
            None.
    """
    if name == PROBE:
        return make_probe(params, width)
    if name not in (LATE_FUSION, EARLY_FUSION):
        raise ValueError(
            f"{name!r} has no estimator here: expected {PROBE}, {LATE_FUSION} or {EARLY_FUSION}."
        )
    if tfidf is None:
        raise ValueError(f"{name} needs the frozen TF-IDF configuration: pass tfidf_params().")
    if name == EARLY_FUSION:
        return make_early_fusion(params, width, tfidf)
    if probe_params is None:
        raise ValueError(f"{name} needs the probe's tuned configuration: pass load_params(PROBE).")
    return make_late_fusion(params, width, probe_params, tfidf)
