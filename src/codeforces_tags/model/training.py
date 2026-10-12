"""Fit the tag model, and score the dev set out of fold.

``build_pipeline`` reads raw problems: it preprocesses them, builds the feature blocks, then
fits one linear model per tag. ``cross_fit`` and ``fit_tag_model`` fit it only through
``CrossFittedTagThresholds`` (docs/adr/0001-model-selection-protocol.md), which scores each dev
problem with a pipeline fitted on the folds that do not hold it, then predicts each fold with
per-tag thresholds chosen on the other folds' scores. ``fit_tag_model`` also refits the pipeline
on every given problem, with the thresholds chosen on all the out-of-fold scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from codeforces_tags.config import MODEL_MAX_ITER, MODEL_SEED
from codeforces_tags.model.features import feature_blocks
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.tag_model import TagModel
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems


def build_pipeline(params: ModelParams) -> Pipeline:
    """Build the unfitted pipeline of a configuration: raw problems in, a score per tag out.

    One-vs-rest fits one binary linear model per tag, all with the same settings. With
    ``class_weight="balanced"``, each tag's model weights a problem by n / (2 n_class) for
    its class in that tag.

    Args:
        params: The configuration.

    Returns:
        The steps ``preprocess`` (``raw_to_problems``, which refuses preprocessed rows),
        ``features`` (``feature_blocks(params)``) and ``classifier`` (a ``OneVsRestClassifier``
        of logistic regressions or linear SVMs). Nothing runs in parallel (no ``n_jobs``):
        joblib would write pickle files.
    """
    return Pipeline(
        [
            ("preprocess", FunctionTransformer(raw_to_problems)),
            ("features", feature_blocks(params)),
            ("classifier", OneVsRestClassifier(_linear_model(params))),
        ]
    )


def _linear_model(params: ModelParams) -> LogisticRegression | LinearSVC:
    """Return the binary model fitted for each tag, seeded with ``MODEL_SEED``."""
    class_weight = None if params.class_weight == "none" else "balanced"
    if params.classifier == "logistic_regression":
        return LogisticRegression(
            solver="liblinear", C=params.c, class_weight=class_weight, random_state=MODEL_SEED
        )
    return LinearSVC(
        C=params.c, class_weight=class_weight, random_state=MODEL_SEED, max_iter=MODEL_MAX_ITER
    )


@dataclass(frozen=True)
class OutOfFold:
    """The out-of-fold scores of the dev set, and the predictions of cross-fitted thresholds.

    Attributes:
        scores: Each problem's score per tag (``decision_function``: logits for logistic
            regression), from the pipeline fitted on the other folds. Index and columns are
            those of the problems and the labels.
        predictions: 0/1 predictions, fold k's with the thresholds of row k of
            ``fold_thresholds``.
        fold_thresholds: Per-tag thresholds of each fold, chosen on the other folds' scores;
            indexed by the sorted fold ids.
        thresholds: The final per-tag thresholds, chosen on every out-of-fold score: the
            thresholds of the tag model fitted on the whole dev set.
    """

    scores: pd.DataFrame
    predictions: pd.DataFrame
    fold_thresholds: pd.DataFrame
    thresholds: pd.Series


def out_of_fold(
    fitted: CrossFittedTagThresholds, labels: pd.DataFrame, folds: pd.Series
) -> OutOfFold:
    """Return the out-of-fold results of a fitted ``CrossFittedTagThresholds`` as frames.

    The estimator keeps them as bare arrays, which name no problem, tag or fold: the frames take
    their names from the labels and folds it was fitted on, which are checked against the
    estimator's.

    Args:
        fitted: A ``CrossFittedTagThresholds`` fitted on ``labels`` with ``groups=folds``.
        labels: The 0/1 labels it was fitted on, indexed by ``sample_id``.
        folds: The dev fold per problem it was fitted on, with the index of ``labels``.

    Returns:
        The out-of-fold scores, predictions and thresholds, with the index and columns of
        ``labels``; the fold thresholds are indexed by the sorted fold ids (index named like
        ``folds``).

    Raises:
        ValueError: If the estimator was fitted on other problems (or in another order), other
            tags or other folds than ``labels`` and ``folds`` hold.
    """
    fold_ids = pd.Index(np.sort(folds.unique()), name=folds.name)
    fitted_on = {
        "problems": (pd.Index(fitted.oof_index_), labels.index),
        "tags": (pd.Index(fitted.tags_), labels.columns),
        "folds": (pd.Index(fitted.fold_ids_), fold_ids),
    }
    for name, (fitted_values, given) in fitted_on.items():
        if not fitted_values.equals(given):
            raise ValueError(
                f"The estimator was fitted on other {name} than these labels and folds: pass the "
                "labels and folds it was fitted on, in the same order."
            )
    return OutOfFold(
        scores=pd.DataFrame(fitted.oof_scores_, index=labels.index, columns=labels.columns),
        predictions=pd.DataFrame(
            fitted.oof_predictions_, index=labels.index, columns=labels.columns
        ),
        fold_thresholds=pd.DataFrame(
            fitted.fold_thresholds_, index=fold_ids, columns=labels.columns
        ),
        thresholds=pd.Series(fitted.thresholds_, index=labels.columns, name="threshold"),
    )


def cross_fit(
    estimator: BaseEstimator, raw_problems: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> OutOfFold:
    """Score the dev set out of fold, and predict it with cross-fitted thresholds (ADR 0001).

    For each fold, a clone of ``estimator`` is fitted on the other folds and scores the fold.
    Fold k is then predicted with thresholds chosen on the other folds' scores and labels, so
    neither the model that scores fold k nor its thresholds were chosen on fold k's labels.
    ``CrossFittedTagThresholds`` does it all, without a refit.

    Args:
        estimator: The unfitted scorer of tags, e.g. ``build_pipeline(params)``; it is cloned,
            never fitted itself.
        raw_problems: Raw dev problems, e.g. ``select_part(read_samples(), split, "dev")``.
        labels: 0/1 focus-tag labels, with the index of ``raw_problems``.
        folds: Dev fold per problem (0 or above), with the index of ``raw_problems``.

    Returns:
        The out-of-fold scores, predictions and thresholds, from ``out_of_fold``.

    Raises:
        TypeError: If ``estimator`` is a ``ModelParams`` (pass ``build_pipeline(params)``), or
            as ``CrossFittedTagThresholds.fit`` does.
        ValueError: If ``labels`` or ``folds`` is not indexed like ``raw_problems``, if a fold is
            negative (the test set's -1), if there are fewer than two folds, or if
            ``raw_problems`` are already preprocessed.
    """
    if isinstance(estimator, ModelParams):
        raise TypeError(
            "cross_fit takes an unfitted estimator, not a ModelParams: pass build_pipeline(params)."
        )
    fitted = CrossFittedTagThresholds(estimator, refit=False).fit(
        raw_problems, labels, groups=folds
    )
    return out_of_fold(fitted, labels, folds)


def fit_tag_model(
    params: ModelParams, raw_problems: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series
) -> TagModel:
    """Cross-fit, choose the thresholds and refit on every problem, in one fit.

    ``CrossFittedTagThresholds`` scores the problems out of fold, chooses each tag's threshold on
    all the out-of-fold scores, then refits ``build_pipeline(params)`` on every problem: the tag
    model's thresholds are those of ``cross_fit`` on the same problems.

    Args:
        params: The configuration.
        raw_problems: Raw problems to fit on, e.g. the whole dev set,
            ``select_part(read_samples(), split, "dev")``.
        labels: 0/1 focus-tag labels, with the index of ``raw_problems``.
        folds: Dev fold per problem (0 or above), with the index of ``raw_problems``, e.g.
            ``dev_raw["fold"]``.

    Returns:
        The tag model.

    Raises:
        ValueError: If ``folds`` is indexed by the tags (thresholds, which the tag model now
            chooses itself), or as ``cross_fit`` does.
    """
    if isinstance(folds, pd.Series) and folds.index.equals(labels.columns):
        raise ValueError(
            "fit_tag_model takes the dev folds, dev_raw['fold'], and chooses the thresholds out "
            "of fold: it takes no thresholds."
        )
    estimator = CrossFittedTagThresholds(build_pipeline(params), refit=True)
    return TagModel(estimator.fit(raw_problems, labels, groups=folds), params)
