"""Out-of-fold predictions of tagging methods that learn 0/1 tags from dev labels.

``cross_predict`` predicts each dev fold with a method fitted on the other folds, so a fitted
0/1 method's dev scores are out of fold (docs/adr/0001, docs/adr/0004). A score method gets its
out-of-fold rows from ``CrossFittedTagThresholds`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from codeforces_tags.methods import TaggingMethod
from codeforces_tags.model.thresholds import check_dev_folds


@runtime_checkable
class FittedMethod(Protocol):  # pylint: disable=too-few-public-methods
    """An unfitted tagging method that learns from raw problems and their labels."""

    def fit(self, raw_problems: pd.DataFrame, labels: pd.DataFrame) -> TaggingMethod:
        """Fit on raw problems.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.
            labels: 0/1 labels, with the index of ``raw_problems``.

        Returns:
            The fitted tagging method.
        """


@runtime_checkable
class CompositeMethod(Protocol):  # pylint: disable=too-few-public-methods
    """An unfitted tagging method that learns from other methods' outputs (e.g. stacking)."""

    def fit_from(self, features: pd.DataFrame, labels: pd.DataFrame) -> TaggingMethod:
        """Fit on the base methods' outputs.

        Args:
            features: The bases' out-of-fold predictions or scores, one row per problem.
            labels: 0/1 labels, with the index of ``features``.

        Returns:
            The fitted tagging method; it also exposes ``predict_from(features)``, which
            returns int 0/1 tags with the index of ``features``.
        """


def cross_predict(
    make_method: Callable[[], object],
    raw_problems: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
    *,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict each fold with a method fitted on the other folds (ADR 0001, ADR 0004).

    Args:
        make_method: Builds a fresh unfitted method, called once per fold: a ``FittedMethod``,
            or a ``CompositeMethod`` when ``features`` is given.
        raw_problems: The dev problems' raw rows, indexed by ``sample_id``.
        labels: 0/1 labels, with the index of ``raw_problems``.
        folds: Dev fold per problem, with the index of ``raw_problems``.
        features: For a composite method, the bases' outputs, indexed like ``raw_problems``
            (their dev out-of-fold rows from ``load_predictions``, or their ``scores.csv``):
            the method is fitted with ``fit_from`` and predicts with ``predict_from`` on them,
            in place of the raw rows.

    Returns:
        The out-of-fold 0/1 predictions, with the index of ``raw_problems``, one column per tag
        of the fitted methods.

    Raises:
        ValueError: As ``check_dev_folds`` does (misaligned labels or folds, a missing or
            negative fold, fewer than two folds), if ``features`` is not indexed like
            ``raw_problems``, or if the methods fitted for two folds have different tags.
        TypeError: If a method built by ``make_method`` lacks ``fit`` (or ``fit_from``, with
            ``features``).
    """
    check_dev_folds(raw_problems.index, labels, folds)
    if features is not None and not features.index.equals(raw_problems.index):
        raise ValueError(
            "features must have the same index as problems: the same problems, in order."
        )
    fold_predictions = [
        _predict_fold(make_method(), raw_problems, labels, features, (folds == fold).to_numpy())
        for fold in np.sort(folds.unique())
    ]
    tags = {tuple(predictions.columns) for predictions in fold_predictions}
    if len(tags) > 1:
        raise ValueError(
            f"The methods fitted for each fold must have the same tags; they have {sorted(tags)}."
        )
    return pd.concat(fold_predictions).loc[raw_problems.index]


def _predict_fold(
    method: object,
    raw_problems: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame | None,
    held_out: np.ndarray,
) -> pd.DataFrame:
    """Fit a fresh method on the rows not held out, and predict the held-out rows."""
    if features is None:
        if not isinstance(method, FittedMethod):
            raise TypeError(f"{type(method).__name__} has no fit(raw_problems, labels).")
        fitted = method.fit(raw_problems[~held_out], labels[~held_out])
        predictions = fitted.predict(raw_problems[held_out])
    else:
        if not isinstance(method, CompositeMethod):
            raise TypeError(
                f"{type(method).__name__} has no fit_from(features, labels): pass features= "
                "for composite methods only."
            )
        fitted = method.fit_from(features[~held_out], labels[~held_out])
        predictions = fitted.predict_from(features[held_out])
    return predictions[list(fitted.tags)]
