"""Measure ``score_atol``, the tolerance of verify on the F2LLM methods' scores.

A frozen method's scores come from the fit-time path: embeddings read from the array cache,
computed on the GPU in batches, and for SetFit statements re-encoded by the body merged on the
GPU. A loaded method takes another path: it encodes each problem alone, live, SetFit's adapter
merged on its device. ``atol`` measures that path on the CPU, the device a user without a GPU
gets and the one furthest from the batched GPU path. The two paths give slightly different
scores.

Before the first freeze, each method is fitted once on part of dev, and both paths score the
same held-out problems: ``score_gaps`` for the three methods on frozen embeddings (one fitted
estimator, two feature frames), ``setfit_score_gap`` for SetFit (its fitted estimator against
the loaded method's path). ``suggest_score_atol`` turns the four gaps into the ``score_atol``
of the registry lines: ``SCORE_ATOL_FACTOR`` times the largest, rounded up to one significant
digit.

Run by ``python -m codeforces_tags.methods.f2llm atol``; it imports neither mlflow nor optuna.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from codeforces_tags.config import FOCUS_TAGS, TUNING_BLAS_THREADS
from codeforces_tags.methods.f2llm.encoder import EncoderConfig, PretrainedRecord
from codeforces_tags.methods.f2llm.method import F2LLMMethod, setfit_adapter
from codeforces_tags.methods.f2llm.settings import SCORE_ATOL_FACTOR, SETFIT_METHOD


def score_gaps(
    build: Callable[[], object],
    fit_rows: pd.DataFrame,
    fit_labels: np.ndarray,
    reference: pd.DataFrame,
    live: pd.DataFrame,
) -> float:
    """Fit one estimator, then measure how far its scores move between two feature frames.

    Args:
        build: Returns the unfitted score estimator (e.g. ``make_estimator`` of a method).
        fit_rows: The feature frame it is fitted on.
        fit_labels: Their 0/1 labels, an int64 (n, n_tags) array.
        reference: The held-out problems' feature frame on the fit-time path (cached
            embeddings).
        live: The same problems, in the same order, embedded on a loaded method's path (live,
            one problem at a time; ``atol`` embeds them on the CPU).

    Returns:
        The largest absolute difference between the estimator's ``decision_function`` on the
        two frames.

    Raises:
        ValueError: If ``reference`` and ``live`` do not hold the same problems in order.
    """
    _check_same_problems(reference, live)
    estimator = _fitted(build, fit_rows, fit_labels)
    return _largest_gap(estimator.decision_function(reference), estimator.decision_function(live))


def setfit_score_gap(
    build: Callable[[], object],
    fit_rows: pd.DataFrame,
    fit_labels: np.ndarray,
    reference: pd.DataFrame,
    raw_problems: pd.DataFrame,
) -> float:
    """Fit SetFit once, then compare its fit-time scores with a loaded method's.

    The fit-time scores re-encode the statements with the body merged on the estimator's
    device. The loaded method's merges the same adapter on ``$CODEFORCES_DEVICE`` (unset, the
    GPU when torch sees one, else the CPU; ``atol`` sets it to the CPU) and scores each
    problem alone, as ``F2LLMMethod`` does after ``load``. It takes the estimator's encoder
    settings and pretrained weights; an estimator that leaves either at None gets the package's
    (``EncoderConfig.from_settings()``, ``PretrainedRecord.from_settings()``), as
    ``SetFitEstimator`` does.

    Args:
        build: Returns the unfitted ``SetFitEstimator`` (e.g. a short smoke training on the GPU).
        fit_rows: The feature frame it is fitted on.
        fit_labels: Their 0/1 labels, an int64 (n, n_tags) array.
        reference: The held-out problems' feature frame on the fit-time path (cached
            embeddings).
        raw_problems: The same problems, in the same order, as ``read_samples()`` gives them.

    Returns:
        The largest absolute difference between the two paths' scores.

    Raises:
        ValueError: If ``reference`` and ``raw_problems`` do not hold the same problems in order.
    """
    _check_same_problems(reference, raw_problems)
    estimator = _fitted(build, fit_rows, fit_labels)
    config, pretrained = _encoder_of(estimator)
    loaded = F2LLMMethod(
        SETFIT_METHOD,
        estimator.head_,
        np.zeros(len(FOCUS_TAGS)),  # thresholds: only the scores are compared
        config,
        pretrained,
        adapter=setfit_adapter(estimator),
    )
    alone = [loaded.scores(raw_problems.iloc[[row]]) for row in range(len(raw_problems))]
    return _largest_gap(estimator.decision_function(reference), pd.concat(alone).to_numpy())


def suggest_score_atol(gaps: Iterable[float]) -> float:
    """Turn the measured gaps into ``score_atol``: a margin over the largest, kept readable.

    Args:
        gaps: The largest score gap of each method, from ``score_gaps`` and
            ``setfit_score_gap``.

    Returns:
        ``SCORE_ATOL_FACTOR`` times the largest gap, rounded up to one significant digit (a gap
        of 7.3e-6 gives 8e-5).

    Raises:
        ValueError: If there is no gap, a gap is negative or not finite, or every gap is 0: the
            two paths then never differed, so they did not run on different devices.
    """
    values = [float(gap) for gap in gaps]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"score_atol needs one finite, non-negative gap per method: {values}.")
    largest = SCORE_ATOL_FACTOR * max(values)
    if largest == 0:
        raise ValueError(
            "Every gap is 0: the fit-time and live paths gave the same scores, so they did not "
            "run on different devices. Measure the cached GPU embeddings against live CPU ones."
        )
    exponent = math.floor(math.log10(largest))
    # Rounded first, so that a quotient such as 7.000000000000001 stays 7.
    digit = math.ceil(round(largest / 10.0**exponent, 9))
    return float(f"{digit}e{exponent}")


def _fitted(build: Callable[[], object], rows: pd.DataFrame, labels: np.ndarray) -> object:
    """Fit a new estimator on the rows, BLAS limited as every fit of the package."""
    with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
        return build().fit(rows, labels)


def _encoder_of(estimator) -> tuple[EncoderConfig, PretrainedRecord]:
    """Return the encoder settings and the weights of a ``SetFitEstimator``, None resolved.

    A parameter left at None means the package's settings, as in ``SetFitEstimator.fit``.
    """
    config = (
        EncoderConfig.from_settings()
        if estimator.encoder_config is None
        else EncoderConfig.from_dict(estimator.encoder_config)
    )
    pretrained = (
        PretrainedRecord.from_settings()
        if estimator.pretrained is None
        else PretrainedRecord.from_dict(estimator.pretrained)
    )
    return config, pretrained


def _check_same_problems(reference: pd.DataFrame, other: pd.DataFrame) -> None:
    """Fail unless two frames hold the same problems, in the same order."""
    if not reference.index.equals(other.index):
        raise ValueError(
            "The two paths must score the same problems in the same order: the indexes differ."
        )


def _largest_gap(reference: np.ndarray, live: np.ndarray) -> float:
    """Return the largest absolute difference between two score arrays of the same shape."""
    return float(np.max(np.abs(np.asarray(reference, float) - np.asarray(live, float))))
