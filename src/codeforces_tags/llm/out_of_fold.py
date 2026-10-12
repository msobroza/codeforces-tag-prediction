"""Out-of-fold requests of prompted methods that learn from dev labels (e.g. few-shot).

``cross_requests`` is the request-level twin of ``methods.out_of_fold.cross_predict``: each dev
fold's requests come from a method fitted on the other folds, so what a problem's prompt shows
(e.g. a few-shot method's examples) never comes from its own fold (docs/adr/0001,
docs/adr/0003). ``llm run`` sends these requests and freezes their answers in the dev answers
file; the method's dev predictions are encoded from them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from codeforces_tags.llm.prompts import LlmRequest
from codeforces_tags.methods.out_of_fold import FittedMethod
from codeforces_tags.model.thresholds import check_dev_folds


def cross_requests(
    make_method: Callable[[], object],
    raw_problems: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
) -> list[LlmRequest]:
    """Build each fold's requests with a method fitted on the other folds (ADR 0001, ADR 0003).

    Args:
        make_method: Builds a fresh unfitted prompted method, called once per fold: its
            ``fit(raw_problems, labels)`` returns a method with ``requests(raw_problems)``
            (e.g. a ``PromptedLlm`` whose ``request_context`` depends on the fit).
        raw_problems: The dev problems' raw rows, indexed by ``sample_id``.
        labels: 0/1 labels, with the index of ``raw_problems``.
        folds: Dev fold per problem, with the index of ``raw_problems``.

    Returns:
        One request per problem, in the order of ``raw_problems``.

    Raises:
        ValueError: As ``check_dev_folds`` does (misaligned labels or folds, a missing or
            negative fold, fewer than two folds), or if the methods fitted for two folds are
            different methods.
        TypeError: If a method built by ``make_method`` lacks ``fit``, or a fitted one lacks
            ``requests``.
    """
    check_dev_folds(raw_problems.index, labels, folds)
    by_problem: dict[int, LlmRequest] = {}
    for fold in np.sort(folds.unique()):
        held_out = (folds == fold).to_numpy()
        fitted = _fit(make_method(), raw_problems[~held_out], labels[~held_out])
        by_problem.update(
            (request.sample_id, request) for request in fitted.requests(raw_problems[held_out])
        )
    names = sorted({request.method for request in by_problem.values()})
    if len(names) > 1:
        raise ValueError(
            f"The methods fitted for each fold must be one method; their requests name {names}."
        )
    return [by_problem[int(sample_id)] for sample_id in raw_problems.index]


def _fit(method: object, raw_problems: pd.DataFrame, labels: pd.DataFrame) -> object:
    """Fit a fresh prompted method, refusing one that cannot fit or build requests."""
    if not isinstance(method, FittedMethod):
        raise TypeError(f"{type(method).__name__} has no fit(raw_problems, labels).")
    fitted = method.fit(raw_problems, labels)
    if not callable(getattr(fitted, "requests", None)):
        raise TypeError(
            f"The fitted {type(fitted).__name__} has no requests(raw_problems): a fitted prompted "
            "method builds its requests with build_requests."
        )
    return fitted
