"""The served records: what the model server answers for raw problems, one record per problem.

A problem's served record is what the command line prints for it, as one row: the tagging
method's name and its 0/1 focus tags; for a score method, each focus tag's score and per-tag
threshold; for a method that calls an API, whether its answer failed and what its request cost.
It comes from the command line's own ``run_method``, so the model server predicts as the command
line does: it first refuses the fields the command line refuses (``check_fields``: a blank
statement or solution), naming the rows of the request, then a score method scores each problem
once and takes its tags from those scores, and a method that calls an API, which takes exactly
one problem per request, is asked once and its answer priced.

The column names are shared with the HTTP client. This module imports neither mlflow nor torch.
"""

from __future__ import annotations

import pandas as pd

from codeforces_tags.cli import check_fields, run_method
from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods import MethodSpec, ScoreMethod, TaggingMethod

# The columns of a served record besides the focus tags and their scores and thresholds: the
# method's name, then, for a method that calls an API, whether its answer failed (the problem
# then gets no tag) and what its request cost, in USD.
METHOD_COLUMN = "method"
FAILED_COLUMN = "failed"
COST_COLUMN = "cost_usd"


def score_column(tag: str) -> str:
    """Name the column of a focus tag's score in a served record.

    Args:
        tag: A focus tag.

    Returns:
        ``"score: <tag>"``.
    """
    return f"score: {tag}"


def threshold_column(tag: str) -> str:
    """Name the column of a focus tag's per-tag threshold in a served record.

    Args:
        tag: A focus tag.

    Returns:
        ``"threshold: <tag>"``.
    """
    return f"threshold: {tag}"


def served_records(
    method: TaggingMethod, spec: MethodSpec, raw_problems: pd.DataFrame
) -> pd.DataFrame:
    """Predict raw problems as the command line does, and return one served record per problem.

    Args:
        method: The loaded tagging method.
        spec: Its spec: ``calls_api`` and ``input_columns``.
        raw_problems: Raw problems, with the method's input columns; other columns are dropped
            before the method sees them.

    Returns:
        A frame indexed like ``raw_problems``: ``method`` (the method's name), then one int 0/1
        column per focus tag, in ``FOCUS_TAGS`` order; for a ``ScoreMethod``, the float columns
        ``score: <tag>``, then ``threshold: <tag>`` (its ``tag_thresholds()``: ``inf`` for a tag
        it never predicts); for a method that calls an API, ``failed`` (bool) and ``cost_usd``
        (float).

    Raises:
        CliError: As ``check_fields``, before the method sees any problem: a problem without a
            description or a solution, or with a field that no method can read, named by its
            row in the request (its index label), never by a field's text; if a method that
            calls an API is not given exactly one problem: each problem is a paid request, and a
            request to the model server carries no cost cap, which more than one paid request
            needs; or as ``run_method``, e.g. when the API does not answer.
    """
    check_fields(raw_problems, sample_name=_request_row)
    if spec.calls_api and len(raw_problems) != 1:
        raise CliError(
            f"{method.name} sends a paid request per problem, and the model server asks a method "
            "that calls an API about exactly one problem per request: a request carries no cost "
            "cap, which more than one paid request needs (--max-cost-usd on the command line). "
            f"Send each problem in a request of its own: this request holds {len(raw_problems)}."
        )
    predictions = run_method(method, spec, raw_problems)
    columns = [predictions.tags]
    if isinstance(method, ScoreMethod):
        thresholds = method.tag_thresholds()
        columns += [
            predictions.scores.rename(columns=score_column),
            pd.DataFrame(
                {threshold_column(tag): float(thresholds[tag]) for tag in FOCUS_TAGS},
                index=predictions.tags.index,
            ),
        ]
    if spec.calls_api:
        columns.append(
            pd.DataFrame({FAILED_COLUMN: predictions.failed, COST_COLUMN: predictions.cost_usd})
        )
    records = pd.concat(columns, axis=1)
    records.insert(0, METHOD_COLUMN, method.name)
    return records


def _request_row(label: object) -> str:
    """Name a problem of a request by its row's index label, as MLflow's frame indexes it."""
    return f"request row {label}"
