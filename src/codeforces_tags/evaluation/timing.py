"""Time a tagging method: one problem at a time, a whole part at once, and its loading.

PROBLEM.md asks that a single-problem prediction finish in under ten seconds, so the latency
harness (``python -m codeforces_tags.tracking latency``) times ``predict`` on one raw problem
at a time, after a warm-up, as the CLI would run it. ``time_bulk`` gives the throughput of one
call over many problems; it is never called on a method that sends paid requests, whose
``latency.csv`` also holds each timed request's tokens, cost and retries (``Latency.calls``).
The clock is ``time.perf_counter``. ``save_latency`` and ``load_latency`` write and read a
method's ``latency.csv`` once timed. No mlflow and no provider SDK here: the timers take any
``TaggingMethod``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import pandas as pd

from codeforces_tags.methods import METHODS, MethodSpec, TaggingMethod, load_method

LATENCY_FILE = "latency.csv"
# The two rows of latency.csv that are not a problem's: one predict call over all the timed
# problems, and loading the method folder.
LATENCY_BULK = "bulk"
LATENCY_LOAD = "load"
# The columns that latency.csv adds for a method that calls an API: each timed request's tokens,
# cost in USD and retries (a retried request's seconds include the SDK's backoff).
LATENCY_CALL_COLUMNS = ("input_tokens", "output_tokens", "cost_usd", "retries")
# The ones that count something, read back as integers.
_COUNT_COLUMNS = ("input_tokens", "output_tokens", "retries")


class Latency(NamedTuple):
    """What a method's ``latency.csv`` holds, in seconds.

    Attributes:
        seconds: One ``predict`` call per problem, indexed by ``sample_id``, named ``seconds``.
        bulk_s: One ``predict`` call over all those problems; for a method that calls an API,
            the time that as many problems took at the pace of its answers files.
        load_s: Loading the method folder.
        calls: For a method that calls an API, its timed requests' ``LATENCY_CALL_COLUMNS``,
            indexed like ``seconds``; None for any other method.
    """

    seconds: pd.Series
    bulk_s: float
    load_s: float
    calls: pd.DataFrame | None = None


def time_single(method: TaggingMethod, raw_problems: pd.DataFrame, *, warmup: int = 1) -> pd.Series:
    """Time ``method.predict`` on one raw problem at a time.

    Args:
        method: The tagging method, loaded.
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.
        warmup: Untimed calls first, on the problems in turn (the first, the second, ...), so
            that a cold start (imports, caches, a first connection) is not timed.

    Returns:
        The seconds of each problem's call, indexed by ``sample_id`` in the order of
        ``raw_problems``, named ``seconds``.

    Raises:
        ValueError: If ``raw_problems`` is empty or ``warmup`` is negative.
    """
    if raw_problems.empty:
        raise ValueError("raw_problems is empty: there is no problem to time.")
    if warmup < 0:
        raise ValueError(f"warmup must be 0 or more, not {warmup}.")
    for call in range(warmup):
        position = call % len(raw_problems)
        method.predict(raw_problems.iloc[position : position + 1])
    seconds = []
    for position in range(len(raw_problems)):
        one_problem = raw_problems.iloc[position : position + 1]
        start = perf_counter()
        method.predict(one_problem)
        seconds.append(perf_counter() - start)
    return pd.Series(seconds, index=raw_problems.index, name="seconds")


def time_bulk(method: TaggingMethod, raw_problems: pd.DataFrame) -> float:
    """Time one ``method.predict`` call over every problem.

    Never called on a method that sends paid requests (``MethodSpec.calls_api``).

    Args:
        method: The tagging method, loaded.
        raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

    Returns:
        The seconds of the call.
    """
    start = perf_counter()
    method.predict(raw_problems)
    return perf_counter() - start


def time_load(
    folder: Path, *, registry: Mapping[str, MethodSpec] = METHODS
) -> tuple[TaggingMethod, float]:
    """Time loading a method folder with ``load_method``.

    Args:
        folder: The method folder, e.g. ``model_dir(name)``.
        registry: The registered methods.

    Returns:
        The loaded method, and the seconds its loading took.
    """
    start = perf_counter()
    method = load_method(folder, registry=registry)
    return method, perf_counter() - start


def latency_summary(seconds: pd.Series) -> pd.Series:
    """Summarise single-problem latencies.

    Args:
        seconds: Seconds per problem, as ``time_single`` returns them.

    Returns:
        ``median_s``, ``p95_s`` (linear interpolation) and ``max_s``, in seconds.

    Raises:
        ValueError: If ``seconds`` is empty.
    """
    if seconds.empty:
        raise ValueError("seconds is empty: there is no time to summarise.")
    return pd.Series(
        {
            "median_s": float(seconds.median()),
            "p95_s": float(seconds.quantile(0.95)),
            "max_s": float(seconds.max()),
        }
    )


def save_latency(directory: Path, latency: Latency) -> None:
    """Write a method's ``latency.csv``, once.

    The file has the columns ``sample_id`` and ``seconds``, then, for a method that calls an
    API, ``LATENCY_CALL_COLUMNS``: a row per problem, in the order of ``latency.seconds``, then
    the rows ``bulk`` and ``load``, whose other columns are empty.

    Args:
        directory: The method's directory, ``method_dir(name)``; it must exist.
        latency: The timings.

    Raises:
        FileExistsError: If the file exists: the committed timings stay the record.
        ValueError: If ``latency.calls`` does not have one row per timed problem, in the order
            of ``latency.seconds``, and the columns ``LATENCY_CALL_COLUMNS``.
    """
    path = Path(directory) / LATENCY_FILE
    if path.exists():
        raise FileExistsError(
            f"{path} exists: latency.csv is written once, so that the committed timings stay "
            "the record."
        )
    rows = {str(sample_id): float(second) for sample_id, second in latency.seconds.items()}
    rows |= {LATENCY_BULK: latency.bulk_s, LATENCY_LOAD: latency.load_s}
    table = pd.Series(rows, name="seconds").rename_axis("sample_id").to_frame()
    if latency.calls is not None:
        table = table.join(_labelled_calls(latency))
    table.to_csv(path)


def load_latency(directory: Path) -> Latency:
    """Read a method's ``latency.csv``, bit for bit.

    Args:
        directory: The method's directory, ``method_dir(name)``.

    Returns:
        The timings: ``seconds`` is indexed by int ``sample_id`` (pass it, not the ``bulk`` and
        ``load`` rows, to ``plot_latency``), and so are ``calls`` when the file has them.

    Raises:
        ValueError: If the file has no ``bulk`` or no ``load`` row, or columns beside
            ``seconds`` that are not ``LATENCY_CALL_COLUMNS``.
    """
    path = Path(directory) / LATENCY_FILE
    table = pd.read_csv(
        path, dtype={"sample_id": str}, index_col="sample_id", float_precision="round_trip"
    )
    missing = [label for label in (LATENCY_BULK, LATENCY_LOAD) if label not in table.index]
    if missing:
        raise ValueError(
            f"{path} has no row for {missing}: it holds the rows of each problem, bulk and load."
        )
    problems = table.drop([LATENCY_BULK, LATENCY_LOAD])
    problems.index = problems.index.astype("int64")
    return Latency(
        problems["seconds"],
        float(table.loc[LATENCY_BULK, "seconds"]),
        float(table.loc[LATENCY_LOAD, "seconds"]),
        _read_calls(path, problems),
    )


def _labelled_calls(latency: Latency) -> pd.DataFrame:
    """Return the timed requests' calls under the row labels of ``latency.csv``."""
    calls = latency.calls
    if list(calls.columns) != list(LATENCY_CALL_COLUMNS) or not calls.index.equals(
        latency.seconds.index
    ):
        raise ValueError(
            "The calls of a method that calls an API must have one row per timed problem, in the "
            f"order of its seconds, and the columns {list(LATENCY_CALL_COLUMNS)}."
        )
    labelled = calls.astype(dict.fromkeys(_COUNT_COLUMNS, "Int64"))  # empty in bulk and load
    labelled.index = calls.index.astype(str)
    return labelled


def _read_calls(path: Path, problems: pd.DataFrame) -> pd.DataFrame | None:
    """Return the calls of the timed problems of ``latency.csv``; None for a local method's."""
    columns = [column for column in problems.columns if column != "seconds"]
    if not columns:
        return None
    if columns != list(LATENCY_CALL_COLUMNS):
        raise ValueError(
            f"{path} has the columns {columns} beside seconds: a method that calls an API adds "
            f"{list(LATENCY_CALL_COLUMNS)}, any other none."
        )
    return problems[columns].astype(dict.fromkeys(_COUNT_COLUMNS, "int64"))
