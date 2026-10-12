"""The latency harness: time a frozen method once, then log its summary (docs/adr/0004).

``python -m codeforces_tags.tracking latency --method NAME`` loads the method folder of a frozen
method and times it on dev problems drawn with ``LATENCY_SEED``, then writes the timings once to
the method's ``latency.csv``. A later run times nothing, and reads no dataset: it logs the
summary of the committed file, so a fresh MLflow store is rebuilt without timing again.

- A local method: one ``predict`` call per problem after a warm-up (``time_single``), one call
  over all of them (``time_bulk``) and the loading (``time_load``). Timing never runs on cached
  arrays: the cache is a fit-time option (``cache.py``), and a method whose timed calls read it
  is refused.
- A method that calls an API (``MethodSpec.calls_api``): one ``apredict_with_answers`` call per
  problem, each a live request that costs money and is written to no answers file, so it needs a
  cost cap (``--max-cost-usd``); no warm-up, since each call opens its own client as a single
  prediction does. ``latency.csv`` also gets each request's tokens, cost and retries; a retried
  request's seconds include the SDK's backoff, and the summary names them. Its bulk time is never
  one call over many paid requests: it comes from the answers files, at the pace (``started_ns``
  and ``latency_s``, at the run's concurrency) of the requests ``llm run`` sent.
- A composite method over a method that calls an API (``depends_on``) is never timed: it has
  no answers files of its own. A calibrated twin (docs/adr/0011) sends its base's request, so
  its freeze copies its base's ``latency.csv``, which the harness logs, timing nothing; without
  one, it is refused.

Run it with the method's own dependency groups.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from codeforces_tags.cache import ArrayCache
from codeforces_tags.config import LATENCY_PROBLEMS, LATENCY_SEED, SPLIT_FILE
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE, load_run_record
from codeforces_tags.evaluation.timing import (
    LATENCY_CALL_COLUMNS,
    LATENCY_FILE,
    Latency,
    latency_summary,
    load_latency,
    save_latency,
    time_bulk,
    time_load,
    time_single,
)
from codeforces_tags.llm.answers import answer_cost_usd, in_flight_seconds, problem_list
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.llm.prompted import PromptedLlm
from codeforces_tags.llm.store import answers_file, read_answers
from codeforces_tags.methods import METHODS, MethodSpec, method_dir, model_dir
from codeforces_tags.provenance import RunRecord, check_dataset
from codeforces_tags.splitting import PARTS, load_split, select_part
from codeforces_tags.tracking.methods import log_run_extras


def run_latency(
    name: str,
    *,
    max_cost_usd: float | None = None,
    data_dir: str | Path | None = None,
    split_path: Path = SPLIT_FILE,
    registry: Mapping[str, MethodSpec] = METHODS,
    **store: Any,
) -> str:
    """Time a frozen method once, then log its latency summary to its method run.

    The first time, it checks the dataset, draws ``LATENCY_PROBLEMS`` dev problems with
    ``LATENCY_SEED``, restricts them to the method's ``input_columns``, times them and writes
    ``latency.csv`` into the method's directory. Every time, it then logs ``latency/median_s``,
    ``latency/p95_s``, ``latency/max_s`` and ``latency/load_s`` from that file to the method's
    run (``log_run_extras`` skips what the run has).

    Args:
        name: The method's registry key.
        max_cost_usd: For a method that calls an API, required when it is timed: no timed
            request is sent once the timed requests cost this much. None for any other method.
        data_dir: Dataset directory, resolved by ``resolve_data_dir``; read only when timing.
        split_path: The frozen split.
        registry: The registered methods.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        A line to print: the file was saved or read, with the summary.

    Raises:
        ValueError: If the method is not registered, is a composite method over a method that
            calls an API with no ``latency.csv`` yet, or has no method run in the store
            (``log_run_extras``); if a method that calls an API is to be timed without a cost
            cap, or another method is given one; or if the cap is not above 0.
        FileNotFoundError: If the method is not frozen (no run record), or an API method to be
            timed has no answers.
        TypeError: If an API method's loaded method has no ``apredict_with_answers``.
        RuntimeError: If a timed call of a local method read the array cache, or an API
            method's timed requests reached the cost cap (nothing is saved).
    """
    spec = _registered(name, registry)
    directory = method_dir(name)
    path = directory / LATENCY_FILE
    saved = not path.exists()
    _check_composite(name, spec, timing=saved)
    _check_cost_cap(name, spec, max_cost_usd, timing=saved)
    record = _frozen_record(directory)
    if saved:
        check_dataset(data_dir)
        problems = _latency_problems(spec, data_dir, split_path)
        if spec.calls_api:
            timings = _time_requests(name, problems, max_cost_usd, registry)
        else:
            timings = _time_locally(name, problems, registry)
        save_latency(directory, timings)
    timings = load_latency(directory)
    log_run_extras(record.recipe_sha256, metrics=_metrics(timings), **store)
    return f"{'saved' if saved else 'read'} {path}: {_summary(timings)}; logged to MLflow"


def _registered(name: str, registry: Mapping[str, MethodSpec]) -> MethodSpec:
    """Return the spec of a registered method that the harness times."""
    if name not in registry:
        raise ValueError(f"{name!r} is not registered: the registry has {sorted(registry)}.")
    return registry[name]


def _check_composite(name: str, spec: MethodSpec, *, timing: bool) -> None:
    """Refuse to time a composite method over an API method: its latency.csv is copied."""
    if timing and spec.calls_api and spec.depends_on:
        raise ValueError(
            f"{name} is a composite method over a method that calls an API (depends_on "
            f"{list(spec.depends_on)}): the harness times a method that sends requests of its "
            "own, and a calibrated twin's freeze copies its base's latency.csv, which the "
            f"harness then logs (docs/adr/0011); {name} has none yet."
        )


def _check_cost_cap(
    name: str, spec: MethodSpec, max_cost_usd: float | None, *, timing: bool
) -> None:
    """Refuse to time an API method without a cost cap, and a cap that has no use or no room."""
    if max_cost_usd is None:
        if timing and spec.calls_api:
            raise ValueError(
                f"{name} calls an API: timing it sends {LATENCY_PROBLEMS} live requests, which "
                "cost money, so it needs a cost cap (--max-cost-usd)."
            )
        return
    if not spec.calls_api:
        raise ValueError(
            f"{name} sends no paid request: a cost cap (--max-cost-usd) is for a method that "
            "calls an API."
        )
    if not max_cost_usd > 0:
        raise ValueError(f"The cost cap must be above 0, not {max_cost_usd}.")


def _frozen_record(directory: Path) -> RunRecord:
    """Read the run record of a frozen method."""
    if not (directory / RUN_RECORD_FILE).is_file():
        raise FileNotFoundError(
            f"{directory} holds no {RUN_RECORD_FILE}: run the method's freeze first."
        )
    return load_run_record(directory)


def _time_locally(name: str, problems: pd.DataFrame, registry: Mapping[str, MethodSpec]) -> Latency:
    """Time a local method's frozen folder on the drawn dev problems."""
    method, load_s = time_load(model_dir(name), registry=registry)
    reads = ArrayCache.reads
    seconds = time_single(method, problems)
    bulk_s = time_bulk(method, problems)
    if ArrayCache.reads != reads:
        raise RuntimeError(
            f"{name} read the array cache while it was timed: the cache is a fit-time option, so "
            "a loaded method has none, and latency is never timed on cached arrays."
        )
    return Latency(seconds, bulk_s, load_s)


def _time_requests(
    name: str, problems: pd.DataFrame, max_cost_usd: float, registry: Mapping[str, MethodSpec]
) -> Latency:
    """Time an API method's frozen folder on the drawn dev problems, one live request each."""
    method, load_s = time_load(model_dir(name), registry=registry)
    if not callable(getattr(method, "apredict_with_answers", None)):
        raise TypeError(
            f"{name} has no apredict_with_answers: the latency of a method that calls an API is "
            "timed through it, as PromptedLlm gives it, to keep each request's tokens and cost."
        )
    bulk_s = _bulk_from_answers(name, len(problems))
    seconds, calls = _time_each_request(name, method, problems, max_cost_usd)
    return Latency(seconds, bulk_s, load_s, calls)


def _bulk_from_answers(name: str, problems: int) -> float:
    """Return the seconds that ``problems`` problems took at the pace of the answers files."""
    answers = [answer for part in PARTS for answer in read_answers(answers_file(name, part))]
    if not answers:
        raise FileNotFoundError(
            f"{name} has no answers: the bulk time of a method that calls an API comes from the "
            f"answers files that `llm run` writes, and none is in {method_dir(name)}."
        )
    return in_flight_seconds(answers) * problems / len(answers)


def _time_each_request(
    name: str, method: PromptedLlm, problems: pd.DataFrame, max_cost_usd: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Time one live request per problem, in turn, until they cost ``max_cost_usd``.

    Returns each request's seconds and call (``LATENCY_CALL_COLUMNS``), indexed like the problems.
    """
    seconds, calls, spent = [], [], 0.0
    for position in range(len(problems)):
        if spent >= max_cost_usd:
            raise RuntimeError(
                f"{name}: the timed requests reached the cost cap of ${max_cost_usd} after "
                f"{position} of {len(problems)} problems, having cost ${spent:.6f}. latency.csv "
                "is written once every problem is timed, so nothing was saved: run it again with "
                "a higher --max-cost-usd."
            )
        one_problem = problems.iloc[position : position + 1]
        start = perf_counter()
        _, [answer] = asyncio.run(method.apredict_with_answers(one_problem))
        seconds.append(perf_counter() - start)
        calls.append(_call(answer))
        spent += calls[-1]["cost_usd"]
    index = problems.index
    return (
        pd.Series(seconds, index=index, name="seconds"),
        pd.DataFrame(calls, index=index, columns=list(LATENCY_CALL_COLUMNS)),
    )


def _call(answer: LlmAnswer) -> dict[str, float | int]:
    """Return what a timed request's answer records: its tokens, cost and retries."""
    return {
        "input_tokens": answer.usage["input_tokens"],
        "output_tokens": answer.usage["output_tokens"],
        "cost_usd": answer_cost_usd(answer)["total_cost"],
        "retries": answer.retries,
    }


def _latency_problems(
    spec: MethodSpec, data_dir: str | Path | None, split_path: Path
) -> pd.DataFrame:
    """Draw the dev problems to time, by sample id, with the columns the method reads."""
    dev = select_part(read_samples(data_dir), load_split(split_path), "dev")
    drawn = dev.sample(n=LATENCY_PROBLEMS, random_state=LATENCY_SEED).sort_index()
    return drawn[list(spec.input_columns)]


def _metrics(timings: Latency) -> dict[str, float]:
    """Return the run metrics of a method's timings: the summary and the load time."""
    summary = latency_summary(timings.seconds)
    return {f"latency/{key}": value for key, value in summary.items()} | {
        "latency/load_s": timings.load_s
    }


def _summary(timings: Latency) -> str:
    """Describe a method's timings in one line."""
    summary = latency_summary(timings.seconds)
    line = (
        f"{len(timings.seconds)} problems, median {summary['median_s']:.6f} s, "
        f"p95 {summary['p95_s']:.6f} s, max {summary['max_s']:.6f} s; "
    )
    if timings.calls is None:
        return line + f"one call {timings.bulk_s:.6f} s; load {timings.load_s:.6f} s"
    return (
        line + f"bulk {timings.bulk_s:.6f} s at the answers files' pace; "
        f"load {timings.load_s:.6f} s; {_calls_summary(timings.calls)}"
    )


def _calls_summary(calls: pd.DataFrame) -> str:
    """Describe the timed requests of an API method: tokens, cost, and the retried ones."""
    retried = calls.index[calls["retries"] > 0]
    flag = "no timed request was retried"
    if len(retried):
        flag = (
            f"retried {len(retried)} of the {len(calls)} timed requests, whose seconds include "
            f"the SDK's backoff: problems {problem_list(retried)}"
        )
    return (
        f"{calls['input_tokens'].sum():,} input and {calls['output_tokens'].sum():,} output "
        f"tokens, ${calls['cost_usd'].sum():.6f}; {flag}"
    )
