"""Tests for the latency harness: a method is timed once, and its summary logged on each run."""

import dataclasses
import importlib
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import mlflow
import pandas as pd
import pytest

from codeforces_tags import methods
from codeforces_tags.config import (
    LATENCY_BUDGET_S,
    LATENCY_PROBLEMS,
    LATENCY_SEED,
    METHODS_ASSETS_DIR,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.evaluation.timing import (
    LATENCY_CALL_COLUMNS,
    LATENCY_FILE,
    Latency,
    latency_summary,
    load_latency,
    save_latency,
)
from codeforces_tags.llm import client
from codeforces_tags.llm.answers import cost_usd
from codeforces_tags.llm.prompted import PromptedLlm
from codeforces_tags.llm.prompts import prompt_snapshot
from codeforces_tags.llm.store import answers_file, append_answer
from codeforces_tags.methods import MODEL_FOLDER, MethodSpec, save_method
from codeforces_tags.methods.zero_shot.settings import METHOD_NAMES
from codeforces_tags.provenance import RunRecord, make_run_record, sha256_json
from codeforces_tags.splitting import PARTS, load_split
from codeforces_tags.tracking import latency
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.latency import run_latency
from codeforces_tags.tracking.methods import log_method, method_run_id

# Dataset-level metrics of a frozen method, as freeze logs them.
METRICS = {"dev-oof": {"macro_f1": 0.5}, "test": {"macro_f1": 0.25}}
# What the run's latency metrics are called.
LATENCY_METRICS = ["latency/load_s", "latency/max_s", "latency/median_s", "latency/p95_s"]
# Four of the six dev problems, drawn with seed 0 (LATENCY_PROBLEMS is set to 4), by sample id.
DRAWN = [2, 3, 4, 6]
LLM_FAKES = Path(__file__).parents[1] / "llm" / "fakes"
# The fake zero-shot prompted method of tests/llm/fakes, as an API method.
API_SPEC = MethodSpec("prompted_fakes.zero_shot", "zero_shot", ("numpy", "pandas"), calls_api=True)
# When each problem's frozen answer was sent (seconds after the first) and how long it took: the
# six dev requests overlap into 3.5 s and the two test ones into 0.75 s, after a gap that is not
# counted, so the answers files hold 8 answers in 4.25 s in flight.
FROZEN_CALLS = {
    1: (0.0, 1.0),
    2: (0.5, 1.0),
    3: (1.0, 1.0),
    4: (1.5, 1.0),
    5: (2.0, 1.0),
    6: (2.5, 1.0),
    7: (100.0, 0.5),
    8: (100.25, 0.5),
}
FIRST_NS = 1_791_000_000_000_000_000


class Env(NamedTuple):
    """What a latency test works with: the registered fakes, a dataset, the assets, a store."""

    registry: dict
    dataset: Path
    split_file: Path
    assets: Path
    store: dict


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record):
    """Write six dev problems (1-6) and two test problems (7-8), and return the dataset folder."""
    return write_samples(
        {
            sid: make_raw_record(
                prob_desc_description=f"Problem {sid} is about a tree.",
                source_code=f"print({sid})\n" * sid,
            )
            for sid in range(1, 9)
        }
    )


@pytest.fixture(name="split_file")
def split_file_fixture(tmp_path):
    """Write the frozen split of the dataset, and return its path."""
    split = pd.DataFrame(
        {
            "part": ["dev"] * 6 + ["test"] * 2,
            "fold": [0, 1, 2, 3, 4, 0, -1, -1],
            "group": range(1, 9),
        },
        index=pd.Index(range(1, 9), name="sample_id"),
    )
    path = tmp_path / "split.csv"
    split.to_csv(path)
    return path


@pytest.fixture(name="assets")
def assets_fixture(tmp_path, monkeypatch):
    """Return the folder of the frozen methods, which ``method_dir`` reads while the test runs."""
    assets = tmp_path / "assets"
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", assets)
    return assets


@pytest.fixture(name="env")
def env_fixture(registry, dataset, split_file, assets, store):
    """Bundle the registered fakes, the dataset, the assets folder and the store."""
    return Env(registry, dataset, split_file, assets, store)


@pytest.fixture(name="other_store")
def other_store_fixture(tmp_path, blank_database):
    """Return a second empty store, as ``store`` is, in another folder."""
    database = tmp_path / "other" / "mlflow.db"
    database.parent.mkdir()
    shutil.copy(blank_database, database)
    return {"tracking_uri": f"sqlite:///{database}", "artifact_dir": tmp_path / "other-artifacts"}


@pytest.fixture(name="spy")
def spy_fixture(monkeypatch):
    """Record what the timers and the dataset check are given, and let the timers run."""
    seen = {"single": [], "bulk": [], "dataset": []}
    time_single, time_bulk = latency.time_single, latency.time_bulk

    def single(method, raw_problems, **kwargs):
        seen["single"].append(raw_problems)
        return time_single(method, raw_problems, **kwargs)

    def bulk(method, raw_problems):
        seen["bulk"].append(raw_problems)
        return time_bulk(method, raw_problems)

    monkeypatch.setattr(latency, "time_single", single)
    monkeypatch.setattr(latency, "time_bulk", bulk)
    monkeypatch.setattr(latency, "check_dataset", seen["dataset"].append)
    monkeypatch.setattr(latency, "LATENCY_PROBLEMS", 4)
    return seen


def freeze(env, fake, *, registry=None, store=None):
    """Write a fake's method folder and run record as ``freeze`` does, and log its method run.

    With ``store`` (a store's arguments), the method run is logged there. Returns the run record.
    """
    registry = registry or env.registry
    directory = env.assets / fake.name
    save_method(fake, directory / MODEL_FOLDER)
    record = make_run_record(
        fake.name,
        sha256_json({"method": fake.name}),
        allow_dirty=True,
        requirements=registry[fake.name].requirements,
        method_entries={"supersedes": []},
        data_dir=env.dataset,
    )
    text = json.dumps(record.to_dict(), allow_nan=False)
    (directory / RUN_RECORD_FILE).write_text(text, encoding="utf-8")
    if store:
        log_method(
            directory / MODEL_FOLDER,
            record,
            METRICS,
            split=load_split(env.split_file),
            registry=registry,
            **store,
        )
    return record


def run(env, *, name="code_rule_fake", registry=None, store=None, max_cost_usd=None):
    """Run the harness on a frozen method, in the store of ``env`` unless another is given."""
    return run_latency(
        name,
        max_cost_usd=max_cost_usd,
        data_dir=env.dataset,
        split_path=env.split_file,
        registry=registry or env.registry,
        **(store or env.store),
    )


def method_run(record, store):
    """Return the id of a method's run, and its metrics."""
    run_id = method_run_id(record.recipe_sha256, **store)
    return run_id, mlflow.get_run(run_id).data.metrics


@pytest.mark.usefixtures("spy")
def test_run_latency_refuses_a_method_that_is_not_registered(env):
    with pytest.raises(ValueError, match="'missing' is not registered"):
        run(env, name="missing")


def test_run_latency_refuses_to_time_a_method_that_calls_an_api_without_a_cost_cap(env, spy):
    with pytest.raises(ValueError, match="fake-text calls an API: .* needs a cost cap"):
        run(env, name="fake-text", registry=env.registry | {"fake-text": API_SPEC})

    assert spy["dataset"] == []  # refused before anything is read


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan")])
def test_run_latency_refuses_a_cost_cap_that_is_not_above_zero(env, cap):
    registry = env.registry | {"fake-text": API_SPEC}

    with pytest.raises(ValueError, match="cost cap must be above 0"):
        run(env, name="fake-text", registry=registry, max_cost_usd=cap)


def test_run_latency_refuses_a_composite_method_over_an_api_method(env, spy):
    composite = MethodSpec(
        "conformance_fakes", "fitted", (), calls_api=True, depends_on=("fake-text",)
    )
    registry = env.registry | {"fake-text": API_SPEC, "stack_fake": composite}

    with pytest.raises(ValueError, match="stack_fake is a composite method over a method that"):
        run(env, name="stack_fake", registry=registry, max_cost_usd=1.0)
    with pytest.raises(ValueError, match="stack_fake has none yet"):
        run(env, name="stack_fake", registry=registry)

    assert spy["dataset"] == []


def test_run_latency_logs_the_copied_latency_of_a_composite_method_over_an_api_method(
    fakes, env, spy, monkeypatch
):
    # docs/adr/0011: a calibrated twin sends its base's request, so its freeze copies its
    # base's latency.csv; the harness logs it, with no cost cap, and times nothing.
    record = freeze(env, fakes.CodeRuleFake(), store=env.store)
    seconds = pd.Series([0.5, 1.5], index=pd.Index([2, 3], name="sample_id"), name="seconds")
    save_latency(env.assets / "code_rule_fake", Latency(seconds, bulk_s=2.0, load_s=0.25))
    composite = dataclasses.replace(
        env.registry["code_rule_fake"], calls_api=True, depends_on=("fake-text",)
    )
    registry = env.registry | {"fake-text": API_SPEC, "code_rule_fake": composite}
    for name in ("time_load", "read_samples"):
        monkeypatch.setattr(latency, name, None)  # calling either fails the test

    message = run(env, registry=registry)

    path = env.assets / "code_rule_fake" / LATENCY_FILE
    assert message.startswith(f"read {path}: 2 problems, median ")
    assert (spy["dataset"], spy["single"]) == ([], [])
    _, metrics = method_run(record, env.store)
    assert (metrics["latency/max_s"], metrics["latency/load_s"]) == (1.5, 0.25)


def test_run_latency_refuses_a_cost_cap_for_a_method_that_sends_no_paid_request(env):
    with pytest.raises(ValueError, match="code_rule_fake sends no paid request"):
        run(env, max_cost_usd=1.0)


def test_run_latency_refuses_a_method_that_is_not_frozen(env):
    with pytest.raises(FileNotFoundError, match="run the method's freeze first"):
        run(env)

    assert not env.assets.exists()


def test_run_latency_times_the_drawn_dev_problems_and_saves_latency_csv(fakes, env, spy):
    freeze(env, fakes.CodeRuleFake(), store=env.store)

    message = run(env)

    [timed] = spy["single"]
    assert list(timed.index) == DRAWN  # dev problems only, by sample id
    assert list(timed.columns) == ["prob_desc_description", "source_code"]  # the input columns
    assert [list(problems.index) for problems in spy["bulk"]] == [DRAWN]
    assert spy["dataset"] == [env.dataset]
    path = env.assets / "code_rule_fake" / LATENCY_FILE
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "sample_id,seconds"
    assert [row.split(",")[0] for row in rows[1:]] == ["2", "3", "4", "6", "bulk", "load"]
    saved = load_latency(path.parent)
    assert message == (
        f"saved {path}: 4 problems, median {saved.seconds.median():.6f} s, "
        f"p95 {saved.seconds.quantile(0.95):.6f} s, max {saved.seconds.max():.6f} s; "
        f"one call {saved.bulk_s:.6f} s; load {saved.load_s:.6f} s; logged to MLflow"
    )


@pytest.mark.usefixtures("spy")
def test_run_latency_logs_the_summary_to_the_method_run(fakes, env):
    record = freeze(env, fakes.CodeRuleFake(), store=env.store)

    run(env)

    saved = load_latency(env.assets / "code_rule_fake")
    summary = latency_summary(saved.seconds)
    _, metrics = method_run(record, env.store)
    assert {key: value for key, value in metrics.items() if key.startswith("latency/")} == {
        "latency/median_s": summary["median_s"],
        "latency/p95_s": summary["p95_s"],
        "latency/max_s": summary["max_s"],
        "latency/load_s": saved.load_s,
    }
    # Run metrics, linked to no model: the model keeps the method's dataset-level metrics only.
    [model] = mlflow.search_logged_models([use_store(**env.store)], output_format="list")
    logged = mlflow.get_logged_model(model.model_id).metrics
    assert {metric.key for metric in logged} == {"macro_f1"}


@pytest.mark.usefixtures("spy")
def test_latency_twice_retimes_nothing(fakes, env, monkeypatch):
    # A second run reads the committed latency.csv, times nothing, needs no
    # dataset, and logs nothing twice.
    record = freeze(env, fakes.CodeRuleFake(), store=env.store)
    run(env)
    path = env.assets / "code_rule_fake" / LATENCY_FILE
    frozen = (path.read_bytes(), path.stat().st_mtime_ns)
    for name in ("time_single", "time_bulk", "time_load", "read_samples", "check_dataset"):
        monkeypatch.setattr(latency, name, None)  # calling any of them fails the test

    message = run_latency("code_rule_fake", registry=env.registry, **env.store)

    assert (path.read_bytes(), path.stat().st_mtime_ns) == frozen
    assert message.startswith(f"read {path}: 4 problems, median ")
    run_id, _ = method_run(record, env.store)
    history = [mlflow.MlflowClient().get_metric_history(run_id, key) for key in LATENCY_METRICS]
    assert [len(values) for values in history] == [1] * 4


@pytest.mark.usefixtures("spy")
def test_latency_rebuilds_a_fresh_store_from_the_committed_csv(
    fakes, env, other_store, monkeypatch
):
    record = freeze(env, fakes.CodeRuleFake(), store=env.store)
    run(env)
    freeze(env, fakes.CodeRuleFake(), store=other_store)  # the method's own freeze mirrors it
    for name in ("time_single", "time_bulk", "time_load", "read_samples"):
        monkeypatch.setattr(latency, name, None)

    run(env, store=other_store)

    _, first = method_run(record, env.store)
    _, rebuilt = method_run(record, other_store)
    assert [rebuilt[key] for key in LATENCY_METRICS] == [first[key] for key in LATENCY_METRICS]


def test_run_latency_refuses_a_method_whose_timed_calls_read_the_array_cache(
    fakes, env, spy, tmp_path, monkeypatch
):
    monkeypatch.setattr(fakes, "CACHE_DIR", tmp_path / "cache")
    spec = MethodSpec(
        module="conformance_fakes",
        kind="other",
        requirements=(),
        input_columns=("prob_desc_description",),
    )
    registry = env.registry | {"cache_reading_fake": spec}
    freeze(env, fakes.CacheReadingFake(), registry=registry)

    with pytest.raises(RuntimeError, match="read the array cache"):
        run(env, name="cache_reading_fake", registry=registry)

    assert not (env.assets / "cache_reading_fake" / LATENCY_FILE).exists()
    assert len(spy["single"]) == 1


@pytest.mark.usefixtures("spy")
def test_run_latency_without_a_method_run_keeps_what_it_timed(fakes, env):
    freeze(env, fakes.CodeRuleFake())

    with pytest.raises(ValueError, match="run the method's freeze first"):
        run(env)

    assert (env.assets / "code_rule_fake" / LATENCY_FILE).exists()


def drawn_dev_problems():
    """Return the sample ids of the dev problems the harness times, drawn from the frozen split."""
    split = load_split()
    dev = pd.DataFrame(index=pd.Index(sorted(split.index[split["part"] == "dev"])))
    return list(dev.sample(n=LATENCY_PROBLEMS, random_state=LATENCY_SEED).sort_index().index)


@pytest.mark.parametrize(
    "name",
    [
        "f2llm_early_fusion",
        "f2llm_late_fusion",
        "f2llm_probe",
        "label_prior",
        "modernbert_base_full",
        "modernbert_lora",
        "tfidf",
    ],
)
def test_the_committed_latency_of_a_method_is_the_drawn_dev_problems_under_the_budget(name):
    # Needs no dataset: the draw is a function of the frozen split, and latency.csv is committed.
    timings = load_latency(METHODS_ASSETS_DIR / name)

    assert list(timings.seconds.index) == drawn_dev_problems()
    assert len(timings.seconds) == 50
    assert timings.seconds.max() < LATENCY_BUDGET_S  # the CLI's eligibility limit (ADR 0004)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_the_committed_latency_of_an_api_method_is_the_drawn_dev_problems_with_their_calls(name):
    # Needs no dataset. Each timed problem was one live request, whose tokens and cost latency.csv
    # keeps. The budget is not asserted: an API method slower than LATENCY_BUDGET_S is a reference
    # the CLI never chooses (ADR 0004), not an error, and notebook 05 reports it.
    timings = load_latency(METHODS_ASSETS_DIR / name)

    assert list(timings.seconds.index) == drawn_dev_problems()
    assert len(timings.seconds) == 50
    assert list(timings.calls.columns) == list(LATENCY_CALL_COLUMNS)
    assert timings.calls.index.equals(timings.seconds.index)
    assert (timings.calls[["input_tokens", "output_tokens", "cost_usd"]] > 0).all().all()
    assert min(timings.seconds.min(), timings.bulk_s, timings.load_s) > 0


# Where a method timed on the GPU keeps its CPU timings, beside the framework's latency.csv.
CPU_LATENCY_DIR = Path("analysis", "cpu")


@pytest.mark.parametrize(
    "name",
    [
        "f2llm_early_fusion",
        "f2llm_late_fusion",
        "f2llm_probe",
        "modernbert_base_full",
        "modernbert_lora",
    ],
)
def test_a_method_timed_on_the_gpu_keeps_its_cpu_timings_of_the_same_problems(name):
    # Needs no dataset. latency.csv, which the CLI and the notebooks read, was timed on the GPU;
    # analysis/cpu/latency.csv keeps the same drawn problems timed on the CPU, slower.
    gpu = load_latency(METHODS_ASSETS_DIR / name)
    cpu = load_latency(METHODS_ASSETS_DIR / name / CPU_LATENCY_DIR)

    assert list(cpu.seconds.index) == drawn_dev_problems()
    assert min(cpu.seconds.min(), cpu.bulk_s, cpu.load_s) > 0
    assert gpu.seconds.median() < cpu.seconds.median()
    assert gpu.seconds.max() < cpu.seconds.max()


def usage_of(sample_id):
    """Return the tokens of the fake Ask's answer to a problem: more input for a larger id."""
    return {
        "input_tokens": 100 + sample_id,
        "cached_tokens": 0,
        "output_tokens": 5,
        "reasoning_tokens": 0,
    }


class Api(NamedTuple):
    """A frozen API method, ``fake-text``, with what its fake Ask was sent."""

    registry: dict
    record: RunRecord
    sent: list


@pytest.fixture(name="api")
def api_fixture(env, make_answer, monkeypatch):
    """Freeze ``fake-text`` with its answers files and method run; fake its Ask and the clock.

    The fake Ask moves the harness's clock by 0.25 s per unit of a problem's sample id, and
    answers with ``usage_of`` that problem and one retry for problem 3. The answers files hold
    ``FROZEN_CALLS``.
    """
    monkeypatch.syspath_prepend(str(LLM_FAKES))
    registry = env.registry | {"fake-text": API_SPEC}
    spec = importlib.import_module("prompted_fakes.zero_shot").llm_spec("fake-text")
    method = PromptedLlm(spec, prompt_snapshot(spec, registry=registry))
    record = freeze(env, method, registry=registry, store=env.store)
    for sample_id, (start_s, seconds) in FROZEN_CALLS.items():
        answer = make_answer(
            request_sha256=f"{sample_id:064x}",
            sample_id=sample_id,
            started_ns=FIRST_NS + round(start_s * 1e9),
            latency_s=seconds,
        )
        append_answer(answers_file("fake-text", "dev" if sample_id <= 6 else "test"), answer)
    sent, clock = [], {"now": 0.0}

    async def ask(request):
        sent.append(request.sample_id)
        clock["now"] += 0.25 * request.sample_id
        return make_answer(
            request_sha256=request.sha256(),
            sample_id=request.sample_id,
            method=request.method,
            usage=usage_of(request.sample_id),
            retries=int(request.sample_id == 3),
        )

    monkeypatch.setattr(client, "ask_for", lambda settings: ask)
    monkeypatch.setattr(latency, "perf_counter", lambda: clock["now"])
    return Api(registry, record, sent)


def answers_bytes():
    """Return the bytes of each answers file of ``fake-text``, by part."""
    return {part: answers_file("fake-text", part).read_bytes() for part in PARTS}


def test_latency_of_an_api_method_sends_one_request_per_drawn_problem_and_records_each_call(
    api, env, spy
):
    frozen = answers_bytes()

    message = run(env, name="fake-text", registry=api.registry, max_cost_usd=1.0)

    assert api.sent == DRAWN  # one live request per problem, and no warm-up request
    assert spy["single"] == spy["bulk"] == []  # never one call over every problem
    assert answers_bytes() == frozen  # the timed requests are written to no answers file
    path = env.assets / "fake-text" / LATENCY_FILE
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "sample_id,seconds,input_tokens,output_tokens,cost_usd,retries"
    assert [row.split(",")[0] for row in rows[1:]] == ["2", "3", "4", "6", "bulk", "load"]
    saved = load_latency(path.parent)
    assert saved.seconds.to_dict() == {2: 0.5, 3: 0.75, 4: 1.0, 6: 1.5}
    assert saved.bulk_s == 2.125  # 4 problems at the answers files' pace: 8 answers in 4.25 s
    costs = [cost_usd(usage_of(sample_id), "gpt-6-luna")["total_cost"] for sample_id in DRAWN]
    assert saved.calls.to_dict("list") == {
        "input_tokens": [102, 103, 104, 106],
        "output_tokens": [5, 5, 5, 5],
        "cost_usd": costs,
        "retries": [0, 1, 0, 0],
    }
    summary = latency_summary(saved.seconds)
    assert message == (
        f"saved {path}: 4 problems, median {summary['median_s']:.6f} s, "
        f"p95 {summary['p95_s']:.6f} s, max 1.500000 s; bulk 2.125000 s at the answers files' "
        f"pace; load {saved.load_s:.6f} s; 415 input and 20 output tokens, ${sum(costs):.6f}; "
        "retried 1 of the 4 timed requests, whose seconds include the SDK's backoff: problems "
        "[3]; logged to MLflow"
    )


@pytest.mark.usefixtures("spy")
def test_latency_of_an_api_method_says_when_no_timed_request_was_retried(api, env, monkeypatch):
    ask = client.ask_for(None)

    async def no_retry(request):
        return dataclasses.replace(await ask(request), retries=0)

    monkeypatch.setattr(client, "ask_for", lambda settings: no_retry)

    message = run(env, name="fake-text", registry=api.registry, max_cost_usd=1.0)

    assert message.endswith("; no timed request was retried; logged to MLflow")


def test_latency_of_an_api_method_twice_sends_nothing_and_needs_no_cost_cap(
    api, env, spy, monkeypatch
):
    run(env, name="fake-text", registry=api.registry, max_cost_usd=1.0)
    path = env.assets / "fake-text" / LATENCY_FILE
    frozen = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(client, "ask_for", None)  # calling it fails the test

    message = run_latency("fake-text", registry=api.registry, **env.store)

    assert (path.read_bytes(), path.stat().st_mtime_ns) == frozen
    assert api.sent == DRAWN
    assert spy["dataset"] == [env.dataset]  # the first run's check only
    assert message.startswith(f"read {path}: 4 problems, median ")
    assert "retried 1 of the 4 timed requests" in message
    _, metrics = method_run(api.record, env.store)
    assert metrics["latency/max_s"] == 1.5


@pytest.mark.usefixtures("spy")
def test_latency_of_an_api_method_stops_sending_at_the_cost_cap_and_saves_nothing(api, env):
    first = cost_usd(usage_of(DRAWN[0]), "gpt-6-luna")["total_cost"]

    with pytest.raises(
        RuntimeError, match=r"reached the cost cap of \$1e-05 after 1 of 4 problems"
    ) as error:
        run(env, name="fake-text", registry=api.registry, max_cost_usd=1e-5)

    assert f"${first:.6f}" in str(error.value)
    assert api.sent == DRAWN[:1]  # the first request cost more than the cap
    assert not (env.assets / "fake-text" / LATENCY_FILE).exists()


@pytest.mark.usefixtures("spy")
def test_latency_of_an_api_method_adds_up_the_cost_of_the_timed_requests(api, env):
    first, second = (cost_usd(usage_of(n), "gpt-6-luna")["total_cost"] for n in DRAWN[:2])
    cap = first + second / 2  # above any one request's cost, reached by the first two

    with pytest.raises(RuntimeError, match=r"after 2 of 4 problems") as error:
        run(env, name="fake-text", registry=api.registry, max_cost_usd=cap)

    assert f"${first + second:.6f}" in str(error.value)
    assert api.sent == DRAWN[:2]
    assert not (env.assets / "fake-text" / LATENCY_FILE).exists()


@pytest.mark.usefixtures("spy")
def test_latency_of_an_api_method_needs_its_answers_files_before_any_request(api, env):
    for part in PARTS:
        answers_file("fake-text", part).unlink()

    with pytest.raises(FileNotFoundError, match="fake-text has no answers: .*`llm run`"):
        run(env, name="fake-text", registry=api.registry, max_cost_usd=1.0)

    assert api.sent == []


@pytest.mark.usefixtures("spy")
def test_latency_of_an_api_method_needs_apredict_with_answers(fakes, env):
    paid = MethodSpec(module="conformance_fakes", kind="zero_shot", requirements=(), calls_api=True)
    registry = env.registry | {"api_fake": paid}
    freeze(env, fakes.ApiFake(), registry=registry)

    with pytest.raises(TypeError, match="api_fake has no apredict_with_answers"):
        run(env, name="api_fake", registry=registry, max_cost_usd=1.0)
