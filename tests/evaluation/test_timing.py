"""Tests for the latency harness's timers, on a clock that moves only when a method works."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.evaluation import timing
from codeforces_tags.evaluation.timing import (
    LATENCY_BULK,
    LATENCY_FILE,
    LATENCY_LOAD,
    Latency,
    latency_summary,
    load_latency,
    save_latency,
    time_bulk,
    time_load,
    time_single,
)
from codeforces_tags.methods import MethodSpec, save_method

FAKES = Path(__file__).parents[1] / "methods" / "fakes"


class Clock:  # pylint: disable=too-few-public-methods
    """A stand-in for ``perf_counter``: its time moves only when a fake method works."""

    def __init__(self):
        """Start at 100 s."""
        self.now = 100.0

    def __call__(self):
        """Return the current time."""
        return self.now


class SlowMethod:  # pylint: disable=too-few-public-methods
    """Takes 0.25 s per problem, plus 1 s on its first call (a cold start)."""

    name = "slow"
    tags = ("math",)

    def __init__(self, clock):
        """Move ``clock`` when predicting."""
        self.clock = clock
        self.calls = []

    def predict(self, timed_problems):
        """Record the problems, move the clock, and predict nothing."""
        cold_start = 0.0 if self.calls else 1.0
        self.calls.append(list(timed_problems.index))
        self.clock.now += cold_start + 0.25 * len(timed_problems)
        return pd.DataFrame(0, index=timed_problems.index, columns=list(self.tags))


@pytest.fixture(name="clock")
def clock_fixture(monkeypatch):
    """Return the clock that the timers read while the test runs."""
    clock = Clock()
    monkeypatch.setattr(timing, "perf_counter", clock)
    return clock


@pytest.fixture(name="timed_problems")
def timed_problems_fixture():
    """Return three raw problems, indexed by sample id out of order."""
    return pd.DataFrame(
        {"source_code": ["print(1)", "print(2)", "print(3)"]},
        index=pd.Index([7, 3, 9], name="sample_id"),
    )


def test_time_single_times_one_problem_at_a_time_after_a_warm_up(clock, timed_problems):
    method = SlowMethod(clock)

    seconds = time_single(method, timed_problems)

    assert method.calls == [[7], [7], [3], [9]]
    assert seconds.to_dict() == {7: 0.25, 3: 0.25, 9: 0.25}
    assert (seconds.name, seconds.index.name) == ("seconds", "sample_id")


def test_time_single_without_a_warm_up_times_the_cold_start(clock, timed_problems):
    seconds = time_single(SlowMethod(clock), timed_problems, warmup=0)

    assert seconds.tolist() == [1.25, 0.25, 0.25]


def test_time_single_warms_up_on_the_problems_in_turn(clock, timed_problems):
    method = SlowMethod(clock)

    time_single(method, timed_problems, warmup=4)

    assert method.calls[:4] == [[7], [3], [9], [7]]


def test_time_single_refuses_no_problem_or_a_negative_warm_up(clock, timed_problems):
    with pytest.raises(ValueError, match="no problem to time"):
        time_single(SlowMethod(clock), timed_problems.iloc[:0])
    with pytest.raises(ValueError, match="warmup must be 0 or more"):
        time_single(SlowMethod(clock), timed_problems, warmup=-1)


def test_time_bulk_times_one_call_over_every_problem(clock, timed_problems):
    method = SlowMethod(clock)

    assert time_bulk(method, timed_problems) == 1.75
    assert method.calls == [[7, 3, 9]]


def test_time_load_returns_the_loaded_method_and_its_load_time(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(FAKES))
    fakes = importlib.import_module("fake_methods")
    save_method(fakes.ConstantMethod("constant", ("math",), ("math",)), tmp_path)
    registry = {"constant": MethodSpec(module="fake_methods", kind="fitted", requirements=())}
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(timing, "perf_counter", lambda: next(ticks))

    method, seconds = time_load(tmp_path, registry=registry)

    assert (method.name, method.tags, seconds) == ("constant", ("math",), 2.5)


def test_latency_summary_gives_the_median_p95_and_max():
    summary = latency_summary(pd.Series([float(second) for second in range(1, 21)]))

    assert list(summary.index) == ["median_s", "p95_s", "max_s"]
    assert summary.tolist() == pytest.approx([10.5, 19.05, 20.0])


def test_latency_summary_refuses_no_time():
    with pytest.raises(ValueError, match="no time to summarise"):
        latency_summary(pd.Series([], dtype=float))


def make_latency():
    """Return the latency of two problems, listed out of sample id order."""
    seconds = pd.Series(
        [0.25, 0.125], index=pd.Index([7, 3], name="sample_id"), name="seconds", dtype=float
    )
    return Latency(seconds, bulk_s=0.5, load_s=1.5)


def test_latency_csv_has_a_row_per_problem_then_the_bulk_and_the_load_rows(tmp_path):
    save_latency(tmp_path, make_latency())

    assert (tmp_path / LATENCY_FILE).read_text(encoding="utf-8").splitlines() == [
        "sample_id,seconds",
        "7,0.25",
        "3,0.125",
        f"{LATENCY_BULK},0.5",
        f"{LATENCY_LOAD},1.5",
    ]


def test_load_latency_reads_back_every_value_exactly(tmp_path):
    seconds = pd.Series(
        [1 / 3, 2 / 3, 1e-4],
        index=pd.Index([30, 10, 20], name="sample_id"),
        name="seconds",
    )
    saved = Latency(seconds, bulk_s=0.1 + 0.2, load_s=1 / 7)
    save_latency(tmp_path, saved)

    loaded = load_latency(tmp_path)

    pd.testing.assert_series_equal(loaded.seconds, seconds)
    assert (loaded.bulk_s, loaded.load_s) == (0.1 + 0.2, 1 / 7)
    assert loaded.calls is None


def test_save_latency_is_written_once(tmp_path):
    save_latency(tmp_path, make_latency())
    saved = (tmp_path / LATENCY_FILE).read_bytes()

    with pytest.raises(FileExistsError, match="written once"):
        save_latency(tmp_path, Latency(make_latency().seconds * 2, bulk_s=0.0, load_s=0.0))

    assert (tmp_path / LATENCY_FILE).read_bytes() == saved


def test_load_latency_refuses_a_file_without_its_bulk_or_load_row(tmp_path):
    (tmp_path / LATENCY_FILE).write_text("sample_id,seconds\n7,0.25\nbulk,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"has no row for \['load'\]"):
        load_latency(tmp_path)


def make_calls(latency):
    """Return the tokens, cost and retries of two timed requests, in the order of ``latency``."""
    return pd.DataFrame(
        {
            "input_tokens": [1_200, 80],
            "output_tokens": [6, 4],
            "cost_usd": [1 / 3 * 1e-4, 1.1e-5],
            "retries": [0, 2],
        },
        index=latency.seconds.index,
    )


def test_latency_csv_of_an_api_method_adds_the_tokens_cost_and_retries_of_each_request(tmp_path):
    latency = make_latency()
    calls = make_calls(latency)

    save_latency(tmp_path, latency._replace(calls=calls))

    assert (tmp_path / LATENCY_FILE).read_text(encoding="utf-8").splitlines() == [
        "sample_id,seconds,input_tokens,output_tokens,cost_usd,retries",
        f"7,0.25,1200,6,{1 / 3 * 1e-4!r},0",
        "3,0.125,80,4,1.1e-05,2",
        f"{LATENCY_BULK},0.5,,,,",
        f"{LATENCY_LOAD},1.5,,,,",
    ]
    loaded = load_latency(tmp_path)
    pd.testing.assert_frame_equal(loaded.calls, calls)
    pd.testing.assert_series_equal(loaded.seconds, latency.seconds)


def test_save_latency_refuses_calls_that_are_not_one_row_per_timed_problem(tmp_path):
    latency = make_latency()
    calls = make_calls(latency)

    for wrong in (calls.iloc[::-1], calls.iloc[:1], calls.drop(columns="retries")):
        with pytest.raises(ValueError, match="one row per timed problem"):
            save_latency(tmp_path, latency._replace(calls=wrong))

    assert not (tmp_path / LATENCY_FILE).exists()


def test_load_latency_refuses_columns_that_are_not_the_calls_of_an_api_method(tmp_path):
    (tmp_path / LATENCY_FILE).write_text(
        "sample_id,seconds,retries\n7,0.25,0\nbulk,0.5,\nload,1.5,\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"has the columns \['retries'\] beside seconds"):
        load_latency(tmp_path)
