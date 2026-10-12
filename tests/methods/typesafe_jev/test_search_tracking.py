"""Tests for the search in MLflow, logged live and replayed from its record, on temporary stores.

A search of three candidates: the seed (0, at iteration 0), then a rewrite of ``graphs`` (1, at
iteration 1) and of the convention (2, at iteration 3); iteration 2's proposal was refused. The
iterations are gepa's, 1-based. Every cost is a binary fraction, so cumulative sums are exact.
"""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import replace

import mlflow
import pandas as pd
import pytest
from mlflow.entities.model_registry import PromptVersion

from codeforces_tags.config import FOCUS_TAGS, MLFLOW_METHODS_EXPERIMENT
from codeforces_tags.methods.typesafe_jev import search_tracking
from codeforces_tags.methods.typesafe_jev.search_record import SearchRecord, candidate_frame
from codeforces_tags.methods.typesafe_jev.search_tracking import (
    LiveSearchLog,
    log_search,
    metric_suffix,
    prompt_name,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    SEARCH_EXPERIMENT,
    SEARCH_RUN_NAME,
    SEARCH_SETTINGS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    PIECES,
    SEED_WORDS,
    candidate_from_words,
    piece_text,
    words_from_candidate,
    words_from_json,
)
from codeforces_tags.tracking.environment import use_store

# Imports the module in a fresh interpreter, then prints whether mlflow is imported.
FRESH_IMPORT = (
    "import sys, codeforces_tags.methods.typesafe_jev.search_tracking; "
    "print('mlflow' in sys.modules)"
)


def rewritten(words, piece, addition):
    """Return words whose piece gets an addition: the convention, or a tag's yes definition."""
    candidate = candidate_from_words(words)
    if piece == "convention":
        candidate[piece] += addition
    else:
        criteria = json.loads(candidate[piece])
        criteria["true"]["definition"] += addition
        candidate[piece] = json.dumps(criteria, ensure_ascii=False, indent=2)
    return words_from_candidate(candidate)


def with_piece(words, piece, source):
    """Return words whose piece is that of other words."""
    return words_from_candidate({**candidate_from_words(words), piece: piece_text(source, piece)})


GRAPHS = rewritten(SEED_WORDS, "graphs", " A grid searched by BFS counts.")
CONVENTION = rewritten(GRAPHS, "convention", " Most problems have two or three tags.")


def candidate_line(index, parents, change, mean, words):
    """Return a line of candidates.jsonl: ``change`` is (iteration, piece changed).

    Its balanced Brier scores fall as its mean score rises.
    """
    iteration, piece = change
    return {
        "index": index,
        "parents": parents,
        "iteration": iteration,
        "piece": piece,
        "words": words.settings(),
        "words_sha256": words.sha256(),
        "selection_scores": {"7": mean},
        "selection_mean": mean,
        "selection_brier": {
            tag: round((rank + 1) / 10 - mean / 2, 6) for rank, tag in enumerate(FOCUS_TAGS)
        },
    }


def history_line(iteration, piece, sums, candidate, costs):
    """Return a line of history.jsonl: (parent, child) minibatch sums, (Jev, Luna) costs."""
    parent_sum, child_sum = sums
    return {
        "iteration": iteration,
        "parent": 0 if candidate is None else candidate - 1,
        "piece": piece,
        "minibatch": [11, 12],
        "parent_sum": parent_sum,
        "child_sum": child_sum,
        "accepted": candidate is not None,
        "refusal": "too_long" if child_sum is None else None,
        "candidate": candidate,
        "jev_cost_usd": costs[0],
        "luna_cost_usd": costs[1],
        "luna_input_tokens": 100,
        "luna_output_tokens": 50,
    }


# The seed is scored on the selection fold before the first iteration: its iteration is 0.
CANDIDATES = (
    candidate_line(0, [], (0, None), 0.5, SEED_WORDS),
    candidate_line(1, [0], (1, "graphs"), 0.6, GRAPHS),
    candidate_line(2, [1], (3, "convention"), 0.65, CONVENTION),
)
HISTORY = (
    history_line(1, "graphs", (30.0, 31.5), 1, (0.25, 0.125)),
    history_line(2, "math", (31.0, None), None, (0.0, 0.0625)),
    history_line(3, "convention", (30.5, 32.0), 2, (0.5, 0.125)),
)


def record_of(candidates, history, chosen, failed=()):
    """Return the frozen record of a search, as read_search_record gives it.

    The candidates of ``failed`` failed the dataset-text check: tuning/ keeps no words of theirs,
    and only the count of the 8-word sequences they shared with a dataset text (here 3).
    """
    copied = {
        line["index"]: ["copied words"] * 3 if line["index"] in failed else []
        for line in candidates
    }
    steps = [{key: value for key, value in line.items() if key != "minibatch"} for line in history]
    return SearchRecord(
        params={"search_settings": json.loads(json.dumps(SEARCH_SETTINGS)), "chosen": chosen},
        candidates=candidate_frame(candidates, copied),
        history=pd.DataFrame(steps).set_index("iteration"),
        words={
            line["index"]: words_from_json(line["words"])
            for line in candidates
            if line["index"] not in failed
        },
    )


RECORD = record_of(CANDIDATES, HISTORY, chosen=2)


def log_live(store, folder, candidates=CANDIDATES, history=HISTORY):
    """Log a search live as tune's recorder does: each iteration once its candidates are logged."""
    live = LiveSearchLog(**store, run_file=folder / "search" / "run.json")
    pending = list(candidates)
    for line in history:
        while pending and pending[0]["iteration"] <= line["iteration"]:
            live.candidate(pending.pop(0))
        live.iteration(line)
    live.close()
    return live.run_id


def snapshot(tracking_uri):
    """Return what a store holds of the search: its runs by name, and its prompts' versions.

    A run gives its parameters, metrics by step, tags (MLflow's own left out), parent run's
    name, linked prompt versions, artifacts and status; a version its number, template, tags and
    aliases.
    """
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(SEARCH_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    names = {run.info.run_id: run.info.run_name for run in runs}
    logged = {
        run.info.run_name: {
            "params": run.data.params,
            "metrics": {
                key: sorted(
                    (metric.step, metric.value)
                    for metric in client.get_metric_history(run.info.run_id, key)
                )
                for key in run.data.metrics
            },
            "tags": {key: value for key, value in run.data.tags.items() if "mlflow." not in key},
            "parent": names.get(run.data.tags.get("mlflow.parentRunId")),
            "prompts": json.loads(run.data.tags.get("mlflow.linkedPrompts", "[]")),
            "artifacts": [artifact.path for artifact in client.list_artifacts(run.info.run_id)],
            "status": run.info.status,
        }
        for run in runs
    }
    prompts = {}
    for piece in PIECES:
        # Only a version read on its own holds its aliases.
        name = prompt_name(piece)
        numbers = sorted(found.version for found in client.search_prompt_versions(name))
        read = [client.get_prompt_version(name, number) for number in numbers]
        prompts[name] = [(each.version, each.template, each.tags, each.aliases) for each in read]
    return logged, prompts


def content(words, piece):
    """Return the content tag of a piece's text."""
    return hashlib.sha256(piece_text(words, piece).encode("utf-8")).hexdigest()


def round_trip(frame, index):
    """Return a frame as pandas reads it back from its CSV text, every float exact."""
    text = frame.to_csv()
    return pd.read_csv(io.StringIO(text), index_col=index, float_precision="round_trip")


@pytest.fixture(name="other_store")
def other_store_fixture(tmp_path, blank_database):
    """Return a second fresh store, beside the ``store`` fixture's."""
    database = tmp_path / "other" / "mlflow.db"
    database.parent.mkdir()
    shutil.copy(blank_database, database)
    return {"tracking_uri": f"sqlite:///{database}", "artifact_dir": tmp_path / "other_artifacts"}


@pytest.mark.parametrize(
    ("piece", "name"),
    [("convention", "jev-gepa-convention"), ("number theory", "jev-gepa-number-theory")],
)
def test_prompt_name_puts_the_piece_after_the_prefix_with_dashes_for_spaces(piece, name):
    assert prompt_name(piece) == name


def test_metric_suffix_writes_a_tags_spaces_as_underscores():
    assert metric_suffix("number theory") == "number_theory"
    assert metric_suffix("graphs") == "graphs"


def test_log_search_logs_the_settings_and_each_iterations_metrics_at_its_step(store):
    run_id = log_search(RECORD, **store)

    logged, _ = snapshot(store["tracking_uri"])
    search = logged[SEARCH_RUN_NAME]
    brier = {
        f"selection_brier.{metric_suffix(tag)}": [
            (1, CANDIDATES[1]["selection_brier"][tag]),
            (2, CANDIDATES[1]["selection_brier"][tag]),
            (3, CANDIDATES[2]["selection_brier"][tag]),
        ]
        for tag in FOCUS_TAGS
    }
    assert mlflow.MlflowClient(store["tracking_uri"]).get_run(run_id).info.run_name == (
        SEARCH_RUN_NAME
    )
    assert search["metrics"] == {
        "selection_best_score": [(1, 0.6), (2, 0.6), (3, 0.65)],
        "minibatch_parent_sum": [(1, 30.0), (2, 31.0), (3, 30.5)],
        "minibatch_child_sum": [(1, 31.5), (3, 32.0)],
        "accepted": [(1, 1.0), (2, 0.0), (3, 1.0)],
        "cost_usd_jev": [(1, 0.25), (2, 0.25), (3, 0.75)],
        "cost_usd_luna": [(1, 0.125), (2, 0.1875), (3, 0.3125)],
        "cost_usd_total": [(1, 0.375), (2, 0.4375), (3, 1.0625)],
        **brier,
    }
    assert search["params"]["held_out_fold"] == "0"
    assert search["params"]["learning_folds"] == "[1, 2, 3]"
    assert search["params"]["reflection.model"] == "gpt-6-luna"
    assert search["params"]["gepa.use_merge"] == "false"
    assert search["tags"] == {"search": GEPA_NAME, "chosen": "2"}
    assert search["artifacts"] == ["candidates.csv", "history.csv"]
    assert search["status"] == "FINISHED"


def test_log_search_logs_a_run_nested_in_the_search_run_per_candidate(store):
    log_search(RECORD, **store)

    logged, _ = snapshot(store["tracking_uri"])
    graphs = logged["candidate-1"]
    assert sorted(name for name in logged if name != SEARCH_RUN_NAME) == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
    ]
    assert {logged[f"candidate-{index}"]["parent"] for index in range(3)} == {SEARCH_RUN_NAME}
    assert logged["candidate-0"]["params"] == {"parents": "", "iteration": "0", "piece": ""}
    assert graphs["params"] == {"parents": "0", "iteration": "1", "piece": "graphs"}
    assert graphs["metrics"] == {"selection_mean": [(0, 0.6)]} | {
        f"selection_brier.{metric_suffix(tag)}": [(0, CANDIDATES[1]["selection_brier"][tag])]
        for tag in FOCUS_TAGS
    }
    assert graphs["tags"] == {
        "candidate": "1",
        "words_sha256": GRAPHS.sha256(),
        "passed_check": "true",
    }
    assert graphs["status"] == "FINISHED"


def test_log_search_adds_a_version_per_new_piece_text_with_create_prompt_version(
    store, monkeypatch
):
    created, started = [], []
    create = mlflow.MlflowClient.create_prompt_version

    def spy(client, **version):
        created.append(version["name"])
        return create(client, **version)

    monkeypatch.setattr(mlflow.MlflowClient, "create_prompt_version", spy)
    monkeypatch.setattr(mlflow.MlflowClient, "register_prompt", lambda *_, **__: pytest.fail())
    monkeypatch.setattr(mlflow.genai, "register_prompt", lambda *_, **__: pytest.fail())
    monkeypatch.setattr(PromptVersion, "format", lambda *_, **__: pytest.fail())
    monkeypatch.setattr(threading.Thread, "start", lambda thread: started.append(thread.name))

    log_search(RECORD, **store)

    logged, prompts = snapshot(store["tracking_uri"])
    assert created == [prompt_name(piece) for piece in PIECES] + [
        "jev-gepa-graphs",
        "jev-gepa-convention",
    ]
    for piece in ("math", "number theory", "probabilities"):
        assert prompts[prompt_name(piece)] == [
            (
                1,
                piece_text(SEED_WORDS, piece),
                {"content_sha256": content(SEED_WORDS, piece), "candidate": "0", "iteration": "0"},
                ["chosen"],
            )
        ]
    assert prompts["jev-gepa-graphs"][1] == (
        2,
        piece_text(GRAPHS, "graphs"),
        {"content_sha256": content(GRAPHS, "graphs"), "candidate": "1", "iteration": "1"},
        ["chosen"],
    )
    assert [version[:2] for version in prompts["jev-gepa-convention"]] == [
        (1, SEED_WORDS.convention),
        (2, CONVENTION.convention),
    ]
    assert prompts["jev-gepa-convention"][1][3] == ["chosen"]
    assert logged["candidate-1"]["prompts"] == [{"name": "jev-gepa-graphs", "version": "2"}]
    assert logged["candidate-0"]["prompts"] == [
        {"name": prompt_name(piece), "version": "1"} for piece in PIECES
    ]
    assert not started
    assert mlflow.MlflowClient(store["tracking_uri"]).get_prompt("jev-gepa-graphs").tags == {}


def test_live_logging_then_log_search_gives_what_log_search_alone_gives(
    store, other_store, tmp_path
):
    log_live(store, tmp_path)
    log_search(RECORD, **store)

    log_search(RECORD, **other_store)

    assert snapshot(store["tracking_uri"]) == snapshot(other_store["tracking_uri"])


def test_the_live_log_writes_its_run_file_and_leaves_what_only_the_export_knows(store, tmp_path):
    run_id = log_live(store, tmp_path)

    logged, prompts = snapshot(store["tracking_uri"])
    experiment = mlflow.MlflowClient(store["tracking_uri"]).get_experiment_by_name(
        SEARCH_EXPERIMENT
    )
    assert json.loads((tmp_path / "search" / "run.json").read_text(encoding="utf-8")) == {
        "run_id": run_id,
        "experiment_id": experiment.experiment_id,
        "tracking_uri": store["tracking_uri"],
    }
    assert logged[SEARCH_RUN_NAME]["tags"] == {"search": GEPA_NAME}
    assert logged[SEARCH_RUN_NAME]["artifacts"] == []
    assert logged[SEARCH_RUN_NAME]["metrics"]["selection_best_score"] == [
        (1, 0.6),
        (2, 0.6),
        (3, 0.65),
    ]
    assert logged["candidate-1"]["tags"] == {"candidate": "1", "words_sha256": GRAPHS.sha256()}
    assert [len(versions) for versions in prompts.values()] == [2, 1, 2, 1, 1, 1, 1, 1, 1]
    assert not [version for versions in prompts.values() for version in versions if version[3]]


def test_an_iteration_logged_before_any_candidate_has_no_best_score(store, tmp_path):
    live = LiveSearchLog(**store, run_file=tmp_path / "run.json")

    live.iteration(HISTORY[0])

    metrics = mlflow.MlflowClient(store["tracking_uri"]).get_run(live.run_id).data.metrics
    assert sorted(metrics) == [
        "accepted",
        "cost_usd_jev",
        "cost_usd_luna",
        "cost_usd_total",
        "minibatch_child_sum",
        "minibatch_parent_sum",
    ]


def test_an_iteration_whose_parent_was_not_scored_logs_no_parent_sum(store, other_store, tmp_path):
    # GEPA ends an iteration whose proposer sampled no minibatch with neither sum.
    history = (HISTORY[0], {**HISTORY[1], "parent_sum": None}, HISTORY[2])

    log_live(store, tmp_path, history=history)
    log_search(record_of(CANDIDATES, history, chosen=2), **other_store)

    for tracking_uri in (store["tracking_uri"], other_store["tracking_uri"]):
        metrics = snapshot(tracking_uri)[0][SEARCH_RUN_NAME]["metrics"]
        assert metrics["minibatch_parent_sum"] == [(1, 30.0), (3, 30.5)]


def test_a_resumed_live_log_adds_only_its_new_iterations(store, other_store, tmp_path):
    # The rerun replays the search from its caches, from the start, then goes on.
    log_live(store, tmp_path, CANDIDATES[:2], HISTORY[:2])
    log_live(store, tmp_path)

    log_live(other_store, tmp_path / "other")

    assert snapshot(store["tracking_uri"]) == snapshot(other_store["tracking_uri"])


def test_a_second_log_search_adds_nothing(store, tmp_path):
    first = log_search(RECORD, **store, run_file=tmp_path / "run.json")
    before = snapshot(store["tracking_uri"])

    second = log_search(RECORD, **store)

    assert second == first
    assert snapshot(store["tracking_uri"]) == before
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["run_id"] == first


def test_log_search_reads_a_record_as_pandas_reads_it_back_from_csv(store, other_store):
    # Read back, the seed's empty parents and piece and the refused child's sum are NaN, and
    # the parents a column of floats.
    read = replace(
        RECORD,
        candidates=round_trip(RECORD.candidates, "index"),
        history=round_trip(RECORD.history, "iteration"),
    )

    log_search(RECORD, **store)
    log_search(read, **other_store)

    assert read.candidates["parents"].isna().sum() == 1
    assert snapshot(store["tracking_uri"]) == snapshot(other_store["tracking_uri"])


def test_a_replayed_store_holds_no_version_of_words_that_failed_the_dataset_text_check(store):
    # Candidate 1 copied a problem's words into graphs at iteration 1; candidate 2, its child,
    # rewrote math and kept them; candidate 3 rewrote graphs again and passed. tuning/ keeps
    # neither 1's nor 2's words, so 3 adds the versions of its graphs and of the math that 2 wrote.
    copied = rewritten(SEED_WORDS, "graphs", " Count the leaves of the tree in the first sample.")
    inherited = rewritten(copied, "math", " Sums of digits count.")
    cleaned = with_piece(inherited, "graphs", GRAPHS)
    candidates = [
        CANDIDATES[0],
        candidate_line(1, [0], (1, "graphs"), 0.62, copied),
        candidate_line(2, [1], (2, "math"), 0.63, inherited),
        candidate_line(3, [2], (3, "graphs"), 0.64, cleaned),
    ]
    history = [
        history_line(iteration, piece, (30.0, 31.0), iteration, (0.25, 0.125))
        for iteration, piece in enumerate(("graphs", "math", "graphs"), start=1)
    ]

    log_search(record_of(candidates, history, chosen=3, failed=(1, 2)), **store)

    logged, prompts = snapshot(store["tracking_uri"])
    assert [version[:3] for version in prompts["jev-gepa-graphs"]] == [
        (
            1,
            piece_text(SEED_WORDS, "graphs"),
            {"content_sha256": content(SEED_WORDS, "graphs"), "candidate": "0", "iteration": "0"},
        ),
        (
            2,
            piece_text(GRAPHS, "graphs"),
            {"content_sha256": content(GRAPHS, "graphs"), "candidate": "3", "iteration": "3"},
        ),
    ]
    assert prompts["jev-gepa-math"][1][:3] == (
        2,
        piece_text(inherited, "math"),
        {"content_sha256": content(inherited, "math"), "candidate": "3", "iteration": "3"},
    )
    assert [prompts[name][-1][3] for name in ("jev-gepa-graphs", "jev-gepa-math")] == [
        ["chosen"],
        ["chosen"],
    ]
    assert logged["candidate-1"]["tags"]["passed_check"] == "false"
    assert logged["candidate-1"]["prompts"] == logged["candidate-2"]["prompts"] == []
    assert logged["candidate-3"]["prompts"] == [
        {"name": "jev-gepa-math", "version": "2"},
        {"name": "jev-gepa-graphs", "version": "2"},
    ]


def test_log_search_refuses_a_store_that_holds_another_search(store):
    log_search(RECORD, **store)
    candidates = RECORD.candidates.copy()
    candidates.loc[1, "words_sha256"] = "f" * 64

    with pytest.raises(ValueError, match=r"candidate 1 of the search run \w+ has the words"):
        log_search(replace(RECORD, candidates=candidates), **store)


def test_a_candidate_with_the_words_of_another_links_their_version_and_adds_none(store):
    # At iteration 2 the reflection rewrote the seed's graphs into candidate 1's text again:
    # candidate 2 holds the same words as candidate 1, and the selection rule keeps the lower index.
    candidates = [*CANDIDATES[:2], candidate_line(2, [0], (2, "graphs"), 0.6, GRAPHS)]
    second = {**history_line(2, "graphs", (30.5, 31.5), 2, (0.25, 0.125)), "parent": 0}
    record = record_of(candidates, [HISTORY[0], second], chosen=1)

    log_search(record, **store)
    before = snapshot(store["tracking_uri"])
    log_search(record, **store)

    logged, prompts = before
    assert snapshot(store["tracking_uri"]) == before
    assert [version[:3] for version in prompts["jev-gepa-graphs"]] == [
        (
            1,
            piece_text(SEED_WORDS, "graphs"),
            {"content_sha256": content(SEED_WORDS, "graphs"), "candidate": "0", "iteration": "0"},
        ),
        (
            2,
            piece_text(GRAPHS, "graphs"),
            {"content_sha256": content(GRAPHS, "graphs"), "candidate": "1", "iteration": "1"},
        ),
    ]
    assert prompts["jev-gepa-graphs"][1][3] == ["chosen"]
    assert sum(len(versions) for versions in prompts.values()) == len(PIECES) + 1
    assert logged["candidate-2"]["tags"]["words_sha256"] == GRAPHS.sha256()
    assert logged["candidate-2"]["prompts"] == [{"name": "jev-gepa-graphs", "version": "2"}]
    assert logged["candidate-1"]["prompts"] == logged["candidate-2"]["prompts"]


def test_log_search_refuses_a_prompt_holding_a_version_with_no_content_tag(store):
    # A version made by hand: no search log wrote it, so no text of it can be matched.
    use_store(store["tracking_uri"], SEARCH_EXPERIMENT, store["artifact_dir"])
    client = mlflow.MlflowClient()
    client.create_prompt("jev-gepa-graphs")
    client.create_prompt_version(name="jev-gepa-graphs", template="Graphs, written by hand.")

    with pytest.raises(
        ValueError,
        match=r"The prompt jev-gepa-graphs of sqlite:///\S+ holds the versions \[1\] with no "
        r"content_sha256 tag: .* Rebuild the store",
    ):
        log_search(RECORD, **store)


def test_log_search_reads_every_page_of_a_prompts_versions(store, other_store, monkeypatch):
    # Three versions of graphs, read two at a time: a reader of the first page only would miss
    # the seed's version 1, and add it again as version 4.
    again = rewritten(GRAPHS, "graphs", " A tree is a graph too.")
    candidates = [*CANDIDATES[:2], candidate_line(2, [1], (2, "graphs"), 0.62, again)]
    history = [HISTORY[0], history_line(2, "graphs", (31.0, 32.0), 2, (0.25, 0.125))]
    record = record_of(candidates, history, chosen=2)
    log_search(record, **other_store)
    monkeypatch.setattr(search_tracking, "VERSIONS_PAGE_SIZE", 2)

    log_search(record, **store)
    log_search(record, **store)

    logged, prompts = snapshot(store["tracking_uri"])
    assert (logged, prompts) == snapshot(other_store["tracking_uri"])
    assert [version[0] for version in prompts["jev-gepa-graphs"]] == [1, 2, 3]


def test_log_search_refuses_a_search_run_with_other_settings(store):
    log_search(RECORD, **store)
    settings = {**RECORD.params["search_settings"], "max_iterations": 100}

    with pytest.raises(ValueError, match=r"other search settings \(\['max_iterations'\]\)"):
        log_search(replace(RECORD, params={**RECORD.params, "search_settings": settings}), **store)


def test_log_search_refuses_an_experiment_with_two_search_runs(store):
    experiment_id = use_store(store["tracking_uri"], SEARCH_EXPERIMENT, store["artifact_dir"])
    client = mlflow.MlflowClient()
    for _ in range(2):
        client.create_run(experiment_id, tags={"search": GEPA_NAME})

    with pytest.raises(ValueError, match=f"holds 2 search runs of {GEPA_NAME}, not one"):
        log_search(RECORD, **store)


def test_the_search_logs_into_its_own_experiment_and_nothing_into_tag_methods(store, tmp_path):
    log_live(store, tmp_path)
    log_search(RECORD, **store)

    client = mlflow.MlflowClient(store["tracking_uri"])
    assert client.get_experiment_by_name(MLFLOW_METHODS_EXPERIMENT) is None
    assert sorted(experiment.name for experiment in client.search_experiments()) == [
        "Default",
        SEARCH_EXPERIMENT,
    ]


def test_importing_the_module_imports_no_mlflow():
    # The package sets none of MLflow's environment: mlflow comes only once use_store set it.
    result = subprocess.run(
        [sys.executable, "-c", FRESH_IMPORT], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
