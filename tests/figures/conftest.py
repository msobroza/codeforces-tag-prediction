"""Fixtures for the tests of the figure render tools: the MLflow UI's and the Streamlit app's.

``notebooks/figures/mlflow_ui.py`` and ``notebooks/figures/streamlit_app.py`` are render tools,
not modules of the package: ``mlflow_ui`` and ``streamlit_capture`` load them by path, once per
session. Every test here runs with MLflow's environment cleared and restored
(``mlflow_environment``, in ``tests/conftest.py``): ``server_env`` sets it, and ``screen_ids``
points MLflow at a store. ``screen_store`` is a store as the MLflow UI capture finds it, built
once per session and copied for each test; ``gepa_store`` builds a store as the capture of the
search's views finds it, a small search replayed by ``log_search``.
"""

import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from types import SimpleNamespace

import mlflow
import pytest
from mlflow.genai.scorers import scorer

from codeforces_tags import methods
from codeforces_tags.config import (
    FOCUS_TAGS,
    MLFLOW_DATASETS,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    MLFLOW_SCREEN_GEPA_TAG,
    PROJECT_ROOT,
)
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.methods.typesafe_jev.search_record import (
    SearchRecord,
    candidate_frame,
    history_frame,
)
from codeforces_tags.methods.typesafe_jev.search_tracking import log_search
from codeforces_tags.methods.typesafe_jev.settings import SEARCH_EXPERIMENT, SEARCH_SETTINGS
from codeforces_tags.methods.typesafe_jev.words import (
    SEED_WORDS,
    candidate_from_words,
    words_from_candidate,
)
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.evaluation import COMPLETE_TAG, EVALUATION_RUN_KIND
from codeforces_tags.tracking.methods import RUN_KIND_TAG

SCRIPT = PROJECT_ROOT / "notebooks" / "figures" / "mlflow_ui.py"
STREAMLIT_CAPTURE = PROJECT_ROOT / "notebooks" / "figures" / "streamlit_app.py"
# The problems of each part of the screen store, and their tags.
SAMPLES = {"dev": (0, 1, 2), "test": (3, 4)}
TAGS = {0: ["math"], 1: ["graphs", "math"], 2: [], 3: ["strings"], 4: ["trees"]}
# The methods of the screen store: the baseline, and the method the views compare with it.
SHOWN = ("tfidf", "descode2tag-json")
# When each evaluation run of the screen store starts and ends, in milliseconds since the epoch:
# the dev runs from 2026-10-09 09:40:05 to 09:52:10 UTC, then, after a pause, the test runs.
TIMES = {
    ("tfidf", "dev"): (1_791_538_805_000, 1_791_539_130_000),
    ("descode2tag-json", "dev"): (1_791_539_160_000, 1_791_539_530_000),
    ("tfidf", "test"): (1_791_539_700_000, 1_791_539_760_000),
    ("descode2tag-json", "test"): (1_791_539_770_000, 1_791_539_830_000),
}
# The dev window of those times, and the prompt the compared method's run record pins.
WINDOW = ("2026-10-09T09:40:00.000Z", "2026-10-09T09:53:00.000Z")
PINNED_PROMPT = "prompts:/descode2tag/2"
# The search of the gepa store: the seed (0, iteration 0), a rewrite of the criteria of the tag
# the views compare (1, iteration 1), then of the convention (2, iteration 2); gepa numbers its
# iterations from 1. Each candidate's parents, iteration and changed piece.
GEPA_CHANGES = {
    0: ((), 0, None),
    1: ((0,), 1, MLFLOW_SCREEN_GEPA_TAG),
    2: ((1,), 2, "convention"),
}


def gepa_words():
    """Return the words of the gepa store's candidates, by index."""
    pieces = candidate_from_words(SEED_WORDS)
    criteria = json.loads(pieces[MLFLOW_SCREEN_GEPA_TAG])
    criteria["true"]["includes"].append("a problem the search added to these criteria")
    rewritten = {
        **pieces,
        MLFLOW_SCREEN_GEPA_TAG: json.dumps(criteria, ensure_ascii=False, indent=2),
    }
    convention = {**rewritten, "convention": f"{pieces['convention']} Tags are few."}
    return {
        0: SEED_WORDS,
        1: words_from_candidate(rewritten),
        2: words_from_candidate(convention),
    }


def gepa_record(chosen):
    """Return the frozen record of the gepa store's search, which chose candidate ``chosen``.

    Its tables are built from the lines the search's recorder writes, as ``export`` builds them;
    every candidate passed the dataset-text check, and every iteration's child was accepted.
    """
    words = gepa_words()
    candidates = [
        {
            "index": index,
            "parents": list(parents),
            "iteration": iteration,
            "piece": piece,
            "selection_mean": 0.5 + index / 10,
            "selection_brier": dict.fromkeys(FOCUS_TAGS, 0.25),
            "words_sha256": words[index].sha256(),
        }
        for index, (parents, iteration, piece) in GEPA_CHANGES.items()
    ]
    history = [
        {
            "iteration": iteration,
            "parent": parents[0],
            "piece": piece,
            "minibatch": [5, 6],
            "parent_sum": 30.0,
            "child_sum": 31.0,
            "accepted": True,
            "refusal": None,
            "candidate": index,
            "jev_cost_usd": 0.5,
            "luna_cost_usd": 0.25,
            "luna_input_tokens": 300,
            "luna_output_tokens": 70,
        }
        for index, (parents, iteration, piece) in GEPA_CHANGES.items()
        if iteration
    ]
    return SearchRecord(
        params={"chosen": chosen, "search_settings": json.loads(json.dumps(SEARCH_SETTINGS))},
        candidates=candidate_frame(candidates, dict.fromkeys(GEPA_CHANGES, [])),
        history=history_frame(history).set_index("iteration"),
        words=words,
    )


@scorer
def same_tags(outputs, expectations):
    """Tell whether a prediction holds the expected tags."""
    return outputs["tags"] == expectations["tags"]


@pytest.fixture(autouse=True)
def isolate_mlflow(mlflow_environment):  # pylint: disable=unused-argument
    """Isolate MLflow's environment and default store in every test of the render tools."""


@pytest.fixture(name="mlflow_ui", scope="session")
def mlflow_ui_fixture():
    """Return the capture script, loaded by path as the module ``mlflow_ui``.

    The module is registered in ``sys.modules`` while the session lasts: its dataclasses look
    their module up there.
    """
    spec = importlib.util.spec_from_file_location("mlflow_ui", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.fixture(name="streamlit_capture", scope="session")
def streamlit_capture_fixture():
    """Return the Streamlit app's capture, loaded by path as the module ``streamlit_capture``.

    It is registered in ``sys.modules`` while the session lasts, as ``mlflow_ui`` is, under a
    name of its own: the app, ``apps/streamlit_app.py``, is never imported.
    """
    spec = importlib.util.spec_from_file_location("streamlit_capture", STREAMLIT_CAPTURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def evaluate(experiment_id, method, part):
    """Evaluate a method's lookup of the true tags on a part, tagged as tracking evaluate tags it.

    The run starts and ends at the method's ``TIMES``, and is tagged complete once it ended.
    """
    dataset = mlflow.genai.datasets.search_datasets(
        experiment_ids=[experiment_id], filter_string=f"name = '{MLFLOW_DATASETS[part]}'"
    )[0]
    start, end = TIMES[method, part]
    tags = {RUN_KIND_TAG: EVALUATION_RUN_KIND, "method": method, "part": part}
    client = mlflow.MlflowClient()
    run = client.create_run(
        experiment_id, start_time=start, tags=tags, run_name=f"eval-{method}-{part}"
    )

    @mlflow.trace(name=method)
    def predict(sample_id, prob_desc_description):  # pylint: disable=unused-argument
        return {"tags": TAGS[sample_id]}

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.genai.evaluate(data=dataset, predict_fn=predict, scorers=[same_tags])
    client.set_terminated(run.info.run_id, end_time=end)
    client.set_tag(run.info.run_id, COMPLETE_TAG, "true")
    return run.info.run_id


def build_screen_store(store):
    """Fill a store as the capture finds it; return its experiment, and its runs by method, part."""
    experiment_id = use_store(**store)
    for part, samples in SAMPLES.items():
        records = [
            {
                "inputs": {"sample_id": sample, "prob_desc_description": f"Problem {sample}."},
                "expectations": {"tags": TAGS[sample]},
            }
            for sample in samples
        ]
        dataset = mlflow.genai.datasets.create_dataset(
            name=MLFLOW_DATASETS[part], experiment_id=experiment_id
        )
        dataset.merge_records(records)
    runs = {
        (method, part): evaluate(experiment_id, method, part)
        for part in SAMPLES
        for method in SHOWN
    }
    return experiment_id, runs


@pytest.fixture(name="built_screen_store", scope="session")
def built_screen_store_fixture(tmp_path_factory, blank_database):
    """Build the screen store once: its database, its artifact folder, and its ids.

    MLflow's environment is cleared while it is built, then restored, as ``mlflow_environment``
    does for a test.
    """
    folder = tmp_path_factory.mktemp("screen_store")
    database = folder / "mlflow.db"
    database.write_bytes(blank_database.read_bytes())
    store = {"tracking_uri": f"sqlite:///{database}", "artifact_dir": folder / "artifacts"}
    with pytest.MonkeyPatch.context() as patch:
        for name in (*MLFLOW_ENV, *MLFLOW_ENV_DEFAULTS):
            patch.setenv(name, "")
            patch.delenv(name)
        experiment_id, runs = build_screen_store(store)
        mlflow.set_tracking_uri(None)
    return SimpleNamespace(
        database=database,
        artifact_dir=store["artifact_dir"],
        experiment_id=experiment_id,
        runs=runs,
    )


def write_run_record(assets, record):
    """Write a method's run record under a methods folder."""
    (assets / record.method).mkdir(parents=True)
    text = json.dumps(record.to_dict())
    (assets / record.method / RUN_RECORD_FILE).write_text(text, encoding="utf-8")


@pytest.fixture(name="screen_store")
def screen_store_fixture(built_screen_store, placeholder_record, monkeypatch, tmp_path):
    """Return a copy of the screen store for one test, with the run records the capture reads.

    The namespace holds ``tracking_uri``, ``artifact_dir``, ``experiment_id`` and the run ids,
    ``runs[method, part]``. The run records are written under a ``METHODS_ASSETS_DIR`` of the
    test's own: that of ``descode2tag-json`` pins its prompt, and that of ``tfidf`` none.
    """
    database = tmp_path / "store" / "mlflow.db"
    database.parent.mkdir()
    with (
        closing(sqlite3.connect(built_screen_store.database)) as source,
        closing(sqlite3.connect(database)) as copy,
    ):
        source.backup(copy)
    assets = tmp_path / "assets"
    write_run_record(assets, placeholder_record("descode2tag-json", prompts=[PINNED_PROMPT]))
    write_run_record(assets, placeholder_record("tfidf"))
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", assets)
    return SimpleNamespace(
        tracking_uri=f"sqlite:///{database}",
        artifact_dir=built_screen_store.artifact_dir,
        experiment_id=built_screen_store.experiment_id,
        runs=built_screen_store.runs,
    )


@pytest.fixture(name="gepa_store")
def gepa_store_fixture(store):
    """Return a builder of the store the search's views read: a search replayed by log_search.

    ``build(chosen)`` replays the search of ``GEPA_CHANGES`` that chose candidate ``chosen``
    (2: the criteria of ``MLFLOW_SCREEN_GEPA_TAG`` and the convention changed; 1: only those
    criteria) into the test's fresh store, and returns ``tracking_uri``, ``artifact_dir``,
    ``experiment_id`` and the search's ``run_id``.
    """

    def build(chosen):
        run_id = log_search(gepa_record(chosen), **store)
        client = mlflow.MlflowClient(store["tracking_uri"])
        return SimpleNamespace(
            tracking_uri=store["tracking_uri"],
            artifact_dir=store["artifact_dir"],
            experiment_id=client.get_experiment_by_name(SEARCH_EXPERIMENT).experiment_id,
            run_id=run_id,
        )

    return build
