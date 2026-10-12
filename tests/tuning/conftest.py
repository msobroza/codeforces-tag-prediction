"""Fixtures for the tuning tests: a tiny dev set, an Optuna journal and an MLflow store.

Every configuration of the search space can fit the dev set. The journal and the store live in
a temporary folder of their own for each test.
"""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from codeforces_tags.data.labels import multi_hot
from codeforces_tags.preprocessing.pipeline import preprocess

# The three kinds of problem of the dev set, in turn: tags, statement, solution, sample input
# and sample output. Each kind has statement words of its own (in a third of the problems) and
# every solution prints, so each block keeps terms whatever min_df and max_df a trial draws.
KINDS = (
    (
        ["graphs"],
        "Find the shortest path between two vertices of a graph.",
        "adj = [[] for _ in range(n)]\nprint(len(adj))",
        "3 2\n1 2\n2 3",
        "2",
    ),
    (
        ["strings"],
        "Find the longest palindrome in a string of letters.",
        "s = input()\nprint(s[::-1])",
        "abacaba",
        "aba",
    ),
    (
        ["greedy"],
        "Find the maximum sum of an array after one swap.",
        "a = list(map(int, input().split()))\nprint(sum(a))",
        "3\n1 2 3",
        "6",
    ),
)
TAGS = ("graphs", "strings")  # the focus tags of the kinds, in FOCUS_TAGS order
PROBLEMS = 36  # 12 of each kind
FOLDS = 3


@pytest.fixture(name="raw_dev")
def raw_dev_fixture():
    """Return 36 raw dev problems, 12 of each kind: graphs, strings, no focus tag.

    Raw rows as ``read_samples()`` gives them, with only the fields a tagging method reads
    (``RAW_INPUT_COLUMNS``: the samples as JSON strings, a missing field NaN) and the tags. Every
    other problem has notes, and every fourth one no input specification.
    """
    rows = []
    for position in range(PROBLEMS):
        tags, statement, source_code, sample_input, sample_output = KINDS[position % len(KINDS)]
        rows.append(
            {
                "prob_desc_description": statement,
                "prob_desc_input_spec": np.nan if position % 4 == 3 else "The input is one line.",
                "prob_desc_output_spec": "Print one line.",
                "prob_desc_notes": "NoteThe sample is explained." if position % 2 == 0 else np.nan,
                "prob_desc_sample_inputs": json.dumps([sample_input]),
                "prob_desc_sample_outputs": json.dumps([sample_output]),
                "source_code": source_code,
                "tags": list(tags),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(range(100, 100 + PROBLEMS), name="sample_id"))


@pytest.fixture(name="problems")
def problems_fixture(raw_dev):
    """Return the raw dev problems preprocessed, as the feature blocks read them."""
    return preprocess(raw_dev)


@pytest.fixture(name="labels")
def labels_fixture(problems):
    """Return the 0/1 labels of the two focus tags the problems carry: graphs and strings."""
    return multi_hot(problems["tags"], TAGS)


@pytest.fixture(name="folds")
def folds_fixture(problems):
    """Deal the problems to folds 0, 1, 2 so that each fold holds four problems of each kind."""
    position = pd.Series(range(PROBLEMS), index=problems.index)
    return (position // len(KINDS) % FOLDS).rename("fold")


@pytest.fixture(name="journal")
def journal_fixture(tmp_path):
    """Return the path of an Optuna journal, in a folder that does not exist yet."""
    return tmp_path / "experiments" / "optuna_journal.jsonl"


@pytest.fixture(name="mlflow_store")
def mlflow_store_fixture(store, monkeypatch):
    """Point MLflow at a fresh store, the shared ``store`` of tests/conftest.py, and return it.

    The shared fixture restores MLflow's environment and forgets the URI after the test.

    Returns:
        The ``tracking_uri`` and ``artifact_dir`` keyword arguments of ``log_study``.
    """
    # MLflow also exports the URI to the environment; monkeypatch restores it after the test.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store["tracking_uri"])
    mlflow.set_tracking_uri(store["tracking_uri"])
    return store
