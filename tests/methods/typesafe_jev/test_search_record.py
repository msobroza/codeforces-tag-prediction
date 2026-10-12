"""Tests for the frozen record of a search and the helpers that export and freeze read.

They read back what export wrote, choose the candidate, read a candidate's parents, list the
dataset texts and compute the run record's search entries.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.export import export_search
from codeforces_tags.methods.typesafe_jev.search_record import (
    BRIER_COLUMNS,
    CANDIDATE_COLUMNS,
    HISTORY_COLUMNS,
    choose_candidate,
    dataset_texts,
    parents_of,
    read_search_record,
    search_entries,
    words_text,
)
from codeforces_tags.methods.typesafe_jev.settings import PARAMS_FILE


@pytest.fixture(name="exported")
def exported_fixture(search_run, tmp_path):
    """Write the finished search of ``search_run``, export it, and return its tuning folder."""
    search_run.write_finished()
    out = tmp_path / "tuning"
    return export_search(
        search_run.folder, search_run.data, search_run.seed_scores, search_run.texts, out
    )


def test_read_search_record_reads_back_what_export_wrote(exported, search_run):
    record = read_search_record(exported)

    assert record.params == json.loads((exported / PARAMS_FILE).read_text(encoding="utf-8"))
    assert {index: words.sha256() for index, words in record.words.items()} == {
        index: search_run.words[index].sha256() for index in (0, 1, 3)
    }
    candidates = record.candidates
    assert candidates.index.tolist() == [0, 1, 2, 3]
    assert candidates.columns.tolist() == list(CANDIDATE_COLUMNS[1:])
    assert candidates["parents"].tolist() == ["", "0", "0", "1"]
    assert candidates["iteration"].tolist() == [0, 1, 2, 4]
    assert candidates["piece"].tolist() == ["", "convention", "convention", "convention"]
    assert candidates["passed_check"].tolist() == [True, True, False, True]
    assert candidates["copied_sequences"].tolist() == [0, 0, 3, 0]
    assert candidates.loc[1, "selection_mean"] == 0.95
    assert candidates.loc[2, list(BRIER_COLUMNS)].tolist() == [0.1] * len(FOCUS_TAGS)
    history = record.history
    assert history.index.tolist() == [1, 2, 3, 4]
    assert history.columns.tolist() == list(HISTORY_COLUMNS[1:])
    assert history["parent"].tolist() == [0, 0, 2, 1]
    assert history["refusal"].tolist() == ["", "", "too_long", ""]
    assert history["candidate"].isna().tolist() == [False, False, True, False]
    assert history["candidate"].dropna().tolist() == [1, 2, 3]
    assert history["child_sum"].isna().tolist() == [False, False, True, False]


def test_the_brier_columns_name_each_focus_tag_by_its_question_id():
    assert BRIER_COLUMNS[FOCUS_TAGS.index("number theory")] == "brier_number_theory"
    assert len(BRIER_COLUMNS) == len(FOCUS_TAGS)


def swap_the_words_of_candidate_1(out, words):
    """Write candidate 3's words under candidate 1's name."""
    (out / "candidates" / "1.json").write_text(words_text(words[3]), encoding="utf-8")


def add_the_words_of_candidate_2(out, words):
    """Write the words of candidate 2, which copies dataset text, into the record."""
    (out / "candidates" / "2.json").write_text(words_text(words[2]), encoding="utf-8")


def swap_the_chosen_words(out, words):
    """Put candidate 3's words in params.json, under candidate 1's index and hash."""
    path = out / PARAMS_FILE
    params = json.loads(path.read_text(encoding="utf-8"))
    params["words"] = words[3].settings()
    path.write_text(json.dumps(params), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (swap_the_words_of_candidate_1, r"\['1.json'\] do not"),
        (add_the_words_of_candidate_2, r"\['2.json'\] do not"),
        (swap_the_chosen_words, "must hold the words of its chosen candidate 1"),
    ],
)
def test_read_search_record_refuses_words_that_are_not_those_of_the_candidates(
    exported, search_run, change, message
):
    change(exported, search_run.words)

    with pytest.raises(ValueError, match=message):
        read_search_record(exported)


def test_choose_candidate_takes_the_best_passing_mean_and_the_lowest_index_on_a_tie():
    candidates = [
        {"index": index, "selection_mean": mean}
        for index, mean in enumerate([0.90, 0.95, 0.99, 0.95])
    ]
    passed = {0: [], 1: [], 2: [], 3: []}
    copying = passed | {2: ["eight words copied from a problem of the dataset"]}

    assert choose_candidate(candidates, passed) == 2
    assert choose_candidate(candidates, copying) == 1
    assert choose_candidate(candidates[::-1], copying) == 1  # whatever the order of the lines


def test_choose_candidate_refuses_candidates_of_which_none_passed_the_check():
    candidates = [{"index": 0, "selection_mean": 0.9}, {"index": 1, "selection_mean": 0.95}]
    copied = dict.fromkeys([0, 1], ["eight words copied from a problem of the dataset"])

    with pytest.raises(ValueError, match="No candidate passed the dataset-text check"):
        choose_candidate(candidates, copied)


@pytest.mark.parametrize(
    ("value", "parents"),
    [
        ("1 0", (1, 0)),
        ("2", (2,)),
        ("", ()),
        (1.0, (1,)),
        (3, (3,)),
        ([1, 0], (1, 0)),
        ([], ()),
        (float("nan"), ()),
        (None, ()),
    ],
    ids=["text", "one index", "seed text", "float", "int", "list", "seed list", "NaN", "None"],
)
def test_parents_of_reads_the_parents_as_every_reader_of_the_record_gives_them(value, parents):
    assert parents_of(value) == parents


@pytest.mark.parametrize("value", [1.5, "1 x", [0.5]], ids=["float", "text", "list"])
def test_parents_of_refuses_a_parent_that_is_not_a_whole_number(value):
    with pytest.raises(ValueError):
        parents_of(value)


def test_dataset_texts_are_the_statement_notes_and_solution_of_every_problem():
    problems = pd.DataFrame(
        {
            "statement": ["Sum the array.", "Sort the array."],
            "prob_desc_notes": ["Look at the first test.", None],
            "source_code": ["print(sum(a))", "print(sorted(a))"],
            "tags": [["math"], ["sortings"]],
        },
        index=pd.Index([4, 9], name="sample_id"),
    )

    assert dataset_texts(problems) == [
        "Sum the array.",
        "Sort the array.",
        "Look at the first test.",
        "print(sum(a))",
        "print(sorted(a))",
    ]


def test_search_entries_score_the_held_out_fold_only():
    index = pd.Index([1, 2, 3, 4], name="sample_id")
    columns = list(FOCUS_TAGS)

    def frame(rows):
        return pd.DataFrame(
            [[int(tag in row) for tag in FOCUS_TAGS] for row in rows], index=index, columns=columns
        )

    labels = frame([{"math", "graphs"}, {"strings"}, {"math"}, {"trees"}])
    # Problem 3 (fold 1) misses math, which would lower math's F1 if fold 1 counted.
    predictions = frame([{"math"}, {"strings", "trees"}, set(), {"trees"}])
    folds = pd.Series([0, 0, 1, 4], index=index)
    params = {"chosen": 5, "cost_usd": {"jev": 8.0, "luna": 1.0, "total": 9.0}}

    entries = search_entries(params, "ab" * 32, predictions, labels, folds)

    assert entries == {
        "held_out_fold": 0,
        "selection_fold": 4,
        "learning_folds": [1, 2, 3],
        "chosen": 5,
        "params_sha256": "ab" * 32,
        "cost_usd": {"jev": 8.0, "luna": 1.0, "total": 9.0},
        # math and strings are found; graphs is missed and trees, absent, predicted once.
        "held_out_macro_f1": 0.25,
        "held_out_f1": dict.fromkeys(FOCUS_TAGS, 0.0) | {"math": 1.0, "strings": 1.0},
    }


def test_the_record_module_imports_neither_gepa_nor_mlflow():
    code = (
        "import sys\n"
        "import codeforces_tags.methods.typesafe_jev.search_record\n"
        "print(sorted({'gepa', 'mlflow', 'optuna'} & set(sys.modules)))\n"
    )

    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert found.stdout.strip() == "[]"
