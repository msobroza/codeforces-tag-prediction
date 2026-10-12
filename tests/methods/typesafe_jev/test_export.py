"""Tests for export_search: the checks before a search's record is frozen, the choice, the files.

Each test writes a finished search by hand (``search_run``, conftest.py): the seed and three
candidates, of which candidate 2 has the best mean score on the selection fold but copies 8
words in a row from a dataset text, and candidates 1 and 3 tie behind it.
"""

import json
import re
import subprocess
import sys

import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.export import EXPORTABLE_STOPS, export_search
from codeforces_tags.methods.typesafe_jev.search_record import (
    CANDIDATE_COLUMNS,
    HISTORY_COLUMNS,
    SELECTION_RULE,
    choose_candidate,
    read_jsonl,
    read_search_record,
    words_text,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_LINES_FILE,
    HISTORY_LINES_FILE,
    OUTCOME_FILE,
    SEARCH_SETTINGS,
)
from codeforces_tags.provenance import json_safe


def export(search_run, out, texts):
    """Export the search of ``search_run`` into ``out``, checked against ``texts``."""
    return export_search(search_run.folder, search_run.data, search_run.seed_scores, texts, out)


def snapshot(folder):
    """Return the bytes and modification time of each file under a folder, by relative path."""
    return {
        path.relative_to(folder): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_export_writes_the_params_of_the_best_passing_candidate_and_the_lowest_index_on_a_tie(
    search_run, tmp_path
):
    search_run.write_finished()
    out = tmp_path / "tuning"

    assert export(search_run, out, search_run.texts) == out

    files = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file())
    assert files == [
        "candidates.csv",
        "candidates/0.json",
        "candidates/1.json",
        "candidates/3.json",
        "history.csv",
        "params.json",
    ]
    chosen = search_run.words[1]
    params = json.loads((out / "params.json").read_text(encoding="utf-8"))
    # Candidate 2 (0.99) copies dataset text; candidates 1 and 3 tie at 0.95: 1 is chosen.
    assert params == {
        "words": json.loads(json.dumps(chosen.settings())),
        "words_sha256": chosen.sha256(),
        "chosen": 1,
        "selection_rule": SELECTION_RULE,
        "seed_selection_mean": search_run.seed_line()["selection_mean"],
        "chosen_selection_mean": 0.95,
        "prevalence": dict.fromkeys(FOCUS_TAGS, 0.125),
        "search_settings": json.loads(json.dumps(json_safe(SEARCH_SETTINGS))),
        "gepa_version": "0.1.4",
        "stop_reason": "iterations",
        "iterations": 4,
        "cost_usd": search_run.outcome["cost_usd"],
        "caps": {"max_cost_usd": 15.0, "max_iterations": 4},
    }
    # The caps this run used follow its cost.
    assert list(params)[-2:] == ["cost_usd", "caps"]
    for index in (0, 1, 3):
        text = (out / "candidates" / f"{index}.json").read_text(encoding="utf-8")
        assert text == words_text(search_run.words[index])


def test_export_keeps_a_copying_candidate_as_a_row_without_its_words_or_the_copied_text(
    search_run, tmp_path
):
    search_run.write_finished()

    out = export(search_run, tmp_path / "tuning", search_run.texts)

    text = (out / "candidates.csv").read_text(encoding="utf-8")
    candidates = pd.read_csv(out / "candidates.csv", index_col="index", keep_default_na=False)
    assert text.splitlines()[0] == ",".join(CANDIDATE_COLUMNS)
    assert candidates["passed_check"].tolist() == [True, True, False, True]
    # Candidate 2 shares 3 sequences of 8 words with the robot's text. The seed and candidate 1
    # share "codeforces tags name the ideas and techniques of" with a dataset text, but those
    # are the seed's own words, which are exempt.
    assert candidates["copied_sequences"].tolist() == [0, 0, 3, 0]
    assert candidates["parents"].astype(str).tolist() == ["", "0", "0", "1"]
    assert candidates["iteration"].tolist() == [0, 1, 2, 4]
    assert candidates["words_sha256"].tolist() == [
        search_run.words[index].sha256() for index in range(4)
    ]
    assert "robot" not in text
    assert not (out / "candidates" / "2.json").exists()


def test_export_chooses_the_best_candidate_once_no_dataset_text_holds_its_words(
    search_run, tmp_path
):
    search_run.write_finished()

    out = export(search_run, tmp_path / "tuning", search_run.texts[1:])

    assert json.loads((out / "params.json").read_text(encoding="utf-8"))["chosen"] == 2
    assert (out / "candidates" / "2.json").is_file()


def test_export_keeps_two_candidates_that_hold_the_same_words_and_reads_both_back(
    search_run, tmp_path
):
    # A reflection on candidate 2 returned the convention of candidate 1, which is not its
    # parent: candidate 3 holds candidate 1's words, so Jev scores it as candidate 1.
    lines = search_run.write_finished()
    lines[3] = search_run.candidate_line(3, search_run.words[1], 0.95, iteration=4, parents=(2,))
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)
    history = read_jsonl(search_run.folder / HISTORY_LINES_FILE)
    history[3]["parent"] = 2  # iteration 4, which accepted candidate 3, reflected on candidate 2
    search_run.write_lines(HISTORY_LINES_FILE, history)

    out = export(search_run, tmp_path / "tuning", search_run.texts)

    candidates = pd.read_csv(out / "candidates.csv", index_col="index", keep_default_na=False)
    assert candidates.index.tolist() == [0, 1, 2, 3]
    assert candidates["parents"].astype(str).tolist() == ["", "0", "0", "2"]
    assert candidates.loc[3, "words_sha256"] == candidates.loc[1, "words_sha256"]
    assert candidates["passed_check"].tolist() == [True, True, False, True]
    assert candidates["copied_sequences"].tolist() == [0, 0, 3, 0]
    first = (out / "candidates" / "1.json").read_text(encoding="utf-8")
    assert (out / "candidates" / "3.json").read_text(encoding="utf-8") == first
    assert first == words_text(search_run.words[1])
    # Equal means: the tie goes to the lower index, whatever the order of the lines; the rule
    # reads the mean first, so a higher mean would take candidate 3 despite the same words.
    params = json.loads((out / "params.json").read_text(encoding="utf-8"))
    assert params["chosen"] == 1
    copied = {0: [], 1: [], 2: ["eight words copied from a problem of the dataset"], 3: []}
    assert choose_candidate(lines[::-1], copied) == 1
    assert choose_candidate([*lines[:3], lines[3] | {"selection_mean": 0.96}], copied) == 3
    record = read_search_record(out)
    assert sorted(record.words) == [0, 1, 3]
    assert record.words[3].sha256() == record.words[1].sha256() == params["words_sha256"]
    assert record.candidates.loc[3, "words_sha256"] == record.candidates.loc[1, "words_sha256"]
    assert record.history.loc[4, "parent"] == 2


def test_export_writes_one_history_row_per_iteration_without_its_minibatch(search_run, tmp_path):
    search_run.write_finished()

    out = export(search_run, tmp_path / "tuning", search_run.texts)

    header, *rows = (out / "history.csv").read_text(encoding="utf-8").splitlines()
    assert header == ",".join(HISTORY_COLUMNS)
    assert rows == [
        "1,0,convention,3.5,3.75,True,,1,0.01,0.002,1500,800",
        "2,0,convention,3.5,3.75,True,,2,0.01,0.002,1500,800",
        "3,2,convention,3.5,,False,too_long,,0.01,0.002,1500,800",
        "4,1,convention,3.5,3.75,True,,3,0.01,0.002,1500,800",
    ]


def test_export_refuses_a_search_that_has_not_stopped_at_a_cap(search_run, tmp_path):
    search_run.write_finished()
    (search_run.folder / OUTCOME_FILE).unlink()
    out = tmp_path / "tuning"

    with pytest.raises(ValueError, match="outcome.json does not exist: the search has not stopped"):
        export(search_run, out, search_run.texts)
    search_run.outcome["stop_reason"] = "error"
    search_run.write_outcome()
    with pytest.raises(ValueError, match="stopped on 'error', not at one of its caps"):
        export(search_run, out, search_run.texts)
    assert EXPORTABLE_STOPS == ("iterations", "cost")
    assert not out.exists()


def test_export_checks_the_seeds_score_against_its_frozen_scores_within_the_tolerance(
    search_run, tmp_path
):
    lines = search_run.write_finished()
    lines[0]["selection_mean"] += 2e-9  # beyond config.TUNING_EXPORT_TOLERANCE (1e-9)
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)
    out = tmp_path / "tuning"

    with pytest.raises(ValueError, match="the search did not read the seed's committed answers"):
        export(search_run, out, search_run.texts)
    assert not out.exists()
    lines[0]["selection_mean"] -= 1.5e-9  # 5e-10 from the frozen scores' value: within it
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)
    assert export(search_run, out, search_run.texts) == out


def test_export_refuses_a_search_whose_candidate_0_is_not_the_seed(search_run, tmp_path):
    lines = search_run.write_finished()
    mean = lines[0]["selection_mean"]
    lines[0] = search_run.candidate_line(0, search_run.words[3], mean, iteration=0, parents=())
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)

    with pytest.raises(ValueError, match="Candidate 0 holds other words than the seed's"):
        export(search_run, tmp_path / "tuning", search_run.texts)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda lines: lines[2].update(words_sha256="0" * 64),
            r"the words_sha256 of the candidates \[2\] is not the hash of their words",
        ),
        (lambda lines: lines.pop(1), r"from the seed, not \[0, 2, 3\]"),
    ],
    ids=["another hash", "a missing candidate"],
)
def test_export_refuses_candidate_lines_that_are_not_numbered_or_hashed_as_written(
    search_run, tmp_path, change, message
):
    lines = search_run.write_finished()
    change(lines)
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)

    with pytest.raises(ValueError, match=message):
        export(search_run, tmp_path / "tuning", search_run.texts)


def test_export_refuses_a_history_line_without_every_key(search_run, tmp_path):
    search_run.write_finished()
    lines = [search_run.history_line(1, candidate=1), search_run.history_line(2, candidate=2)]
    del lines[1]["luna_output_tokens"]
    search_run.write_lines(HISTORY_LINES_FILE, lines)

    with pytest.raises(ValueError, match=r"The history lines of the iterations \[2\]"):
        export(search_run, tmp_path / "tuning", search_run.texts)


def test_export_writes_once_then_only_compares_byte_for_byte(search_run, tmp_path):
    search_run.write_finished()
    out = export(search_run, tmp_path / "tuning", search_run.texts)
    before = snapshot(out)

    assert export(search_run, out, search_run.texts) == out

    assert snapshot(out) == before
    # No staging folder is left next to it.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["experiments", "tuning"]


def test_export_refuses_a_search_that_now_gives_another_record(search_run, tmp_path):
    lines = search_run.write_finished()
    out = export(search_run, tmp_path / "tuning", search_run.texts)
    lines[3]["selection_mean"] = 0.97  # candidate 3 would now be chosen
    search_run.write_lines(CANDIDATE_LINES_FILE, lines)

    with pytest.raises(ValueError, match=re.escape("['candidates.csv', 'params.json'] differ")):
        export(search_run, out, search_run.texts)
    search_run.write_finished()  # the search as it was exported
    (out / "candidates" / "2.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("['candidates/2.json'] differ")):
        export(search_run, out, search_run.texts)


def test_the_export_module_imports_neither_gepa_nor_mlflow():
    # It names the search's data under TYPE_CHECKING only: export reads a finished search.
    code = (
        "import sys\n"
        "import codeforces_tags.methods.typesafe_jev.export\n"
        "print(sorted({'gepa', 'mlflow', 'optuna'} & set(sys.modules)))\n"
    )

    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert found.stdout.strip() == "[]"
