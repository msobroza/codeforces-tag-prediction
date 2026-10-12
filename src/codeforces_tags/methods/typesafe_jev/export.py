"""The export of a finished search over Jev's words into its frozen record (docs/adr/0010).

``export_search`` reads the search folder that ``tuning.run_search`` wrote (candidates.jsonl,
history.jsonl and outcome.json) and freezes the search's record (``search_record``) in the
searched method's ``tuning/`` once, after the checks of docs/adr/0010: the search stopped at a
cap, its seed reproduces the frozen scores of jev-statement-code, and no kept word set copies 8
words in a row of a dataset text. A second export only compares the files byte for byte.

It imports neither gepa nor mlflow: it reads the search's files, and names the search's data
(``tuning.SearchData``) for type checking only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS, TUNING_EXPORT_TOLERANCE
from codeforces_tags.methods.typesafe_jev.objective import problem_scores
from codeforces_tags.methods.typesafe_jev.search_record import (
    SELECTION_RULE,
    choose_candidate,
    read_jsonl,
    record_files,
    write_once,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_LINES_FILE,
    GEPA_NAME,
    GEPA_SEED_NAME,
    HISTORY_LINES_FILE,
    OUTCOME_FILE,
    SEARCH_SETTINGS,
)
from codeforces_tags.methods.typesafe_jev.words import (
    SEED_WORDS,
    copied_sequences,
    words_from_json,
)

if TYPE_CHECKING:  # tuning imports gepa, which export never needs
    from codeforces_tags.methods.typesafe_jev.tuning import SearchData

# How a search must have stopped for export to freeze its record: at a cap, never on an error.
EXPORTABLE_STOPS: tuple[str, ...] = ("iterations", "cost")


def export_search(
    folder: Path,
    data: SearchData,
    seed_scores: pd.DataFrame,
    dataset_texts: Sequence[str],
    out: Path,
) -> Path:
    """Freeze a finished search's record in ``out`` once, after the checks of docs/adr/0010.

    The search must have stopped at a cap (outcome.json), candidate 0 must be the seed, and its
    mean score on the selection fold must equal the one the seed's frozen scores give, within
    ``config.TUNING_EXPORT_TOLERANCE``. A candidate whose words share 8 words in a row with a
    dataset text, beyond the seed's own words, fails the dataset-text check: it keeps its row,
    without its words, and the selection rule (``search_record.SELECTION_RULE``) passes it by.
    It takes about two seconds per distinct candidate on the full dataset.

    Args:
        folder: The search folder (``tuning.search_folder()``): candidates.jsonl, history.jsonl
            and outcome.json.
        data: The search's problems; its selection fold, labels and weights score the seed.
        seed_scores: The frozen dev probabilities of jev-statement-code, indexed by sample id.
        dataset_texts: The statement, notes and solution of every problem of the dataset
            (``search_record.dataset_texts``).
        out: The record's folder, the searched method's ``tuning/``.

    Returns:
        ``out``, written once; a second export only compares its files byte for byte.

    Raises:
        ValueError: If the search did not stop at a cap, if its lines are not numbered from the
            seed or not hashed as written, if candidate 0 is not the seed or its score differs
            from the frozen one, if a history line lacks a key, or if ``out`` holds another
            record.
    """
    folder = Path(folder)
    outcome = _finished_outcome(folder)
    candidates = read_jsonl(folder / CANDIDATE_LINES_FILE)
    _check_candidate_lines(folder, candidates)
    _check_seed(candidates[0], data, seed_scores)
    copied = _copied_by_candidate(candidates, dataset_texts)
    chosen = choose_candidate(candidates, copied)
    params = _search_params(candidates[chosen], candidates[0], data, outcome)
    history = read_jsonl(folder / HISTORY_LINES_FILE)
    write_once(record_files(params, candidates, copied, history), out)
    return out


def _finished_outcome(folder: Path) -> dict[str, object]:
    """Read outcome.json, or fail unless the search stopped at one of its caps."""
    path = folder / OUTCOME_FILE
    if not path.is_file():
        raise ValueError(
            f"{path} does not exist: the search has not stopped at its iteration or cost cap "
            "(it is running, or it stopped on an error). Run `python -m "
            f"codeforces_tags.methods.typesafe_jev tune --method {GEPA_NAME}` until it does."
        )
    outcome = json.loads(path.read_text(encoding="utf-8"))
    if outcome.get("stop_reason") not in EXPORTABLE_STOPS:
        raise ValueError(
            f"{path} says that the search stopped on {outcome.get('stop_reason')!r}, not at one "
            f"of its caps {list(EXPORTABLE_STOPS)}: only a finished search is exported "
            "(docs/adr/0010)."
        )
    return outcome


def _check_candidate_lines(folder: Path, candidates: Sequence[Mapping[str, object]]) -> None:
    """Fail unless the candidates are numbered from the seed and each hash is its words'."""
    path = folder / CANDIDATE_LINES_FILE
    indices = [line["index"] for line in candidates]
    if not candidates or indices != list(range(len(candidates))):
        raise ValueError(
            f"{path} must number its candidates 0, 1, 2, ... from the seed, not {indices}."
        )
    wrong = [
        line["index"]
        for line in candidates
        if words_from_json(line["words"]).sha256() != line["words_sha256"]
    ]
    if wrong:
        raise ValueError(
            f"{path}: the words_sha256 of the candidates {wrong} is not the hash of their words."
        )


def _check_seed(seed: Mapping[str, object], data: SearchData, seed_scores: pd.DataFrame) -> None:
    """Fail unless candidate 0 is the seed, scored as the frozen seed's probabilities score."""
    if seed["words_sha256"] != SEED_WORDS.sha256():
        raise ValueError(
            "Candidate 0 holds other words than the seed's, the frozen words of "
            f"{GEPA_SEED_NAME}: the search did not start from its seed (docs/adr/0010)."
        )
    ids = data.selection_ids
    frozen = problem_scores(
        seed_scores.loc[ids, list(FOCUS_TAGS)], data.labels.loc[ids], data.weights
    )
    expected = float(frozen.mean())
    found = seed["selection_mean"]
    if not math.isclose(found, expected, rel_tol=0.0, abs_tol=TUNING_EXPORT_TOLERANCE):
        raise ValueError(
            f"The seed's mean score on the selection fold is {found!r} in the search, but "
            f"{expected!r} from the frozen scores of {GEPA_SEED_NAME}: the search did not read "
            "the seed's committed answers (docs/adr/0010)."
        )


def _copied_by_candidate(
    candidates: Sequence[Mapping[str, object]], dataset_texts: Sequence[str]
) -> dict[int, list[str]]:
    """Return the sequences each candidate shares with the dataset texts, once per word set."""
    distinct = {line["words_sha256"]: line["words"] for line in candidates}
    texts = list(dataset_texts)
    shared = {
        sha256: copied_sequences(words_from_json(words), texts)
        for sha256, words in distinct.items()
    }
    return {line["index"]: shared[line["words_sha256"]] for line in candidates}


def _search_params(
    chosen: Mapping[str, object],
    seed: Mapping[str, object],
    data: SearchData,
    outcome: Mapping[str, object],
) -> dict[str, object]:
    """Return what params.json holds: the chosen words, the rule, the scores, the settings."""
    learning = data.labels.loc[data.learning_ids]
    return {
        "words": words_from_json(chosen["words"]).settings(),
        "words_sha256": chosen["words_sha256"],
        "chosen": chosen["index"],
        "selection_rule": SELECTION_RULE,
        "seed_selection_mean": seed["selection_mean"],
        "chosen_selection_mean": chosen["selection_mean"],
        "prevalence": {tag: float(learning[tag].mean()) for tag in FOCUS_TAGS},
        "search_settings": SEARCH_SETTINGS,
        "gepa_version": outcome["gepa_version"],
        "stop_reason": outcome["stop_reason"],
        "iterations": outcome["iterations"],
        "cost_usd": outcome["cost_usd"],
        "caps": {
            "max_cost_usd": outcome["max_cost_usd"],
            "max_iterations": outcome["max_iterations"],
        },
    }
