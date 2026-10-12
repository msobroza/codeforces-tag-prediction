"""The frozen record of a search over Jev's words, in the searched method's ``tuning/`` folder.

``export`` (``export.export_search``) writes it once, after the checks of docs/adr/0010:

- ``params.json``: the chosen words and candidate, the selection rule, the seed's and the chosen
  candidate's mean scores on the selection fold, each focus tag's prevalence in the learning
  folds, every search setting, GEPA's version, the stop reason, the iterations, the costs and the
  caps the search ran under;
- ``candidates.csv``: one row per candidate scored on the selection fold (``CANDIDATE_COLUMNS``);
- ``candidates/<index>.json``: the words of each candidate that passed the dataset-text check;
- ``history.csv``: one row per iteration of the search (``HISTORY_COLUMNS``).

A candidate whose words share 8 words in a row with a problem of the dataset keeps its row, with
the number of sequences it shares, but neither its words nor the sequences, which are dataset
text (docs/adr/0004). This module writes and reads these files, applies the selection rule
(``choose_candidate``), reads a candidate's parents however they were read (``parents_of``) and
computes the run record's search entries (``search_entries``). It imports neither gepa nor
mlflow: ``freeze``, ``log-search`` and notebook 10 read the record without them.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.typesafe_jev.questions import question_id
from codeforces_tags.methods.typesafe_jev.settings import (
    CANDIDATE_WORDS_FOLDER,
    CANDIDATES_FILE,
    HISTORY_FILE,
    PARAMS_FILE,
    SEARCH_HELD_OUT_FOLD,
    SEARCH_LEARNING_FOLDS,
    SEARCH_SELECTION_FOLD,
)
from codeforces_tags.methods.typesafe_jev.words import JevWords, read_words, words_from_json
from codeforces_tags.provenance import json_safe

# The columns of history.csv: the keys of a history.jsonl line but its minibatch, in order.
HISTORY_COLUMNS: tuple[str, ...] = (
    "iteration",
    "parent",
    "piece",
    "parent_sum",
    "child_sum",
    "accepted",
    "refusal",
    "candidate",
    "jev_cost_usd",
    "luna_cost_usd",
    "luna_input_tokens",
    "luna_output_tokens",
)
# A candidate's balanced Brier score of each focus tag on the selection fold, a column each,
# named by the tag's question id ("number theory" is brier_number_theory).
BRIER_COLUMNS: tuple[str, ...] = tuple(f"brier_{question_id(tag)}" for tag in FOCUS_TAGS)
# The columns of candidates.csv. copied_sequences counts the sequences of 8 words a candidate
# shares with the dataset's texts (0 when it passed the check); the sequences are never written.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "index",
    "parents",
    "iteration",
    "piece",
    "selection_mean",
    *BRIER_COLUMNS,
    "words_sha256",
    "passed_check",
    "copied_sequences",
)
# The texts of a problem that no kept word set may copy: its statement, notes and solution.
DATASET_TEXT_COLUMNS: tuple[str, ...] = ("statement", "prob_desc_notes", "source_code")
# The selection rule, fixed before the search (docs/adr/0010) and written into params.json.
SELECTION_RULE = (
    "the candidate with the highest mean score on the selection fold among those whose words "
    "pass the dataset-text check; ties go to the lowest candidate index"
)
# The columns of each file that hold a text, "" for none: an empty cell elsewhere is missing.
_CANDIDATE_TEXTS = ("parents", "piece", "words_sha256")
_HISTORY_TEXTS = ("piece", "refusal")


@dataclass(frozen=True)
class SearchRecord:
    """The frozen record of a search, as ``read_search_record`` reads it from ``tuning/``.

    Attributes:
        params: ``params.json`` (the keys are listed in the module's docstring).
        candidates: ``candidates.csv``, indexed by ``index`` (the seed is 0), with the columns of
            ``CANDIDATE_COLUMNS`` after it. ``parents`` holds the parents' indices separated by
            spaces (``parents_of`` reads them) and ``piece`` the piece that changed, both "" for
            the seed.
        history: ``history.csv``, indexed by ``iteration`` (from 1), with the columns of
            ``HISTORY_COLUMNS`` after it. ``refusal`` is "" unless the proposal was refused,
            ``child_sum`` NaN when no child was scored, and ``candidate`` (Int64) missing unless
            the child was accepted.
        words: The words of each candidate that passed the dataset-text check, by index.
    """

    params: dict[str, object]
    candidates: pd.DataFrame
    history: pd.DataFrame
    words: dict[int, JevWords]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a file of JSON lines, such as the search's candidates.jsonl or history.jsonl.

    Args:
        path: The file.

    Returns:
        One dict per line that is not blank, in order.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def dataset_texts(problems: pd.DataFrame) -> list[str]:
    """List the texts of the problems that no kept word set may copy.

    Args:
        problems: Preprocessed problems (``raw_to_problems``), every problem of the dataset.

    Returns:
        The texts of ``DATASET_TEXT_COLUMNS``, column by column, each in the problems' order; a
        missing text (e.g. a problem without notes) is left out.
    """
    return [
        text
        for column in DATASET_TEXT_COLUMNS
        for text in problems[column]
        if isinstance(text, str)
    ]


def choose_candidate(
    candidates: Sequence[Mapping[str, object]], copied: Mapping[int, Sequence[str]]
) -> int:
    """Apply the selection rule (``SELECTION_RULE``) to the candidates of a search.

    Args:
        candidates: The lines of candidates.jsonl, each with its ``index`` and ``selection_mean``.
        copied: The sequences each candidate shares with the dataset's texts, by index; a
            candidate passes the dataset-text check when it shares none.

    Returns:
        The index of the passing candidate with the highest mean score on the selection fold,
        the lowest index among equal scores.

    Raises:
        ValueError: If no candidate passes (a search's seed always does: its own words are
            exempt).
    """
    passing = [line for line in candidates if not copied[line["index"]]]
    if not passing:
        raise ValueError(
            "No candidate passed the dataset-text check among the candidates "
            f"{[line['index'] for line in candidates]}: the seed's own words are exempt, so a "
            "search's candidate 0 always passes."
        )
    best = max(passing, key=lambda line: (line["selection_mean"], -line["index"]))
    return int(best["index"])


def parents_of(value: object) -> tuple[int, ...]:
    """Read a candidate's parents, as a line, the record or a plain CSV read gives them.

    Args:
        value: A list of indices (a line of candidates.jsonl), the indices separated by spaces
            (``SearchRecord.candidates``), one index as a number (candidates.csv read by
            ``pd.read_csv`` alone: a float once the seed's empty cell is NaN), or the seed's
            ``""``, ``[]``, NaN or None.

    Returns:
        The parents' indices, in order; ``()`` for the seed.

    Raises:
        ValueError: If an index is not a whole number.
    """
    if isinstance(value, str):
        return tuple(int(part) for part in value.split())
    if isinstance(value, Sequence):
        return tuple(_whole(part) for part in value)
    if pd.isna(value):
        return ()
    return (_whole(value),)


def _whole(value: object) -> int:
    """Return a candidate index given as a number, which must be whole."""
    if float(value) != int(value):
        raise ValueError(f"A candidate index is a whole number, not {value!r}.")
    return int(value)


def candidate_frame(
    candidates: Sequence[Mapping[str, object]], copied: Mapping[int, Sequence[str]]
) -> pd.DataFrame:
    """Build the table of candidates.csv from the lines of candidates.jsonl.

    Args:
        candidates: The lines of candidates.jsonl, the seed first.
        copied: The sequences each candidate shares with the dataset's texts, by index.

    Returns:
        One row per candidate, indexed by ``index``, with the other columns of
        ``CANDIDATE_COLUMNS``; it holds the number of shared sequences, never their text.
    """
    rows = [_candidate_row(line, copied[line["index"]]) for line in candidates]
    return pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS)).set_index("index")


def _candidate_row(line: Mapping[str, object], copied: Sequence[str]) -> dict[str, object]:
    """Return one candidate's row of candidates.csv."""
    brier = line["selection_brier"]
    return {
        "index": line["index"],
        "parents": " ".join(str(parent) for parent in line["parents"]),
        "iteration": line["iteration"],
        "piece": line["piece"] or "",
        "selection_mean": line["selection_mean"],
        **{column: brier[tag] for column, tag in zip(BRIER_COLUMNS, FOCUS_TAGS, strict=True)},
        "words_sha256": line["words_sha256"],
        "passed_check": not copied,
        "copied_sequences": len(copied),
    }


def history_frame(history: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Build the table of history.csv from the lines of history.jsonl.

    Args:
        history: The lines of history.jsonl, one per iteration, in order.

    Returns:
        The columns of ``HISTORY_COLUMNS``, one row per iteration: the minibatch is left out,
        a missing refusal is "" and ``candidate`` is Int64.

    Raises:
        ValueError: If a line does not hold exactly the keys of ``HISTORY_COLUMNS`` and
            ``minibatch``.
    """
    keys = {*HISTORY_COLUMNS, "minibatch"}
    odd = [line.get("iteration") for line in history if set(line) != keys]
    if odd:
        raise ValueError(
            f"The history lines of the iterations {odd} do not hold exactly the keys "
            f"{sorted(keys)}: the search's recorder wrote another history than export reads."
        )
    frame = pd.DataFrame(list(history), columns=list(HISTORY_COLUMNS))
    return frame.assign(
        refusal=frame["refusal"].fillna(""), candidate=frame["candidate"].astype("Int64")
    )


def words_text(words: JevWords) -> str:
    """Return the text of a candidate's words file, ``candidates/<index>.json``.

    Args:
        words: The candidate's words.

    Returns:
        ``words.settings()`` as indented JSON, non-ASCII kept, with a final newline.
    """
    return json.dumps(words.settings(), indent=2, ensure_ascii=False) + "\n"


def record_files(
    params: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    copied: Mapping[int, Sequence[str]],
    history: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    """Render every file of a search's record, by its path relative to ``tuning/``.

    Args:
        params: What params.json holds.
        candidates: The lines of candidates.jsonl, the seed first.
        copied: The sequences each candidate shares with the dataset's texts, by index.
        history: The lines of history.jsonl.

    Returns:
        The bytes of params.json, candidates.csv, history.csv and of the words file of each
        candidate that passed the dataset-text check, by POSIX path.

    Raises:
        ValueError: As ``history_frame``.
    """
    text = json.dumps(json_safe(params), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    files = {
        PARAMS_FILE: text.encode("utf-8"),
        CANDIDATES_FILE: candidate_frame(candidates, copied).to_csv(lineterminator="\n").encode(),
        HISTORY_FILE: history_frame(history).to_csv(index=False, lineterminator="\n").encode(),
    }
    for line in candidates:
        if not copied[line["index"]]:
            words = words_from_json(line["words"])
            files[f"{CANDIDATE_WORDS_FOLDER}/{line['index']}.json"] = words_text(words).encode()
    return files


def write_once(files: Mapping[str, bytes], out: Path, *, owner: str = "the search") -> None:
    """Write a record's files into ``out`` once; when ``out`` exists, only compare them.

    The files are written in a staging folder next to ``out``, which is renamed into place
    once all are written: an interrupted export leaves no half record.

    Args:
        files: The bytes of each file, by POSIX path relative to ``out``.
        out: The record's folder: the searched method's ``tuning/``, or a calibrated twin's
            (``platt.export_platt``, docs/adr/0011).
        owner: What the record is of, which a refusal names: the search by default.

    Raises:
        ValueError: If ``out`` exists and holds another file, or lacks one, or one differs.
    """
    out = Path(out)
    if out.exists():
        _check_same_files(files, out, owner)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as staging:
        staged = Path(staging) / out.name
        for relative, content in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        staged.replace(out)


def _check_same_files(files: Mapping[str, bytes], out: Path, owner: str) -> None:
    """Fail unless ``out`` holds exactly ``files``, byte for byte."""
    found = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    differ = sorted(
        name for name in files.keys() | found.keys() if files.get(name) != found.get(name)
    )
    if differ:
        raise ValueError(
            f"{out} holds another record of {owner} than the one computed now: {differ} differ. "
            "A record is written once (docs/adr/0010, docs/adr/0011): restore it from git if it "
            "was edited; another search or threshold rule is a new method under a new name."
        )


def read_search_record(tuning: Path) -> SearchRecord:
    """Read the frozen record of a search back from its folder, and check its words.

    Args:
        tuning: The searched method's ``tuning/`` folder, as ``export`` wrote it.

    Returns:
        Its params, candidates, history and the words of each candidate that passed.

    Raises:
        ValueError: If the words files are not exactly those of the candidates that passed,
            under their ``words_sha256``, or if params.json does not hold the words of its
            chosen candidate.
    """
    tuning = Path(tuning)
    params = json.loads((tuning / PARAMS_FILE).read_text(encoding="utf-8"))
    candidates = _read_table(tuning / CANDIDATES_FILE, "index", _CANDIDATE_TEXTS, {})
    history = _read_table(
        tuning / HISTORY_FILE, "iteration", _HISTORY_TEXTS, {"candidate": "Int64"}
    )
    passed = [int(index) for index in candidates.index[candidates["passed_check"]]]
    words = {
        index: read_words(tuning / CANDIDATE_WORDS_FOLDER / f"{index}.json") for index in passed
    }
    _check_words(tuning, params, candidates, words)
    return SearchRecord(params=params, candidates=candidates, history=history, words=words)


def _read_table(
    path: Path, index: str, texts: Sequence[str], dtypes: Mapping[str, str]
) -> pd.DataFrame:
    """Read a CSV file of the record: "" in a text column, NaN in any other empty cell."""
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(
        path,
        index_col=index,
        dtype=dict.fromkeys(texts, "str") | dict(dtypes),
        keep_default_na=False,
        na_values={column: [""] for column in header if column not in texts},
        float_precision="round_trip",
    )


def _check_words(
    tuning: Path,
    params: Mapping[str, object],
    candidates: pd.DataFrame,
    words: Mapping[int, JevWords],
) -> None:
    """Fail unless the words files and params.json hold the words the candidates name."""
    folder = tuning / CANDIDATE_WORDS_FOLDER
    found = {path.name for path in folder.iterdir()}
    wrong = sorted(found ^ {f"{index}.json" for index in words})
    wrong += [
        f"{index}.json"
        for index, value in words.items()
        if value.sha256() != candidates.at[index, "words_sha256"]
    ]
    if wrong:
        raise ValueError(
            f"{folder} must hold the words of each candidate that passed the dataset-text "
            f"check, under its words_sha256, and nothing else: {wrong} do not."
        )
    chosen = words.get(params["chosen"])
    stated = read_words(tuning / PARAMS_FILE).sha256()
    if chosen is None or len({stated, params["words_sha256"], chosen.sha256()}) != 1:
        raise ValueError(
            f"{tuning / PARAMS_FILE} must hold the words of its chosen candidate "
            f"{params['chosen']}, a candidate that passed the dataset-text check, under their "
            "words_sha256."
        )


def search_entries(
    params: Mapping[str, object],
    params_sha256: str,
    dev_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.Series,
) -> dict[str, object]:
    """Return what the searched method's run record keeps of its search (``"search"``).

    Args:
        params: The searched method's params.json.
        params_sha256: The sha256 of that file, which the run record pins.
        dev_predictions: The frozen 0/1 dev predictions, indexed by sample id, a column per
            focus tag (out of fold for the thresholds).
        labels: The 0/1 focus tags of the dev problems, indexed like ``dev_predictions``.
        folds: Each dev problem's fold, indexed by sample id.

    Returns:
        The folds of the search, the chosen candidate, ``params_sha256``, the search's cost,
        and the macro F1 and each focus tag's F1 on the held-out fold, the one fold whose
        problems neither the search nor the words saw.
    """
    held_out = folds.index[folds == SEARCH_HELD_OUT_FOLD]
    tags = list(FOCUS_TAGS)
    scores = evaluate(labels.loc[held_out, tags], dev_predictions.loc[held_out, tags])
    return {
        "held_out_fold": SEARCH_HELD_OUT_FOLD,
        "selection_fold": SEARCH_SELECTION_FOLD,
        "learning_folds": list(SEARCH_LEARNING_FOLDS),
        "chosen": params["chosen"],
        "params_sha256": params_sha256,
        "cost_usd": params["cost_usd"],
        "held_out_macro_f1": scores.macro_f1,
        "held_out_f1": {tag: float(f1) for tag, f1 in scores.tag_f1.items()},
    }
