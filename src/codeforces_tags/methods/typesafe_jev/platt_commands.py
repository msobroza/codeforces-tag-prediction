"""The handlers of a calibrated twin's commands, and the gates on its base's files (docs/adr/0011).

``__main__`` runs the gates every command shares (the dataset, the clean tree, the committed
split) and hands these functions what they read. A calibrated twin (``settings.PLATT_BASES``)
sends no request: it reads its base's committed answers files, ``scores.csv`` and
``run_record.json``, which every gate here checks by git, and names by sha256 and commit.

- ``export`` writes the twin's dev-only record in its ``tuning/`` once (``platt.export_platt``),
  from the dev rows of its base's ``scores.csv`` only; a second run checks the base's files
  against the record (``platt.check_base_files``), then only compares.
- ``check_freeze_inputs``, ``check_derived``, ``record_entries``, ``check_record`` and
  ``copy_base_files`` are the twin's own steps of ``freeze``, and ``verify_twin`` those of
  ``verify``; ``threshold_class`` names the class whose fit chooses a Jev method's thresholds;
  ``check_base_history`` refuses a base file whose recorded commit a rewritten history lost.

It imports neither gepa nor mlflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from codeforces_tags.data.labels import focus_labels
from codeforces_tags.evaluation.predictions import SCORES_FILE, load_run_record, load_scores
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev import platt
from codeforces_tags.methods.typesafe_jev.settings import (
    PARAMS_FILE,
    PLATT_BASES,
    PLATT_FILES,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.words import base_of, settings_entry
from codeforces_tags.model.platt_lipton import PlattLiptonThresholds
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.provenance import (
    committed_unchanged,
    committed_versions,
    file_sha256,
    json_safe,
    last_commit,
)

# The command line of the package, which the messages name.
_PROGRAM = "python -m codeforces_tags.methods.typesafe_jev"


def base_files(base: str) -> dict[str, dict[str, str]]:
    """Name each file of a base that its twin reads by its sha256 and its last commit.

    Args:
        base: The twin's base, a frozen Jev method.

    Returns:
        ``{"sha256": ..., "commit": ...}`` of each file of ``platt.base_paths(base)``, by file
        name.
    """
    return {
        file_name: {"sha256": file_sha256(path), "commit": last_commit(path)}
        for file_name, path in platt.base_paths(base).items()
    }


def check_base_committed(twin: str, *others: Path) -> None:
    """Refuse to go on until a twin's base's files that it reads are committed and unchanged.

    Args:
        twin: A calibrated twin.
        others: More of its base's files to check, e.g. the ``latency.csv`` that freeze copies.

    Raises:
        ValueError: If one of the files is not committed, or has changed since its commit.
    """
    base = base_of(twin)
    for path in (*platt.base_paths(base).values(), *others):
        if not committed_unchanged(path):
            raise ValueError(
                f"{path} is not committed, or has changed since: {twin} reads the committed "
                f"answers files, scores and run record of {base}, and its freeze copies the "
                f"base's latency.csv and per_problem.csv (docs/adr/0011). Freeze and time {base} "
                "and commit its files, or restore them from git."
            )


def write_record(twin: str, raw_dev: pd.DataFrame, problems: pd.DataFrame) -> None:
    """Write a twin's tuning/ record from its base's committed dev scores, or compare it.

    Args:
        twin: A calibrated twin.
        raw_dev: The dev part's raw problems, with their fold and group (``select_part``).
        problems: Every problem of the dataset, preprocessed: the labels of the dev ones are
            read.

    Raises:
        ValueError: As ``platt.export_platt``.
    """
    base = base_of(twin)
    dev = platt.DevScores(
        scores=load_scores(method_dir(base)).loc["dev"],  # the dev rows only: no test score
        labels=focus_labels(problems).loc[raw_dev.index],
        folds=raw_dev["fold"],
        groups=raw_dev["group"],
    )
    platt.export_platt(twin, method_dir(twin) / TUNING_FOLDER, dev, base_files(base))


def export(twin: str, raw_dev: pd.DataFrame, problems: pd.DataFrame) -> int:
    """Write a twin's dev-only record in its tuning/ once, from its base's committed files.

    Args:
        twin: A calibrated twin.
        raw_dev: The dev part's raw problems, with their fold and group (``select_part``).
        problems: Every problem of the dataset, preprocessed.

    Returns:
        0, once tuning/ holds the record, written now or found byte for byte.

    Raises:
        ValueError: If a file of the base is not committed and unchanged; if the record exists
            and names a commit the history of HEAD no longer holds (``check_base_history``), or
            other files of the base than those committed now (``platt.check_base_files``, so
            that the refusal names the base's files, not the record); or as
            ``platt.export_platt``.
    """
    check_base_committed(twin)
    out = method_dir(twin) / TUNING_FOLDER
    existed = out.exists()
    if existed:
        check_base_history(twin, platt.read_platt_params(out)["base_files"])
        platt.check_base_files(out, base_files(base_of(twin)))
    write_record(twin, raw_dev, problems)
    params = platt.read_platt_params(out)
    state = "holds the same record" if existed else "now holds the record"
    print(
        f"{out} {state} of the threshold rule {params['rule']} on the dev scores of "
        f"{params['base']}; the rule chosen inside each fold's other folds would be "
        f"{params['nested_choice']}, for a nested macro F1 of {params['nested_macro_f1']:.4f}."
    )
    if not existed:
        print(f"Commit it, then freeze the twin with `{_PROGRAM} freeze --method {twin}`.")
    return 0


def threshold_class(name: str) -> type[CrossFittedTagThresholds]:
    """Return the class whose fit chooses a Jev method's thresholds.

    Args:
        name: A Jev method.

    Returns:
        ``PlattLiptonThresholds`` for a calibrated twin (docs/adr/0011), else
        ``CrossFittedTagThresholds``, the F1 argmax.
    """
    return PlattLiptonThresholds if name in PLATT_BASES else CrossFittedTagThresholds


def check_base_history(twin: str, recorded: Mapping[str, Mapping[str, str]]) -> None:
    """Fail when a commit recorded for a base file is no longer in the file's history at HEAD.

    Args:
        twin: A calibrated twin.
        recorded: The sha256 and commit of each base file, by file name, as the twin's
            ``tuning/params.json`` records them.

    Raises:
        ValueError: If the history of HEAD holds no commit of that sha that changed the file:
            a squash, a rebase or a shallow clone rewrote it.
    """
    for file_name, path in platt.base_paths(base_of(twin)).items():
        commit = recorded.get(file_name, {}).get("commit")
        if commit is not None and commit not in {sha for sha, _ in committed_versions(path)}:
            raise ValueError(
                f"{path} is named by the commit {commit}, which the history of HEAD no longer "
                f"holds: the history of the files {twin} reads was rewritten, by a squash, a "
                "rebase or a shallow clone. A calibrated twin names its base's files by commit "
                "(docs/adr/0011), so the pull request that holds them keeps its history: merge "
                "it with a merge commit, never squash or rebase it, and work in a full clone."
            )


def check_freeze_inputs(twin: str) -> None:
    """Refuse a twin's freeze until its record and its base's files are committed and agree.

    Args:
        twin: A calibrated twin.

    Raises:
        ValueError: If a file of the twin's ``tuning/`` (``settings.PLATT_FILES``) is not
            committed and unchanged; if a file of its base that it reads (the answers files,
            scores, run record, ``latency.csv`` and ``per_problem.csv``) is not; if params.json
            names a commit that the history of HEAD no longer holds; or if it names other
            versions of them.
    """
    tuning = method_dir(twin) / TUNING_FOLDER
    for file_name in PLATT_FILES:
        if not committed_unchanged(tuning / file_name):
            raise ValueError(
                f"{tuning / file_name} is not committed, or has changed since: a calibrated "
                "twin's dev-only record is committed before it is frozen (docs/adr/0011). Write "
                f"it with `{_PROGRAM} export --method {twin}`, then commit it."
            )
    base = base_of(twin)
    check_base_committed(twin, *(method_dir(base) / name for name in platt.BASE_COPIES))
    check_base_history(twin, platt.read_platt_params(tuning)["base_files"])
    platt.check_base_files(tuning, base_files(base))


def check_derived(twin: str, scores: pd.DataFrame, thresholds: Sequence[float]) -> None:
    """Fail unless a twin's scores are its base's frozen ones and its thresholds its record's.

    Args:
        twin: A calibrated twin.
        scores: The probabilities its base's answers files give, as ``scores.csv`` holds them.
        thresholds: The final thresholds its rule chose on them, in ``FOCUS_TAGS`` order.

    Raises:
        ValueError: If ``scores`` is not the base's ``scores.csv``, or the thresholds are not
            those of the twin's ``tuning/params.json``.
    """
    base = base_of(twin)
    if not load_scores(method_dir(base)).equals(scores):
        raise ValueError(
            f"The probabilities that the answers files of {base} give are not its frozen "
            f"{SCORES_FILE}: run `{_PROGRAM} verify --method {base}` (docs/adr/0011)."
        )
    platt.check_thresholds(method_dir(twin) / TUNING_FOLDER, thresholds)


def record_entries(twin: str) -> dict[str, object]:
    """Return what a twin's run record keeps of its base, its rule and its disclosure.

    Args:
        twin: A calibrated twin.

    Returns:
        ``platt.twin_entries`` of its base's files as they are now and of its base's run record.
    """
    base = base_of(twin)
    entries = load_run_record(method_dir(base)).method_entries
    return platt.twin_entries(twin, base_files(base), entries)


def check_record(directory: Path, twin: str) -> None:
    """Fail unless a twin's frozen run record holds every entry that its freeze records now.

    Each entry its freeze writes is compared with the record's ``method_entries``: ``settings``
    (``words.settings_entry``: the model, the problem fields and the settings hash), then each
    key of ``record_entries(twin)``: its base's files as they are now, its base's served models
    and ``dev_in_sample``, its threshold rule, its disclosure ``test_informed`` and
    ``latency_from``. ``test_informed`` keeps both twins out of the command line's eligible
    methods, and ``dev_in_sample`` keeps the GEPA twin out too. An entry that a freeze does not
    write is not compared.

    Args:
        directory: The twin's directory.
        twin: A calibrated twin.

    Raises:
        ValueError: If the record's ``base`` entry differs from the base's files now; else if
            another of those entries differs or is missing, naming each one.
    """
    base = base_of(twin)
    recorded = load_run_record(directory).method_entries
    current = json_safe({"settings": settings_entry(twin), **record_entries(twin)})
    differ = [key for key, value in current.items() if recorded.get(key) != value]
    if "base" in differ:
        raise ValueError(
            f"The run record in {directory} names other files of {base} than those committed "
            "now, and each must have the sha256 and, as its last commit, the commit that the "
            f"record names. If the record was edited, restore it from git. {platt.BASE_CHANGED}"
        )
    if differ:
        raise ValueError(
            f"The run record in {directory} holds other entries than the freeze of {twin} "
            f"records now: {differ} differ. The record was edited, or the code records other "
            "values, after the twin was frozen, and a frozen file never changes (docs/adr/0003, "
            "docs/adr/0011): restore the record from git."
        )


def copy_base_files(twin: str) -> str:
    """Copy its base's ``latency.csv`` and ``per_problem.csv`` into a twin's folder, once.

    Args:
        twin: A calibrated twin, whose directory exists.

    Returns:
        A line per file to print: it was copied, or the twin's is already the base's.

    Raises:
        ValueError: As ``platt.copy_from_base``.
    """
    base_directory, directory = method_dir(base_of(twin)), method_dir(twin)
    copies = [platt.copy_from_base(base_directory, directory, name) for name in platt.BASE_COPIES]
    return "\n".join(copies)


def verify_twin(
    twin: str,
    scores: pd.DataFrame,
    thresholds: Sequence[float],
    raw_dev: pd.DataFrame,
    problems: pd.DataFrame,
) -> str:
    """Run the checks of a twin's verify that its base's answers files do not cover; write nothing.

    Args:
        twin: A calibrated twin, frozen.
        scores: The probabilities its base's answers files give, as ``scores.csv`` holds them.
        thresholds: The final thresholds its rule chose on them, in ``FOCUS_TAGS`` order.
        raw_dev: The dev part's raw problems, with their fold and group (``select_part``).
        problems: Every problem of the dataset, preprocessed.

    Returns:
        The lines to print, one per group of checks, once all passed.

    Raises:
        ValueError: As ``check_base_history``, ``check_record``, ``platt.check_base_files``,
            ``platt.check_copy``, ``check_derived`` and ``write_record`` (which only compares
            the tuning/ record it re-derives, since it exists).
        FileNotFoundError: If the twin's ``tuning/params.json`` does not exist.
    """
    base, directory = base_of(twin), method_dir(twin)
    tuning = directory / TUNING_FOLDER
    check_base_history(twin, platt.read_platt_params(tuning)["base_files"])
    check_record(directory, twin)
    platt.check_base_files(tuning, base_files(base))
    for file_name in platt.BASE_COPIES:
        platt.check_copy(method_dir(base), directory, file_name)
    check_derived(twin, scores, thresholds)
    write_record(twin, raw_dev, problems)
    return (
        f"the run record and tuning/{PARAMS_FILE} name the files of {base} by their sha256 and "
        f"last commit, a commit of HEAD's history, and {' and '.join(platt.BASE_COPIES)} are "
        f"{base}'s\n"
        f"the answers of {base} give its {SCORES_FILE}, and the twin's rule the thresholds of "
        f"tuning/{PARAMS_FILE}\n"
        f"re-derived the tuning/ record of {twin} from the dev scores of {base}: it is equal"
    )
