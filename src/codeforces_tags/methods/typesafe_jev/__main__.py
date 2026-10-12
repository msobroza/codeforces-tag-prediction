"""Command line of the TypeSafe Jev methods (docs/adr/0007).

    python -m codeforces_tags.methods.typesafe_jev render --method M --part P --limit N --out DIR
    python -m codeforces_tags.methods.typesafe_jev run --method M --part P [--max-cost-usd X]
        [--allow-dirty]
    python -m codeforces_tags.methods.typesafe_jev freeze --method M [--allow-dirty]
    python -m codeforces_tags.methods.typesafe_jev verify --method M
    python -m codeforces_tags.methods.typesafe_jev tune --method jev-statement-code-gepa
        [--max-cost-usd X] [--max-iterations N] [--no-mlflow] [--allow-dirty]
    python -m codeforces_tags.methods.typesafe_jev export --method jev-statement-code-gepa
        [--allow-dirty]
    python -m codeforces_tags.methods.typesafe_jev export --method T [--allow-dirty]
    python -m codeforces_tags.methods.typesafe_jev log-search --method jev-statement-code-gepa

M is jev-statement, jev-statement-code or jev-statement-code-gepa (the method whose words a
search chose, docs/adr/0010), or a calibrated twin, jev-statement-code-platt or
jev-statement-code-gepa-platt (docs/adr/0011), and P is dev or test. run and freeze also take
--tracking-uri URI and --artifact-dir DIR (the MLflow store, whose prompt registry holds the
prompt jev-tags, the words of the two frozen methods), whose defaults come from
codeforces_tags.config.
Every command reads the dataset of $CODEFORCES_DATA_DIR (else config.DEFAULT_DATA_DIR) and the
committed split, config.SPLIT_FILE: no option names another file, so no gate can be pointed at
another one. run sends paid requests: it needs the llm-typesafe dependency group and
TYPESAFE_API_KEY, e.g. `uv run --env-file <file holding the key> --group llm-typesafe python -m
codeforces_tags.methods.typesafe_jev run --method M --part P`. freeze needs the group (log_method
pins typesafe-sdk) and no key; render and verify need neither.

render writes the request body of the first N problems of a part to DIR/M/P/<sample_id>.json, and
prints each request's hash, for review. It runs no gate, opens no client and sends nothing.
render and run refuse a calibrated twin before anything else: it sends no request of its own and
reads its base's answers.

run asks Jev the requests of a part that have no line in assets/methods/M/answers_P.jsonl yet,
and appends each answer as it arrives (docs/adr/0003). Before anything is sent, it checks the
dataset, that the code is committed (unless --allow-dirty), that the split is committed and
unchanged, that the self-identifying slice is committed and is the slice of this dataset, that
each answers file of M holds every line of its committed versions, that every line of both is
an answer to its problem's current request, that every completed answer was served by the
pinned model and, for the test part, that the dev answers file of every Jev method of M's study
(settings.STUDIES) is committed, unchanged and complete (which --allow-dirty does not lift), or,
for the dev part, that M's test answers file holds no line. For jev-statement-code-gepa it first
checks that the words its search chose, tuning/params.json, are committed and unchanged. For a
frozen method, whose words are the prompt file templates/jev-tags.v1.yaml, it then registers the
prompt jev-tags in the store's prompt registry once, one version per prompt file, and later only
checks it (tracking.register_jev_prompt); the searched method's words are not a prompt file's
(docs/adr/0010), so its run registers none. It opens the SDK's client, which reads
TYPESAFE_API_KEY, only when requests are left to send, and stops sending once its answers cost
--max-cost-usd (2 USD by default) or after 20 failed answers in a row
(settings.MAX_FAILED_IN_A_ROW: a refused request costs nothing, so no cap stops a run of them).
It exits with 1 while requests are left (a stop, a transient error): run it again to send them,
once the failed lines are understood.

freeze, once both answers files of M are committed, unchanged, complete and answered by the
pinned model, fits cross-fitted thresholds on the dev probabilities and freezes predictions.csv
(dev out of fold, test once), scores.csv (the probabilities), per_problem.csv, run_record.json
and the method folder, each once, then logs the method to MLflow (log_method). A frozen
method's run record made now pins the version of jev-tags that the requests were built from
(prompts:/jev-tags/1), which freeze registers first. A second run rewrites nothing: it re-derives
every file from the answers files, checks the frozen ones against them, then mirrors them to
MLflow, which adds only what its store lacks; a frozen record is kept as it is, even one that
pins no prompt. Every run of a frozen method then links the version of jev-tags to the method's
logged model and run (tracking.link_jev_prompt). The searched method's words are its search's
frozen record, not a version of jev-tags (docs/adr/0010): its record pins no prompt and freeze
links none.

verify, offline and writing nothing (no key, no SDK), rebuilds every request and checks it
against its answer, checks the answers files against their committed versions and the run
record, checks the model that served them, re-derives predictions.csv, scores.csv,
per_problem.csv and the method folder's thresholds from the answers, runs check_frozen_method,
then prints the scores.

tune, export and log-search serve the search over the words of jev-statement-code-gepa
(docs/adr/0010); their handlers live in search_commands. tune checks the dataset, the clean tree
(--allow-dirty), the committed split and self-identifying slice, and that the frozen dev
answers, scores and run record of jev-statement-code, the seed, are committed and unchanged; it
refuses once the search is exported (the method's tuning/params.json exists: its words are
final). It then runs the search, or replays it from its caches under experiments/ and goes on,
logs it live to MLflow (experiment jev-gepa) unless --no-mlflow, and exits with 1 when the
search is interrupted inside an iteration: after a transient error of Jev or the reflection
model, or a run of Jev's failed answers, run it again; after its cost cap was spent inside a
call, run it again with a higher --max-cost-usd (the same cap stops it at the same point). It
needs OPENAI_API_KEY and TYPESAFE_API_KEY only while uncached requests are left. export checks
the same gates but the slice, then that the search stopped at a cap, that its seed reproduces
the frozen scores of jev-statement-code and that no word set it keeps copies 8 words in a row
of a problem's text, and writes the searched method's tuning/ once; a second run only compares.
log-search replays tuning/ into the MLflow store and writes the search run's id to the search
folder's run.json; it needs no key. tune and log-search also take --tracking-uri and
--artifact-dir. freeze and verify of jev-statement-code-gepa also record, then check, the sha256
of tuning/params.json and the macro F1 and per-tag F1 of the held-out fold, which the search
never read.

export --method T, for a calibrated twin T (settings.PLATT_BASES, docs/adr/0011), checks the
dataset, the clean tree (--allow-dirty) and the committed split, and that the answers files,
scores.csv and run_record.json of T's base are committed and unchanged. It then writes T's
dev-only record in T's tuning/ once (platt_commands.export), from the dev rows of the base's
scores.csv only: the three threshold rules cross-fitted on dev, the nested choice among them,
the stability of their fold thresholds, and params.json, which names the base's files by sha256
and commit. A second run checks the base's files against params.json, then only compares. Commit
it before freezing T.

freeze and verify --method T read the committed answers files of T's base, never answers of
its own, and run T's own gates and checks in platt_commands: T's tuning/ and its base's files
committed, unchanged and named by params.json; the Platt + Lipton thresholds; the base's
latency.csv and per_problem.csv copied byte for byte; nothing written in the base's folder.
Like the searched method's, T's record pins no prompt and freeze links none to T: T sends no
request of its own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from codeforces_tags.config import (
    FOCUS_TAGS,
    MLFLOW_METRIC_DATASETS,
    SELF_IDENTIFYING_FILE,
    SPLIT_FILE,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.contamination import load_slice, self_identifying_solutions
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    load_scores,
    stack_parts,
)
from codeforces_tags.llm.answers import per_problem_frame
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.llm.store import (
    answers_file,
    check_append_only,
    check_complete,
    check_same_requests,
    read_answers,
    requests_to_send,
)
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    load_method,
    method_dir,
    model_dir,
    save_method,
)
from codeforces_tags.methods.typesafe_jev import (
    client,
    platt_commands,
    questions,
    runner,
    search_commands,
)
from codeforces_tags.methods.typesafe_jev.answers import (
    answers_summary,
    check_served_model,
    probabilities,
)
from codeforces_tags.methods.typesafe_jev.method import JevProbabilities, JevTagger
from codeforces_tags.methods.typesafe_jev.search_record import search_entries
from codeforces_tags.methods.typesafe_jev.settings import (
    FROZEN_NAMES,
    GEPA_NAME,
    GEPA_SEED_NAME,
    MAX_COST_USD,
    MODEL,
    NAMES,
    PARAMS_FILE,
    PLATT_BASES,
    STUDIES,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.tracking import link_jev_prompt, register_jev_prompt
from codeforces_tags.methods.typesafe_jev.words import (
    base_of,
    method_settings,
    settings_entry,
    words_of,
)
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.provenance import (
    RunRecord,
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    committed_versions,
    file_sha256,
    json_safe,
    last_commit,
    make_run_record,
    sha256_json,
)
from codeforces_tags.splitting import PARTS, load_split, select_part
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.methods import log_method, standard_metrics

# The hex digits of a request's hash that render prints.
_HASH_DIGITS = 12


class _Data(NamedTuple):
    """The dataset as the commands read it: the committed split, each part, every problem."""

    split: pd.DataFrame  # load_split(SPLIT_FILE)
    raw_parts: dict[str, pd.DataFrame]  # each part's raw problems, with their fold and group
    problems: pd.DataFrame  # every problem, preprocessed (raw_to_problems)


class _Derived(NamedTuple):
    """What a method's answers files give once its thresholds are cross-fitted."""

    wrapper: CrossFittedTagThresholds  # fitted on the dev probabilities, the one saved
    predictions: pd.DataFrame  # 0/1 by (part, sample_id): dev out of fold, then test
    scores: pd.DataFrame  # the probabilities, with the index and columns of the predictions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command`` and ``method``; for render also ``part``, ``limit``
        and ``out``; for run ``part``, ``max_cost_usd``, ``allow_dirty``, ``tracking_uri`` and
        ``artifact_dir``; for freeze ``allow_dirty``, ``tracking_uri`` and ``artifact_dir``.
        verify takes nothing more; for tune also ``max_cost_usd``, ``max_iterations``,
        ``no_mlflow``, ``allow_dirty``, ``tracking_uri`` and ``artifact_dir``; for export
        ``allow_dirty``; for log-search ``tracking_uri`` and ``artifact_dir``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.methods.typesafe_jev",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="write a few request bodies, for review")
    render.add_argument("--limit", required=True, type=_limit, help="how many problems")
    render.add_argument("--out", required=True, type=Path, help="folder of the request bodies")
    run = commands.add_parser("run", help="ask Jev the requests of a part that have no answer")
    run.add_argument(
        "--max-cost-usd",
        type=_cost_cap,
        default=MAX_COST_USD,
        help="stop sending once this call's answers cost this much (USD)",
    )
    freeze = commands.add_parser("freeze", help="freeze the answered method, then log it")
    verify = commands.add_parser(
        "verify", help="re-derive every frozen file offline from the answers"
    )
    for command in (render, run):
        command.add_argument("--part", required=True, choices=PARTS, help="dev or test")
    for command in (run, freeze):
        command.add_argument(
            "--allow-dirty", action="store_true", help="accept uncommitted code (recorded)"
        )
        add_store_arguments(command)
    for command in (render, run, freeze, verify):
        command.add_argument("--method", required=True, choices=NAMES, help="the Jev method")
    # The search's commands take the searched method only (search_commands).
    search_commands.add_search_commands(commands, _cost_cap)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command of the TypeSafe Jev methods.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status: 0, or 1 when ``run`` leaves requests without an answer (the cost cap,
        a run of failed answers, a transient error) or when ``tune`` is interrupted inside an
        iteration. A refused gate or a fatal API error raises.
    """
    args = parse_args(argv)
    commands = {
        "render": _render,
        "run": _run,
        "freeze": _freeze,
        "verify": _verify,
        "tune": _tune,
        "export": _export,
        "log-search": search_commands.log_search,
    }
    return commands[args.command](args)


def _limit(text: str) -> int:
    """Read --limit for argparse: a number of problems, at least 1."""
    number = int(text)
    if number >= 1:
        return number
    raise argparse.ArgumentTypeError(f"the limit must be at least 1 problem, not {text}")


def _cost_cap(text: str) -> float:
    """Read --max-cost-usd for argparse: a finite amount above 0 USD (a NaN cap stops nothing)."""
    amount = float(text)
    if math.isfinite(amount) and amount > 0:
        return amount
    raise argparse.ArgumentTypeError(f"the cost cap must be finite and above 0 USD, not {text}")


def _read_data() -> _Data:
    """Read the dataset of $CODEFORCES_DATA_DIR, and split it as the committed split does."""
    split = load_split(SPLIT_FILE)
    raw = read_samples()
    raw_parts = {part: select_part(raw, split, part) for part in PARTS}
    return _Data(split=split, raw_parts=raw_parts, problems=raw_to_problems(raw))


def _render(args: argparse.Namespace) -> int:
    """Write the request bodies of the first problems of a part, for review; send nothing."""
    _refuse_twin(args.method)
    first = _read_data().raw_parts[args.part].iloc[: args.limit]
    folder = args.out / args.method / args.part
    folder.mkdir(parents=True, exist_ok=True)
    for request in questions.requests(first, args.method):
        path = folder / f"{request.sample_id}.json"
        body = json.dumps(request.body(), indent=2, ensure_ascii=False)
        path.write_text(f"{body}\n", encoding="utf-8")
        print(f"{path}: request {request.sha256()[:_HASH_DIGITS]}")
    return 0


def _store(args: argparse.Namespace) -> dict[str, Any]:
    """Return the MLflow store the command was given, as ``use_store``'s keywords."""
    return {"tracking_uri": args.tracking_uri, "artifact_dir": args.artifact_dir}


def _requests(name: str, data: _Data) -> dict[str, list[questions.JevRequest]]:
    """Build the request of every problem of each part, for one method, with its words."""
    words = words_of(name)
    return {part: questions.requests(data.raw_parts[part], name, words) for part in PARTS}


def _answers(name: str) -> dict[str, list[LlmAnswer]]:
    """Read the answers file of each part of a method; a file not written yet holds none."""
    return {part: read_answers(answers_file(name, part)) for part in PARTS}


def _refuse_twin(name: str) -> None:
    """Refuse a calibrated twin: it has no request of its own to render or send (docs/adr/0011)."""
    if name in PLATT_BASES:
        raise ValueError(
            f"{name}: a calibrated twin sends no request: it reads {PLATT_BASES[name]}'s answers "
            "(docs/adr/0011-platt-lipton-threshold-rule.md)."
        )


def _run(args: argparse.Namespace) -> int:
    """Check the gates of docs/adr/0003, register the prompt, then ask Jev what has no answer."""
    name = args.method
    _refuse_twin(name)
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    data = _read_data()
    _check_slice(data)
    _check_words(name)
    requests = _requests(name, data)
    _check_append_only(name)
    answers = _answers(name)
    for part in PARTS:
        check_same_requests(requests[part], answers[part])
    check_served_model([*answers["dev"], *answers["test"]])
    _check_part_order(name, args.part, data, answers)
    if name in FROZEN_NAMES:  # the searched method's words are no prompt file's (docs/adr/0010)
        register_jev_prompt(**_store(args))
    return _send(name, args.part, requests[args.part], answers[args.part], args.max_cost_usd)


def _check_split() -> None:
    """Refuse to go on until the split is committed and unchanged: the parts are its parts."""
    if not committed_unchanged(SPLIT_FILE):
        raise ValueError(
            f"{SPLIT_FILE} is not committed, or has changed since: the parts, and the gate that "
            "asks dev before test, are those of the committed split (docs/adr/0003)."
        )


def _check_slice(data: _Data) -> None:
    """Refuse a request until the self-identifying slice is committed and is the data's slice."""
    if not committed_unchanged(SELF_IDENTIFYING_FILE):
        raise ValueError(
            f"{SELF_IDENTIFYING_FILE} is not committed, or has changed since: freeze it with "
            "`python -m codeforces_tags.evaluation slices`, then commit it, before the first "
            "request, so that its rule is fixed before any answer is read."
        )
    found = self_identifying_solutions(data.problems, data.split)
    if not load_slice(SELF_IDENTIFYING_FILE).equals(found):
        raise ValueError(
            f"{SELF_IDENTIFYING_FILE} is not the slice that self_identifying_solutions finds "
            "in this dataset and split: the data, the split or the rule changed since it was "
            "frozen."
        )


def _check_words(name: str) -> None:
    """Refuse a request of the searched method until the words its search chose are committed."""
    if name != GEPA_NAME:
        return
    path = method_dir(name) / TUNING_FOLDER / PARAMS_FILE
    if not committed_unchanged(path):
        raise ValueError(
            f"{path} is not committed, or has changed since: {name} sends the words its search "
            "chose, fixed before its first request (docs/adr/0003, docs/adr/0010). Write them "
            f"with `python -m codeforces_tags.methods.typesafe_jev export --method {name}`, then "
            "commit them. --allow-dirty, which concerns the code, does not lift this check."
        )


def _study_of(name: str) -> tuple[str, ...]:
    """Return the methods of a method's study (``STUDIES``): the methods compared with it."""
    return next(study for study in STUDIES if name in study)


def _check_append_only(name: str) -> None:
    """Refuse an answers file of the method that lost or changed a line of a committed version."""
    for path in (answers_file(name, part) for part in PARTS):
        check_append_only(path, committed_versions(path))


def _check_answered(
    name: str, part: str, requests: Sequence[questions.JevRequest], answers: Sequence[LlmAnswer]
) -> None:
    """Refuse a part until each of its requests has an answer under its current hash.

    The toolkit's message names its own ``run``, which refuses a score method: it names this
    package's instead.
    """
    try:
        check_complete(requests, answers)
    except ValueError as error:
        command = (
            f"python -m codeforces_tags.methods.typesafe_jev run --method {name} --part {part}"
        )
        raise ValueError(
            str(error).replace("python -m codeforces_tags.llm run", command)
        ) from error


def _check_part_order(
    name: str, part: str, data: _Data, answers: Mapping[str, Sequence[LlmAnswer]]
) -> None:
    """Refuse dev once test has answers, and test until its study's dev parts are answered."""
    if part == "dev":
        if answers["test"]:
            raise ValueError(
                f"{answers_file(name, 'test')} has answers: the dev part of {name} is closed, "
                "since every dev problem is answered before the first test request "
                "(docs/adr/0003)."
            )
        return
    study = _study_of(name)
    for other in study:
        path = answers_file(other, "dev")
        if not committed_unchanged(path):
            raise ValueError(
                f"{path} is not committed, or has changed since: no Jev method asks its test "
                f"part until the dev answers of every method of its study ({', '.join(study)}) "
                "are committed and complete, so that they are compared on dev alone "
                "(docs/adr/0003). --allow-dirty, which concerns the code, does not lift this "
                "check."
            )
        dev_requests = questions.requests(data.raw_parts["dev"], other)
        _check_answered(other, "dev", dev_requests, read_answers(path))


def _send(
    name: str,
    part: str,
    requests: Sequence[questions.JevRequest],
    answers: Sequence[LlmAnswer],
    max_cost_usd: float,
) -> int:
    """Ask Jev the part's requests that have no answer; open its client only when some are left."""
    path = answers_file(name, part)
    if requests_to_send(requests, answers):
        return asyncio.run(_ask_pending(name, requests, path, max_cost_usd))
    print(f"{path}: every request has an answer: nothing to send.")
    return 0


async def _ask_pending(
    name: str, requests: Sequence[questions.JevRequest], path: Path, max_cost_usd: float
) -> int:
    """Send the requests that have no line in ``path`` through one Jev session of this loop."""
    async with client.jev_session(name) as ask:
        return await runner.send(requests, ask, path, max_cost_usd=max_cost_usd)


def _tune(args: argparse.Namespace) -> int:
    """Check the gates of a search, then run or replay it from the seed's committed answers."""
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    data = _read_data()
    _check_slice(data)
    _check_seed_frozen()
    seed_answers = read_answers(answers_file(GEPA_SEED_NAME, "dev"))
    return search_commands.tune(args, data.raw_parts["dev"], seed_answers)


def _export(args: argparse.Namespace) -> int:
    """Check the gates of an export, then freeze the method's dev-only record in tuning/ once.

    For the searched method, the search's record: the texts no kept word set may copy are those
    of every problem, dev and test. For a calibrated twin, the record of its threshold rule on
    its base's dev scores (docs/adr/0011).
    """
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    data = _read_data()
    if args.method in PLATT_BASES:
        return platt_commands.export(args.method, data.raw_parts["dev"], data.problems)
    _check_seed_frozen()
    seed_scores = load_scores(method_dir(GEPA_SEED_NAME)).loc["dev"]
    return search_commands.export(data.raw_parts["dev"], data.problems, seed_scores)


def _check_seed_frozen() -> None:
    """Refuse a search or its export until the seed's frozen files are committed and unchanged."""
    directory = method_dir(GEPA_SEED_NAME)
    for path in (
        answers_file(GEPA_SEED_NAME, "dev"),
        directory / SCORES_FILE,
        directory / RUN_RECORD_FILE,
    ):
        if not committed_unchanged(path):
            raise ValueError(
                f"{path} is not committed, or has changed since: a search starts from the frozen "
                f"words and dev answers of {GEPA_SEED_NAME}, and export checks its seed against "
                "its frozen scores (docs/adr/0010)."
            )


def _search_entries(name: str, data: _Data, predictions: pd.DataFrame) -> dict[str, object] | None:
    """Return the run record's search entries of the searched method; None for another."""
    if name != GEPA_NAME:
        return None
    path = method_dir(name) / TUNING_FOLDER / PARAMS_FILE
    dev = data.raw_parts["dev"]
    return search_entries(
        json.loads(path.read_text(encoding="utf-8")),
        file_sha256(path),
        predictions.loc["dev"],
        focus_labels(data.problems).loc[dev.index],
        dev["fold"],
    )


def _check_record_search(directory: Path, search: Mapping[str, object] | None) -> None:
    """Fail unless the frozen run record holds the search entries that the files give now."""
    recorded = load_run_record(directory).method_entries.get("search")
    current = None if search is None else json.loads(json.dumps(json_safe(search)))
    if recorded != current:
        raise ValueError(
            f"The run record in {directory} holds another search than tuning/{PARAMS_FILE} and "
            "the held-out fold's predictions give: params.json changed after the method was "
            "frozen, and an exported search is final (docs/adr/0010). Restore it from git."
        )


def recipe_sha256(name: str) -> str:
    """Hash what defines a Jev method: its name, its tags and the settings its requests hold.

    The two frozen methods share ``SETTINGS``, and ``log_method`` finds a method's MLflow run by
    its recipe alone: without the name, the second method to be frozen would be logged as the
    first. The searched method's settings hold the words its search chose. The answers, which
    are the method's outcome, are not hashed.

    Args:
        name: The method, one of ``NAMES``.

    Returns:
        ``sha256_json`` of the name, ``FOCUS_TAGS`` and ``method_settings`` of the method's
        words: the ``recipe_sha256`` of the method's run record.
    """
    settings = method_settings(name, words_of(name))
    return sha256_json({"method": name, "tags": list(FOCUS_TAGS), "settings": settings})


def _freeze(args: argparse.Namespace) -> int:
    """Freeze a method once from its committed answers files, then mirror it to MLflow.

    A calibrated twin reads its base's answers files, once its tuning/ record is committed, and
    copies its base's latency.csv and per_problem.csv (docs/adr/0011).

    A frozen method's new run record pins the prompt, registered first; a frozen one is kept as
    it is (the records frozen before the prompt registry pin none). After the log, every run of
    a frozen method links the prompt to the method's logged model and run. The searched method
    and the calibrated twins pin and link none (``_pinned_prompts``, ``_linked_prompt``).
    """
    name, base = args.method, base_of(args.method)
    twin = name in PLATT_BASES
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    _check_words(base)
    if twin:
        platt_commands.check_freeze_inputs(name)
    data = _read_data()
    answers = _committed_answers(base, _requests(base, data))
    _check_append_only(base)
    check_served_model([*answers["dev"], *answers["test"]])
    directory = method_dir(name)
    frozen = (directory / RUN_RECORD_FILE).exists()
    derived = _derive(data, answers, platt_commands.threshold_class(name))
    if twin:
        platt_commands.check_derived(name, derived.scores, derived.wrapper.thresholds_)
    search = _search_entries(name, data, derived.predictions)
    if frozen:
        if twin:
            platt_commands.check_record(directory, name)
        else:
            _check_record_answers(directory, name, answers)
        _check_record_search(directory, search)
    prompts = () if frozen else _pinned_prompts(name, args)
    record = _run_record(name, args.allow_dirty, answers, prompts, search)
    print(
        freeze_predictions(
            derived.predictions,
            directory,
            record,
            scores=derived.scores,
            score_atol=METHODS[name].score_atol,
        )
    )
    print(platt_commands.copy_base_files(name) if twin else _freeze_per_problem(directory, answers))
    print(_freeze_folder(name, derived.wrapper))
    model_id = log_method(
        model_dir(name),
        load_run_record(directory),  # the frozen record: a second run makes another created_at
        _metrics(data, derived.predictions),
        split=data.split,
        registry=METHODS,
        **_store(args),
    )
    print(f"logged {name} to MLflow: model {model_id}{_linked_prompt(name, model_id, args)}")
    return 0


def _pinned_prompts(name: str, args: argparse.Namespace) -> tuple[str, ...]:
    """Register ``jev-tags``, then return what a frozen method's new run record pins.

    The searched method's words are its search's frozen record, not a version of the prompt
    file (docs/adr/0010): its record pins no prompt, and nothing is registered for it.
    """
    if name not in FROZEN_NAMES:
        return ()
    return (register_jev_prompt(**_store(args)).uri,)


def _linked_prompt(name: str, model_id: str, args: argparse.Namespace) -> str:
    """Link ``jev-tags`` to a frozen method's logged model and run; return what to print of it.

    The searched method reads no version of ``jev-tags`` (docs/adr/0010): nothing is linked
    to it, and nothing is printed.
    """
    if name not in FROZEN_NAMES:
        return ""
    return f", prompt {link_jev_prompt(model_id, **_store(args)).uri}"


def _committed_answers(
    name: str, requests: Mapping[str, Sequence[questions.JevRequest]]
) -> dict[str, list[LlmAnswer]]:
    """Read both answers files of a method, each committed, unchanged and complete."""
    answers = {}
    for part in PARTS:
        answers[part] = _read_committed(answers_file(name, part))
        _check_answered(name, part, requests[part], answers[part])
    return answers


def _read_committed(path: Path) -> list[LlmAnswer]:
    """Read an answers file that is committed and unchanged since its last commit."""
    if not committed_unchanged(path):
        raise ValueError(
            f"{path} is not committed, or has changed since: freeze reads committed answers "
            "only, whose commit the run record names (docs/adr/0003). Commit it, or restore it "
            "from git."
        )
    return read_answers(path)


def _answers_entries(
    name: str, answers: Mapping[str, Sequence[LlmAnswer]]
) -> dict[str, dict[str, object]]:
    """Return what the run record keeps of each answers file: its sha256 and its summary."""
    return {
        "answers_sha256": {part: file_sha256(answers_file(name, part)) for part in PARTS},
        "answers": {part: answers_summary(answers[part]) for part in PARTS},
    }


def _check_record_answers(
    directory: Path, name: str, answers: Mapping[str, Sequence[LlmAnswer]]
) -> None:
    """Fail unless the frozen run record holds the hashes and summaries of the answers files."""
    recorded = load_run_record(directory).method_entries
    differ = [
        key for key, value in _answers_entries(name, answers).items() if recorded.get(key) != value
    ]
    if differ:
        raise ValueError(
            f"The run record in {directory} holds other {' and '.join(differ)} than the "
            "answers files give: a line changed after the method was frozen, and a line is "
            "never rewritten (docs/adr/0003). Restore the answers files from git."
        )


def _run_record(
    name: str,
    allow_dirty: bool,
    answers: Mapping[str, Sequence[LlmAnswer]],
    prompts: Sequence[str],
    search: Mapping[str, object] | None = None,
) -> RunRecord:
    """Make the run record: the recipe, the settings, what was read and the pinned prompts.

    That is the method's answers files, or, for a calibrated twin, its base's files, its
    threshold rule and its disclosure (``platt_commands.record_entries``, docs/adr/0011). The
    searched method's record also holds ``search``, what ``_search_entries`` gives.
    """
    entries: dict[str, object] = {"settings": settings_entry(name)}
    if name in PLATT_BASES:
        entries |= platt_commands.record_entries(name)
    else:
        entries |= _answers_record(name, answers)
    if search is not None:
        entries["search"] = dict(search)
    return make_run_record(
        name,
        recipe_sha256(name),
        allow_dirty=allow_dirty,
        requirements=METHODS[name].requirements,
        prompts=prompts,
        method_entries=entries,
        split_path=SPLIT_FILE,
    )


def _answers_record(name: str, answers: Mapping[str, Sequence[LlmAnswer]]) -> dict[str, object]:
    """Return what the run record keeps of a method's own answers files."""
    files = _answers_entries(name, answers)
    return {
        "answers_sha256": files["answers_sha256"],
        # The dev answers files were committed before the first test request (docs/adr/0003).
        "answers_commit": {part: last_commit(answers_file(name, part)) for part in PARTS},
        "answers": files["answers"],
        "served_models": sorted(
            {
                answer.served_model
                for part in PARTS
                for answer in answers[part]
                if answer.served_model
            }
        ),
        # The searched method's words were chosen on dev folds 1 to 4 (docs/adr/0010).
        "dev_in_sample": name == GEPA_NAME,
    }


def _derive(
    data: _Data,
    answers: Mapping[str, Sequence[LlmAnswer]],
    rule: type[CrossFittedTagThresholds],
) -> _Derived:
    """Cross-fit the thresholds on the dev probabilities; predict dev out of fold, then test.

    ``rule`` chooses the thresholds: ``platt_commands.threshold_class`` of the method.
    """
    dev, test = data.raw_parts["dev"], data.raw_parts["test"]
    dev_scores = probabilities(answers["dev"], dev.index)
    test_scores = probabilities(answers["test"], test.index)
    labels = focus_labels(data.problems).loc[dev.index]
    wrapper = rule(JevProbabilities()).fit(dev_scores, labels, groups=dev["fold"])
    columns = list(FOCUS_TAGS)
    predictions = stack_parts(
        dev=pd.DataFrame(wrapper.oof_predictions_, index=dev.index, columns=columns),
        test=pd.DataFrame(wrapper.predict(test_scores), index=test.index, columns=columns),
    )
    return _Derived(wrapper, predictions, stack_parts(dev=dev_scores, test=test_scores))


def _per_problem_text(answers: Mapping[str, Sequence[LlmAnswer]]) -> str:
    """Return the text of per_problem.csv: each problem's recorded call, dev rows then test rows."""
    rows = pd.concat([per_problem_frame(answers[part], part) for part in PARTS], ignore_index=True)
    return rows.to_csv(index=False)


def _freeze_per_problem(directory: Path, answers: Mapping[str, Sequence[LlmAnswer]]) -> str:
    """Write per_problem.csv once, staged then renamed; later, only compare it with the answers."""
    path = directory / PER_PROBLEM_FILE
    text = _per_problem_text(answers)
    if not path.exists():
        partial = directory / f".{PER_PROBLEM_FILE}.partial"
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
        count = sum(len(lines) for lines in answers.values())
        return f"saved each problem's call to {path} ({count:,} problems)"
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(
            f"{path} is not the calls of the answers files, of which it is a pure function: "
            "restore it from git."
        )
    return f"{path} holds the calls of the answers files"


def _check_folder(name: str, folder: Path, wrapper: CrossFittedTagThresholds) -> None:
    """Fail unless a method folder loads with the method's words and the wrapper's thresholds.

    The words are those its answers were asked with (``words_of``): a folder saved again with
    other words still loads, since its ``jev.json`` hashes its own ``words.json``. It never
    predicts.
    """
    loaded = load_method(folder, registry=METHODS)
    if loaded.words != words_of(name):
        raise ValueError(
            f"The method folder {folder} holds other words than those {name} sends (for the "
            f"searched method, those of its {TUNING_FOLDER}/{PARAMS_FILE}), which its answers "
            "were asked with and its thresholds chosen on: restore it from git."
        )
    stored = loaded.thresholds.thresholds_
    if not np.array_equal(stored, wrapper.thresholds_):
        raise ValueError(
            f"The method folder {folder} holds other thresholds ({stored.tolist()}) than those "
            f"the answers files give ({wrapper.thresholds_.tolist()}): restore it from git."
        )


def _freeze_folder(name: str, wrapper: CrossFittedTagThresholds) -> str:
    """Save the method folder once, moved into place once it loads back; later, only check it.

    It is saved in a temporary folder of the method's directory: a folder that fails the check
    is never left behind, and the next run saves it again.
    """
    folder = model_dir(name)
    if (folder / METHOD_FILE).exists():
        _check_folder(name, folder, wrapper)
        return f"the method folder {folder} holds the re-derived thresholds"
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=method_dir(name)) as staging:
        staged = Path(staging) / MODEL_FOLDER
        save_method(JevTagger(name, wrapper, words_of(name)), staged)
        _check_folder(name, staged, wrapper)
        staged.replace(folder)
    return f"saved the method folder {folder}"


def _metrics(data: _Data, predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Score each part's predictions on the focus labels, under its MLflow dataset's name."""
    labels = focus_labels(data.problems)
    rows = {part: predictions.loc[part] for part in MLFLOW_METRIC_DATASETS}
    return {
        dataset: standard_metrics(labels.loc[rows[part].index], rows[part], FOCUS_TAGS)
        for part, dataset in MLFLOW_METRIC_DATASETS.items()
    }


def _verify(args: argparse.Namespace) -> int:
    """Re-derive every frozen file of a method from its answers files, offline; print its scores.

    A calibrated twin's are re-derived from its base's answers files, and its own checks
    (``platt_commands.verify_twin``) take the place of the answers files' records.
    """
    name, base = args.method, base_of(args.method)
    check_dataset()
    data = _read_data()
    directory = method_dir(name)
    check_frozen_method(name, assets_dir=directory.parent, split=data.split, registry=METHODS)
    print(f"the frozen files of {name} are complete and consistent with the split")
    _check_recipe(directory, name)
    print(f"the run record of {name} holds the recipe of the code")
    requests, answers = _requests(base, data), _answers(base)
    for part in PARTS:
        _check_answered(base, part, requests[part], answers[part])
    count = sum(len(part_requests) for part_requests in requests.values())
    print(f"rebuilt {count:,} requests: each has its answer, under its hash")
    check_served_model([*answers["dev"], *answers["test"]])
    print(f"every completed answer was served by {MODEL}")
    _check_append_only(base)
    print("each answers file holds every line of its committed versions, then new ones")
    derived = _derive(data, answers, platt_commands.threshold_class(name))
    if name in PLATT_BASES:
        dev, thresholds = data.raw_parts["dev"], derived.wrapper.thresholds_
        print(platt_commands.verify_twin(name, derived.scores, thresholds, dev, data.problems))
    else:
        _check_record_answers(directory, name, answers)
        _check_answers_commits(directory, name)
        print("the run record holds the hashes, the commits and the summaries of the answers files")
    _check_frozen_files(directory, derived, answers)
    print(
        f"re-derived {len(derived.predictions):,} predictions, their scores and "
        f"{PER_PROBLEM_FILE} from the answers: they equal the frozen ones"
    )
    _check_folder(name, model_dir(name), derived.wrapper)
    print(f"the method folder {model_dir(name)} holds the re-derived thresholds")
    search = _search_entries(name, data, derived.predictions)
    _check_record_search(directory, search)
    if search is not None:
        print(
            "the run record holds the sha256 of tuning/params.json and the held-out fold's scores"
        )
    for dataset, metrics in _metrics(data, derived.predictions).items():
        print(f"{dataset}: macro F1 {metrics['macro_f1']:.4f}, micro F1 {metrics['micro_f1']:.4f}")
    return 0


def _check_recipe(directory: Path, name: str) -> None:
    """Fail unless the frozen run record holds the recipe that the code gives the method."""
    frozen = load_run_record(directory).recipe_sha256
    current = recipe_sha256(name)
    if frozen != current:
        raise ValueError(
            f"{directory} was frozen with another recipe ({frozen}, not {current}): the model, "
            f"the questions or the tag definitions of {name} changed since. A changed word is a "
            "new method under a new name (docs/adr/0003)."
        )


def _check_answers_commits(directory: Path, name: str) -> None:
    """Fail unless the run record names the commit that last changed each answers file."""
    recorded = load_run_record(directory).method_entries.get("answers_commit", {})
    for part in PARTS:
        path = answers_file(name, part)
        commit = last_commit(path)
        if recorded.get(part) != commit:
            raise ValueError(
                f"The run record in {directory} names the commit {recorded.get(part)} for "
                f"{path.name}, whose last commit is {commit}: the answers file was committed "
                "again, or its history rewritten, after the method was frozen (docs/adr/0003)."
            )


def _check_frozen_files(
    directory: Path, derived: _Derived, answers: Mapping[str, Sequence[LlmAnswer]]
) -> None:
    """Fail unless predictions.csv, scores.csv and per_problem.csv are what the answers give."""
    calls = directory / PER_PROBLEM_FILE
    same = {
        PREDICTIONS_FILE: load_predictions(directory).equals(derived.predictions),
        SCORES_FILE: load_scores(directory).equals(derived.scores),
        PER_PROBLEM_FILE: calls.is_file()
        and calls.read_text(encoding="utf-8") == _per_problem_text(answers),
    }
    differ = [file_name for file_name, equal in same.items() if not equal]
    if differ:
        raise ValueError(
            f"The frozen files {differ} in {directory} are not what the answers files give, of "
            "which they are a pure function: restore them from git."
        )


if __name__ == "__main__":
    sys.exit(main())
