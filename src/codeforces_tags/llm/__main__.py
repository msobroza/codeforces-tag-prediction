"""Command line of the LLM toolkit: the commands of every registered prompted method.

    python -m codeforces_tags.llm render --method NAME --part PART --limit N --out DIR
    python -m codeforces_tags.llm run --method NAME --part PART [--max-cost-usd X] [--allow-dirty]
    python -m codeforces_tags.llm score --method NAME [--allow-dirty]
    python -m codeforces_tags.llm verify --method NAME

run and score also take --tracking-uri URI and --artifact-dir DIR (the MLflow store), whose
defaults come from codeforces_tags.config. The dataset is the folder of $CODEFORCES_DATA_DIR (else
config.DEFAULT_DATA_DIR) and the split is the frozen one, config.SPLIT_FILE; no option names
another, so no gate can be pointed at another file: a fitted method's load reads its examples
from that folder too (docs/adr/0004), so a command and the method it loads read the same
problems. Run them with the method's dependency groups, e.g.
`uv run --group llm python -m codeforces_tags.llm ...`.

They serve any registered prompted method, through its package (prompted_package): load,
llm_spec, prompt_family, make_method for a fitted method, and optionally verify_extra. A fitted
method's dev requests come from cross_requests, fold by fold; its test requests, and its method
folder, from the method fitted on all of dev.

render writes the exact prompt and request body of the first N problems of a part to
DIR/NAME/PART/, for review. It sends nothing.

run sends the requests of a part that have no answer yet, and appends each answer to
assets/methods/NAME/answers_PART.jsonl as it arrives (docs/adr/0003). Before anything is sent, it
checks the dataset, that the code is committed (unless --allow-dirty), that the split is
committed and unchanged, that the self-identifying slice is committed and is the slice of this
dataset, that each answers file holds every line of its committed versions (it is only appended
to), that every answer of both answers files is to its problem's current request and, for the
test part, that the dev answers file is committed, unchanged and complete (which --allow-dirty
does not lift), or, for the dev part, that no dev request is left once the test part has
answers; then it registers the method's prompts. It needs the provider's API key only when
requests are left to send, and exits with 1 while some are left (at the cost cap, after a
transient error): run it again to send them.

score, once the split is committed and unchanged and both answers files are committed, unchanged,
complete and only appended to, encodes the answers and freezes predictions.csv, per_problem.csv,
run_record.json and the method folder, each once, with the prompt version pinned; the method
folder is moved into place only once it builds the answered requests. It then logs the method to
MLflow (log_method). A second run rewrites nothing: it checks the frozen files, the answers
files' hashes in the run record among them, then mirrors them to MLflow, which adds only what
its store lacks.

verify, offline and writing nothing, re-renders every request and checks it against its answer,
re-parses every answer against the parse it stores, checks the answers files' hashes and summary
in the run record and that each answers file was only appended to, re-derives predictions.csv and
per_problem.csv from the answers, checks that the method folder holds the package's prompt and
builds the answered requests, runs the package's verify_extra, then prints the scores.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import importlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import pandas as pd
from mlflow.entities.model_registry import PromptVersion

from codeforces_tags.config import (
    ALL_TAGS,
    LLM_CONCURRENCY,
    SELF_IDENTIFYING_FILE,
    SPLIT_FILE,
    TEXT_PARSER_VERSION,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.contamination import load_slice, self_identifying_solutions
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    stack_parts,
)
from codeforces_tags.llm import client
from codeforces_tags.llm.answers import (
    ParsedTags,
    answer_summary,
    encode_answers,
    parse_answer,
    per_problem_frame,
    problem_list,
)
from codeforces_tags.llm.client import LlmAnswer
from codeforces_tags.llm.out_of_fold import cross_requests
from codeforces_tags.llm.prompted import PromptedLlm
from codeforces_tags.llm.prompts import LlmMethodSpec, LlmRequest, PromptSnapshot, prompt_snapshot
from codeforces_tags.llm.runner import answer_each
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
    MethodSpec,
    load_method,
    method_dir,
    model_dir,
    save_method,
)
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.provenance import (
    RunRecord,
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    committed_versions,
    file_sha256,
    last_commit,
    make_run_record,
    sha256_json,
)
from codeforces_tags.splitting import PARTS, load_split, select_part
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.methods import log_method, part_metrics
from codeforces_tags.tracking.prompts import register_prompts, set_prompt_aliases

# What every prompted method's package gives (the prompted-method package contract).
PACKAGE_FUNCTIONS = ("load", "llm_spec", "prompt_family")
# The method kinds the toolkit serves: nothing learned from dev, or learned out of fold.
PROMPTED_KINDS = ("zero_shot", "fitted")
# How many hex digits of a hash a message shows.
_SHORT_HASH = 12


class _Prompted(NamedTuple):
    """A registered prompted method, as its package describes it."""

    name: str
    kind: str
    package: ModuleType
    spec: LlmMethodSpec
    prompt: PromptSnapshot  # read from the package: prompt_snapshot(spec)


class _Parts(NamedTuple):
    """The dataset as the commands use it."""

    split: pd.DataFrame
    raw: dict[str, pd.DataFrame]  # each part's raw problems, with their fold and group
    problems: pd.DataFrame  # every problem, preprocessed (raw_to_problems)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command`` and ``method``; for render also ``part``, ``limit``
        and ``out``; for run ``part`` and ``max_cost_usd``; for run and score ``allow_dirty``,
        ``tracking_uri`` and ``artifact_dir``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.llm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="write the prompts of a few problems, for review")
    render.add_argument("--limit", required=True, type=_count, help="how many problems")
    render.add_argument("--out", required=True, type=Path, help="folder to write the prompts to")
    run = commands.add_parser("run", help="send the requests of a part that have no answer yet")
    run.add_argument(
        "--max-cost-usd", type=_cost, help="stop sending once this run's answers cost this much"
    )
    for command in (render, run):
        command.add_argument("--part", required=True, choices=PARTS, help="dev or test")
    score = commands.add_parser("score", help="freeze the answered method once, log it to MLflow")
    for command in (run, score):
        command.add_argument(
            "--allow-dirty", action="store_true", help="run from uncommitted code (recorded)"
        )
        add_store_arguments(command)
    commands.add_parser("verify", help="re-derive the frozen files from the answers")
    for command in commands.choices.values():
        command.add_argument("--method", required=True, help="the prompted method's registry key")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command of the LLM toolkit.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status: 0, or 1 when ``run`` leaves requests without an answer. A refused
        dataset, slice, tree, answers file or frozen file raises.
    """
    args = parse_args(argv)
    commands = {"render": _render, "run": _run, "score": _score, "verify": _verify}
    return commands[args.command](args)


def prompted_package(name: str, *, registry: Mapping[str, MethodSpec] = METHODS) -> ModuleType:
    """Import the package of a registered prompted method, checking that it serves the toolkit.

    Args:
        name: The method's registry key.
        registry: The registered methods.

    Returns:
        The module that ``registry[name].module`` names.

    Raises:
        ValueError: If the method is not registered, is not a ``calls_api`` method of kind
            ``zero_shot`` or ``fitted``, or its module lacks a function of the contract:
            ``load``, ``llm_spec`` and ``prompt_family``, and ``make_method`` for a fitted one.
    """
    if name not in registry:
        raise ValueError(f"{name!r} is not registered: the registry has {sorted(registry)}.")
    spec = registry[name]
    if spec.kind not in PROMPTED_KINDS or not spec.calls_api:
        raise ValueError(
            f"{name} is not a prompted method: the toolkit serves the methods of kind "
            f"{PROMPTED_KINDS} that call an API, and it is of kind {spec.kind!r}, with "
            f"calls_api={spec.calls_api}."
        )
    package = importlib.import_module(spec.module)
    wanted = [*PACKAGE_FUNCTIONS, *(["make_method"] if spec.kind == "fitted" else [])]
    missing = [function for function in wanted if not callable(getattr(package, function, None))]
    if missing:
        raise ValueError(
            f"{spec.module}, the package of {name}, lacks {missing}: a prompted method's package "
            "gives load, llm_spec and prompt_family, and make_method for a fitted method."
        )
    return package


def recipe_sha256(spec: LlmMethodSpec, prompt: PromptSnapshot) -> str:
    """Hash what defines a prompted method, as the recipe of its run record.

    Not its answers, which are its outcome, nor its prompt's version in the registry, which is
    informational.

    Args:
        spec: The method's spec.
        prompt: Its prompt snapshot, as its package gives it (``prompt_snapshot``).

    Returns:
        ``sha256_json`` of the method, its settings, its prompt's content hash, its schema name,
        fields, constants and extra, its tags and ``config.TEXT_PARSER_VERSION``.
    """
    return sha256_json(
        {
            "method": spec.name,
            "llm_settings": spec.settings.to_dict(),
            "prompt_content_sha256": prompt.content_sha256,
            "schema_name": spec.schema_name,
            "fields": list(spec.fields),
            "constants": dict(spec.constants),
            "extra": dict(spec.extra),
            "tags": list(spec.all_tags),
            "text_parser_version": TEXT_PARSER_VERSION,
        }
    )


def _count(text: str) -> int:
    """Read a count of at least 1, for argparse."""
    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError(f"expected a count of at least 1, not {count}")
    return count


def _cost(text: str) -> float:
    """Read a cost cap, for argparse: a finite amount above 0 USD (NaN would be no cap)."""
    cost = float(text)
    if not (math.isfinite(cost) and cost > 0):
        raise argparse.ArgumentTypeError(f"expected a finite cost above 0 USD, not {text}")
    return cost


def _prompted(name: str) -> _Prompted:
    """Describe a registered prompted method from its package."""
    package = prompted_package(name, registry=METHODS)
    spec = package.llm_spec(name)
    if spec.name != name:
        raise ValueError(
            f"llm_spec({name!r}) of {package.__name__} gives the spec of {spec.name!r}."
        )
    prompt = prompt_snapshot(spec, registry=METHODS)
    return _Prompted(name, METHODS[name].kind, package, spec, prompt)


def _family(prompted: _Prompted) -> list[tuple[LlmMethodSpec, PromptSnapshot]]:
    """Return the methods of the method's prompt with their snapshots, in registration order."""
    specs = prompted.package.prompt_family(prompted.spec.prompt_name)
    if prompted.spec not in specs:
        raise ValueError(
            f"prompt_family({prompted.spec.prompt_name!r}) of {prompted.package.__name__} lacks "
            f"the spec of {prompted.name}."
        )
    return [(spec, prompt_snapshot(spec, registry=METHODS)) for spec in specs]


def _read_parts() -> _Parts:
    """Read the dataset and the frozen split: each part's raw problems, every one preprocessed."""
    raw = read_samples()
    split = load_split(SPLIT_FILE)
    return _Parts(
        split=split,
        raw={part: select_part(raw, split, part) for part in PARTS},
        problems=raw_to_problems(raw),
    )


def _truth(parts: _Parts) -> pd.DataFrame:
    """Return every problem's 0/1 tags, one column per tag of ``ALL_TAGS``."""
    return multi_hot(parts.problems["tags"], ALL_TAGS)


def _derive(prompted: _Prompted, parts: _Parts) -> tuple[dict[str, list[LlmRequest]], PromptedLlm]:
    """Build each part's requests, and the method that is saved.

    A zero-shot method builds them all itself. A fitted method's dev requests come from
    ``cross_requests``, and its test requests from the method fitted on all of dev, which is
    the one saved.
    """
    if prompted.kind == "zero_shot":
        method = PromptedLlm(prompted.spec, prompted.prompt)
        return {part: method.requests(parts.raw[part]) for part in PARTS}, method
    dev = parts.raw["dev"]
    labels = _truth(parts).loc[dev.index, list(prompted.spec.all_tags)]
    make = functools.partial(prompted.package.make_method, prompted.name)
    method = make().fit(dev, labels)
    requests = {
        "dev": cross_requests(make, dev, labels, dev["fold"]),
        "test": method.requests(parts.raw["test"]),
    }
    return requests, method


def _answers(name: str) -> dict[str, list[LlmAnswer]]:
    """Read both answers files of a method."""
    return {part: read_answers(answers_file(name, part)) for part in PARTS}


def _store(args: argparse.Namespace) -> dict[str, Any]:
    """Return the MLflow store the command was given, as ``use_store``'s keywords."""
    return {"tracking_uri": args.tracking_uri, "artifact_dir": args.artifact_dir}


def _render(args: argparse.Namespace) -> int:
    """Write the prompt and the request body of the first problems of a part."""
    prompted = _prompted(args.method)
    requests, _ = _derive(prompted, _read_parts())
    folder = args.out / prompted.name / args.part
    folder.mkdir(parents=True, exist_ok=True)
    for request in requests[args.part][: args.limit]:
        prompt = folder / f"{request.sample_id}.txt"
        prompt.write_text(request.input, encoding="utf-8")
        body = json.dumps(request.body(), indent=2, ensure_ascii=False)
        (folder / f"{request.sample_id}.json").write_text(body + "\n", encoding="utf-8")
        print(f"{prompt}: request {request.sha256()[:_SHORT_HASH]}")
    return 0


def _run(args: argparse.Namespace) -> int:
    """Check the gates of ADR 0003, register the prompts, then send what has no answer yet."""
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    parts = _read_parts()
    _check_slice(parts)
    prompted = _prompted(args.method)
    requests, _ = _derive(prompted, parts)
    _check_append_only(prompted.name)
    answers = _answers(prompted.name)
    for part in PARTS:
        check_same_requests(requests[part], answers[part])
    _check_dev_first(prompted.name, args.part, requests, answers)
    register_prompts(prompted.spec.prompt_name, _family(prompted), **_store(args))
    return _send(prompted, requests[args.part], answers[args.part], args)


def _check_split() -> None:
    """Refuse to send or score until the frozen split is committed and unchanged.

    The dev-first gate is computed on it: a split that parked an unanswered dev problem elsewhere
    would let the test part through.
    """
    if not committed_unchanged(SPLIT_FILE):
        raise ValueError(
            f"{SPLIT_FILE} is not committed, or has changed since: the parts and the dev-first "
            "gate are those of the committed split (docs/adr/0003)."
        )


def _check_append_only(name: str) -> None:
    """Refuse an answers file that lost or changed a line of one of its committed versions."""
    for part in PARTS:
        path = answers_file(name, part)
        check_append_only(path, committed_versions(path))


def _check_dev_first(
    name: str,
    part: str,
    requests: Mapping[str, Sequence[LlmRequest]],
    answers: Mapping[str, Sequence[LlmAnswer]],
) -> None:
    """Refuse a test request until dev is frozen, and a dev request once test has answers."""
    if part == "test":
        _check_dev_frozen(name, requests["dev"], answers["dev"])
        return
    pending = [request.sample_id for request in requests_to_send(requests["dev"], answers["dev"])]
    if answers["test"] and pending:
        raise ValueError(
            f"{answers_file(name, 'test')} has answers, yet the dev problems "
            f"{problem_list(pending)} have none: every dev problem is answered before the first "
            "test request (docs/adr/0003), and an answers file is only appended to. Restore the "
            "dev answers file from git."
        )


def _check_slice(parts: _Parts) -> None:
    """Refuse to send a request until the self-identifying slice is frozen and committed."""
    if not committed_unchanged(SELF_IDENTIFYING_FILE):
        raise ValueError(
            f"{SELF_IDENTIFYING_FILE} is not committed, or has changed since: freeze the slice "
            "with `python -m codeforces_tags.evaluation slices` and commit it before the first "
            "request, so that its rule is fixed before any answer is seen."
        )
    found = self_identifying_solutions(parts.problems, parts.split)
    if not load_slice(SELF_IDENTIFYING_FILE).equals(found):
        raise ValueError(
            f"{SELF_IDENTIFYING_FILE} is not the slice that self_identifying_solutions finds on "
            "this dataset and split: the data, the split or the rule changed since it was frozen."
        )


def _check_dev_frozen(
    name: str, requests: Sequence[LlmRequest], answers: Sequence[LlmAnswer]
) -> None:
    """Refuse the test part until the dev answers are committed, unchanged and complete."""
    path = answers_file(name, "dev")
    if not committed_unchanged(path):
        raise ValueError(
            f"{path} is not committed, or has changed since: the test part is asked only once "
            "every dev problem has its answer, committed (docs/adr/0003). --allow-dirty, which "
            "concerns the code, does not lift this check."
        )
    check_complete(requests, answers)


def _send(
    prompted: _Prompted,
    requests: Sequence[LlmRequest],
    answers: Sequence[LlmAnswer],
    args: argparse.Namespace,
) -> int:
    """Send the requests that have no answer yet; return 1 while some are left without one."""
    path = answers_file(prompted.name, args.part)
    if not requests_to_send(requests, answers):
        print(f"{path}: every request has an answer: nothing to send.")
        return 0
    settings = prompted.spec.settings
    coroutine = answer_each(
        requests,
        client.ask_for(settings),
        concurrency=LLM_CONCURRENCY,
        answers_file=path,
        output_format=settings.output_format,
        all_tags=prompted.spec.all_tags,
        max_cost_usd=args.max_cost_usd,
    )
    asyncio.run(coroutine)
    left = requests_to_send(requests, read_answers(path))
    if left:
        print(
            f"{path}: {len(left)} of {len(requests)} requests have no answer yet: run it again "
            "to send them.",
            file=sys.stderr,
        )
        return 1
    return 0


def _score(args: argparse.Namespace) -> int:
    """Freeze the answered method once, then log it to MLflow."""
    check_dataset()
    check_clean_tree(args.allow_dirty)
    _check_split()
    parts = _read_parts()
    prompted = _prompted(args.method)
    requests, method = _derive(prompted, parts)
    answers = _frozen_answers(prompted.name, requests)
    _check_append_only(prompted.name)
    directory = method_dir(prompted.name)
    if (directory / RUN_RECORD_FILE).exists():
        _check_record_entries(directory, prompted, answers)
    version = _pin_prompt(prompted, method, _store(args))
    predictions = _encode(prompted, parts, answers)
    record = _run_record(args, prompted, version, answers)
    print(freeze_predictions(predictions, directory, record))
    print(_freeze_per_problem(directory, answers))
    print(_freeze_folder(method, prompted, parts, answers))
    model_id = log_method(
        model_dir(prompted.name),
        load_run_record(directory),  # what is frozen: a second run's record is made anew
        _metrics(prompted, parts, predictions),
        split=parts.split,
        registry=METHODS,
        **_store(args),
    )
    print(f"logged {prompted.name} to MLflow: model {model_id}, prompt {version.uri}")
    return 0


def _frozen_answers(
    name: str, requests: Mapping[str, Sequence[LlmRequest]]
) -> dict[str, list[LlmAnswer]]:
    """Read both answers files, refusing one that is not committed, unchanged and complete."""
    answers = {}
    for part in PARTS:
        path = answers_file(name, part)
        if not committed_unchanged(path):
            raise ValueError(
                f"{path} is not committed, or has changed since: score freezes committed "
                "answers only (docs/adr/0003)."
            )
        answers[part] = read_answers(path)
        check_complete(requests[part], answers[part])
    return answers


def _pin_prompt(
    prompted: _Prompted, method: PromptedLlm, store: Mapping[str, Any]
) -> PromptVersion:
    """Register the prompts, set the method's alias, and pin its version in its prompt."""
    versions = register_prompts(prompted.spec.prompt_name, _family(prompted), **store)
    version = versions[prompted.name]
    set_prompt_aliases(
        prompted.spec.prompt_name,
        {prompted.name: version},
        {prompted.name: prompted.spec},
        **store,
    )
    method.prompt = dataclasses.replace(method.prompt, version=version.version, uri=version.uri)
    return version


def _encode(
    prompted: _Prompted, parts: _Parts, answers: Mapping[str, Sequence[LlmAnswer]]
) -> pd.DataFrame:
    """Encode both parts' answers as the method's 0/1 predictions, indexed by part and problem."""
    output_format = prompted.spec.settings.output_format
    encoded = {
        part: encode_answers(
            answers[part], parts.raw[part].index, output_format, prompted.spec.all_tags
        )
        for part in PARTS
    }
    return stack_parts(**encoded)


def _run_record(
    args: argparse.Namespace,
    prompted: _Prompted,
    version: PromptVersion,
    answers: Mapping[str, Sequence[LlmAnswer]],
) -> RunRecord:
    """Make the run record: the recipe, the pinned prompt version and the answers' summary."""
    answered = _answers_entries(prompted, answers)
    entries = {
        "prompt": {
            "name": prompted.prompt.name,
            "version": version.version,
            "uri": version.uri,
            "content_sha256": prompted.prompt.content_sha256,
        },
        "llm_settings": prompted.spec.settings.to_dict(),
        "fields": list(prompted.spec.fields),
        "constants": dict(prompted.spec.constants),
        "extra": dict(prompted.spec.extra),
        "text_parser_version": TEXT_PARSER_VERSION,
        "answers_sha256": answered["answers_sha256"],
        # The commit of each answers file: the dev one was committed before any test request.
        "answers_commit": {part: last_commit(answers_file(prompted.name, part)) for part in PARTS},
        "answers": answered["answers"],
        "dev_in_sample": False,
    }
    return make_run_record(
        prompted.name,
        recipe_sha256(prompted.spec, prompted.prompt),
        allow_dirty=args.allow_dirty,
        requirements=METHODS[prompted.name].requirements,
        prompts=(version.uri,),
        method_entries=entries,
        split_path=SPLIT_FILE,
    )


def _answers_entries(
    prompted: _Prompted, answers: Mapping[str, Sequence[LlmAnswer]]
) -> dict[str, dict[str, Any]]:
    """Return what the run record keeps of the answers files: their hashes and summaries."""
    output_format = prompted.spec.settings.output_format
    return {
        "answers_sha256": {part: file_sha256(answers_file(prompted.name, part)) for part in PARTS},
        "answers": {
            part: answer_summary(answers[part], output_format, prompted.spec.all_tags)
            for part in PARTS
        },
    }


def _check_record_entries(
    directory: Path, prompted: _Prompted, answers: Mapping[str, Sequence[LlmAnswer]]
) -> None:
    """Fail unless the frozen run record holds the answers files' hashes and summaries."""
    entries = load_run_record(directory).method_entries
    found = _answers_entries(prompted, answers)
    differ = [key for key, value in found.items() if entries.get(key) != value]
    if differ:
        raise ValueError(
            f"The run record in {directory} holds other {' and '.join(differ)} than the answers "
            "files: an answers file changed after the method was scored, and its answers are "
            "never rewritten (docs/adr/0003). Restore it from git."
        )


def _per_problem_text(answers: Mapping[str, Sequence[LlmAnswer]]) -> str:
    """Return the text of per_problem.csv: each problem's recorded call, dev then test."""
    frames = [per_problem_frame(answers[part], part) for part in PARTS]
    return pd.concat(frames, ignore_index=True).to_csv(index=False)


def _check_per_problem(path: Path, text: str) -> None:
    """Fail unless per_problem.csv holds the rows derived from the answers files."""
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        raise ValueError(
            f"{path} is missing, or is not the rows derived from the answers files: restore it "
            "from git; it is a pure function of the answers."
        )


def _freeze_per_problem(directory: Path, answers: Mapping[str, Sequence[LlmAnswer]]) -> str:
    """Save per_problem.csv once; later, only check it."""
    path = directory / PER_PROBLEM_FILE
    text = _per_problem_text(answers)
    if path.exists():
        _check_per_problem(path, text)
        return f"{PER_PROBLEM_FILE} matches the answers files"
    staged = directory / f".{PER_PROBLEM_FILE}.partial"
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, path)
    problems = sum(len(answers[part]) for part in PARTS)
    return f"saved the calls of {problems:,} problems to {path}"


def _freeze_folder(
    method: PromptedLlm,
    prompted: _Prompted,
    parts: _Parts,
    answers: Mapping[str, Sequence[LlmAnswer]],
) -> str:
    """Save the method folder once it builds the answered requests; later, only check it.

    It is saved in a temporary folder, moved into place once the parity check passes: a folder
    that fails it is never left behind, and the next run saves it again.
    """
    folder = model_dir(prompted.name)
    if (folder / METHOD_FILE).exists():
        _check_parity(prompted, parts, answers, folder)
        return f"the method folder {folder} builds the answered requests"
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=folder.parent) as staging:
        staged = Path(staging) / folder.name
        save_method(method, staged)
        _check_parity(prompted, parts, answers, staged)
        staged.replace(folder)
    return f"saved the method folder {folder}"


def _check_parity(
    prompted: _Prompted, parts: _Parts, answers: Mapping[str, Sequence[LlmAnswer]], folder: Path
) -> None:
    """Fail unless a method folder holds the package's prompt and builds the answered requests.

    A fitted method's folder is the method fitted on all of dev: it builds the test requests
    only, its dev requests coming from ``cross_requests``.
    """
    loaded = load_method(folder, registry=METHODS)
    stored = loaded.prompt.content_sha256
    if stored != prompted.prompt.content_sha256:
        raise ValueError(
            f"The method folder {folder} holds the prompt {stored[:_SHORT_HASH]}, but the "
            "package's template, schema, settings and extra hash to "
            f"{prompted.prompt.content_sha256[:_SHORT_HASH]}."
        )
    for part in PARTS if prompted.kind == "zero_shot" else ("test",):
        try:
            check_complete(loaded.requests(parts.raw[part]), answers[part])
        except ValueError as error:
            raise ValueError(
                f"The method folder {folder} builds other {part} requests than the answered "
                f"ones: {error}"
            ) from error


def _metrics(
    prompted: _Prompted, parts: _Parts, predictions: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Score each part's predictions, under the name of its MLflow dataset."""
    return part_metrics(_truth(parts), predictions, prompted.spec.all_tags)


def _verify(args: argparse.Namespace) -> int:
    """Re-derive every frozen file of the method from its answers, offline, and report scores."""
    check_dataset()
    parts = _read_parts()
    prompted = _prompted(args.method)
    directory = method_dir(prompted.name)
    check_frozen_method(
        prompted.name, assets_dir=directory.parent, split=parts.split, registry=METHODS
    )
    print(f"the frozen files of {prompted.name} are complete and consistent with the split")
    _check_recipe(directory, prompted)
    answers = _checked_answers(prompted, parts)
    _check_record_entries(directory, prompted, answers)
    print("the run record holds the hashes and the summaries of the answers files")
    _check_append_only(prompted.name)
    print("each answers file holds every line of its committed versions, then new ones")
    frozen = _check_predictions(directory, prompted, parts, answers)
    print(
        f"re-derived {len(frozen):,} predictions and {PER_PROBLEM_FILE} from the answers: they "
        "equal the frozen ones"
    )
    _check_parity(prompted, parts, answers, model_dir(prompted.name))
    print(f"the method folder {model_dir(prompted.name)} builds the answered requests")
    _verify_extra(prompted, parts)
    for dataset, metrics in _metrics(prompted, parts, frozen).items():
        print(f"{dataset}: macro F1 {metrics['macro_f1']:.4f}, micro F1 {metrics['micro_f1']:.4f}")
    return 0


def _check_recipe(directory: Path, prompted: _Prompted) -> None:
    """Fail unless the frozen run record holds the recipe of the package."""
    stored = load_run_record(directory).recipe_sha256
    recipe = recipe_sha256(prompted.spec, prompted.prompt)
    if stored != recipe:
        raise ValueError(
            f"{directory} was frozen with another recipe ({stored}, not {recipe}): the prompt, "
            f"the settings or the parser of {prompted.name} changed since."
        )


def _checked_answers(prompted: _Prompted, parts: _Parts) -> dict[str, list[LlmAnswer]]:
    """Re-render every request and re-parse every answer; return the answers of each part."""
    requests, _ = _derive(prompted, parts)
    answers = _answers(prompted.name)
    for part in PARTS:
        check_complete(requests[part], answers[part])
        _check_stored_parse(prompted, answers[part])
    count = sum(len(answers[part]) for part in PARTS)
    print(f"re-rendered {count:,} requests: each has its answer, and each answer its stored parse")
    return answers


def _check_stored_parse(prompted: _Prompted, answers: Sequence[LlmAnswer]) -> None:
    """Fail unless each answer's stored tags, failure and unknown pieces are the parser's."""
    output_format = prompted.spec.settings.output_format
    differ = [
        answer.sample_id
        for answer in answers
        if ParsedTags(answer.tags, answer.unknown_pieces, answer.failed)
        != parse_answer(answer, output_format, prompted.spec.all_tags)
    ]
    if differ:
        raise ValueError(
            f"The stored tags, failed or unknown_pieces of the answers to the problems "
            f"{problem_list(differ)} are not what the parser reads in their text and status: "
            "the answers file was edited, or the parser changed."
        )


def _check_predictions(
    directory: Path,
    prompted: _Prompted,
    parts: _Parts,
    answers: Mapping[str, Sequence[LlmAnswer]],
) -> pd.DataFrame:
    """Fail unless the frozen predictions and calls are those the answers give; return them."""
    frozen = load_predictions(directory)
    if not frozen.equals(_encode(prompted, parts, answers)):
        raise ValueError(
            f"The frozen {PREDICTIONS_FILE} in {directory} differ from the predictions encoded "
            "from the answers files."
        )
    _check_per_problem(directory / PER_PROBLEM_FILE, _per_problem_text(answers))
    return frozen


def _verify_extra(prompted: _Prompted, parts: _Parts) -> None:
    """Run the package's own checks on every dev and test problem, when it has some."""
    verify_extra = getattr(prompted.package, "verify_extra", None)
    if verify_extra is None:
        return
    raw = pd.concat([parts.raw[part] for part in PARTS])
    verify_extra(raw)
    print(f"verify_extra of {prompted.package.__name__} passed on {len(raw):,} problems")


if __name__ == "__main__":
    sys.exit(main())
