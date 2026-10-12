"""The command line of the project: list the tagging methods, predict focus tags, evaluate.

    codeforces-tags methods [--data DIR]
    codeforces-tags predict INPUT [--method M] [--max-cost-usd USD]
    codeforces-tags evaluate [DATA] [--method M] [--part P] [--out DIR] [--max-cost-usd USD]

It also runs as ``python -m codeforces_tags`` and ``python -m codeforces_tags.cli``. ``methods``
lists every registered tagging method: its kind, its frozen dev (out of fold) and test macro F1,
scored on the challenge dataset of --data DIR or $CODEFORCES_DATA_DIR (empty, with a note, when
$CODEFORCES_DATA_DIR is not it), its latency (its latency.csv: median and slowest problem, loading,
and their sum, the slowest prediction of a fresh command), whether ADR 0004 makes it eligible and
names it best on dev, whether it is the default method (docs/adr/0008), what it needs (a uv
dependency group, weights, an API key) and whether it runs here.

``predict`` reads INPUT, one sample_<id>.json file (the dataset's record format) or a directory of
them, and prints one JSON line per exercise: its sample id and predicted focus tags, plus its
scores for a score method, and the cost of its request for a method that calls an API, marked
failed when its answer failed (it then gets no tag). Every command reports the focus tags only. A
sample needs a text in prob_desc_description and source_code; any other field may be left out, or
holds a text (the sample inputs and outputs: a list of texts, or its JSON text, as the dataset
writes it). The method is jev-statement-code unless --method names another (docs/adr/0008). A
method that calls an API sends one paid request per exercise: more than one request needs
--max-cost-usd, which the estimate from the method's latency.csv must not exceed, and the run stops
once its answers cost that much.

``evaluate`` scores a method on DATA, a directory of sample_<id>.json files with their tags
($CODEFORCES_DATA_DIR by default): --part test or dev of the frozen split, on the challenge
dataset only, or all of its problems; without --part, the test part of the challenge dataset, and
every problem of any other directory. It prints each tag's support, confusion counts, precision,
recall and F1, the macro and micro F1 (with group-bootstrap intervals on a part of the split) and
the tag confusion matrix; --out DIR writes them, with the predictions, as JSON and CSV files.

A method runs only once it is frozen and every need is met. When one is not, the command fails
with one line naming every unmet need, and --method tfidf, which runs offline and for free: it
never falls back on another method by itself. Notes (skipped files, the part chosen, the spend,
dev problems scored in sample, focus tags that no problem has) go to stderr, so that stdout holds
only the listing, the JSON lines or the evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from codeforces_tags.cli_methods import (
    PROG,
    CliError,
    api_keys,
    format_method_table,
    load_cli_method,
    method_table,
    offline_hint,
    resolve_method,
)
from codeforces_tags.config import (
    API_SDK_ERRORS,
    CLI_API_BATCH,
    CLI_COST_DECIMALS,
    CLI_DEFAULT_METHOD,
    CLI_JSON_LIST_FIELDS,
    CLI_OFFLINE_METHOD,
    CLI_REQUIRED_FIELDS,
    CLI_UNCAPPED_REQUESTS,
    DATA_DIR_ENV_VAR,
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    PYPROJECT_FILE,
    RAW_INPUT_COLUMNS,
    SPLIT_FILE,
)
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import (
    dataset_path,
    non_sample_files,
    read_sample_file,
    read_samples,
)
from codeforces_tags.evaluation.report import evaluate, report_text, save_report
from codeforces_tags.evaluation.timing import LATENCY_FILE, load_latency
from codeforces_tags.methods import METHODS, MethodSpec, ScoreMethod, TaggingMethod
from codeforces_tags.provenance import check_dataset
from codeforces_tags.splitting import load_split, select_part

# evaluate --part: a part of the frozen split, on the challenge dataset only, or every problem.
EVALUATED_PARTS = ("test", "dev", "all")
# The most sample files a message names; it counts the others.
_FILES_NAMED = 10
# What --help says of the command line, and of each command.
_DESCRIPTION = (
    "Predict the focus tags of Codeforces exercises, and evaluate the tagging methods. Without "
    f"--method, predict and evaluate run {CLI_DEFAULT_METHOD} (docs/adr/0008); --method "
    f"{CLI_OFFLINE_METHOD} runs offline, for free, and `{PROG} methods` lists every method and "
    "what it needs."
)
_COMMANDS = {
    "methods": (
        "list the methods: frozen scores, latency, eligibility, needs",
        "List the registered tagging methods: their frozen dev (out of fold) and test macro F1 "
        "on the challenge dataset, their latency, whether ADR 0004 makes them eligible and "
        "names one best on dev, the default method, what each needs and whether it runs here.",
    ),
    "predict": (
        "print each exercise's predicted focus tags, one JSON line each",
        "Print the predicted focus tags of each exercise of INPUT, one JSON line each: its "
        "sample id, its tags, its scores for a score method (compared with per-tag thresholds "
        "chosen out of fold), and its request's cost for a method that calls an API.",
    ),
    "evaluate": (
        "score a method on the frozen test part, or on a directory of samples",
        "Score a method on the frozen test part of the challenge dataset, its dev part, or "
        "every exercise of a directory of samples with their tags: per-tag precision, recall "
        "and F1, the macro and micro F1, and the tag confusion matrix.",
    ),
}


@dataclass(frozen=True)
class Predictions:
    """What a method predicted for the problems of an input, in their order.

    Attributes:
        tags: Int 0/1, indexed by ``sample_id``, one column per focus tag (``run_method`` drops
            a method's other tags).
        scores: A score method's scores, with the rows and columns of ``tags``; None otherwise.
        cost_usd: For a method that calls an API, what each problem's request cost, indexed by
            ``sample_id`` and named ``cost_usd``; None for a local method.
        failed: For a method that calls an API, whether each problem's answer failed (a request
            the API refused, or an answer the method cannot read: such a problem gets no tag),
            indexed by ``sample_id`` and named ``failed``; None for a local method.
    """

    tags: pd.DataFrame
    scores: pd.DataFrame | None = None
    cost_usd: pd.Series | None = None
    failed: pd.Series | None = None


@dataclass(frozen=True)
class EvaluationRows:
    """The problems an evaluation scores, with their true focus tags.

    Attributes:
        part: ``test`` or ``dev`` of the frozen split, or ``all``: every problem of a directory.
        data: The directory read.
        raw: The raw problems, indexed by ``sample_id``, with their tags.
        labels: Their 0/1 focus labels, one column per tag of ``FOCUS_TAGS``.
        groups: Their near-duplicate groups in the frozen split, which the bootstrap's intervals
            resample; None for ``all``.
        in_dev: How many of them carry the sample id of a dev problem of the frozen split.
    """

    part: str
    data: Path
    raw: pd.DataFrame
    labels: pd.DataFrame
    groups: pd.Series | None
    in_dev: int


def challenge_labels(data_dir: str | os.PathLike[str] | None) -> pd.DataFrame:
    """Read the focus labels of the challenge dataset, which score the frozen predictions.

    Args:
        data_dir: The dataset directory; None reads ``$CODEFORCES_DATA_DIR``, then the default.

    Returns:
        The 0/1 focus labels of every problem, indexed by ``sample_id``.

    Raises:
        CliError: If the directory does not exist, or is not the challenge dataset unchanged
            (``check_dataset``); both messages name ``--data`` and ``$CODEFORCES_DATA_DIR``.
    """
    path = _dataset_dir(data_dir)
    reason = _not_the_challenge(path)
    if reason is not None:
        raise CliError(f"{path} is not the challenge dataset: {reason}, or give it as --data DIR.")
    return focus_labels(read_samples(path))


def read_input(path: Path) -> pd.DataFrame:
    """Read the samples to predict: one ``sample_<id>.json`` file, or a directory of them.

    A directory's entries that are not sample files are named in a note on stderr.

    Args:
        path: The file, or the directory.

    Returns:
        The raw problems indexed by ``sample_id`` (``read_sample_file``, ``read_samples``), with
        each column of ``RAW_INPUT_COLUMNS`` that no sample holds added as None (object dtype:
        what ``read_samples`` gives a field that is null in every file), and each list of a
        field of ``CLI_JSON_LIST_FIELDS`` written as its JSON text, as the dataset writes it.

    Raises:
        CliError: If the path does not exist; if a file is not named ``sample_<id>.json``, cannot
            be read, is not JSON or holds no JSON object of fields (the message names it); if a
            directory holds no sample file, or two files of one sample id; or as
            ``check_fields``.
    """
    path = Path(path)
    if not path.exists():
        raise CliError(
            f"{path} does not exist: give a sample_<id>.json file or a directory of them."
        )
    try:
        raw = _read_directory(path) if path.is_dir() else read_sample_file(path)
    except (FileNotFoundError, ValueError) as error:
        raise CliError(str(error)) from error
    check_fields(raw)
    return _as_the_dataset_writes(raw)


def _sample_file(sample_id: object) -> str:
    """Name a sample by its file, as the dataset names it: ``sample_<id>.json``."""
    return f"sample_{sample_id}.json"


def check_fields(raw: pd.DataFrame, *, sample_name: Callable[[object], str] = _sample_file) -> None:
    """Refuse samples whose fields a method cannot read.

    Args:
        raw: Raw problems indexed by ``sample_id``, as their sample files hold them.
        sample_name: How a message names a sample, from its index label: by its file,
            ``sample_<id>.json``, unless the caller names it otherwise (the model server names
            the rows of a request). A message never quotes a field's text.

    Raises:
        CliError: If a field of ``CLI_REQUIRED_FIELDS`` holds no text (it is missing, null, not
            a string, or blank) in some sample; or else if another field of
            ``RAW_INPUT_COLUMNS`` holds neither a text nor null, or a field of
            ``CLI_JSON_LIST_FIELDS`` neither a list of texts, its JSON text nor null. Each
            message names every such field and its samples, on one line.
    """
    lacking = {field: _without_text(raw, field) for field in CLI_REQUIRED_FIELDS}
    named = [
        f"{field} holds none in {_named_samples(ids, sample_name)}"
        for field, ids in lacking.items()
        if ids
    ]
    if named:
        raise CliError(
            f"every sample needs a text in {' and '.join(CLI_REQUIRED_FIELDS)}, but "
            f"{'; '.join(named)}."
        )
    optional = [field for field in RAW_INPUT_COLUMNS if field not in CLI_REQUIRED_FIELDS]
    unreadable = [reason for field in optional if (reason := _unreadable(raw, field, sample_name))]
    if unreadable:
        raise CliError(f"a sample holds a field that no method can read: {'; '.join(unreadable)}.")


def estimate_cost_usd(
    name: str, problems: int, *, assets_dir: Path = METHODS_ASSETS_DIR
) -> float | None:
    """Estimate what a method that calls an API spends on some problems, before any request.

    Args:
        name: The method's registry key.
        problems: How many problems it would be asked about, one request each.
        assets_dir: The folder of the frozen methods.

    Returns:
        ``problems`` times the mean ``cost_usd`` of the timed requests of the method's
        ``latency.csv``; None without that file, or when it records no cost (a local method).
    """
    directory = assets_dir / name
    if not (directory / LATENCY_FILE).is_file():
        return None
    calls = load_latency(directory).calls
    if calls is None:
        return None
    return problems * float(calls["cost_usd"].mean())


def check_cost_cap(
    name: str,
    spec: MethodSpec,
    problems: int,
    max_cost_usd: float | None,
    *,
    assets_dir: Path = METHODS_ASSETS_DIR,
) -> None:
    """Check the cost cap of a run before the method loads and before any request.

    Args:
        name: The method's registry key.
        spec: Its spec.
        problems: How many problems it would be asked about.
        max_cost_usd: The ``--max-cost-usd`` given, or None.
        assets_dir: The folder of the frozen methods, whose ``latency.csv`` gives the estimate.

    Raises:
        CliError: If a cap is given to a method that sends no paid request; if no cap is given
            for more than ``CLI_UNCAPPED_REQUESTS`` paid requests (the message gives the
            estimate); if the cap is not above 0 (NaN included); or if the estimate
            (``estimate_cost_usd``) is above it. Each message ends with
            ``offline_hint(name)``.
    """
    if not spec.calls_api:
        if max_cost_usd is not None:
            raise CliError(
                f"{name} sends no paid request: --max-cost-usd is for a method that calls an "
                f"API{offline_hint(name)}."
            )
        return
    estimate = estimate_cost_usd(name, problems, assets_dir=assets_dir)
    if max_cost_usd is None:
        _check_uncapped(name, problems, estimate)
    elif not max_cost_usd > 0:
        raise CliError(
            f"the cost cap (--max-cost-usd) must be above 0, not {max_cost_usd}"
            f"{offline_hint(name)}."
        )
    elif estimate is not None and estimate > max_cost_usd:
        raise CliError(
            f"{name}: {_counted(problems, 'request')} would cost about ${estimate:.4f}, above the "
            f"cost cap of ${max_cost_usd}: raise --max-cost-usd{offline_hint(name)}."
        )


def run_method(
    method: TaggingMethod,
    spec: MethodSpec,
    raw_problems: pd.DataFrame,
    *,
    max_cost_usd: float | None = None,
) -> Predictions:
    """Predict the tags of raw problems, handing the method only its input columns.

    A local method predicts in one call; a ``ScoreMethod`` scores in one call and takes its tags
    from those scores (``predict_from_scores``), so each problem goes through it once. A method that
    calls an API is asked ``CLI_API_BATCH`` problems at a time, in order, through
    ``apredict_with_answers``, each batch in an event loop of its own; each answer is priced
    with the LLM toolkit's ``answer_cost_usd``, and its scores come from the same answers
    through the method's ``scores_from_answers`` when it has one (no second paid request).

    Args:
        method: The loaded method.
        spec: Its spec: ``calls_api`` and ``input_columns``.
        raw_problems: Raw problems indexed by ``sample_id``; other columns (tags, split) are
            dropped before the method sees them.
        max_cost_usd: For a method that calls an API, the spend after which no batch is sent.

    Returns:
        The predictions of the focus tags, with the scores of a score method, and the cost and
        failed answers of an API method.

    Raises:
        CliError: Once the answers cost ``max_cost_usd`` (nothing is returned), and when the
            API does not answer (``TransientApiError``: no network, a rate limit) or its SDK
            refuses the request (an error of ``API_SDK_ERRORS``: a refused key, no credit),
            named by its class and status, never its text.
        TypeError: If a method that calls an API has no ``apredict_with_answers``.
    """
    inputs = raw_problems[list(spec.input_columns)]
    if spec.calls_api:
        predictions = _ask_in_batches(method, spec, inputs, max_cost_usd)
    elif isinstance(method, ScoreMethod):
        scores = method.scores(inputs)
        predictions = Predictions(method.predict_from_scores(scores), scores)
    else:
        predictions = Predictions(method.predict(inputs), None)
    return _focus_tags_only(predictions)


def prediction_lines(predictions: Predictions) -> list[str]:
    """Write each problem's predictions as one JSON object, in the order of the problems.

    Args:
        predictions: What ``run_method`` returned.

    Returns:
        One line per problem: ``sample_id`` and ``tags`` (its predicted tags, in the order of
        ``FOCUS_TAGS``), then ``scores`` (by tag) for a score method, ``cost_usd`` (rounded to
        ``CLI_COST_DECIMALS``) for a method that calls an API, and ``"failed": true`` for a
        problem whose answer failed.

    Raises:
        ValueError: If a score is not a finite number: the line would not be JSON.
    """
    lines = []
    for sample_id, row in predictions.tags.iterrows():
        line = {"sample_id": int(sample_id), "tags": [tag for tag, value in row.items() if value]}
        if predictions.scores is not None:
            scores = predictions.scores.loc[sample_id]
            line["scores"] = {tag: float(score) for tag, score in scores.items()}
        if predictions.cost_usd is not None:
            line["cost_usd"] = round(float(predictions.cost_usd.loc[sample_id]), CLI_COST_DECIMALS)
        if predictions.failed is not None and predictions.failed.loc[sample_id]:
            line["failed"] = True
        lines.append(json.dumps(line, allow_nan=False))
    return lines


def evaluation_rows(
    data_dir: str | os.PathLike[str] | None, part: str | None, *, split_path: Path = SPLIT_FILE
) -> EvaluationRows:
    """Read the problems to evaluate: a part of the frozen split, or every problem of a directory.

    Without a part, the test part of the challenge dataset is read, and every problem of any
    other directory, with a note on stderr that says so.

    Args:
        data_dir: A directory of ``sample_<id>.json`` files with their tags; None reads
            ``$CODEFORCES_DATA_DIR``, then the default.
        part: ``test`` or ``dev`` (by the frozen split, on the challenge dataset only), ``all``
            (every problem of the directory), or None.
        split_path: The frozen split.

    Returns:
        The problems, their focus labels, their near-duplicate groups (for a part of the split)
        and how many of them are dev problems.

    Raises:
        CliError: If the directory does not exist; if ``test`` or ``dev`` is asked of another
            directory than the challenge dataset (``check_dataset``); if a problem has no list of
            tags; or as ``read_input``.
    """
    path = _dataset_dir(data_dir, argument="DATA")
    if part != "all":
        reason = _not_the_challenge(path)
        if reason is None:
            return _frozen_part(path, part or "test", split_path)
        if part is not None:
            raise CliError(
                f"--part {part} selects problems by the frozen split, on the challenge dataset "
                f"only, and {path} is not it: {reason}, or give it as DATA; or pass --part all to "
                f"evaluate every problem of {path}."
            )
    rows = _every_problem(path, split_path)
    if part is None:
        _note(
            f"{path} is not the challenge dataset: evaluating its {len(rows.raw)} problems "
            "(--part all)"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the parser of the command line, one subcommand per command.

    Returns:
        The parser: ``methods [--data DIR]``, ``predict INPUT [--method M] [--max-cost-usd
        USD]`` and ``evaluate [DATA] [--method M] [--part P] [--out DIR] [--max-cost-usd USD]``.
    """
    parser = argparse.ArgumentParser(prog=PROG, description=_DESCRIPTION)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    added = {
        name: commands.add_parser(name, help=help_text, description=description)
        for name, (help_text, description) in _COMMANDS.items()
    }
    added["methods"].add_argument(
        "--data",
        metavar="DIR",
        help="the challenge dataset, to score the frozen predictions on "
        "(default: $CODEFORCES_DATA_DIR)",
    )
    added["predict"].add_argument(
        "input", type=Path, metavar="INPUT", help="a sample_<id>.json file, or a directory of them"
    )
    _add_method_options(added["predict"])
    evaluation = added["evaluate"]
    evaluation.add_argument(
        "data",
        nargs="?",
        metavar="DATA",
        help="a directory of sample_<id>.json files with their tags (default: "
        "$CODEFORCES_DATA_DIR)",
    )
    evaluation.add_argument(
        "--part",
        choices=EVALUATED_PARTS,
        help="test or dev: that part of the frozen split, on the challenge dataset only; all: "
        "every problem of DATA (default: test on the challenge dataset, all on any other "
        "directory)",
    )
    evaluation.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="a new or empty folder to write the report in (report.json, CSV tables, predictions)",
    )
    _add_method_options(evaluation)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status: 0, or 1 after a ``CliError``, printed on one line of stderr, or once
        the reader of stdout has gone (``predict ... | head -1``). A usage error exits with
        status 2, from argparse.
    """
    args = build_parser().parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "methods": _methods,
        "predict": _predict,
        "evaluate": _evaluate,
    }
    try:
        status = commands[args.command](args)
        sys.stdout.flush()  # a reader that has gone fails the flush here, rather than at exit
        return status
    except CliError as error:
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # Python's recipe: what is left of stdout goes to /dev/null, so that the flush at exit
        # does not fail again with a traceback.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


def _methods(args: argparse.Namespace) -> int:
    """List the registered methods; with no F1 and a note when $CODEFORCES_DATA_DIR is not set."""
    try:
        labels = challenge_labels(args.data)
    except CliError as error:
        if args.data is not None:
            raise  # a --data that is wrong is an error, not a dataset left out
        _note(f"no F1 scores without the challenge dataset: {error}")
        labels = None
    table = method_table(
        labels, registry=METHODS, assets_dir=METHODS_ASSETS_DIR, pyproject=PYPROJECT_FILE
    )
    print(format_method_table(table))
    return 0


def _predict(args: argparse.Namespace) -> int:
    """Print the predicted tags of each sample of the input, one JSON line each."""
    spec = _resolve(args.method)
    predictions = _run(args, spec, read_input(args.input))
    print("\n".join(prediction_lines(predictions)))
    _note_spend(predictions)
    _note_failed(predictions)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Score a method on problems of DATA: print the scores, and write them under --out."""
    spec = _resolve(args.method)
    if args.out is not None:
        _prepare_report_folder(args.out)
    rows = evaluation_rows(args.data, args.part, split_path=SPLIT_FILE)
    problems = len(rows.raw)
    predictions = _run(args, spec, rows.raw)
    evaluation = evaluate(rows.labels, predictions.tags, groups=rows.groups)
    print(f"{args.method} on {rows.part}: {problems} problems\n{report_text(evaluation)}")
    _note_spend(predictions)
    _note_failed(predictions)
    _note_caveats(rows)
    if args.out is not None:
        cost_usd = None if predictions.cost_usd is None else float(predictions.cost_usd.sum())
        failed = None if predictions.failed is None else int(predictions.failed.sum())
        details = {"method": args.method, "part": rows.part, "data": str(rows.data)}
        details |= {"problems": problems, "cost_usd": cost_usd, "failed": failed}
        tables = {"predictions.csv": predictions.tags, "scores.csv": predictions.scores}
        save_report(evaluation, args.out, details, tables)
        _note(f"wrote the report in {args.out}")
    return 0


def _resolve(name: str) -> MethodSpec:
    """Resolve a method against the registry, the frozen methods and the groups of the project."""
    return resolve_method(
        name, registry=METHODS, assets_dir=METHODS_ASSETS_DIR, pyproject=PYPROJECT_FILE
    )


def _run(args: argparse.Namespace, spec: MethodSpec, raw: pd.DataFrame) -> Predictions:
    """Check the cost cap of the run, then load the method and predict the raw problems."""
    check_cost_cap(args.method, spec, len(raw), args.max_cost_usd, assets_dir=METHODS_ASSETS_DIR)
    method = load_cli_method(args.method, registry=METHODS, assets_dir=METHODS_ASSETS_DIR)
    return run_method(method, spec, raw, max_cost_usd=args.max_cost_usd)


def _add_method_options(command: argparse.ArgumentParser) -> None:
    """Add the options that choose the method and cap what it spends."""
    command.add_argument(
        "--method",
        default=CLI_DEFAULT_METHOD,
        metavar="M",
        help=f"the tagging method (default: {CLI_DEFAULT_METHOD}; {CLI_OFFLINE_METHOD} runs "
        f"offline, for free; `{PROG} methods` lists them all)",
    )
    command.add_argument(
        "--max-cost-usd",
        type=float,
        metavar="USD",
        help="for a method that calls an API: the most its requests may cost, in USD (needed for "
        "more than one request)",
    )


def _note(message: str) -> None:
    """Print a note on stderr, which keeps stdout for the listing or the JSON lines."""
    print(f"{PROG}: note: {message}", file=sys.stderr)


def _dataset_dir(data_dir: str | os.PathLike[str] | None, argument: str = "--data DIR") -> Path:
    """Return the dataset directory, or raise a CliError naming the argument and the variable."""
    path = dataset_path(data_dir)
    if not path.is_dir():
        state = "is a file" if path.is_file() else "does not exist"
        raise CliError(
            f"{path} {state}: {argument} must be a directory of sample_<id>.json files (default: "
            f"${DATA_DIR_ENV_VAR})."
        )
    return path


def _not_the_challenge(path: Path) -> str | None:
    """Say why a directory is not the challenge dataset (``check_dataset``); None when it is.

    The reason starts lower case and ends without its period, to go after a colon.
    """
    try:
        check_dataset(path)
    except (OSError, ValueError) as error:
        reason = str(error).rstrip(".")
        return reason[:1].lower() + reason[1:]
    return None


def _read_directory(path: Path) -> pd.DataFrame:
    """Read every sample file of a directory, after a note that names the entries it skips."""
    skipped = non_sample_files(path)
    if skipped:
        what = "entry that is not a sample_<id>.json file"
        if len(skipped) > 1:
            what = "entries that are not sample_<id>.json files"
        _note(f"skipped {len(skipped)} {what}: {', '.join(skipped)}")
    return read_samples(path)


def _as_the_dataset_writes(raw: pd.DataFrame) -> pd.DataFrame:
    """Add each raw input column that no sample holds, as None; write each list as JSON text."""
    missing = [column for column in RAW_INPUT_COLUMNS if column not in raw.columns]
    raw = raw.assign(**dict.fromkeys(missing, None))
    for field in CLI_JSON_LIST_FIELDS:
        raw[field] = raw[field].map(
            lambda value: json.dumps(value) if isinstance(value, list) else value
        )
    return raw


def _without_text(raw: pd.DataFrame, field: str) -> list[int]:
    """Return the samples whose field is missing, null, not a string or blank."""
    if field not in raw.columns:
        return list(raw.index)
    return [
        sample_id
        for sample_id, value in raw[field].items()
        if not (isinstance(value, str) and value.strip())
    ]


def _unreadable(raw: pd.DataFrame, field: str, sample_name: Callable[[object], str]) -> str | None:
    """Say which samples hold a value of an optional field that no method reads; None if none."""
    if field not in raw.columns:
        return None
    lists = field in CLI_JSON_LIST_FIELDS
    readable = _text_list if lists else _is_text
    ids = [
        sample_id
        for sample_id, value in raw[field].items()
        if not (_is_null(value) or readable(value))
    ]
    if not ids:
        return None
    kind = "a list of texts (or its JSON text)" if lists else "a text"
    return f"{field} holds neither {kind} nor null in {_named_samples(ids, sample_name)}"


def _is_null(value: object) -> bool:
    """Tell whether a raw value is null: JSON's null, or NaN where a sample lacks the field."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_text(value: object) -> bool:
    """Tell whether a raw value is a text."""
    return isinstance(value, str)


def _text_list(value: object) -> bool:
    """Tell whether a raw value is a list of texts, or the JSON text of one."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _named_samples(sample_ids: list, sample_name: Callable[[object], str] = _sample_file) -> str:
    """Name some samples, by their files unless told otherwise, at most ``_FILES_NAMED``."""
    named = ", ".join(sample_name(sample_id) for sample_id in sample_ids[:_FILES_NAMED])
    others = len(sample_ids) - _FILES_NAMED
    return named if others <= 0 else f"{named} and {others} more"


def _check_uncapped(name: str, problems: int, estimate: float | None) -> None:
    """Refuse more than ``CLI_UNCAPPED_REQUESTS`` paid requests without a cost cap."""
    if problems <= CLI_UNCAPPED_REQUESTS:
        return
    cost = "at an unknown cost" if estimate is None else f"about ${estimate:.4f}"
    raise CliError(
        f"{name} sends one paid request per problem: {problems} requests, {cost}, need a cost "
        f"cap: pass --max-cost-usd X{offline_hint(name)}."
    )


def _ask_in_batches(
    method: TaggingMethod, spec: MethodSpec, inputs: pd.DataFrame, max_cost_usd: float | None
) -> Predictions:
    """Ask a method that calls an API, batch by batch, until done or the cap is reached."""
    if not hasattr(method, "apredict_with_answers"):
        raise TypeError(
            f"{method.name} calls an API but has no apredict_with_answers: the command line asks "
            "such a method through it, as tracking latency does."
        )
    batches: list[Predictions] = []
    spent = 0.0
    for start in range(0, len(inputs), CLI_API_BATCH):
        if max_cost_usd is not None and spent >= max_cost_usd:
            raise CliError(
                f"{method.name} reached the cost cap of ${max_cost_usd}: it spent ${spent:.6f} "
                f"on {start} of {len(inputs)} problems, and nothing is reported"
                f"{offline_hint(method.name)}."
            )
        try:
            batch = _ask_batch(method, spec, inputs.iloc[start : start + CLI_API_BATCH])
        except CliError:
            if start:
                _note(
                    f"{start} of {len(inputs)} problems were answered before this failure, for "
                    f"${spent:.6f}; nothing is reported"
                )
            raise
        batches.append(batch)
        spent += float(batch.cost_usd.sum())
    return _join_batches(batches)


def _ask_batch(method: TaggingMethod, spec: MethodSpec, batch: pd.DataFrame) -> Predictions:
    """Ask a method that calls an API about one batch; name an error of the API, or of its SDK."""
    # Loading the method imported the LLM toolkit: these imports add nothing, and keep the
    # toolkit, which sets MLflow's environment, out of the command line's own imports.
    from codeforces_tags.llm.answers import (  # pylint: disable=import-outside-toplevel
        answer_cost_usd,
    )
    from codeforces_tags.llm.client import (  # pylint: disable=import-outside-toplevel
        TransientApiError,
    )

    try:
        tags, answers = asyncio.run(method.apredict_with_answers(batch))
    except TransientApiError as error:
        # Named by its cause, as an SDK's error is: an adapter's text may quote the SDK's.
        raise CliError(
            f"{method.name} got no answer from its API (no network, a rate limit or a server "
            f"error): {_error_kind(error.__cause__ or error)}{offline_hint(method.name)}."
        ) from error
    except _sdk_errors(spec.groups) as error:  # evaluated once raised: the SDK is imported then
        raise CliError(_refused_message(method.name, spec, error)) from error
    by_problem = {answer.sample_id: answer for answer in answers}
    ordered = [by_problem[int(sample_id)] for sample_id in batch.index]
    scorer = getattr(method, "scores_from_answers", None)
    return Predictions(
        tags=tags,
        scores=None if scorer is None else scorer(answers, batch.index),
        cost_usd=pd.Series(
            [answer_cost_usd(answer)["total_cost"] for answer in ordered],
            index=batch.index,
            name="cost_usd",
        ),
        failed=pd.Series(
            [answer.failed or answer.status != "completed" for answer in ordered],
            index=batch.index,
            name="failed",
        ),
    )


def _join_batches(batches: list[Predictions]) -> Predictions:
    """Join the predictions of the batches of one run, in order."""
    scores = [batch.scores for batch in batches if batch.scores is not None]
    return Predictions(
        tags=pd.concat([batch.tags for batch in batches]),
        scores=pd.concat(scores) if scores else None,
        cost_usd=pd.concat([batch.cost_usd for batch in batches]),
        failed=pd.concat([batch.failed for batch in batches]),
    )


def _focus_tags_only(predictions: Predictions) -> Predictions:
    """Keep the focus tags of a method's predictions and scores, which every command reports."""
    focus = list(FOCUS_TAGS)
    scores = None if predictions.scores is None else predictions.scores[focus]
    return replace(predictions, tags=predictions.tags[focus], scores=scores)


def _sdk_errors(groups: tuple[str, ...]) -> tuple[type[BaseException], ...]:
    """Return the base error classes of the groups' SDKs that are imported; import none."""
    found = []
    for group in groups:
        module, _, name = API_SDK_ERRORS.get(group, "").rpartition(".")
        error = getattr(sys.modules.get(module), name, None)
        if isinstance(error, type) and issubclass(error, BaseException):
            found.append(error)
    return tuple(found)


def _error_kind(error: BaseException) -> str:
    """Name an error by its class and status, never by its text, which can echo the state."""
    status = getattr(error, "status", None)
    return type(error).__name__ if status is None else f"{type(error).__name__} {status}"


def _refused_message(name: str, spec: MethodSpec, error: BaseException) -> str:
    """Say that a method's API refused a request: its SDK's error, by class and status."""
    return (
        f"{name}: its API refused the request ({_error_kind(error)}): check that "
        f"{' or '.join(api_keys(spec))} holds a valid key, with credit{offline_hint(name)}."
    )


def _note_spend(predictions: Predictions) -> None:
    """Note how many paid requests a method that calls an API sent, and what they cost."""
    cost_usd = predictions.cost_usd
    if cost_usd is None:
        return
    _note(f"{_counted(len(cost_usd), 'paid request')} cost ${cost_usd.sum():.6f}")


def _note_failed(predictions: Predictions) -> None:
    """Note the problems whose answer failed, which got no tag."""
    failed = predictions.failed
    if failed is None or not failed.any():
        return
    ids = failed.index[failed].tolist()
    _note(
        f"{len(ids)} of {len(failed)} answers failed (the API refused the request, or its answer "
        f"was not one the method reads): {_named_samples(ids)} got no tag"
    )


def _counted(count: int, noun: str) -> str:
    """Count a noun: ``"1 request"``, ``"2 requests"``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _note_caveats(rows: EvaluationRows) -> None:
    """Note what makes an evaluation's scores read otherwise: dev problems, absent focus tags."""
    if rows.in_dev:
        are = "is a dev problem" if rows.in_dev == 1 else "are dev problems"
        _note(
            f"{rows.in_dev} of these problems {are} of the frozen split, by sample id: every "
            "frozen method was fitted or thresholded on the dev part, so their scores are in "
            f"sample ({PROG} methods lists the out-of-fold dev macro F1)"
        )
    absent = [tag for tag in rows.labels.columns if not rows.labels[tag].any()]
    if absent:
        _note(
            f"no problem here has the focus tags {', '.join(absent)}: the F1 of each is 0 (as "
            "with scikit-learn's zero_division=0), which lowers the macro F1"
        )


def _frozen_part(path: Path, part: str, split_path: Path) -> EvaluationRows:
    """Read a part of the frozen split of the challenge dataset, with its labels and groups."""
    rows = select_part(read_samples(path), load_split(split_path), part)
    in_dev = len(rows) if part == "dev" else 0
    return EvaluationRows(part, path, rows, focus_labels(rows), rows["group"], in_dev)


def _every_problem(path: Path, split_path: Path) -> EvaluationRows:
    """Read every problem of a directory, with its labels; count those with a dev problem's id."""
    raw = read_input(path)
    tags = raw["tags"] if "tags" in raw else pd.Series(None, index=raw.index, dtype=object)
    untagged = [sample_id for sample_id, value in tags.items() if not isinstance(value, list)]
    if untagged:
        raise CliError(
            "every problem to evaluate needs its tags, as a list, but tags holds none in "
            f"{_named_samples(untagged)}."
        )
    split = load_split(split_path)
    in_dev = int(raw.index.isin(split.index[split["part"] == "dev"]).sum())
    return EvaluationRows("all", path, raw, focus_labels(raw), None, in_dev)


def _prepare_report_folder(out: Path) -> None:
    """Create the report folder before the run; refuse one that holds something, or cannot be."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CliError(
            f"{out} is not an empty folder: give --out a new or empty folder, so that a report is "
            "never mixed with other files."
        )
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CliError(
            f"cannot create the report folder {out}: {error.strerror or error}."
        ) from error


if __name__ == "__main__":
    sys.exit(main())
