"""Command line of the tracking package.

    python -m codeforces_tags.tracking latency --method NAME [--max-cost-usd X] [--data DIR]
        [--split FILE]
    python -m codeforces_tags.tracking datasets [--data DIR] [--split FILE]
    python -m codeforces_tags.tracking evaluate --part dev|test [--method NAME]
    python -m codeforces_tags.tracking discard --run RUN_ID
    python -m codeforces_tags.tracking calibrate [--data DIR] [--split FILE]

Every command also takes --tracking-uri URI and --artifact-dir DIR (the MLflow store). The
defaults come from codeforces_tags.config.

latency times a frozen method once (50 dev problems one at a time, one call over them and the
loading), writes them to assets/methods/NAME/latency.csv, and logs their summary to the method's
MLflow run. A later run times nothing: it logs the summary of the committed file, and needs no
dataset. Run it after the method's freeze, with the method's dependency groups. A method that
calls an API is timed on 50 live requests, which cost money: the first run needs --max-cost-usd,
and its bulk time comes from its answers files instead of one call over the 50 problems.

datasets checks the dataset, then copies the problems of each part once into the MLflow
evaluation datasets codeforces-dev and codeforces-test: one record per problem, with its inputs,
expected tags, fold and group. A later run only checks them, and refuses any change.

evaluate scores the frozen predictions of a method (without --method, of every registered method
that is frozen) on a part's evaluation dataset: one run eval-NAME-PART, one trace per problem with
its expected and predicted tags, its scores and, for an API method, its recorded cost. Run
datasets and the method's freeze first. A complete run is only checked again, and the test part
is evaluated once per method. Without --method, it refuses before the first run when a frozen
method has no logged model in the store: log each with its freeze (llm score for a prompted
method), or pass --method NAME for each method to evaluate.

discard deletes an evaluation run that is not complete (it stopped, or its traces were refused)
with its traces, so that its part can be evaluated again; deleting it in the MLflow UI would keep
its traces beside the next run's. A complete run is never discarded.

calibrate logs to the method run of tfidf what notebook 04's section 5 shows: how well four
probabilities built from its out-of-fold dev scores are calibrated (Brier score, log loss and
Brier skill, as metrics, and with their intervals in calibration/calibration_table.csv), and two
reliability diagrams. It loads the frozen method folder, never refits it, and cross-fits the
first look on the dev problems. Run it after the freeze of tfidf. Each run computes everything
again and logs only the keys and files the run lacks: it never updates one, even after a change
of code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codeforces_tags.config import DATASET_SHA256, SPLIT_FILE
from codeforces_tags.data.loading import load_dataset
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.methods import METHODS, method_dir
from codeforces_tags.provenance import check_dataset, file_sha256
from codeforces_tags.splitting import PARTS, load_split
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.evaluation import (
    CONTENT_TAG,
    discard_evaluation_run,
    ensure_datasets,
    evaluate_method,
    unlogged_methods,
)
from codeforces_tags.tracking.latency import run_latency


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, ``tracking_uri`` and ``artifact_dir``; for latency
        ``method``, ``max_cost_usd``, ``data`` and ``split``; for datasets ``data`` and ``split``;
        for evaluate ``part`` and ``method``; for discard ``run``; for calibrate ``data`` and
        ``split``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.tracking",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    latency = commands.add_parser("latency", help="time a frozen method once, log its summary")
    latency.add_argument("--method", required=True, help="the method's registry key")
    latency.add_argument(
        "--max-cost-usd",
        type=float,
        help="for a method that calls an API: stop sending timed requests once they cost this much",
    )
    datasets = commands.add_parser(
        "datasets", help="copy each part's problems into an MLflow evaluation dataset once"
    )
    evaluate = commands.add_parser(
        "evaluate", help="log a frozen method's predictions, one trace per problem"
    )
    evaluate.add_argument("--part", required=True, choices=PARTS, help="the part to score")
    evaluate.add_argument(
        "--method",
        choices=sorted(METHODS),
        help="the method's registry key (default: every frozen registered method)",
    )
    discard = commands.add_parser(
        "discard", help="delete an incomplete evaluation run and its traces"
    )
    discard.add_argument("--run", required=True, help="the evaluation run's id")
    calibrate = commands.add_parser(
        "calibrate", help="log the calibration of tfidf's out-of-fold scores to its method run"
    )
    for command in (latency, datasets, calibrate):
        command.add_argument("--data", help="dataset folder (default: $CODEFORCES_DATA_DIR)")
        command.add_argument("--split", type=Path, default=SPLIT_FILE, help="frozen split")
    for command in commands.choices.values():
        add_store_arguments(command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of the tracking package.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status, 0; a method that is not frozen or not registered, a refused cost cap
        or one reached (latency), a refused dataset, an evaluation dataset or run that must
        not be redone, a frozen method with no logged model (evaluate without --method), or a
        tfidf that is not frozen or not logged (calibrate) raises.
    """
    args = parse_args(argv)
    commands = {
        "latency": _latency,
        "datasets": _datasets,
        "evaluate": _evaluate,
        "discard": _discard,
        "calibrate": _calibrate,
    }
    return commands[args.command](args)


def _latency(args: argparse.Namespace) -> int:
    """Time the method, or read its committed timings, and log their summary."""
    print(
        run_latency(
            args.method,
            max_cost_usd=args.max_cost_usd,
            data_dir=args.data,
            split_path=args.split,
            tracking_uri=args.tracking_uri,
            artifact_dir=args.artifact_dir,
        )
    )
    return 0


def _datasets(args: argparse.Namespace) -> int:
    """Make the evaluation datasets of the challenge's dataset once, or check them."""
    check_dataset(args.data)
    split = load_split(args.split)
    datasets = ensure_datasets(
        load_dataset(args.data),
        split,
        dataset_sha256=DATASET_SHA256,
        split_sha256=file_sha256(args.split),
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
    )
    for part, dataset in datasets.items():
        records = int((split["part"] == part).sum())
        print(f"{dataset.name}: {records:,} records, content_sha256 {dataset.tags[CONTENT_TAG]}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Evaluate the method, or every frozen registered method, on the part."""
    store = {"tracking_uri": args.tracking_uri, "artifact_dir": args.artifact_dir}
    names = [args.method] if args.method else _logged_frozen_methods(store)
    for name in names:
        run_id = evaluate_method(name, args.part, **store)
        print(f"eval-{name}-{args.part}: run {run_id}, traces checked")
    return 0


def _logged_frozen_methods(store: dict[str, object]) -> list[str]:
    """Return the frozen registered methods, refusing if the store lacks one's logged model."""
    names = _frozen_methods()
    missing = unlogged_methods(names, registry=METHODS, **store)
    if missing:
        raise ValueError(
            f"No logged model in the store for {missing}: log each with its freeze or llm score "
            "command, or pass --method NAME for each method to evaluate."
        )
    return names


def _discard(args: argparse.Namespace) -> int:
    """Delete an incomplete evaluation run and its traces."""
    deleted = discard_evaluation_run(
        args.run, tracking_uri=args.tracking_uri, artifact_dir=args.artifact_dir
    )
    print(f"discarded the evaluation run {args.run} and its {deleted} traces")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    """Log the calibration of the frozen tfidf's out-of-fold scores to its method run."""
    # Imported here: calibrate pulls in the tag model (skops), first_look and matplotlib, which
    # the other commands do not need, and which latency would otherwise import before its timer
    # starts, leaving their import out of the load time it measures.
    from codeforces_tags.tracking.calibration import (  # pylint: disable=import-outside-toplevel
        run_calibration,
    )

    print(
        run_calibration(
            data_dir=args.data,
            split_path=args.split,
            tracking_uri=args.tracking_uri,
            artifact_dir=args.artifact_dir,
        )
    )
    return 0


def _frozen_methods() -> list[str]:
    """Return the registered methods that are frozen (they have a run record), by name."""
    return [name for name in sorted(METHODS) if (method_dir(name) / RUN_RECORD_FILE).is_file()]


if __name__ == "__main__":
    sys.exit(main())
