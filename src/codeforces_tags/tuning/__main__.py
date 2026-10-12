"""Command line of the tuning study, and of the tag model it tuned, the tagging method tfidf.

    python -m codeforces_tags.tuning run [--trials N] [--workers W]
    python -m codeforces_tags.tuning worker --trials N --seed S
    python -m codeforces_tags.tuning export [--out DIR]
    python -m codeforces_tags.tuning freeze [--allow-dirty]
    python -m codeforces_tags.tuning verify

Every command also takes --data DIR (the dataset) and --split FILE (the frozen split); run,
worker and export take --journal FILE (the Optuna journal); freeze and verify take --results DIR
(the study's frozen results) and --assets DIR (the folder of the frozen methods); run, export and
freeze take --tracking-uri URI and --artifact-dir DIR (the MLflow store), and refuse a URI that
is not sqlite:/// before they do any work. The defaults come from codeforces_tags.config.

run starts the workers as processes of their own, so that trials run in parallel without
scikit-learn's n_jobs (joblib would write pickle files), then logs the study to MLflow from
this process alone. Each worker runs BLAS on TUNING_BLAS_THREADS threads: otherwise the BLAS
calls of liblinear start one thread per CPU in every worker, and the threads compete.

freeze fits the tag model with the study's frozen configuration, checks that it reproduces the
study, and freezes it once as the tagging method tfidf in assets/methods/tfidf/, then logs it to
MLflow; verify checks those files offline. Neither re-runs the study. This module only wires the
steps together: the search is in search.py, the MLflow logging of the study in tracking.py, the
frozen results in results.py, and freeze and verify in freeze.py.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from subprocess import Popen

import optuna
import pandas as pd
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState
from threadpoolctl import threadpool_limits

from codeforces_tags.config import (
    OPTUNA_JOURNAL,
    SPLIT_FILE,
    TFIDF_BASELINE_DIR,
    TUNING_BLAS_THREADS,
    TUNING_EXPORT_TOLERANCE,
    TUNING_SEED,
    TUNING_TRIALS,
    TUNING_WORKERS,
)
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods.parts import add_dataset_arguments
from codeforces_tags.model.training import build_pipeline, cross_fit
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tuning.freeze import freeze, verify
from codeforces_tags.tuning.results import RESULT_FILES, TuningResults, save_tuning_results
from codeforces_tags.tuning.search import (
    best_params,
    create_study,
    run_worker,
    trial_scores,
    worker_commands,
)
from codeforces_tags.tuning.tracking import log_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, ``data`` and ``split``; ``journal`` for run, worker
        and export; ``trials`` and ``workers`` for run, ``trials`` and ``seed`` for worker,
        ``out`` for export; ``results`` and ``assets`` for freeze and verify, ``allow_dirty``
        for freeze; and ``tracking_uri`` and ``artifact_dir`` for run, export and freeze.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.tuning",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the workers, then log the study to MLflow")
    run.add_argument("--trials", type=int, default=TUNING_TRIALS, help="complete trials wanted")
    run.add_argument("--workers", type=int, default=TUNING_WORKERS, help="worker processes")
    worker = commands.add_parser("worker", help="add trials until the study has --trials")
    worker.add_argument("--trials", type=int, required=True, help="complete trials wanted")
    worker.add_argument("--seed", type=int, required=True, help="seed of the sampler")
    export = commands.add_parser("export", help="freeze the best trial's results, log the study")
    export.add_argument("--out", type=Path, default=TFIDF_BASELINE_DIR, help="results folder")
    freezing = commands.add_parser("freeze", help="freeze the tag model as tfidf, then log it")
    freezing.add_argument(
        "--allow-dirty", action="store_true", help="freeze from uncommitted code (recorded)"
    )
    verifying = commands.add_parser("verify", help="re-derive the frozen files of tfidf")
    for command in (run, worker, export):
        command.add_argument("--data", help="dataset folder (default: $CODEFORCES_DATA_DIR)")
        command.add_argument("--split", type=Path, default=SPLIT_FILE, help="frozen split")
        command.add_argument("--journal", type=Path, default=OPTUNA_JOURNAL, help="Optuna journal")
    for command in (freezing, verifying):
        add_dataset_arguments(command)  # --data, --split and --assets, as every local method's
        command.add_argument(
            "--results", type=Path, default=TFIDF_BASELINE_DIR, help="frozen results of the study"
        )
    for command in (run, export, freezing):
        add_store_arguments(command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of the tuning study.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status: 0, or 1 when run is asked for fewer than one trial or worker, when a
        worker failed, when the study has no complete trial, or when export scores the best
        configuration at another value than the study recorded. freeze and verify raise
        instead: on a refused dataset, a dirty tree, or a study or frozen file that the code
        does not reproduce.

    Raises:
        SystemExit: With status 2, from the parser, on an option it refuses, e.g. a
            ``--tracking-uri`` that is not ``sqlite:///``: before any work.
    """
    args = parse_args(argv)
    commands = {
        "run": _run,
        "worker": _worker,
        "export": _export,
        "freeze": freeze,
        "verify": verify,
    }
    return commands[args.command](args)


def _run(args: argparse.Namespace) -> int:
    """Create the study, run the workers in parallel, then log the study to MLflow."""
    for option, value in (("--trials", args.trials), ("--workers", args.workers)):
        if value < 1:
            print(f"{option} must be at least 1, not {value}.", file=sys.stderr)
            return 1
    study = create_study(args.journal)
    failed = _start_workers(args)
    if failed:
        print(
            f"{failed} of {args.workers} workers failed: see their errors above. "
            "Nothing was logged to MLflow.",
            file=sys.stderr,
        )
        return 1
    if _complete_trial_count(study) == 0:
        print(
            f"{args.journal} has no complete trial after the workers: nothing was logged to "
            "MLflow.",
            file=sys.stderr,
        )
        return 1
    run_id = log_study(study, tracking_uri=args.tracking_uri, artifact_dir=args.artifact_dir)
    print(
        f"{_complete_trial_count(study)} complete trials, best macro F1 {study.best_value:.4f} "
        f"(trial {study.best_trial.number}); MLflow run {run_id}."
    )
    return 0


def _start_workers(args: argparse.Namespace) -> int:
    """Start the worker processes side by side, wait for them, and count those that failed.

    If this process stops early, because starting a worker or waiting for one raised (Ctrl-C
    included), the workers still running are terminated rather than left writing to the
    journal.
    """
    commands = worker_commands(
        args.trials, args.workers, data_dir=args.data, split_file=args.split, journal=args.journal
    )
    with ExitStack() as stack:
        processes = []
        for command in commands:
            process = stack.enter_context(Popen(command))
            # Callbacks run last in, first out: this one runs before Popen's exit waits.
            stack.callback(_stop_if_running, process)
            processes.append(process)
        return sum(process.wait() != 0 for process in processes)


def _stop_if_running(process: Popen) -> None:
    """Terminate a worker that is still running when this process stops early."""
    if process.poll() is None:
        process.terminate()


def _worker(args: argparse.Namespace) -> int:
    """Add trials to the study, scored on the dev set, until it has ``args.trials``.

    BLAS runs on ``TUNING_BLAS_THREADS`` threads meanwhile, and on as many as before once the
    trials are done.
    """
    raw_problems, labels, folds = _dev_set(args)
    study = create_study(args.journal, seed=args.seed)
    with threadpool_limits(limits=TUNING_BLAS_THREADS, user_api="blas"):
        run_worker(study, args.trials, raw_problems, labels, folds)
    return 0


def _export(args: argparse.Namespace) -> int:
    """Freeze the results of the best trial, then log the study to MLflow.

    The best configuration is cross-fitted again on the dev set, which gives its out-of-fold
    scores and thresholds. Nothing is saved or logged unless the re-run scores the study's best
    value, to within ``TUNING_EXPORT_TOLERANCE``: otherwise the code, data or split differ from
    the study's. No tag model is fitted or logged: the tag model is a tagging method, logged
    through ``log_method``.
    """
    study = create_study(args.journal)
    if _complete_trial_count(study) == 0:
        print(
            f"{args.journal} has no complete trial: run python -m codeforces_tags.tuning run.",
            file=sys.stderr,
        )
        return 1
    raw_problems, labels, folds = _dev_set(args)
    params = best_params(study)
    out_of_fold = cross_fit(build_pipeline(params), raw_problems, labels, folds)
    evaluation = evaluate(labels, out_of_fold.predictions)
    if abs(evaluation.macro_f1 - study.best_value) > TUNING_EXPORT_TOLERANCE:
        print(
            f"The best configuration scores {evaluation.macro_f1:.6f} out of fold, which "
            f"differs from the study's best value {study.best_value:.6f}: the code, data or "
            "split differs from the study's. Nothing was saved or logged.",
            file=sys.stderr,
        )
        return 1
    results = TuningResults(
        params=params,
        thresholds=out_of_fold.thresholds,
        trials=_trials_table(study),
        importances=_importances(study),
        out_of_fold=evaluation.overall_f1().to_frame("f1"),
    )
    save_tuning_results(results, args.out)
    run_id = log_study(study, tracking_uri=args.tracking_uri, artifact_dir=args.artifact_dir)
    print(
        f"Saved {', '.join(RESULT_FILES)} to {args.out}: out-of-fold macro F1 "
        f"{evaluation.macro_f1:.4f}{_importances_note(results.importances)}. "
        f"Study logged to MLflow run {run_id}."
    )
    return 0


def _dev_set(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the raw dev problems (the pipeline preprocesses them), their labels and folds."""
    raw_problems = select_part(read_samples(args.data), load_split(args.split), "dev")
    return raw_problems, focus_labels(raw_problems), raw_problems["fold"]


def _complete_trial_count(study: optuna.Study) -> int:
    """Return the number of complete trials of the study."""
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


def _trials_table(study: optuna.Study) -> pd.DataFrame:
    """Return one row per trial, with the columns that ``TuningResults.trials`` describes."""
    trials = study.trials_dataframe(attrs=("number", "state", "value", "duration", "params"))
    trials["duration"] = trials["duration"].dt.total_seconds()
    scores = pd.DataFrame(
        [trial_scores(trial) for trial in study.get_trials(deepcopy=False)], index=trials.index
    )
    return trials.rename(columns={"duration": "duration_s"}).join(scores)


def _importances(study: optuna.Study) -> pd.Series:
    """Return the fANOVA importance of each parameter drawn in every trial, seeded.

    The importances share out the variance of the complete trials' values: there are none (an
    empty series) when fewer than two complete trials differ in value. fANOVA fails on values
    that are all equal, and would split a single trial's value evenly.
    """
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    importances = {}
    if len({trial.value for trial in complete}) > 1:
        evaluator = FanovaImportanceEvaluator(seed=TUNING_SEED)
        importances = get_param_importances(study, evaluator=evaluator)
    return pd.Series(importances, name="importance", dtype=float).rename_axis("parameter")


def _importances_note(importances: pd.Series) -> str:
    """Return what the export line adds about the importances: why there are none, if so."""
    if importances.empty:
        return "; no parameter importances, as fewer than two complete trials differ in value"
    return ""


if __name__ == "__main__":
    sys.exit(main())
