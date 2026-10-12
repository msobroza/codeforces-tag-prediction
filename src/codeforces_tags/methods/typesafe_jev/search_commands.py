"""The handlers of the search's commands: tune, export and log-search (docs/adr/0010).

``__main__`` registers their options (``add_search_commands``), runs their gates (the dataset,
the clean tree, the committed split and slice, the seed's committed frozen files) and hands each
handler what the gates read:

- ``tune`` runs the search over the words of jev-statement-code-gepa (``tuning.run_search``), or
  replays it from its caches in ``experiments/jev-statement-code-gepa/search/`` and goes on, and
  logs it live to MLflow unless ``--no-mlflow``. It returns 1 when the search is interrupted
  inside an iteration: after a transient error, running it again resumes; after its cost cap was
  spent inside a call, only running it again with a higher ``--max-cost-usd`` does. It refuses
  once the search is exported (``tuning/params.json`` exists), whose words are final.
- ``export`` freezes the finished search's record in the method's ``tuning/`` once
  (``export.export_search``); a second run only compares. Its ``--method`` also takes a
  calibrated twin, whose export ``__main__`` hands to ``platt_commands`` (docs/adr/0011).
- ``log_search`` replays that record into an MLflow store, with no key and no paid call, and
  writes the search run's id to the search folder's run.json.

It reaches gepa through ``tuning`` and mlflow through ``search_tracking``: only ``__main__``
imports it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

import pandas as pd

from codeforces_tags.config import EXPERIMENTS_DIR
from codeforces_tags.llm.client import LlmAnswer, TransientApiError
from codeforces_tags.methods import method_dir
from codeforces_tags.methods.typesafe_jev import client, search_tracking
from codeforces_tags.methods.typesafe_jev.export import export_search
from codeforces_tags.methods.typesafe_jev.jev_cache import CostCapSpent, SearchInterrupted
from codeforces_tags.methods.typesafe_jev.search_record import dataset_texts, read_search_record
from codeforces_tags.methods.typesafe_jev.settings import (
    GEPA_NAME,
    PARAMS_FILE,
    PLATT_BASES,
    RUN_ID_FILE,
    SEARCH_EXPERIMENT,
    SEARCH_MAX_COST_USD,
    SEARCH_MAX_ITERATIONS,
    TUNING_FOLDER,
)
from codeforces_tags.methods.typesafe_jev.tuning import (
    SearchOutcome,
    run_search,
    search_data,
    search_folder,
)
from codeforces_tags.tracking.environment import add_store_arguments

# The command line of the package, which the messages name.
_PROGRAM = "python -m codeforces_tags.methods.typesafe_jev"


def add_search_commands(
    commands: argparse._SubParsersAction, cost_cap: Callable[[str], float]
) -> tuple[argparse.ArgumentParser, ...]:
    """Add the parsers of tune, export and log-search, each with its options.

    Args:
        commands: The subparsers of ``__main__``'s parser.
        cost_cap: The argparse type of a cost cap (``__main__``'s, which ``run`` uses too).

    Returns:
        The parsers of tune, export and log-search; tune and log-search take ``--method`` with
        the searched method as its only choice, and export the searched method or a calibrated
        twin (``settings.PLATT_BASES``, docs/adr/0011).
    """
    tune_parser = commands.add_parser("tune", help="search the words of the method (resumable)")
    tune_parser.add_argument(
        "--max-cost-usd",
        type=cost_cap,
        default=SEARCH_MAX_COST_USD,
        help="stop once the search's answers cost this much, Jev and Luna together (USD)",
    )
    tune_parser.add_argument(
        "--max-iterations",
        type=_iterations,
        default=SEARCH_MAX_ITERATIONS,
        help="stop after this many iterations",
    )
    tune_parser.add_argument(
        "--no-mlflow", action="store_true", help="log nothing to MLflow while searching"
    )
    add_store_arguments(tune_parser)
    export_parser = commands.add_parser(
        "export", help="freeze the search's record, or a calibrated twin's, in tuning/"
    )
    log_parser = commands.add_parser("log-search", help="replay the search's record into MLflow")
    add_store_arguments(log_parser)
    for parser in (tune_parser, export_parser):
        parser.add_argument("--allow-dirty", action="store_true", help="accept uncommitted code")
    for parser in (tune_parser, log_parser):
        parser.add_argument(
            "--method", required=True, choices=(GEPA_NAME,), help="the searched Jev method"
        )
    export_parser.add_argument(
        "--method",
        required=True,
        choices=(GEPA_NAME, *PLATT_BASES),
        help="the searched Jev method, or a calibrated twin (docs/adr/0011)",
    )
    return tune_parser, export_parser, log_parser


def _iterations(text: str) -> int:
    """Read --max-iterations for argparse: a number of iterations, at least 1."""
    number = int(text)
    if number >= 1:
        return number
    raise argparse.ArgumentTypeError(f"the search needs at least 1 iteration, not {text}")


def tune(args: argparse.Namespace, raw_dev: pd.DataFrame, seed_answers: Sequence[LlmAnswer]) -> int:
    """Run the search, or replay it from its caches and go on, then print how it stopped.

    The live MLflow log, unless ``--no-mlflow``, is closed however the search stops.

    Args:
        args: ``max_cost_usd``, ``max_iterations``, ``no_mlflow``, ``tracking_uri`` and
            ``artifact_dir``.
        raw_dev: The dev part's raw problems with their fold (``select_part``).
        seed_answers: The committed dev answers of jev-statement-code, which the seed's cache
            starts from.

    Returns:
        0 when the search stopped at its iteration or cost cap between iterations; 1 when it
        was interrupted inside one: by a transient error of Jev or Luna, or a run of Jev's
        failed answers (run it again), or by its cost cap spent inside a call (run it again
        with a higher ``--max-cost-usd``).

    Raises:
        ValueError: If the search was exported already.
    """
    _check_not_exported()
    folder = search_folder(EXPERIMENTS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    data = search_data(raw_dev)
    live = None
    if not args.no_mlflow:
        live = search_tracking.LiveSearchLog(
            tracking_uri=args.tracking_uri,
            artifact_dir=args.artifact_dir,
            run_file=folder / RUN_ID_FILE,
        )
    try:
        outcome = run_search(
            data,
            folder,
            max_cost_usd=args.max_cost_usd,
            max_iterations=args.max_iterations,
            committed_seed_answers=seed_answers,
            jev_session=client.jev_session,
            live=live,
        )
    except (SearchInterrupted, TransientApiError) as error:
        print(
            f"{folder}: the search was interrupted inside an iteration, on "
            f"{type(error).__name__}: {error} {_how_to_go_on(error)}",
            file=sys.stderr,
        )
        return 1
    finally:
        if live is not None:
            live.close()
    print("\n".join(_outcome_lines(outcome)))
    return 0


def _how_to_go_on(error: Exception) -> str:
    """Say how a rerun of tune goes on after ``error`` interrupted the search."""
    if isinstance(error, CostCapSpent):
        # The same cap would stop the rerun at the same point, before it sends anything.
        return (
            "Run tune again with a higher --max-cost-usd: it replays every cached answer for "
            "free, then goes on."
        )
    return "Run tune again: it replays every cached answer for free, then goes on."


def _check_not_exported() -> None:
    """Refuse a search once its record is exported: the exported words are final."""
    path = method_dir(GEPA_NAME) / TUNING_FOLDER / PARAMS_FILE
    if path.exists():
        raise ValueError(
            f"{path} exists: the search was exported, so its words are final (docs/adr/0010). "
            "log-search replays the exported search into MLflow; another search is a new "
            "method under a new name."
        )


def _outcome_lines(outcome: SearchOutcome) -> list[str]:
    """Say how the search stopped, what it cost and which command comes next."""
    cost = outcome.cost_usd
    return [
        f"the search stopped at its {outcome.stop_reason} cap after {outcome.iterations} "
        f"iterations, for ${cost['total']:.2f} (Jev ${cost['jev']:.2f}, Luna "
        f"${cost['luna']:.2f}); GEPA's best candidate is {outcome.best_index}.",
        f"Next: {_PROGRAM} export --method {GEPA_NAME}",
    ]


def export(raw_dev: pd.DataFrame, problems: pd.DataFrame, seed_scores: pd.DataFrame) -> int:
    """Freeze the finished search's record in the searched method's tuning/ once; print it.

    Args:
        raw_dev: The dev part's raw problems with their fold (``select_part``).
        problems: Every problem of the dataset, dev and test, preprocessed: the texts no kept
            word set may copy.
        seed_scores: The frozen dev probabilities of jev-statement-code, by sample id.

    Returns:
        0, once tuning/ holds the record, written now or found byte for byte.

    Raises:
        ValueError: As ``export.export_search``.
    """
    out = method_dir(GEPA_NAME) / TUNING_FOLDER
    existed = out.exists()
    export_search(
        search_folder(EXPERIMENTS_DIR),
        search_data(raw_dev),
        seed_scores,
        dataset_texts(problems),
        out,
    )
    params = json.loads((out / PARAMS_FILE).read_text(encoding="utf-8"))
    state = "holds the same record" if existed else "now holds the record"
    print(
        f"{out} {state} of the search: candidate {params['chosen']} is chosen, with a mean "
        f"score of {params['chosen_selection_mean']:.4f} on the selection fold, against the "
        f"seed's {params['seed_selection_mean']:.4f}."
    )
    if not existed:
        print(
            f"Commit it, then answer the dev part with `{_PROGRAM} run --method {GEPA_NAME} "
            "--part dev`."
        )
    return 0


def log_search(args: argparse.Namespace) -> int:
    """Replay the searched method's frozen tuning/ record into an MLflow store.

    ``search_tracking.log_search`` finds the search run by its search tag
    (``search_tracking.SEARCH_TAG``) in the store, or creates it, and writes its id to the search
    folder's run.json: after a store is rebuilt, run.json names the new store's run, not the one
    the live log made in the old store.

    Args:
        args: ``tracking_uri`` and ``artifact_dir``, the store.

    Returns:
        0, once the store holds the search's runs, metrics and prompt versions.
    """
    record = read_search_record(method_dir(GEPA_NAME) / TUNING_FOLDER)
    run_id = search_tracking.log_search(
        record,
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
        run_file=search_folder(EXPERIMENTS_DIR) / RUN_ID_FILE,
    )
    print(
        f"logged the search's record to MLflow: run {run_id} of the experiment {SEARCH_EXPERIMENT}"
    )
    return 0
