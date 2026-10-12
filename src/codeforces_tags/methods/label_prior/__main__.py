"""Command line of the label prior, the worked example of a tagging method's commands.

    python -m codeforces_tags.methods.label_prior freeze [--allow-dirty]
    python -m codeforces_tags.methods.label_prior verify

Both take --data DIR (the dataset), --split FILE (the frozen split) and --assets DIR (the folder of
the frozen methods); freeze also takes --tracking-uri URI and --artifact-dir DIR (the MLflow
store). The defaults come from codeforces_tags.config.

freeze checks the dataset and that the code is committed (unless --allow-dirty, which the run
record records), predicts dev out of fold (cross_predict) and test once (the method fitted on all
of dev), and saves predictions.csv, run_record.json and the method folder, each once. It then logs
the frozen record and the method's scores to MLflow (log_method). A second run rewrites nothing:
it re-derives the predictions and checks the frozen files and the method folder against them,
then mirrors the frozen record to MLflow, which adds only what its store lacks.

verify, offline and writing nothing, checks that the frozen files are complete and consistent with
the split, that their recipe is the code's, that the re-derived predictions equal the frozen ones
and that the method folder predicts the frozen test rows, then prints the scores recomputed from
predictions.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from codeforces_tags.config import (
    FOCUS_TAGS,
)
from codeforces_tags.conformance import check_frozen_method
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    freeze_predictions,
    load_predictions,
    load_run_record,
    stack_parts,
)
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    load_method,
    save_method,
    sibling_methods,
)
from codeforces_tags.methods.label_prior.method import LabelPrior
from codeforces_tags.methods.label_prior.settings import NAME, SETTINGS
from codeforces_tags.methods.out_of_fold import cross_predict
from codeforces_tags.methods.parts import Parts, add_dataset_arguments, read_parts
from codeforces_tags.provenance import (
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    make_run_record,
    sha256_json,
)
from codeforces_tags.tracking.environment import add_store_arguments
from codeforces_tags.tracking.methods import log_method, part_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, ``data``, ``split`` and ``assets``; for freeze also
        ``allow_dirty``, ``tracking_uri`` and ``artifact_dir``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.methods.label_prior",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze the files once, then log to MLflow")
    freeze.add_argument(
        "--allow-dirty", action="store_true", help="freeze from uncommitted code (recorded)"
    )
    add_store_arguments(freeze)
    commands.add_parser("verify", help="re-derive the frozen files and report their scores")
    for command in commands.choices.values():
        add_dataset_arguments(command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of the label prior.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status, 0; a refused dataset, a dirty tree or a frozen file that the code no
        longer reproduces raises.
    """
    args = parse_args(argv)
    commands = {"freeze": _freeze, "verify": _verify}
    return commands[args.command](args)


def recipe_sha256() -> str:
    """Hash what defines the method: its name, its tags and its settings.

    Not its fitted prior, which is an outcome of the data (docs/adr/0004).

    Returns:
        The sha256 hex digest, the ``recipe_sha256`` of its run record.
    """
    return sha256_json({"method": NAME, "tags": list(LabelPrior().tags), "settings": SETTINGS})


def supersedes(assets: Path) -> list[str]:
    """Name the sibling methods whose predictions are committed, as the freeze records them.

    A sibling is another registered method of this package (``sibling_methods``). Siblings
    frozen before either is committed give ``[]`` (docs/adr/0004).

    Args:
        assets: The folder of the frozen methods.

    Returns:
        The sorted names of the siblings whose ``predictions.csv`` is tracked by git and
        unchanged since ``HEAD``.
    """
    return sorted(
        other
        for other in sibling_methods(NAME, METHODS)
        if committed_unchanged(assets / other / PREDICTIONS_FILE)
    )


def _derive(parts: Parts) -> tuple[pd.DataFrame, LabelPrior]:
    """Predict dev out of fold and test once, as the method is frozen.

    Returns:
        The predictions indexed by ``(part, sample_id)``, and the method fitted on all of dev
        (the one that predicts test, and that is saved).
    """
    dev_labels = parts.labels.loc[parts.dev.index]
    out_of_fold = cross_predict(LabelPrior, parts.dev, dev_labels, parts.dev["fold"])
    method = LabelPrior().fit(parts.dev, dev_labels)
    return stack_parts(dev=out_of_fold, test=method.predict(parts.test)), method


def _check_folder(folder: Path, test: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Fail unless the method folder predicts the frozen test rows, exactly."""
    if not load_method(folder).predict(test).equals(frozen):
        raise ValueError(
            f"The method folder {folder} predicts other tags for the test problems than the "
            f"frozen {PREDICTIONS_FILE}: the folder and the frozen files come from different fits."
        )


def _freeze_folder(
    method: LabelPrior, folder: Path, test: pd.DataFrame, frozen: pd.DataFrame
) -> str:
    """Save the method folder once; then check that it predicts the frozen test rows."""
    saved = not (folder / METHOD_FILE).exists()
    if saved:
        save_method(method, folder)
    _check_folder(folder, test, frozen)
    if saved:
        return f"saved the method folder {folder}"
    return f"the method folder {folder} predicts the frozen test rows"


def _freeze(args: argparse.Namespace) -> int:
    """Freeze the predictions, the run record and the method folder, then log to MLflow."""
    check_dataset(args.data)
    check_clean_tree(args.allow_dirty)
    parts = read_parts(args.data, args.split)
    predictions, method = _derive(parts)
    directory = args.assets / NAME
    record = make_run_record(
        NAME,
        recipe_sha256(),
        allow_dirty=args.allow_dirty,
        requirements=METHODS[NAME].requirements,
        method_entries={
            "settings": SETTINGS,
            "dev_in_sample": False,
            "supersedes": supersedes(args.assets),
        },
        data_dir=args.data,
        split_path=args.split,
    )
    print(freeze_predictions(predictions, directory, record, score_atol=METHODS[NAME].score_atol))
    folder = directory / MODEL_FOLDER
    print(_freeze_folder(method, folder, parts.test, predictions.loc["test"]))
    # What is frozen, not the record made now: a second run's has another created_at.
    model_id = log_method(
        folder,
        load_run_record(directory),
        part_metrics(parts.labels, predictions, FOCUS_TAGS),
        split=parts.split,
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
    )
    print(f"logged {NAME} to MLflow: model {model_id}")
    return 0


def _check_recipe(directory: Path) -> None:
    """Fail unless the frozen run record holds the recipe of the code."""
    stored = load_run_record(directory).recipe_sha256
    if stored != recipe_sha256():
        raise ValueError(
            f"{directory} was frozen with another recipe ({stored}, not {recipe_sha256()}): the "
            f"settings of {NAME} changed since."
        )


def _frozen_predictions(directory: Path, parts: Parts) -> pd.DataFrame:
    """Return the frozen predictions, after checking that re-deriving them gives the same."""
    frozen = load_predictions(directory)
    derived, _ = _derive(parts)
    if not frozen.equals(derived):
        raise ValueError(
            f"The re-derived predictions differ from the frozen {PREDICTIONS_FILE} in "
            f"{directory}: the code, data or split changed since they were frozen."
        )
    return frozen


def _verify(args: argparse.Namespace) -> int:
    """Check the frozen files against the dataset and the code, and report their scores."""
    check_dataset(args.data)
    parts = read_parts(args.data, args.split)
    check_frozen_method(NAME, assets_dir=args.assets, split=parts.split)
    print(f"the frozen files of {NAME} are complete and consistent with the split")
    directory = args.assets / NAME
    _check_recipe(directory)
    frozen = _frozen_predictions(directory, parts)
    print(f"re-derived {len(frozen):,} predictions: they equal the frozen ones")
    folder = directory / MODEL_FOLDER
    _check_folder(folder, parts.test, frozen.loc["test"])
    print(f"the method folder {folder} predicts the frozen test rows")
    for dataset, metrics in part_metrics(parts.labels, frozen, FOCUS_TAGS).items():
        print(f"{dataset}: macro F1 {metrics['macro_f1']:.4f}, micro F1 {metrics['micro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
