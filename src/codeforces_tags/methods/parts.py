"""The dataset as a local method's commands read it: the split, the raw parts, the labels.

The ``freeze`` and ``verify`` of every local method (``label_prior``, ``tfidf``'s ``tuning
freeze`` and ``tuning verify``, the ModernBERT methods) read the raw problems once, split them
as the frozen split says, and score both parts against the focus labels of the same
preprocessing. Each of these commands takes the same options to name the dataset, the split and
the frozen methods, as do the F2LLM methods' ``freeze`` and ``verify``; their other commands read
the parts with ``read_parts`` too, from ``--data`` and ``--split`` alone. The LLM toolkit's
commands read the same parts without these options, and score every tag (docs/adr/0003).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from codeforces_tags.config import METHODS_ASSETS_DIR, SPLIT_FILE
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.splitting import load_split, select_part


class Parts(NamedTuple):
    """The dataset as a method's commands read it.

    Attributes:
        split: The frozen split: part, fold and group per problem.
        dev: The raw dev problems, with their fold and group.
        test: The raw test problems, with their fold and group.
        labels: The focus labels of every problem.
    """

    split: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    labels: pd.DataFrame


def read_parts(data_dir: str | os.PathLike[str] | None, split_path: Path) -> Parts:
    """Read the raw problems, split them, and label every problem with its focus tags.

    Args:
        data_dir: Dataset directory, resolved by ``read_samples`` (``$CODEFORCES_DATA_DIR`` or
            the default when None).
        split_path: The frozen split, e.g. the ``--split`` of a method's ``freeze``.

    Returns:
        The split, the raw dev and test problems (``select_part``) and the focus labels.
    """
    raw = read_samples(data_dir)
    split = load_split(split_path)
    return Parts(
        split=split,
        dev=select_part(raw, split, "dev"),
        test=select_part(raw, split, "test"),
        labels=focus_labels(raw_to_problems(raw)),
    )


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that name the dataset, the split and the frozen methods to a command.

    Args:
        parser: The command's parser; it gains ``--data`` (``read_samples``'s directory),
            ``--split`` (default ``config.SPLIT_FILE``) and ``--assets`` (default
            ``config.METHODS_ASSETS_DIR``).
    """
    parser.add_argument("--data", help="dataset folder (default: $CODEFORCES_DATA_DIR)")
    parser.add_argument("--split", type=Path, default=SPLIT_FILE, help="frozen split")
    parser.add_argument(
        "--assets", type=Path, default=METHODS_ASSETS_DIR, help="folder of the frozen methods"
    )
