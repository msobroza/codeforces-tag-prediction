"""Command line of the evaluation package.

    python -m codeforces_tags.evaluation slices [--data DIR] [--split FILE] [--out FILE]

slices refuses any dataset but the challenge's (check_dataset), finds the self-identifying
solutions of every problem (contamination.py) and saves them to
assets/slices/self_identifying_solutions.csv the first time; later runs only check the file,
and fail if the data, the split or the rule changed. The defaults come from
codeforces_tags.config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from codeforces_tags.config import SELF_IDENTIFYING_FILE, SPLIT_FILE
from codeforces_tags.data.loading import load_dataset
from codeforces_tags.evaluation.contamination import (
    RULE_SEPARATOR,
    SELF_IDENTIFYING_RULES,
    freeze_slice,
    self_identifying_solutions,
)
from codeforces_tags.provenance import check_dataset
from codeforces_tags.splitting import PARTS, load_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, and ``data``, ``split`` and ``out`` for slices.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    slices = commands.add_parser("slices", help="freeze the self-identifying solutions once")
    slices.add_argument("--data", help="dataset folder (default: $CODEFORCES_DATA_DIR)")
    slices.add_argument("--split", type=Path, default=SPLIT_FILE, help="frozen split")
    slices.add_argument("--out", type=Path, default=SELF_IDENTIFYING_FILE, help="slice file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command of the evaluation package.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit status, 0; a refused dataset or a changed slice raises ``ValueError``.
    """
    args = parse_args(argv)
    commands = {"slices": _slices}
    return commands[args.command](args)


def _slices(args: argparse.Namespace) -> int:
    """Check the dataset, then freeze the self-identifying slice or check the frozen one."""
    check_dataset(args.data)
    slice_ = self_identifying_solutions(load_dataset(args.data), load_split(args.split))
    print(freeze_slice(slice_, args.out))
    print(_summary(slice_))
    return 0


def _summary(slice_: pd.DataFrame) -> str:
    """Count the slice's problems per part and per rule (a problem can match several rules)."""
    parts = slice_["part"].value_counts()
    rules = slice_["rules"].str.split(RULE_SEPARATOR).explode().value_counts()
    by_part = ", ".join(f"{parts.get(part, 0)} {part}" for part in PARTS)
    by_rule = ", ".join(f"{rule} {rules.get(rule, 0)}" for rule in SELF_IDENTIFYING_RULES)
    return f"{len(slice_)} problems ({by_part}); by rule: {by_rule}"


if __name__ == "__main__":
    sys.exit(main())
