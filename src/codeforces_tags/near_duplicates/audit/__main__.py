"""Command line of the near-duplicate audit.

Run it with ``uv run``, adding ``--group audit`` to judge:

    python -m codeforces_tags.near_duplicates.audit build-pairs DIR [--recall-check]
    python -m codeforces_tags.near_duplicates.audit judge first_pass DIR
    python -m codeforces_tags.near_duplicates.audit judge second_pass_round_1 DIR
    python -m codeforces_tags.near_duplicates.audit judge second_pass_round_2 DIR
    python -m codeforces_tags.near_duplicates.audit merge DIR

``judge --dry-run`` writes the prompts without calling any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from codeforces_tags.data.loading import load_dataset
from codeforces_tags.near_duplicates.audit.pipeline import build_pairs, merge, run_pass
from codeforces_tags.near_duplicates.audit.prompts import PASSES
from codeforces_tags.near_duplicates.audit.verdicts import confirmed_pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments: ``command``, ``folder``, ``recall_check`` for ``build-pairs``,
        and the pass and ``dry_run`` for ``judge``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codeforces_tags.near_duplicates.audit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-pairs", help="write the pairs to judge, and their batches")
    build.add_argument(
        "--recall-check",
        action="store_true",
        help="also judge pairs below the threshold, as the recorded audit did",
    )
    judge = commands.add_parser("judge", help="run the LLM judges of one pass")
    judge.add_argument("pass_name", choices=PASSES)
    judge.add_argument("--dry-run", action="store_true", help="write the prompts only")
    commands.add_parser("merge", help="merge the judges' answers into verdicts")
    for command in commands.choices.values():
        command.add_argument("folder", type=Path, help="working folder of the audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run one step of the audit.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    if args.command == "build-pairs":
        pairs = build_pairs(load_dataset(), args.folder, recall_check=args.recall_check)
        print(f"{len(pairs)} pairs to judge in {args.folder}")
    elif args.command == "judge":
        prompts = run_pass(args.pass_name, args.folder, dry_run=args.dry_run)
        action = "written" if args.dry_run else "judged"
        print(f"{args.pass_name}: {len(prompts)} prompts {action} in {args.folder}")
    else:
        judgments = merge(args.folder)
        print(f"{len(confirmed_pairs(judgments))} duplicates in {args.folder}")


if __name__ == "__main__":
    main()
