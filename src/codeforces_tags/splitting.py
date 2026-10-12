"""Split the problems into dev and test, and dev into cross-validation folds.

Whole near-duplicate groups go to one side, so no near copy of a test problem is trained on.
Groups with a focus tag and groups without one are split separately (two strata), each by
iterative stratification: on the focus tags, their frequent pairs and the release year in the
first, on the release year alone in the second. Every fold then has about the same share of
each focus tag, of each frequent pair of them, of problems without any, and of each year.

The split is frozen in ``SPLIT_FILE`` with each problem's near-duplicate group, so that code
downstream reads the groups instead of grouping a subset again: ``freeze_split`` writes it once,
``load_split`` reads it back and ``select_part`` keeps the problems of one part.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from codeforces_tags.config import (
    SPLIT_FILE,
    SPLIT_FOLDS,
    SPLIT_MIN_PAIR_PROBLEMS,
    SPLIT_SEED,
    SPLIT_TEST_FOLD,
)
from codeforces_tags.data.labels import focus_labels, frequent_pairs

# The columns of the frozen split, after its sample_id index, and the parts it divides into.
SPLIT_COLUMNS = ("part", "fold", "group")
PARTS = ("dev", "test")


def release_years(problems: pd.DataFrame) -> pd.Series:
    """Return the year each problem was released.

    Args:
        problems: Preprocessed problems, with the ``prob_desc_created_at`` datetime.

    Returns:
        The year per problem.
    """
    return problems["prob_desc_created_at"].dt.year.rename("year")


def stratified_group_folds(
    labels: pd.DataFrame, groups: pd.Series, n_folds: int, seed: int
) -> pd.Series:
    """Assign whole groups to folds by iterative stratification of their labels.

    Iterative stratification (Sechidis, Tsoumakas and Vlahavas, "On the Stratification of
    Multi-Label Data", ECML PKDD 2011) takes the label with the fewest remaining items first,
    and puts each of its items in the fold that lacks that label most. A group's labels are the
    union of its members' labels, so a group counts once for every label any member has.

    Args:
        labels: 0/1 columns to balance across folds, one row per item.
        groups: Group id per item, aligned on ``labels``.
        n_folds: Number of folds.
        seed: Seed of the stratifier.

    Returns:
        The fold per item, from 0 to ``n_folds - 1``.

    Raises:
        ValueError: If there are fewer groups than folds.
    """
    group_labels = labels.groupby(groups.to_numpy()).max()
    if len(group_labels) < n_folds:
        raise ValueError(f"{len(group_labels)} groups cannot fill {n_folds} folds.")
    if group_labels.shape[1] == 1:  # the stratifier needs two columns: add the complement
        group_labels = group_labels.assign(other=1 - group_labels.iloc[:, 0])
    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    group_fold = pd.Series(0, index=group_labels.index)
    split = splitter.split(np.zeros(len(group_labels)), group_labels.to_numpy())
    for fold, (_, in_fold) in enumerate(split):
        group_fold.iloc[in_fold] = fold
    return pd.Series(group_fold.loc[groups.to_numpy()].to_numpy(), index=labels.index, name="fold")


def two_stage_folds(
    problems: pd.DataFrame, groups: pd.Series, n_folds: int = SPLIT_FOLDS, seed: int = SPLIT_SEED
) -> pd.Series:
    """Assign problems to folds, splitting the focus and no-focus strata separately.

    A group belongs to the focus stratum when any member has a focus tag. That stratum is
    balanced on the focus tags, the pairs of them shared by at least
    ``SPLIT_MIN_PAIR_PROBLEMS`` problems, and the release year; the no-focus stratum on the
    year alone. Each stratum needs at least ``n_folds`` groups, or is empty.

    Args:
        problems: Preprocessed problems, with ``tags`` and ``prob_desc_created_at``.
        groups: Near-duplicate group per problem, aligned on ``problems``.
        n_folds: Number of folds.
        seed: Seed of the stratifier.

    Returns:
        The fold per problem, from 0 to ``n_folds - 1``.
    """
    focus = focus_labels(problems)
    pairs = frequent_pairs(focus, SPLIT_MIN_PAIR_PROBLEMS)
    years = pd.get_dummies(release_years(problems), prefix="year", dtype=int)
    in_focus_stratum = focus.any(axis=1).groupby(groups).transform("any")
    folds = pd.Series(0, index=problems.index, name="fold")
    strata = ((in_focus_stratum, [focus, pairs, years]), (~in_focus_stratum, [years]))
    for stratum, label_blocks in strata:
        if not stratum.any():
            continue
        labels = pd.concat([block[stratum] for block in label_blocks], axis=1)
        folds[stratum] = stratified_group_folds(labels, groups[stratum], n_folds, seed)
    return folds


def make_split(problems: pd.DataFrame, groups: pd.Series, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Split problems into dev and test, then dev into cross-validation folds.

    Fold ``SPLIT_TEST_FOLD`` of ``two_stage_folds`` (whole groups, stratified in two strata) is
    the test set (one fold in ``SPLIT_FOLDS``); the same rule, applied to the rest, gives the
    dev folds. The frequent tag pairs are counted again on dev, so the dev folds balance the
    pairs frequent in dev.

    Args:
        problems: Preprocessed problems, indexed by ``sample_id``.
        groups: Near-duplicate group per problem, aligned on ``problems``.
        seed: Seed of the stratifier, for both splits.

    Returns:
        Columns ``part`` (``dev`` or ``test``), ``fold`` (the dev fold, -1 in test) and
        ``group`` (the near-duplicate group, as int64), one row per problem.
    """
    is_test = two_stage_folds(problems, groups, seed=seed) == SPLIT_TEST_FOLD
    dev_folds = two_stage_folds(problems[~is_test], groups[~is_test], seed=seed)
    return pd.DataFrame(
        {
            "part": np.where(is_test, "test", "dev"),
            "fold": dev_folds.reindex(problems.index, fill_value=-1),
            "group": groups.astype("int64"),
        },
        index=problems.index,
    )


def freeze_split(split: pd.DataFrame, path: Path = SPLIT_FILE) -> str:
    """Save the split once, then only check that it has not changed.

    The test set must never change, so an existing file is compared with ``split`` rather
    than overwritten.

    Args:
        split: The split from ``make_split``, indexed by ``sample_id``.
        path: CSV file of the frozen split.

    Returns:
        A line to print: the file was saved, or it matches ``split``.

    Raises:
        ValueError: If the file exists and holds another split.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        split[list(SPLIT_COLUMNS)].to_csv(path, index_label="sample_id")
        return f"saved {len(split):,} problems to {path.name}"
    if not load_split(path).equals(split[list(SPLIT_COLUMNS)]):
        raise ValueError(
            f"make_split no longer gives the frozen split in {path}: the data, the near-duplicate "
            "groups or the method changed. Delete the file only to freeze a new test set."
        )
    return f"{path.name} matches the split ({len(split):,} problems)"


def load_split(path: Path = SPLIT_FILE) -> pd.DataFrame:
    """Read the frozen split, and check that every near-duplicate group stays whole.

    Args:
        path: CSV written by ``freeze_split``, with ``sample_id``, ``part``, ``fold`` and
            ``group``.

    Returns:
        Columns ``part``, ``fold`` and ``group``, indexed by ``sample_id``.

    Raises:
        ValueError: If a column is missing, or if a group spans two parts or two folds.
    """
    split = pd.read_csv(path, index_col="sample_id")
    for column in SPLIT_COLUMNS:
        if column not in split.columns:
            raise ValueError(
                f"{path} has no {column!r} column. Delete it and re-run "
                "notebooks/03_train_test_split.ipynb, which writes it again."
            )
    for column in ("part", "fold"):
        values_per_group = split.groupby("group")[column].nunique()
        if (values_per_group > 1).any():
            raise ValueError(
                f"Near-duplicate group {values_per_group.idxmax()} is split across {column}s in "
                f"{path}: one of its problems would be trained on while another is scored."
            )
    return split[list(SPLIT_COLUMNS)]


def select_part(problems: pd.DataFrame, split: pd.DataFrame, part: str) -> pd.DataFrame:
    """Keep the problems of one part, with their fold and near-duplicate group.

    Args:
        problems: Problems indexed by ``sample_id``: raw (``read_samples()``, as tagging methods
            take them) or preprocessed (``load_dataset()``).
        split: The frozen split, from ``load_split``, over the same problems.
        part: ``dev`` or ``test``.

    Returns:
        The problems of ``part``, in the order of ``problems``, with the ``fold`` and ``group``
        columns of ``split`` appended.

    Raises:
        ValueError: If ``part`` is not a part, or if ``problems`` and ``split`` do not hold the
            same sample ids.
    """
    if part not in PARTS:
        raise ValueError(f"Unknown part {part!r}: expected one of {PARTS}.")
    not_in_split = problems.index.difference(split.index)
    not_in_problems = split.index.difference(problems.index)
    if len(not_in_split) or len(not_in_problems):
        raise ValueError(
            f"The problems and the split differ: {len(not_in_split)} problems are not in the "
            f"split, {len(not_in_problems)} split rows are not in the problems. Pass the full "
            "dataset, as load_dataset() returns it, with load_split()."
        )
    in_part = split["part"].reindex(problems.index) == part
    return problems[in_part].join(split[["fold", "group"]])
