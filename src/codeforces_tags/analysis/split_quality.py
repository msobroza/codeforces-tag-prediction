"""Measure how good a dev/test split is, and compare the split used with random ones."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from codeforces_tags.config import (
    SPLIT_FOLDS,
    SPLIT_MIN_PAIR_PROBLEMS,
    SPLIT_TEST_FOLD,
)
from codeforces_tags.data.labels import focus_labels, frequent_pairs
from codeforces_tags.splitting import release_years, two_stage_folds

# Picks the test set: (problems, near-duplicate groups, seed) -> True for test problems.
TestPicker = Callable[[pd.DataFrame, pd.Series, int], pd.Series]


def random_problem_test(problems: pd.DataFrame, groups: pd.Series, seed: int) -> pd.Series:
    """Pick one problem in ``SPLIT_FOLDS`` at random for test, ignoring the groups.

    Args:
        problems: Preprocessed problems.
        groups: Near-duplicate group per problem (unused: this split ignores them).
        seed: Seed of the random draw.

    Returns:
        True for the test problems.
    """
    del groups
    order = np.random.default_rng(seed).permutation(len(problems))
    return pd.Series(order % SPLIT_FOLDS == SPLIT_TEST_FOLD, index=problems.index)


def random_group_test(problems: pd.DataFrame, groups: pd.Series, seed: int) -> pd.Series:
    """Pick one near-duplicate group in ``SPLIT_FOLDS`` at random for test.

    Args:
        problems: Preprocessed problems.
        groups: Near-duplicate group per problem.
        seed: Seed of the random draw.

    Returns:
        True for the test problems.
    """
    ids = groups.unique()
    order = np.random.default_rng(seed).permutation(len(ids))
    group_in_test = pd.Series(order % SPLIT_FOLDS == SPLIT_TEST_FOLD, index=ids)
    return pd.Series(group_in_test.loc[groups.to_numpy()].to_numpy(), index=problems.index)


def stratified_group_test(problems: pd.DataFrame, groups: pd.Series, seed: int) -> pd.Series:
    """Pick the test fold of the split used: whole groups, stratified in two strata.

    Args:
        problems: Preprocessed problems.
        groups: Near-duplicate group per problem.
        seed: Seed of the stratifier.

    Returns:
        True for the test problems.
    """
    return two_stage_folds(problems, groups, seed=seed) == SPLIT_TEST_FOLD


# The split methods compared, from the simplest to the one used. Each name says what is drawn
# (a problem or a near-duplicate group) and how (at random, or stratified).
SPLIT_METHODS: dict[str, TestPicker] = {
    "random by problem": random_problem_test,
    "random by group": random_group_test,
    "stratified by group": stratified_group_test,
}

# Each measure of split_quality() and which value is better: lower (↓) or higher (↑). Higher
# test problems per tag is better up to the rarest tag's expected count; above it, the tag is
# over-represented, which the tag share deviations catch. The test share is a control rather
# than a quality: every method aims at one fold in SPLIT_FOLDS.
BETTER = {
    "leaked duplicate pairs": "↓",
    "max tag share deviation (%)": "↓",
    "mean tag share deviation (%)": "↓",
    "max tag-pair share deviation (%)": "↓",
    "mean tag-pair share deviation (%)": "↓",
    "min test problems per tag": "↑",
    "no-focus share deviation (%)": "↓",
    "year mix distance (%)": "↓",
    "test share (%)": f"≈ {100 / SPLIT_FOLDS:.0f}",
}
# Table labels: each measure followed by its better direction, e.g. "leaked duplicate pairs ↓".
MEASURE_LABELS = {name: f"{name} {better}" for name, better in BETTER.items()}


def leaked_pairs(groups: pd.Series, is_test: pd.Series) -> int:
    """Count the near-duplicate pairs split between dev and test.

    Args:
        groups: Near-duplicate group per problem.
        is_test: True for the test problems, aligned on ``groups``.

    Returns:
        The number of pairs with one problem in dev and the other in test.
    """
    in_test = is_test.groupby(groups).sum()
    size = groups.value_counts().reindex(in_test.index)
    return int((in_test * (size - in_test)).sum())


def share_deviation(subset_share: float | pd.Series, share: float | pd.Series) -> float | pd.Series:
    """Measure how far a share in a subset is from the share in all problems.

    Args:
        subset_share: Share of problems with a property (a tag, say) in the subset.
        share: Share of problems with that property in all problems; not zero.

    Returns:
        The absolute difference, in percent of ``share``: 10 means the subset has 10% more or
        10% fewer such problems than its size calls for.
    """
    return 100 * abs(subset_share - share) / share


def split_quality(problems: pd.DataFrame, groups: pd.Series, is_test: pd.Series) -> pd.Series:
    """Measure how well a test set represents the whole dataset.

    Args:
        problems: Preprocessed problems, with ``tags`` and ``prob_desc_created_at``.
        groups: Near-duplicate group per problem.
        is_test: True for the test problems.

    Returns:
        The measures, named as in ``BETTER``, which says whether lower or higher is better:
        - ``leaked duplicate pairs``: near-duplicate pairs with one problem in dev and the
          other in test;
        - ``max tag share deviation (%)``: the largest ``share_deviation`` of a focus tag's
          share in test, over the focus tags present in the data;
        - ``mean tag share deviation (%)``: the same deviation, averaged over those tags;
        - ``max tag-pair share deviation (%)``: the largest ``share_deviation`` of the share of
          a pair of focus tags in test, over the pairs shared by at least
          ``SPLIT_MIN_PAIR_PROBLEMS`` problems (NaN when there is none);
        - ``mean tag-pair share deviation (%)``: the same deviation, averaged over those pairs;
        - ``min test problems per tag``: test problems of the focus tag with the fewest of them;
        - ``no-focus share deviation (%)``: the ``share_deviation`` of the share of problems
          with no focus tag;
        - ``year mix distance (%)``: the total variation distance between the release years
          of test and of all problems, in percent: the smallest share of test problems whose
          year would have to change for test to have the years of all problems (0 when
          identical, 100 when no year is shared);
        - ``test share (%)``: the share of problems in test.
    """
    focus = focus_labels(problems)
    focus = focus.loc[:, focus.any()]  # focus tags present in the data
    tag_deviations = share_deviation(focus[is_test].mean(), focus.mean())
    pairs = frequent_pairs(focus, SPLIT_MIN_PAIR_PROBLEMS)
    pair_deviations = share_deviation(pairs[is_test].mean(), pairs.mean())
    no_focus = ~focus.astype(bool).any(axis=1)
    years = release_years(problems)
    year_shares = pd.concat(
        [years.value_counts(normalize=True), years[is_test].value_counts(normalize=True)], axis=1
    ).fillna(0)
    measures = {
        "leaked duplicate pairs": leaked_pairs(groups, is_test),
        "max tag share deviation (%)": tag_deviations.max(),
        "mean tag share deviation (%)": tag_deviations.mean(),
        "max tag-pair share deviation (%)": pair_deviations.max(),
        "mean tag-pair share deviation (%)": pair_deviations.mean(),
        "min test problems per tag": int(focus[is_test].sum().min()),
        "no-focus share deviation (%)": share_deviation(no_focus[is_test].mean(), no_focus.mean()),
        "year mix distance (%)": 50 * (year_shares.iloc[:, 0] - year_shares.iloc[:, 1]).abs().sum(),
        "test share (%)": 100 * is_test.mean(),
    }
    return pd.Series(measures)[list(BETTER)]  # a KeyError as soon as a name drifts from BETTER


def compare_split_methods(
    problems: pd.DataFrame,
    groups: pd.Series,
    seeds: Iterable[int],
    methods: dict[str, TestPicker] | None = None,
) -> pd.DataFrame:
    """Measure every split method on the same seeds.

    Args:
        problems: Preprocessed problems.
        groups: Near-duplicate group per problem.
        seeds: Seeds to draw a split with, the same for every method.
        methods: Test pickers by name; defaults to ``SPLIT_METHODS``.

    Returns:
        One row per method and seed, with the columns of ``split_quality``.
    """
    seeds = list(seeds)  # iterated once per method
    rows = {
        (name, seed): split_quality(problems, groups, pick(problems, groups, seed))
        for name, pick in (methods or SPLIT_METHODS).items()
        for seed in seeds
    }
    return pd.DataFrame(rows).T.rename_axis(["method", "seed"])
