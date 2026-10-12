"""Tests for the group bootstrap."""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.evaluation.bootstrap import percentile_interval, resample_weights


@pytest.fixture(name="groups")
def groups_fixture():
    """Return 6 rows in 4 groups: rows 0-1 and 2-3 are pairs."""
    return pd.Series([7, 7, 3, 3, 5, 9])


def test_resample_weights_gives_one_count_per_resample_and_row(groups):
    weights = resample_weights(groups, resamples=50, seed=0)

    assert weights.shape == (50, 6)
    assert weights.dtype == np.int64


def test_resample_weights_draws_each_group_whole(groups):
    weights = resample_weights(groups, resamples=200, seed=0)

    assert (weights[:, 0] == weights[:, 1]).all()
    assert (weights[:, 2] == weights[:, 3]).all()


def test_resample_weights_draws_as_many_groups_as_there_are(groups):
    weights = resample_weights(groups, resamples=200, seed=0)

    one_row_per_group = weights[:, [0, 2, 4, 5]]
    assert (one_row_per_group.sum(axis=1) == 4).all()


def test_resample_weights_draws_groups_uniformly(groups):
    weights = resample_weights(groups, resamples=20_000, seed=0)

    assert np.allclose(weights.mean(axis=0), 1, atol=0.03)  # each group drawn once on average


def test_resample_weights_depends_on_the_seed_only(groups):
    first = resample_weights(groups, resamples=20, seed=1)

    assert np.array_equal(first, resample_weights(groups, resamples=20, seed=1))
    assert not np.array_equal(first, resample_weights(groups, resamples=20, seed=2))


def test_resample_weights_of_no_row_draw_nothing():
    weights = resample_weights(pd.Series([], dtype=np.int64), resamples=5, seed=0)

    assert weights.shape == (5, 0)
    assert weights.dtype == np.int64


def test_resample_weights_rejects_a_row_without_a_group():
    with pytest.raises(ValueError, match="Every row needs a group"):
        resample_weights(pd.Series([0, None, 1]), resamples=10, seed=0)


def test_percentile_interval_cuts_both_tails_along_the_first_axis():
    values = np.column_stack([np.arange(101.0), np.arange(101.0) * 2])

    low, high = percentile_interval(values, confidence=0.9)

    assert np.allclose(low, [5, 10])
    assert np.allclose(high, [95, 190])
