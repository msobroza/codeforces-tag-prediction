"""Tests for the search's stratified minibatches of learning-fold problems."""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.sampling import StratifiedSampler
from codeforces_tags.methods.typesafe_jev.settings import (
    SEARCH_MINIBATCH_SIZE,
    SEARCH_POSITIVES_PER_TAG,
    SEARCH_SEED,
)

# Rows of other folds (the selection and held-out folds): positive on every tag, so a sampler
# that drew from the labels' rows instead of its sample ids would pick them first.
OTHER_FOLD_IDS = list(range(5000, 5020))


def pool(count=120, positives=None):
    """Return learning-fold sample ids, in a shuffled order, and labels for them and other folds.

    Each focus tag is true on a random fifth of the learning problems (at least 3 of them),
    unless ``positives`` maps it to the number of learning problems it is true on.
    """
    ids = [1000 + 3 * number for number in range(count)]
    random.Random(1).shuffle(ids)
    values = np.random.default_rng(0).random((count, len(FOCUS_TAGS))) < 0.2
    values[:3, :] = True
    for column, tag in enumerate(FOCUS_TAGS):
        if positives is not None and tag in positives:
            values[:, column] = False
            values[: positives[tag], column] = True
    learning = pd.DataFrame(values.astype(int), index=ids, columns=list(FOCUS_TAGS))
    other = pd.DataFrame(1, index=OTHER_FOLD_IDS, columns=list(FOCUS_TAGS))
    labels = pd.concat([other, learning]).rename_axis("sample_id")
    return ids, labels


def drawn_labels(ids, labels, positions):
    """Return the labels of the problems at ``positions`` of ``ids``."""
    return labels.loc[[ids[position] for position in positions]]


def test_every_minibatch_holds_size_distinct_positions_with_per_tag_positives_of_each_tag():
    ids, labels = pool()
    sampler = StratifiedSampler(ids, labels, size=24, per_tag=2, seed=3)

    for iteration in range(30):
        positions = sampler.minibatch(iteration)

        assert len(positions) == 24
        assert positions == sorted(set(positions))
        assert all(0 <= position < len(ids) for position in positions)
        # Positions index sample_ids in its own (shuffled) order, not the labels' rows.
        counts = drawn_labels(ids, labels, positions).sum()
        assert (counts >= 2).all(), (iteration, counts.to_dict())


def test_a_minibatch_never_holds_a_problem_outside_sample_ids():
    ids, labels = pool()
    sampler = StratifiedSampler(ids, labels, size=40, per_tag=2)

    drawn = {ids[position] for iteration in range(50) for position in sampler.minibatch(iteration)}

    assert drawn <= set(ids)
    assert not drawn & set(OTHER_FOLD_IDS)


def test_a_tag_with_fewer_positives_than_per_tag_gives_all_of_them():
    # probabilities is true on one learning problem and games on none: every minibatch takes
    # all of their positives (one, and none), then fills up to its size.
    ids, labels = pool(positives={"probabilities": 1, "games": 0})
    only = [
        position for position, sample_id in enumerate(ids) if labels.at[sample_id, "probabilities"]
    ]
    sampler = StratifiedSampler(ids, labels, size=30, per_tag=3)

    for iteration in range(20):
        positions = sampler.minibatch(iteration)

        assert len(positions) == 30
        assert set(only) <= set(positions)
        counts = drawn_labels(ids, labels, positions).sum()
        assert (counts.drop(["probabilities", "games"]) >= 3).all(), iteration
        assert counts["games"] == 0
    assert len(only) == 1


def test_the_same_seed_and_iteration_draw_the_same_minibatch_and_others_differ():
    ids, labels = pool()
    first = StratifiedSampler(ids, labels, size=24, per_tag=2, seed=7)
    again = StratifiedSampler(ids, labels, size=24, per_tag=2, seed=7)
    other_seed = StratifiedSampler(ids, labels, size=24, per_tag=2, seed=8)

    # A replayed search builds a new sampler: it must draw what the first run drew.
    assert [first.minibatch(i) for i in range(10)] == [again.minibatch(i) for i in range(10)]
    assert first.minibatch(5) == first.minibatch(5)
    assert first.minibatch(5) != first.minibatch(6)
    assert first.minibatch(5) != other_seed.minibatch(5)


def test_next_minibatch_ids_draws_the_minibatch_of_gepas_current_iteration():
    ids, labels = pool()
    sampler = StratifiedSampler(ids, labels, size=24, per_tag=2)
    loader = list(range(len(ids)))  # GEPA's ListDataLoader: its ids are positions

    assert sampler.next_minibatch_ids(loader, SimpleNamespace(i=0)) == sampler.minibatch(0)
    assert sampler.next_minibatch_ids(loader, SimpleNamespace(i=4)) == sampler.minibatch(4)
    with pytest.raises(ValueError, match="the loader holds 119 problems, the sampler 120"):
        sampler.next_minibatch_ids(loader[:-1], SimpleNamespace(i=0))


def test_the_defaults_are_the_search_settings():
    ids, labels = pool()

    sampler = StratifiedSampler(ids, labels)

    assert (sampler.size, sampler.per_tag, sampler.seed) == (
        SEARCH_MINIBATCH_SIZE,
        SEARCH_POSITIVES_PER_TAG,
        SEARCH_SEED,
    )
    assert len(sampler.minibatch(0)) == SEARCH_MINIBATCH_SIZE


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"size": 15, "per_tag": 2}, "size 15 is smaller than the stratified draw of 16"),
        ({"size": 121, "per_tag": 2}, "size 121 is larger than the 120 problems to draw from"),
        ({"size": 10, "per_tag": -1}, "per_tag must not be negative, got -1"),
    ],
)
def test_construction_refuses_a_size_the_draw_cannot_meet(options, message):
    ids, labels = pool()

    with pytest.raises(ValueError, match=message):
        StratifiedSampler(ids, labels, **options)


def test_construction_refuses_repeated_ids_and_ids_without_labels():
    ids, labels = pool()

    with pytest.raises(ValueError, match="repeats a sample id"):
        StratifiedSampler([*ids, ids[0]], labels, size=24)
    with pytest.raises(ValueError, match=r"labels has no row for sample ids \[1\]"):
        StratifiedSampler([1, *ids], labels, size=24)
