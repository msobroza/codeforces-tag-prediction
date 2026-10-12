"""Fixtures for the model tests: the problems of ``raw_dev`` preprocessed, their labels and folds.

``raw_dev`` and ``focus_raw_dev``, the small dev sets whose words make focus tags learnable, are in
``tests/conftest.py``: the tracking tests fit a tag model on ``focus_raw_dev`` too.
"""

import pytest

from codeforces_tags.data.labels import multi_hot
from codeforces_tags.preprocessing.pipeline import preprocess

TAGS = ("graphs", "strings", "games")  # the focus tags of raw_dev's kinds, in FOCUS_TAGS order
FOLDS = 3


@pytest.fixture(name="problems")
def problems_fixture(raw_dev):
    """Return the raw dev problems preprocessed, as the feature blocks read them."""
    return preprocess(raw_dev)


@pytest.fixture(name="labels")
def labels_fixture(problems):
    """Return the 0/1 labels of the three focus tags the problems carry: graphs, strings, games."""
    return multi_hot(problems["tags"], TAGS)


@pytest.fixture(name="folds")
def folds_fixture(labels):
    """Deal the problems of each label combination to folds 0, 1, 2 in turn.

    Every fold then holds positives of every tag (five of each) and five no-focus problems.
    """
    rank = labels.groupby(list(labels.columns)).cumcount()
    return (rank % FOLDS).rename("fold")
