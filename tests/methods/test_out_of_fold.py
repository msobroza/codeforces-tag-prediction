"""Tests for the out-of-fold predictions of fitted and composite tagging methods."""

import numpy as np
import pandas as pd
import pytest

from codeforces_tags.methods.out_of_fold import CompositeMethod, FittedMethod, cross_predict

TAGS = ("math", "games")


class Majority:
    """A fitted 0/1 method: predicts, for every problem, the tags of most fit problems."""

    def __init__(self):
        """Start unfitted."""
        self.fit_ids = None
        self.predicted = None

    @property
    def tags(self):
        """The output columns, in order."""
        return TAGS

    def fit(self, twelve_problems, labels):
        """Keep the tags positive for at least half of ``labels``; a method is fitted once."""
        assert self.fit_ids is None
        self.fit_ids = set(twelve_problems.index)
        self.predicted = (labels.mean() >= 0.5).astype(int)
        return self

    def predict(self, twelve_problems):
        """Predict the kept tags for every problem, none of which was fitted on."""
        assert not self.fit_ids & set(twelve_problems.index)
        return pd.DataFrame(
            [self.predicted.to_list()] * len(twelve_problems),
            index=twelve_problems.index,
            columns=TAGS,
        )


class PositiveTags:
    """A fitted 0/1 method whose tags are the tags with a positive in its fit data."""

    def __init__(self):
        """Start unfitted, with no tag."""
        self.tags = ()

    def fit(self, twelve_problems, labels):
        """Keep the tags with a positive."""
        assert labels.index.equals(twelve_problems.index)
        self.tags = tuple(labels.columns[labels.max() == 1])
        return self

    def predict(self, twelve_problems):
        """Predict every kept tag."""
        return pd.DataFrame(1, index=twelve_problems.index, columns=list(self.tags))


class Stack:
    """A composite method: predicts a tag where its base's score is above the fit rows' mean."""

    def __init__(self):
        """Start unfitted."""
        self.cutoffs = None

    @property
    def tags(self):
        """The output columns, in order."""
        return TAGS

    def fit_from(self, features, labels):
        """Keep each tag's mean base score over the fit rows."""
        assert features.index.equals(labels.index)
        self.cutoffs = features[list(TAGS)].mean()
        return self

    def predict_from(self, features):
        """Predict the tags whose base score is above the fit rows' mean."""
        return features[list(TAGS)].gt(self.cutoffs).astype(int)


@pytest.fixture(name="twelve_problems")
def twelve_problems_fixture():
    """Return 12 raw problems, indexed by sample id."""
    return pd.DataFrame(
        {"prob_desc_description": [f"Problem {i}." for i in range(12)]},
        index=pd.Index(range(200, 212), name="sample_id"),
    )


@pytest.fixture(name="labels")
def labels_fixture(twelve_problems):
    """Return 0/1 labels: math on half of fold 1 and all of fold 2, games on two problems."""
    return pd.DataFrame(
        {
            "math": [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
            "games": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        },
        index=twelve_problems.index,
    )


@pytest.fixture(name="folds")
def folds_fixture(twelve_problems):
    """Assign the problems to folds 0, 1, 2, 0, 1, 2, ..."""
    return pd.Series([0, 1, 2] * 4, index=twelve_problems.index, name="fold")


@pytest.fixture(name="features")
def features_fixture(twelve_problems):
    """Return a base method's scores per tag, indexed like the problems."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(12, 2)), index=twelve_problems.index, columns=list(TAGS))


def test_cross_predict_predicts_each_fold_with_a_method_fitted_on_the_other_folds(
    twelve_problems, labels, folds
):
    predictions = cross_predict(Majority, twelve_problems, labels, folds)

    expected = pd.concat(
        Majority()
        .fit(twelve_problems[folds != fold], labels[folds != fold])
        .predict(twelve_problems[folds == fold])
        for fold in (0, 1, 2)
    ).loc[twelve_problems.index]
    pd.testing.assert_frame_equal(predictions, expected)
    assert predictions["math"].tolist() == [1, 1, 0] * 4  # the folds' fits disagree


def test_cross_predict_builds_a_fresh_method_per_fold(twelve_problems, labels, folds):
    built = []

    def make_method():
        built.append(Majority())
        return built[-1]

    cross_predict(make_method, twelve_problems, labels, folds)

    assert [len(method.fit_ids) for method in built] == [8, 8, 8]


def test_cross_predict_fits_a_composite_method_on_the_base_features(
    twelve_problems, labels, folds, features
):
    predictions = cross_predict(Stack, twelve_problems, labels, folds, features=features)

    expected = pd.concat(
        Stack()
        .fit_from(features[folds != fold], labels[folds != fold])
        .predict_from(features[folds == fold])
        for fold in (0, 1, 2)
    ).loc[twelve_problems.index]
    pd.testing.assert_frame_equal(predictions, expected)


def test_cross_predict_refuses_the_test_fold(twelve_problems, labels, folds):
    # A dev frame holding test rows passed by mistake: -1 is never a cross-fitting fold.
    with pytest.raises(ValueError, match="4 problems have a negative fold"):
        cross_predict(Majority, twelve_problems, labels, folds.mask(folds == 2, -1))


def test_cross_predict_refuses_labels_of_other_problems(twelve_problems, labels, folds):
    with pytest.raises(ValueError, match="labels must have the same index"):
        cross_predict(Majority, twelve_problems, labels.iloc[::-1], folds)


def test_cross_predict_refuses_features_of_other_problems(twelve_problems, labels, folds, features):
    with pytest.raises(ValueError, match="features must have the same index"):
        cross_predict(Stack, twelve_problems, labels, folds, features=features.iloc[1:])


def test_cross_predict_refuses_fold_methods_with_other_tags(twelve_problems, labels, folds):
    # One games positive, in fold 1: the method fitted for fold 1 (on folds 0 and 2) has no
    # games column.
    labels = labels.assign(games=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    with pytest.raises(ValueError, match="same tags"):
        cross_predict(PositiveTags, twelve_problems, labels, folds)


def test_cross_predict_needs_a_method_of_the_right_protocol(
    twelve_problems, labels, folds, features
):
    assert isinstance(Majority(), FittedMethod)
    assert isinstance(Stack(), CompositeMethod)
    with pytest.raises(TypeError, match="Majority has no fit_from"):
        cross_predict(Majority, twelve_problems, labels, folds, features=features)
    with pytest.raises(TypeError, match=r"Stack has no fit\(raw_problems"):
        cross_predict(Stack, twelve_problems, labels, folds)
