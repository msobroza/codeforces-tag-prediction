"""Tests for notebook 04's first look: TF-IDF of the statement and a logistic regression per tag."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from codeforces_tags.config import MODEL_SEED
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.model.features import normalise_statement
from codeforces_tags.model.first_look import first_look_pipeline
from codeforces_tags.model.thresholds import CrossFittedTagThresholds, per_tag_thresholds
from codeforces_tags.model.training import cross_fit
from codeforces_tags.preprocessing.pipeline import raw_to_problems


def notebook_first_look():
    """Return the first look as notebook 04 built it before the pipeline read raw problems."""
    vectorizer = TfidfVectorizer(
        preprocessor=normalise_statement, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    linear_model = LogisticRegression(solver="liblinear", random_state=MODEL_SEED)
    return Pipeline([("tfidf", vectorizer), ("classifier", OneVsRestClassifier(linear_model))])


def test_the_first_look_reads_raw_problems_then_the_statement_alone():
    pipeline = first_look_pipeline()

    assert list(pipeline.named_steps) == ["preprocess", "features", "classifier"]
    assert pipeline.named_steps["preprocess"].func is raw_to_problems
    features = pipeline.named_steps["features"]
    assert isinstance(features, ColumnTransformer)
    [(name, vectorizer, column)] = features.transformers
    # A scalar column name hands the vectorizer a 1-D Series; a list would hand it a frame.
    assert (name, column) == ("statement", "statement")
    expected = notebook_first_look()
    assert vectorizer.get_params() == expected.named_steps["tfidf"].get_params()
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, OneVsRestClassifier)
    linear_model = expected.named_steps["classifier"].estimator
    assert classifier.estimator.get_params() == linear_model.get_params()


def test_the_first_look_scores_raw_problems_as_the_notebook_scored_their_statements(
    raw_dev, problems
):
    all_tags = multi_hot(raw_dev["tags"])  # every tag of the problems, not only the focus ones

    scores = first_look_pipeline().fit(raw_dev, all_tags).decision_function(raw_dev)

    statements = problems["statement"]
    expected = notebook_first_look().fit(statements, all_tags).decision_function(statements)
    np.testing.assert_array_equal(scores, expected)


def test_a_refit_first_look_keeps_the_thresholds_chosen_on_its_out_of_fold_scores(raw_dev, folds):
    # Notebook 04's reference test score comes from this refit: its thresholds are those the
    # notebook chose on the first look's out-of-fold scores of every tag.
    all_tags = multi_hot(raw_dev["tags"])
    scores = cross_fit(first_look_pipeline(), raw_dev, all_tags, folds).scores

    refit = CrossFittedTagThresholds(first_look_pipeline(), refit=True)
    refit.fit(raw_dev, all_tags, groups=folds)

    assert refit.tags_ == tuple(all_tags.columns)
    np.testing.assert_array_equal(
        refit.thresholds_, per_tag_thresholds(scores, all_tags).to_numpy()
    )
