"""Notebook 04's first look: a logistic regression per tag on the TF-IDF of the statement.

``first_look_pipeline`` is the reference model of notebook 04's first section, before any
tuning: word unigrams and bigrams of the normalised statement, then one liblinear logistic
regression per tag, fitted on every tag of the dataset. It reads raw problems, as the tag model
does, and is fitted the same way, through ``CrossFittedTagThresholds``. It is in the package so
that every caller fits the very same model. This module is on the prediction path: it never
imports mlflow or optuna.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from codeforces_tags.config import MODEL_SEED
from codeforces_tags.model.features import normalise_statement
from codeforces_tags.preprocessing.pipeline import raw_to_problems


def first_look_pipeline() -> Pipeline:
    """Build the first look, unfitted: raw problems in, a score per tag out.

    The statement alone, with fixed settings: the unigrams and bigrams of the normalised
    statement (``normalise_statement``) found in at least two problems, with sublinear term
    frequencies, then one logistic regression per tag of whatever labels it is fitted on.

    Returns:
        The steps ``preprocess`` (``raw_to_problems``, which refuses preprocessed rows),
        ``features`` (a ``ColumnTransformer`` that hands the vectorizer the ``statement``
        column as a 1-D Series, which ``TfidfVectorizer`` needs) and ``classifier`` (a
        ``OneVsRestClassifier`` of liblinear logistic regressions, seeded with
        ``MODEL_SEED``). Nothing runs in parallel (no ``n_jobs``): joblib would write pickle
        files.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=normalise_statement, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    linear_model = LogisticRegression(solver="liblinear", random_state=MODEL_SEED)
    return Pipeline(
        [
            ("preprocess", FunctionTransformer(raw_to_problems)),
            ("features", ColumnTransformer([("statement", vectorizer, "statement")])),
            ("classifier", OneVsRestClassifier(linear_model)),
        ]
    )
