"""Tests for the score estimators of the frozen-encoder methods."""

import json

import numpy as np
import pytest
import skops.io as sio
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.svm import LinearSVC

from codeforces_tags.config import MODEL_MAX_ITER, MODEL_SEED, TFIDF_BASELINE_DIR
from codeforces_tags.methods.f2llm.embedding import SOLUTION, STATEMENT, embedding_columns
from codeforces_tags.methods.f2llm.estimators import (
    HEADS,
    ScoreFusion,
    embedding_blocks,
    make_early_fusion,
    make_estimator,
    make_head,
    make_late_fusion,
    make_probe,
    make_raw_tfidf,
    make_tfidf,
    tfidf_params,
)
from codeforces_tags.methods.f2llm.settings import (
    EARLY_FUSION,
    FUSION_INNER_FOLDS,
    LATE_FUSION,
    LBFGS_MAX_ITER,
    PROBE,
    PROBE_GRID,
    SETFIT_METHOD,
)
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.preprocessing.pipeline import raw_to_problems

PROBE_PARAMS = {"head": "ridge", "c": 1.0, "class_weight": "none", "solution_weight": 0.5}
EARLY_PARAMS = {"c": 0.62, "statement_weight": 0.25, "solution_weight": 0.1}
LATE_PARAMS = {"weight": 0.4}
# The types skops does not trust by default in each fitted estimator, by import path (sorted, as
# skops lists them): the TF-IDF blocks' functions, and ScoreFusion itself.
TFIDF_FUNCTIONS = [
    "codeforces_tags.model.features.code_tokens",
    "codeforces_tags.model.features.normalise_statement",
    "codeforces_tags.model.features.statement_text",
]
UNTRUSTED_TYPES = {
    PROBE: [],
    EARLY_FUSION: TFIDF_FUNCTIONS,
    LATE_FUSION: ["codeforces_tags.methods.f2llm.estimators.ScoreFusion", *TFIDF_FUNCTIONS],
}


class ConstantScorer(BaseEstimator):
    """A scorer that gives every row and tag the score 0."""

    def fit(self, X, y):  # pylint: disable=unused-argument
        """Set ``classes_``."""
        self.classes_ = np.arange(np.asarray(y).shape[1])
        return self

    def decision_function(self, X):
        """Return zeros, one row per row of ``X``."""
        return np.zeros((len(X), len(self.classes_)))

    def predict(self, X):
        """Predict every tag (score 0 is at least 0), as the contract's ``predict`` does."""
        return (self.decision_function(X) >= 0).astype(int)


@pytest.fixture(name="build")
def build_fixture(embedding_width, small_tfidf):
    """Return a builder of each method's unfitted estimator, by method name."""

    def build(name):
        params = {PROBE: PROBE_PARAMS, EARLY_FUSION: EARLY_PARAMS, LATE_FUSION: LATE_PARAMS}
        return make_estimator(
            name,
            params[name],
            width=embedding_width,
            probe_params=PROBE_PARAMS,
            tfidf=small_tfidf,
        )

    return build


@pytest.fixture(name="y")
def y_fixture(synthetic_labels):
    """Return the labels as the int64 array the score-estimator contract fits on."""
    return synthetic_labels.to_numpy(dtype=np.int64)


def test_embedding_blocks_scale_then_normalise_each_input_of_weight_above_zero():
    blocks = embedding_blocks({STATEMENT: 1.0, SOLUTION: 0.25}, 3)

    assert [(name, columns) for name, _, columns in blocks.transformers] == [
        ("statement_embedding", embedding_columns(STATEMENT, 3)),
        ("solution_embedding", embedding_columns(SOLUTION, 3)),
    ]
    assert blocks.transformer_weights == {"statement_embedding": 1.0, "solution_embedding": 0.25}
    assert blocks.remainder == "drop"
    steps = blocks.transformers[0][1].steps
    assert [name for name, _ in steps] == ["scale", "normalise"]


def test_embedding_blocks_leave_an_input_of_weight_zero_out():
    blocks = embedding_blocks({STATEMENT: 1.0, SOLUTION: 0.0}, 3)

    assert [name for name, _, _ in blocks.transformers] == ["statement_embedding"]
    assert blocks.transformer_weights == {"statement_embedding": 1.0}


def test_embedding_blocks_give_each_row_the_norm_of_its_block_weight(
    embedding_frame, embedding_width
):
    blocks = embedding_blocks({STATEMENT: 1.0, SOLUTION: 0.5}, embedding_width)

    output = blocks.fit_transform(embedding_frame)

    assert output.shape == (len(embedding_frame), 2 * embedding_width)
    norms = np.linalg.norm(output[:, :embedding_width], axis=1)
    np.testing.assert_allclose(norms, 1.0)
    np.testing.assert_allclose(np.linalg.norm(output[:, embedding_width:], axis=1), 0.5)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({STATEMENT: 1.0, "notes": 1.0}, "Unknown inputs"),
        ({STATEMENT: 1.0, SOLUTION: -0.5}, "0 or above"),
        ({STATEMENT: 0.0, SOLUTION: 0.0}, "at least one input"),
    ],
)
def test_embedding_blocks_refuse_an_unknown_input_a_negative_weight_or_no_block(weights, message):
    with pytest.raises(ValueError, match=message):
        embedding_blocks(weights, 3)


def test_make_head_builds_a_lbfgs_logistic_regression():
    head = make_head("logistic_regression", 0.215, "balanced")

    assert isinstance(head, LogisticRegression)
    assert (head.solver, head.C, head.class_weight) == ("lbfgs", 0.215, "balanced")
    assert head.max_iter == LBFGS_MAX_ITER


def test_make_head_builds_a_seeded_linear_svm():
    head = make_head("linear_svm", 4.64, "none")

    assert isinstance(head, LinearSVC)
    assert (head.C, head.class_weight) == (4.64, None)
    assert (head.max_iter, head.random_state) == (MODEL_MAX_ITER, MODEL_SEED)


def test_make_head_builds_a_ridge_classifier_with_the_penalty_of_c():
    head = make_head("ridge", 2.0, "balanced")

    assert isinstance(head, RidgeClassifier)
    assert (head.alpha, head.class_weight) == (0.25, "balanced")


@pytest.mark.parametrize(
    ("head", "c", "class_weight", "message"),
    [
        ("knn", 1.0, "none", "head must be one of"),
        ("ridge", 1.0, "auto", "class_weight must be one of"),
        ("ridge", 0.0, "none", "c must be above 0"),
    ],
)
def test_make_head_refuses_an_unknown_head_class_weight_or_c(head, c, class_weight, message):
    with pytest.raises(ValueError, match=message):
        make_head(head, c, class_weight)


def test_every_head_of_the_probe_grid_is_a_head_make_head_builds():
    assert {params["head"] for params in PROBE_GRID} == set(HEADS)


def test_make_probe_puts_one_head_per_tag_on_the_embedding_blocks(embedding_width):
    probe = make_probe(PROBE_PARAMS | {"solution_weight": 0.0}, embedding_width)

    assert [name for name, _ in probe.steps] == ["features", "classifier"]
    assert probe.named_steps["features"].transformer_weights == {"statement_embedding": 1.0}
    assert isinstance(probe.named_steps["classifier"].estimator, RidgeClassifier)


def test_tfidf_params_are_the_frozen_tfidf_configuration():
    stored = json.loads((TFIDF_BASELINE_DIR / "params.json").read_text(encoding="utf-8"))

    params = tfidf_params()

    assert params == ModelParams.from_dict(stored)
    assert params.classifier == "linear_svm"


def test_make_tfidf_puts_a_seeded_linear_svm_on_the_frozen_feature_blocks(small_tfidf):
    tfidf = make_tfidf(small_tfidf)

    assert [name for name, _ in tfidf.steps] == ["features", "classifier"]
    assert [name for name, _, _ in tfidf.named_steps["features"].transformers] == [
        "statement",
        "code",
    ]
    assert tfidf.named_steps["features"].transformer_weights == {"statement": 1.0, "code": 0.5}
    svm = tfidf.named_steps["classifier"].estimator
    assert isinstance(svm, LinearSVC)
    assert (svm.C, svm.class_weight, svm.max_iter, svm.random_state) == (
        1.0,
        None,
        MODEL_MAX_ITER,
        MODEL_SEED,
    )


def test_make_tfidf_refuses_a_configuration_that_is_not_a_linear_svm(small_tfidf):
    params = ModelParams.from_dict(small_tfidf.to_dict() | {"classifier": "logistic_regression"})

    with pytest.raises(ValueError, match="linear_svm"):
        make_tfidf(params)


def test_make_raw_tfidf_scores_raw_problems_as_make_tfidf_scores_preprocessed_ones(
    synthetic_raw, y, small_tfidf
):
    raw = make_raw_tfidf(small_tfidf).fit(synthetic_raw, y)
    problems = raw_to_problems(synthetic_raw)
    preprocessed = make_tfidf(small_tfidf).fit(problems, y)

    assert [name for name, _ in raw.steps] == ["problems", "features", "classifier"]
    np.testing.assert_array_equal(
        raw.decision_function(synthetic_raw), preprocessed.decision_function(problems)
    )


def test_make_early_fusion_holds_the_tfidf_and_embedding_blocks_with_their_weights(
    embedding_width, small_tfidf
):
    fusion = make_early_fusion(EARLY_PARAMS, embedding_width, small_tfidf)

    features = fusion.named_steps["features"]
    assert [name for name, _, _ in features.transformers] == [
        "statement",
        "code",
        "statement_embedding",
        "solution_embedding",
    ]
    assert features.transformer_weights == {
        "statement": 1.0,
        "code": 0.5,
        "statement_embedding": 0.25,
        "solution_embedding": 0.1,
    }
    assert (features.remainder, features.sparse_threshold) == ("drop", 1.0)
    assert fusion.named_steps["classifier"].estimator.C == 0.62


def test_make_early_fusion_leaves_the_solution_block_out_at_weight_zero(
    embedding_width, small_tfidf
):
    fusion = make_early_fusion(
        EARLY_PARAMS | {"solution_weight": 0.0}, embedding_width, small_tfidf
    )

    assert "solution_embedding" not in fusion.named_steps["features"].transformer_weights


@pytest.mark.parametrize("name", [PROBE, EARLY_FUSION, LATE_FUSION])
def test_each_estimator_follows_the_score_estimator_contract(name, build, embedding_frame, y):
    estimator = clone(build(name)).fit(embedding_frame, y)

    scores = estimator.decision_function(embedding_frame)

    np.testing.assert_array_equal(estimator.classes_, np.arange(y.shape[1]))
    assert scores.shape == y.shape
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(estimator.predict(embedding_frame), (scores >= 0).astype(int))


@pytest.mark.parametrize("name", [PROBE, EARLY_FUSION, LATE_FUSION])
def test_each_estimator_works_under_cross_fitted_tag_thresholds(
    name, build, embedding_frame, synthetic_labels
):
    model = CrossFittedTagThresholds(build(name)).fit(
        embedding_frame, synthetic_labels, groups=embedding_frame["fold"]
    )

    assert model.oof_scores_.shape == synthetic_labels.shape
    predicted = model.predict(embedding_frame.drop(columns=["fold", "group", "tags"]))
    assert predicted.shape == synthetic_labels.shape


@pytest.mark.parametrize("name", [PROBE, EARLY_FUSION, LATE_FUSION])
def test_skops_saves_each_fitted_estimator_with_only_the_pinned_untrusted_types(
    name, build, embedding_frame, y
):
    fitted = clone(build(name)).fit(embedding_frame, y)
    data = sio.dumps(fitted)

    assert sio.get_untrusted_types(data=data) == UNTRUSTED_TYPES[name]
    loaded = sio.loads(data, trusted=UNTRUSTED_TYPES[name])
    np.testing.assert_array_equal(
        loaded.decision_function(embedding_frame), fitted.decision_function(embedding_frame)
    )


def test_score_fusion_standardises_with_inner_group_out_of_fold_scores(build, embedding_frame, y):
    fusion = build(LATE_FUSION)
    fitted = clone(fusion).fit(embedding_frame, y)

    for position, scorer in enumerate((fusion.first, fusion.second)):
        held_out = cross_val_predict(
            clone(scorer),
            embedding_frame,
            y,
            cv=GroupKFold(n_splits=FUSION_INNER_FOLDS),
            groups=embedding_frame["group"],
            method="decision_function",
        )
        np.testing.assert_array_equal(fitted.means_[position], held_out.mean(axis=0))
        np.testing.assert_array_equal(fitted.scales_[position], held_out.std(axis=0))
    # The training rows' own scores are spread otherwise: they are not what is used.
    in_sample = fitted.first_.decision_function(embedding_frame).std(axis=0)
    assert not np.allclose(fitted.scales_[0], in_sample)


def test_score_fusion_fold_models_learn_their_statistics_from_their_own_rows(
    build, embedding_frame, synthetic_labels
):
    fusion = build(LATE_FUSION).set_params(inner_folds=3)
    folds = embedding_frame["fold"]

    model = CrossFittedTagThresholds(fusion, refit=False).fit(
        embedding_frame, synthetic_labels, groups=folds
    )

    for fold in sorted(folds.unique()):
        held_out = (folds == fold).to_numpy()
        fold_model = clone(fusion).fit(
            embedding_frame[~held_out], synthetic_labels[~held_out].to_numpy(dtype=np.int64)
        )
        np.testing.assert_array_equal(
            model.oof_scores_[held_out], fold_model.decision_function(embedding_frame[held_out])
        )


@pytest.mark.parametrize(("weight", "position"), [(1.0, 0), (0.0, 1)])
def test_score_fusion_at_either_end_is_one_scorer_standardised(
    weight, position, build, embedding_frame, y
):
    fitted = clone(build(LATE_FUSION)).set_params(weight=weight).fit(embedding_frame, y)

    scorer = (fitted.first_, fitted.second_)[position]
    expected = (scorer.decision_function(embedding_frame) - fitted.means_[position]) / (
        fitted.scales_[position]
    )
    np.testing.assert_allclose(
        fitted.decision_function(embedding_frame), expected, rtol=0, atol=1e-12
    )


def test_score_fusion_keeps_scale_one_for_constant_inner_scores(build, embedding_frame, y):
    probe = build(PROBE)
    fitted = ScoreFusion(ConstantScorer(), probe, weight=0.5).fit(embedding_frame, y)

    np.testing.assert_array_equal(fitted.first_.predict(embedding_frame), 1)
    np.testing.assert_array_equal(fitted.means_[0], 0.0)
    np.testing.assert_array_equal(fitted.scales_[0], 1.0)
    second = (fitted.second_.decision_function(embedding_frame) - fitted.means_[1]) / (
        fitted.scales_[1]
    )
    np.testing.assert_allclose(fitted.decision_function(embedding_frame), 0.5 * second)


def test_score_fusion_needs_the_group_column(build, embedding_frame, y):
    with pytest.raises(ValueError, match="'group' column"):
        clone(build(LATE_FUSION)).fit(embedding_frame.drop(columns="group"), y)


def test_score_fusion_refuses_a_weight_outside_zero_to_one(build, embedding_frame, y):
    with pytest.raises(ValueError, match="from 0 to 1"):
        clone(build(LATE_FUSION)).set_params(weight=1.5).fit(embedding_frame, y)


def test_score_fusion_does_not_score_before_fit(build, embedding_frame):
    with pytest.raises(NotFittedError):
        build(LATE_FUSION).decision_function(embedding_frame)


def test_make_late_fusion_fuses_the_tfidf_scorer_first_and_the_probe_second(
    embedding_width, small_tfidf
):
    fusion = make_late_fusion(LATE_PARAMS, embedding_width, PROBE_PARAMS, small_tfidf)

    assert (fusion.weight, fusion.inner_folds) == (0.4, FUSION_INNER_FOLDS)
    assert [name for name, _, _ in fusion.first.named_steps["features"].transformers] == [
        "statement",
        "code",
    ]
    assert isinstance(fusion.second.named_steps["classifier"].estimator, RidgeClassifier)


def test_make_estimator_builds_each_method_s_estimator(build):
    probe, early, late = build(PROBE), build(EARLY_FUSION), build(LATE_FUSION)

    assert list(probe.named_steps["features"].transformer_weights) == [
        "statement_embedding",
        "solution_embedding",
    ]
    assert "code" in early.named_steps["features"].transformer_weights
    assert isinstance(late, ScoreFusion)


@pytest.mark.parametrize(
    ("name", "missing", "message"),
    [
        (SETFIT_METHOD, (), "has no estimator here"),
        (EARLY_FUSION, ("tfidf",), "frozen TF-IDF configuration"),
        (LATE_FUSION, ("probe_params",), "probe's tuned configuration"),
    ],
)
def test_make_estimator_refuses_another_method_or_a_missing_configuration(
    name, missing, message, small_tfidf, embedding_width
):
    given = {"probe_params": PROBE_PARAMS, "tfidf": small_tfidf}
    configurations = {key: None if key in missing else value for key, value in given.items()}

    with pytest.raises(ValueError, match=message):
        make_estimator(name, LATE_PARAMS, width=embedding_width, **configurations)
