"""Tests for fitting the tag model and scoring the dev set out of fold."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from codeforces_tags.config import MODEL_MAX_ITER, MODEL_SEED
from codeforces_tags.model.params import CLASS_WEIGHTS, CLASSIFIERS
from codeforces_tags.model.tag_model import TagModel
from codeforces_tags.model.thresholds import (
    CrossFittedTagThresholds,
    apply_thresholds,
    cross_fitted_thresholds,
    per_tag_thresholds,
)
from codeforces_tags.model.training import build_pipeline, cross_fit, fit_tag_model, out_of_fold
from codeforces_tags.preprocessing.pipeline import raw_to_problems

# The results of the code from before the pipeline read raw problems (cross_fit, then
# fit_tag_model with its thresholds) on the noisy labels of the fixture's problems, written once
# by a script run on that code. Two configurations: make_params() as it is, and a linear SVM on
# the statement and code blocks.
GOLDEN = Path(__file__).parent / "golden" / "characterisation.npz"
GOLDEN_CONFIGURATIONS = {
    "logistic": {},
    "svm": {"classifier": "linear_svm", "blocks": ("statement", "code")},
}
# The arrays of each configuration, named <configuration>_<array> in the file.
GOLDEN_ARRAYS = (
    "oof_scores",
    "fold_thresholds",
    "thresholds",
    "oof_predictions",
    "refit_predictions",
)
# Calls written for the earlier signatures, before the pipeline read raw problems: what each now
# raises, and the words of its message that say what to pass instead.
OLD_STYLE_CALLS = {
    "cross_fit(params, ...)": (
        lambda old: cross_fit(old.params, old.problems, old.labels, old.folds),
        TypeError,
        r"pass build_pipeline\(params\)",
    ),
    "cross_fit on preprocessed problems": (
        lambda old: cross_fit(build_pipeline(old.params), old.problems, old.labels, old.folds),
        ValueError,
        "already preprocessed",
    ),
    "fit_tag_model(..., thresholds)": (
        lambda old: fit_tag_model(old.params, old.problems, old.labels, old.thresholds),
        ValueError,
        r"takes the dev folds, dev_raw\['fold'\]",
    ),
    "TagModel.predict on preprocessed problems": (
        lambda old: fit_tag_model(old.params, old.raw_dev, old.labels, old.folds).predict(
            old.problems
        ),
        ValueError,
        "already preprocessed",
    ),
}


@pytest.mark.parametrize(
    ("classifier", "model_type", "settings"),
    [
        ("logistic_regression", LogisticRegression, {"solver": "liblinear"}),
        ("linear_svm", LinearSVC, {"max_iter": MODEL_MAX_ITER}),
    ],
)
def test_build_pipeline_preprocesses_then_fits_one_linear_model_per_tag_on_the_feature_blocks(
    make_params, classifier, model_type, settings
):
    pipeline = build_pipeline(make_params(classifier=classifier, c=0.3, blocks=("code", "cues")))

    features, one_vs_rest = pipeline.named_steps["features"], pipeline.named_steps["classifier"]
    assert [name for name, _ in pipeline.steps] == ["preprocess", "features", "classifier"]
    preprocess = pipeline.named_steps["preprocess"]
    assert isinstance(preprocess, FunctionTransformer)
    assert preprocess.func is raw_to_problems
    assert [name for name, _, _ in features.transformers] == ["code", "cues"]
    assert isinstance(one_vs_rest, OneVsRestClassifier)
    assert one_vs_rest.n_jobs is None
    assert isinstance(one_vs_rest.estimator, model_type)
    expected = settings | {"C": 0.3, "random_state": MODEL_SEED}
    assert {name: one_vs_rest.estimator.get_params()[name] for name in expected} == expected


@pytest.mark.parametrize(("class_weight", "expected"), [("none", None), ("balanced", "balanced")])
@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_build_pipeline_passes_the_class_weight_to_the_linear_model(
    make_params, classifier, class_weight, expected
):
    pipeline = build_pipeline(make_params(classifier=classifier, class_weight=class_weight))

    assert pipeline.named_steps["classifier"].estimator.class_weight == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"classifier": classifier, "class_weight": class_weight}
        for classifier in CLASSIFIERS
        for class_weight in CLASS_WEIGHTS
    ],
    ids=lambda settings: f"{settings['classifier']}, {settings['class_weight']}",
)
def test_cross_fit_learns_the_tags_with_every_classifier_and_class_weight(
    raw_dev, labels, folds, make_params, settings
):
    oof = cross_fit(build_pipeline(make_params(**settings)), raw_dev, labels, folds)

    pd.testing.assert_frame_equal(oof.predictions, labels)


def test_cross_fit_learns_the_tags_from_the_cue_block_alone(raw_dev, labels, folds, make_params):
    oof = cross_fit(build_pipeline(make_params(blocks=("cues",))), raw_dev, labels, folds)

    pd.testing.assert_frame_equal(oof.predictions, labels)


def test_cross_fit_scores_every_problem_in_the_order_of_the_problems(
    raw_dev, labels, folds, make_params
):
    oof = cross_fit(build_pipeline(make_params()), raw_dev, labels, folds)

    for frame in (oof.scores, oof.predictions):
        assert frame.index.equals(raw_dev.index)
        assert frame.columns.equals(labels.columns)
        assert (frame.index.name, frame.columns.name) == ("sample_id", "label")
    assert oof.scores.notna().all().all()
    assert oof.predictions.dtypes.eq("int64").all()
    assert oof.fold_thresholds.index.tolist() == [0, 1, 2]
    assert oof.fold_thresholds.index.name == "fold"
    assert oof.fold_thresholds.columns.equals(labels.columns)
    assert oof.thresholds.index.equals(labels.columns)
    assert oof.thresholds.name == "threshold"


def test_cross_fit_scores_each_fold_with_a_pipeline_fitted_on_the_other_folds(
    raw_dev, labels, folds, make_params
):
    params = make_params()

    oof = cross_fit(build_pipeline(params), raw_dev, labels, folds)

    for fold in folds.unique():
        held_out = folds == fold
        pipeline = build_pipeline(params).fit(raw_dev[~held_out], labels[~held_out].to_numpy())
        np.testing.assert_array_equal(
            oof.scores[held_out].to_numpy(), pipeline.decision_function(raw_dev[held_out])
        )


def test_cross_fit_scores_a_fold_without_its_labels(raw_dev, labels, folds, make_params):
    flipped = labels.mask(folds == 1, 1 - labels, axis=0)

    before = cross_fit(build_pipeline(make_params()), raw_dev, labels, folds)
    after = cross_fit(build_pipeline(make_params()), raw_dev, flipped, folds)

    in_fold = folds == 1
    pd.testing.assert_frame_equal(after.scores[in_fold], before.scores[in_fold], check_exact=True)
    assert not after.scores[~in_fold].equals(before.scores[~in_fold])  # the other folds learn them


@pytest.fixture(name="noisy_labels")
def noisy_labels_fixture(labels):
    """Return the labels with every seventh problem's labels flipped, so that thresholds vary."""
    noisy = labels.copy()
    noisy.iloc[::7] = 1 - noisy.iloc[::7]
    return noisy


def test_cross_fit_predicts_each_fold_with_thresholds_chosen_on_the_other_folds(
    raw_dev, noisy_labels, folds, make_params
):
    oof = cross_fit(build_pipeline(make_params(c=0.1)), raw_dev, noisy_labels, folds)

    expected = cross_fitted_thresholds(oof.scores, noisy_labels, folds)
    pd.testing.assert_frame_equal(oof.fold_thresholds, expected)
    for fold in expected.index:
        in_fold = folds == fold
        pd.testing.assert_frame_equal(
            oof.predictions[in_fold],
            apply_thresholds(oof.scores[in_fold], expected.loc[fold]),
        )
    # The final thresholds would predict otherwise: the predictions are cross-fitted.
    pooled = apply_thresholds(oof.scores, oof.thresholds)
    assert not oof.predictions.equals(pooled)


def test_cross_fit_chooses_the_final_thresholds_on_all_out_of_fold_scores(
    raw_dev, noisy_labels, folds, make_params
):
    oof = cross_fit(build_pipeline(make_params(c=0.1)), raw_dev, noisy_labels, folds)

    expected = per_tag_thresholds(oof.scores, noisy_labels)
    pd.testing.assert_series_equal(oof.thresholds, expected)


def test_cross_fit_fits_one_pipeline_per_fold_and_no_refit(
    raw_dev, labels, folds, make_params, monkeypatch
):
    # A refit on every problem would change no result, only the cost of each trial.
    fitted_rows = []
    fit = Pipeline.fit

    def counted_fit(self, X, y=None, **params):
        fitted_rows.append(len(X))
        return fit(self, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", counted_fit)

    cross_fit(build_pipeline(make_params()), raw_dev, labels, folds)

    assert len(fitted_rows) == folds.nunique()
    assert max(fitted_rows) < len(raw_dev)  # none on every problem


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_cross_fit_gives_the_same_result_twice(raw_dev, labels, folds, make_params, classifier):
    pipeline = build_pipeline(make_params(classifier=classifier, class_weight="balanced"))

    first = cross_fit(pipeline, raw_dev, labels, folds)
    second = cross_fit(pipeline, raw_dev, labels, folds)

    pd.testing.assert_frame_equal(first.scores, second.scores, check_exact=True)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.fold_thresholds, second.fold_thresholds)
    pd.testing.assert_series_equal(first.thresholds, second.thresholds)


@pytest.fixture(name="wide_raw_dev")
def wide_raw_dev_fixture(raw_dev):
    """Return the raw problems with 400 description words of their own each.

    OpenBLAS splits only long dot products over threads: with these words, the model of each
    fold has some 16,000 features, enough for the thread count to change the order of the sums.
    """
    wide = raw_dev.copy()
    wide["prob_desc_description"] = [
        f"{description} " + " ".join(f"word{position}x{rank}" for rank in range(400))
        for position, description in enumerate(raw_dev["prob_desc_description"])
    ]
    return wide


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_cross_fit_predicts_the_same_on_one_blas_thread(
    wide_raw_dev, labels, folds, make_params, classifier
):
    pipeline = build_pipeline(make_params(classifier=classifier, class_weight="balanced"))

    with threadpool_limits(limits=1, user_api="blas"):  # as in a tuning worker
        one_thread = cross_fit(pipeline, wide_raw_dev, labels, folds)
    default = cross_fit(pipeline, wide_raw_dev, labels, folds)

    # Summed in another order, the scores of logistic regression differ by rounding errors.
    pd.testing.assert_frame_equal(one_thread.scores, default.scores, rtol=0, atol=1e-9)
    pd.testing.assert_frame_equal(one_thread.predictions, default.predictions, check_exact=True)


def test_cross_fit_rejects_labels_of_other_problems(raw_dev, labels, folds, make_params):
    with pytest.raises(ValueError, match="same index"):
        cross_fit(build_pipeline(make_params()), raw_dev, labels.iloc[::-1], folds)


def test_cross_fit_rejects_folds_of_other_problems(raw_dev, labels, folds, make_params):
    with pytest.raises(ValueError, match="same index"):
        cross_fit(build_pipeline(make_params()), raw_dev, labels, folds.iloc[1:])


def test_cross_fit_rejects_test_problems(raw_dev, labels, folds, make_params):
    with_test = folds.mask(folds == 2, -1)

    with pytest.raises(ValueError, match="20 problems have a negative fold"):
        cross_fit(build_pipeline(make_params()), raw_dev, labels, with_test)


def test_cross_fit_needs_two_folds(raw_dev, labels, folds, make_params):
    with pytest.raises(ValueError, match="at least two folds"):
        cross_fit(build_pipeline(make_params()), raw_dev, labels, folds * 0)


@pytest.fixture(name="fitted")
def fitted_fixture(raw_dev, labels, folds, make_params):
    """Return make_params()'s pipeline cross-fitted on the dev set, without a refit."""
    estimator = CrossFittedTagThresholds(build_pipeline(make_params()), refit=False)
    return estimator.fit(raw_dev, labels, groups=folds)


@pytest.mark.parametrize(
    ("other", "differs"),
    [
        (lambda labels, folds: (labels.iloc[::-1], folds.iloc[::-1]), "problems"),
        (lambda labels, folds: (labels[labels.columns[::-1]], folds), "tags"),
        (lambda labels, folds: (labels, folds + 1), "folds"),
    ],
    ids=["problems in another order", "tags in another order", "other fold ids"],
)
def test_out_of_fold_refuses_labels_or_folds_the_estimator_was_not_fitted_on(
    fitted, labels, folds, other, differs
):
    with pytest.raises(ValueError, match=f"fitted on other {differs}"):
        out_of_fold(fitted, *other(labels, folds))


def test_fit_tag_model_chooses_the_thresholds_out_of_fold_then_refits_on_every_problem(
    raw_dev, noisy_labels, folds, make_params
):
    params = make_params(classifier="linear_svm")

    model = fit_tag_model(params, raw_dev, noisy_labels, folds)

    assert isinstance(model, TagModel)
    assert model.params == params
    assert isinstance(model.estimator, CrossFittedTagThresholds)
    assert model.estimator.refit
    oof = cross_fit(build_pipeline(params), raw_dev, noisy_labels, folds)
    pd.testing.assert_series_equal(model.thresholds, oof.thresholds)
    expected = build_pipeline(params).fit(raw_dev, noisy_labels.to_numpy())
    np.testing.assert_array_equal(
        model.pipeline.decision_function(raw_dev), expected.decision_function(raw_dev)
    )


def test_fit_tag_model_rejects_labels_of_other_problems(raw_dev, labels, folds, make_params):
    with pytest.raises(ValueError, match="same index"):
        fit_tag_model(make_params(), raw_dev, labels.iloc[::-1], folds)


@pytest.fixture(name="old")
def old_fixture(raw_dev, problems, labels, folds, make_params):
    """Return what a caller of the old signatures holds: preprocessed problems and thresholds."""
    return SimpleNamespace(
        params=make_params(),
        raw_dev=raw_dev,
        problems=problems,  # preprocessed, as load_dataset() gives them
        labels=labels,
        folds=folds,
        thresholds=pd.Series(0.0, index=labels.columns, name="threshold"),
    )


@pytest.mark.parametrize("call", OLD_STYLE_CALLS)
def test_old_style_calls_fail_with_what_to_pass(old, call):
    run, error, message = OLD_STYLE_CALLS[call]

    with pytest.raises(error, match=message):
        run(old)


@pytest.mark.parametrize("configuration", GOLDEN_CONFIGURATIONS)
def test_cross_fit_and_the_refit_reproduce_mains_results_on_the_fixture(
    raw_dev, noisy_labels, folds, make_params, configuration
):
    params = make_params(**GOLDEN_CONFIGURATIONS[configuration])

    oof = cross_fit(build_pipeline(params), raw_dev, noisy_labels, folds)
    refit = fit_tag_model(params, raw_dev, noisy_labels, folds)

    with np.load(GOLDEN) as golden:  # refuses pickles: allow_pickle is False by default
        expected = {name: golden[f"{configuration}_{name}"] for name in GOLDEN_ARRAYS}
    close = {"rtol": 0, "atol": 1e-9}  # CI's BLAS may sum in another order
    np.testing.assert_allclose(oof.scores.to_numpy(), expected["oof_scores"], **close)
    np.testing.assert_allclose(oof.fold_thresholds.to_numpy(), expected["fold_thresholds"], **close)
    np.testing.assert_allclose(oof.thresholds.to_numpy(), expected["thresholds"], **close)
    np.testing.assert_array_equal(oof.predictions.to_numpy(), expected["oof_predictions"])
    np.testing.assert_array_equal(refit.predict(raw_dev).to_numpy(), expected["refit_predictions"])
