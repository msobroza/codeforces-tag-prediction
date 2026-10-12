"""Tests for the tag model: scores, predictions and feature weights, on raw problems.

The tag model is also the tagging method ``tfidf``: its method folder, its conformance and its
MLflow round trip.
"""

import dataclasses
import json
import math
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import skops.io
from sklearn.preprocessing import FunctionTransformer
from skops.io.exceptions import UntrustedTypesFoundException

from codeforces_tags.config import RAW_INPUT_COLUMNS, SKOPS_COMPRESSION
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.methods import ScoreMethod
from codeforces_tags.model.features import SAMPLE_IO_CUES, STORED_FUNCTIONS, feature_blocks
from codeforces_tags.model.tag_model import MODEL_FILE, NAME, STORED_TYPES, TAG_MODEL_FILE, load
from codeforces_tags.model.thresholds import apply_thresholds
from codeforces_tags.model.training import build_pipeline, cross_fit, fit_tag_model
from codeforces_tags.preprocessing.pipeline import preprocess
from codeforces_tags.tracking.methods import load_logged_method


@pytest.fixture(name="params")
def params_fixture(make_params):
    """Return parameters with all three blocks, weighted 1, 0.5 and 0.25."""
    return make_params(code_weight=0.5, cue_weight=0.25)


@pytest.fixture(name="tag_model")
def tag_model_fixture(raw_dev, labels, folds, params):
    """Return a tag model fitted on every problem, predicting a tag when its logit is >= 0."""
    return with_thresholds(fit_tag_model(params, raw_dev, labels, folds), 0.0, 0.0, 0.0)


def with_thresholds(tag_model, *thresholds):
    """Give the tag model other thresholds, one per tag in tag order, and return it.

    A tag model reads its thresholds from its fitted ``CrossFittedTagThresholds``: they are set
    there.
    """
    tag_model.estimator.thresholds_ = np.array(thresholds, dtype=float)
    return tag_model


def linear_models(tag_model):
    """Return the fitted linear model of each tag, in tag order."""
    return tag_model.pipeline.named_steps["classifier"].estimators_


def test_tags_are_the_label_columns_in_order(tag_model):
    assert tag_model.tags == ("graphs", "strings", "games")


def test_pipeline_and_thresholds_are_those_of_the_fitted_estimator(tag_model):
    estimator = tag_model.estimator

    assert tag_model.pipeline is estimator.estimator_
    assert list(tag_model.pipeline.named_steps) == ["preprocess", "features", "classifier"]
    expected = pd.Series(
        estimator.thresholds_, index=pd.Index(estimator.tags_, name="label"), name="threshold"
    )
    pd.testing.assert_series_equal(tag_model.thresholds, expected)
    returned = tag_model.tag_thresholds()
    returned["graphs"] = math.inf  # a caller changes its copy, never the model
    pd.testing.assert_series_equal(tag_model.tag_thresholds(), expected)


def test_scores_are_each_tag_decision_function_indexed_like_the_problems(raw_dev, tag_model):
    some = raw_dev.iloc[[5, 2, 9]]

    scores = tag_model.scores(some)

    assert scores.index.equals(some.index)
    assert scores.columns.tolist() == ["graphs", "strings", "games"]
    assert scores.columns.name == "label"
    np.testing.assert_array_equal(scores.to_numpy(), tag_model.pipeline.decision_function(some))


def test_predict_applies_each_tag_threshold_to_its_scores(raw_dev, tag_model):
    model = with_thresholds(tag_model, 0.0, math.inf, -math.inf)

    predicted = model.predict(raw_dev)

    pd.testing.assert_frame_equal(
        predicted, apply_thresholds(model.scores(raw_dev), model.thresholds)
    )
    assert predicted["strings"].eq(0).all()
    assert predicted["games"].eq(1).all()


def test_predict_recovers_the_labels_the_model_was_fitted_on(raw_dev, labels, tag_model):
    pd.testing.assert_frame_equal(tag_model.predict(raw_dev), labels)


def test_predicted_tags_list_each_problem_tags_and_nothing_for_a_no_focus_problem(
    raw_dev, tag_model
):
    one_of_each_kind = raw_dev.iloc[:4]  # graphs, strings, games, no focus tag

    predicted = tag_model.predicted_tags(one_of_each_kind)

    assert predicted.name == "predicted_tags"
    assert predicted.index.equals(one_of_each_kind.index)
    assert predicted.tolist() == [["graphs"], ["strings"], ["games"], []]


def test_predicted_tags_come_in_tag_order(raw_dev, tag_model):
    model = with_thresholds(tag_model, -math.inf, math.inf, -math.inf)

    assert model.predicted_tags(raw_dev.iloc[[0]]).tolist() == [["graphs", "games"]]


def test_out_of_fold_scores_are_the_cross_fitted_scores_of_the_problems_fitted_on(
    raw_dev, labels, folds, params, tag_model
):
    expected = cross_fit(build_pipeline(params), raw_dev, labels, folds).scores

    pd.testing.assert_frame_equal(tag_model.out_of_fold_scores(), expected, check_exact=True)


def test_feature_weights_have_one_row_per_feature_of_each_block(problems, tag_model):
    weights = tag_model.feature_weights()

    assert weights.index.names == ["block", "feature"]
    assert weights.columns.tolist() == ["graphs", "strings", "games"]
    assert weights.index.unique("block").tolist() == ["statement", "code", "cues"]
    assert ("statement", "palindrome") in weights.index
    assert ("code", "adj") in weights.index
    assert weights.loc["cues"].index.tolist() == list(SAMPLE_IO_CUES)
    assert len(weights) == tag_model.pipeline.named_steps["features"].transform(problems).shape[1]


def test_feature_weights_cover_only_the_fitted_blocks(raw_dev, labels, folds, make_params):
    tag_model = fit_tag_model(make_params(blocks=("code", "cues")), raw_dev, labels, folds)

    assert tag_model.feature_weights().index.unique("block").tolist() == ["code", "cues"]


def test_feature_weights_times_the_unweighted_features_give_the_scores(
    raw_dev, problems, params, tag_model
):
    unweighted = dataclasses.replace(params, code_weight=1.0, cue_weight=1.0)
    values = feature_blocks(unweighted).fit(problems).transform(problems)
    intercepts = [model.intercept_[0] for model in linear_models(tag_model)]

    scores = values @ tag_model.feature_weights().to_numpy() + intercepts

    np.testing.assert_allclose(scores, tag_model.scores(raw_dev).to_numpy())


def test_feature_weights_of_a_cue_are_its_coefficient_times_the_cue_weight(tag_model):
    cue_columns = tag_model.pipeline.named_steps["features"].output_indices_["cues"]
    coefficients = np.column_stack([model.coef_.ravel() for model in linear_models(tag_model)])

    weights = tag_model.feature_weights().loc["cues"]

    expected = 0.25 * coefficients[cue_columns]
    np.testing.assert_array_equal(weights.to_numpy(), expected)
    assert weights.at["graph_edge_list", "graphs"] > 0  # only the graph problems have it
    assert weights.at["player_output", "games"] > 0  # only the game problems have it


def test_a_tag_model_is_the_score_method_tfidf(tag_model):
    assert isinstance(tag_model, ScoreMethod)
    assert tag_model.name == NAME == "tfidf"


def test_save_writes_the_fitted_estimator_and_the_tags_and_parameters(tag_model, params, tmp_path):
    tag_model.save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [MODEL_FILE, TAG_MODEL_FILE]
    stored = json.loads((tmp_path / TAG_MODEL_FILE).read_text(encoding="utf-8"))
    assert stored == {"tags": ["graphs", "strings", "games"], "params": params.to_dict()}


def test_save_writes_a_compressed_model_file(tag_model, tmp_path):
    # The frozen model's file is 53.85 MB uncompressed, above
    # METHOD_FILE_MAX_BYTES, but a synthetic one is a few KB: only the zip entries tell a lost
    # compression= before the real freeze fails on check_method_folder.
    tag_model.save(tmp_path)

    with zipfile.ZipFile(tmp_path / MODEL_FILE) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    assert SKOPS_COMPRESSION == zipfile.ZIP_DEFLATED


def test_load_returns_the_saved_tag_model(raw_dev, tag_model, params, tmp_path):
    saved = with_thresholds(tag_model, 0.5, -0.5, 0.0)
    saved.save(tmp_path)

    loaded = load(tmp_path)

    assert (loaded.name, loaded.tags, loaded.params) == (NAME, saved.tags, params)
    pd.testing.assert_series_equal(loaded.thresholds, saved.thresholds)
    pd.testing.assert_frame_equal(loaded.scores(raw_dev), saved.scores(raw_dev), check_exact=True)
    pd.testing.assert_frame_equal(
        loaded.out_of_fold_scores(), saved.out_of_fold_scores(), check_exact=True
    )


def test_stored_types_are_the_types_skops_asks_to_trust_in_a_saved_tag_model(tag_model, tmp_path):
    # The tag_model fixture keeps the three blocks, so its file stores every function; the frozen
    # configuration, without the cue block, stores all but sample_io_cues.
    tag_model.save(tmp_path)

    assert tuple(skops.io.get_untrusted_types(file=tmp_path / MODEL_FILE)) == STORED_TYPES
    assert set(STORED_FUNCTIONS) < set(STORED_TYPES)


def test_load_refuses_a_model_file_that_stores_a_type_it_does_not_trust(tag_model, tmp_path):
    tag_model.save(tmp_path)
    skops.io.dump(FunctionTransformer(preprocess), tmp_path / MODEL_FILE)

    with pytest.raises(UntrustedTypesFoundException, match="pipeline.preprocess"):
        load(tmp_path)


def test_load_refuses_tags_that_differ_from_the_fitted_estimators(tag_model, tmp_path):
    tag_model.save(tmp_path)
    path = tmp_path / TAG_MODEL_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(stored | {"tags": ["games", "strings", "graphs"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="stores the tags"):
        load(tmp_path)


def test_a_focus_tag_model_passes_check_tagging_method(focus_raw_dev, focus_tag_model, tmp_path):
    # Raw problems as select_part gives them: NaN fields, and the tags, fold and group columns.
    check_tagging_method(focus_tag_model, focus_raw_dev, tmp_path / "model")


@pytest.fixture(name="logged_uri")
def logged_uri_fixture(log_tag_model, tmp_path):
    """Save the focus tag model, log it as freeze does (``log_tag_model``), and return its URI."""
    return f"models:/{log_tag_model(tmp_path / 'model')}"


def test_a_logged_tag_model_predicts_raw_problems_as_the_method_does(
    focus_raw_dev, focus_tag_model, logged_uri, store
):
    predicted = load_logged_method(logged_uri, **store).predict(focus_raw_dev)

    pd.testing.assert_frame_equal(predicted, focus_tag_model.predict(focus_raw_dev))


def test_logged_tag_model_predicts_rows_with_missing_fields_as_the_method_does(
    focus_raw_dev, focus_tag_model, logged_uri, store
):
    # MLflow's wrapper hands the method its raw string columns, NaN kept as NaN
    # (pandas' str dtype), in a batch and one problem at a time as raw.loc[[sample_id]] gives
    # it. A NaN read back as "nan" would put "nan" into the statement and the notes.
    fields = focus_raw_dev[["prob_desc_input_spec", "prob_desc_notes"]]
    missing = focus_raw_dev[fields.isna().any(axis=1)]
    logged = load_logged_method(logged_uri, **store)
    wrapper = logged.unwrap_python_model()
    method, given = wrapper.method, []

    def recording_predict(raw_problems):
        given.append(raw_problems)
        return method.predict(raw_problems)

    wrapper.method = SimpleNamespace(predict=recording_predict)

    batch = logged.predict(missing)
    one_at_a_time = pd.concat(
        [logged.predict(missing.loc[[sample_id]]) for sample_id in missing.index]
    )

    expected = focus_tag_model.predict(missing)
    pd.testing.assert_frame_equal(batch, expected)
    pd.testing.assert_frame_equal(one_at_a_time, expected)
    assert len(given) == 1 + len(missing)
    for raw_problems in given:
        pd.testing.assert_frame_equal(
            raw_problems, missing.loc[raw_problems.index, list(RAW_INPUT_COLUMNS)]
        )
