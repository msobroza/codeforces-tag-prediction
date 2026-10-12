"""Tests for the parameter table and the trace of one problem through the fitted pipeline."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from codeforces_tags.analysis.pipeline_trace import NOT_USED, parameter_table, trace_problem
from codeforces_tags.config import RAW_INPUT_COLUMNS
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.model.features import SAMPLE_IO_CUES, block_feature_names, normalise_statement
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.model.training import build_pipeline, fit_tag_model
from codeforces_tags.preprocessing.pipeline import preprocess

STRINGS, MATH = 101, 102  # the sample ids of a strings problem and of a math one, in raw_dev
# A value for every field that differs from the fixture's defaults and from the other fields', so
# that a parameter name paired with another field shows.
DISTINCT = {
    "classifier": "linear_svm",
    "c": 0.3,
    "class_weight": "balanced",
    "include_notes": True,
    "statement_ngram_max": 2,
    "statement_min_df": 2,
    "statement_max_df": 0.9,
    "statement_sublinear_tf": True,
    "code_weight": 0.7,
    "code_ngram_max": 2,
    "code_min_df": 3,
    "code_sublinear_tf": True,
    "cue_weight": 0.4,
}
ALL_STEPS = ["raw fields", "statement", "normalised statement", "code tokens", "sample I/O cues"]


@pytest.fixture(name="raw_dev")
def raw_dev_fixture(raw_problems):
    """Return 18 raw dev problems: six copies of the three of ``raw_problems``, in turn.

    Sample ids 100-117: a trees problem, a strings problem (101) and a math and number theory
    problem (102), six times. Each of the three folds holds two copies of each problem.
    """
    position = np.arange(18)
    copies = pd.concat([raw_problems] * 6).set_axis(pd.Index(100 + position, name="sample_id"))
    return copies.assign(fold=position // 3 % 3, group=position)


@pytest.fixture(name="labels")
def labels_fixture(raw_dev):
    """Return the 0/1 labels of the tags of the problems: math, number theory, strings, trees."""
    return multi_hot(raw_dev["tags"])


@pytest.fixture(name="make_tag_model")
def make_tag_model_fixture(raw_dev, labels, make_params):
    """Return a builder of tag models fitted on every problem, parameters overridable by keyword.

    Bigrams and the notes in the statement block, every term kept, all three blocks unless the
    keywords say otherwise.
    """

    def make(**overrides):
        params = make_params(include_notes=True, statement_ngram_max=2, **overrides)
        return fit_tag_model(params, raw_dev, labels, raw_dev["fold"])

    return make


@pytest.fixture(name="tag_model")
def tag_model_fixture(make_tag_model):
    """Return the tag model that keeps all three blocks."""
    return make_tag_model()


def test_the_parameter_table_has_a_row_per_field_of_model_params_in_field_order(make_params):
    params = make_params()

    table = parameter_table(params)

    assert table.index.name == "parameter"
    assert table.index.tolist() == [field.name for field in fields(ModelParams)]
    assert table.columns.tolist() == ["value", "sets"]
    expected = {field.name: getattr(params, field.name) for field in fields(ModelParams)}
    assert table["value"].to_dict() == expected


def test_each_parameter_names_the_estimator_parameter_that_holds_its_value(make_params):
    params = make_params(**DISTINCT)
    table = parameter_table(params)
    held = CrossFittedTagThresholds(build_pipeline(params), refit=False).get_params()

    def held_by(field):
        return held[table.at[field, "sets"]]

    assert isinstance(held_by("classifier"), LinearSVC)
    assert held_by("c") == 0.3
    assert held_by("class_weight") == "balanced"
    assert held_by("include_notes") == {"include_notes": True}
    assert held_by("statement_ngram_max") == (1, 2)
    assert held_by("statement_min_df") == 2
    assert held_by("statement_max_df") == 0.9
    assert held_by("statement_sublinear_tf") is True
    assert held_by("code_weight")["code"] == 0.7
    assert held_by("code_ngram_max") == (1, 2)
    assert held_by("code_min_df") == 3
    assert held_by("code_sublinear_tf") is True
    assert held_by("cue_weight")["cues"] == 0.4
    assert [name for name, _, _ in held_by("blocks")] == ["statement", "code", "cues"]


def test_the_block_weights_are_set_through_the_transformer_weights_of_the_feature_blocks(
    make_params,
):
    table = parameter_table(make_params())

    assert table.at["code_weight", "sets"] == "estimator__features__transformer_weights"
    assert table.at["cue_weight", "sets"] == "estimator__features__transformer_weights"


@pytest.mark.parametrize(
    ("blocks", "not_used"),
    [
        (("statement", "code", "cues"), []),
        (("statement", "code"), ["cue_weight"]),
        (
            ("statement", "cues"),
            ["code_weight", "code_ngram_max", "code_min_df", "code_sublinear_tf"],
        ),
        (
            ("code", "cues"),
            [
                "include_notes",
                "statement_ngram_max",
                "statement_min_df",
                "statement_max_df",
                "statement_sublinear_tf",
            ],
        ),
    ],
)
def test_a_parameter_of_a_block_that_is_not_kept_is_not_used(make_params, blocks, not_used):
    params = make_params(blocks=blocks)

    table = parameter_table(params)

    assert table.index[table["sets"] == NOT_USED].tolist() == not_used
    assert NOT_USED == "not used: block not kept"
    assert table.at["cue_weight", "value"] == params.cue_weight  # its value is shown all the same


def test_trace_problem_traces_one_problem(tag_model, raw_dev):
    for problems in (raw_dev.iloc[:0], raw_dev.iloc[:2]):
        with pytest.raises(ValueError, match=f"traces one problem, not {len(problems)}"):
            trace_problem(tag_model, problems)


def test_trace_problem_needs_the_tags_of_the_problem(tag_model, raw_dev):
    without_tags = raw_dev.loc[[MATH]].drop(columns="tags")

    with pytest.raises(ValueError, match="tags column"):
        trace_problem(tag_model, without_tags)


def test_trace_problem_refuses_a_problem_that_is_already_preprocessed(tag_model, raw_dev):
    with pytest.raises(ValueError, match="already preprocessed"):
        trace_problem(tag_model, preprocess(raw_dev.loc[[MATH]]))


def test_the_steps_follow_a_problem_from_its_raw_fields_through_the_three_blocks(
    tag_model, raw_dev
):
    raw = raw_dev.loc[[STRINGS]]
    problem = preprocess(raw).iloc[0]

    steps = trace_problem(tag_model, raw).steps

    assert list(steps) == ALL_STEPS
    pd.testing.assert_series_equal(steps["raw fields"], raw.iloc[0][list(RAW_INPUT_COLUMNS)])
    assert steps["statement"] == f"{problem['statement']}\n\n{problem['prob_desc_notes']}"
    assert steps["normalised statement"] == normalise_statement(steps["statement"])
    assert steps["normalised statement"].startswith("reverse each of the strings. some strings.")
    assert steps["code tokens"] == ["input", "print"]  # one-letter names drop out
    cues = steps["sample I/O cues"]
    assert cues.to_dict() == dict.fromkeys(SAMPLE_IO_CUES, 0) | {"letters_output": 1}
    assert pd.api.types.is_integer_dtype(cues)  # yes/no facts: 1, not 1.0


def test_a_missing_input_specification_stays_missing_in_the_raw_fields_and_the_statement(
    tag_model, raw_dev
):
    steps = trace_problem(tag_model, raw_dev.loc[[MATH]]).steps

    assert pd.isna(steps["raw fields"]["prob_desc_input_spec"])
    expected = "Print the gcd, a number theory classic.\n\nOne integer.\n\nUse math.gcd."
    assert steps["statement"] == expected


@pytest.mark.parametrize(
    ("blocks", "steps"),
    [
        (("statement", "code"), ALL_STEPS[:4]),
        (("statement",), ALL_STEPS[:3]),
        (("code", "cues"), ["raw fields", "code tokens", "sample I/O cues"]),
    ],
)
def test_the_steps_of_a_block_that_is_not_kept_are_left_out(make_tag_model, raw_dev, blocks, steps):
    trace = trace_problem(make_tag_model(blocks=blocks), raw_dev.loc[[STRINGS]])

    assert list(trace.steps) == steps
    assert list(trace.block_values) == list(blocks)


def test_the_block_values_scaled_by_the_block_weights_are_the_pipelines_feature_row(
    tag_model, raw_dev
):
    raw = raw_dev.loc[[STRINGS]]
    trace = trace_problem(tag_model, raw)
    features = tag_model.pipeline.named_steps["features"]
    row = tag_model.pipeline[:2].transform(raw).toarray()[0]

    rebuilt = np.zeros_like(row)
    for block, values in trace.block_values.items():
        names = block_feature_names(features, block)
        first = features.output_indices_[block].start
        for feature, value in values.items():
            rebuilt[first + names.index(feature)] = value * features.transformer_weights[block]

    np.testing.assert_allclose(rebuilt, row, rtol=1e-12)
    assert list(trace.block_values) == ["statement", "code", "cues"]
    assert all(values.gt(0).all() for values in trace.block_values.values())


def test_features_list_a_blocks_non_zero_values_largest_first(tag_model, raw_dev):
    trace = trace_problem(tag_model, raw_dev.loc[[MATH]])

    statement = trace.features("statement")

    assert statement.is_monotonic_decreasing
    assert statement.index.name == "feature"
    assert {"gcd", "number theory"} <= set(statement.index)
    assert set(statement.index) == set(trace.block_values["statement"].index)
    assert trace.features("code").to_dict() == {"print": 1.0}


def test_features_of_the_cue_block_are_the_cues_that_fire(tag_model, raw_dev):
    cues = trace_problem(tag_model, raw_dev.loc[[STRINGS]]).features("cues")

    assert cues.to_dict() == {"letters_output": 1.0}


def test_features_refuse_a_block_the_pipeline_does_not_keep(make_tag_model, raw_dev):
    trace = trace_problem(make_tag_model(blocks=("statement", "code")), raw_dev.loc[[MATH]])

    with pytest.raises(ValueError, match=r"keeps no 'cues' block.*\['statement', 'code'\]"):
        trace.features("cues")


def test_the_weights_are_the_tag_models_feature_weights(tag_model, raw_dev):
    trace = trace_problem(tag_model, raw_dev.loc[[MATH]])

    pd.testing.assert_frame_equal(trace.weights, tag_model.feature_weights())


def test_the_contributions_to_a_tag_are_value_times_weight_and_add_up_to_its_score(
    tag_model, raw_dev
):
    trace = trace_problem(tag_model, raw_dev.loc[[MATH]])

    for tag in tag_model.tags:
        table = trace.contributions(tag)

        assert table.index.names == ["block", "feature"]
        assert table.columns.tolist() == ["value", "weight", "contribution"]
        np.testing.assert_array_equal(table["contribution"], table["value"] * table["weight"])
        assert table["contribution"].is_monotonic_decreasing
        weights = trace.weights[tag].loc[table.index]
        np.testing.assert_array_equal(table["weight"], weights)
        decision = trace.decision.loc[tag]
        total = table["contribution"].sum() + decision["intercept"]
        assert total == pytest.approx(decision["score"], rel=1e-9)


def test_the_contributions_of_a_tag_the_model_does_not_predict_are_refused(tag_model, raw_dev):
    trace = trace_problem(tag_model, raw_dev.loc[[MATH]])

    with pytest.raises(ValueError, match=r"predicts no tag 'dp'.*\['math', 'number theory'"):
        trace.contributions("dp")


def test_the_decision_gives_each_tags_score_intercept_threshold_prediction_and_truth(
    tag_model, raw_dev
):
    raw = raw_dev.loc[[MATH]]

    decision = trace_problem(tag_model, raw).decision

    assert decision.index.tolist() == ["math", "number theory", "strings", "trees"]
    assert decision.index.name == "label"
    assert decision.columns.tolist() == ["score", "intercept", "threshold", "predicted", "true"]
    np.testing.assert_array_equal(decision["score"], tag_model.scores(raw).iloc[0])
    np.testing.assert_array_equal(decision["threshold"], tag_model.thresholds)
    np.testing.assert_array_equal(decision["predicted"], tag_model.predict(raw).iloc[0])
    assert (decision["score"] >= decision["threshold"]).astype(int).equals(decision["predicted"])
    assert decision["true"].to_dict() == {"math": 1, "number theory": 1, "strings": 0, "trees": 0}


def test_the_decision_tells_the_prediction_from_the_truth(tag_model, raw_dev):
    # The model predicts the problems it was fitted on as they are tagged. With no threshold
    # within reach, it predicts no tag, while the math problem keeps its two.
    tag_model.estimator.thresholds_ = np.full_like(tag_model.estimator.thresholds_, np.inf)

    decision = trace_problem(tag_model, raw_dev.loc[[MATH]]).decision

    assert decision["predicted"].eq(0).all()
    assert decision["true"].to_dict() == {"math": 1, "number theory": 1, "strings": 0, "trees": 0}


def test_the_intercept_of_a_tag_is_its_score_when_no_feature_is_present(tag_model, raw_dev):
    raw = raw_dev.loc[[MATH]]
    columns = tag_model.pipeline[:2].transform(raw).shape[1]
    classifier = tag_model.pipeline.named_steps["classifier"]

    decision = trace_problem(tag_model, raw).decision

    nothing_present = classifier.decision_function(np.zeros((1, columns)))[0]
    np.testing.assert_allclose(decision["intercept"], nothing_present)


def test_a_problem_none_of_whose_features_is_present_scores_its_intercept(make_tag_model, raw_dev):
    # No sample I/O cue fires on the math problem.
    trace = trace_problem(make_tag_model(blocks=("cues",)), raw_dev.loc[[MATH]])

    assert trace.features("cues").empty
    assert all(trace.contributions(tag).empty for tag in trace.decision.index)
    np.testing.assert_array_equal(trace.decision["score"], trace.decision["intercept"])
