"""Tests for the F2LLM tagging methods: the method, its folder, its loader, its registry lines."""

import dataclasses
import json
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from skops.io.exceptions import UntrustedTypesFoundException

from codeforces_tags.cache import ArrayCache
from codeforces_tags.config import DEVICE_ENV_VAR, FOCUS_TAGS
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MethodSpec,
    ScoreMethod,
    load_method,
    save_method,
)
from codeforces_tags.methods.f2llm.embedding import live_feature_frame
from codeforces_tags.methods.f2llm.encoder import (
    ENCODER_FILE,
    PRETRAINED_FILE,
    EncoderConfig,
    PretrainedRecord,
    frozen_encoder,
)
from codeforces_tags.methods.f2llm.estimators import make_estimator
from codeforces_tags.methods.f2llm.method import (
    MODEL_FILE,
    THRESHOLDS_FILE,
    TRUSTED_TYPES,
    F2LLMMethod,
    from_fit,
    load,
    setfit_adapter,
)
from codeforces_tags.methods.f2llm.setfit import (
    ADAPTER_FILE,
    ADAPTER_RECORD_FILE,
    SetFitEstimator,
    adapter_record,
)
from codeforces_tags.methods.f2llm.settings import (
    EARLY_FUSION,
    EARLY_FUSION_GRID,
    LATE_FUSION,
    NAMES,
    PROBE,
    PROBE_GRID,
    SETFIT,
    SETFIT_METHOD,
)
from codeforces_tags.model.params import ModelParams
from codeforces_tags.model.thresholds import CrossFittedTagThresholds

# A configuration of each grid with both inputs, so that every embedding block is fitted.
PROBE_PARAMS = next(p for p in PROBE_GRID if p["c"] == 1 and p["solution_weight"] == 0.5)
# The configuration of each method's estimator in these tests.
PARAMS = {
    PROBE: PROBE_PARAMS,
    LATE_FUSION: {"weight": 0.5},
    EARLY_FUSION: next(p for p in EARLY_FUSION_GRID if p["c"] == 1 and p["solution_weight"] == 0.5),
}
# SetFit's settings with a handful of pairs: a short training of the tiny model.
SMOKE_SETFIT = SETFIT | {"pairs_per_tag_positive": 2, "batch_pairs": 4, "micro_batch_pairs": 2}
# The distributions every F2LLM method needs, as its registry line names them.
REQUIREMENTS = "numpy pandas peft safetensors scikit-learn scipy skops torch transformers"
# The problems check_tagging_method scores in a batch, then one at a time.
CHECKED = 6
# The tolerance of the tiny model's scores in these tests: its batch-versus-single CPU drift,
# whatever score_atol the registry holds (the atol command measures the GPU-to-CPU value
# written there, which may be smaller).
TINY_ATOL = 1e-4


def tiny_registry():
    """Return the registry with the four F2LLM lines at the tests' tolerance, ``TINY_ATOL``."""
    tiny = {name: dataclasses.replace(METHODS[name], score_atol=TINY_ATOL) for name in NAMES}
    return {**METHODS, **tiny}


class Tiny(NamedTuple):
    """The synthetic dev problems and their labels, with the tiny model's files and constants.

    ``tfidf`` is the conftest's ``small_tfidf``: the frozen TF-IDF's blocks (statement and code,
    no cues), with document-frequency cuts that keep the small vocabulary of the problems.
    """

    raw: pd.DataFrame
    labels: pd.DataFrame
    pretrained: PretrainedRecord
    config: EncoderConfig
    tfidf: ModelParams


@pytest.fixture(name="tiny")
def tiny_fixture(synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config, small_tfidf):
    """Return the synthetic problems, their focus labels, the tiny model and a TF-IDF."""
    return Tiny(synthetic_raw, synthetic_labels, tiny_pretrained, tiny_config, small_tfidf)


def make_test_estimator(name, width, tiny):
    """Return the unfitted score estimator of a method, on the tiny model."""
    if name == SETFIT_METHOD:
        return SetFitEstimator(
            head_params=PROBE_PARAMS,
            settings=SMOKE_SETFIT,
            device="cpu",
            pretrained=tiny.pretrained.to_dict(),
            encoder_config=tiny.config.to_dict(),
            resume=False,
        )
    return make_estimator(
        name, PARAMS[name], width=width, probe_params=PROBE_PARAMS, tfidf=tiny.tfidf
    )


def fit(name, tiny, *, labels=None, refit=True):
    """Fit a method's ``CrossFittedTagThresholds`` on the live feature frame of the problems.

    Returns:
        The fitted ``CrossFittedTagThresholds``, and the feature frame it was fitted on.
    """
    encoder = frozen_encoder(tiny.pretrained, tiny.config, "cpu")
    frame = live_feature_frame(tiny.raw, encoder, encoder)
    model = CrossFittedTagThresholds(make_test_estimator(name, encoder.width, tiny), refit=refit)
    labels = tiny.labels if labels is None else labels
    return model.fit(frame, labels, groups=frame["fold"]), frame


@pytest.fixture(name="probe")
def probe_fixture(tiny):
    """Return an ``f2llm_probe`` fitted on the synthetic problems."""
    fitted, _ = fit(PROBE, tiny)
    return from_fit(PROBE, fitted, tiny.config, tiny.pretrained)


def rename_method(folder, name):
    """Rewrite the method name that a saved ``method.json`` stores."""
    path = folder / METHOD_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(stored | {"method": name}), encoding="utf-8")


def test_trusted_types_are_pinned_per_method():
    functions = [
        "codeforces_tags.model.features.code_tokens",
        "codeforces_tags.model.features.normalise_statement",
        "codeforces_tags.model.features.statement_text",
    ]
    expected = {
        EARLY_FUSION: functions,
        LATE_FUSION: ["codeforces_tags.methods.f2llm.estimators.ScoreFusion", *functions],
        PROBE: [],
        SETFIT_METHOD: [],
    }

    assert expected == TRUSTED_TYPES
    assert list(TRUSTED_TYPES) == list(NAMES)


def test_the_four_methods_are_registered_alike_as_score_methods_of_the_f2llm_group():
    atol = METHODS[PROBE].score_atol
    expected = MethodSpec(
        module="codeforces_tags.methods.f2llm.method",
        kind="score",
        requirements=tuple(REQUIREMENTS.split()),
        groups=("f2llm",),
        score_atol=atol,
    )

    assert {name: METHODS[name] for name in NAMES} == dict.fromkeys(NAMES, expected)
    assert 0 < atol < 1e-2  # the GPU-to-CPU drift, measured before the first freeze


@pytest.mark.parametrize("name", NAMES)
def test_each_method_passes_check_tagging_method_without_any_cache(
    name, tiny, tmp_path, monkeypatch
):
    fitted, frame = fit(name, tiny)
    method = from_fit(name, fitted, tiny.config, tiny.pretrained)
    problems, folder = tiny.raw.iloc[:CHECKED], tmp_path / "model"
    # The method, fitted or loaded, encodes live: with the cache's methods gone, any use fails.
    monkeypatch.delattr(ArrayCache, "get_or_compute")
    monkeypatch.delattr(ArrayCache, "get_or_compute_many")
    reads = ArrayCache.reads

    check_tagging_method(method, problems, folder, registry=tiny_registry())

    assert ArrayCache.reads == reads
    expected = fitted.decision_function(frame.iloc[:CHECKED])
    np.testing.assert_allclose(method.scores(problems), expected, rtol=0, atol=TINY_ATOL)
    adapter = [ADAPTER_FILE, ADAPTER_RECORD_FILE] if name == SETFIT_METHOD else []
    files = [ENCODER_FILE, METHOD_FILE, MODEL_FILE, PRETRAINED_FILE, THRESHOLDS_FILE, *adapter]
    assert sorted(path.name for path in folder.iterdir()) == sorted(files)
    assert sio.get_untrusted_types(file=folder / MODEL_FILE) == TRUSTED_TYPES[name]


def test_from_fit_predicts_with_the_thresholds_of_the_fit(tiny):
    fitted, frame = fit(PROBE, tiny)

    method = from_fit(PROBE, fitted, tiny.config, tiny.pretrained)

    assert isinstance(method, ScoreMethod)
    assert (method.name, method.tags, method.device) == (PROBE, FOCUS_TAGS, "cpu")
    np.testing.assert_array_equal(method.thresholds, fitted.thresholds_)
    predictions = method.predict(tiny.raw)
    assert (predictions.dtypes == "int64").all()
    np.testing.assert_array_equal(predictions, fitted.predict(frame))


def test_from_fit_refuses_a_fit_without_refit(tiny):
    fitted, _ = fit(PROBE, tiny, refit=False)

    with pytest.raises(ValueError, match="refit=True"):
        from_fit(PROBE, fitted, tiny.config, tiny.pretrained)


def test_from_fit_refuses_labels_other_than_the_focus_tags_in_order(tiny):
    fitted, _ = fit(PROBE, tiny, labels=tiny.labels[list(reversed(FOCUS_TAGS))])

    with pytest.raises(ValueError, match="in FOCUS_TAGS order"):
        from_fit(PROBE, fitted, tiny.config, tiny.pretrained)


@pytest.mark.parametrize(
    ("name", "adapter", "thresholds", "match"),
    [
        ("label_prior", None, np.zeros(8), "'label_prior' is not an F2LLM method"),
        (PROBE, ({}, {}), np.zeros(8), "f2llm_probe takes no adapter"),
        (SETFIT_METHOD, None, np.zeros(8), "f2llm_setfit needs an adapter"),
        (PROBE, None, np.zeros(7), r"one threshold per focus tag, 8, not an array of shape \(7,\)"),
    ],
)
def test_f2llm_method_refuses_a_name_adapter_or_thresholds_that_do_not_fit(
    tiny, name, adapter, thresholds, match
):
    with pytest.raises(ValueError, match=match):
        F2LLMMethod(name, None, thresholds, tiny.config, tiny.pretrained, adapter=adapter)


def test_scores_and_predictions_of_no_problem_are_empty_frames(probe, tiny):
    none = tiny.raw.iloc[:0]

    scores, predictions = probe.scores(none), probe.predict(none)

    assert scores.shape == predictions.shape == (0, len(FOCUS_TAGS))
    assert (scores.dtypes == "float64").all()
    assert (predictions.dtypes == "int64").all()


def test_thresholds_json_holds_each_focus_tag_threshold_in_order(probe, tmp_path):
    save_method(probe, tmp_path)

    stored = json.loads((tmp_path / THRESHOLDS_FILE).read_text(encoding="utf-8"))

    assert list(stored) == list(FOCUS_TAGS)
    assert list(stored.values()) == probe.thresholds.tolist()


def test_load_refuses_thresholds_of_other_tags(probe, tmp_path):
    save_method(probe, tmp_path)
    reordered = dict(zip(reversed(FOCUS_TAGS), probe.thresholds.tolist(), strict=True))
    (tmp_path / THRESHOLDS_FILE).write_text(json.dumps(reordered), encoding="utf-8")

    with pytest.raises(ValueError, match="not the focus tags in order"):
        load(tmp_path)


def test_load_dispatches_on_the_name_that_method_json_stores(probe, tiny, tmp_path):
    save_method(probe, tmp_path)
    rename_method(tmp_path, EARLY_FUSION)  # trusts more types than the probe's model holds

    loaded = load(tmp_path)

    assert (loaded.name, loaded.adapter) == (EARLY_FUSION, None)
    pd.testing.assert_frame_equal(loaded.scores(tiny.raw), probe.scores(tiny.raw))


def test_load_refuses_a_folder_of_another_method(probe, tmp_path):
    save_method(probe, tmp_path)
    rename_method(tmp_path, "label_prior")

    with pytest.raises(ValueError, match="'label_prior', not one of the F2LLM methods"):
        load(tmp_path)


def test_load_trusts_only_the_skops_types_of_the_stored_method(tiny, tmp_path):
    fitted, _ = fit(EARLY_FUSION, tiny)
    save_method(from_fit(EARLY_FUSION, fitted, tiny.config, tiny.pretrained), tmp_path)
    rename_method(tmp_path, PROBE)  # the probe trusts no type beyond skops's defaults

    with pytest.raises(UntrustedTypesFoundException, match="statement_text"):
        load(tmp_path)


def test_load_encodes_on_the_device_that_the_environment_names(probe, tmp_path, monkeypatch):
    save_method(probe, tmp_path)
    monkeypatch.setenv(DEVICE_ENV_VAR, "tpu")

    with pytest.raises(ValueError, match="tpu"):
        load_method(tmp_path)


def test_the_probe_shares_the_frozen_encoder_and_setfit_merges_its_adapter(probe, tiny):
    fitted, _ = fit(SETFIT_METHOD, tiny)

    setfit = from_fit(SETFIT_METHOD, fitted, tiny.config, tiny.pretrained)

    frozen = frozen_encoder(tiny.pretrained, tiny.config, "cpu").model
    assert probe.statement_encoder.model is probe.solution_encoder.model is frozen
    assert setfit.solution_encoder.model is frozen
    assert setfit.statement_encoder.model is not frozen
    assert setfit.adapter[1] == adapter_record(SMOKE_SETFIT)
    assert probe.reads_solution and setfit.reads_solution  # both heads keep it at weight 0.5


def test_a_method_whose_scorer_drops_the_solution_never_encodes_one(tiny, monkeypatch):
    monkeypatch.setitem(PARAMS, PROBE, PROBE_PARAMS | {"solution_weight": 0.0})
    fitted, _ = fit(PROBE, tiny)
    method = from_fit(PROBE, fitted, tiny.config, tiny.pretrained)
    encoded = []
    encode = method.solution_encoder.encode  # the probe's statement encoder too

    def recording_encode(texts):
        encoded.extend(texts)
        return encode(texts)

    monkeypatch.setattr(method.solution_encoder, "encode", recording_encode)

    scores = method.scores(tiny.raw)

    assert not method.reads_solution
    assert scores.shape == (len(tiny.raw), len(FOCUS_TAGS))
    assert len(encoded) == len(tiny.raw)  # one input per problem: its statement
    assert all(text.startswith(tiny.config.statement_instruction) for text in encoded)


@pytest.mark.parametrize(("device", "calls"), [("cpu", "one per input"), ("cuda", "one per text")])
def test_on_the_gpu_scores_encodes_each_problem_in_its_own_call(
    probe, tiny, monkeypatch, device, calls
):
    batched = probe.scores(tiny.raw)
    sizes = []
    encode = probe.solution_encoder.encode  # the probe's statement encoder too

    def recording_encode(texts):
        sizes.append(len(texts))
        return encode(texts)

    monkeypatch.setattr(probe.solution_encoder, "encode", recording_encode)
    monkeypatch.setattr(probe, "device", device)  # the tiny encoder stays on the CPU

    scores = probe.scores(tiny.raw)

    expected = [len(tiny.raw)] * 2 if calls == "one per input" else [1] * (2 * len(tiny.raw))
    assert sizes == expected  # the statement, then the solution, of each call
    pd.testing.assert_index_equal(scores.index, tiny.raw.index)
    np.testing.assert_allclose(scores, batched, rtol=0, atol=TINY_ATOL)


def test_setfit_adapter_records_setfit_settings_unless_the_estimator_has_its_own():
    state = {}
    default = SimpleNamespace(settings=None, adapter_state_=state)
    smaller = SimpleNamespace(settings=SETFIT | {"lora_r": 4}, adapter_state_=state)

    assert setfit_adapter(default)[0] is state
    assert setfit_adapter(default)[1] == adapter_record(SETFIT)
    assert setfit_adapter(smaller)[1] == adapter_record(SETFIT | {"lora_r": 4})
