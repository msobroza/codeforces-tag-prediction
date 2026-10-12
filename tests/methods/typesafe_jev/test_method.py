"""Tests for the TypeSafe Jev tagging method: its scorer, thresholds, predictions and folder."""

import asyncio
import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.conformance import check_tagging_method
from codeforces_tags.llm.answers import answer_cost_usd
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.methods import METHODS, MethodSpec, installed_methods, save_method
from codeforces_tags.methods.typesafe_jev import questions
from codeforces_tags.methods.typesafe_jev.method import (
    JevProbabilities,
    JevTagger,
    fitted_thresholds,
    load,
)
from codeforces_tags.methods.typesafe_jev.settings import (
    FROZEN_NAMES,
    GEPA_NAME,
    METHOD_FILE_NAME,
    MODEL,
    NAMES,
    PLATT_BASES,
    SETTINGS,
    WORDS_FILE_NAME,
)
from codeforces_tags.methods.typesafe_jev.words import SEED_WORDS, method_settings, words_from_json
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.provenance import sha256_json

# What the fake Jev (conftest.py) answers for the raw problems of tests/conftest.py, in
# FOCUS_TAGS order: problem 30 counts the leaves of a tree, problem 20 prints a gcd, and
# problem 10 holds no word of a focus tag.
FAKE_PROBABILITIES = {
    30: [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],
    10: [0.1] * 8,
    20: [0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1],
}


def tagger(name, thresholds=(0.5,) * 8, words=None):
    """Return the Jev method ``name`` whose fitted wrapper holds ``thresholds``, with ``words``."""
    return JevTagger(name, fitted_thresholds(list(thresholds), FOCUS_TAGS), words)


def searched_words():
    """Return the words of a searched method: the seed's, with another games definition."""
    values = SEED_WORDS.settings()
    values["tag_definitions"]["games"]["true"]["definition"] = "Two players play to win."
    return words_from_json(values)


def dev_set():
    """Return ten dev problems in five folds: Jev's probabilities, the labels and the folds.

    A problem has a tag when its probability is at least 0.5, but no problem has geometry,
    whose threshold is then inf.
    """
    index = pd.Index(range(1, 11), name="sample_id")
    values = np.random.default_rng(0).random((10, len(FOCUS_TAGS))).round(2)
    found = pd.DataFrame(values, index=index, columns=list(FOCUS_TAGS))
    labels = (found >= 0.5).astype(np.int64)
    labels["geometry"] = 0
    return found, labels, pd.Series(np.arange(10) // 2, index=index, name="fold")


def test_jev_probabilities_learn_nothing_and_return_the_tag_columns():
    found, labels, _ = dev_set()
    # The focus tags in another order, beside a column that is not a tag.
    shuffled = found[list(reversed(FOCUS_TAGS))].assign(statement="Count the leaves.")

    scorer = clone(JevProbabilities()).fit(shuffled, labels.to_numpy())

    assert scorer.get_params() == {}
    assert scorer.classes_.tolist() == list(range(len(FOCUS_TAGS)))
    assert JevProbabilities().fit(found, labels).classes_.tolist() == scorer.classes_.tolist()
    assert scorer.decision_function(shuffled).dtype == np.float64
    assert np.array_equal(scorer.decision_function(shuffled), found.to_numpy())
    assert np.array_equal(scorer.predict(shuffled), np.ones(found.shape))


def test_cross_fitted_thresholds_score_jev_probabilities_out_of_fold_as_they_are():
    found, labels, folds = dev_set()

    wrapper = CrossFittedTagThresholds(JevProbabilities()).fit(found, labels, groups=folds)

    assert np.array_equal(wrapper.oof_scores_, found.to_numpy())
    assert np.array_equal(wrapper.decision_function(found), found.to_numpy())
    assert np.isinf(wrapper.thresholds_).tolist() == [tag == "geometry" for tag in FOCUS_TAGS]
    assert wrapper.tags_ == FOCUS_TAGS


def test_fitted_thresholds_rebuild_the_wrapper_fitted_at_freeze():
    found, labels, folds = dev_set()
    fitted = CrossFittedTagThresholds(JevProbabilities()).fit(found, labels, groups=folds)

    rebuilt = fitted_thresholds(fitted.thresholds_.tolist(), FOCUS_TAGS)

    assert np.array_equal(rebuilt.thresholds_, fitted.thresholds_)
    assert rebuilt.tags_ == fitted.tags_
    assert np.array_equal(rebuilt.estimator_.classes_, fitted.estimator_.classes_)
    assert np.array_equal(rebuilt.decision_function(found), fitted.decision_function(found))
    assert np.array_equal(rebuilt.predict(found), fitted.predict(found))


def test_fitted_thresholds_refuse_a_length_other_than_the_tags():
    with pytest.raises(ValueError, match="7 thresholds for 8 tags"):
        fitted_thresholds([0.5] * 7, FOCUS_TAGS)


def test_scores_and_predict_ask_one_request_per_problem_in_one_session(jev_api, raw_problems):
    method = tagger("jev-statement")

    scores = method.scores(raw_problems)
    predictions = method.predict(raw_problems)

    expected = pd.DataFrame(
        [FAKE_PROBABILITIES[sample_id] for sample_id in raw_problems.index],
        index=raw_problems.index,
        columns=list(FOCUS_TAGS),
    )
    pd.testing.assert_frame_equal(scores, expected)
    pd.testing.assert_frame_equal(predictions, (expected >= 0.5).astype(int))
    assert jev_api.sessions == ["jev-statement", "jev-statement"]
    assert jev_api.sent == [("jev-statement", sample_id) for sample_id in (30, 10, 20)] * 2


@pytest.mark.usefixtures("jev_api")
def test_predict_counts_a_probability_equal_to_its_threshold_as_a_tag(raw_problems):
    # Problems 30 and 20 have 0.8 of trees and of number theory: the threshold of trees, and
    # just under that of number theory.
    thresholds = dict.fromkeys(FOCUS_TAGS, 0.8) | {"number theory": math.nextafter(0.8, 1.0)}

    predictions = tagger("jev-statement", thresholds.values()).predict(raw_problems)

    assert predictions.loc[30, "trees"] == 1
    assert predictions.loc[20, "number theory"] == 0
    assert predictions.to_numpy().sum() == 1


def test_a_failed_answer_scores_0_and_meets_a_0_threshold(jev_api, raw_problems):
    # A line that is not completed scores 0.0 on every tag, which a threshold of 0.0 counts as
    # the tag: problem 10's math.
    method = tagger("jev-statement", [0.0] + [0.5] * 7)
    jev_api.outcomes[10] = {"status": "api_error:422"}

    predictions, answers = asyncio.run(method.apredict_with_answers(raw_problems))
    jev_api.outcomes[10] = {"status": "malformed"}
    scores = method.scores(raw_problems)

    assert [answer.status for answer in answers] == ["completed", "api_error:422", "completed"]
    assert scores.loc[10].tolist() == [0.0] * 8
    assert predictions.loc[10].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_a_prediction_raises_the_first_error_in_order_once_every_request_ended(
    jev_api, raw_problems
):
    # A live prediction has no answers file to come back to: it fails rather than skip a
    # problem, and only once every request has ended.
    jev_api.outcomes[10] = TransientApiError("TypeSafe did not answer problem 10.")
    jev_api.outcomes[20] = RuntimeError("problem 20 failed too")

    with pytest.raises(TransientApiError, match="problem 10"):
        tagger("jev-statement").predict(raw_problems)

    assert jev_api.sent == [("jev-statement", sample_id) for sample_id in (30, 10, 20)]


def test_apredict_with_answers_returns_the_predictions_and_one_answer_per_problem(
    jev_api, raw_problems
):
    method = tagger("jev-statement")

    predictions, answers = asyncio.run(method.apredict_with_answers(raw_problems))
    # As tracking latency calls it: one problem, whose one answer gives its tokens and cost.
    _, [answer] = asyncio.run(method.apredict_with_answers(raw_problems.iloc[[1]]))

    requests = questions.requests(raw_problems, "jev-statement")
    assert [line.request_sha256 for line in answers] == [request.sha256() for request in requests]
    assert [line.sample_id for line in answers] == [30, 10, 20]
    pd.testing.assert_frame_equal(predictions, method.predict(raw_problems))
    assert (answer.sample_id, answer.method, answer.status) == (10, "jev-statement", "completed")
    assert answer_cost_usd(answer)["total_cost"] == pytest.approx(2500 * 0.042 / 1e6)
    assert jev_api.sessions == ["jev-statement"] * 3


def test_scores_from_answers_scores_the_answers_already_received_without_a_request(
    jev_api, raw_problems
):
    # The command line reads a prediction's scores from the answers that gave its tags: a second
    # request would cost again, and Jev would answer it with a new sample.
    method = tagger("jev-statement")
    jev_api.outcomes[10] = {"status": "api_error:422"}
    predictions, answers = asyncio.run(method.apredict_with_answers(raw_problems))

    scores = method.scores_from_answers(answers, raw_problems.index)

    expected = pd.DataFrame(
        [FAKE_PROBABILITIES[sample_id] for sample_id in raw_problems.index],
        index=raw_problems.index,
        columns=list(FOCUS_TAGS),
    )
    expected.loc[10] = 0.0
    pd.testing.assert_frame_equal(scores, expected)
    pd.testing.assert_frame_equal(predictions, (scores >= 0.5).astype(int))
    assert len(jev_api.sent) == 3


def test_apredict_with_answers_opens_a_client_per_call_in_its_own_loop(
    server, noul_response, raw_problems
):
    # tracking latency runs each call in an event loop of its own: the SDK's client opens and
    # closes in it, never in the loop of the call before.
    low = dict.fromkeys(FOCUS_TAGS, 0.1)
    server.outcomes.extend(
        [noul_response(low | {"trees": 0.9}), noul_response(low | {"strings": 0.7})]
    )
    method = tagger("jev-statement")

    first, [one] = asyncio.run(method.apredict_with_answers(raw_problems.iloc[[0]]))
    second, [two] = asyncio.run(method.apredict_with_answers(raw_problems.iloc[[1]]))

    assert (len(server.clients), server.closed) == (2, 2)
    assert [call["model"] for call in server.calls] == [MODEL, MODEL]
    assert [(line.sample_id, line.status) for line in (one, two)] == [
        (30, "completed"),
        (10, "completed"),
    ]
    assert first.loc[30].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    assert second.loc[10].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]


def test_no_problem_opens_no_session(jev_api, raw_problems):
    method = tagger("jev-statement")
    nothing = raw_problems.iloc[:0]

    predictions, answers = asyncio.run(method.apredict_with_answers(nothing))
    scores = method.scores(nothing)

    assert (jev_api.sessions, jev_api.sent, answers) == ([], [], [])
    assert predictions.shape == scores.shape == (0, len(FOCUS_TAGS))
    assert list(predictions.columns) == list(scores.columns) == list(FOCUS_TAGS)
    assert (predictions.dtypes == "int64").all()
    assert (scores.dtypes == "float64").all()


def test_save_and_load_keep_the_name_and_an_infinite_threshold(tmp_path):
    thresholds = [0.45, 0.0, 0.45, 0.5, 0.45, math.inf, 0.45, math.inf]
    tagger("jev-statement-code", thresholds).save(tmp_path)

    text = (tmp_path / METHOD_FILE_NAME).read_text(encoding="utf-8")
    loaded = load(tmp_path)

    stored = json.loads(text)
    assert list(stored) == ["method", "model", "settings_sha256", "tags", "thresholds"]
    assert stored == {
        "method": "jev-statement-code",
        "model": "jev-1.13.0",
        "settings_sha256": sha256_json(SETTINGS),
        "tags": list(FOCUS_TAGS),
        "thresholds": [0.45, 0.0, 0.45, 0.5, 0.45, "inf", 0.45, "inf"],
    }
    assert text.endswith("}\n") and str(tmp_path) not in text
    assert [path.name for path in tmp_path.iterdir()] == [METHOD_FILE_NAME]
    assert loaded.name == "jev-statement-code"
    assert loaded.thresholds.thresholds_.tolist() == thresholds


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("method", "label_prior", r"jev\.json holds the method 'label_prior', which is not a Jev"),
        ("model", "jev-latest", r"jev\.json holds \{'model': 'jev-latest'\}"),
        ("settings_sha256", "0" * 64, r"jev\.json holds \{'settings_sha256': '0{64}'\}"),
        ("tags", ["math"], r"jev\.json holds \{'tags': \['math'\]\}"),
    ],
)
def test_load_refuses_another_model_settings_hash_name_or_tags(tmp_path, key, value, match):
    tagger("jev-statement").save(tmp_path)
    path = tmp_path / METHOD_FILE_NAME
    stored = json.loads(path.read_text(encoding="utf-8")) | {key: value}
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load(tmp_path)


def test_a_jev_tagger_needs_the_name_of_a_jev_method():
    with pytest.raises(ValueError, match=r"'label_prior' is not a Jev method"):
        tagger("label_prior")


@pytest.mark.parametrize("name", FROZEN_NAMES)
def test_jev_tagger_passes_check_tagging_method(name, raw_problems, jev_api, tmp_path):
    # Its isolation subprocess loads the folder without predicting: a Jev method calls an API.
    check_tagging_method(tagger(name), raw_problems, tmp_path / "model")

    # predict and scores open a session each, on the batch, with every column, on each problem
    # alone, then once loaded: 12 sessions, 24 requests.
    assert jev_api.sessions == [name] * 12
    assert len(jev_api.sent) == 24


def test_the_method_module_imports_and_a_folder_loads_without_the_sdk(tmp_path):
    # check_import_isolation looks for mlflow, optuna, gepa and the LLM toolkit's provider SDKs:
    # here typesafe_sdk cannot be imported at all, and the method module and its folder load.
    save_method(tagger("jev-statement"), tmp_path / "model")
    code = (
        "import sys\n"
        "sys.modules['typesafe_sdk'] = None\n"
        "from pathlib import Path\n"
        "import codeforces_tags.methods.typesafe_jev.method\n"
        "from codeforces_tags.methods import load_method\n"
        "method = load_method(Path(sys.argv[1]))\n"
        "print(method.name, sorted({'gepa', 'mlflow', 'optuna'} & set(sys.modules)))\n"
    )

    loaded = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "model")],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.strip() == "jev-statement []"


def test_the_jev_methods_are_registered_as_score_methods_that_call_the_api():
    expected = MethodSpec(
        calls_api=True,
        groups=("llm-typesafe",),
        kind="score",
        module="codeforces_tags.methods.typesafe_jev.method",
        requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
    )

    assert [METHODS[name] for name in (*FROZEN_NAMES, GEPA_NAME)] == [expected] * 3
    # A calibrated twin is a composite over its base: it reads the base's answers (ADR 0011).
    assert {twin: METHODS[twin] for twin in PLATT_BASES} == {
        twin: dataclasses.replace(expected, depends_on=(base,))
        for twin, base in PLATT_BASES.items()
    }
    # pyproject.toml defines the group: all are installed with it, or none is.
    installed = installed_methods()
    assert len({name in installed for name in NAMES}) == 1


def test_the_searched_method_needs_its_words():
    with pytest.raises(
        ValueError, match=r"jev-statement-code-gepa needs its words: .* words\.json"
    ):
        tagger(GEPA_NAME)


def test_a_frozen_method_takes_only_the_seed_words():
    assert tagger("jev-statement").words is SEED_WORDS
    assert tagger("jev-statement-code", words=SEED_WORDS).words is SEED_WORDS
    with pytest.raises(ValueError, match="jev-statement sends the seed words"):
        tagger("jev-statement", words=searched_words())


def test_the_searched_method_asks_jev_with_its_words(jev_api, raw_problems):
    words = searched_words()

    _, answers = asyncio.run(tagger(GEPA_NAME, words=words).apredict_with_answers(raw_problems))

    asked = questions.requests(raw_problems, GEPA_NAME, words)
    seed = questions.requests(raw_problems, GEPA_NAME, SEED_WORDS)
    assert [answer.request_sha256 for answer in answers] == [r.sha256() for r in asked]
    assert not {answer.request_sha256 for answer in answers} & {r.sha256() for r in seed}
    assert jev_api.sessions == [GEPA_NAME]


def test_the_searched_method_saves_and_loads_its_words(tmp_path):
    words = searched_words()
    tagger(GEPA_NAME, [0.3] * 8, words).save(tmp_path)

    loaded = load(tmp_path)

    stored = json.loads((tmp_path / METHOD_FILE_NAME).read_text(encoding="utf-8"))
    text = (tmp_path / WORDS_FILE_NAME).read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == [METHOD_FILE_NAME, WORDS_FILE_NAME]
    assert stored["method"] == GEPA_NAME
    assert stored["settings_sha256"] == sha256_json(method_settings(GEPA_NAME, words))
    assert stored["settings_sha256"] != sha256_json(SETTINGS)
    assert text == json.dumps(words.settings(), indent=2, ensure_ascii=False) + "\n"
    assert f"p{chr(0xB7)}q^-1" in text  # a middle dot: non-ASCII text kept
    assert (loaded.name, loaded.words) == (GEPA_NAME, words)
    assert loaded.thresholds.thresholds_.tolist() == [0.3] * 8


def test_load_refuses_words_that_are_not_those_the_thresholds_were_chosen_with(tmp_path):
    tagger(GEPA_NAME, words=searched_words()).save(tmp_path)
    path = tmp_path / WORDS_FILE_NAME
    values = json.loads(path.read_text(encoding="utf-8"))
    values["convention"] = "Tags name the setters' ideas."
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match=r"jev\.json holds \{'settings_sha256': '[0-9a-f]{64}'\}"):
        load(tmp_path)


def test_load_refuses_a_searched_method_folder_without_its_words_or_with_words_of_another_shape(
    tmp_path,
):
    tagger(GEPA_NAME, words=searched_words()).save(tmp_path)
    path = tmp_path / WORDS_FILE_NAME
    path.write_text(json.dumps({"convention": "Tags."}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"words\.json: Jev's words hold exactly the keys"):
        load(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match=r"words\.json"):
        load(tmp_path)


def test_the_searched_method_passes_check_tagging_method(raw_problems, jev_api, tmp_path):
    check_tagging_method(
        tagger(GEPA_NAME, words=searched_words()), raw_problems, tmp_path / "model"
    )

    assert sorted(path.name for path in (tmp_path / "model").iterdir()) == [
        METHOD_FILE_NAME,
        "method.json",
        WORDS_FILE_NAME,
    ]
    assert jev_api.sessions == [GEPA_NAME] * 12
    assert len(jev_api.sent) == 24


# The words of each calibrated twin in these tests: its base's.
TWIN_WORDS = {
    "jev-statement-code-platt": SEED_WORDS,
    "jev-statement-code-gepa-platt": searched_words(),
}


@pytest.mark.parametrize(("twin", "base"), list(PLATT_BASES.items()))
def test_a_calibrated_twin_asks_jev_its_bases_request(twin, base, jev_api, raw_problems):
    words = TWIN_WORDS[twin]

    _, answers = asyncio.run(tagger(twin, words=words).apredict_with_answers(raw_problems))

    asked = questions.requests(raw_problems, base, words)
    assert [answer.request_sha256 for answer in answers] == [r.sha256() for r in asked]
    assert [list(request.state) for request in asked] == [["statement", "solution"]] * 3
    assert jev_api.sessions == [twin]


def test_the_twin_of_a_frozen_method_takes_only_the_seed_words():
    assert tagger("jev-statement-code-platt").words is SEED_WORDS
    with pytest.raises(ValueError, match="jev-statement-code sends the seed words"):
        tagger("jev-statement-code-platt", words=searched_words())


def test_the_twin_of_the_searched_method_needs_its_words():
    with pytest.raises(
        ValueError,
        match=r"jev-statement-code-gepa-platt needs its words: .* jev-statement-code-gepa .* "
        r"words\.json",
    ):
        tagger("jev-statement-code-gepa-platt")


@pytest.mark.parametrize("twin", list(PLATT_BASES))
def test_a_calibrated_twin_saves_and_loads_its_thresholds_and_its_bases_words(twin, tmp_path):
    words = TWIN_WORDS[twin]
    tagger(twin, [0.695, 0.825, 0.835, 0.765, 0.655, math.inf, 0.815, 0.595], words).save(tmp_path)

    loaded = load(tmp_path)

    stored = json.loads((tmp_path / METHOD_FILE_NAME).read_text(encoding="utf-8"))
    files = sorted(path.name for path in tmp_path.iterdir())
    searched = PLATT_BASES[twin] == GEPA_NAME
    assert files == ([METHOD_FILE_NAME, WORDS_FILE_NAME] if searched else [METHOD_FILE_NAME])
    assert stored["method"] == twin
    assert stored["settings_sha256"] == sha256_json(method_settings(twin, words))
    assert stored["thresholds"] == [0.695, 0.825, 0.835, 0.765, 0.655, "inf", 0.815, 0.595]
    assert (loaded.name, loaded.words) == (twin, words)


@pytest.mark.parametrize(("twin", "base"), list(PLATT_BASES.items()))
def test_load_refuses_a_bases_folder_renamed_as_its_twin(twin, base, tmp_path):
    # The base's thresholds were chosen by another rule: its settings hash is not the twin's.
    tagger(base, words=TWIN_WORDS[twin]).save(tmp_path)
    path = tmp_path / METHOD_FILE_NAME
    stored = json.loads(path.read_text(encoding="utf-8")) | {"method": twin}
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match=r"jev\.json holds \{'settings_sha256': '[0-9a-f]{64}'\}"):
        load(tmp_path)


@pytest.mark.parametrize("twin", list(PLATT_BASES))
def test_a_calibrated_twin_passes_check_tagging_method(twin, raw_problems, jev_api, tmp_path):
    check_tagging_method(tagger(twin, words=TWIN_WORDS[twin]), raw_problems, tmp_path / "model")

    assert jev_api.sessions == [twin] * 12
    assert len(jev_api.sent) == 24
