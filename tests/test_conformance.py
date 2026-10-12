"""Tests for the conformance check, the executable contract of every tagging method."""

import importlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio

from codeforces_tags import conformance
from codeforces_tags.config import (
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    PRETRAINED_DIR_ENV_VAR,
    SKOPS_COMPRESSION,
)
from codeforces_tags.conformance import (
    DATA_SUFFIXES,
    PICKLE_SUFFIXES,
    TRACKING_MODULES,
    ConformanceError,
    check_frozen_method,
    check_import_isolation,
    check_method_folder,
    check_tagging_method,
    starts_like_pickle,
)
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    freeze_predictions,
    stack_parts,
)
from codeforces_tags.llm.providers import sdk_modules
from codeforces_tags.methods import (
    METHOD_FILE,
    METHOD_KINDS,
    MODEL_FOLDER,
    MethodSpec,
    load_method,
    save_method,
)
from codeforces_tags.model.thresholds import CrossFittedTagThresholds
from codeforces_tags.splitting import select_part

FAKES = Path(__file__).parent / "methods" / "fakes"
# What the isolation subprocess prepends to sys.path to import conformance_fakes.
SYS_PATH = (str(FAKES),)
# sample_id -> description, solution, tags.
PROBLEMS = {
    1: (
        "Count the leaves of a tree with $$$n$$$ vertices.",
        "n = int(input())\nprint(n - 1)\n",
        ["trees", "graphs"],
    ),
    2: (
        "Print the gcd of $$$a$$$ and $$$b$$$: number theory.",
        "import math\nprint(math.gcd(4, 6))\n",
        ["math", "number theory"],
    ),
    3: ("Two players take stones in turn; print who wins.", "print('First')\n", ["games"]),
    4: (
        "Reverse the strings of the input.",
        "s = input()\nprint(s[::-1])\n",
        ["strings", "implementation"],
    ),
    5: ("Find a shortest path in these graphs.", "print(0)\n", ["graphs", "shortest paths"]),
    6: ("Print the area of a polygon, a geometry task.", "print(0.5)\n", ["geometry", "math"]),
    7: ("Print the probabilities of each outcome.", "print(1 / 6)\n", ["probabilities", "math"]),
    8: ("Sort the array, then print it.", "a = sorted(input().split())\nprint(*a)\n", ["sortings"]),
    9: ("Count the palindromes among the strings.", "print(1)\n", ["strings"]),
}
CODE_RULE_COLUMNS = ("prob_desc_description", "source_code")
# {"a": 1} pickled at protocols 2 and 4, as pickle.dumps writes it (pickle itself is banned here).
PICKLE_BYTES = (
    b"\x80\x02}q\x00X\x01\x00\x00\x00aq\x01K\x01s.",
    b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x01a\x94K\x01s.",
)
# A safetensors file whose 640-byte JSON header gives it the first two bytes of a pickle.
SAFETENSORS_640 = (640).to_bytes(8, "little") + b'{"__metadata__":{}}'.ljust(640)


def make_spec(kind, **fields):
    """Return the spec of a fake served by ``conformance_fakes``."""
    return MethodSpec(module="conformance_fakes", kind=kind, requirements=(), **fields)


REGISTRY = {
    "api_fake": make_spec("zero_shot", calls_api=True),
    "code_rule_fake": make_spec("other", input_columns=CODE_RULE_COLUMNS),
    "edge_fake": make_spec("score", score_atol=1e-5),
    "importing_fake": make_spec("other"),
    "keyword_fake": make_spec("zero_shot"),
    "length_cut_fake": make_spec("fitted"),
    "pickle_file_fake": make_spec("other"),
    "pretrained_fake": make_spec("other"),
    "score_fake": make_spec("score"),
    "zeros_fake": make_spec("other"),
}


@pytest.fixture(name="fakes")
def fakes_fixture(monkeypatch):
    """Return the module ``conformance_fakes``, importable by that name while the test runs."""
    monkeypatch.syspath_prepend(str(FAKES))
    return importlib.import_module("conformance_fakes")


@pytest.fixture(name="dev_problems")
def dev_problems_fixture(write_samples, make_raw_record):
    """Return nine dev problems as ``select_part(read_samples(), split, "dev")`` gives them.

    Their fold, group and tags columns come along; problem 1 has no notes and problem 2 no
    input specification (JSON null, read as NaN), as some problems of the dataset.
    """
    records = {
        sample_id: make_raw_record(prob_desc_description=description, source_code=code, tags=tags)
        for sample_id, (description, code, tags) in PROBLEMS.items()
    }
    records[1]["prob_desc_notes"] = None
    records[2]["prob_desc_input_spec"] = None
    raw = read_samples(write_samples(records))
    split = pd.DataFrame({"part": "dev", "fold": [0, 1, 2] * 3, "group": range(9)}, index=raw.index)
    return select_part(raw, split, "dev")


@pytest.fixture(name="conforming")
def conforming_fixture(fakes, dev_problems):
    """Return a conforming fake of each method kind, fitted on the raw problems."""
    labels = focus_labels(dev_problems)
    model = CrossFittedTagThresholds(fakes.LengthScorer())
    model.fit(dev_problems, labels, groups=dev_problems["fold"])
    return {
        "score": fakes.ScoreFake(model),
        "fitted": fakes.LengthCutFake().fit(dev_problems, labels),
        "zero_shot": fakes.KeywordFake(),
        "other": fakes.CodeRuleFake(),
    }


def check(method, dev_problems, tmp_path, registry=None):
    """Run ``check_tagging_method`` with the fakes' registry and folder."""
    check_tagging_method(
        method,
        dev_problems,
        tmp_path / "model",
        registry=REGISTRY if registry is None else registry,
        sys_path=SYS_PATH,
    )


@pytest.mark.parametrize("kind", METHOD_KINDS)
def test_check_tagging_method_passes_on_a_conforming_fake_of_each_kind(
    kind, conforming, dev_problems, tmp_path
):
    method = conforming[kind]

    check(method, dev_problems, tmp_path)

    assert REGISTRY[method.name].kind == kind
    assert (tmp_path / "model" / METHOD_FILE).is_file()


def test_check_tagging_method_accepts_nan_notes(fakes, dev_problems, tmp_path):
    # A raw frame as select_part(read_samples(), split, part) returns it: fold, group and tags
    # columns, a problem without notes (the first, which FirstRowFake requires the isolation
    # subprocess to predict) and one without input specification. KeywordFake reads the notes.
    assert {"fold", "group", "tags"} <= set(dev_problems.columns)
    assert dev_problems["prob_desc_notes"].isna().tolist()[:2] == [True, False]
    assert pd.isna(dev_problems.loc[2, "prob_desc_input_spec"])
    before = dev_problems.copy()

    check(fakes.FirstRowFake(), dev_problems, tmp_path)

    pd.testing.assert_frame_equal(dev_problems, before)


def test_the_score_fake_wraps_a_scorer_under_the_score_estimator_contract(conforming, dev_problems):
    scorer = conforming["score"].model.estimator_

    np.testing.assert_array_equal(
        scorer.predict(dev_problems), scorer.decision_function(dev_problems) >= 0
    )
    np.testing.assert_array_equal(scorer.classes_, np.arange(len(FOCUS_TAGS)))


def test_the_score_fake_saves_a_compressed_skops_file_of_two_untrusted_types(
    fakes, conforming, tmp_path
):
    save_method(conforming["score"], tmp_path)

    path = tmp_path / fakes.MODEL_FILE
    assert sio.get_untrusted_types(file=path) == fakes.TRUSTED_TYPES
    with zipfile.ZipFile(path) as archive:
        assert {info.compress_type for info in archive.infolist()} == {SKOPS_COMPRESSION}


def test_check_tagging_method_refuses_what_is_not_a_tagging_method(dev_problems, tmp_path):
    with pytest.raises(ConformanceError, match="TaggingMethod"):
        check(object(), dev_problems, tmp_path)


def test_check_tagging_method_refuses_an_unregistered_method(fakes, dev_problems, tmp_path):
    with pytest.raises(ConformanceError, match="'zeros_fake' is not registered"):
        check(fakes.ZerosFake(), dev_problems, tmp_path, registry={})


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        (list(FOCUS_TAGS), "a tuple of str"),
        (FOCUS_TAGS + ("math",), r"repeat \['math'\]"),
        (FOCUS_TAGS + ("recursion",), r"hold \['recursion'\], which are not tags of ALL_TAGS"),
        (FOCUS_TAGS[1:], r"lack the focus tags \['math'\]"),
        (tuple(sorted(FOCUS_TAGS)), "in FOCUS_TAGS order"),
        (FOCUS_TAGS + ("implementation",), "in ALL_TAGS order"),
    ],
)
def test_check_tagging_method_refuses_tags_that_break_the_tag_rules(
    fakes, dev_problems, tmp_path, tags, match
):
    with pytest.raises(ConformanceError, match=match):
        check(fakes.ZerosFake(tags), dev_problems, tmp_path)


def test_check_tagging_method_refuses_a_score_method_without_scores(fakes, dev_problems, tmp_path):
    registry = REGISTRY | {"zeros_fake": make_spec("score")}

    with pytest.raises(ConformanceError, match="kind 'score' implements ScoreMethod"):
        check(fakes.ZerosFake(), dev_problems, tmp_path, registry)


def test_check_tagging_method_refuses_a_score_method_that_cannot_predict_from_scores(
    fakes, dev_problems, tmp_path
):
    # The command line prints a score method's scores and its tags from those same scores.
    method = fakes.ScoresOnlyFake()

    assert method.scores(dev_problems).eq(0.0).all().all()
    with pytest.raises(ConformanceError, match=r"predict_from_scores\(scores\)"):
        check(method, dev_problems, tmp_path)


def test_check_tagging_method_names_the_error_of_predict_from_scores(fakes, dev_problems, tmp_path):
    with pytest.raises(
        ConformanceError, match=r"edge_fake\.predict_from_scores raised KeyError\('expected other"
    ):
        check(fakes.RaisingFromScoresFake(offset=0.0, threshold=0.5), dev_problems, tmp_path)


def test_check_tagging_method_refuses_tags_from_scores_that_are_not_its_predictions(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(
        ConformanceError,
        match=r"edge_fake\.predict_from_scores\(scores\) must return predict's 0/1 tags",
    ):
        check(fakes.DisagreeingFake(offset=0.0, threshold=0.5), dev_problems, tmp_path)


def test_check_tagging_method_refuses_a_method_that_changes_the_scores_it_predicts_from(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match="changed the scores it was given"):
        check(fakes.ScoreMutatingFake(offset=0.0, threshold=0.5), dev_problems, tmp_path)


def test_check_tagging_method_refuses_a_score_tolerance_without_scores(
    fakes, dev_problems, tmp_path
):
    registry = REGISTRY | {"zeros_fake": make_spec("other", score_atol=1e-3)}

    with pytest.raises(ConformanceError, match="score_atol 0.001, which needs scores"):
        check(fakes.ZerosFake(), dev_problems, tmp_path, registry)


def test_check_tagging_method_needs_raw_problems_with_the_input_columns(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ValueError, match=r"lack the input columns \['prob_desc_notes'\]"):
        check(fakes.ZerosFake(), dev_problems.drop(columns="prob_desc_notes"), tmp_path)


def test_check_tagging_method_needs_two_raw_problems(fakes, dev_problems, tmp_path):
    with pytest.raises(ValueError, match="at least two raw problems"):
        check(fakes.ZerosFake(), dev_problems.iloc[:1], tmp_path)


def test_check_tagging_method_needs_a_new_folder(fakes, dev_problems, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "left.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        check(fakes.ZerosFake(), dev_problems, tmp_path)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda predicted: predicted.to_numpy(), "a DataFrame, not ndarray"),
        (lambda predicted: predicted.iloc[::-1], "the index of its input"),
        (lambda predicted: predicted.rename_axis("id"), "the index of its input"),
        (lambda predicted: predicted.iloc[:, ::-1], "the columns of its tags"),
        (lambda predicted: predicted.astype(float), "int 0/1"),
        (lambda predicted: predicted.astype(bool), "int 0/1"),
        (lambda predicted: predicted + 2, "int 0/1"),
    ],
)
def test_check_tagging_method_refuses_predictions_that_break_the_output_rules(
    fakes, dev_problems, tmp_path, change, match
):
    with pytest.raises(ConformanceError, match=match):
        check(fakes.MalformedFake(change), dev_problems, tmp_path)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda scores: scores.to_numpy(),
            r"edge_fake\.scores must return a DataFrame, not ndarray",
        ),
        (lambda scores: scores.iloc[::-1], r"edge_fake\.scores must return the index of its input"),
        (
            lambda scores: scores.iloc[:, ::-1],
            r"edge_fake\.scores must return the columns of its tags",
        ),
        (lambda scores: scores.astype(int), "float scores without NaN"),
    ],
)
def test_check_tagging_method_refuses_scores_that_break_the_output_rules(
    fakes, dev_problems, tmp_path, change, match
):
    with pytest.raises(ConformanceError, match=match):
        check(fakes.MalformedScoresFake(change), dev_problems, tmp_path)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda thresholds: thresholds.to_numpy(), "must return a Series, not ndarray"),
        (lambda thresholds: thresholds.iloc[::-1], "must be indexed by its tags, in order"),
        (lambda thresholds: thresholds.astype(int), "must return float thresholds without NaN"),
        (lambda thresholds: thresholds.where(thresholds > 1), "must return float thresholds"),
        (lambda thresholds: thresholds - 1.0, "must be the thresholds that predict_from_scores"),
    ],
)
def test_check_tagging_method_refuses_thresholds_that_break_the_threshold_rules(
    fakes, dev_problems, tmp_path, change, match
):
    # The fake predicts at 0.5; it gives that threshold as an array, reversed, as an int, as NaN,
    # or as -0.5, which would predict every tag of the batch's 0.0 scores.
    with pytest.raises(ConformanceError, match=rf"edge_fake\.tag_thresholds\(\) {match}"):
        check(fakes.MalformedThresholdsFake(change), dev_problems, tmp_path)


def test_check_tagging_method_accepts_an_infinite_threshold(fakes, dev_problems, tmp_path):
    # inf is the threshold of a tag that is never predicted, e.g. one that no dev problem has.
    method = fakes.EdgeFake(offset=0.0, threshold=np.inf)

    check(method, dev_problems, tmp_path)

    assert method.tag_thresholds().tolist() == [np.inf] * len(FOCUS_TAGS)


def test_check_tagging_method_refuses_a_method_that_changes_its_input(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match="changed its input"):
        check(fakes.MutatingFake(), dev_problems, tmp_path)
    assert (dev_problems["source_code"] != "").all()


def test_check_tagging_method_refuses_a_method_that_reads_an_undeclared_column(
    fakes, dev_problems, tmp_path
):
    registry = REGISTRY | {
        "code_rule_fake": make_spec("other", input_columns=CODE_RULE_COLUMNS[:1])
    }

    with pytest.raises(ConformanceError, match="raised KeyError.*input_columns"):
        check(fakes.CodeRuleFake(), dev_problems, tmp_path, registry)


def test_check_tagging_method_refuses_a_method_that_reads_the_true_tags(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match="given extra columns"):
        check(fakes.LeakyFake(), dev_problems, tmp_path)


def test_check_tagging_method_refuses_a_row_that_depends_on_the_batch(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match=r"one problem at a time \(sample_id 1\)"):
        check(fakes.BatchFake(), dev_problems, tmp_path)


def test_check_tagging_method_refuses_a_method_that_does_not_survive_its_folder(
    conforming, fakes, dev_problems, tmp_path
):
    labels = focus_labels(dev_problems)
    method = fakes.ForgetfulFake().fit(dev_problems, labels)

    assert method.predict(dev_problems).equals(conforming["fitted"].predict(dev_problems))
    with pytest.raises(ConformanceError, match="after save_method and load_method"):
        check(method, dev_problems, tmp_path)


def test_check_tagging_method_lets_a_tag_flip_within_score_atol_with_its_scores(
    fakes, dev_problems, tmp_path
):
    # One problem alone scores 1e-6 above the batch, across the threshold: within score_atol
    # (1e-5), each tag flips to 1 where its score is the higher one.
    method = fakes.EdgeFake(offset=1e-6, threshold=5e-7)

    assert method.predict(dev_problems.iloc[[0]]).to_numpy().tolist() == [[1] * 8]
    assert method.predict(dev_problems).to_numpy().sum() == 0
    check(method, dev_problems, tmp_path)


def test_check_tagging_method_refuses_one_row_scores_beyond_score_atol(
    fakes, dev_problems, tmp_path
):
    registry = REGISTRY | {"edge_fake": make_spec("score")}

    with pytest.raises(ConformanceError, match="scores differ by up to 1e-06, more than 1e-09"):
        check(fakes.EdgeFake(offset=1e-6, threshold=5e-7), dev_problems, tmp_path, registry)


def test_check_tagging_method_refuses_a_tag_that_flips_against_its_scores(
    fakes, dev_problems, tmp_path
):
    # Alone, a problem scores 1e-6 above the batch, but its threshold rose by 2e-6: each call's
    # tags follow that call's thresholds, yet the tags flip to 0 where the scores rose.
    method = fakes.EdgeFake(offset=1e-6, threshold=0.0, moved=2e-6)

    with pytest.raises(ConformanceError, match="flips against its scores"):
        check(method, dev_problems, tmp_path)


@pytest.mark.parametrize(("threshold", "moved"), [(0.0, 1e-7), (1e-7, -1e-7)])
def test_check_tagging_method_refuses_a_tag_that_flips_between_equal_scores(
    fakes, dev_problems, tmp_path, threshold, moved
):
    # The scores are 0.0 in a batch and alone: a flip from 1 to 0 is as wrong as one from 0 to 1.
    with pytest.raises(ConformanceError, match="flips against its scores"):
        check(fakes.EdgeFake(offset=0.0, threshold=threshold, moved=moved), dev_problems, tmp_path)


def test_check_tagging_method_compares_the_scores_beside_an_infinite_one(
    fakes, dev_problems, tmp_path
):
    # The first tag scores -inf in a batch and alone: equal infinities agree, and they hide no
    # difference of the other tags' scores.
    check(fakes.LogProbabilityFake(offset=1e-6, threshold=5e-7), dev_problems, tmp_path)

    with pytest.raises(ConformanceError, match="scores differ by up to 1, more than 1e-05"):
        check(fakes.LogProbabilityFake(offset=1.0, threshold=2.0), dev_problems, tmp_path / "2")


def test_check_tagging_method_refuses_a_score_method_that_loads_without_scores(
    fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match="loses its scores .* returns a ZerosFake"):
        check(fakes.ScorelessLoadFake(offset=0.0, threshold=0.5), dev_problems, tmp_path)


def test_check_tagging_method_refuses_scores_with_nan(fakes, dev_problems, tmp_path):
    with pytest.raises(ConformanceError, match="float scores without NaN"):
        check(fakes.EdgeFake(offset=float("nan"), threshold=0.5), dev_problems, tmp_path)


def test_check_tagging_method_checks_the_method_folder(fakes, dev_problems, tmp_path):
    with pytest.raises(ConformanceError, match=r"state\.pkl .* is a pickle format"):
        check(fakes.PickleFileFake(), dev_problems, tmp_path)


@pytest.mark.parametrize("library", ["gepa", "mlflow", "optuna"])
def test_check_tagging_method_refuses_a_method_whose_prediction_path_imports_a_tracking_library(
    library, fakes, dev_problems, tmp_path
):
    with pytest.raises(ConformanceError, match=rf"imported \['{library}'\]"):
        check(fakes.ImportingFake([library]), dev_problems, tmp_path)


def test_the_isolation_subprocess_inherits_the_environment(
    fakes, dev_problems, tmp_path, monkeypatch
):
    # A method that reads pretrained weights from $CODEFORCES_PRETRAINED_DIR, set by its test.
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "weights.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(weights))

    check(fakes.PretrainedFake(), dev_problems, tmp_path)


def test_check_import_isolation_refuses_a_method_that_imports_tracking_or_provider_modules(
    fakes, tmp_path, monkeypatch
):
    modules = (*TRACKING_MODULES, *sdk_modules())
    folder = tmp_path / "model"
    save_method(fakes.ImportingFake(modules), folder)
    # Stand-ins of the libraries: in this process, already imported ones; in the subprocess,
    # empty modules first on its sys.path.
    shadows = tmp_path / "shadows"
    shadows.mkdir()
    for module in modules:
        monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
        (shadows / f"{module}.py").write_text("", encoding="utf-8")

    assert load_method(folder, registry=REGISTRY).modules == list(modules)
    with pytest.raises(
        ConformanceError, match=r"imported \['gepa', 'mlflow', 'openai', 'optuna'\]"
    ):
        check_import_isolation(folder, registry=REGISTRY, sys_path=(*SYS_PATH, str(shadows)))


def test_check_import_isolation_never_predicts_with_a_method_that_calls_an_api(
    fakes, dev_problems, tmp_path
):
    save_method(fakes.ApiFake(), tmp_path)

    check_import_isolation(tmp_path, registry=REGISTRY, sys_path=SYS_PATH)
    with pytest.raises(RuntimeError, match="paid request"):
        fakes.ApiFake().predict(dev_problems)


def test_check_import_isolation_predicts_a_synthetic_problem_by_default(fakes, tmp_path):
    save_method(fakes.KeywordFake(), tmp_path)

    check_import_isolation(tmp_path, registry=REGISTRY, sys_path=SYS_PATH)


def test_check_import_isolation_needs_the_sys_path_of_the_method_module(fakes, tmp_path):
    save_method(fakes.ZerosFake(), tmp_path)

    with pytest.raises(ConformanceError, match="fresh interpreter failed(.|\n)*conformance_fakes"):
        check_import_isolation(tmp_path, registry=REGISTRY)


@pytest.mark.parametrize(
    ("text", "match"),
    [("oops", "is not valid JSON"), ("[]", "lacks the keys"), ("{}", "lacks the keys")],
)
def test_check_import_isolation_refuses_a_malformed_method_json(tmp_path, text, match):
    (tmp_path / METHOD_FILE).write_text(text, encoding="utf-8")

    with pytest.raises(ConformanceError, match=match):
        check_import_isolation(tmp_path, registry=REGISTRY)


def test_check_import_isolation_needs_a_registered_method_folder(fakes, tmp_path):
    with pytest.raises(ConformanceError, match="holds no method.json"):
        check_import_isolation(tmp_path, registry=REGISTRY)
    save_method(fakes.ZerosFake(), tmp_path)
    with pytest.raises(ConformanceError, match="'zeros_fake' is not registered"):
        check_import_isolation(tmp_path, registry={})


def write_files(folder, files):
    """Write ``{relative path: bytes}`` under ``folder``."""
    for name, data in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def numpy_bytes(suffix, array, version=None):
    """Return an array as the bytes of an .npy file (of a header ``version``), or of an .npz."""
    buffer = io.BytesIO()
    if suffix == ".npz":
        np.savez(buffer, weights=array)
    else:
        np.lib.format.write_array(buffer, array, version=version)
    return buffer.getvalue()


def test_check_method_folder_accepts_data_files_in_sub_folders(tmp_path):
    files = {f"part_{index}/data{suffix}": b"{}" for index, suffix in enumerate(DATA_SUFFIXES)}
    for suffix in (".npy", ".npz"):
        files[f"part_{DATA_SUFFIXES.index(suffix)}/data{suffix}"] = numpy_bytes(suffix, np.ones(3))
    files["arrays/words.npy"] = numpy_bytes(".npy", np.array(["leaf", "tree"]), version=(2, 0))
    write_files(tmp_path, files | {"adapter/model.safetensors": SAFETENSORS_640})

    check_method_folder(tmp_path)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_check_method_folder_refuses_an_object_array(tmp_path, suffix):
    # numpy stores an object array as a pickle, which np.load(allow_pickle=True) would run.
    write_files(tmp_path, {f"weights{suffix}": numpy_bytes(suffix, np.array([{}], dtype=object))})

    with pytest.raises(ConformanceError, match="holds an object array, which numpy stores as a"):
        check_method_folder(tmp_path)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_check_method_folder_refuses_a_numpy_file_that_numpy_did_not_write(tmp_path, suffix):
    # {} pickled at protocol 0, which has no header: np.load(allow_pickle=True) would unpickle it.
    write_files(tmp_path, {f"weights{suffix}": b"(dp0\n."})

    with pytest.raises(
        ConformanceError, match=rf"weights\{suffix} .* is not a valid \{suffix} file"
    ):
        check_method_folder(tmp_path)


@pytest.mark.parametrize("suffix", [*PICKLE_SUFFIXES, ".PKL"])
def test_check_method_folder_refuses_pickle_formats(tmp_path, suffix):
    write_files(tmp_path, {f"sub/state{suffix}": b"{}"})

    with pytest.raises(ConformanceError, match=rf"sub/state\{suffix} .* is a pickle format"):
        check_method_folder(tmp_path)


@pytest.mark.parametrize("name", ["weights.bin", "LICENSE", "load.py"])
def test_check_method_folder_refuses_files_that_are_not_data(tmp_path, name):
    write_files(tmp_path, {name: b"{}"})

    with pytest.raises(ConformanceError, match="is not a data file"):
        check_method_folder(tmp_path)


@pytest.mark.parametrize("data", PICKLE_BYTES)
def test_check_method_folder_refuses_a_pickle_under_a_data_name(tmp_path, data):
    write_files(tmp_path, {"weights.npy": data})

    with pytest.raises(ConformanceError, match="starts with pickle header bytes"):
        check_method_folder(tmp_path)


@pytest.mark.parametrize(
    ("data", "pickle_like"),
    [
        (PICKLE_BYTES[0], True),
        (PICKLE_BYTES[1], True),
        (b"\x80\x05" + bytes(7), True),
        (SAFETENSORS_640, False),
        (b"\x93NUMPY\x01\x00", False),
        (b"PK\x03\x04", False),
        (b"\x80", False),
        (b"", False),
    ],
)
def test_starts_like_pickle_tells_pickles_from_data_files(tmp_path, data, pickle_like):
    path = tmp_path / "file"
    path.write_bytes(data)

    assert starts_like_pickle(path) is pickle_like


def test_check_method_folder_caps_the_size_of_each_file(tmp_path, monkeypatch):
    monkeypatch.setattr(conformance, "METHOD_FILE_MAX_BYTES", 4)
    write_files(tmp_path, {"small.json": b"{}", "large.json": b'{"a":1}'})

    with pytest.raises(ConformanceError, match=r"large\.json .* 7 bytes, above 4"):
        check_method_folder(tmp_path)


def test_check_method_folder_caps_the_size_of_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(conformance, "METHOD_FOLDER_MAX_BYTES", 5)
    write_files(tmp_path, {"a.json": b"{}", "sub/b.json": b"[1]"})

    check_method_folder(tmp_path)
    write_files(tmp_path, {"sub/c.json": b"{}"})
    with pytest.raises(ConformanceError, match="7 bytes in all, above 5"):
        check_method_folder(tmp_path)


def test_check_method_folder_refuses_a_symbolic_link(tmp_path):
    write_files(tmp_path, {"outside.json": b"{}"})
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "link.json").symlink_to(tmp_path / "outside.json")

    with pytest.raises(ConformanceError, match="link.json .* is a symbolic link"):
        check_method_folder(tmp_path / "model")


def test_check_method_folder_refuses_a_file_naming_the_folder(tmp_path):
    folder = tmp_path / "model"
    write_files(folder, {"paths.json": json.dumps({"weights": str(folder / "w.npz")}).encode()})

    with pytest.raises(ConformanceError, match="holds the folder's absolute path"):
        check_method_folder(folder)


def test_check_method_folder_needs_a_folder(tmp_path):
    with pytest.raises(ConformanceError, match="is not a folder"):
        check_method_folder(tmp_path / "missing")


# A split of three problems: two dev, one test.
SPLIT = pd.DataFrame(
    {"part": ["dev", "dev", "test"], "fold": [0, 1, -1], "group": [0, 1, 2]},
    index=pd.Index([4, 5, 6], name="sample_id"),
)
FROZEN_REGISTRY = {"zeros_fake": make_spec("score")}
# The same frozen files, as those of a method that calls an API.
API_REGISTRY = {"zeros_fake": make_spec("zero_shot", calls_api=True)}


@pytest.fixture(name="assets")
def assets_fixture(fakes, tmp_path, placeholder_record):
    """Return an assets folder that holds one frozen method of kind score, ``zeros_fake``."""
    parts = {
        part: pd.DataFrame(0, index=SPLIT.index[SPLIT["part"] == part], columns=list(FOCUS_TAGS))
        for part in ("dev", "test")
    }
    predictions = stack_parts(**parts)
    directory = tmp_path / "zeros_fake"
    record = placeholder_record("zeros_fake")
    freeze_predictions(predictions, directory, record, scores=predictions.astype(float))
    save_method(fakes.ZerosFake(), directory / MODEL_FOLDER)
    return tmp_path


def write_answers(folder, part, sample_ids, make_answer):
    """Write an answers file of a part: one answer per sample id, each under a hash of its own."""
    lines = [
        make_answer(request_sha256=f"{line:032x}{sample_id:032x}", sample_id=sample_id).to_json()
        for line, sample_id in enumerate(sample_ids)
    ]
    (folder / f"answers_{part}.jsonl").write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )


@pytest.fixture(name="api_assets")
def api_assets_fixture(assets, make_answer):
    """Return the assets folder of ``assets``, with an answers file per part of the split."""
    write_answers(assets / "zeros_fake", "dev", [4, 5], make_answer)
    write_answers(assets / "zeros_fake", "test", [6], make_answer)
    return assets


def check_frozen(assets, registry=None):
    """Run ``check_frozen_method`` on ``zeros_fake`` with the three-problem split."""
    check_frozen_method(
        "zeros_fake",
        assets_dir=assets,
        split=SPLIT,
        registry=FROZEN_REGISTRY if registry is None else registry,
    )


def edit_csv(path, change):
    """Rewrite a CSV file with ``change(frame)``."""
    change(pd.read_csv(path)).to_csv(path, index=False)


def edit_json(path, change):
    """Rewrite a JSON file with ``change(values)``."""
    values = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(values)), encoding="utf-8")


def test_check_frozen_method_passes_on_complete_consistent_frozen_files(assets):
    check_frozen(assets)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        pytest.param(
            lambda folder: (folder / PREDICTIONS_FILE).unlink(),
            "has no predictions.csv",
            id="no predictions",
        ),
        pytest.param(
            lambda folder: edit_csv(folder / PREDICTIONS_FILE, lambda frame: frame.iloc[:-1]),
            "1 missing, 0 not in the split, 0 repeated",
            id="a missing problem",
        ),
        pytest.param(
            lambda folder: edit_csv(
                folder / PREDICTIONS_FILE, lambda frame: frame.replace({"sample_id": {6: 99}})
            ),
            "1 missing, 1 not in the split, 0 repeated",
            id="a problem not in the split",
        ),
        pytest.param(
            lambda folder: edit_csv(
                folder / PREDICTIONS_FILE, lambda frame: pd.concat([frame, frame.iloc[:1]])
            ),
            "0 missing, 0 not in the split, 1 repeated",
            id="a repeated problem",
        ),
        pytest.param(
            lambda folder: edit_csv(folder / PREDICTIONS_FILE, lambda frame: frame.assign(math=2)),
            "values other than 0 and 1",
            id="a value of 2",
        ),
        pytest.param(
            lambda folder: edit_csv(
                folder / PREDICTIONS_FILE,
                lambda frame: frame[["part", "sample_id", *reversed(FOCUS_TAGS)]],
            ),
            "columns of .*predictions.csv must be the tags of method.json",
            id="reordered columns",
        ),
        pytest.param(
            lambda folder: (folder / SCORES_FILE).unlink(),
            "kind 'score' freezes its scores",
            id="no scores",
        ),
        pytest.param(
            lambda folder: edit_csv(folder / SCORES_FILE, lambda frame: frame.iloc[::-1]),
            "rows and columns of predictions.csv",
            id="reordered scores",
        ),
        pytest.param(
            lambda folder: edit_json(
                folder / RUN_RECORD_FILE,
                lambda record: {key: value for key, value in record.items() if key != "git"},
            ),
            r"missing \['git'\]",
            id="a run record key missing",
        ),
        pytest.param(
            lambda folder: edit_json(
                folder / RUN_RECORD_FILE, lambda record: record | {"method": "x"}
            ),
            "names the method 'x', not 'zeros_fake'",
            id="a run record of another method",
        ),
        pytest.param(
            lambda folder: edit_json(
                folder / RUN_RECORD_FILE, lambda record: record | {"method_entries": {}}
            ),
            r"method_entries\['supersedes'\]",
            id="no supersedes",
        ),
        pytest.param(
            lambda folder: (folder / MODEL_FOLDER / METHOD_FILE).unlink(),
            "holds no method.json",
            id="no method.json",
        ),
        pytest.param(
            lambda folder: (folder / MODEL_FOLDER / METHOD_FILE).write_text("oops"),
            "method.json is not valid JSON",
            id="method.json not JSON",
        ),
        pytest.param(
            lambda folder: edit_json(folder / MODEL_FOLDER / METHOD_FILE, lambda stored: {}),
            r"method.json lacks the keys \['method', 'format_version', 'tags'\]",
            id="method.json without keys",
        ),
        pytest.param(
            lambda folder: edit_json(
                folder / MODEL_FOLDER / METHOD_FILE, lambda stored: stored | {"method": "x"}
            ),
            "names the method 'x', not 'zeros_fake'",
            id="a method folder of another method",
        ),
        pytest.param(
            lambda folder: edit_json(
                folder / MODEL_FOLDER / METHOD_FILE,
                lambda stored: stored | {"tags": sorted(stored["tags"])},
            ),
            "in FOCUS_TAGS order",
            id="tags out of order",
        ),
        pytest.param(
            lambda folder: (folder / MODEL_FOLDER / "state.pkl").write_bytes(b""),
            "is a pickle format",
            id="a pickle in the method folder",
        ),
    ],
)
def test_check_frozen_method_refuses_incomplete_or_inconsistent_frozen_files(assets, change, match):
    change(assets / "zeros_fake")

    with pytest.raises(ConformanceError, match=match):
        check_frozen(assets)


def test_check_frozen_method_refuses_an_unregistered_method(assets):
    with pytest.raises(ConformanceError, match="'zeros_fake' is not registered"):
        check_frozen(assets, registry={})


def test_check_frozen_method_lets_a_method_of_another_kind_have_no_scores(assets):
    (assets / "zeros_fake" / SCORES_FILE).unlink()

    check_frozen(assets, registry={"zeros_fake": make_spec("fitted")})


def test_check_frozen_method_needs_supersedes_from_local_methods_only(api_assets):
    edit_json(
        api_assets / "zeros_fake" / RUN_RECORD_FILE, lambda record: record | {"method_entries": {}}
    )

    check_frozen(api_assets, registry=API_REGISTRY)


def test_check_frozen_method_passes_on_an_api_method_with_one_answer_per_problem(api_assets):
    check_frozen(api_assets, registry=API_REGISTRY)


def test_check_frozen_method_needs_no_answers_file_from_a_composite_api_method(assets):
    # It calls the API through its base, whose answers files are the base's own.
    registry = {
        "api_base_fake": make_spec("zero_shot", calls_api=True),
        "zeros_fake": make_spec("fitted", calls_api=True, depends_on=("api_base_fake",)),
    }

    check_frozen(assets, registry=registry)


def test_check_frozen_method_needs_no_answers_file_from_a_local_method(api_assets):
    for part in ("dev", "test"):
        (api_assets / "zeros_fake" / f"answers_{part}.jsonl").unlink()

    check_frozen(api_assets)


@pytest.mark.parametrize(
    ("part", "sample_ids", "match"),
    [
        pytest.param("dev", [4], "1 missing, 0 not in the dev part, 0 repeated", id="missing"),
        pytest.param(
            "dev", [4, 5, 6], "0 missing, 1 not in the dev part, 0 repeated", id="another part"
        ),
        pytest.param("test", [6, 6], "0 missing, 0 not in the test part, 1 repeated", id="twice"),
    ],
)
def test_check_frozen_method_refuses_answers_files_without_one_line_per_problem(
    api_assets, make_answer, part, sample_ids, match
):
    write_answers(api_assets / "zeros_fake", part, sample_ids, make_answer)

    with pytest.raises(ConformanceError, match=f"answers_{part}.jsonl must hold one line per"):
        check_frozen(api_assets, registry=API_REGISTRY)
    with pytest.raises(ConformanceError, match=match):
        check_frozen(api_assets, registry=API_REGISTRY)


def test_check_frozen_method_refuses_an_api_method_without_an_answers_file(api_assets):
    (api_assets / "zeros_fake" / "answers_test.jsonl").unlink()

    with pytest.raises(ConformanceError, match="has no answers_test.jsonl: run `llm run"):
        check_frozen(api_assets, registry=API_REGISTRY)


def test_check_frozen_method_reads_the_answers_files_with_read_answers(api_assets):
    path = api_assets / "zeros_fake" / "answers_dev.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "not an answer\n", encoding="utf-8")

    with pytest.raises(ConformanceError, match=r"answers_dev.jsonl is malformed: .* line 3 is not"):
        check_frozen(api_assets, registry=API_REGISTRY)


def test_check_frozen_method_reads_the_committed_split_by_default(assets):
    with pytest.raises(ConformanceError, match="each dev and test problem of split.csv once"):
        check_frozen_method("zeros_fake", assets_dir=assets, registry=FROZEN_REGISTRY)


def frozen_failures(names, **kwargs):
    """Map each method whose frozen files fail ``check_frozen_method`` to its error message."""
    failures = {}
    for name in names:
        try:
            check_frozen_method(name, **kwargs)
        except ConformanceError as error:
            failures[name] = str(error)
    return failures


def test_frozen_failures_name_the_methods_that_fail(assets):
    registry = FROZEN_REGISTRY | {"missing_fake": make_spec("fitted")}

    failures = frozen_failures(
        ["zeros_fake", "missing_fake"], assets_dir=assets, split=SPLIT, registry=registry
    )

    assert list(failures) == ["missing_fake"]
    assert "holds no method.json" in failures["missing_fake"]


def test_every_committed_method_passes_check_frozen_method():
    # One test over every committed assets/methods/<name>/ (none before label_prior was frozen):
    # a test parametrized over no folder would be skipped, its body never run.
    committed = sorted(path.name for path in METHODS_ASSETS_DIR.glob("*/"))

    assert not frozen_failures(committed)
