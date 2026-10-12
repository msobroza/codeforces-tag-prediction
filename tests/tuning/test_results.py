"""Tests for the frozen results of the tuning study."""

import json
import math
import re
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from codeforces_tags.model.params import ModelParams
from codeforces_tags.tuning.results import (
    RESULT_FILES,
    TuningResults,
    load_tuning_results,
    save_tuning_results,
)

TAGS = ["math", "graphs", "games"]


@pytest.fixture(name="results")
def results_fixture():
    """Return results with a conditional parameter, an infinite threshold and tricky floats.

    pandas' default CSV parser reads 1/7, 3/7 and 0.1 + 0.2 back rounded.
    """
    params = ModelParams(
        classifier="linear_svm",
        c=0.123456789012345,
        class_weight="balanced",
        include_notes=True,
        statement_ngram_max=2,
        statement_min_df=3,
        statement_max_df=0.87654321,
        statement_sublinear_tf=True,
        code_weight=1 / 3,
        code_ngram_max=1,
        code_min_df=5,
        code_sublinear_tf=False,
        cue_weight=0.0,
        blocks=("statement", "code"),
    )
    tags = pd.Index(TAGS, name="label")
    trials = pd.DataFrame(
        {
            "number": [0, 1, 2],
            "state": ["COMPLETE", "COMPLETE", "FAIL"],
            "value": [0.61, 3 / 7, math.nan],
            "duration_s": [12.5, 30.25, 0.75],
            "params_classifier": ["logistic_regression", "linear_svm", "linear_svm"],
            "params_c_logistic_regression": [1.5, math.nan, math.nan],
            "params_c_linear_svm": [math.nan, 0.123456789012345, 0.01],
            "params_include_notes": [False, True, True],
            "params_statement_ngram_max": [1, 2, 3],
            "micro_f1": [0.7, 1 / 7, math.nan],
            **{f"f1/{tag}": [0.5, 0.6, math.nan] for tag in TAGS},
        }
    )
    importances = pd.Series(
        [0.5, 0.1 + 0.2, 0.2],
        index=pd.Index(["classifier", "code_weight", "statement_min_df"], name="parameter"),
        name="importance",
    )
    out_of_fold = pd.DataFrame(
        {"f1": [3 / 7, 1 / 7, 0.5, 0.6, 0.7]},
        index=pd.Index(["macro_f1", "micro_f1", *TAGS], name="metric"),
    )
    thresholds = pd.Series([-0.123456789012345, 1 / 3, math.inf], index=tags, name="threshold")
    return TuningResults(params, thresholds, trials, importances, out_of_fold)


def test_load_tuning_results_reads_back_what_save_tuning_results_wrote(tmp_path, results):
    directory = tmp_path / "assets" / "tfidf_baseline"

    save_tuning_results(results, directory)
    loaded = load_tuning_results(directory)

    assert loaded.params == results.params
    pd.testing.assert_series_equal(loaded.thresholds, results.thresholds, check_exact=True)
    pd.testing.assert_frame_equal(loaded.trials, results.trials, check_exact=True)
    pd.testing.assert_series_equal(loaded.importances, results.importances, check_exact=True)
    pd.testing.assert_frame_equal(loaded.out_of_fold, results.out_of_fold, check_exact=True)


def test_save_tuning_results_writes_the_params_and_thresholds_as_json(tmp_path, results):
    save_tuning_results(results, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(RESULT_FILES)
    params = json.loads((tmp_path / "params.json").read_text(encoding="utf-8"))
    assert params == results.params.to_dict()
    thresholds = json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))
    assert list(thresholds) == TAGS


@pytest.mark.parametrize("name", RESULT_FILES)
def test_load_tuning_results_names_the_missing_file_and_the_command_that_writes_it(
    tmp_path, results, name
):
    save_tuning_results(results, tmp_path)
    (tmp_path / name).unlink()

    message = rf"{re.escape(name)}.*python -m codeforces_tags\.tuning export"
    with pytest.raises(FileNotFoundError, match=message):
        load_tuning_results(tmp_path)


def test_the_results_the_model_and_the_evaluation_import_no_heavy_library():
    # A fresh interpreter, so that the imports of other tests do not count. Ruff's TID253 sees
    # only each file's own module-level imports, not what the modules it imports pull in.
    script = textwrap.dedent(
        """
        import importlib
        import pkgutil
        import sys

        import codeforces_tags.evaluation
        import codeforces_tags.model
        import codeforces_tags.tuning.results

        for package in (codeforces_tags.model, codeforces_tags.evaluation):
            for module in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
                importlib.import_module(module.name)
        heavy = ("optuna", "mlflow", "matplotlib", "seaborn")
        print(sorted(name for name in heavy if name in sys.modules))
        """
    )

    imported = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout

    assert imported.strip() == "[]"
