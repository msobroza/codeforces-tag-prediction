"""Fixtures for the tracking tests: MLflow isolated in each test, and fake tagging methods.

Every test here runs with MLflow's environment cleared and restored, and a fresh store on request
(``mlflow_environment`` and ``store``, in ``tests/conftest.py``). The fakes are the conforming
methods of ``tests/methods/fakes/conformance_fakes.py``, registered in ``METHODS`` while a test
runs, as the entry file of a logged model reads it.
"""

import importlib
from pathlib import Path

import pytest

from codeforces_tags import methods
from codeforces_tags.methods import MethodSpec, save_method

FAKES = Path(__file__).parents[1] / "methods" / "fakes"
# The two fakes the tests log: one reads two raw columns and predicts the focus tags and
# "implementation"; the other reads every raw column and predicts all 37 tags.
REGISTRY = {
    "code_rule_fake": MethodSpec(
        module="conformance_fakes",
        kind="other",
        requirements=("pandas",),
        input_columns=("prob_desc_description", "source_code"),
    ),
    "keyword_fake": MethodSpec(
        module="conformance_fakes", kind="zero_shot", requirements=("numpy", "pandas")
    ),
}


@pytest.fixture(autouse=True)
def isolate_mlflow(mlflow_environment):  # pylint: disable=unused-argument
    """Isolate MLflow's environment and default store in every tracking test."""


@pytest.fixture(name="registry")
def registry_fixture(monkeypatch):
    """Register the fakes in ``METHODS``, make them importable, and return the registry."""
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setattr(methods, "METHODS", REGISTRY)
    return REGISTRY


@pytest.fixture(name="fakes")
def fakes_fixture(registry):  # pylint: disable=unused-argument
    """Return the module ``conformance_fakes``, its fakes registered."""
    return importlib.import_module("conformance_fakes")


@pytest.fixture(name="folders")
def folders_fixture(fakes, tmp_path):
    """Save each fake to a method folder, and return the folders by method name."""
    saved = {"code_rule_fake": fakes.CodeRuleFake(), "keyword_fake": fakes.KeywordFake()}
    for name, method in saved.items():
        save_method(method, tmp_path / "methods" / name)
    return {name: tmp_path / "methods" / name for name in saved}
