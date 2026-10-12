"""Tests for the models-from-code entry file every tagging method is logged through.

MLflow runs a copy of the entry file, kept in each logged model, which coverage does not see:
these tests also import the file and call its wrapper directly.
"""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from codeforces_tags import methods
from codeforces_tags.tracking import pyfunc_model
from codeforces_tags.tracking.methods import PYFUNC_ENTRY

# The names that every logged model's copy of the entry file runs: renaming or moving one makes
# the logged models unloadable (docs/adr/0002, safeguard 6).
FROZEN_NAMES = (
    "codeforces_tags.methods.METHODS",
    "codeforces_tags.methods.load_method",
    "codeforces_tags.tracking.pyfunc_model.TaggingMethodModel",
)


def loaded_wrapper(folder):
    """Return the wrapper with the method folder loaded, as MLflow loads a logged model."""
    model = pyfunc_model.TaggingMethodModel()
    model.load_context(SimpleNamespace(artifacts={"method": str(folder)}))
    return model


def test_the_wrapper_predicts_raw_problems_as_its_method_does(fakes, folders, raw_problems):
    model = loaded_wrapper(folders["keyword_fake"])

    predicted = model.predict(None, raw_problems)

    pd.testing.assert_frame_equal(predicted, fakes.KeywordFake().predict(raw_problems))


def test_the_wrapper_loads_through_the_registry_of_the_loading_code(folders, monkeypatch):
    monkeypatch.setattr(methods, "METHODS", {})

    with pytest.raises(ValueError, match="not registered"):
        loaded_wrapper(folders["keyword_fake"])


def test_the_entry_file_is_the_wrapper_module():
    last_line = PYFUNC_ENTRY.read_text(encoding="utf-8").rstrip().splitlines()[-1]

    assert Path(pyfunc_model.__file__) == PYFUNC_ENTRY
    assert last_line == "mlflow.models.set_model(TaggingMethodModel())"


def test_the_entry_file_reaches_the_framework_only_through_frozen_names():
    tree = ast.parse(PYFUNC_ENTRY.read_text(encoding="utf-8"))
    imported = [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module.startswith("codeforces_tags")
        for alias in node.names
    ]
    used = {
        f"codeforces_tags.methods.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "methods"
    }

    assert imported == [("codeforces_tags", "methods")]
    assert used | {f"{pyfunc_model.__name__}.TaggingMethodModel"} == set(FROZEN_NAMES)
    for name in FROZEN_NAMES:
        module, attribute = name.rsplit(".", 1)
        assert hasattr(importlib.import_module(module), attribute), name
