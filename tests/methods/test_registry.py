"""Tests for the tagging-method registry and method folders."""

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from codeforces_tags.config import METHOD_FORMAT_VERSION, METHODS_ASSETS_DIR, RAW_INPUT_COLUMNS
from codeforces_tags.methods import (
    METHOD_FILE,
    METHOD_KINDS,
    METHODS,
    MODEL_FOLDER,
    MethodSpec,
    ScoreMethod,
    TaggingMethod,
    installed_methods,
    load_method,
    method_dir,
    missing_groups,
    model_dir,
    save_method,
    sibling_methods,
)

FAKES = Path(__file__).parent / "fakes"
TAGS = ("math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities")
# The core (what every method and the tracking code import), and the modules held to its rule
# (the device setting, the served records, the HTTP client, the app's data): importing any of
# them imports no heavy or tracking library. This test sees what an import loads, not a lazy
# import inside a function, which carries a pylint disable for review to read.
CORE_MODULES = (
    "codeforces_tags.methods",
    "codeforces_tags.methods.out_of_fold",
    "codeforces_tags.preprocessing.pipeline",
    "codeforces_tags.model.thresholds",
    "codeforces_tags.evaluation.report",
    "codeforces_tags.evaluation.predictions",
    "codeforces_tags.evaluation.timing",
    "codeforces_tags.evaluation.contamination",
    "codeforces_tags.provenance",
    "codeforces_tags.cache",
    "codeforces_tags.pretrained",
    "codeforces_tags.devices",
    "codeforces_tags.conformance",
    "codeforces_tags.serving.records",
    "codeforces_tags.serving.client",
    "codeforces_tags.serving.problems",
)
HEAVY_MODULES = ("mlflow", "optuna", "gepa", "openai", "torch", "typesafe_sdk")


@pytest.fixture(name="fakes")
def fakes_fixture(monkeypatch):
    """Return the module ``fake_methods``, importable by that name while the test runs."""
    monkeypatch.syspath_prepend(str(FAKES))
    return importlib.import_module("fake_methods")


@pytest.fixture(name="registry")
def registry_fixture():
    """Return a registry of one fitted method, ``constant``, served by ``fake_methods``."""
    return {"constant": MethodSpec(module="fake_methods", kind="fitted", requirements=())}


@pytest.fixture(name="two_problems")
def two_problems_fixture():
    """Return two raw problems, indexed by sample id."""
    return pd.DataFrame(
        {"prob_desc_description": ["Find n.", "Sort it."]},
        index=pd.Index([7, 9], name="sample_id"),
    )


def make_spec(**overrides):
    """Return a ``MethodSpec`` of a fitted method, with fields overridable by keyword."""
    return MethodSpec(
        **({"module": "fake_methods", "kind": "fitted", "requirements": ()} | overrides)
    )


def composite_errors(registry):
    """Name the composites whose bases are unregistered, or call the API or need other groups."""
    return [
        name
        for name, spec in registry.items()
        if not set(spec.depends_on) <= set(registry)
        or any(registry[base].calls_api and not spec.calls_api for base in spec.depends_on)
        or not all(set(registry[base].groups) <= set(spec.groups) for base in spec.depends_on)
    ]


def test_methods_are_registered_one_line_each_sorted_by_name():
    assert list(METHODS) == sorted(METHODS)
    with pytest.raises(TypeError):
        METHODS["new"] = make_spec()  # read-only: no test can change the registry
    assert all(isinstance(spec, MethodSpec) for spec in METHODS.values())


def test_every_composite_method_declares_its_bases_api_calls_and_groups():
    assert composite_errors(METHODS) == []


def test_composite_errors_name_the_composites_that_hide_their_bases():
    registry = {
        "base": make_spec(calls_api=True, groups=("llm",)),
        "stack": make_spec(calls_api=True, groups=("llm",), depends_on=("base",)),
        "hides_api": make_spec(groups=("llm",), depends_on=("base",)),
        "hides_group": make_spec(calls_api=True, depends_on=("base",)),
        "orphan": make_spec(depends_on=("missing",)),
    }

    assert composite_errors(registry) == ["hides_api", "hides_group", "orphan"]


def test_every_registered_module_to_import_exposes_load():
    # The modules of every installed method, and of every API method (its provider SDK is
    # imported only when it asks), import in this job and expose load(folder).
    modules = {spec.module for spec in installed_methods().values()}
    modules |= {spec.module for spec in METHODS.values() if spec.calls_api}

    assert [
        name for name in sorted(modules) if not hasattr(importlib.import_module(name), "load")
    ] == []


def test_every_committed_method_folder_names_a_registered_method():
    stored = [
        json.loads(path.read_text(encoding="utf-8"))["method"]
        for path in METHODS_ASSETS_DIR.glob(f"*/model/{METHOD_FILE}")
    ]

    assert [name for name in stored if name not in METHODS] == []


def test_method_spec_defaults_to_a_local_bit_reproducible_method_of_the_raw_columns():
    spec = make_spec()

    assert (spec.groups, spec.calls_api, spec.score_atol, spec.depends_on) == ((), False, 0.0, ())
    assert spec.input_columns == RAW_INPUT_COLUMNS
    assert METHOD_KINDS == ("score", "fitted", "zero_shot", "other")


def test_method_spec_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="'llm' is not a method kind"):
        make_spec(kind="llm")


@pytest.mark.parametrize("field", ["requirements", "groups", "depends_on", "input_columns"])
def test_method_spec_refuses_a_str_where_a_tuple_is_due(field):
    with pytest.raises(TypeError, match=rf"MethodSpec.{field} must be a tuple of str"):
        make_spec(**{field: "llm"})


@pytest.mark.parametrize("column", ["tags", "part", "fold", "group"])
def test_method_spec_refuses_input_columns_that_hold_the_labels_or_the_split(column):
    # check_tagging_method refuses a method that reads the labels only by feeding it its
    # input_columns: a spec that lists them would let it read the true tags.
    with pytest.raises(ValueError, match=rf"input_columns hold \['{column}'\]"):
        make_spec(input_columns=(*RAW_INPUT_COLUMNS, column))


def test_method_spec_survives_a_json_round_trip():
    spec = make_spec(requirements=("numpy",), groups=("llm",), calls_api=True, score_atol=1e-6)

    assert MethodSpec(**json.loads(json.dumps(asdict(spec)))) == spec


def test_installed_methods_keep_the_specs_whose_groups_are_all_installed(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[dependency-groups]\n"
        'present = ["pandas>=2.2", "numpy"]\n'
        'absent = ["numpy", "surely-not-an-installed-distribution>=1"]\n'
        'nested = [{include-group = "present"}]\n'
        'nested_absent = ["numpy", {include-group = "absent"}]\n',
        encoding="utf-8",
    )
    registry = {
        "core": make_spec(),
        "present": make_spec(groups=("present",)),
        "absent": make_spec(groups=("present", "absent")),
        "nested": make_spec(groups=("nested",)),
        "nested_absent": make_spec(groups=("nested_absent",)),
    }

    installed = installed_methods(registry, pyproject=pyproject)

    assert list(installed) == ["core", "present", "nested"]


def test_installed_methods_refuse_a_group_that_pyproject_does_not_define(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[dependency-groups]\ndev = ["pytest"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="'llm', which .* does not define"):
        installed_methods({"prompted": make_spec(groups=("llm",))}, pyproject=pyproject)


def test_installed_methods_read_the_project_dependency_groups():
    registry = {"tracked": make_spec(groups=("modeling",))}

    assert installed_methods(registry) == registry


def test_missing_groups_name_the_groups_of_one_method_that_are_not_installed_in_its_order(
    tmp_path,
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[dependency-groups]\n"
        'present = ["numpy"]\n'
        'absent = ["surely-not-an-installed-distribution>=1"]\n'
        'other = ["surely-not-an-installed-distribution>=1"]\n',
        encoding="utf-8",
    )
    spec = make_spec(groups=("other", "present", "absent"))

    assert missing_groups("three", spec, pyproject=pyproject) == ["other", "absent"]
    assert missing_groups("core", make_spec(), pyproject=pyproject) == []
    with pytest.raises(ValueError, match="'llm', which .* does not define"):
        missing_groups("prompted", make_spec(groups=("llm",)), pyproject=pyproject)


def test_method_dir_and_model_dir_live_under_the_methods_assets():
    assert method_dir("label_prior") == METHODS_ASSETS_DIR / "label_prior"
    assert model_dir("label_prior") == METHODS_ASSETS_DIR / "label_prior" / "model"
    assert model_dir("label_prior").name == MODEL_FOLDER


@pytest.mark.parametrize("name", ["/etc", "label_prior/", "../label_prior", "Label_Prior", ""])
def test_method_dir_refuses_a_name_that_is_not_a_method_name(name):
    # A path would read the frozen files of another folder, or alias a method's own.
    with pytest.raises(ValueError, match="is not a method name"):
        method_dir(name)


def test_a_constant_method_is_a_tagging_method_but_not_a_score_method(fakes):
    method = fakes.ConstantMethod("constant", TAGS, ["math"])

    assert isinstance(method, TaggingMethod)
    assert not isinstance(method, ScoreMethod)
    assert not isinstance(object(), TaggingMethod)


def test_a_tagging_method_with_scores_is_a_score_method(fakes, two_problems):
    method = fakes.ScoredMethod("constant", TAGS, ["math"])

    assert isinstance(method, ScoreMethod)
    assert method.scores(two_problems)["math"].tolist() == [1.0, 1.0]
    assert method.predict_from_scores(method.scores(two_problems)).equals(
        method.predict(two_problems)
    )


def test_sibling_methods_are_the_other_methods_of_the_package_whatever_their_module():
    def spec(module):
        return MethodSpec(module=module, kind="score", requirements=())

    registry = METHODS | {
        "modernbert_other": spec("codeforces_tags.methods.modernbert.other"),
        "elsewhere": spec("codeforces_tags.methods.elsewhere.method"),
    }

    assert sibling_methods("modernbert_lora") == ["modernbert_base_full"]
    assert sibling_methods("modernbert_lora", registry) == [
        "modernbert_base_full",
        "modernbert_other",
    ]
    assert sibling_methods("label_prior", registry) == []
    assert sibling_methods("tfidf", registry) == []


def test_save_method_writes_method_json_then_the_method_files(fakes, tmp_path):
    folder = tmp_path / "model"

    save_method(fakes.ConstantMethod("constant", TAGS, ["math"]), folder)

    stored = json.loads((folder / METHOD_FILE).read_text(encoding="utf-8"))
    assert stored == {
        "method": "constant",
        "format_version": METHOD_FORMAT_VERSION,
        "tags": list(TAGS),
    }
    assert sorted(path.name for path in folder.iterdir()) == ["constant.json", METHOD_FILE]


def test_load_method_returns_the_saved_method(fakes, registry, two_problems, tmp_path):
    method = fakes.ConstantMethod("constant", TAGS, ["math", "trees"])
    save_method(method, tmp_path / "model")

    loaded = load_method(tmp_path / "model", registry=registry)

    assert (loaded.name, loaded.tags) == ("constant", TAGS)
    pd.testing.assert_frame_equal(loaded.predict(two_problems), method.predict(two_problems))


def test_a_moved_method_folder_loads(fakes, registry, two_problems, tmp_path):
    method = fakes.ConstantMethod("constant", TAGS, ["games"])
    save_method(method, tmp_path / "here" / "model")

    moved = Path(shutil.move(tmp_path / "here" / "model", tmp_path / "elsewhere"))

    for path in moved.rglob("*"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    loaded = load_method(moved, registry=registry)
    pd.testing.assert_frame_equal(loaded.predict(two_problems), method.predict(two_problems))


def test_load_method_needs_a_method_folder(registry, tmp_path):
    with pytest.raises(FileNotFoundError, match="method.json"):
        load_method(tmp_path, registry=registry)


def rewrite_method_file(folder, **changes):
    """Change keys of a saved ``method.json``."""
    path = folder / METHOD_FILE
    path.write_text(
        json.dumps(json.loads(path.read_text(encoding="utf-8")) | changes), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("oops", "method.json is not valid JSON"),
        ("[]", r"lacks the keys \['method', 'format_version', 'tags'\]"),
        ('{"method": "constant"}', r"lacks the keys \['format_version', 'tags'\]"),
    ],
)
def test_load_method_refuses_a_malformed_method_json(registry, tmp_path, text, match):
    (tmp_path / METHOD_FILE).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_method(tmp_path, registry=registry)


def test_load_method_refuses_another_format_version(fakes, registry, tmp_path):
    save_method(fakes.ConstantMethod("constant", TAGS, []), tmp_path)
    rewrite_method_file(tmp_path, format_version=METHOD_FORMAT_VERSION + 1)

    with pytest.raises(ValueError, match="format version 2"):
        load_method(tmp_path, registry=registry)


def test_load_method_refuses_an_unregistered_method(fakes, tmp_path):
    save_method(fakes.ConstantMethod("constant", TAGS, []), tmp_path)

    with pytest.raises(ValueError, match="'constant', which is not registered"):
        load_method(tmp_path)


def test_load_method_refuses_a_method_of_another_name(fakes, registry, tmp_path):
    save_method(fakes.ConstantMethod("constant", TAGS, []), tmp_path)
    rewrite_method_file(tmp_path, method="renamed")

    with pytest.raises(ValueError, match="loaded 'constant'"):
        load_method(tmp_path, registry=registry | {"renamed": registry["constant"]})


def test_load_method_refuses_tags_that_differ_from_method_json(fakes, registry, tmp_path):
    save_method(fakes.ConstantMethod("constant", TAGS, []), tmp_path)
    rewrite_method_file(tmp_path, tags=list(reversed(TAGS)))

    with pytest.raises(ValueError, match="tags"):
        load_method(tmp_path, registry=registry)


def test_importing_the_core_imports_no_heavy_library():
    imports = "; ".join(f"import {module}" for module in CORE_MODULES)
    code = f"import sys; {imports}; print(sorted(set({HEAVY_MODULES!r}) & set(sys.modules)))"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


def test_label_prior_is_registered_as_a_fitted_method_served_by_its_method_module():
    assert METHODS["label_prior"] == MethodSpec(
        module="codeforces_tags.methods.label_prior.method",
        kind="fitted",
        requirements=("numpy", "pandas"),
    )
    assert "label_prior" in installed_methods()


def test_tfidf_is_registered_as_a_score_method_served_by_the_tag_model_module():
    assert METHODS["tfidf"] == MethodSpec(
        module="codeforces_tags.model.tag_model",
        kind="score",
        requirements=("numpy", "pandas", "scikit-learn", "scipy", "skops"),
    )
    assert "tfidf" in installed_methods()


API_METHODS = [
    "code2tag-json",
    "code2tag-text",
    "descode2tag-json",
    "descode2tag-text",
    "jev-statement",
    "jev-statement-code",
    "jev-statement-code-gepa",
    "jev-statement-code-gepa-platt",
    "jev-statement-code-platt",
]


def test_every_api_method_lists_pyyaml_since_its_module_imports_the_prompt_file_reader():
    api = {name: spec for name, spec in METHODS.items() if spec.calls_api}
    modules = sorted({spec.module for spec in api.values()})
    code = (
        "import importlib, sys\n"
        f"for module in {modules!r}:\n"
        "    importlib.import_module(module)\n"
        "print('yaml' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert sorted(api) == API_METHODS
    assert result.stdout.strip() == "True"
    assert [name for name, spec in api.items() if "pyyaml" not in spec.requirements] == []
