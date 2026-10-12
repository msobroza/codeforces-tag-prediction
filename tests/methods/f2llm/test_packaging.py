"""Tests for the wiring of the f2llm package: its dependency group, lint, coverage and CI job."""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from codeforces_tags.config import PROJECT_ROOT, PYPROJECT_FILE, UV_LOCK_FILE

CI_FILE = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
COVERAGE_FILE = Path(__file__).with_name("coverage.toml")
# The group's requirements. The floors match the `finetuning` group's (the ModernBERT methods), so
# that both groups lock the same torch, transformers and peft.
GROUP = {
    "huggingface-hub": ">=1.0",
    "peft": ">=0.21",
    "safetensors": ">=0.4",
    "torch": ">=2.9",
    "transformers": ">=5.19,<6",
}
# The modules of the group that the package imports: pylint must not look for them in the default
# job, which does not install them.
GROUP_MODULES = ["huggingface_hub", "peft", "safetensors", "tokenizers", "torch", "transformers"]
# The fresh import of every command: the default job, without torch, skips the f2llm command, so
# the f2llm job runs it.
FRESH_IMPORT_TEST = (
    "tests/tracking/test_environment.py::test_no_command_lets_mlflow_start_telemetry_or_hint"
)


@pytest.fixture(name="pyproject")
def pyproject_fixture():
    """Return the parsed ``pyproject.toml``."""
    return tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))


def test_the_f2llm_group_holds_the_gpu_stack_and_is_not_installed_by_default(pyproject):
    requirements = pyproject["dependency-groups"]["f2llm"]
    split = [re.fullmatch(r"([A-Za-z0-9_.-]+)(.*)", item).groups() for item in requirements]

    assert dict(split) == GROUP
    assert "f2llm" not in pyproject["tool"]["uv"]["default-groups"]


def test_the_lock_file_pins_the_group_that_the_ci_job_syncs():
    lock = tomllib.loads(UV_LOCK_FILE.read_text(encoding="utf-8"))
    project = next(package for package in lock["package"] if package["name"] == "codeforces-tags")
    locked = project["metadata"]["requires-dev"]["f2llm"]

    assert {item["name"]: item["specifier"] for item in locked} == GROUP
    assert {"torch", "transformers", "peft"} <= {package["name"] for package in lock["package"]}


def test_the_default_job_ignores_the_f2llm_tests_and_leaves_the_package_out_of_its_coverage(
    pyproject,
):
    omitted = pyproject["tool"]["coverage"]["run"]["omit"]

    assert "--ignore=tests/methods/f2llm" in pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert {"src/codeforces_tags/methods/f2llm/*", "tests/methods/f2llm/*"} <= set(omitted)


def test_pylint_does_not_report_the_libraries_of_the_group_as_missing_imports(pyproject):
    ignored = pyproject["tool"]["pylint"]["typecheck"]["ignored-modules"]

    assert sorted(ignored) == GROUP_MODULES


def test_the_group_measures_its_package_and_its_tests_and_omits_nothing():
    coverage = tomllib.loads(COVERAGE_FILE.read_text(encoding="utf-8"))["tool"]["coverage"]

    assert coverage == {
        "run": {
            "source_dirs": ["src/codeforces_tags/methods/f2llm", "tests/methods/f2llm"],
            "branch": True,
        }
    }


def test_the_f2llm_job_syncs_the_locked_group_offline_and_holds_the_package_to_its_floors():
    workflow = yaml.safe_load(CI_FILE.read_text(encoding="utf-8"))
    job = workflow["jobs"]["f2llm"]
    commands = [step["run"] for step in job["steps"] if "run" in step]

    assert "check" in workflow["jobs"]
    assert job["env"] == {"HF_HUB_OFFLINE": "1"}
    assert commands[0] == "uv sync --locked --group f2llm"
    assert commands[1].splitlines() == [
        "uv run pytest tests/methods/test_registry.py tests/methods/f2llm "
        f'"{FRESH_IMPORT_TEST}" --cov-config=tests/methods/f2llm/coverage.toml',
        'uv run coverage report --include="src/codeforces_tags/methods/f2llm/*" --fail-under=95',
        'uv run coverage report --include="tests/methods/f2llm/*" --fail-under=100',
    ]
