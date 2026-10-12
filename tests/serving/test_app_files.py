"""Tests for the files of the Streamlit app: its dependency group, its settings and its imports.

The app (``apps/streamlit_app.py``, docs/adr/0012) runs on the user's machine only, with the
``app`` group, which CI never installs: these tests read its files and never import streamlit.
"""

import ast
import json
import tomllib

from codeforces_tags.config import PROJECT_ROOT

PYPROJECT = PROJECT_ROOT / "pyproject.toml"
STREAMLIT_APP = PROJECT_ROOT / "apps" / "streamlit_app.py"
# The settings Streamlit reads when `streamlit run` starts from the repository root.
STREAMLIT_CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"
# Runs the import statements of argv[1], then prints, as JSON, which of mlflow and torch they
# loaded.
HEAVY_AFTER_IMPORTS = """
import json, sys
exec(sys.argv[1])
print(json.dumps(sorted({"mlflow", "torch"} & {name.split(".")[0] for name in sys.modules})))
"""


def test_the_app_dependency_group_holds_streamlit_and_is_not_installed_by_default():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    [requirement] = pyproject["dependency-groups"]["app"]

    assert requirement.startswith("streamlit")
    assert "app" not in pyproject["tool"]["uv"]["default-groups"]


def test_streamlit_sends_no_usage_statistics_and_listens_on_this_machine_only():
    config = tomllib.loads(STREAMLIT_CONFIG.read_text(encoding="utf-8"))

    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "127.0.0.1"


def test_the_app_imports_streamlit_altair_and_the_package_but_neither_mlflow_nor_torch(
    agent_python,
):
    # The app holds layout only: every rule it applies comes from codeforces_tags. Its imports of
    # the package must name what exists, and load neither mlflow nor torch, in a fresh
    # interpreter of a coding agent's shell (agent_python, tests/conftest.py).
    tree = ast.parse(STREAMLIT_APP.read_text(encoding="utf-8"))
    froms = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    names = [
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    ]
    libraries = {name.split(".")[0] for name in names + [str(node.module) for node in froms]}
    package = [ast.unparse(node) for node in froms if str(node.module).startswith("codeforces_")]

    output, _ = agent_python(HEAVY_AFTER_IMPORTS, "\n".join(package))

    assert {"altair", "streamlit"} <= libraries
    assert not {"mlflow", "torch"} & libraries
    assert any("codeforces_tags.serving.mlflow_links" in line for line in package)
    assert json.loads(output) == []
