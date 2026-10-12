"""Tests for the documentation of serving: ADR 0012, the glossary, the agent instructions, README.

``tests/test_project_files.py`` checks the shape of every ADR and glossary entry, that the
agent instructions link every ADR and name every module, and that every relative link resolves;
these tests check what the documents say about serving a tagging method (docs/adr/0012).
"""

import ast
import itertools
import re

from codeforces_tags.config import DEVICE_ENV_VAR, LEGACY_DEVICE_ENV_VAR, PROJECT_ROOT

ADR_0012 = PROJECT_ROOT / "docs" / "adr" / "0012-serving-tagging-methods.md"
CLAUDE = PROJECT_ROOT / "CLAUDE.md"
GLOSSARY = PROJECT_ROOT / "CONTEXT.md"
README = PROJECT_ROOT / "README.md"
APP = PROJECT_ROOT / "apps" / "streamlit_app.py"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
TRACKING_INIT = PROJECT_ROOT / "src" / "codeforces_tags" / "tracking" / "__init__.py"
SERVING = PROJECT_ROOT / "src" / "codeforces_tags" / "serving"
# The serving modules that TID253's per-file-ignores let import mlflow at module level.
MLFLOW_MODULES = ("serving/entry.py", "serving/package.py", "serving/__main__.py")
# An import of mlflow, at module level or inside a function.
MLFLOW_IMPORT = re.compile(r"^\s*(?:import|from) mlflow\b", flags=re.MULTILINE)
# The commands that package, build, run, check, clean up, deploy and link a served method, then
# start all of it in one terminal and run notebook 12, in the order the Serving part of the agent
# instructions runs them.
SERVING_COMMANDS = [
    "python -m codeforces_tags.serving package --method f2llm_late_fusion",
    "python -m codeforces_tags.serving build --method f2llm_late_fusion",
    "docker compose -f deploy/docker-compose.yml --profile cpu up --wait",
    "python -m codeforces_tags.serving smoke --method f2llm_late_fusion",
    "python -m codeforces_tags.serving clean-images --method f2llm_late_fusion",
    "kind create cluster --name codeforces-tags",
    "python -m codeforces_tags.serving manifests --method tfidf --device cpu",
    "kubectl --context kind-codeforces-tags apply -k build/k8s/tfidf-cpu",
    "python -m codeforces_tags.serving smoke --url http://127.0.0.1:18080 --method tfidf",
    "kind delete cluster --name codeforces-tags",
    "deploy/sagemaker/deploy.sh",
    "deploy/sagemaker/invoke.sh payload.json answer.json",
    "deploy/sagemaker/delete.sh",
    "python -m codeforces_tags.serving mlflow-ui",
    "uv run --group app streamlit run apps/streamlit_app.py",
    "uv run --group app python -m codeforces_tags.serving up --method f2llm_late_fusion\n",
    "serving up --method f2llm_late_fusion --device gpu",
    "--group f2llm python -m codeforces_tags.serving up --method f2llm_late_fusion --local",
    "--inplace notebooks/12_model_serving.ipynb",
]
# The options ADR 0012 rejected: those of each of its decisions, and MLflow's client in the app.
OPTIONS = (
    "- **A Flask app of our own**",
    "- **`mlflow models build-docker`**",
    "- **Weights mounted at run time**",
    "- **Two images per method",
    f"- **Keeping the F2LLM-only `{LEGACY_DEVICE_ENV_VAR}`**",
    "- **Tags only, as the logged wrapper returns them**",
    "- **The logged model of the MLflow store**",
    "- **A Streamlit container in the compose file**",
    "- **`MlflowClient` in the app**",
    "- **A fully live notebook 12, or an explanatory one**",
    "- **A real SageMaker endpoint, or manifests only**",
)


def serving_commands() -> str:
    """Return the text of the Serving part of the agent instructions' Commands."""
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]
    return commands.split("\n### Serving\n")[1]


def test_adr_0012_serves_a_frozen_method_from_its_committed_folder_with_mlflows_server():
    text = ADR_0012.read_text(encoding="utf-8")
    decision, options = text.split("\n## Considered options\n")
    options, consequences = options.split("\n## Consequences\n")

    assert text.startswith(
        "# A tagging method is served by MLflow's model server, from its committed folder\n"
    )
    for expected in (
        "`assets/methods/<name>/model/`",
        "`mlflow.pyfunc.save_model`",
        "`serving/entry.py`",
        "`mlflow.models.container._serve('local')`",
        "`MLFLOW_MODELS_WORKERS=1`",
        "`ARG METHOD`",
        f"`{DEVICE_ENV_VAR}`",
        f"`{LEGACY_DEVICE_ENV_VAR}`",
        "`ScoreMethod.tag_thresholds()`",
        "exactly one problem per request",
        "`serving mlflow-ui`",
        "prompt registry",
        "`jev-tags`",
        "recorded outputs",
        "[ADR 0001](0001-model-selection-protocol.md)",
        "[ADR 0002](0002-method-folders-and-mlflow-wrapper.md)",
        "[ADR 0003](0003-api-methods-are-fixed-reference-models.md)",
        "[ADR 0006](0006-trained-weight-cache.md)",
        "[ADR 0008](0008-cli-default-method.md)",
    ):
        assert expected in decision
    for option in OPTIONS:
        assert option in options
    for expected in (
        "never the store",
        "`load_logged_method`",
        "`config.PROJECT_ROOT` is the model folder",
        "chooses nothing",
        "SageMaker",
    ):
        assert expected in consequences


def test_the_serving_documents_say_what_the_recorded_runs_found():
    # Compose's gpu profile ran once the NVIDIA Container Toolkit was installed, and measured a
    # worker's GPU memory; the method's run page shows no parameter (the logged model holds
    # them), and build's prune drops every cached build context of the host.
    adr = " ".join(ADR_0012.read_text(encoding="utf-8").split())
    links = ast.get_docstring(ast.parse((SERVING / "mlflow_links.py").read_text(encoding="utf-8")))
    run_page = "its logged model, which holds the run record's parameters and the test metrics"

    assert "compose on the CPU and on the GPU" in adr
    assert "took 0.148 s at the median and 0.262 s at the slowest" in adr
    assert "2,432 MiB of GPU memory for `f2llm_late_fusion`, as its one worker held" in adr
    assert "pending" not in adr
    for text in (adr, links, APP.read_text(encoding="utf-8")):
        assert run_page in " ".join(text.split())
    assert "drops every cached build context on the host, not only this method's" in adr
    assert "every one on the host, not only this method's" in serving_commands()


def test_the_glossary_defines_the_model_server_apart_from_sagemakers_endpoint():
    text = GLOSSARY.read_text(encoding="utf-8")
    entry = re.search(r"^\*\*Model server\*\*:\n([^\n]+)\n_Avoid_: ([^\n]+)$", text, re.MULTILINE)

    assert "`/ping` and `/invocations`" in entry.group(1)
    assert "endpoint (SageMaker's deployment)" in entry.group(2)


def test_the_layout_names_the_deploy_files_and_the_streamlit_app():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    deploy = next(line for line in lines if line.startswith("- `deploy/`: "))
    apps = next(line for line in lines if line.startswith("- `apps/`: "))

    for name in ("`Dockerfile`", "`docker-compose.yml`", "`k8s/`", "`sagemaker/`"):
        assert name in deploy
    for name in ("`.env.serving.example`", "`build/serving/<method>/`", "`build/k8s/"):
        assert name in deploy
    assert "docs/adr/0012-serving-tagging-methods.md" in deploy
    for name in ("`streamlit_app.py`", "`.streamlit/config.toml`", "`serving/mlflow_links.py`"):
        assert name in apps


def test_the_commands_package_run_check_deploy_and_link_a_served_method_in_this_order():
    section = serving_commands()

    positions = [section.find(command) for command in SERVING_COMMANDS]

    assert -1 not in positions
    assert positions == sorted(positions)
    assert "docs/adr/0012-serving-tagging-methods.md" in section


def test_the_serving_commands_hold_kubectl_to_its_context_and_run_what_this_machine_can():
    # Every kubectl call names its context, SageMaker's example is a CPU endpoint (deploy.sh
    # refuses a GPU one on this machine's aws CLI), invoke.sh's payload is said, and the MLflow
    # UI, which holds its shell, leaves the app to another one.
    section = serving_commands()
    calls = [line for line in section.splitlines() if line.startswith("kubectl ")]
    calls += re.findall(r"`(kubectl [^`]+)`", section)

    assert any("port-forward" in call for call in calls)  # the scan sees the block's calls
    assert any("create secret" in call for call in calls)  # and the one in the text
    assert all(call.startswith("kubectl --context kind-codeforces-tags ") for call in calls)
    assert "export METHOD=f2llm_late_fusion DEVICE=cpu " in section
    assert "`serving.client.request_payload`" in section
    assert "serving mlflow-ui   # then, in another shell:" in section


def test_every_list_of_the_modules_that_import_mlflow_names_the_serving_modules_that_may():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    tuning = next(line for line in lines if line.startswith("  - `tuning/`: "))
    heavy_imports = next(line for line in lines if line.startswith("- **Heavy imports "))
    import_time = next(line for line in lines if line.startswith("- **MLflow's import-time "))
    docstring = ast.get_docstring(ast.parse(TRACKING_INIT.read_text(encoding="utf-8")))
    ignores = PYPROJECT.read_text(encoding="utf-8").split("\n[tool.ruff.lint.per-file-ignores]\n")
    comment = itertools.takewhile(lambda line: line.startswith("#"), ignores[1].splitlines())
    importers = [
        f"serving/{path.name}"
        for path in SERVING.glob("*.py")
        if MLFLOW_IMPORT.search(path.read_text(encoding="utf-8"))
    ]

    assert "serving/entry.py" in importers  # the scan sees the entry file's import
    assert set(importers) <= set(MLFLOW_MODULES)
    for text in (tuning[: tuning.index("; only ")], heavy_imports):
        assert all(f"`{module}`" in text for module in MLFLOW_MODULES)
    for text in (" ".join(docstring.split()), " ".join(comment)):
        assert all(module in text for module in MLFLOW_MODULES)
    assert "`serving/`" in import_time  # its __init__.py sets MLflow's import-time variables


def test_the_readme_tries_a_method_with_up_first_then_serves_tfidf_step_by_step():
    readme = README.read_text(encoding="utf-8")
    serving = readme.split("\n## Serving\n")[1].split("\n## ")[0]
    steps = [
        "uv run --group app python -m codeforces_tags.serving up --method f2llm_late_fusion",
        "uv run python -m codeforces_tags.serving build --method tfidf",
        "METHOD=tfidf docker compose -f deploy/docker-compose.yml --profile cpu up --wait",
        "uv run python -m codeforces_tags.serving smoke --method tfidf",
        "uv run --group app streamlit run apps/streamlit_app.py",
    ]

    positions = [serving.find(step) for step in steps]

    assert -1 not in positions
    assert positions == sorted(positions)
    assert "docs/adr/0012-serving-tagging-methods.md" in serving
    assert "mlflow-ui" in serving
