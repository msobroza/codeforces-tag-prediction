"""Tests for the project's own files: lint rules, ignores, ADRs, glossary and agent instructions."""

import ast
import itertools
import json
import re
import subprocess
import sys
import tomllib

import pytest

from codeforces_tags.config import (
    CLI_DEFAULT_METHOD,
    METHODS_ASSETS_DIR,
    MLFLOW_SCREEN_COMPARE,
    MLFLOW_UI_PORT,
    PROJECT_ROOT,
)
from codeforces_tags.evaluation.timing import latency_summary, load_latency
from codeforces_tags.methods import METHODS
from codeforces_tags.model.tag_model import STORED_TYPES

# Libraries that no prediction path imports at module level (ruff's TID253).
HEAVY_LIBRARIES = [
    "claude_agent_sdk",
    "gepa",
    "matplotlib",
    "mlflow",
    "openai",
    "optuna",
    "playwright",
    "seaborn",
    "typesafe_sdk",
]
# Where the prediction path lives: the core and a method's module.
PREDICTION_PATH_FILE = "src/codeforces_tags/methods/example/method.py"
# The files that may import them, whatever their folder holds (per-file-ignores of pyproject.toml).
EXEMPT_FILES = [
    "src/codeforces_tags/analysis/example.py",
    "src/codeforces_tags/tuning/search.py",
    "src/codeforces_tags/tuning/tracking.py",
    "src/codeforces_tags/tuning/__main__.py",
    "src/codeforces_tags/tracking/example.py",
    "src/codeforces_tags/llm/__main__.py",
    "src/codeforces_tags/methods/example/__main__.py",
    "src/codeforces_tags/methods/example/tracking.py",
    "src/codeforces_tags/methods/example/tuning.py",
    "src/codeforces_tags/methods/typesafe_jev/search_commands.py",
    "notebooks/figures/example.py",
    "tests/example.py",
]
# The other files of the LLM toolkit: only its command is exempt, so none of these imports mlflow
# or a provider's SDK at module level (an adapter imports its SDK inside its ask).
LLM_TOOLKIT_FILES = [
    "src/codeforces_tags/llm/prompts.py",
    "src/codeforces_tags/llm/runner.py",
    "src/codeforces_tags/llm/providers/openai.py",
]
# The modules of the TypeSafe Jev package that are not its command: none imports mlflow or the
# TypeSafe SDK at module level (the SDK is imported inside client.jev_session).
TYPESAFE_FILES = [
    "src/codeforces_tags/methods/typesafe_jev/__init__.py",
    "src/codeforces_tags/methods/typesafe_jev/settings.py",
    "src/codeforces_tags/methods/typesafe_jev/words.py",
    "src/codeforces_tags/methods/typesafe_jev/questions.py",
    "src/codeforces_tags/methods/typesafe_jev/client.py",
    "src/codeforces_tags/methods/typesafe_jev/answers.py",
    "src/codeforces_tags/methods/typesafe_jev/runner.py",
    "src/codeforces_tags/methods/typesafe_jev/method.py",
    "src/codeforces_tags/methods/typesafe_jev/search_record.py",
    "src/codeforces_tags/methods/typesafe_jev/export.py",
    "src/codeforces_tags/methods/typesafe_jev/jev_cache.py",
    "src/codeforces_tags/methods/typesafe_jev/platt.py",
    "src/codeforces_tags/methods/typesafe_jev/platt_commands.py",
    "src/codeforces_tags/methods/typesafe_jev/search_tracking.py",
]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
ADR_DIR = PROJECT_ROOT / "docs" / "adr"
ADRS = sorted(ADR_DIR.glob("*.md"))
GLOSSARY = PROJECT_ROOT / "CONTEXT.md"
CLAUDE = PROJECT_ROOT / "CLAUDE.md"
AGENTS = PROJECT_ROOT / "AGENTS.md"
PACKAGE = PROJECT_ROOT / "src" / "codeforces_tags"
STUDY_README = PROJECT_ROOT / "assets" / "tfidf_baseline" / "README.md"
README = PROJECT_ROOT / "README.md"
NOTEBOOKS_README = PROJECT_ROOT / "notebooks" / "README.md"
# The six methods of notebook 05, in the order the store's evaluation runs are made: the two
# references, then the zero-shot methods, with the one the MLflow screenshots compare last.
EVALUATED = (
    "label_prior",
    "tfidf",
    "code2tag-text",
    "code2tag-json",
    "descode2tag-text",
    "descode2tag-json",
)
# The three commands of the command line that PROBLEM.md asks for, as they are run.
CLI_COMMANDS = ["codeforces-tags methods", "codeforces-tags predict", "codeforces-tags evaluate"]
# The commands that run the tuning study and freeze, check, time and calibrate the tfidf method,
# in the order they run.
STUDY_COMMANDS = [
    "tuning run",
    "tuning export",
    "tuning freeze",
    "tuning verify",
    "tracking latency",
    "tracking calibrate",
]
# The terms of the glossary that come with the tagging-method framework, its LLM toolkit, the
# zero-shot methods and the TypeSafe methods.
FRAMEWORK_TERMS = [
    "Tagging method",
    "Method kind",
    "Method folder",
    "Run record",
    "Self-identifying solution",
    "Evaluation dataset",
    "Evaluation run",
    "All-tag macro F1",
    "Prompted method",
    "Output format",
    "Answer",
    "Zero-shot setting",
    "Zero-shot method",
    "Tag definition",
    "Prompt file",
]
# The terms of the glossary that come with the F2LLM embedding methods.
EMBEDDING_TERMS = ["Statement input / solution input", "Encoder input", "Embedding block"]
# The terms of the glossary that come with the search over Jev's words (docs/adr/0010).
SEARCH_TERMS = [
    "Search",
    "Learning folds",
    "Selection fold",
    "Held-out fold",
    "Candidate words",
    "Reflection model",
    "Balanced Brier score",
]
# The ADR of the search over Jev's words.
ADR_0010 = ADR_DIR / "0010-searched-jev-words.md"
# The searched Jev method (docs/adr/0010) and its package.
SEARCHED = "jev-statement-code-gepa"
JEV_PACKAGE = PACKAGE / "methods" / "typesafe_jev"
# The commands that search, record, answer, freeze, check and time the searched Jev method, in the
# order they run.
SEARCH_COMMANDS = [
    f"methods.typesafe_jev tune --method {SEARCHED} --max-cost-usd 15",
    f"methods.typesafe_jev export --method {SEARCHED}",
    f"methods.typesafe_jev log-search --method {SEARCHED}",
    f"methods.typesafe_jev run --method {SEARCHED} --part dev",
    f"methods.typesafe_jev run --method {SEARCHED} --part test",
    f"methods.typesafe_jev freeze --method {SEARCHED}",
    f"methods.typesafe_jev verify --method {SEARCHED}",
    f"tracking latency --method {SEARCHED} --max-cost-usd 0.05",
    f"tracking evaluate --method {SEARCHED} --part dev",
    f"tracking evaluate --method {SEARCHED} --part test",
    "--inplace notebooks/10_jev_gepa.ipynb",
]
F2LLM_PACKAGE = PACKAGE / "methods" / "f2llm"
# The ADR of the versioned prompt files, and the link to it from another ADR.
ADR_0009 = ADR_DIR / "0009-versioned-prompt-files.md"
ADR_0009_LINK = "[ADR 0009](0009-versioned-prompt-files.md)"
# The target of a markdown link to a file of the repository (not a web page or a heading).
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#)([^)#\s]+)")
# An import of optuna, at module level or inside a function.
OPTUNA_IMPORT = re.compile(r"^\s*(?:import|from) optuna\b", flags=re.MULTILINE)


def ruff_codes(source: str, filename: str) -> list[str]:
    """Return the codes of the banned-import and banned-API rules that ruff reports.

    Args:
        source: The Python source to lint.
        filename: The path to lint it as, relative to the project root: ruff picks the
            per-file-ignores of ``pyproject.toml`` by it.

    Returns:
        The rule codes, one per violation, in the order of the file.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "TID251,TID253",
            "--output-format",
            "json",
            "--stdin-filename",
            filename,
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,  # ruff exits with 1 when it reports a violation
    )
    return [violation["code"] for violation in json.loads(result.stdout)]


@pytest.mark.parametrize("library", HEAVY_LIBRARIES)
def test_a_module_level_import_of_a_heavy_library_fails_lint_on_the_prediction_path(library):
    assert ruff_codes(f"import {library}\n", PREDICTION_PATH_FILE) == ["TID253"]


@pytest.mark.parametrize("filename", EXEMPT_FILES)
def test_tracking_tuning_analysis_figure_and_test_files_may_import_the_heavy_libraries(filename):
    source = "".join(f"import {library}\n" for library in HEAVY_LIBRARIES)

    assert not ruff_codes(source, filename)


@pytest.mark.parametrize("filename", LLM_TOOLKIT_FILES)
def test_only_the_llm_command_may_import_mlflow_or_a_provider_sdk_at_module_level(filename):
    assert ruff_codes("import mlflow\nimport openai\n", filename) == ["TID253", "TID253"]


@pytest.mark.parametrize("filename", TYPESAFE_FILES)
def test_the_typesafe_modules_may_import_neither_mlflow_nor_the_sdk_at_module_level(filename):
    assert ruff_codes("import mlflow\nimport typesafe_sdk\n", filename) == ["TID253", "TID253"]


def test_the_llm_dependency_group_holds_openai_and_is_not_installed_by_default():
    # CI installs the default groups only, so it installs no openai: every test fakes it.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert any(
        requirement.startswith("openai") for requirement in pyproject["dependency-groups"]["llm"]
    )
    assert "llm" not in pyproject["tool"]["uv"]["default-groups"]


def test_the_llm_typesafe_dependency_group_holds_typesafe_sdk_and_is_not_installed_by_default():
    # CI installs the default groups only, so it installs no typesafe-sdk: every test fakes it.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert any(
        requirement.startswith("typesafe-sdk")
        for requirement in pyproject["dependency-groups"]["llm-typesafe"]
    )
    assert "llm-typesafe" not in pyproject["tool"]["uv"]["default-groups"]


def test_the_figures_dependency_group_pins_playwright_and_is_not_installed_by_default():
    # CI installs the default groups only, so it installs no playwright: the estimator diagram and
    # the MLflow screenshots are rendered by hand (notebooks/README.md, Figures), and an exact pin
    # keeps their render stable.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    [requirement] = pyproject["dependency-groups"]["figures"]

    assert requirement.startswith("playwright==")
    assert "figures" not in pyproject["tool"]["uv"]["default-groups"]


def test_the_modeling_group_pins_gepa_whose_lock_entry_needs_no_other_package():
    # The search over Jev's words (docs/adr/0010) calls gepa.optimize directly. gepa 0.1.4 is pure
    # Python: without its extras (litellm, mlflow, wandb, ...) it requires no other package, and
    # the default groups (so CI) install it.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))

    [gepa] = [package for package in lock["package"] if package["name"] == "gepa"]

    assert "gepa==0.1.4" in pyproject["dependency-groups"]["modeling"]
    assert "modeling" in pyproject["tool"]["uv"]["default-groups"]
    assert gepa["version"] == "0.1.4"
    assert not gepa.get("dependencies")


@pytest.mark.parametrize("call", ["torch.save(model, path)", "torch.load(path)"])
def test_torch_save_and_torch_load_fail_lint_because_they_pickle(call):
    assert ruff_codes(f"import torch\n\n{call}\n", PREDICTION_PATH_FILE) == ["TID251"]


@pytest.mark.parametrize(
    "source",
    [
        "import torch\n\ntorch.serialization.load(path)\n",
        "import torch\n\ntorch.serialization.save(model, path)\n",
        "from torch.serialization import load\n",
        "from torch.serialization import save\n",
        "import _pickle\n",
        "from pandas.io.pickle import read_pickle\n",
        "import pandas as pd\n\npd.io.pickle.read_pickle(path)\n",
    ],
)
def test_the_other_names_of_the_pickle_functions_fail_lint_too(source):
    # torch re-exports torch.serialization's load and save, pandas.read_pickle is
    # pandas.io.pickle's, and pickle's functions are _pickle's.
    assert ruff_codes(source, PREDICTION_PATH_FILE) == ["TID251"]


@pytest.mark.parametrize(
    "path",
    [
        "mlflow.db",
        "mlflow.db-wal",
        "notebooks/mlflow.db",
        "notebooks/mlflow.db-shm",
        "mlartifacts/1/a/artifacts/MLmodel",
        "mlruns/0/meta.yaml",
        ".playwright-mcp/page-1.png",
    ],
)
def test_git_ignores_the_default_mlflow_store_and_playwright_output(path):
    # MLflow 3.16's default store is ./mlflow.db (with its artifacts in ./mlartifacts or
    # ./mlruns), written wherever a client runs before a store is chosen.
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", path], cwd=PROJECT_ROOT, check=False
    )

    assert ignored.returncode == 0


@pytest.mark.parametrize(
    ("path", "status"), [("deploy/.env.serving", 0), ("deploy/.env.serving.example", 1)]
)
def test_git_ignores_the_model_servers_variables_but_not_their_example(path, status):
    # deploy/.env.serving holds API keys; its example is committed. git check-ignore exits with 0
    # for an ignored path and 1 for one that is not; --no-index tests the patterns themselves,
    # which a tracked file would otherwise pass whatever they say.
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path], cwd=PROJECT_ROOT, check=False
    )

    assert checked.returncode == status


def test_the_adrs_of_the_tagging_method_framework_are_recorded():
    assert {"0001", "0002", "0003", "0004"} <= {adr.name[:4] for adr in ADRS}


@pytest.mark.parametrize("adr", ADRS, ids=lambda adr: adr.name)
def test_an_adr_states_its_decision_then_its_options_then_its_consequences(adr):
    text = adr.read_text(encoding="utf-8")
    options, consequences = "\n## Considered options\n", "\n## Consequences\n"

    assert text.startswith("# ")
    assert 0 < text.index(options) < text.index(consequences)


def test_adr_0002_states_each_of_its_seven_safeguards():
    text = (ADR_DIR / "0002-method-folders-and-mlflow-wrapper.md").read_text(encoding="utf-8")

    numbers = re.findall(r"^- \*\*Safeguard (\d), ", text, flags=re.MULTILINE)

    assert numbers == ["1", "2", "3", "4", "5", "6", "7"]


def test_adr_0008_records_the_default_method_of_the_command_line_as_the_users_choice():
    text = (ADR_DIR / "0008-cli-default-method.md").read_text(encoding="utf-8")

    assert text.startswith(
        f"# The command line runs {CLI_DEFAULT_METHOD} by default, a choice of the user\n"
    )
    assert "`config.CLI_DEFAULT_METHOD`" in text
    assert "[ADR 0004](0004-tagging-method-framework.md)" in text
    assert "never on a test score" in text


def test_adr_0008_quotes_the_committed_latency_of_the_default_and_of_the_rules_choice():
    text = (ADR_DIR / "0008-cli-default-method.md").read_text(encoding="utf-8")
    rule = text.split("\n\n")[2]

    assert "ADR 0004's rule names `descode2tag-json`" in rule
    assert "The reason is latency, not cost." in text
    for method in (CLI_DEFAULT_METHOD, "descode2tag-json"):
        summary = latency_summary(load_latency(METHODS_ASSETS_DIR / method).seconds)
        for seconds in (summary["median_s"], summary["max_s"]):
            assert f"{seconds:.3f} s" in text


def test_adr_0004_leaves_the_default_method_of_the_command_line_to_adr_0008():
    text = (ADR_DIR / "0004-tagging-method-framework.md").read_text(encoding="utf-8")
    choosing = next(
        line
        for line in text.splitlines()
        if line.startswith("- **Choosing among frozen methods.**")
    )
    consequences = text.split("\n## Consequences\n")[1]

    assert "[ADR 0008](0008-cli-default-method.md)" in choosing
    assert "[ADR 0008](0008-cli-default-method.md)" in consequences
    assert "The CLI's method is then a mechanical choice" not in consequences


def test_adr_0001_says_that_a_methods_test_predictions_are_logged_once_per_problem():
    text = (ADR_DIR / "0001-model-selection-protocol.md").read_text(encoding="utf-8")

    assert "logged per problem once, in one evaluation run per method" in text
    assert "per-problem comparisons read dev only" in text


def test_adr_0010_fixes_the_protocol_of_a_search_before_it_runs():
    text = ADR_0010.read_text(encoding="utf-8")
    decision, options = text.split("\n## Considered options\n")
    options = options.split("\n## Consequences\n")[0]

    assert text.startswith("# Jev's words may be searched, as a new method read in sample\n")
    for expected in (
        "| 1-3 | the learning folds:",
        "| 4 | the selection fold:",
        "| 0 | the held-out fold:",
        "**The selection rule is fixed here, before the search.**",
        "**A search's requests are records of the search, not a method's answers.**",
        "**The seed check**",
        "**The dataset-text check**",
        "8-word sequence",
        "`config.TUNING_EXPORT_TOLERANCE`",
        "200 iterations",
        "$15",
        '`method_entries["dev_in_sample"] = true`',
        "`settings.STUDIES`",
        "`model/words.json`",
        "[ADR 0001](0001-model-selection-protocol.md)",
        "[ADR 0003](0003-api-methods-are-fixed-reference-models.md)",
        "[ADR 0004](0004-tagging-method-framework.md)",
        "[ADR 0007](0007-api-score-methods-share-the-llm-toolkit.md)",
        "[ADR 0008](0008-cli-default-method.md)",
    ):
        assert expected in decision
    for option in (
        "- **MLflow's `optimize_prompts`**",
        "- **Nested searches, one per dev fold**",
        "- **All of dev for the search**",
    ):
        assert option in options


def test_adr_0009_keeps_every_prompt_version_of_an_api_method_in_a_versioned_file():
    text = ADR_0009.read_text(encoding="utf-8")
    decision, rest = text.split("\n## Considered options\n")
    options = rest.split("\n## Consequences\n")[0]

    for expected in (
        "`templates/<name>.v<N>.yaml`",
        "`codeforces_tags.prompt_files`",
        "prompt registry",
        "[ADR 0003](0003-api-methods-are-fixed-reference-models.md)",
        "[ADR 0007](0007-api-score-methods-share-the-llm-toolkit.md)",
        "`{tag}`",
        "`prompts: []`",
        "`settings_sha256`",
        "`.j2`",
    ):
        assert expected in decision
    for option in ("Python literals", "JSON files", "One YAML file per prompt name", "MLflow"):
        assert f"- **{option}" in options


def test_adrs_0003_and_0007_say_that_the_words_live_in_versioned_prompt_files():
    for name in (
        "0003-api-methods-are-fixed-reference-models.md",
        "0007-api-score-methods-share-the-llm-toolkit.md",
    ):
        decision = (
            (ADR_DIR / name).read_text(encoding="utf-8").split("\n## Considered options\n")[0]
        )

        assert ADR_0009_LINK in decision


def test_the_agent_instructions_say_where_the_prompt_files_are_and_what_reads_them():
    text = CLAUDE.read_text(encoding="utf-8")
    entry = next(line for line in text.splitlines() if line.startswith("  - `prompt_files.py`: "))
    toolkit = text[text.index("\n### LLM toolkit\n") : text.index("\n### TypeSafe methods\n")]
    typesafe = text[text.index("\n### TypeSafe methods\n") : text.index("\n## Commands\n")]

    assert "pyyaml" in entry
    for section in (entry, toolkit, typesafe):
        assert "docs/adr/0009-versioned-prompt-files.md" in section


def test_the_glossary_defines_the_terms_of_the_tagging_method_framework():
    text = GLOSSARY.read_text(encoding="utf-8")
    defined = re.findall(r"^\*\*(.+?)\*\*:$", text, flags=re.MULTILINE)

    assert set(FRAMEWORK_TERMS) <= set(defined)


def test_the_glossary_defines_the_terms_of_the_embedding_methods():
    text = GLOSSARY.read_text(encoding="utf-8")
    defined = re.findall(r"^\*\*(.+?)\*\*:$", text, flags=re.MULTILINE)

    assert set(EMBEDDING_TERMS) <= set(defined)


def test_the_glossary_defines_the_terms_of_the_search():
    text = GLOSSARY.read_text(encoding="utf-8")
    defined = re.findall(r"^\*\*(.+?)\*\*:$", text, flags=re.MULTILINE)

    assert set(SEARCH_TERMS) <= set(defined)


def test_the_glossary_keeps_the_folds_of_a_search_apart_from_the_train_validation_and_test_sets():
    text = GLOSSARY.read_text(encoding="utf-8")
    avoided = dict(
        re.findall(r"^\*\*(.+?)\*\*:\n[^\n]+\n_Avoid_: ([^\n]+)$", text, flags=re.MULTILINE)
    )

    assert "train set" in avoided["Learning folds"]
    assert "validation set" in avoided["Selection fold"]
    assert "test set" in avoided["Held-out fold"]


def test_the_feature_block_names_the_embedding_blocks():
    text = GLOSSARY.read_text(encoding="utf-8")
    definition = re.search(r"^\*\*Feature block\*\*:\n([^\n]+)\n", text, flags=re.MULTILINE)

    assert "embedding blocks" in definition.group(1)


def test_every_term_of_the_glossary_has_a_definition_and_words_to_avoid():
    text = GLOSSARY.read_text(encoding="utf-8")
    terms = re.findall(r"^\*\*(.+?)\*\*:$", text, flags=re.MULTILINE)
    complete = re.findall(r"^\*\*(.+?)\*\*:\n[^\n]+\n_Avoid_: [^\n]+$", text, flags=re.MULTILINE)

    assert complete == terms


def test_the_glossary_keeps_a_calibrated_probability_apart_from_a_cross_fitted_threshold():
    text = GLOSSARY.read_text(encoding="utf-8")
    entry = re.search(
        r"^\*\*Calibrated probability\*\*:\n[^\n]+\n_Avoid_: ([^\n]+)$", text, flags=re.MULTILINE
    )

    assert entry
    assert "calibrated threshold" in entry.group(1)


def test_every_relative_link_of_the_agent_instructions_the_glossary_and_the_adrs_resolves():
    documents = [CLAUDE, AGENTS, GLOSSARY, *ADRS]
    broken = [
        f"{document.name} links to {target}"
        for document in documents
        for target in RELATIVE_LINK.findall(document.read_text(encoding="utf-8"))
        if not (document.parent / target).exists()
    ]

    assert not broken


def test_agents_md_is_claude_md_with_its_own_title_line():
    claude = CLAUDE.read_text(encoding="utf-8").splitlines()
    agents = AGENTS.read_text(encoding="utf-8").splitlines()

    assert (claude[0], agents[0]) == ("# CLAUDE.md", "# AGENTS.md")
    assert agents[1:] == claude[1:]


def test_the_agent_instructions_say_how_to_add_a_tagging_method_and_test_its_dependency_group():
    text = CLAUDE.read_text(encoding="utf-8")

    for heading in (
        "## Tagging methods",
        "### Adding a tagging method",
        "### Tests and CI for a method's dependency group",
    ):
        assert f"\n{heading}\n" in text


def test_the_agent_instructions_describe_the_llm_toolkit():
    text = CLAUDE.read_text(encoding="utf-8")
    lines = text.splitlines()
    tuning = next(line for line in lines if line.startswith("  - `tuning/`: "))
    heavy_imports = next(line for line in lines if line.startswith("- **Heavy imports "))

    assert "\n### LLM toolkit\n" in text
    assert "needs no `__main__.py`" in text  # checklist step 1, for a prompted method
    assert "added with the toolkit" not in text  # checklist step 2 named a subsection to come
    assert "`llm/__main__.py`" in tuning
    assert "`llm/__main__.py`" in heavy_imports


def test_the_agent_instructions_describe_the_typesafe_methods():
    text = CLAUDE.read_text(encoding="utf-8")
    heavy_imports = next(
        line for line in text.splitlines() if line.startswith("- **Heavy imports ")
    )
    toolkit = text.index("\n### LLM toolkit\n")
    typesafe = text.index("\n### TypeSafe methods\n")
    commands = text.index("\n## Commands\n")
    section = text[typesafe:commands]

    assert toolkit < typesafe < commands
    assert "\n### " not in text[toolkit + 1 : typesafe]  # right after the toolkit's subsection
    for expected in (
        "docs/adr/0007-api-score-methods-share-the-llm-toolkit.md",
        "--env-file",
        "--group llm-typesafe",
        "scores_from_answers",
    ):
        assert expected in section
    assert "typesafe_sdk" in heavy_imports


def test_the_agent_instructions_link_every_adr():
    text = CLAUDE.read_text(encoding="utf-8")

    unlinked = [adr.name for adr in ADRS if f"docs/adr/{adr.name}" not in text]

    assert not unlinked


def test_the_layout_names_every_module_of_the_f2llm_package_and_the_commands_run_it():
    text = CLAUDE.read_text(encoding="utf-8")
    bullet = next(line for line in text.splitlines() if line.startswith("    - `f2llm/`:"))
    names = [path.name for path in F2LLM_PACKAGE.glob("*.py") if path.name != "__init__.py"]

    assert [name for name in names if f"`{name}`" not in bullet] == []
    assert "`assets/f2llm/`" in text
    assert "uv sync --group f2llm" in text


def test_the_layout_names_every_module_and_package_directly_under_the_package():
    text = CLAUDE.read_text(encoding="utf-8")
    names = [
        f"`{path.name}/`" if path.is_dir() else f"`{path.name}`"
        for path in PACKAGE.iterdir()
        if path.name not in {"__init__.py", "__pycache__"}
    ]

    unnamed = [name for name in names if name not in text]

    assert not unnamed


def test_the_agent_instructions_name_every_module_that_imports_optuna():
    text = CLAUDE.read_text(encoding="utf-8")
    bullet = next(line for line in text.splitlines() if line.startswith("  - `tuning/`:"))
    clause = bullet[bullet.index("; only ") : bullet.index(" import optuna")]
    importers = [
        path
        for path in PACKAGE.rglob("*.py")
        if OPTUNA_IMPORT.search(path.read_text(encoding="utf-8"))
    ]

    # A module of tuning/ is named by its folder, a module of a method by its file name.
    unnamed = [
        str(path.relative_to(PACKAGE))
        for path in importers
        if f"`{path.parent.name}/`" not in clause and f"`{path.name}`" not in clause
    ]

    assert PACKAGE / "tuning" / "search.py" in importers  # the scan sees the study's imports
    assert not unnamed


def test_the_layout_names_every_module_of_each_package_in_the_entry_of_the_package():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    unnamed = []
    for package in sorted(path for path in PACKAGE.iterdir() if (path / "__init__.py").is_file()):
        start = next(
            i for i, line in enumerate(lines) if line.startswith(f"  - `{package.name}/`: ")
        )
        sub_bullets = itertools.takewhile(
            lambda line: line.startswith("    - "), lines[start + 1 :]
        )
        entry = "\n".join([lines[start], *sub_bullets])
        command = f"python -m codeforces_tags.{package.name}"
        unnamed += [
            f"{package.name}/{module.name}"
            for module in sorted(package.glob("*.py"))
            if module.name != "__init__.py"
            and f"`{module.name}`" not in entry
            and not (module.name == "__main__.py" and command in entry)
        ]

    assert not unnamed


def test_the_analysis_entry_of_the_layout_says_that_the_tracking_commands_log_its_figures():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    analysis = next(line for line in lines if line.startswith("  - `analysis/`: "))

    assert "code the notebooks use, and the figures the tracking commands log" in analysis
    assert "code only the notebooks use" not in analysis
    assert "reliability diagrams" in analysis


def test_the_frozen_import_path_bullet_names_every_type_that_a_saved_tag_model_stores():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    bullet = next(line for line in lines if line.startswith("- **Frozen import path**"))

    unnamed = [path for path in STORED_TYPES if f"`{path.rsplit('.', 1)[-1]}`" not in bullet]

    assert "`STORED_TYPES`" in bullet
    assert not unnamed


def test_the_commands_and_the_readme_run_the_three_commands_of_the_command_line():
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]
    readme = README.read_text(encoding="utf-8")

    for document in (commands, readme):
        assert all(f"uv run {command}" in document for command in CLI_COMMANDS)
        assert "--method tfidf" in document
        assert "docs/adr/0008-cli-default-method.md" in document


def test_the_agent_instructions_say_how_the_command_line_chooses_and_imports():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    requirement = next(line for line in lines if line.startswith("- **CLI**: "))
    selection = next(line for line in lines if line.startswith("- **Model selection**: "))
    heavy_imports = next(line for line in lines if line.startswith("- **Heavy imports "))

    assert "`codeforces-tags`" in requirement
    assert "docs/adr/0008-cli-default-method.md" in selection
    assert "the CLI's method is chosen on dev out-of-fold focus macro F1 alone" not in selection
    assert "`cli.py`" in heavy_imports
    assert "tests/test_cli.py" in heavy_imports


def command_positions(text: str) -> list[int]:
    """Find the first place of each command of ``STUDY_COMMANDS`` in a text.

    Args:
        text: The text of a document, or of a section of one.

    Returns:
        One position per command, in the order of ``STUDY_COMMANDS``; -1 for a command that the
        text does not run as ``uv run python -m codeforces_tags.<command>``.
    """
    return [text.find(f"uv run python -m codeforces_tags.{step}") for step in STUDY_COMMANDS]


def test_the_commands_tune_freeze_verify_time_and_calibrate_the_tfidf_method_in_this_order():
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]

    positions = command_positions(commands)

    assert -1 not in positions
    assert positions == sorted(positions)


def test_the_readme_of_the_tuning_study_gives_the_same_commands_in_the_same_order():
    positions = command_positions(STUDY_README.read_text(encoding="utf-8"))

    assert -1 not in positions
    assert positions == sorted(positions)


def package_docstring(package: str) -> str:
    """Read the docstring of a package's ``__init__.py`` without importing the package.

    Args:
        package: The path of a package under ``codeforces_tags``, e.g. ``methods/typesafe_jev``.

    Returns:
        Its docstring, with the indentation removed.
    """
    source = (PACKAGE / package / "__init__.py").read_text(encoding="utf-8")
    return ast.get_docstring(ast.parse(source)) or ""


@pytest.mark.parametrize("package", ["analysis", "evaluation", "tracking", "methods/typesafe_jev"])
def test_the_docstring_of_a_package_that_lists_its_modules_names_each_of_them(package):
    docstring = package_docstring(package)
    modules = sorted(
        path.stem for path in (PACKAGE / package).glob("*.py") if path.name != "__init__.py"
    )

    unnamed = [module for module in modules if f"``{module}``" not in docstring]

    assert not unnamed


def test_the_analysis_docstring_says_that_the_tracking_package_draws_figures_too():
    docstring = " ".join(package_docstring("analysis").split())  # whatever the line wrapping

    assert "``tracking/``" in docstring
    assert "the only package meant to import matplotlib" not in docstring


def test_the_layout_names_every_method_package_in_the_methods_entry():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    entry = next(line for line in lines if line.startswith("  - `methods/`: "))
    packages = [
        path.name for path in (PACKAGE / "methods").iterdir() if (path / "__init__.py").is_file()
    ]

    unnamed = [name for name in packages if f"`{name}/`" not in entry]

    assert {"label_prior", "zero_shot"} <= set(packages)
    assert not unnamed


def test_the_assets_entry_names_every_committed_method_folder():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    entry = next(line for line in lines if line.startswith("- `assets/`: "))
    folders = [path.name for path in METHODS_ASSETS_DIR.iterdir() if path.is_dir()]

    unnamed = [name for name in folders if f"`assets/methods/{name}/`" not in entry]

    assert {"label_prior", "tfidf", "descode2tag-json"} <= set(folders)
    assert not unnamed


def test_the_commands_render_run_score_verify_and_time_the_zero_shot_methods_in_this_order():
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]
    llm = "uv run --group llm python -m codeforces_tags"
    method = "--method descode2tag-json"
    steps = [
        f"{llm}.llm render {method}",
        f"{llm}.llm run {method} --part dev",
        f"{llm}.llm run {method} --part test",
        f"{llm}.llm score {method}",
        f"{llm}.llm verify {method}",
        f"{llm}.tracking latency {method}",
        "--inplace notebooks/05_zero_shot_llm.ipynb",
    ]

    positions = [commands.find(step) for step in steps]

    assert -1 not in positions
    assert positions == sorted(positions)


def test_the_checklist_lists_the_zero_shot_exception_to_step_8_and_numbers_notebooks_at_merge():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    step_8 = next(line for line in lines if line.startswith("8. **MLflow and latency.**"))
    step_9 = next(line for line in lines if line.startswith("9. **Notebook.**"))

    assert "zero-shot" in step_8
    assert "next free number at merge time" in step_9
    assert "from 06" not in step_9


def test_the_comment_of_the_figures_group_names_every_render_tool():
    before = PYPROJECT.read_text(encoding="utf-8").split("\nfigures = [")[0].splitlines()
    comment = " ".join(itertools.takewhile(lambda line: line.startswith("#"), reversed(before)))
    tools = sorted(path.name for path in (PROJECT_ROOT / "notebooks" / "figures").glob("*.py"))

    unnamed = [tool for tool in tools if tool not in comment]

    assert tools == ["estimator_diagram.py", "mlflow_ui.py", "streamlit_app.py"]
    assert not unnamed


def test_the_layout_names_the_settings_the_render_tools_and_the_tests_of_the_mlflow_screenshots():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    config = next(line for line in lines if line.startswith("  - `config.py`: "))
    tests = next(line for line in lines if line.startswith("- `tests/`: "))
    notebooks = next(line for line in lines if line.startswith("- `notebooks/`: "))
    tools = sorted(path.name for path in (PROJECT_ROOT / "notebooks" / "figures").glob("*.py"))

    unnamed = [tool for tool in tools if f"`{tool}`" not in notebooks]

    assert "MLflow UI screenshots" in config
    assert "`tests/figures/`" in tests
    assert "`figures/mlflow/`" in notebooks and "`figures/mlflow_docs/`" in notebooks
    assert not unnamed


def test_step_8_of_the_checklist_says_how_the_zero_shot_evaluation_runs_join_a_store():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    step_8 = next(line for line in lines if line.startswith("8. **MLflow and latency.**"))

    assert "later pull request" not in step_8
    assert "with no paid call" in step_8
    assert "before the store's first test run" in step_8


def test_the_evaluated_methods_are_the_two_references_and_the_zero_shot_methods_compared_last():
    zero_shot = sorted(name for name, spec in METHODS.items() if spec.kind == "zero_shot")

    assert EVALUATED[:2] == ("label_prior", "tfidf")
    assert sorted(EVALUATED[2:]) == zero_shot
    assert EVALUATED[-1] == MLFLOW_SCREEN_COMPARE


def test_the_notebooks_readme_rebuilds_the_store_dev_first_then_takes_the_mlflow_screenshots():
    readme = NOTEBOOKS_README.read_text(encoding="utf-8")
    steps = readme.split("\n### The MLflow screenshots\n")[1].split("\n## ")[0]
    loop = f"for method in {' '.join(EVALUATED)}; do"
    order = [
        "tracking datasets",
        "tracking evaluate --part dev",
        "sleep 120",
        "tracking evaluate --part test",
        "unshare -rn",
        "notebooks/figures/mlflow_ui.py",
        "--inplace notebooks/05_zero_shot_llm.ipynb",
    ]

    positions = [steps.find(step) for step in order]

    assert steps.count(loop) == 2
    assert -1 not in positions
    assert positions == sorted(positions)
    assert f"127.0.0.1:{MLFLOW_UI_PORT}" in steps


def test_the_layout_names_every_module_of_the_typesafe_jev_package():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    entry = next(line for line in lines if line.startswith("  - `methods/`: "))
    sentence = entry[entry.index("`typesafe_jev/` holds") : entry.index("`f2llm/` holds")]
    names = sorted(path.name for path in JEV_PACKAGE.glob("*.py") if path.name != "__init__.py")

    unnamed = [name for name in names if f"`{name}`" not in sentence]

    assert "search_commands.py" in names  # the scan sees the modules of the search
    assert "`templates/gepa/`" in sentence
    assert f"`{SEARCHED}`" in sentence
    assert not unnamed


def test_the_agent_instructions_describe_the_searched_jev_method():
    text = CLAUDE.read_text(encoding="utf-8")
    typesafe = text.index("\n### TypeSafe methods\n")
    searched = text.index("\n### The searched Jev method\n")
    commands = text.index("\n## Commands\n")
    section = text[searched:commands]

    assert typesafe < searched < commands
    for expected in (
        "docs/adr/0010-searched-jev-words.md",
        f"`{SEARCHED}`",
        "`dev_in_sample`",
        "learning folds",
        "selection fold",
        "held-out fold",
        "balanced Brier",
        "`tuning/params.json`",
        "`model/words.json`",
        f"`experiments/{SEARCHED}/search/`",
        "`settings.STUDIES`",
        "`jev-gepa`",
        "`create_prompt_version`",
    ):
        assert expected in section


def test_the_commands_search_answer_freeze_and_time_the_searched_jev_method_in_this_order():
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]
    section = commands.split("\n### The GEPA search over Jev's words\n")[1]

    positions = [section.find(f"python -m codeforces_tags.{step}") for step in SEARCH_COMMANDS[:-1]]
    positions.append(section.find(SEARCH_COMMANDS[-1]))

    assert -1 not in positions
    assert positions == sorted(positions)
    assert "--group llm --group llm-typesafe" in section


def test_the_agent_instructions_keep_gepa_off_the_prediction_path():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    heavy_imports = next(line for line in lines if line.startswith("- **Heavy imports "))
    tuning = next(line for line in lines if line.startswith("  - `tuning/`: "))

    assert "optuna, mlflow, gepa," in heavy_imports
    assert "`methods/typesafe_jev/search_commands.py`" in heavy_imports
    assert "never imports mlflow, optuna or gepa" in heavy_imports
    assert "import gepa" in tuning
    assert "`search_commands.py`" in tuning


def test_the_layout_names_the_files_of_the_search_and_its_screenshots():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    assets = next(line for line in lines if line.startswith("- `assets/`: "))
    experiments = next(line for line in lines if line.startswith("- `experiments/`: "))
    notebooks = next(line for line in lines if line.startswith("- `notebooks/`: "))
    analysis = next(line for line in lines if line.startswith("  - `analysis/`: "))

    assert f"`assets/methods/{SEARCHED}/`" in assets
    assert f"`assets/methods/{SEARCHED}/tuning/`" in assets
    assert f"`{SEARCHED}/search/`" in experiments
    assert "`figures/mlflow/gepa/`" in notebooks
    assert "`--set gepa`" in notebooks
    assert "`prompt_search.py`" in analysis
    assert "`plot_search_progress()`" in analysis


def test_the_notebooks_readme_runs_notebook_10_and_takes_the_search_screenshots():
    readme = NOTEBOOKS_README.read_text(encoding="utf-8")
    table, running = readme.split("\n## Running\n")
    running = running.split("\n## Figures\n")[0]
    shots = readme.split("\n### The search's MLflow screenshots\n")[1].split("\n## ")[0]
    order = [
        f"log-search --method {SEARCHED}",
        "unshare -rn",
        "notebooks/figures/mlflow_ui.py --set gepa",
        "--inplace notebooks/10_jev_gepa.ipynb",
    ]

    positions = [shots.find(step) for step in order]

    assert "| [10_jev_gepa.ipynb](10_jev_gepa.ipynb) |" in table
    assert "--inplace notebooks/10_jev_gepa.ipynb" in running
    assert "`figures/mlflow/gepa/*.png`" in shots
    assert -1 not in positions
    assert positions == sorted(positions)
    assert readme.index("\n### The MLflow screenshots\n") < readme.index(
        "\n### The search's MLflow screenshots\n"
    )
