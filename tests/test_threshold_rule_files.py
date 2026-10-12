"""Tests for the files that record the threshold rule and the calibrated Jev twins (ADR 0011)."""

import ast
import inspect
import re

from codeforces_tags import config
from codeforces_tags.config import PROJECT_ROOT
from codeforces_tags.model import platt_lipton

ADR_DIR = PROJECT_ROOT / "docs" / "adr"
ADR_0011 = ADR_DIR / "0011-platt-lipton-threshold-rule.md"
GLOSSARY = PROJECT_ROOT / "CONTEXT.md"
CLAUDE = PROJECT_ROOT / "CLAUDE.md"
NOTEBOOKS_README = PROJECT_ROOT / "notebooks" / "README.md"
JEV_PACKAGE = PROJECT_ROOT / "src" / "codeforces_tags" / "methods" / "typesafe_jev"
# The calibrated Jev twins, the commands that export, freeze, check, time and evaluate them in
# the order they run, and the command that runs notebook 11.
TWINS = ("jev-statement-code-platt", "jev-statement-code-gepa-platt")
TWIN_COMMANDS = [
    'methods.typesafe_jev export --method "$twin"',
    'methods.typesafe_jev freeze --method "$twin"',
    'methods.typesafe_jev verify --method "$twin"',
    'tracking latency --method "$twin"',
    'tracking evaluate --method "$twin" --part dev',
    'tracking evaluate --method "$twin" --part test',
]
TWIN_NOTEBOOK = "--inplace notebooks/11_jev_platt.ipynb"
# The ADRs that ADR 0011 builds on and links.
LINKED_ADRS = (
    "0001-model-selection-protocol",
    "0003-api-methods-are-fixed-reference-models",
    "0004-tagging-method-framework",
    "0007-api-score-methods-share-the-llm-toolkit",
    "0008-cli-default-method",
    "0010-searched-jev-words",
)
# The entries of RULE_SETTINGS that hold a project-wide setting, and its name in config.py.
CONFIG_SETTINGS = {
    "clip": "PLATT_CLIP",
    "solver": "PLATT_SOLVER",
    "tol": "PLATT_TOL",
    "grid": "SCORE_GRID",
}


def _listed_as(key: str, value: object) -> str:
    """The text by which ADR 0011 lists one entry of ``RULE_SETTINGS``."""
    if key in CONFIG_SETTINGS:
        return f"`config.{CONFIG_SETTINGS[key]} = "
    if key == "C":
        return f"`C={value}`"
    return f'`"{value}"`'


def test_adr_0011_fixes_the_threshold_rule_before_any_number_of_the_twins():
    text = ADR_0011.read_text(encoding="utf-8")
    decision, options = text.split("\n## Considered options\n")
    options = options.split("\n## Consequences\n")[0]
    title = "# A threshold rule that reads calibrated probabilities, and methods proposed after"

    assert text.startswith(f"{title} a test score\n")
    for expected in (
        "**The threshold rule R2 is fixed here, before any new number.**",
        "half the best F1 (Lipton et al. 2014)",
        "`model.platt_lipton.rule_calibrator()`",
        "`RULE_SETTINGS`",
        "`PlattLiptonThresholds`",
        "**Two calibrated twins, composites over their bases' answers.**",
        "`jev-statement-code-platt`",
        "`jev-statement-code-gepa-platt`",
        "`depends_on`",
        "`per_problem.csv` are the base's, copied byte for byte",
        "**The comparison is declared, and dev only.**",
        "the nested choice is reported, never acted on",
        "**The twins are `test_informed`.**",
        'method_entries["test_informed"]',
        "`cli_methods._eligible`",
        "**`dev_in_sample` follows the base.**",
        "`dev_in_sample: false`",
        "`dev_in_sample: true`",
    ):
        assert expected in decision
    for stem in LINKED_ADRS:
        assert f"[ADR {stem[:4]}]({stem}.md)" in decision
    for option in (
        "- **The naive per-fold Platt scorer**",
        "- **A smoothed F1 curve**",
        "- **Platt and a fixed 0.5**",
        "- **Fresh paid answers for the twins**",
        "- **`dev_in_sample` for `jev-statement-code-platt`**",
    ):
        assert option in options


def test_adr_0011_quotes_the_dev_evidence_of_the_rule():
    text = ADR_0011.read_text(encoding="utf-8")
    table = text.split("The dev evidence, out of fold, as the committed records hold it")[1]
    table = table.split(":\n\n", 1)[1]
    table = table.split("\n\n")[0]

    for expected in (
        "| `jev-statement-code` | 0.729 | 0.722 | **0.738** (+0.009 [+0.001, +0.017]) |",
        "| `jev-statement-code-gepa` | 0.730 | 0.721 | 0.737 (+0.006 [−0.002, +0.015]) |",
    ):
        assert expected in table
    assert "68% of resamples" in text
    assert "13.5%" in text
    assert "against 41 under R0" in text


def test_adr_0011_lists_every_setting_of_the_rule_as_the_code_holds_it():
    # Regression: the ADR's list of RULE_SETTINGS left out the solver and its tolerance, which every
    # twin's recipe and jev.json hash hold, so changing either looked harmless to its reader.
    text = ADR_0011.read_text(encoding="utf-8")
    opening = "The rule's settings are `RULE_SETTINGS` in `model/platt_lipton.py` ("
    listed = text.split(opening)[1].split("); they are in each twin's recipe")[0]
    quoted = dict(re.findall(r"`config\.(\w+) = ([^`]+)`", listed))
    rule = platt_lipton.RULE_SETTINGS

    missing = [key for key, value in rule.items() if _listed_as(key, value) not in listed]

    assert {key: rule[key] for key in CONFIG_SETTINGS} == {
        key: getattr(config, name) for key, name in CONFIG_SETTINGS.items()
    }
    assert not missing
    assert {name: ast.literal_eval(value) for name, value in quoted.items()} == {
        name: getattr(config, name) for name in CONFIG_SETTINGS.values()
    }
    assert "The clip, the grid, the solver and its tolerance live in `config.py`" in text


def test_the_documents_tell_the_curves_peaks_and_what_was_chosen_on_dev_as_they_are():
    # Regression: the dev F1 curve of jev-statement-code-gepa's probabilities has four runs of cuts
    # near its maximum, not two peaks; the nested choice measures only the choice among R0, R1
    # and R2, which were declared after the curves and a test score were read.
    adr = ADR_0011.read_text(encoding="utf-8")
    instructions = CLAUDE.read_text(encoding="utf-8")
    readme = NOTEBOOKS_README.read_text(encoding="utf-8")

    for run in ("0.125-0.145 (F1 0.7602)", "0.315-0.325 (0.7600)", "0.405-0.435", "0.735-0.815"):
        assert run in adr
    assert "measures only the choice among the three declared rules" in adr
    assert "one of three declared in advance" not in adr
    assert "It is fixed in ADR 0011 before any new number" in instructions
    for document in (adr, instructions, readme):
        assert "two peaks" not in document
        assert "two almost equal" not in document


def test_adr_0004_makes_a_test_informed_method_ineligible_like_a_dev_in_sample_one():
    text = (ADR_DIR / "0004-tagging-method-framework.md").read_text(encoding="utf-8")
    link = "[ADR 0011](0011-platt-lipton-threshold-rule.md)"
    bullets = {line.split("**")[1]: line for line in text.splitlines() if line.startswith("- **")}
    choosing = bullets["Choosing among frozen methods."]
    in_sample = bullets["Dev scores read in sample."]

    assert "neither `dev_in_sample` nor `test_informed`" in choosing
    assert link in choosing
    assert 'method_entries["test_informed"]' in in_sample
    assert link in in_sample


def test_the_glossary_defines_the_threshold_rule_apart_from_a_calibrated_threshold():
    text = GLOSSARY.read_text(encoding="utf-8")
    entry = re.search(
        r"^\*\*Threshold rule\*\*:\n([^\n]+)\n_Avoid_: ([^\n]+)$", text, flags=re.MULTILINE
    )

    assert entry
    assert "Platt" in entry.group(1)
    assert "Lipton" in entry.group(1)
    assert "cross-fitted" in entry.group(1)
    assert "calibrated threshold" in entry.group(2)


def test_the_layout_names_the_threshold_rule_the_twins_and_their_folders():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    model = next(line for line in lines if line.startswith("  - `model/`: "))
    methods = next(line for line in lines if line.startswith("  - `methods/`: "))
    assets = next(line for line in lines if line.startswith("- `assets/`: "))
    cli_methods = next(line for line in lines if line.startswith("  - `cli_methods.py`: "))
    tests = next(line for line in lines if line.startswith("- `tests/`: "))
    sentence = methods[methods.index("`typesafe_jev/` holds") : methods.index("`f2llm/` holds")]

    for name in ("`platt_lipton.py`", "`PlattLiptonThresholds`", "is never edited"):
        assert name in model
    assert "docs/adr/0011-platt-lipton-threshold-rule.md" in model
    for twin in TWINS:
        assert f"`{twin}`" in sentence
        assert f"`assets/methods/{twin}/`" in assets
    for name in ("`platt.py`", "`platt_commands.py`", "`base_of()`", "`TEST_INFORMED`"):
        assert name in sentence
    twin_record = assets[assets.index("`assets/methods/jev-statement-code-platt/tuning/`") :]
    for name in ("`params.json`", "`rules.csv`", "`nested.csv`", "`stability.csv`"):
        assert name in twin_record
    assert "`test_informed`" in cli_methods
    assert "`latency.csv` and `per_problem.csv` are copied from its base" in cli_methods
    assert "logs that copy without timing anything" in cli_methods
    assert "`test_threshold_rule_files.py`" in tests


def test_the_layout_names_every_public_function_of_the_threshold_rule_module():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    model = next(line for line in lines if line.startswith("  - `model/`: "))
    entry = model[model.index("`platt_lipton.py`") : model.index("`training.py`")]
    public = [
        name
        for name, value in vars(platt_lipton).items()
        if inspect.isfunction(value)
        and value.__module__ == platt_lipton.__name__
        and not name.startswith("_")
    ]

    missing = [name for name in public if f"`{name}()`" not in entry]

    assert {"lipton_thresholds", "rule_calibrator"} <= set(public)
    assert not missing
    for setting in ("PLATT_CLIP", "SCORE_GRID", "PLATT_SOLVER", "PLATT_TOL"):
        assert f"`config.{setting}`" in entry


def test_the_docstring_of_the_model_package_names_the_threshold_rule_module():
    source = (PROJECT_ROOT / "src" / "codeforces_tags" / "model" / "__init__.py").read_text(
        encoding="utf-8"
    )
    docstring = source.split('"""')[1]

    assert "``platt_lipton``" in docstring
    assert (
        "``CrossFittedTagThresholds`` is never edited, and ``PlattLiptonThresholds`` subclasses it."
        in docstring
    )


def test_the_typesafe_jev_sentence_of_the_layout_names_only_modules_that_exist():
    lines = CLAUDE.read_text(encoding="utf-8").splitlines()
    entry = next(line for line in lines if line.startswith("  - `methods/`: "))
    sentence = entry[entry.index("`typesafe_jev/` holds") : entry.index("`f2llm/` holds")]

    named = re.findall(r"`([a-z_]+\.py)`", sentence)
    missing = [name for name in named if not (JEV_PACKAGE / name).is_file()]

    assert {"platt.py", "platt_commands.py", "__main__.py"} <= set(named)
    assert not missing


def test_the_agent_instructions_describe_the_calibrated_jev_twins():
    text = CLAUDE.read_text(encoding="utf-8")
    searched = text.index("\n### The searched Jev method\n")
    twins = text.index("\n### The calibrated Jev twins\n")
    commands = text.index("\n## Commands\n")
    section = text[twins:commands]

    assert searched < twins < commands
    for expected in (
        "docs/adr/0011-platt-lipton-threshold-rule.md",
        *(f"`{twin}`" for twin in TWINS),
        "`PlattLiptonThresholds`",
        "`MethodSpec.depends_on`",
        "`settings.PLATT_BASES`",
        "`settings.TEST_INFORMED`",
        "`test_informed`",
        "`dev_in_sample`",
        "`tuning/params.json`",
        "`rules.csv`",
        "`nested.csv`",
        "`stability.csv`",
        "never reads a test row",
        "`latency.csv`",
        "`per_problem.csv`",
        "byte for byte",
        "without timing anything",
        "`notebooks/11_jev_platt.ipynb`",
    ):
        assert expected in section


def test_the_agent_instructions_say_that_a_test_informed_method_is_never_eligible():
    text = CLAUDE.read_text(encoding="utf-8")
    checklist = text.split("### Adding a tagging method\n")[1].split("\n### ")[0]
    gotchas = text.split("\n## Data gotchas\n")[1]

    assert 'also records `method_entries["test_informed"]`' in checklist
    assert "reads its base's committed files instead and sends no request" in checklist
    assert "`test_informed`" in gotchas
    assert "never eligible and their test scores are references" in gotchas


def test_the_commands_export_freeze_verify_time_and_evaluate_the_calibrated_twins_in_this_order():
    commands = CLAUDE.read_text(encoding="utf-8").split("\n## Commands\n")[1].split("\n## ")[0]
    section = commands.split("\n### Freezing the calibrated Jev twins\n")[1].split("\n### ")[0]
    block = section.split("```bash\n")[1].split("```")[0]

    positions = [block.find(f"python -m codeforces_tags.{step}") for step in TWIN_COMMANDS]
    positions.append(block.find(TWIN_NOTEBOOK))

    assert -1 not in positions
    assert positions == sorted(positions)
    assert "--env-file" not in block  # no key is read
    assert "--max-cost-usd" not in block  # no request is sent
    assert "for twin in jev-statement-code-platt jev-statement-code-gepa-platt; do" in block


def test_the_notebooks_readme_lists_and_runs_notebook_11_and_names_its_helpers():
    readme = NOTEBOOKS_README.read_text(encoding="utf-8")
    table, running = readme.split("\n## Running\n")
    running = running.split("\n## Figures\n")[0]
    row = next(line for line in table.splitlines() if "11_jev_platt.ipynb" in line)
    adr = "../docs/adr/0011-platt-lipton-threshold-rule.md"

    assert row.startswith("| ")
    assert f"({adr})" in row
    assert (NOTEBOOKS_README.parent / adr).is_file()
    assert table.index("10_jev_gepa.ipynb") < table.index("11_jev_platt.ipynb")
    assert TWIN_NOTEBOOK in running
    assert "`analysis/threshold_rules.py`" in readme
    assert "`analysis/calibrators.py`" in readme
