"""The tagging methods as the command line sees them: whether each runs here, and their listing.

``resolve_method`` checks a method before anything of it is imported or loaded: it must be
registered and frozen, its dependency groups installed, the weights its folder names in their
caches (a ModernBERT or F2LLM method, docs/adr/0006; the message names the fetch command of the
method's own package, ``package_command``) and, for a method that calls an API, its key set.
When one is not, its ``CliError`` names every unmet need and the offline method
(``--method tfidf``): the command line never falls back on another method by itself.
``load_cli_method`` then loads the frozen method folder. ``method_table`` is the ``methods``
listing: each method's frozen dev (out of fold) and test macro F1, its latency (and what one
prediction by a fresh command takes: loading, then the slowest problem), whether ADR 0004 makes
it eligible and names it best on dev, whether it is the default (docs/adr/0008), what it needs
and whether it runs here.

``CliError`` and ``PROG`` belong to the whole command line (``codeforces_tags.cli``), which
reports a ``CliError`` on one line of stderr, with status 1.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from codeforces_tags.config import (
    API_KEY_ENV_VARS,
    CLI_DEFAULT_METHOD,
    CLI_OFFLINE_METHOD,
    LATENCY_BUDGET_S,
    METHODS_ASSETS_DIR,
    PYPROJECT_FILE,
)
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    load_predictions,
    load_run_record,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.evaluation.timing import LATENCY_FILE, latency_summary, load_latency
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    MethodSpec,
    TaggingMethod,
    load_method,
    missing_groups,
)
from codeforces_tags.methods.modernbert.weights import missing_caches, weight_caches
from codeforces_tags.splitting import PARTS

PROG = "codeforces-tags"
# The columns of the methods listing, in order (method_table).
METHOD_TABLE_COLUMNS = (
    "kind",
    "dev_macro_f1",
    "test_macro_f1",
    "median_s",
    "max_s",
    "load_s",
    "cold_s",
    "eligible",
    "best_dev",
    "default",
    "needs",
    "runs_here",
)
# How the listing prints each column of floats; NaN is printed "-".
_LISTED_FORMATS = {
    "dev_macro_f1": "{:.4f}",
    "test_macro_f1": "{:.4f}",
    "median_s": "{:.3f}",
    "max_s": "{:.3f}",
    "load_s": "{:.3f}",
    "cold_s": "{:.3f}",
}


class CliError(Exception):
    """A failure the CLI reports as one line on stderr ("codeforces-tags: error: ..."), status 1."""


def offline_hint(name: str) -> str:
    """Name the offline method at the end of a failure, unless it is the method that failed.

    Args:
        name: The method that cannot run.

    Returns:
        ``" (--method tfidf runs offline, for free)"`` for any method but
        ``CLI_OFFLINE_METHOD``, and ``""`` for it.
    """
    if name == CLI_OFFLINE_METHOD:
        return ""
    return f" (--method {CLI_OFFLINE_METHOD} runs offline, for free)"


def resolve_method(
    name: str,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    assets_dir: Path = METHODS_ASSETS_DIR,
    pyproject: Path = PYPROJECT_FILE,
) -> MethodSpec:
    """Check that a method can run here, before anything of it is imported or loaded.

    It needs to be registered and frozen (its method folder holds ``method.json``), its
    dependency groups installed, the weights its folder's weight lists name in their caches
    (``missing_caches``: present, their hashes checked by loading; missing pretrained weights
    are fetched by ``package_command(spec)``'s ``fetch``), and, for a method that calls an API,
    the key of each of its groups (``API_KEY_ENV_VARS``) set and not empty: only the key's
    presence is read.

    Args:
        name: The method's registry key, as ``--method`` gives it.
        registry: The registered methods.
        assets_dir: The folder of the frozen methods.
        pyproject: The ``pyproject.toml`` that defines the dependency groups.

    Returns:
        The method's spec.

    Raises:
        CliError: If the method is not registered, not frozen, or misses a dependency group,
            weights or a key; one message names every unmet need, with the command that brings
            the groups and keys, and the offline method.
        ValueError: If ``pyproject`` does not define a group of the method (``missing_groups``),
            or if the method folder's ``pretrained.json`` is in neither shape that
            ``pinned_files`` reads, or names a repository or revision that is not one folder
            name of the pretrained cache (``missing_caches``): a corrupt frozen file, reported
            as such rather than as a method that cannot run here.
    """
    if name not in registry:
        raise CliError(
            f"{name!r} is not a registered method: the registry has {list(registry)}"
            f"{offline_hint(name)}."
        )
    folder = assets_dir / name / MODEL_FOLDER
    if not (folder / METHOD_FILE).is_file():
        raise CliError(
            f"{name} is not frozen: {folder} holds no {METHOD_FILE}, so run its freeze first"
            f"{offline_hint(name)}."
        )
    spec = registry[name]
    groups = missing_groups(name, spec, pyproject=pyproject)
    keys = [key for key in api_keys(spec) if not os.environ.get(key)]
    caches = missing_caches(folder, name, command=package_command(spec))
    if groups or keys or caches:
        raise CliError(_unmet_needs_message(name, groups, keys, caches))
    return spec


def load_cli_method(
    name: str,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    assets_dir: Path = METHODS_ASSETS_DIR,
) -> TaggingMethod:
    """Load a frozen method's folder, once ``resolve_method`` has passed.

    Args:
        name: The method's registry key.
        registry: The registered methods.
        assets_dir: The folder of the frozen methods.

    Returns:
        The method, as ``load_method`` loads ``assets_dir / name / MODEL_FOLDER``.

    Raises:
        CliError: If loading it fails for want of a module or a file, or on a file it refuses
            (e.g. a ModernBERT cache whose weights differ from the frozen ones); the message
            names the error and the offline method.
    """
    folder = assets_dir / name / MODEL_FOLDER
    try:
        return load_method(folder, registry=registry)
    except (ImportError, OSError, ValueError) as error:
        raise CliError(
            f"{name} could not be loaded from {folder}: {type(error).__name__}: "
            f"{str(error).rstrip('.')}{offline_hint(name)}."
        ) from error


def api_keys(spec: MethodSpec) -> list[str]:
    """Name the API keys a method reads: one per group of a method that calls an API.

    Args:
        spec: The method's spec.

    Returns:
        The variable of ``API_KEY_ENV_VARS`` of each of its groups that has one, in its order;
        none for a method that calls no API.
    """
    if not spec.calls_api:
        return []
    return [API_KEY_ENV_VARS[group] for group in spec.groups if group in API_KEY_ENV_VARS]


def package_command(spec: MethodSpec) -> str:
    """Write the command that runs a method's own package, e.g. its ``fetch``.

    Args:
        spec: The method's spec.

    Returns:
        ``uv run --group <group> python -m <package>``, with one ``--group`` per group of the
        spec, in its order, and ``<package>`` the spec's module without its last component:
        ``uv run --group f2llm python -m codeforces_tags.methods.f2llm`` for an F2LLM method,
        ``uv run --group finetuning python -m codeforces_tags.methods.modernbert`` for a
        ModernBERT method.
    """
    groups = "".join(f" --group {group}" for group in spec.groups)
    package = spec.module.rpartition(".")[0]
    return f"uv run{groups} python -m {package}"


def method_table(
    labels: pd.DataFrame | None,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    assets_dir: Path = METHODS_ASSETS_DIR,
    pyproject: Path = PYPROJECT_FILE,
) -> pd.DataFrame:
    """Describe every registered method: its frozen scores, latency, eligibility and needs.

    Args:
        labels: The focus labels of the challenge dataset (``cli.challenge_labels``), or None:
            the F1 columns are then NaN.
        registry: The registered methods.
        assets_dir: The folder of the frozen methods.
        pyproject: The ``pyproject.toml`` that defines the dependency groups.

    Returns:
        One row per method, in registry order, indexed by ``method``, with the columns of
        ``METHOD_TABLE_COLUMNS``: ``kind``; ``dev_macro_f1`` and ``test_macro_f1``, the macro
        F1 of the frozen ``predictions.csv`` of each part (NaN without labels or predictions);
        ``median_s`` and ``max_s`` of its timed problems and ``load_s`` (NaN without
        ``latency.csv``); ``cold_s``, ``load_s + max_s``, the slowest prediction of one sample
        by a fresh command, which loads the method first (the interpreter's start and imports
        aside); ``eligible`` (ADR 0004: frozen, not ``dev_in_sample``, not ``test_informed``
        (docs/adr/0011), slowest problem under ``LATENCY_BUDGET_S``, once loaded);
        ``best_dev``, the eligible method with the best dev macro F1 (none without labels);
        ``default``, ``CLI_DEFAULT_METHOD``; ``needs``, its groups, weights and API keys, e.g.
        ``"group llm-typesafe, TYPESAFE_API_KEY"``; ``runs_here``, whether ``resolve_method``
        passes.

    Raises:
        ValueError: If ``pyproject`` does not define a group of a method (``missing_groups``),
            or if a method folder's ``pretrained.json`` is in neither shape that
            ``pinned_files`` reads, or names a repository or revision that is not one folder
            name of the pretrained cache (``resolve_method``): ``runs_here`` catches only
            ``CliError``, so a corrupt frozen file stops the listing.
    """
    rows = {
        name: _method_row(name, spec, labels, assets_dir, pyproject)
        for name, spec in registry.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "method"
    eligible_dev = table["dev_macro_f1"].where(table["eligible"])
    table["best_dev"] = eligible_dev.notna() & (eligible_dev == eligible_dev.max())
    return table[list(METHOD_TABLE_COLUMNS)]


def format_method_table(table: pd.DataFrame) -> str:
    """Write the methods listing as the ``methods`` command prints it.

    Args:
        table: What ``method_table`` returned.

    Returns:
        The table as text: F1 with 4 decimals, seconds with 3, NaN as ``-``.
    """
    formatters = {column: text.format for column, text in _LISTED_FORMATS.items()}
    return table.to_string(formatters=formatters, na_rep="-")


def _unmet_needs_message(name: str, groups: list[str], keys: list[str], caches: list[str]) -> str:
    """Say what a method needs, and the one command that brings its groups and keys."""
    needs = [f"the dependency group {group}" for group in groups] + [f"{key} set" for key in keys]
    message = f"{name} cannot run here: it needs {' and '.join(needs + caches)}"
    if needs:
        options = [f"--group {group}" for group in groups] + (["--env-file FILE"] if keys else [])
        holding = f", FILE holding {' and '.join(f'{key}=...' for key in keys)}" if keys else ""
        message += f". Run it as uv run {' '.join(options)} {PROG} ...{holding}"
    return f"{message}{offline_hint(name)}."


def _method_row(
    name: str,
    spec: MethodSpec,
    labels: pd.DataFrame | None,
    assets_dir: Path,
    pyproject: Path,
) -> dict[str, object]:
    """Return a method's row of the listing, but for best_dev, which compares the rows."""
    directory = assets_dir / name
    latency = _latency_columns(directory)
    weights = weight_caches(directory / MODEL_FOLDER)
    return {
        "kind": spec.kind,
        **_frozen_macro_f1(directory, labels),
        **latency,
        "cold_s": latency["load_s"] + latency["max_s"],
        "eligible": _eligible(directory, latency["max_s"]),
        "default": name == CLI_DEFAULT_METHOD,
        "needs": ", ".join([f"group {group}" for group in spec.groups] + weights + api_keys(spec)),
        "runs_here": _runs_here(name, spec, assets_dir, pyproject),
    }


def _frozen_macro_f1(directory: Path, labels: pd.DataFrame | None) -> dict[str, float]:
    """Return the macro F1 of each part's frozen predictions; NaN without them or labels."""
    if labels is None or not (directory / PREDICTIONS_FILE).is_file():
        return {f"{part}_macro_f1": math.nan for part in PARTS}
    predictions = load_predictions(directory)
    scores = {}
    for part in PARTS:
        rows = predictions.loc[part]
        truth = labels.loc[rows.index]
        scores[f"{part}_macro_f1"] = evaluate(truth, rows[list(truth.columns)]).macro_f1
    return scores


def _latency_columns(directory: Path) -> dict[str, float]:
    """Return the median and slowest problem of a latency.csv, and its loading; NaN without it."""
    if not (directory / LATENCY_FILE).is_file():
        return dict.fromkeys(("median_s", "max_s", "load_s"), math.nan)
    latency = load_latency(directory)
    summary = latency_summary(latency.seconds)
    return {"median_s": summary["median_s"], "max_s": summary["max_s"], "load_s": latency.load_s}


def _eligible(directory: Path, max_s: float) -> bool:
    """Tell whether ADR 0004 lets the CLI choose a method: frozen, out of fold, fast enough.

    A method proposed after a test score was read (``method_entries["test_informed"]``,
    docs/adr/0011) is never eligible: nothing is chosen from test (ADR 0001).
    """
    frozen = (directory / MODEL_FOLDER / METHOD_FILE).is_file()
    if not frozen or not (directory / RUN_RECORD_FILE).is_file():
        return False
    entries = load_run_record(directory).method_entries
    in_sample = entries.get("dev_in_sample") is True
    return not in_sample and not entries.get("test_informed") and max_s < LATENCY_BUDGET_S


def _runs_here(name: str, spec: MethodSpec, assets_dir: Path, pyproject: Path) -> bool:
    """Tell whether ``resolve_method`` passes for a method."""
    try:
        resolve_method(name, registry={name: spec}, assets_dir=assets_dir, pyproject=pyproject)
    except CliError:
        return False
    return True
