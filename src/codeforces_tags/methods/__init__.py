"""Tagging methods: one contract, one registry and one saved form for every method.

A tagging method turns raw problems, as ``read_samples()`` returns them, into 0/1 tags
(``TaggingMethod``). Each one registers one ``MethodSpec`` in ``METHODS`` and saves itself to a
method folder: ``method.json`` (``save_method``), which names the method and its tags, plus the
method's own data files. ``load_method`` reads the folder back through the module the registry
names, so a folder never says which code loads it. This package imports neither mlflow, optuna
nor any library of a single method (docs/adr/0004).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import pandas as pd

from codeforces_tags.config import (
    LABEL_AND_SPLIT_COLUMNS,
    METHOD_FORMAT_VERSION,
    METHODS_ASSETS_DIR,
    PYPROJECT_FILE,
    RAW_INPUT_COLUMNS,
)

METHOD_FILE = "method.json"
# The keys of method.json, in the order save_method writes them.
METHOD_FILE_KEYS = ("method", "format_version", "tags")
# The method folder of a frozen method, in its directory: model_dir(name).
MODEL_FOLDER = "model"
# How a method gets honest dev predictions: cross-fitted thresholds on its scores ("score"),
# cross_predict ("fitted"), nothing learned from dev ("zero_shot"), or another way ("other").
METHOD_KINDS = ("score", "fitted", "zero_shot", "other")
# A method's registry key, e.g. "tfidf" or "descode2tag-json": never a path, so method_dir(name)
# is always a folder of METHODS_ASSETS_DIR.
_METHOD_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
# The name of a distribution at the start of a requirement such as "mlflow>=3.16".
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@runtime_checkable
class TaggingMethod(Protocol):
    """Turns raw problems into 0/1 tags (CONTEXT.md, Tagging method)."""

    @property
    def name(self) -> str:
        """Its key in ``METHODS``, e.g. ``"tfidf"`` or ``"descode2tag-json"``."""

    @property
    def tags(self) -> tuple[str, ...]:
        """The output columns, in order: distinct tags of ``ALL_TAGS``."""

    def predict(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of raw problems.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Int 0/1, with the index of ``raw_problems``, one column per tag of ``tags``.
        """

    def save(self, folder: Path) -> None:
        """Write the method's own data files into ``folder`` (``save_method`` writes the rest).

        Args:
            folder: An existing directory.
        """


@runtime_checkable
class ScoreMethod(TaggingMethod, Protocol):
    """A tagging method of kind "score": it also exposes its scores and per-tag thresholds."""

    def scores(self, raw_problems: pd.DataFrame) -> pd.DataFrame:
        """Score raw problems.

        Args:
            raw_problems: Rows of ``read_samples()``, indexed by ``sample_id``.

        Returns:
            Floats, with the index of ``raw_problems``, one column per tag of ``tags``: the
            ``decision_function`` of the method's fitted ``CrossFittedTagThresholds``, or, for
            a bag of fine-tuned fold models, their mean probability (docs/adr/0005).
        """

    def predict_from_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Predict the tags of problems already scored, without scoring them again.

        ``predict(raw_problems)`` returns ``predict_from_scores(scores(raw_problems))``, so the
        command line prints a problem's scores and the tags taken from those same scores, at
        the cost of one ``scores``.

        Args:
            scores: What ``scores`` returned for some raw problems.

        Returns:
            Int 0/1, with the index and columns of ``scores``: 1 where a score reaches its
            tag's threshold.
        """

    def tag_thresholds(self) -> pd.Series:
        """Return the per-tag thresholds that ``predict_from_scores`` applies.

        A tag is predicted where its score reaches its threshold (``>=``), so
        ``apply_thresholds(scores, tag_thresholds())`` is ``predict_from_scores(scores)``, which
        ``check_tagging_method`` checks; a caller can then show each score beside the threshold
        it was compared with.

        Returns:
            Floats indexed by ``tags``, in order: ``inf`` for a tag that is never predicted
            (e.g. one that no dev problem has).
        """


@dataclass(frozen=True)
class MethodSpec:  # pylint: disable=too-many-instance-attributes
    """How the code finds, runs and logs one registered tagging method.

    Attributes:
        module: The module that serves the method, imported lazily; it exposes
            ``load(folder) -> TaggingMethod``.
        kind: One of ``METHOD_KINDS``.
        requirements: The packages the method's logged MLflow model needs, pinned from
            ``uv.lock`` when it is logged.
        groups: The uv dependency groups that run it, e.g. ``("llm",)``.
        calls_api: Whether the method sends paid requests.
        score_atol: The tolerance of ``verify`` on its scores; 0.0 when bit-reproducible.
        depends_on: The base methods of a composite method.
        input_columns: The raw columns ``predict`` reads: the logged model's signature. Never
            a column of ``LABEL_AND_SPLIT_COLUMNS``.
    """

    module: str
    kind: str
    requirements: tuple[str, ...]
    groups: tuple[str, ...] = ()
    calls_api: bool = False
    score_atol: float = 0.0
    depends_on: tuple[str, ...] = ()
    input_columns: tuple[str, ...] = RAW_INPUT_COLUMNS

    def __post_init__(self) -> None:
        """Check the kind and the input columns; store every sequence field as a tuple.

        Sequence fields become tuples because JSON gives lists.

        Raises:
            ValueError: If ``kind`` is not one of ``METHOD_KINDS``, or if ``input_columns``
                holds a label or split column (``LABEL_AND_SPLIT_COLUMNS``).
            TypeError: If a sequence field is a single str.
        """
        if self.kind not in METHOD_KINDS:
            raise ValueError(f"{self.kind!r} is not a method kind: expected one of {METHOD_KINDS}.")
        for field in ("requirements", "groups", "depends_on", "input_columns"):
            value = getattr(self, field)
            if isinstance(value, str):
                raise TypeError(
                    f"MethodSpec.{field} must be a tuple of str, not the str {value!r}: write "
                    f"({value!r},)."
                )
            object.__setattr__(self, field, tuple(value))
        read = [column for column in self.input_columns if column in LABEL_AND_SPLIT_COLUMNS]
        if read:
            raise ValueError(
                f"MethodSpec.input_columns hold {read}: a method never reads the labels or the "
                "split, so check_tagging_method can refuse one that does."
            )


# The four F2LLM methods (codeforces_tags.methods.f2llm) share their requirements and their
# verify tolerance, written here because this package never imports a method's package. The
# tolerance covers the GPU-to-CPU drift of their scores. It is ten times the largest gap that
# python -m codeforces_tags.methods.f2llm atol measured, rounded up to one significant digit.
_F2LLM_REQUIREMENTS = (
    "numpy",
    "pandas",
    "peft",
    "safetensors",
    "scikit-learn",
    "scipy",
    "skops",
    "torch",
    "transformers",
)
_F2LLM_SCORE_ATOL = 9e-05
# Both ModernBERT bags: re-scored anywhere but in the GPU batches that froze them, a score moves
# by up to 8.25e-3 on the dev folds (float32 on the CPU, modernbert_lora; docs/adr/0005).
_MODERNBERT_SCORE_ATOL = 1e-2

# One line per method, sorted by name. Read-only: a test that needs another registry passes
# its own (``METHODS | {...}`` gives a dict).
METHODS: Mapping[str, MethodSpec] = MappingProxyType(
    {
        "code2tag-json": MethodSpec(
            module="codeforces_tags.methods.zero_shot.methods",
            kind="zero_shot",
            requirements=("numpy", "pandas", "jinja2", "openai", "pyyaml"),
            groups=("llm",),
            calls_api=True,
        ),
        "code2tag-text": MethodSpec(
            module="codeforces_tags.methods.zero_shot.methods",
            kind="zero_shot",
            requirements=("numpy", "pandas", "jinja2", "openai", "pyyaml"),
            groups=("llm",),
            calls_api=True,
        ),
        "descode2tag-json": MethodSpec(
            module="codeforces_tags.methods.zero_shot.methods",
            kind="zero_shot",
            requirements=("numpy", "pandas", "jinja2", "openai", "pyyaml"),
            groups=("llm",),
            calls_api=True,
        ),
        "descode2tag-text": MethodSpec(
            module="codeforces_tags.methods.zero_shot.methods",
            kind="zero_shot",
            requirements=("numpy", "pandas", "jinja2", "openai", "pyyaml"),
            groups=("llm",),
            calls_api=True,
        ),
        "f2llm_early_fusion": MethodSpec(
            module="codeforces_tags.methods.f2llm.method",
            kind="score",
            requirements=_F2LLM_REQUIREMENTS,
            groups=("f2llm",),
            score_atol=_F2LLM_SCORE_ATOL,
        ),
        "f2llm_late_fusion": MethodSpec(
            module="codeforces_tags.methods.f2llm.method",
            kind="score",
            requirements=_F2LLM_REQUIREMENTS,
            groups=("f2llm",),
            score_atol=_F2LLM_SCORE_ATOL,
        ),
        "f2llm_probe": MethodSpec(
            module="codeforces_tags.methods.f2llm.method",
            kind="score",
            requirements=_F2LLM_REQUIREMENTS,
            groups=("f2llm",),
            score_atol=_F2LLM_SCORE_ATOL,
        ),
        "f2llm_setfit": MethodSpec(
            module="codeforces_tags.methods.f2llm.method",
            kind="score",
            requirements=_F2LLM_REQUIREMENTS,
            groups=("f2llm",),
            score_atol=_F2LLM_SCORE_ATOL,
        ),
        "jev-statement": MethodSpec(
            module="codeforces_tags.methods.typesafe_jev.method",
            kind="score",
            requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
            groups=("llm-typesafe",),
            calls_api=True,
        ),
        "jev-statement-code": MethodSpec(
            module="codeforces_tags.methods.typesafe_jev.method",
            kind="score",
            requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
            groups=("llm-typesafe",),
            calls_api=True,
        ),
        # Jev with the words a search chose from jev-statement-code's (docs/adr/0010).
        "jev-statement-code-gepa": MethodSpec(
            module="codeforces_tags.methods.typesafe_jev.method",
            kind="score",
            requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
            groups=("llm-typesafe",),
            calls_api=True,
        ),
        # The calibrated twins: each reads its base's answers and chooses its thresholds by the
        # Platt + Lipton threshold rule (docs/adr/0011); a prediction asks its base's request.
        "jev-statement-code-gepa-platt": MethodSpec(
            module="codeforces_tags.methods.typesafe_jev.method",
            kind="score",
            requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
            groups=("llm-typesafe",),
            calls_api=True,
            depends_on=("jev-statement-code-gepa",),
        ),
        "jev-statement-code-platt": MethodSpec(
            module="codeforces_tags.methods.typesafe_jev.method",
            kind="score",
            requirements=("numpy", "pandas", "jinja2", "pyyaml", "scikit-learn", "typesafe-sdk"),
            groups=("llm-typesafe",),
            calls_api=True,
            depends_on=("jev-statement-code",),
        ),
        "label_prior": MethodSpec(
            module="codeforces_tags.methods.label_prior.method",
            kind="fitted",
            requirements=("numpy", "pandas"),
        ),
        "modernbert_base_full": MethodSpec(
            module="codeforces_tags.methods.modernbert.method",
            kind="score",
            requirements=("numpy", "pandas", "peft", "safetensors", "torch", "transformers"),
            groups=("finetuning",),
            score_atol=_MODERNBERT_SCORE_ATOL,
        ),
        "modernbert_lora": MethodSpec(
            module="codeforces_tags.methods.modernbert.method",
            kind="score",
            requirements=("numpy", "pandas", "peft", "safetensors", "torch", "transformers"),
            groups=("finetuning",),
            score_atol=_MODERNBERT_SCORE_ATOL,
        ),
        "tfidf": MethodSpec(
            module="codeforces_tags.model.tag_model",
            kind="score",
            requirements=("numpy", "pandas", "scikit-learn", "scipy", "skops"),
        ),
    }
)


def installed_methods(
    registry: Mapping[str, MethodSpec] = METHODS, *, pyproject: Path = PYPROJECT_FILE
) -> dict[str, MethodSpec]:
    """Keep the methods that this environment can run.

    Args:
        registry: The methods to filter.
        pyproject: The ``pyproject.toml`` whose ``[dependency-groups]`` list each group's
            requirements.

    Returns:
        The specs, in registry order, whose ``groups`` are empty or whose groups' distributions
        are all installed.

    Raises:
        ValueError: If a spec names a group that ``pyproject`` does not define.
    """
    groups = _dependency_groups(pyproject)
    return {
        name: spec
        for name, spec in registry.items()
        if all(_group_installed(groups, group, name) for group in spec.groups)
    }


def missing_groups(name: str, spec: MethodSpec, *, pyproject: Path = PYPROJECT_FILE) -> list[str]:
    """Name the dependency groups of one method that this environment lacks.

    Args:
        name: The method's registry key, which an error names.
        spec: Its spec.
        pyproject: The ``pyproject.toml`` whose ``[dependency-groups]`` list each group's
            requirements.

    Returns:
        The groups of ``spec.groups``, in its order, whose distributions are not all installed;
        empty when the method can run here (``installed_methods`` keeps it).

    Raises:
        ValueError: If ``pyproject`` does not define one of the method's groups.
    """
    groups = _dependency_groups(pyproject)
    return [group for group in spec.groups if not _group_installed(groups, group, name)]


def sibling_methods(name: str, registry: Mapping[str, MethodSpec] = METHODS) -> list[str]:
    """Name a method's siblings: the other registered methods of its package (docs/adr/0004).

    A method's package is the ``codeforces_tags.methods.<package>`` prefix of its
    ``MethodSpec.module``, so two methods served by different modules of one package are
    siblings. A local method's ``freeze`` supersedes the siblings whose predictions are committed.

    Args:
        name: A method of ``registry``.
        registry: The methods.

    Returns:
        The other methods of ``registry`` in the package of ``name``, in registry order.
    """
    package = _package(registry[name].module)
    return [
        other
        for other, spec in registry.items()
        if other != name and _package(spec.module) == package
    ]


def save_method(method: TaggingMethod, folder: Path) -> None:
    """Save a tagging method to a method folder.

    Writes ``method.json`` (the method's name, ``METHOD_FORMAT_VERSION`` and its tags), then
    calls ``method.save(folder)``. Nothing written may hold an absolute path, so the folder
    loads from anywhere.

    Args:
        method: The method to save.
        folder: The method folder, created when missing.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stored = {
        "method": method.name,
        "format_version": METHOD_FORMAT_VERSION,
        "tags": list(method.tags),
    }
    (folder / METHOD_FILE).write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    method.save(folder)


def load_method(folder: Path, *, registry: Mapping[str, MethodSpec] = METHODS) -> TaggingMethod:
    """Load the tagging method saved in a method folder.

    The module that loads it comes from ``registry``, never from the folder. Its import path
    is frozen: logged MLflow models call it by that path.

    Args:
        folder: A folder written by ``save_method``.
        registry: The registered methods.

    Returns:
        The method, as the ``load(folder)`` of its registered module returns it.

    Raises:
        FileNotFoundError: If ``folder`` holds no ``method.json``.
        ValueError: If ``method.json`` is malformed (``read_method_file``), has another format
            version or names an unregistered method, or if the loaded method's name or tags
            differ from those it stores.
    """
    stored = read_method_file(Path(folder))
    if stored["format_version"] != METHOD_FORMAT_VERSION:
        raise ValueError(
            f"{Path(folder) / METHOD_FILE} has format version {stored['format_version']}; this "
            f"code reads version {METHOD_FORMAT_VERSION}."
        )
    name = stored["method"]
    if name not in registry:
        raise ValueError(
            f"{folder} holds the method {name!r}, which is not registered: the registry has "
            f"{sorted(registry)}."
        )
    method = importlib.import_module(registry[name].module).load(Path(folder))
    if method.name != name:
        raise ValueError(f"{folder} stores the method {name!r}, but loaded {method.name!r}.")
    if tuple(method.tags) != tuple(stored["tags"]):
        raise ValueError(
            f"{folder} stores the tags {stored['tags']}, but the loaded method has the tags "
            f"{list(method.tags)}."
        )
    return method


def method_dir(name: str) -> Path:
    """Return the directory of a method's frozen files.

    Args:
        name: The method's registry key.

    Returns:
        ``METHODS_ASSETS_DIR / name``.

    Raises:
        ValueError: If ``name`` is not a method name (lower-case letters, digits, ``_`` and
            ``-``): a path would name another folder.
    """
    if not _METHOD_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a method name: a registry key holds lower-case letters, digits, "
            "'_' and '-' only."
        )
    return METHODS_ASSETS_DIR / name


def model_dir(name: str) -> Path:
    """Return the method folder of a frozen method.

    Args:
        name: The method's registry key.

    Returns:
        ``method_dir(name) / MODEL_FOLDER``.
    """
    return method_dir(name) / MODEL_FOLDER


def read_method_file(folder: Path) -> dict:
    """Read a method folder's ``method.json``, without checking its values.

    Args:
        folder: A folder written by ``save_method``.

    Returns:
        Its keys, ``METHOD_FILE_KEYS`` among them.

    Raises:
        FileNotFoundError: If ``folder`` holds no ``method.json``.
        ValueError: If ``method.json`` is not JSON, or not an object with every key of
            ``METHOD_FILE_KEYS``.
    """
    path = Path(folder) / METHOD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder} holds no {METHOD_FILE}: save it with save_method().")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(
            f"{path} is not valid JSON ({error}): save it with save_method()."
        ) from error
    missing = [key for key in METHOD_FILE_KEYS if not isinstance(stored, dict) or key not in stored]
    if missing:
        raise ValueError(f"{path} lacks the keys {missing}: save it with save_method().")
    return stored


def _package(module: str) -> str:
    """Return ``codeforces_tags.methods.<package>``, the package a method's module is in."""
    return ".".join(module.split(".")[:3])


def _dependency_groups(pyproject: Path) -> dict[str, list]:
    """Read the ``[dependency-groups]`` table of a ``pyproject.toml``; empty without one."""
    return tomllib.loads(pyproject.read_text(encoding="utf-8")).get("dependency-groups", {})


def _group_installed(groups: Mapping[str, list], group: str, method: str) -> bool:
    """Whether every distribution of a dependency group, and of the groups it includes, is there."""
    if group not in groups:
        raise ValueError(
            f"The method {method!r} needs the dependency group {group!r}, which pyproject.toml "
            "does not define."
        )
    for entry in groups[group]:
        if isinstance(entry, dict):
            installed = _group_installed(groups, entry["include-group"], method)
        else:
            installed = _distribution_installed(_DISTRIBUTION.match(entry).group(0))
        if not installed:
            return False
    return True


def _distribution_installed(distribution: str) -> bool:
    """Whether a distribution is installed in this environment."""
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True
