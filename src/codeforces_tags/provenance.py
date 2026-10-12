"""Fingerprints of what a frozen result was computed from, gathered in a run record.

A run record (CONTEXT.md) pins the code (git commit, uncommitted changes), the data (the
dataset, split and near-duplicate verdict files), the environment (Python, package versions,
``uv.lock``, BLAS thread pools) and the settings of a frozen result. Every method's ``freeze``
writes one next to its predictions, with ``make_run_record``; ``check_dataset`` and
``check_clean_tree`` refuse to freeze from another dataset or from uncommitted code.

Standard library and threadpoolctl only: the core never imports mlflow or a deep-learning
library (``torch`` is described only when a method has already imported it).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

from threadpoolctl import threadpool_info

from codeforces_tags import config
from codeforces_tags.config import (
    DATA_DIR_ENV_VAR,
    DATASET_FILES,
    DATASET_SHA256,
    NEAR_DUPLICATE_VERDICTS,
    PROJECT_ROOT,
    RECORDED_PACKAGES,
    RECORDED_SETTINGS,
    SPLIT_FILE,
    UV_LOCK_FILE,
)
from codeforces_tags.data.loading import resolve_data_dir

# The files read_samples() reads: one per problem.
_PROBLEM_FILE = re.compile(r"sample_\d+\.json")
# What git_state() counts as code: a change elsewhere (assets/, notebooks/) leaves it clean.
_CODE_PATHS = ("src", "pyproject.toml", "uv.lock")
# The keys of threadpool_info() a run record keeps: its file paths name the machine.
_THREADPOOL_KEYS = (
    "user_api",
    "internal_api",
    "prefix",
    "version",
    "num_threads",
    "threading_layer",
    "architecture",
)


def json_safe(obj: object) -> object:
    """Return ``obj`` with JSON's shapes: lists for tuples, strings for non-finite floats.

    JSON has no infinity or NaN (Python would write the non-standard ``Infinity``), so
    ``inf``, ``-inf`` and ``nan``, such as the threshold of a tag with no positive, become the
    strings ``"inf"``, ``"-inf"`` and ``"nan"``.

    Args:
        obj: JSON types (``numpy.float64`` is a float), tuples, and nested mappings or
            sequences of them.

    Returns:
        A copy; ``obj`` is left unchanged.
    """
    if isinstance(obj, Mapping):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(float(obj))
    return obj


def sha256_json(obj: object) -> str:
    """Hash the canonical JSON of ``obj``: sorted keys, compact separators, ASCII.

    Args:
        obj: Anything ``json_safe`` accepts; non-finite floats are hashed as the strings
            ``"inf"``, ``"-inf"`` and ``"nan"``.

    Returns:
        The sha256 hex digest.
    """
    text = json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash the bytes of a file.

    Args:
        path: The file.

    Returns:
        The sha256 hex digest.
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _problem_files(data_dir: str | os.PathLike[str] | None) -> list[Path]:
    """Return the dataset's problem files, in file name order."""
    path = resolve_data_dir(data_dir)
    return sorted(
        (file for file in path.iterdir() if _PROBLEM_FILE.fullmatch(file.name)),
        key=lambda file: file.name,
    )


def _files_sha256(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8") + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


def dataset_sha256(data_dir: str | os.PathLike[str] | None = None) -> str:
    """Hash the dataset: every problem file's name and bytes, in file name order.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir`` (``$CODEFORCES_DATA_DIR``
            when omitted).

    Returns:
        The sha256 hex digest over the ``(file name, NUL, bytes, NUL)`` of each
        ``sample_<id>.json`` file; other files are left out.
    """
    return _files_sha256(_problem_files(data_dir))


def check_dataset(data_dir: str | os.PathLike[str] | None = None) -> None:
    """Refuse any dataset but the one the frozen results were computed on.

    Args:
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.

    Raises:
        ValueError: Unless the dataset has ``config.DATASET_FILES`` problem files and its
            ``dataset_sha256`` equals ``config.DATASET_SHA256``.
    """
    files = _problem_files(data_dir)
    digest = _files_sha256(files)
    if len(files) != DATASET_FILES or digest != DATASET_SHA256:
        raise ValueError(
            f"The dataset has {len(files)} problem files and the sha256 {digest}; the frozen "
            f"results were computed on {DATASET_FILES} files with the sha256 {DATASET_SHA256}. "
            f"Point ${DATA_DIR_ENV_VAR} at the challenge dataset, unchanged."
        )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def git_state(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Describe the code a result is computed with: its commit, and whether it has changes.

    Args:
        root: The repository root.

    Returns:
        ``{"commit": <HEAD's sha>, "code_dirty": <bool>}``; ``code_dirty`` is true when
        ``src/``, ``pyproject.toml`` or ``uv.lock`` has a change or an untracked file
        (changes under ``assets/`` or ``notebooks/`` do not count).

    Raises:
        subprocess.CalledProcessError: If ``root`` is not in a git repository with a commit.
    """
    commit = _git(root, "rev-parse", "HEAD")
    commit.check_returncode()
    status = _git(root, "status", "--porcelain", "--", *_CODE_PATHS)
    status.check_returncode()
    return {"commit": commit.stdout.strip(), "code_dirty": bool(status.stdout.strip())}


def paths_dirty(paths: Sequence[str | os.PathLike[str]], root: Path = PROJECT_ROOT) -> list[str]:
    """List the uncommitted changes of some files and folders, as ``git status`` shows them.

    Unlike ``git_state``, whose three code paths feed a run record's ``git``, it looks at the
    paths it is given, e.g. a Dockerfile and a frozen method folder that an image is built from.
    It passes ``--untracked-files=all``, so each untracked file is listed by name, even in a new
    folder, whatever the user's git config: ``status.showUntrackedFiles=no`` would hide them
    all. A file that git ignores (e.g. a stray ``*.pkl``) is not a change and is not listed;
    ``conformance.check_method_folder`` refuses one in a method folder.

    Args:
        paths: Files or folders, absolute (inside ``root``) or relative to ``root``.
        root: The repository root.

    Returns:
        One line of ``git status --porcelain`` per change under ``paths``: modified, staged,
        deleted or untracked, e.g. ``" M deploy/Dockerfile"`` or ``"?? assets/new.csv"``, its
        path relative to the repository root. Empty when every path is clean, and for no path
        (``git status`` given none would list the whole tree).

    Raises:
        subprocess.CalledProcessError: If ``root`` is not in a git repository, or a path lies
            outside it.
    """
    if not paths:
        return []
    status = _git(
        root, "status", "--porcelain", "--untracked-files=all", "--", *map(os.fspath, paths)
    )
    status.check_returncode()
    return status.stdout.splitlines()


def committed_unchanged(path: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> bool:
    """Tell whether a file is tracked by git and identical to its committed version.

    Unlike ``git_state``'s ``code_dirty``, it looks at one file, e.g. one under ``assets/``.

    Args:
        path: The file, absolute or relative to ``root``.
        root: The repository root.

    Returns:
        True when ``path`` is tracked and equals its version at ``HEAD``; False when it is
        modified, staged but not committed, untracked or missing.
    """
    tracked = _git(root, "ls-files", "--error-unmatch", "--", str(path)).returncode == 0
    return tracked and _git(root, "diff", "--quiet", "HEAD", "--", str(path)).returncode == 0


def committed_versions(
    path: str | os.PathLike[str], root: Path = PROJECT_ROOT
) -> list[tuple[str, bytes]]:
    """Read a file at each commit of ``HEAD``'s history that changed it, oldest first.

    Unlike ``committed_unchanged``, which compares the file with ``HEAD`` alone, it shows a
    commit that removed or changed what an earlier one held, e.g. a line of an answers file.

    Args:
        path: The file, absolute (inside ``root``) or relative to ``root``.
        root: The repository root.

    Returns:
        ``(sha, bytes)`` for each commit that changed the file, parents before children, with
        ``b""`` for a commit that deleted it; none when no commit changed it.

    Raises:
        subprocess.CalledProcessError: If ``root`` is not in a git repository with a commit.
    """
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.resolve().relative_to(Path(root).resolve())
    log = _git(root, "log", "--topo-order", "--reverse", "--format=%H", "--", str(relative))
    log.check_returncode()
    return [(sha, _committed_bytes(root, sha, relative)) for sha in log.stdout.split()]


def _committed_bytes(root: Path, sha: str, relative: Path) -> bytes:
    """Return a file's bytes at a commit, as git stores them; none when the commit deleted it."""
    shown = subprocess.run(
        ["git", "show", f"{sha}:./{relative.as_posix()}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return shown.stdout if shown.returncode == 0 else b""


def last_commit(path: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> str:
    """Name the commit that last changed a file, e.g. a prompted method's dev answers file.

    Args:
        path: The file, absolute or relative to ``root``.
        root: The repository root.

    Returns:
        The sha of the last commit of ``HEAD``'s history that changed ``path``.

    Raises:
        ValueError: If no commit changed it (an untracked file, or one never committed).
        subprocess.CalledProcessError: If ``root`` is not in a git repository with a commit.
    """
    log = _git(root, "log", "-1", "--format=%H", "--", str(path))
    log.check_returncode()
    commit = log.stdout.strip()
    if not commit:
        raise ValueError(f"No commit changed {path}: commit it first.")
    return commit


def check_clean_tree(allow_dirty: bool, root: Path = PROJECT_ROOT) -> None:
    """Refuse to freeze a result from uncommitted code, unless it is allowed.

    Args:
        allow_dirty: True to let uncommitted code through (a ``--allow-dirty`` option, which
            the run record records).
        root: The repository root.

    Raises:
        RuntimeError: If ``git_state(root)`` finds the code dirty and ``allow_dirty`` is False.
    """
    if not allow_dirty and git_state(root)["code_dirty"]:
        raise RuntimeError(
            "src/, pyproject.toml or uv.lock has uncommitted changes, which the run record could "
            "not pin: commit them first, or pass --allow-dirty (recorded in the run record)."
        )


def canonical_name(name: str) -> str:
    """Return a distribution's normalised name (PEP 503), as ``uv.lock`` writes it.

    Args:
        name: A distribution name, e.g. ``"Scikit_Learn"``.

    Returns:
        Its lowercase name with runs of ``-``, ``_`` and ``.`` turned into ``-``, e.g.
        ``"scikit-learn"``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock: Path = UV_LOCK_FILE) -> dict[str, tuple[str, ...]]:
    """Read the versions ``uv.lock`` pins each package at.

    Args:
        lock: The ``uv.lock`` file.

    Returns:
        Each package's sorted versions: a package has several when the lock forks by Python
        version or platform (e.g. numpy).
    """
    with lock.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    versions: dict[str, set[str]] = {}
    for package in packages:
        versions.setdefault(package["name"], set()).add(package["version"])
    return {name: tuple(sorted(found)) for name, found in versions.items()}


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _threadpools() -> list[dict[str, object]]:
    return [
        {key: pool[key] for key in _THREADPOOL_KEYS if key in pool} for pool in threadpool_info()
    ]


def _torch() -> dict[str, object] | None:
    """Describe torch when a method has imported it; never import it here."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {"version": torch.__version__, "cuda": torch.version.cuda, "devices": devices}


def environment(
    requirements: Sequence[str] = (), *, lock: Path = UV_LOCK_FILE
) -> dict[str, object]:
    """Describe the environment a result is computed in.

    Args:
        requirements: The method's own requirements (``MethodSpec.requirements``), recorded
            after ``config.RECORDED_PACKAGES``.
        lock: The ``uv.lock`` file the versions are compared with.

    Returns:
        ``python`` and ``platform``; ``packages``, each recorded package's installed version
        (None when absent); ``lock_mismatches``, ``{name: {"installed", "locked"}}`` for each
        one whose installed version is not among its locked versions; ``uv_lock_sha256``;
        ``threadpools``, the BLAS and OpenMP pools loaded (``threadpool_info()`` without file
        paths); and ``torch`` (its version, CUDA version and device names) when a method has
        imported it, else None.
    """
    names = list(dict.fromkeys([*RECORDED_PACKAGES, *requirements]))
    packages = {name: _installed_version(name) for name in names}
    locked = locked_versions(lock)
    mismatches = {}
    for name, installed in packages.items():
        versions = locked.get(canonical_name(name), ())
        if installed not in versions:
            mismatches[name] = {"installed": installed, "locked": list(versions)}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "lock_mismatches": mismatches,
        "uv_lock_sha256": file_sha256(lock),
        "threadpools": _threadpools(),
        "torch": _torch(),
    }


def settings_snapshot() -> dict[str, object]:
    """Return the project-wide settings a run record holds.

    Returns:
        ``{name: value}`` for each name of ``config.RECORDED_SETTINGS``, in that order, as JSON
        values (tuples become lists).
    """
    return {name: json_safe(getattr(config, name)) for name in RECORDED_SETTINGS}


@dataclass(frozen=True)
class RunRecord:  # pylint: disable=too-many-instance-attributes
    """What a frozen result was computed from (CONTEXT.md, Run record).

    The four dict fields are stored as ``json_safe`` copies and ``prompts`` as a tuple, so a
    record equals itself read back from its JSON file.

    Attributes:
        method: The method's name, its ``METHODS`` key.
        recipe_sha256: ``sha256_json`` of the method's committed inputs (its settings,
            parameters, frozen thresholds, prompts, tags), never of re-derived floats.
        git: ``git_state()``.
        environment: ``environment(requirements)``.
        dataset_sha256: ``dataset_sha256()`` of the dataset read.
        split_sha256: ``file_sha256`` of the split file the result was computed on.
        verdicts_sha256: ``file_sha256`` of the near-duplicate verdicts.
        settings: Always ``settings_snapshot()``.
        allow_dirty: Whether uncommitted code was let through.
        created_at: When the record was made, ISO 8601 in UTC.
        prompts: Pinned ``prompts:/<name>/<version>`` URIs (prompted methods).
        method_entries: What the method adds, e.g. its own ``settings``, ``dev_in_sample``,
            ``supersedes`` (ADR 0004), its parameters and thresholds.
    """

    method: str
    recipe_sha256: str
    git: dict[str, object]
    environment: dict[str, object]
    dataset_sha256: str
    split_sha256: str
    verdicts_sha256: str
    settings: dict[str, object]
    allow_dirty: bool
    created_at: str
    prompts: tuple[str, ...] = ()
    method_entries: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store the prompts as a tuple and the dict fields as JSON values.

        Raises:
            TypeError: If ``prompts`` is a single str.
        """
        if isinstance(self.prompts, str):
            raise TypeError("RunRecord.prompts must be a sequence of prompt URIs, not one str.")
        object.__setattr__(self, "prompts", tuple(self.prompts))
        for name in ("git", "environment", "settings", "method_entries"):
            object.__setattr__(self, name, json_safe(getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        """Return the record as JSON values, keyed by ``RUN_RECORD_KEYS`` in order.

        Returns:
            A dict that ``json.dumps(..., allow_nan=False)`` accepts.
        """
        return asdict(self) | {"prompts": list(self.prompts)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> RunRecord:
        """Build a record from ``to_dict``'s output, e.g. read from ``run_record.json``.

        Args:
            values: Exactly the keys of ``RUN_RECORD_KEYS``.

        Returns:
            The record.

        Raises:
            ValueError: If a key is missing or unknown.
        """
        missing = [key for key in RUN_RECORD_KEYS if key not in values]
        unknown = sorted(set(values) - set(RUN_RECORD_KEYS))
        if missing or unknown:
            raise ValueError(
                f"A run record has exactly the keys {list(RUN_RECORD_KEYS)}: missing {missing}, "
                f"unknown {unknown}."
            )
        return cls(**{key: values[key] for key in RUN_RECORD_KEYS})


# The keys of run_record.json, in order: check_frozen_method checks that a frozen one has them.
RUN_RECORD_KEYS: tuple[str, ...] = tuple(item.name for item in fields(RunRecord))


def make_run_record(  # pylint: disable=too-many-arguments
    method: str,
    recipe_sha256: str,
    *,
    allow_dirty: bool,
    requirements: Sequence[str] = (),
    prompts: Sequence[str] = (),
    method_entries: Mapping[str, object] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    split_path: str | os.PathLike[str] = SPLIT_FILE,
) -> RunRecord:
    """Fingerprint the code, data, environment and settings of a result about to be frozen.

    It records; it does not refuse: a method's ``freeze`` calls ``check_dataset`` and
    ``check_clean_tree`` first.

    Args:
        method: The method's name.
        recipe_sha256: ``sha256_json`` of the method's committed inputs.
        allow_dirty: Whether uncommitted code was let through.
        requirements: The method's own requirements, recorded by ``environment``.
        prompts: Pinned ``prompts:/<name>/<version>`` URIs.
        method_entries: What the method adds (e.g. ``settings``, ``supersedes``).
        data_dir: Dataset directory, resolved by ``resolve_data_dir``.
        split_path: The split file the result was computed on, e.g. the ``--split`` of a
            method's ``freeze``.

    Returns:
        The run record, created now.
    """
    return RunRecord(
        method=method,
        recipe_sha256=recipe_sha256,
        git=git_state(),
        environment=environment(requirements),
        dataset_sha256=dataset_sha256(data_dir),
        split_sha256=file_sha256(split_path),
        verdicts_sha256=file_sha256(NEAR_DUPLICATE_VERDICTS),
        settings=settings_snapshot(),
        allow_dirty=allow_dirty,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prompts=tuple(prompts),
        method_entries=dict(method_entries or {}),
    )
