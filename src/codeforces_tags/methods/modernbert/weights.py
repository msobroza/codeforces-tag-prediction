"""The weights a ModernBERT method reads from outside its folder, named with their hashes.

Two caches, both under the gitignored ``experiments/``, each overridden by an environment variable
that is read at call time (as ``read_samples`` reads ``$CODEFORCES_DATA_DIR``) and counts only
when it is not empty:

- the pretrained snapshot (``config.PRETRAINED_CACHE_DIR``, ``$CODEFORCES_PRETRAINED_DIR``),
  filled by the ``fetch`` command and named by ``pretrained.json`` (docs/adr/0004). Its folder,
  ``<cache>/<org>--<name>/<revision>``, is ``snapshot_folder``'s, imported from
  ``codeforces_tags.pretrained``, which the F2LLM methods use too;
- the fold models (``config.TRAINED_CACHE_DIR``, ``$CODEFORCES_TRAINED_DIR``), filled by ``train``
  or ``adopt`` and named by ``trained.json`` (docs/adr/0006), under ``trained_root()``.

A weight list (``pretrained.json``, ``trained.json``) names each file relative to its cache, with
its sha256, so a folder holds no absolute path and loads from anywhere the caches are; loading
checks every hash first. ``weight_caches`` and ``missing_caches`` read a method folder's weight
lists without hashing, for the command line, which checks a method before it imports torch. They
read any method folder, an F2LLM method's too: ``pinned_files``, imported from
``codeforces_tags.pretrained``, reads both shapes of ``pretrained.json``, and the fetch hint
names the command of the method's own package. A pinned snapshot is fetched again byte for byte,
but GPU training is not bit-reproducible: a frozen method's fold models are restored by copying
its trained cache, and training again makes another method.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from codeforces_tags.config import TRAINED_CACHE_DIR, TRAINED_DIR_ENV_VAR
from codeforces_tags.methods.modernbert.settings import PRETRAINED_FILE, TRAINED_FILE
from codeforces_tags.pretrained import pinned_files, snapshot_folder
from codeforces_tags.provenance import file_sha256

_COMMAND = "uv run --group finetuning python -m codeforces_tags.methods.modernbert"


class MissingWeightsError(FileNotFoundError):
    """A file a weight list names is not in its cache; the message says how to fill the cache."""


class ChangedWeightsError(ValueError):
    """A file in a cache does not have the sha256 its weight list records."""


def trained_root() -> Path:
    """Return the trained cache, read at call time.

    Returns:
        ``$CODEFORCES_TRAINED_DIR`` when it is set and not empty, else
        ``config.TRAINED_CACHE_DIR``, as ``codeforces_tags.pretrained.pretrained_root`` reads
        the pretrained cache.
    """
    return Path(os.environ.get(TRAINED_DIR_ENV_VAR) or TRAINED_CACHE_DIR)


def pretrained_weight_list(repo_id: str, revision: str, files: Iterable[str]) -> dict:
    """Hash the files of a fetched snapshot, for ``pretrained.json``.

    Args:
        repo_id: The model's repository.
        revision: The commit the snapshot holds.
        files: The snapshot's files to pin, by name.

    Returns:
        ``{"repo_id", "revision", "files": {name: sha256}}``, names sorted.

    Raises:
        ValueError: If the repository or the revision is not one folder name of the pretrained
            cache (``snapshot_folder`` says which).
        MissingWeightsError: If a file is not in the snapshot.
    """
    folder = snapshot_folder(repo_id, revision)
    names = sorted(files)
    _require(folder, names, _fetch_hint())
    return {
        "repo_id": repo_id,
        "revision": revision,
        "files": {name: file_sha256(folder / name) for name in names},
    }


def trained_weight_list(root: str, files: Mapping[str, str]) -> dict:
    """Hash the fold models in the trained cache, for ``trained.json``.

    Args:
        root: The method's folder in the trained cache (its name).
        files: The weight file of each fold model, by fold name, relative to that folder.

    Returns:
        ``{"root", "folds": {fold: {"file", "sha256", "bytes"}}}``.

    Raises:
        MissingWeightsError: If a file is not in the cache.
    """
    folder = trained_root() / root
    _require(folder, files.values(), f"run `{_COMMAND} train` or `adopt --method {root}` first.")
    return {
        "root": root,
        "folds": {
            fold: {
                "file": file,
                "sha256": file_sha256(folder / file),
                "bytes": (folder / file).stat().st_size,
            }
            for fold, file in files.items()
        },
    }


def check_pretrained(weight_list: Mapping) -> Path:
    """Find a pinned snapshot in the pretrained cache, and check every file's hash.

    Args:
        weight_list: The content of ``pretrained.json``.

    Returns:
        The snapshot's folder.

    Raises:
        ValueError: If the weight list's repository or revision is not one folder name of the
            pretrained cache (``snapshot_folder`` says which).
        MissingWeightsError: If a file is missing (the message names the fetch command).
        ChangedWeightsError: If a file's hash differs.
    """
    folder = snapshot_folder(weight_list["repo_id"], weight_list["revision"])
    _check(folder, dict(weight_list["files"]), _fetch_hint())
    return folder


def check_trained(weight_list: Mapping) -> dict[str, Path]:
    """Find the fold models in the trained cache, and check every file's hash.

    Args:
        weight_list: The content of ``trained.json``.

    Returns:
        The path of each fold model's weights, by fold name, in the weight list's order.

    Raises:
        MissingWeightsError: If a file is missing (the message says where to copy it from).
        ChangedWeightsError: If a file's hash differs.
    """
    folder = trained_root() / weight_list["root"]
    folds = weight_list["folds"]
    hashes = {entry["file"]: entry["sha256"] for entry in folds.values()}
    _check(folder, hashes, _copy_hint(weight_list["root"]))
    return {fold: folder / entry["file"] for fold, entry in folds.items()}


def weight_caches(folder: Path) -> list[str]:
    """Name the weights a method folder needs from outside it: those its weight lists name.

    Args:
        folder: A method folder.

    Returns:
        ``"pretrained weights"`` when it holds ``pretrained.json``, then ``"trained weights"``
        when it holds ``trained.json``; empty for a folder without weight lists.
    """
    lists = {"pretrained weights": PRETRAINED_FILE, "trained weights": TRAINED_FILE}
    return [name for name, file in lists.items() if (folder / file).is_file()]


def missing_caches(folder: Path, method: str, *, command: str = _COMMAND) -> list[str]:
    """Say which caches lack a file that a method folder's weight lists name, without hashing.

    Args:
        folder: A method folder; one without weight lists needs no cache. Its
            ``pretrained.json`` is in either shape that ``pinned_files`` reads (a ModernBERT or
            an F2LLM method's).
        method: The method's registry key, which the fetch command takes.
        command: The command of the method's package, whose ``fetch --method <method>`` fills
            the pretrained cache; by default the ModernBERT methods'.

    Returns:
        For each cache that lacks a file, which weights and how to bring them, e.g. ``"the
        pretrained weights of <repo> in <folder> (fetch them: `<command> fetch --method
        <method>`)"``; empty when every file is there. Their hashes are checked when the method
        loads.

    Raises:
        ValueError: If ``pretrained.json`` is in neither shape (``pinned_files``), or names a
            repository or revision that is not one folder name of the pretrained cache.
    """
    missing = []
    if (folder / PRETRAINED_FILE).is_file():
        pretrained = json.loads((folder / PRETRAINED_FILE).read_text(encoding="utf-8"))
        snapshot = snapshot_folder(pretrained["repo_id"], pretrained["revision"])
        if _absent(snapshot, pinned_files(pretrained)):
            missing.append(
                f"the pretrained weights of {pretrained['repo_id']} in {snapshot} (fetch them: "
                f"`{command} fetch --method {method}`)"
            )
    if (folder / TRAINED_FILE).is_file():
        trained = json.loads((folder / TRAINED_FILE).read_text(encoding="utf-8"))
        root = trained_root() / trained["root"]
        if _absent(root, [entry["file"] for entry in trained["folds"].values()]):
            missing.append(
                f"the trained weights in {root} (copy {trained['root']}/ from the trained cache "
                "of the machine that froze the method, docs/adr/0006)"
            )
    return missing


def _absent(folder: Path, names: Iterable[str]) -> bool:
    """Tell whether a file named in ``names`` is not in ``folder``."""
    return any(not (folder / name).is_file() for name in names)


def _require(folder: Path, names: Iterable[str], hint: str) -> None:
    """Fail with how to fill the cache unless every named file is in ``folder``."""
    missing = sorted(name for name in names if not (folder / name).is_file())
    if missing:
        raise MissingWeightsError(f"{folder} lacks {missing}: {hint}")


def _check(folder: Path, hashes: Mapping[str, str], hint: str) -> None:
    """Fail unless every file of ``hashes`` is in ``folder`` with the recorded sha256."""
    _require(folder, hashes, hint)
    changed = sorted(name for name, sha in hashes.items() if file_sha256(folder / name) != sha)
    if changed:
        raise ChangedWeightsError(
            f"{changed} in {folder} differ from the hashes the method folder records: the cache "
            "holds other weights than the frozen method's."
        )


def _fetch_hint() -> str:
    """Say how to fill the pretrained cache."""
    return f"run `{_COMMAND} fetch --method <name>` on this machine first."


def _copy_hint(root: str) -> str:
    """Say how to fill the trained cache of a frozen method."""
    return (
        f"copy {root}/ from the trained cache of the machine that froze the method (trained.json "
        f"lists each file's bytes and sha256). There, `{_COMMAND} adopt --method {root}` restores "
        "it from the notebooks' runs. Training again (`train --root`) makes other weights: a "
        "new method, frozen under a new name."
    )
