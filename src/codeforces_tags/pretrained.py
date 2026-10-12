"""Where the pretrained weights are on this machine: one cache, one folder per pinned snapshot.

A method that encodes with pretrained weights (the ModernBERT and F2LLM methods) downloads them
once per machine, with its ``fetch`` command, into the pretrained cache:
``config.PRETRAINED_CACHE_DIR`` under the gitignored ``experiments/``, or the folder that
``$CODEFORCES_PRETRAINED_DIR`` names, read at call time (as ``read_samples`` reads
``$CODEFORCES_DATA_DIR``). Its method folder names the snapshot by repository and revision in
``pretrained.json``, never by path (docs/adr/0004), so it loads wherever the cache is.

Every method lays the cache out the same way, one folder per repository and one per revision
under it: ``<cache>/<org>--<name>/<revision>``, e.g.
``answerdotai--ModernBERT-base/8949b909...``, the repository's ``/`` written ``--``. The two
packages write ``pretrained.json`` in two shapes; ``pinned_files`` reads the file names of
either, for the command line, which checks a method's snapshot before it imports torch.

This module imports only the standard library and ``codeforces_tags.config``, so the core and
every method import it without a deep-learning library.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from codeforces_tags.config import PRETRAINED_CACHE_DIR, PRETRAINED_DIR_ENV_VAR

# What a folder name of the cache may not be, or hold: it would not be one folder, one level down.
_NOT_A_FOLDER = (".", "..")
_SEPARATORS = ("/", "\\")


def pretrained_root() -> Path:
    """Return the pretrained cache, read at call time.

    Returns:
        ``$CODEFORCES_PRETRAINED_DIR`` when it is set and not empty (the tests set it, and the
        isolation subprocess of ``check_tagging_method`` inherits it), else
        ``config.PRETRAINED_CACHE_DIR``.
    """
    return Path(os.environ.get(PRETRAINED_DIR_ENV_VAR) or PRETRAINED_CACHE_DIR)


def snapshot_folder(repo_id: str, revision: str) -> Path:
    """Return the folder of one pinned snapshot in the pretrained cache.

    Args:
        repo_id: The model's repository on the Hugging Face Hub, e.g.
            ``answerdotai/ModernBERT-base``.
        revision: The commit the snapshot holds, one folder name.

    Returns:
        ``pretrained_root() / repo_id.replace("/", "--") / revision``.

    Raises:
        ValueError: If ``repo_id`` or ``revision`` is empty, or if either folder name is ``.``
            or ``..`` or still holds a path separator (a slash or a backslash) once the
            repository's slashes are written ``--``.
    """
    repository = repo_id.replace("/", "--")
    _check_folder_name("repository", repo_id, repository)
    _check_folder_name("revision", revision, revision)
    return pretrained_root() / repository / revision


def pinned_files(weight_list: Mapping) -> list[str]:
    """Name the files a ``pretrained.json`` pins, from either of its two shapes.

    - The ModernBERT methods' (``pretrained_weight_list``, in
      ``codeforces_tags.methods.modernbert.weights``):
      ``{"repo_id", "revision", "files": {name: sha256}}``.
    - The F2LLM methods' (``PretrainedRecord.to_dict()``, in
      ``codeforces_tags.methods.f2llm.encoder``):
      ``{"repo_id", "revision", "files": [{"path": name, "sha256": sha256}, ...]}``.

    Args:
        weight_list: The content of a ``pretrained.json``.

    Returns:
        Each file's name, relative to the snapshot's folder (``snapshot_folder``): the keys of
        ``files``, or the ``"path"`` of each of its entries, in the order it lists them.

    Raises:
        ValueError: If the weight list has no ``files``, if ``files`` is neither a mapping nor a
            list of mappings that each hold a ``"path"``, or if a name is not a string; the
            message names both shapes.
    """
    files = weight_list.get("files")
    if isinstance(files, Mapping):
        names = list(files)
    elif isinstance(files, list):
        names = _entry_paths(files)
    else:
        raise ValueError(_shape_error(f"its files are {files!r}"))
    strange = [name for name in names if not isinstance(name, str)]
    if strange:
        raise ValueError(_shape_error(f"the names {strange!r} are not strings"))
    return names


def _entry_paths(entries: list) -> list:
    """Return the ``"path"`` of each entry of a list of files, refusing an entry without one."""
    pathless = [entry for entry in entries if not isinstance(entry, Mapping) or "path" not in entry]
    if pathless:
        raise ValueError(_shape_error(f"the entries {pathless!r} hold no path"))
    return [entry["path"] for entry in entries]


def _shape_error(what: str) -> str:
    """Say that a ``pretrained.json`` is in neither shape, and what is wrong with it."""
    return (
        f"The weight list is in neither shape of pretrained.json ({what}): "
        '{"files": {name: sha256}} (the ModernBERT methods) or '
        '{"files": [{"path": name, "sha256": sha256}, ...]} (the F2LLM methods).'
    )


def _check_folder_name(what: str, given: str, name: str) -> None:
    """Refuse a folder name that is empty, ``.`` or ``..``, or that holds a path separator."""
    if not name or name in _NOT_A_FOLDER or any(sep in name for sep in _SEPARATORS):
        raise ValueError(
            f"The {what} {given!r} gives the folder name {name!r}, which is not one folder of "
            "the pretrained cache: a snapshot is <cache>/<org>--<name>/<revision>."
        )
