"""Cache costly arrays on disk, one ``.npy`` file per key, such as an encoder's embeddings.

For encoders and other costly stateless steps; never scikit-learn's ``Pipeline(memory=...)``,
whose joblib files are pickles. Arrays are written with ``np.save`` and read with
``np.load(allow_pickle=False)``; an object array, which only pickle could store, is refused.
A method's cache lives in ``config.ARRAY_CACHE_DIR / <method name>``, under the gitignored
``experiments/``. It is a fit-time option only: a method's ``load`` returns it without a
cache, so latency is never timed on cached arrays (``ArrayCache.reads`` lets the latency
harness check it).
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from codeforces_tags.provenance import sha256_json

_KEY = re.compile(r"[0-9a-f]{64}")


def cache_key(*parts: str) -> str:
    """Hash what an array is computed from into a cache key.

    For an encoder: its ``repo_id``, ``revision``, pooling, max length (as a str) and the text.

    Args:
        *parts: The inputs, in a fixed order.

    Returns:
        ``sha256_json(list(parts))``: a sha256 hex digest, distinct for ``("ab", "c")`` and
        ``("a", "bc")``.

    Raises:
        TypeError: If a part is not a str.
    """
    if not all(isinstance(part, str) for part in parts):
        raise TypeError("cache_key takes str parts: convert a number (e.g. a max length) first.")
    return sha256_json(list(parts))


class ArrayCache:
    """A folder of ``<key>.npy`` files: each array is computed once, then read.

    Attributes:
        reads: Cache hits, counted over every ``ArrayCache`` (a class attribute), so the
            latency harness can check that no timed call read the cache.
        directory: The folder of the files, created at the first write.
    """

    reads: ClassVar[int] = 0

    def __init__(self, directory: Path) -> None:
        """Use ``directory`` for the files.

        Args:
            directory: The folder, e.g. ``config.ARRAY_CACHE_DIR / <method name>``.
        """
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ValueError(f"{key!r} is not a cache key: pass a cache_key(...) digest.")
        return self.directory / f"{key}.npy"

    def _read(self, path: Path) -> np.ndarray:
        ArrayCache.reads += 1
        return np.load(path, allow_pickle=False)

    def _write(self, path: Path, array: np.ndarray) -> None:
        """Save under a temporary name, then rename: a reader never sees half a file."""
        if array.dtype.hasobject:
            raise TypeError("An object array can only be saved as a pickle: return numbers.")
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as file:
            np.save(file, array, allow_pickle=False)
        os.replace(file.name, path)

    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Read the array of ``key``, or compute and save it.

        Args:
            key: A ``cache_key`` digest.
            compute: Computes the array when it is not cached.

        Returns:
            The array.

        Raises:
            ValueError: If ``key`` is not a sha256 hex digest.
            TypeError: If the computed array holds Python objects.
        """
        path = self._path(key)
        if path.exists():
            return self._read(path)
        array = np.asarray(compute())
        self._write(path, array)
        return array

    def get_or_compute_many(
        self, keys: Sequence[str], compute_batch: Callable[[list[str]], np.ndarray]
    ) -> np.ndarray:
        """Read the arrays of ``keys``, computing the missing ones in a single batch.

        Args:
            keys: ``cache_key`` digests; a key may repeat.
            compute_batch: Called once, on the missing keys only (each once, in the order of
                their first appearance), when any is missing; returns one array per key.

        Returns:
            The arrays stacked in the order of ``keys``.

        Raises:
            ValueError: If ``keys`` is empty or holds a key that is not a sha256 hex digest,
                or if ``compute_batch`` returns another number of arrays than it was given keys.
            TypeError: If a computed array holds Python objects.
        """
        if not keys:
            raise ValueError("get_or_compute_many got no keys: there is nothing to stack.")
        paths = {key: self._path(key) for key in keys}
        missing = [key for key, path in paths.items() if not path.exists()]
        computed = {}
        if missing:
            arrays = np.asarray(compute_batch(missing))
            if len(arrays) != len(missing):
                raise ValueError(
                    f"compute_batch returned {len(arrays)} arrays for {len(missing)} keys."
                )
            for key, array in zip(missing, arrays, strict=True):
                self._write(paths[key], array)
                computed[key] = array
        return np.stack(
            [computed[key] if key in computed else self._read(paths[key]) for key in keys]
        )
