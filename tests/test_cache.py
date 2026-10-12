"""Tests for the on-disk cache of costly arrays."""

import subprocess

import numpy as np
import pytest

from codeforces_tags.cache import ArrayCache, cache_key
from codeforces_tags.config import ARRAY_CACHE_DIR, EXPERIMENTS_DIR, PROJECT_ROOT
from codeforces_tags.provenance import sha256_json

KEYS = {name: cache_key("encoder", "rev", "mean", "512", name) for name in ("a", "b", "c")}


def make_encoder():
    """Return a stand-in for a costly batch step, and the list of the batches it was given."""
    calls = []
    names = {key: name for name, key in KEYS.items()}

    def encode(keys):
        calls.append(list(keys))
        return np.array([[ord(names[key]), 0.5] for key in keys])

    return encode, calls


@pytest.fixture(name="cache")
def cache_fixture(tmp_path, monkeypatch):
    """Return an empty cache under ``tmp_path``, with the read counter at 0."""
    monkeypatch.setattr(ArrayCache, "reads", 0)
    return ArrayCache(tmp_path / "cache")


def test_the_cache_lives_in_the_gitignored_experiments_folder():
    cached = ARRAY_CACHE_DIR.relative_to(PROJECT_ROOT) / f"{'0' * 64}.npy"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(cached)], cwd=PROJECT_ROOT, check=False
    )

    assert ARRAY_CACHE_DIR.is_relative_to(EXPERIMENTS_DIR)
    assert ignored.returncode == 0


def test_cache_key_hashes_its_parts_in_order():
    key = cache_key("encoder", "rev", "mean", "512", "text")

    assert key == sha256_json(["encoder", "rev", "mean", "512", "text"])
    assert len(key) == 64
    assert cache_key("ab", "c") != cache_key("a", "bc")


def test_cache_key_refuses_a_part_that_is_not_a_str():
    with pytest.raises(TypeError, match="str"):
        cache_key("encoder", 512)


def test_get_or_compute_computes_once_then_reads(cache):
    calls = []

    def compute():
        calls.append(1)
        return np.arange(3.0)

    first = cache.get_or_compute(KEYS["a"], compute)
    second = cache.get_or_compute(KEYS["a"], compute)

    np.testing.assert_array_equal(first, np.arange(3.0))
    np.testing.assert_array_equal(second, first)
    assert (len(calls), ArrayCache.reads) == (1, 1)
    assert [path.name for path in cache.directory.iterdir()] == [f"{KEYS['a']}.npy"]


def test_cached_arrays_are_npy_files_that_load_without_pickle(cache):
    cache.get_or_compute(KEYS["a"], lambda: np.eye(2, dtype=np.float32))

    path = cache.directory / f"{KEYS['a']}.npy"
    assert path.read_bytes().startswith(b"\x93NUMPY")
    loaded = np.load(path, allow_pickle=False)
    assert loaded.dtype == np.float32


def test_get_or_compute_refuses_an_object_array(cache):
    with pytest.raises(TypeError, match="object"):
        cache.get_or_compute(KEYS["a"], lambda: np.array([{"a": 1}], dtype=object))
    assert not cache.directory.exists()


@pytest.mark.parametrize("key", ["../outside", "A" * 64, "a" * 63])
def test_get_or_compute_refuses_a_key_that_is_not_a_sha256_digest(cache, key):
    with pytest.raises(ValueError, match="not a cache key"):
        cache.get_or_compute(key, lambda: np.zeros(1))


def test_get_or_compute_many_computes_only_the_missing_keys_in_one_batch(cache):
    encoder, calls = make_encoder()
    cache.get_or_compute(KEYS["b"], lambda: np.array([98.0, 0.5]))

    arrays = cache.get_or_compute_many([KEYS["a"], KEYS["b"], KEYS["c"], KEYS["a"]], encoder)

    assert calls == [[KEYS["a"], KEYS["c"]]]
    np.testing.assert_array_equal(arrays[:, 0], [97, 98, 99, 97])
    assert ArrayCache.reads == 1


def test_get_or_compute_many_reads_every_cached_key_without_computing(cache):
    encoder, calls = make_encoder()
    cache.get_or_compute_many([KEYS["a"], KEYS["b"]], encoder)

    arrays = cache.get_or_compute_many([KEYS["b"], KEYS["a"]], encoder)

    assert calls == [[KEYS["a"], KEYS["b"]]]
    np.testing.assert_array_equal(arrays[:, 0], [98, 97])
    assert ArrayCache.reads == 2


def test_get_or_compute_many_refuses_a_batch_of_another_length(cache):
    with pytest.raises(ValueError, match="2 arrays for 1 keys"):
        cache.get_or_compute_many([KEYS["a"]], lambda keys: np.zeros((2, 3)))


def test_get_or_compute_many_refuses_no_keys(cache):
    with pytest.raises(ValueError, match="no keys"):
        cache.get_or_compute_many([], make_encoder()[0])


def test_reads_count_the_hits_of_every_cache(tmp_path, cache):
    other = ArrayCache(tmp_path / "other")
    for each in (cache, other, other):
        each.get_or_compute(KEYS["c"], lambda: np.ones(2))

    assert ArrayCache.reads == other.reads == 1
