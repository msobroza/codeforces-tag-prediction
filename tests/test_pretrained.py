"""Tests for the pretrained-weight cache: where it is, each pinned snapshot's folder and files."""

import subprocess
import sys

import pytest

from codeforces_tags.config import PRETRAINED_CACHE_DIR, PRETRAINED_DIR_ENV_VAR
from codeforces_tags.pretrained import pinned_files, pretrained_root, snapshot_folder

REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"


@pytest.fixture(name="root")
def root_fixture(monkeypatch, tmp_path):
    """Point ``$CODEFORCES_PRETRAINED_DIR`` at a folder under ``tmp_path``, and return it."""
    root = tmp_path / "pretrained"
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(root))
    return root


def test_the_cache_defaults_to_the_config_without_its_environment_variable(monkeypatch):
    monkeypatch.delenv(PRETRAINED_DIR_ENV_VAR, raising=False)

    assert pretrained_root() == PRETRAINED_CACHE_DIR


def test_an_empty_environment_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, "")

    assert pretrained_root() == PRETRAINED_CACHE_DIR


def test_the_cache_is_the_environment_variable_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "a"))
    assert pretrained_root() == tmp_path / "a"

    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "b"))

    assert pretrained_root() == tmp_path / "b"


@pytest.mark.parametrize(
    ("repo_id", "folder"),
    [
        ("answerdotai/ModernBERT-large", "answerdotai--ModernBERT-large"),
        ("codefuse-ai/F2LLM-0.6B", "codefuse-ai--F2LLM-0.6B"),
        ("org/sub/model", "org--sub--model"),
        ("model", "model"),
    ],
)
def test_a_snapshot_is_one_folder_per_repository_then_one_per_revision(root, repo_id, folder):
    assert snapshot_folder(repo_id, REVISION) == root / folder / REVISION


def test_a_snapshot_folder_reads_the_cache_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "a"))
    before = snapshot_folder("org/model", REVISION)

    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(tmp_path / "b"))

    assert (before, snapshot_folder("org/model", REVISION)) == (
        tmp_path / "a" / "org--model" / REVISION,
        tmp_path / "b" / "org--model" / REVISION,
    )


@pytest.mark.parametrize(
    ("repo_id", "revision", "what"),
    [
        ("", REVISION, "repository"),
        (".", REVISION, "repository"),
        ("..", REVISION, "repository"),
        ("org\\model", REVISION, "repository"),
        ("org/model", "", "revision"),
        ("org/model", ".", "revision"),
        ("org/model", "..", "revision"),
        ("org/model", "main/sub", "revision"),
        ("org/model", "/abs", "revision"),
        ("org/model", "main\\sub", "revision"),
    ],
)
def test_a_name_that_is_not_one_folder_of_the_cache_is_refused(root, repo_id, revision, what):
    with pytest.raises(ValueError, match=f"^The {what} .* not one folder of the pretrained cache"):
        snapshot_folder(repo_id, revision)
    assert not root.exists()


def test_pinned_files_reads_a_modernbert_weight_list_in_its_order():
    weight_list = {"repo_id": "org/model", "revision": REVISION, "files": {"b.json": "1", "a": "2"}}

    assert pinned_files(weight_list) == ["b.json", "a"]


def test_pinned_files_reads_an_f2llm_weight_list_in_its_order():
    files = [{"path": "model.safetensors", "sha256": "1"}, {"path": "1_Pooling/config.json"}]
    weight_list = {"repo_id": "org/model", "revision": REVISION, "files": files}

    assert pinned_files(weight_list) == ["model.safetensors", "1_Pooling/config.json"]


@pytest.mark.parametrize(
    ("weight_list", "what"),
    [
        ({"repo_id": "org/model"}, "its files are None"),
        ({"files": "model.safetensors"}, "its files are 'model.safetensors'"),
        ({"files": [{"sha256": "1"}]}, "the entries [{'sha256': '1'}] hold no path"),
        ({"files": ["model.safetensors"]}, "the entries ['model.safetensors'] hold no path"),
        ({"files": [{"path": 3, "sha256": "1"}]}, "the names [3] are not strings"),
        ({"files": {None: "1"}}, "the names [None] are not strings"),
    ],
)
def test_pinned_files_refuses_a_weight_list_in_neither_shape(weight_list, what):
    with pytest.raises(ValueError) as error:
        pinned_files(weight_list)

    assert str(error.value) == (
        f"The weight list is in neither shape of pretrained.json ({what}): "
        '{"files": {name: sha256}} (the ModernBERT methods) or '
        '{"files": [{"path": name, "sha256": sha256}, ...]} (the F2LLM methods).'
    )


def test_the_module_imports_only_the_standard_library_and_the_config():
    code = (
        "import sys; before = set(sys.modules); import codeforces_tags.pretrained; "
        "new = set(sys.modules) - before; "
        "print(sorted(m for m in new if m.startswith('codeforces_tags'))); "
        "print(sorted({m.partition('.')[0] for m in new} - sys.stdlib_module_names "
        "- {'codeforces_tags'}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.splitlines() == [
        "['codeforces_tags', 'codeforces_tags.config', 'codeforces_tags.pretrained']",
        "[]",
    ]
