"""Tests for the weight caches of the ModernBERT methods: where they are, and their hashed lists."""

import json
import re

import pytest

from codeforces_tags.config import PRETRAINED_DIR_ENV_VAR, TRAINED_CACHE_DIR, TRAINED_DIR_ENV_VAR
from codeforces_tags.methods.modernbert.settings import PRETRAINED_FILE, TRAINED_FILE
from codeforces_tags.methods.modernbert.weights import (
    ChangedWeightsError,
    MissingWeightsError,
    check_pretrained,
    check_trained,
    missing_caches,
    pretrained_weight_list,
    trained_root,
    trained_weight_list,
    weight_caches,
)
from codeforces_tags.provenance import file_sha256

COMMAND = "uv run --group finetuning python -m codeforces_tags.methods.modernbert"
REPO, REVISION = "answerdotai/ModernBERT-base", "0123abcd"
SNAPSHOT_FILES = {"tokenizer.json": b"{}", "config.json": b'{"a": 1}', "model.safetensors": b"w"}
FOLD_FILES = {"fold_1": "fold_1/weights.safetensors", "fold_0": "fold_0/weights.safetensors"}


@pytest.fixture(name="roots")
def roots_fixture(monkeypatch, tmp_path):
    """Point both caches at empty temporary folders, and return them by name."""
    roots = {"pretrained": tmp_path / "pretrained", "trained": tmp_path / "trained"}
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(roots["pretrained"]))
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(roots["trained"]))
    return roots


@pytest.fixture(name="snapshot")
def snapshot_fixture(roots):
    """Write a small snapshot at ``REPO`` and ``REVISION``, and return its folder."""
    folder = roots["pretrained"] / "answerdotai--ModernBERT-base" / REVISION
    folder.mkdir(parents=True)
    for name, content in SNAPSHOT_FILES.items():
        (folder / name).write_bytes(content)
    return folder


@pytest.fixture(name="fold_models")
def fold_models_fixture(roots):
    """Write two fold models of different sizes under ``method/`` in the trained cache."""
    folder = roots["trained"] / "method"
    for size, file in enumerate(FOLD_FILES.values(), start=3):
        (folder / file).parent.mkdir(parents=True)
        (folder / file).write_bytes(b"x" * size)
    return folder


def test_missing_weights_are_a_missing_file_and_changed_weights_a_wrong_value():
    assert issubclass(MissingWeightsError, FileNotFoundError)
    assert issubclass(ChangedWeightsError, ValueError)


def test_the_trained_cache_is_its_environment_variable_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(tmp_path / "a"))
    assert trained_root() == tmp_path / "a"

    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(tmp_path / "b"))

    assert trained_root() == tmp_path / "b"


def test_the_trained_cache_defaults_to_the_config_without_its_environment_variable(monkeypatch):
    monkeypatch.delenv(TRAINED_DIR_ENV_VAR, raising=False)

    assert trained_root() == TRAINED_CACHE_DIR


def test_an_empty_trained_cache_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, "")

    assert trained_root() == TRAINED_CACHE_DIR


def test_a_pretrained_weight_list_hashes_each_file_under_its_sorted_name(snapshot):
    weight_list = pretrained_weight_list(REPO, REVISION, ["tokenizer.json", "config.json"])

    assert weight_list == {
        "repo_id": REPO,
        "revision": REVISION,
        "files": {
            "config.json": file_sha256(snapshot / "config.json"),
            "tokenizer.json": file_sha256(snapshot / "tokenizer.json"),
        },
    }
    assert list(weight_list["files"]) == ["config.json", "tokenizer.json"]


def test_a_pretrained_weight_list_names_the_fetch_command_for_missing_files(snapshot):
    files = ["config.json", "vocab.txt", "merges.txt"]

    with pytest.raises(MissingWeightsError) as raised:
        pretrained_weight_list(REPO, REVISION, files)

    message = str(raised.value)
    assert message.startswith(f"{snapshot} lacks ['merges.txt', 'vocab.txt']: ")
    assert f"run `{COMMAND} fetch --method <name>` on this machine first." in message


def test_a_trained_weight_list_gives_each_fold_its_file_hash_and_bytes(fold_models):
    weight_list = trained_weight_list("method", FOLD_FILES)

    assert weight_list == {
        "root": "method",
        "folds": {
            fold: {
                "file": file,
                "sha256": file_sha256(fold_models / file),
                "bytes": size,
            }
            for (fold, file), size in zip(FOLD_FILES.items(), [3, 4], strict=True)
        },
    }


def test_a_trained_weight_list_says_to_train_or_adopt_for_missing_files(roots):
    with pytest.raises(MissingWeightsError) as raised:
        trained_weight_list("method", FOLD_FILES)

    message = str(raised.value)
    assert message.startswith(f"{roots['trained'] / 'method'} lacks ")
    assert f"run `{COMMAND} train` or `adopt --method method` first." in message


def test_check_pretrained_returns_the_snapshot_folder_of_an_unchanged_weight_list(snapshot):
    weight_list = pretrained_weight_list(REPO, REVISION, SNAPSHOT_FILES)

    assert check_pretrained(json.loads(json.dumps(weight_list))) == snapshot


def test_check_pretrained_names_the_fetch_command_for_a_missing_file(snapshot):
    weight_list = pretrained_weight_list(REPO, REVISION, SNAPSHOT_FILES)
    (snapshot / "model.safetensors").unlink()

    with pytest.raises(MissingWeightsError, match=re.escape(f"`{COMMAND} fetch --method <name>`")):
        check_pretrained(weight_list)


def test_check_pretrained_refuses_a_changed_file(snapshot):
    weight_list = pretrained_weight_list(REPO, REVISION, SNAPSHOT_FILES)
    (snapshot / "config.json").write_bytes(b'{"a": 2}')

    with pytest.raises(ChangedWeightsError, match=re.escape(f"['config.json'] in {snapshot}")):
        check_pretrained(weight_list)


@pytest.mark.usefixtures("snapshot")
def test_check_pretrained_refuses_a_revision_that_is_not_one_folder_name():
    weight_list = pretrained_weight_list(REPO, REVISION, SNAPSHOT_FILES)

    with pytest.raises(ValueError, match="^The revision .* not one folder") as raised:
        check_pretrained({**weight_list, "revision": f"../{REVISION}"})
    assert raised.type is ValueError


def test_check_trained_returns_each_folds_weights_in_the_weight_lists_order(fold_models):
    weight_list = trained_weight_list("method", FOLD_FILES)

    paths = check_trained(json.loads(json.dumps(weight_list)))

    assert paths == {fold: fold_models / file for fold, file in FOLD_FILES.items()}
    assert list(paths) == ["fold_1", "fold_0"]


def test_check_trained_says_to_copy_the_trained_cache_for_a_missing_file(fold_models):
    weight_list = trained_weight_list("method", FOLD_FILES)
    (fold_models / FOLD_FILES["fold_0"]).unlink()

    with pytest.raises(MissingWeightsError) as raised:
        check_trained(weight_list)

    message = str(raised.value)
    assert message.startswith(f"{fold_models} lacks ['fold_0/weights.safetensors']: ")
    assert "copy method/ from the trained cache of the machine that froze the method" in message
    assert f"`{COMMAND} adopt --method method` restores it" in message


def test_check_trained_refuses_a_changed_file(fold_models):
    weight_list = trained_weight_list("method", FOLD_FILES)
    (fold_models / FOLD_FILES["fold_1"]).write_bytes(b"yyy")  # same size, other bytes

    with pytest.raises(ChangedWeightsError, match=re.escape("['fold_1/weights.safetensors']")):
        check_trained(weight_list)


@pytest.fixture(name="method_folder")
def method_folder_fixture(tmp_path, snapshot, fold_models):  # pylint: disable=unused-argument
    """Write the weight lists of a method folder whose weights (``snapshot``) are in the caches."""
    folder = tmp_path / "model"
    folder.mkdir()
    pretrained = pretrained_weight_list(REPO, REVISION, SNAPSHOT_FILES)
    (folder / PRETRAINED_FILE).write_text(json.dumps(pretrained), encoding="utf-8")
    trained = trained_weight_list(fold_models.name, FOLD_FILES)
    (folder / TRAINED_FILE).write_text(json.dumps(trained), encoding="utf-8")
    return folder


def test_weight_caches_name_the_weights_that_a_method_folders_weight_lists_name(
    method_folder, tmp_path
):
    (tmp_path / "plain").mkdir()

    assert weight_caches(method_folder) == ["pretrained weights", "trained weights"]
    assert not weight_caches(tmp_path / "plain")
    assert not missing_caches(tmp_path / "plain", "lora")


def test_missing_caches_say_how_to_fill_each_cache_that_lacks_a_file_without_hashing(
    method_folder, snapshot, fold_models
):
    assert not missing_caches(method_folder, "lora")
    (snapshot / "config.json").write_bytes(b"changed")  # hashing is loading's job
    assert not missing_caches(method_folder, "lora")

    (snapshot / "model.safetensors").unlink()
    (fold_models / FOLD_FILES["fold_0"]).unlink()

    assert missing_caches(method_folder, "lora") == [
        f"the pretrained weights of {REPO} in {snapshot} (fetch them: `{COMMAND} fetch "
        "--method lora`)",
        f"the trained weights in {fold_models} (copy method/ from the trained cache of the "
        "machine that froze the method, docs/adr/0006)",
    ]


def test_missing_caches_read_an_f2llm_weight_list_and_name_the_given_command(snapshot, tmp_path):
    # Regression: the command line checks every method folder, and an F2LLM method's
    # pretrained.json lists {"path", "sha256"} entries, which missing_caches iterated as names.
    folder = tmp_path / "model"
    folder.mkdir()
    files = [{"path": name, "sha256": "0"} for name in sorted(SNAPSHOT_FILES)]
    pretrained = {"repo_id": REPO, "revision": REVISION, "files": files}
    (folder / PRETRAINED_FILE).write_text(json.dumps(pretrained), encoding="utf-8")
    command = "uv run --group f2llm python -m codeforces_tags.methods.f2llm"

    assert not missing_caches(folder, "probe", command=command)

    (snapshot / "tokenizer.json").unlink()

    assert missing_caches(folder, "probe", command=command) == [
        f"the pretrained weights of {REPO} in {snapshot} (fetch them: `{command} fetch "
        "--method probe`)"
    ]
