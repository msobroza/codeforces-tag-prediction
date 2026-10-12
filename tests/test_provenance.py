"""Tests for the fingerprints of code, data, environment and settings, and the run records."""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from codeforces_tags import config, provenance
from codeforces_tags.config import (
    DATA_DIR_ENV_VAR,
    DATASET_FILES,
    DATASET_SHA256,
    FOCUS_TAGS,
    NEAR_DUPLICATE_VERDICTS,
    PROJECT_ROOT,
    RECORDED_PACKAGES,
    RECORDED_SETTINGS,
    SPLIT_FILE,
    UV_LOCK_FILE,
)
from codeforces_tags.provenance import (
    RUN_RECORD_KEYS,
    RunRecord,
    canonical_name,
    check_clean_tree,
    check_dataset,
    committed_unchanged,
    committed_versions,
    dataset_sha256,
    environment,
    file_sha256,
    git_state,
    json_safe,
    last_commit,
    locked_versions,
    make_run_record,
    paths_dirty,
    settings_snapshot,
    sha256_json,
)

LOCK = """\
version = 1

[[package]]
name = "numpy"
version = "0.0.1"

[[package]]
name = "numpy"
version = "0.0.2"

[[package]]
name = "pandas"
version = "{pandas}"
"""


def sha256_text(text):
    """Return the sha256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(root, *args):
    """Run git in ``root`` with a fixed identity and no signing, and return its output."""
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    return subprocess.run(
        ["git", *identity, "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(name="repo")
def repo_fixture(tmp_path, monkeypatch):
    """Return a git repository whose code, lock and assets are committed."""
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)
    root = tmp_path / "repo"
    for name in ("src/module.py", "pyproject.toml", "uv.lock", "assets/predictions.csv"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("first\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "First")
    return root


@pytest.fixture(name="data_dir")
def data_dir_fixture(write_samples, make_record):
    """Return a dataset directory of two problems and a file that is not a problem."""
    data = write_samples({12: make_record(), 3: make_record(tags=["graphs"])})
    (data / "README.md").write_text("Not a problem.\n", encoding="utf-8")
    return data


def test_sha256_json_hashes_canonical_json():
    digest = sha256_json({"b": 1, "a": ["x", 0.5]})

    assert digest == sha256_text('{"a":["x",0.5],"b":1}')
    assert sha256_json({"a": ("x", 0.5), "b": 1}) == digest


def test_sha256_json_accepts_infinite_thresholds():
    thresholds = {"games": math.inf, "math": 0.25, "trees": -math.inf, "geometry": math.nan}

    digest = sha256_json({"thresholds": thresholds})

    assert digest == sha256_text(
        '{"thresholds":{"games":"inf","geometry":"nan","math":0.25,"trees":"-inf"}}'
    )
    assert sha256_json({"thresholds": thresholds | {"games": 0.5}}) != digest


def test_json_safe_turns_tuples_into_lists_and_non_finite_floats_into_strings():
    values = {"tags": ("math", "games"), "scores": [np.float64(np.inf), 1.5], "n": 2}

    safe = json_safe(values)

    assert safe == {"tags": ["math", "games"], "scores": ["inf", 1.5], "n": 2}
    assert values["tags"] == ("math", "games")
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe


def test_file_sha256_hashes_the_bytes_of_a_file(tmp_path):
    path = tmp_path / "file.csv"
    path.write_bytes(b"part,sample_id\ndev,1\n")

    assert file_sha256(path) == hashlib.sha256(b"part,sample_id\ndev,1\n").hexdigest()


def test_dataset_sha256_hashes_each_problem_file_name_and_bytes_in_name_order(data_dir):
    expected = hashlib.sha256()
    for name in ("sample_12.json", "sample_3.json"):
        expected.update(name.encode("utf-8") + b"\0" + (data_dir / name).read_bytes() + b"\0")

    assert dataset_sha256(data_dir) == expected.hexdigest()


def test_dataset_sha256_changes_with_a_byte_or_a_file_name(data_dir):
    before = dataset_sha256(data_dir)
    (data_dir / "sample_3.json").rename(data_dir / "sample_4.json")
    renamed = dataset_sha256(data_dir)
    path = data_dir / "sample_12.json"
    path.write_bytes(path.read_bytes().replace(b"Find", b"Fond"))

    assert len({before, renamed, dataset_sha256(data_dir)}) == 3


def test_dataset_sha256_reads_the_data_dir_of_the_environment(data_dir, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(data_dir))

    assert dataset_sha256() == dataset_sha256(data_dir)


def test_the_frozen_dataset_fingerprint_counts_the_task_statements_problems_and_is_a_sha256():
    # Whether it is the sha256 of the dataset itself needs the dataset: a method's freeze and
    # verify check it (check_dataset).
    statement = (PROJECT_ROOT / "PROBLEM.md").read_text(encoding="utf-8")

    assert f"holds {DATASET_FILES:,} problems" in statement
    assert re.fullmatch(r"[0-9a-f]{64}", DATASET_SHA256)


def test_check_dataset_accepts_the_frozen_dataset(data_dir, monkeypatch):
    monkeypatch.setattr(provenance, "DATASET_SHA256", dataset_sha256(data_dir))
    monkeypatch.setattr(provenance, "DATASET_FILES", 2)

    check_dataset(data_dir)


def test_check_dataset_refuses_another_dataset(data_dir, monkeypatch):
    monkeypatch.setattr(provenance, "DATASET_FILES", 2)

    with pytest.raises(ValueError, match="2 problem files and the sha256"):
        check_dataset(data_dir)


def test_check_dataset_refuses_a_dataset_with_another_file_count(data_dir, monkeypatch):
    monkeypatch.setattr(provenance, "DATASET_SHA256", dataset_sha256(data_dir))

    with pytest.raises(ValueError, match=f"{DATASET_FILES} files"):
        check_dataset(data_dir)


def test_git_state_names_the_commit_of_a_clean_tree(repo):
    assert git_state(repo) == {
        "commit": git(repo, "rev-parse", "HEAD").strip(),
        "code_dirty": False,
    }


def test_git_state_ignores_changes_outside_the_code(repo):
    (repo / "assets" / "predictions.csv").write_text("second\n", encoding="utf-8")
    (repo / "notebooks").mkdir()
    (repo / "notebooks" / "new.ipynb").write_text("{}\n", encoding="utf-8")

    assert git_state(repo)["code_dirty"] is False


@pytest.mark.parametrize("name", ["src/module.py", "src/new.py", "pyproject.toml", "uv.lock"])
def test_git_state_sees_a_changed_or_new_code_file(repo, name):
    (repo / name).write_text("second\n", encoding="utf-8")

    assert git_state(repo)["code_dirty"] is True


def test_git_state_of_the_project_names_a_commit():
    commit = git_state()["commit"]

    assert len(commit) == 40
    assert int(commit, 16) >= 0


def modify(repo):
    """Change a committed file of ``assets/``."""
    (repo / "assets" / "predictions.csv").write_text("second\n", encoding="utf-8")


def stage(repo):
    """Add a new file of ``assets/`` to the index, without committing it."""
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")
    git(repo, "add", "assets/scores.csv")


def delete(repo):
    """Delete a committed file of ``assets/``."""
    (repo / "assets" / "predictions.csv").unlink()


def add_untracked(repo):
    """Write a file into ``assets/`` that git does not track."""
    (repo / "assets" / "new.csv").write_text("first\n", encoding="utf-8")


def add_untracked_folder(repo):
    """Write a file into a new folder of ``assets/``, which git does not track either."""
    (repo / "assets" / "new").mkdir()
    (repo / "assets" / "new" / "weights.csv").write_text("first\n", encoding="utf-8")


def test_paths_dirty_of_clean_paths_is_empty(repo):
    assert paths_dirty(["assets", "uv.lock"], repo) == []


@pytest.mark.parametrize(
    ("change", "line"),
    [
        (modify, " M assets/predictions.csv"),
        (stage, "A  assets/scores.csv"),
        (delete, " D assets/predictions.csv"),
        (add_untracked, "?? assets/new.csv"),
        (add_untracked_folder, "?? assets/new/weights.csv"),  # the file, not "?? assets/new/"
    ],
)
def test_paths_dirty_lists_each_change_under_a_path_as_git_status_shows_it(repo, change, line):
    change(repo)

    assert paths_dirty(["assets"], repo) == [line]
    assert paths_dirty([repo / "assets", "uv.lock"], repo) == [line]


def test_paths_dirty_lists_an_untracked_file_whatever_the_git_config(repo, monkeypatch):
    # A user's status.showUntrackedFiles=no hides every untracked file from git status
    # --porcelain; GIT_CONFIG_COUNT sets it for every git command that runs from here on.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "status.showUntrackedFiles")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "no")
    add_untracked(repo)

    assert paths_dirty(["assets"], repo) == ["?? assets/new.csv"]


def test_paths_dirty_ignores_changes_outside_its_paths(repo):
    (repo / "src" / "module.py").write_text("second\n", encoding="utf-8")
    (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")

    assert paths_dirty(["assets", "uv.lock"], repo) == []
    assert paths_dirty([], repo) == []  # git status given no path would list the whole tree


def test_paths_dirty_refuses_a_root_outside_any_repository_or_a_path_outside_its_own(
    repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # git looks no higher
    (tmp_path / "plain").mkdir()

    with pytest.raises(subprocess.CalledProcessError):
        paths_dirty(["assets"], tmp_path / "plain")
    with pytest.raises(subprocess.CalledProcessError):
        paths_dirty([tmp_path / "plain"], repo)


def test_committed_unchanged_holds_for_a_tracked_clean_file(repo):
    assert committed_unchanged(repo / "assets" / "predictions.csv", repo)
    assert committed_unchanged("assets/predictions.csv", repo)


def test_committed_unchanged_fails_for_a_modified_file(repo):
    (repo / "assets" / "predictions.csv").write_text("second\n", encoding="utf-8")

    assert not committed_unchanged(repo / "assets" / "predictions.csv", repo)


def test_committed_unchanged_fails_for_an_untracked_or_missing_file(repo):
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")

    assert not committed_unchanged(repo / "assets" / "scores.csv", repo)
    assert not committed_unchanged(repo / "assets" / "missing.csv", repo)


def test_committed_unchanged_fails_for_a_staged_but_uncommitted_file(repo):
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")
    git(repo, "add", "assets/scores.csv")

    assert not committed_unchanged(repo / "assets" / "scores.csv", repo)


def test_last_commit_names_the_commit_that_last_changed_a_file(repo):
    first = git(repo, "rev-parse", "HEAD").strip()
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")
    git(repo, "add", "assets/scores.csv")
    git(repo, "commit", "-q", "-m", "Second")

    assert last_commit(repo / "assets" / "predictions.csv", repo) == first
    assert last_commit("assets/scores.csv", repo) == git(repo, "rev-parse", "HEAD").strip()


def test_last_commit_refuses_a_file_that_no_commit_changed(repo):
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No commit changed"):
        last_commit(repo / "assets" / "scores.csv", repo)


def test_committed_versions_reads_a_file_at_each_commit_that_changed_it(repo):
    path = repo / "assets" / "predictions.csv"
    first = git(repo, "rev-parse", "HEAD").strip()
    path.write_text("first\nsecond\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "Second")
    second = git(repo, "rev-parse", "HEAD").strip()
    (repo / "src" / "module.py").write_text("third\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "Third")  # leaves the file alone
    git(repo, "rm", "-q", "assets/predictions.csv")
    git(repo, "commit", "-q", "-m", "Fourth")
    fourth = git(repo, "rev-parse", "HEAD").strip()

    versions = [(first, b"first\n"), (second, b"first\nsecond\n"), (fourth, b"")]
    assert committed_versions(path, repo) == versions  # a deleted file reads as no bytes
    assert committed_versions("assets/predictions.csv", repo) == versions


def test_committed_versions_of_a_file_that_no_commit_changed_is_empty(repo):
    (repo / "assets" / "scores.csv").write_text("first\n", encoding="utf-8")

    assert committed_versions(repo / "assets" / "scores.csv", repo) == []


def test_check_clean_tree_accepts_a_clean_tree(repo):
    check_clean_tree(allow_dirty=False, root=repo)


def test_check_clean_tree_refuses_uncommitted_code(repo):
    (repo / "src" / "module.py").write_text("second\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        check_clean_tree(allow_dirty=False, root=repo)


def test_check_clean_tree_lets_uncommitted_code_through_when_allowed(repo):
    (repo / "src" / "module.py").write_text("second\n", encoding="utf-8")

    check_clean_tree(allow_dirty=True, root=repo)


def test_locked_versions_keep_every_version_a_package_is_locked_at(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(LOCK.format(pandas="3.0.6"), encoding="utf-8")

    assert locked_versions(lock) == {"numpy": ("0.0.1", "0.0.2"), "pandas": ("3.0.6",)}


def test_environment_records_python_platform_packages_lock_and_thread_pools():
    env = environment(["threadpoolctl", "scikit-learn"])

    assert env["python"] == platform.python_version()
    assert env["platform"] == platform.platform()
    assert list(env["packages"]) == [*RECORDED_PACKAGES, "threadpoolctl"]
    assert env["packages"]["numpy"] == np.__version__
    assert env["uv_lock_sha256"] == file_sha256(UV_LOCK_FILE)
    assert all("filepath" not in pool for pool in env["threadpools"])
    assert json.loads(json.dumps(env, allow_nan=False)) == env


def test_environment_lists_the_packages_whose_version_is_not_locked(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(LOCK.format(pandas=environment()["packages"]["pandas"]), encoding="utf-8")

    env = environment(["surely-not-an-installed-distribution"], lock=lock)

    mismatches = env["lock_mismatches"]
    assert mismatches["numpy"] == {"installed": np.__version__, "locked": ["0.0.1", "0.0.2"]}
    assert mismatches["surely-not-an-installed-distribution"] == {"installed": None, "locked": []}
    assert "pandas" not in mismatches
    assert env["uv_lock_sha256"] == file_sha256(lock)


def test_environment_records_torch_only_when_a_method_imported_it(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert environment()["torch"] is None

    cuda = SimpleNamespace(device_count=lambda: 1, get_device_name=lambda index: f"GPU {index}")
    fake = SimpleNamespace(__version__="2.9.0", version=SimpleNamespace(cuda="12.8"), cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert environment()["torch"] == {"version": "2.9.0", "cuda": "12.8", "devices": ["GPU 0"]}


def test_settings_snapshot_holds_the_recorded_settings_as_json_values():
    snapshot = settings_snapshot()

    assert tuple(snapshot) == RECORDED_SETTINGS
    assert snapshot["FOCUS_TAGS"] == list(FOCUS_TAGS)
    assert snapshot["MODEL_SEED"] == config.MODEL_SEED


def test_run_record_keys_are_the_field_names_in_order():
    assert tuple(field.name for field in fields(RunRecord)) == RUN_RECORD_KEYS
    assert RUN_RECORD_KEYS[:2] == ("method", "recipe_sha256")
    assert RUN_RECORD_KEYS[-2:] == ("prompts", "method_entries")


def test_a_run_record_survives_a_json_round_trip(placeholder_record):
    entries = {"thresholds": {"games": math.inf, "math": 0.25}, "tags": ("math", "games")}
    record = placeholder_record(method_entries=entries, prompts=("prompts:/code2tag/1",))

    text = json.dumps(record.to_dict(), allow_nan=False)

    assert RunRecord.from_dict(json.loads(text)) == record
    assert record.method_entries == {
        "thresholds": {"games": "inf", "math": 0.25},
        "tags": ["math", "games"],
    }
    assert record.prompts == ("prompts:/code2tag/1",)
    assert list(record.to_dict()) == list(RUN_RECORD_KEYS)


def test_run_record_from_dict_refuses_missing_or_unknown_keys(placeholder_record):
    values = placeholder_record().to_dict()
    del values["created_at"]

    with pytest.raises(ValueError, match=r"missing \['created_at'\], unknown \['extra'\]"):
        RunRecord.from_dict(values | {"extra": 1})
    with pytest.raises(ValueError, match=r"missing \[\], unknown \['extra'\]"):
        RunRecord.from_dict(placeholder_record().to_dict() | {"extra": 1})


def test_run_record_refuses_one_prompt_given_as_a_str(placeholder_record):
    with pytest.raises(TypeError, match="prompts"):
        placeholder_record(prompts="prompts:/code2tag/1")


def test_make_run_record_fingerprints_code_data_split_environment_and_settings(data_dir):
    record = make_run_record(
        "label_prior",
        "f" * 64,
        allow_dirty=True,
        requirements=["threadpoolctl"],
        prompts=["prompts:/code2tag/1"],
        method_entries={"supersedes": []},
        data_dir=data_dir,
    )

    assert (record.method, record.recipe_sha256, record.allow_dirty) == (
        "label_prior",
        "f" * 64,
        True,
    )
    assert record.git["commit"] == git_state()["commit"]
    assert record.environment["packages"]["threadpoolctl"] is not None
    assert record.dataset_sha256 == dataset_sha256(data_dir)
    assert record.split_sha256 == file_sha256(SPLIT_FILE)
    assert record.verdicts_sha256 == file_sha256(NEAR_DUPLICATE_VERDICTS)
    assert record.settings == settings_snapshot()
    assert record.prompts == ("prompts:/code2tag/1",)
    assert record.method_entries == {"supersedes": []}
    assert datetime.fromisoformat(record.created_at).utcoffset().total_seconds() == 0


def test_make_run_record_fingerprints_the_split_it_is_given(data_dir, tmp_path):
    # A freeze run with --split another.csv records the split its predictions came from.
    split_file = tmp_path / "split.csv"
    split_file.write_text("sample_id,part,fold,group\n1,dev,0,1\n", encoding="utf-8")

    record = make_run_record(
        "label_prior", "f" * 64, allow_dirty=False, data_dir=data_dir, split_path=split_file
    )

    assert record.split_sha256 == file_sha256(split_file) != file_sha256(SPLIT_FILE)


def test_make_run_record_defaults_to_no_prompts_and_no_method_entries(data_dir):
    record = make_run_record("label_prior", "f" * 64, allow_dirty=False, data_dir=data_dir)

    assert (record.prompts, record.method_entries) == ((), {})


def test_canonical_name_normalises_a_distribution_name_as_uv_lock_writes_it():
    assert canonical_name("Scikit_Learn") == "scikit-learn"
    assert canonical_name("zope.interface") == "zope-interface"
