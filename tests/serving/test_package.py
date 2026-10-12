"""Tests for packaging a frozen tagging method as the build context of its image.

Packaging reads committed method folders (label_prior's and tfidf's, and modernbert_lora's and
f2llm_late_fusion's for the missing-weights refusals), never the dataset, and writes under a build
folder of ``tmp_path``. git is faked (a test sets what it answers), except where a test runs it on
a repository of ``tmp_path`` (``mirror``), and so is ``uv export``; weights are written into
caches under ``tmp_path``.
"""

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from mlflow.models import Model

import codeforces_tags
from codeforces_tags.config import (
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    PRETRAINED_DIR_ENV_VAR,
    PROJECT_ROOT,
    PYPROJECT_FILE,
    RAW_INPUT_COLUMNS,
    TRAINED_DIR_ENV_VAR,
    UV_LOCK_FILE,
)
from codeforces_tags.conformance import PICKLE_SUFFIXES, starts_like_pickle
from codeforces_tags.methods import METHODS, load_method, model_dir
from codeforces_tags.pretrained import pinned_files, snapshot_folder
from codeforces_tags.provenance import git_state, paths_dirty
from codeforces_tags.serving import package
from codeforces_tags.serving.package import (
    DOCKERFILE,
    PACKAGE_DIR,
    SERVING_ENTRY,
    SERVING_GROUP,
    PackagingError,
    copy_dockerfile,
    lock_requirements,
    package_method,
    run_command,
    served_signature,
    short_sha,
    stage_weights,
)
from codeforces_tags.serving.records import served_records
from codeforces_tags.tracking.methods import CODE_MODEL_LOADER

# The commit the fake git has checked out.
COMMIT = "c0ffee" + "0" * 34
# Every input of label_prior's build context that git checks, as packaging names them.
LABEL_PRIOR_INPUTS = [
    PACKAGE_DIR,
    PYPROJECT_FILE,
    UV_LOCK_FILE,
    DOCKERFILE,
    METHODS_ASSETS_DIR / "label_prior" / "model",
]
# The files of a uv project, as the working directory may hold them: with MLflow's defaults,
# save_model copies them into the model whatever its requirements say (docs/adr/0002,
# safeguard 3).
DECOYS = {
    "uv.lock": "version = 1\n",
    "pyproject.toml": '[project]\nname = "decoy"\nversion = "0"\n',
    ".python-version": "3.12\n",
}
# Everything a packaged model holds at its top level.
MODEL_TOP_LEVEL = [
    "MLmodel",
    "artifacts",
    "code",
    "conda.yaml",
    "entry.py",
    "python_env.yaml",
    "requirements.txt",
]
# The project's dependencies (pyproject.toml), which every packaged model pins first.
PROJECT_DEPENDENCIES = (
    "iterative-stratification",
    "jinja2",
    "numpy",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "skops",
)
# The outputs that a score method adds to the method and its focus tags, then those of a method
# that calls an API.
SCORE_OUTPUTS = [
    *({"type": "double", "name": f"score: {tag}", "required": True} for tag in FOCUS_TAGS),
    *({"type": "double", "name": f"threshold: {tag}", "required": True} for tag in FOCUS_TAGS),
]
API_OUTPUTS = [
    {"type": "boolean", "name": "failed", "required": True},
    {"type": "double", "name": "cost_usd", "required": True},
]
# What the fake `uv export` prints: one requirement, pinned with its hash.
LOCKED = "mlflow==3.16.1 \\\n    --hash=sha256:" + "0" * 64 + "\n"
# The command that exports the requirements of a method without dependency groups: uv.lock as
# it is, without the default groups and the project itself, in pip's format, with the server's.
UV_EXPORT = [
    *("uv", "export", "--frozen", "--no-default-groups", "--no-emit-project"),
    *("--format", "requirements-txt", "--group", "serving"),
]
# What the weight caches hold, by path in their cache: a pretrained snapshot with a file at its
# top and one in a subfolder (as F2LLM's has), and the weights of one fold model.
PRETRAINED = {
    "org--encoder/0123abcd/config.json": b'{"hidden_size": 8}\n',
    "org--encoder/0123abcd/1_Pooling/config.json": b'{"pooling_mode": "mean"}\n',
}
TRAINED = {"weighted_fake/fold_0/weights.safetensors": b"the weights of fold 0\n"}
# The same files in the weights/ folder of a build context, in the order they are staged.
STAGED = {
    **{f"pretrained/{name}": data for name, data in PRETRAINED.items()},
    **{f"trained/{name}": data for name, data in TRAINED.items()},
}
# Names of a fold model's file that would stage it outside weights/: three folders up from
# weights/trained/weighted_fake, and an absolute path. From the trained cache's folder
# weighted_fake (tmp_path/cache/trained/weighted_fake), both name tmp_path/escaped.safetensors.
ESCAPING = {
    "dotdot": lambda tmp_path: "../../../escaped.safetensors",
    "absolute": lambda tmp_path: str(tmp_path / "escaped.safetensors"),
}
# Loads a packaged model as a server does, in a fresh interpreter, predicts the raw problems of
# a JSON file (pandas' "table" orient), then prints where codeforces_tags was imported from and
# the predictions, with their index, as JSON.
LOAD_AND_PREDICT = """
import json, sys
import mlflow
import pandas as pd
model = mlflow.pyfunc.load_model(sys.argv[1])
predicted = model.predict(pd.read_json(sys.argv[2], orient="table"))
package = sys.modules["codeforces_tags"].__file__
print(json.dumps({"package": package, "predicted": predicted.reset_index().to_dict("list")}))
"""


def sha256(data):
    """Return the sha256 hex digest of some bytes."""
    return hashlib.sha256(data).hexdigest()


def as_entries(hashes):
    """List file hashes as the F2LLM methods' ``pretrained.json`` does: path and sha256."""
    return [{"path": name, "sha256": digest} for name, digest in hashes.items()]


def pin_weights(folder, caches, shape=dict):
    """Fill the caches with ``PRETRAINED`` and ``TRAINED``, and pin them in a method folder.

    Args:
        folder: The method folder that gains ``pretrained.json`` and ``trained.json``.
        caches: The ``caches`` fixture.
        shape: How ``pretrained.json`` lists the hashes of its files: ``dict`` (the ModernBERT
            methods' ``{name: sha256}``) or ``as_entries`` (the F2LLM methods' list).
    """
    for cache, held in ((caches.pretrained, PRETRAINED), (caches.trained, TRAINED)):
        for name, data in held.items():
            (cache / name).parent.mkdir(parents=True, exist_ok=True)
            (cache / name).write_bytes(data)
    snapshot = {name.split("/", 2)[2]: sha256(data) for name, data in PRETRAINED.items()}
    pretrained = {"repo_id": "org/encoder", "revision": "0123abcd", "files": shape(snapshot)}
    [(name, data)] = TRAINED.items()
    fold = {"file": name.split("/", 1)[1], "sha256": sha256(data), "bytes": len(data)}
    trained = {"root": "weighted_fake", "folds": {"fold_0": fold}}
    (folder / "pretrained.json").write_text(json.dumps(pretrained), encoding="utf-8")
    (folder / "trained.json").write_text(json.dumps(trained), encoding="utf-8")


def name_fold_file(folder, name, tmp_path):
    """Name the fold model's file ``name`` in ``trained.json``, and write the file it names.

    Args:
        folder: A method folder that ``pin_weights`` filled.
        name: The file's name in ``trained.json``, one of ``ESCAPING``'s.
        tmp_path: The test's folder, whose ``escaped.safetensors`` gets ``TRAINED``'s bytes, so
            that the cache holds the file (``missing_caches``) and its hash is the pinned one.
    """
    weight_list = json.loads((folder / "trained.json").read_text(encoding="utf-8"))
    weight_list["folds"]["fold_0"]["file"] = name
    (folder / "trained.json").write_text(json.dumps(weight_list), encoding="utf-8")
    [data] = TRAINED.values()
    (tmp_path / "escaped.safetensors").write_bytes(data)


def files(root):
    """Return each file under ``root`` but those of a ``__pycache__``, by path, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    }


@pytest.fixture(name="build_dir", autouse=True)
def build_dir_fixture(tmp_path, monkeypatch):
    """Point ``SERVING_BUILD_DIR`` at a folder of ``tmp_path``, and return it resolved."""
    folder = tmp_path / "build" / "serving"
    monkeypatch.setattr(package, "SERVING_BUILD_DIR", folder)
    return folder.resolve()


@pytest.fixture(name="git", autouse=True)
def git_fixture(monkeypatch):
    """Stand in for git: ``COMMIT`` checked out, with no change unless a test sets some.

    Returns:
        A namespace: ``changes``, the lines of ``git status --porcelain`` that ``paths_dirty``
        answers; ``asked``, the paths of each call.
    """
    fake = SimpleNamespace(changes=[], asked=[])

    def fake_paths_dirty(paths):
        fake.asked.append(list(paths))
        return list(fake.changes)

    monkeypatch.setattr(package, "git_state", lambda: {"commit": COMMIT, "code_dirty": False})
    monkeypatch.setattr(package, "paths_dirty", fake_paths_dirty)
    return fake


@pytest.fixture(name="mirror")
def mirror_fixture(tmp_path, monkeypatch):
    """Run packaging's git on a repository of ``tmp_path`` that mirrors label_prior's inputs.

    The repository commits one file where each of ``LABEL_PRIOR_INPUTS`` lies in the project,
    then every git command reads a user's ``status.showUntrackedFiles=no``, which hides every
    untracked file from a plain ``git status``. ``git_state`` and ``paths_dirty`` run the real
    git there, each path packaging asks about mapped to its place in the mirror.

    Returns:
        The mirror's root.
    """
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)
    root = tmp_path / "mirror"
    for path in LABEL_PRIOR_INPUTS:
        placed = root / path.relative_to(PROJECT_ROOT)
        committed = placed / "committed.txt" if path.is_dir() else placed
        committed.parent.mkdir(parents=True, exist_ok=True)
        committed.write_text("committed\n", encoding="utf-8")
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    identity += ["-c", "commit.gpgsign=false"]
    for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "Inputs"]):
        subprocess.run(["git", *identity, *command], cwd=root, check=True, capture_output=True)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "status.showUntrackedFiles")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "no")
    monkeypatch.setattr(package, "git_state", lambda: git_state(root))
    monkeypatch.setattr(
        package,
        "paths_dirty",
        lambda paths: paths_dirty([root / path.relative_to(PROJECT_ROOT) for path in paths], root),
    )
    return root


@pytest.fixture(name="commands", autouse=True)
def commands_fixture(monkeypatch):
    """Answer ``uv export`` with ``LOCKED``, run any other command, and record each one.

    Returns:
        The ``(command, cwd)`` of every call of ``package.run_command``, in order.
    """
    recorded = []

    def run(command, *, cwd=None):
        recorded.append((list(command), cwd))
        return LOCKED if list(command[:2]) == ["uv", "export"] else run_command(command, cwd=cwd)

    monkeypatch.setattr(package, "run_command", run)
    return recorded


@pytest.fixture(name="caches")
def caches_fixture(tmp_path, monkeypatch):
    """Point the pretrained and the trained weight caches at empty folders, and return them.

    Both are in ``tmp_path / "cache"``, two folders down, so that a weight list's ``..`` name that
    leaves a build context's ``weights/`` still names a file of ``tmp_path`` (``ESCAPING``).
    """
    cache = tmp_path / "cache"
    caches = SimpleNamespace(pretrained=cache / "pretrained", trained=cache / "trained")
    monkeypatch.setenv(PRETRAINED_DIR_ENV_VAR, str(caches.pretrained))
    monkeypatch.setenv(TRAINED_DIR_ENV_VAR, str(caches.trained))
    return caches


@pytest.fixture(name="previous_build")
def previous_build_fixture(build_dir):
    """Return a writer of an earlier build of a method, which returns the build folder's files."""

    def write(name):
        (build_dir / name / "model").mkdir(parents=True)
        (build_dir / name / "model" / "MLmodel").write_text("an earlier build\n", encoding="utf-8")
        (build_dir / name / "stale.txt").write_text("a file no build writes\n", encoding="utf-8")
        return files(build_dir)

    return write


@pytest.fixture(name="linked_build_dir")
def linked_build_dir_fixture(tmp_path, monkeypatch):
    """Make ``SERVING_BUILD_DIR`` a symbolic link to a folder elsewhere, as on another disk.

    Returns:
        A namespace: ``link``, the build folder as the settings name it; ``target``, the folder
        it links to, resolved.
    """
    target = tmp_path / "other_disk" / "serving"
    target.mkdir(parents=True)
    link = tmp_path / "repository" / "build" / "serving"
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(package, "SERVING_BUILD_DIR", link)
    return SimpleNamespace(link=link, target=target.resolve())


@pytest.mark.parametrize(
    ("name", "added"),
    [
        ("label_prior", []),
        ("tfidf", SCORE_OUTPUTS),
        ("descode2tag-json", API_OUTPUTS),
        ("jev-statement-code", SCORE_OUTPUTS + API_OUTPUTS),
    ],
)
def test_served_signature_takes_raw_strings_and_gives_the_served_columns(name, added):
    signature = served_signature(METHODS[name])

    assert signature.inputs.to_dict() == [
        {"type": "string", "name": column, "required": True} for column in RAW_INPUT_COLUMNS
    ]
    assert signature.outputs.to_dict() == [
        {"type": "string", "name": "method", "required": True},
        *({"type": "long", "name": tag, "required": True} for tag in FOCUS_TAGS),
        *added,
    ]


def test_run_command_returns_what_the_command_printed_in_the_folder_it_ran_in(tmp_path):
    printed = run_command([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path)

    assert printed.splitlines() == [str(tmp_path.resolve())]


def test_run_command_raises_with_what_a_failed_command_printed_on_its_error_output():
    with pytest.raises(subprocess.CalledProcessError) as failed:
        run_command([sys.executable, "-c", "import sys; sys.exit('no such image')"])

    assert (failed.value.returncode, failed.value.stderr) == (1, "no such image\n")


def test_short_sha_abbreviates_the_commit_checked_out():
    sha = short_sha()

    assert len(sha) >= 7
    assert git_state()["commit"].startswith(sha)


def test_package_writes_a_model_from_code_without_the_working_directorys_uv_files(
    tmp_path, monkeypatch, build_dir
):
    monkeypatch.chdir(tmp_path)
    for name, text in DECOYS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setenv("MLFLOW_UV_AUTO_DETECT", "true")
    monkeypatch.setenv("MLFLOW_LOG_UV_FILES", "true")

    out = package_method("label_prior")

    assert out == build_dir / "label_prior"
    assert sorted(path.name for path in (out / "model").iterdir()) == MODEL_TOP_LEVEL


def test_the_model_holds_the_package_entry_file_and_method_folder_byte_for_byte():
    model = package_method("tfidf") / "model"

    assert files(model / "code" / "codeforces_tags") == files(Path(codeforces_tags.__file__).parent)
    assert (model / "entry.py").read_bytes() == SERVING_ENTRY.read_bytes()
    assert files(model / "artifacts" / "model") == files(model_dir("tfidf"))


def test_the_model_holds_no_bytecode_and_no_pickle():
    model = package_method("tfidf") / "model"

    paths = list(model.rglob("*"))
    assert [path for path in paths if path.name == "__pycache__" or path.suffix == ".pyc"] == []
    assert [
        path
        for path in paths
        if path.is_file() and (path.suffix in PICKLE_SUFFIXES or starts_like_pickle(path))
    ] == []


def test_the_model_loads_from_code_and_records_its_method_commit_and_signature():
    model = Model.load(package_method("tfidf") / "model")

    assert model.flavors["python_function"]["loader_module"] == CODE_MODEL_LOADER
    assert model.metadata == {"method": "tfidf", "git_commit": COMMIT, "code_dirty": False}
    assert model.signature == served_signature(METHODS["tfidf"])


def test_the_model_pins_the_projects_dependencies_the_methods_requirements_and_mlflow():
    spec = dataclasses.replace(METHODS["label_prior"], requirements=("numpy", "threadpoolctl"))

    model = package_method("label_prior", registry={"label_prior": spec}) / "model"

    names = (*PROJECT_DEPENDENCIES, "threadpoolctl", "mlflow")
    assert (model / "requirements.txt").read_text(encoding="utf-8").split() == [
        f"{name}=={version(name)}" for name in names
    ]


@pytest.mark.parametrize("name", ["label_prior", "tfidf"])
def test_a_packaged_model_serves_from_its_own_code_in_a_fresh_interpreter(
    name, tmp_path, raw_problems
):
    model = package_method(name) / "model"
    problems = tmp_path / "problems.json"
    raw_problems[list(RAW_INPUT_COLUMNS)].to_json(problems, orient="table")
    # The image's environment: MLflow's variables, and no bytecode written into the model.
    env = os.environ | MLFLOW_ENV_DEFAULTS | MLFLOW_ENV | {"PYTHONDONTWRITEBYTECODE": "1"}

    printed = subprocess.run(
        [sys.executable, "-c", LOAD_AND_PREDICT, str(model), str(problems)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    served = json.loads(printed)
    predicted = pd.DataFrame(served["predicted"]).set_index("sample_id")
    method = load_method(model_dir(name))
    assert Path(served["package"]).is_relative_to(model / "code")
    pd.testing.assert_frame_equal(predicted, served_records(method, METHODS[name], raw_problems))
    own = method.predict(raw_problems)[list(FOCUS_TAGS)].rename_axis(columns=None)
    pd.testing.assert_frame_equal(predicted[list(FOCUS_TAGS)], own)
    assert not list(model.rglob("__pycache__"))


def test_package_replaces_an_earlier_build_of_which_no_file_survives(previous_build, build_dir):
    previous_build("label_prior")
    # The staging folder of a run that was killed while it wrote.
    staging = build_dir / ".label_prior.partial"
    staging.mkdir()
    (staging / "stale.txt").write_text("a file a killed run left\n", encoding="utf-8")

    out = package_method("label_prior")

    assert not (out / "stale.txt").exists()
    assert [path.name for path in build_dir.iterdir()] == ["label_prior"]
    assert Model.load(out / "model").metadata["method"] == "label_prior"


def test_package_writes_an_explicit_out_inside_the_build_folder(build_dir):
    out = package_method("label_prior", out=build_dir / "elsewhere" / ".." / "copy")

    assert out == build_dir / "copy"
    assert [path.name for path in build_dir.iterdir()] == ["copy"]
    assert Model.load(out / "model").metadata["method"] == "label_prior"


def test_package_refuses_an_unregistered_method():
    with pytest.raises(PackagingError, match="'nope' is not a registered method"):
        package_method("nope")


def test_package_refuses_a_method_that_is_not_frozen(tmp_path):
    with pytest.raises(PackagingError, match=r"tfidf is not frozen: .* holds no method\.json"):
        package_method("tfidf", assets_dir=tmp_path / "assets")


def test_package_refuses_a_method_folder_that_holds_a_pickle_before_deleting_anything(
    tmp_path, previous_build, build_dir
):
    # A gitignored file (*.pkl) that git status does not show, and the image would copy.
    folder = tmp_path / "assets" / "label_prior" / "model"
    shutil.copytree(model_dir("label_prior"), folder)
    (folder / "cache.pkl").write_bytes(b"")
    before = previous_build("label_prior")

    with pytest.raises(PackagingError, match=r"cannot be packaged: cache\.pkl in .* is a pickle"):
        package_method("label_prior", assets_dir=tmp_path / "assets")

    assert files(build_dir) == before


@pytest.mark.usefixtures("caches")
@pytest.mark.parametrize(
    ("name", "command"),
    [
        (
            "modernbert_lora",
            "uv run --group finetuning python -m codeforces_tags.methods.modernbert",
        ),
        ("f2llm_late_fusion", "uv run --group f2llm python -m codeforces_tags.methods.f2llm"),
    ],
    ids=["modernbert_lora", "f2llm_late_fusion"],
)
def test_package_refuses_missing_weights_and_leaves_the_previous_build_intact(
    name, command, previous_build, build_dir
):
    # A machine without the weights a method folder pins, e.g. modernbert_lora's.
    before = previous_build(name)

    with pytest.raises(PackagingError, match=re.escape(f"`{command} fetch --method {name}`")):
        package_method(name)

    assert files(build_dir) == before


def test_package_refuses_a_missing_trained_cache_before_deleting_anything(
    caches, previous_build, build_dir
):
    # Modernbert_lora's pretrained snapshot is here, its trained weights are not.
    # missing_caches looks for the files without hashing them, so empty ones stand in.
    weight_list = model_dir("modernbert_lora") / "pretrained.json"
    pinned = json.loads(weight_list.read_text(encoding="utf-8"))
    snapshot = snapshot_folder(pinned["repo_id"], pinned["revision"])
    for name in pinned_files(pinned):
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"")
    before = previous_build("modernbert_lora")
    message = re.escape(f"the trained weights in {caches.trained / 'modernbert_lora'}")

    with pytest.raises(PackagingError, match=message) as refused:
        package_method("modernbert_lora")

    assert "docs/adr/0006" in str(refused.value)
    assert "fetch --method" not in str(refused.value)
    assert files(build_dir) == before


@pytest.mark.parametrize(
    "relative",
    ["../elsewhere", "../serving-other", ".", "tfidf/../.."],
    ids=["outside", "a_sibling_with_its_prefix", "the_build_folder", "its_parent"],
)
def test_package_deletes_and_writes_only_a_folder_inside_the_build_folder(build_dir, relative):
    kept = [build_dir / "tfidf" / "kept.txt", build_dir.parent / "elsewhere" / "kept.txt"]
    for path in kept:
        path.parent.mkdir(parents=True)
        path.write_text("kept\n", encoding="utf-8")

    with pytest.raises(PackagingError, match="is not a folder of"):
        package_method("label_prior", out=build_dir / relative)

    assert [path.read_text(encoding="utf-8") for path in kept] == ["kept\n", "kept\n"]


def test_package_writes_through_a_build_folder_that_is_a_symbolic_link(linked_build_dir):
    # build/serving may link to a folder on another disk: the guard compares resolved paths.
    default = package_method("label_prior")
    explicit = package_method("label_prior", out=linked_build_dir.link / "copy")

    assert (default, explicit) == (
        linked_build_dir.target / "label_prior",
        linked_build_dir.target / "copy",
    )
    assert sorted(path.name for path in linked_build_dir.link.iterdir()) == ["copy", "label_prior"]
    assert Model.load(linked_build_dir.link / "label_prior" / "model").metadata["method"] == (
        "label_prior"
    )


@pytest.mark.parametrize(
    "out",
    [
        lambda link: link / ".." / "elsewhere",
        lambda link: link.parent / "elsewhere",
        lambda link: link.resolve().parent / "elsewhere",
        lambda link: link,
        lambda link: link / "label_prior" / "..",
    ],
    ids=[
        "escaping_the_link_with_dotdot",
        "beside_the_link",
        "beside_its_target",
        "the_link_itself",
        "the_build_folder_through_dotdot",
    ],
)
def test_package_refuses_a_folder_outside_a_linked_build_folder_and_deletes_nothing(
    linked_build_dir, out
):
    # The link resolves first, so link/.. is the target's parent, not the link's.
    kept = [
        linked_build_dir.target / "label_prior" / "kept.txt",
        linked_build_dir.target.parent / "elsewhere" / "kept.txt",
        linked_build_dir.link.parent / "elsewhere" / "kept.txt",
    ]
    for path in kept:
        path.parent.mkdir(parents=True)
        path.write_text("kept\n", encoding="utf-8")
    before = files(linked_build_dir.target.parent.parent)

    with pytest.raises(PackagingError, match="is not a folder of"):
        package_method("label_prior", out=out(linked_build_dir.link))

    assert files(linked_build_dir.target.parent.parent) == before


def test_package_refuses_a_method_whose_requirements_are_not_installed(previous_build, build_dir):
    spec = dataclasses.replace(METHODS["label_prior"], requirements=("surely-not-installed",))
    before = previous_build("label_prior")

    with pytest.raises(PackagingError, match="surely-not-installed is not installed"):
        package_method("label_prior", registry={"label_prior": spec})

    assert files(build_dir) == before


def test_package_refuses_uncommitted_code_and_leaves_the_previous_build_intact(
    git, previous_build, build_dir
):
    git.changes = [" M src/codeforces_tags/config.py", "?? src/codeforces_tags/stray.py"]
    named = "(M src/codeforces_tags/config.py; ?? src/codeforces_tags/stray.py)"
    before = previous_build("label_prior")

    with pytest.raises(PackagingError, match=re.escape(f"uncommitted changes {named}")):
        package_method("label_prior")

    assert files(build_dir) == before


def test_package_refuses_an_untracked_module_of_the_package_whatever_the_git_config(mirror):
    # model/code ships every file of the package, so an untracked one is an uncommitted input,
    # even where the user's git config hides it from git_state.
    stray = mirror / PACKAGE_DIR.relative_to(PROJECT_ROOT) / "stray.py"
    stray.write_text("print('never committed')\n", encoding="utf-8")

    with pytest.raises(PackagingError, match=re.escape("(?? src/codeforces_tags/stray.py)")):
        package_method("label_prior")
    model = Model.load(package_method("label_prior", allow_dirty=True) / "model")

    commit = git_state(mirror)
    assert commit["code_dirty"] is False  # the plain git status of git_state does not see it
    assert model.metadata == {
        "method": "label_prior",
        "git_commit": commit["commit"],
        "code_dirty": True,
    }


@pytest.mark.parametrize(
    "change",
    [
        " M deploy/Dockerfile",
        "M  assets/methods/label_prior/model/label_prior.json",
        " D assets/methods/label_prior/model/method.json",
        "?? assets/methods/label_prior/model/notes.json",
    ],
    ids=["modified_dockerfile", "staged_file", "deleted_file", "untracked_file"],
)
def test_package_refuses_an_uncommitted_change_to_the_dockerfile_or_the_method_folder(git, change):
    git.changes = [change]

    with pytest.raises(PackagingError, match=re.escape(f"({change.strip()})")):
        package_method("label_prior")

    assert git.asked == [LABEL_PRIOR_INPUTS]


def test_package_lets_uncommitted_inputs_through_when_allowed_and_says_so(git):
    git.changes = [" M deploy/Dockerfile"]

    model = Model.load(package_method("label_prior", allow_dirty=True) / "model")

    assert model.metadata == {"method": "label_prior", "git_commit": COMMIT, "code_dirty": True}


def test_package_keeps_the_previous_build_when_the_model_would_not_load_from_code(
    monkeypatch, previous_build, build_dir
):
    def save_model(path, **arguments):  # pylint: disable=unused-argument
        path.mkdir(parents=True)
        (path / "MLmodel").write_text(
            "flavors:\n  python_function:\n    loader_module: mlflow.pyfunc.model\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(package.mlflow.pyfunc, "save_model", save_model)
    before = previous_build("label_prior")

    with pytest.raises(PackagingError, match="loads with mlflow.pyfunc.model"):
        package_method("label_prior")

    assert files(build_dir) == before
    assert [path.name for path in build_dir.iterdir()] == ["label_prior"]


@pytest.mark.parametrize(("name", "groups"), [("tfidf", []), ("f2llm_late_fusion", ["f2llm"])])
def test_lock_requirements_exports_the_servers_and_the_methods_groups_from_uv_lock(
    name, groups, commands, tmp_path
):
    lock_requirements(METHODS[name], tmp_path / "requirements.lock.txt")

    exported = [*UV_EXPORT, *(option for group in groups for option in ("--group", group))]
    assert commands == [(exported, PROJECT_ROOT)]
    assert (tmp_path / "requirements.lock.txt").read_text(encoding="utf-8") == LOCKED


def test_the_serving_group_holds_mlflow_out_of_the_default_groups_and_uv_lock_locks_it():
    pyproject = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))
    lock = tomllib.loads(UV_LOCK_FILE.read_text(encoding="utf-8"))

    [project] = [entry for entry in lock["package"] if entry["name"] == "codeforces-tags"]

    assert pyproject["dependency-groups"][SERVING_GROUP] == ["mlflow>=3.16"]
    assert SERVING_GROUP not in pyproject["tool"]["uv"]["default-groups"]
    assert project["dev-dependencies"][SERVING_GROUP] == [{"name": "mlflow"}]


def test_copy_dockerfile_copies_a_recipe_byte_for_byte(tmp_path):
    recipe = tmp_path / "recipe"
    recipe.write_bytes(b"FROM scratch\r\nCOPY model /opt/ml/model\n")
    (tmp_path / "context").mkdir()

    copy = copy_dockerfile(tmp_path / "context", recipe)

    assert copy == tmp_path / "context" / "Dockerfile"
    assert copy.read_bytes() == recipe.read_bytes()


@pytest.mark.parametrize("shape", [dict, as_entries], ids=["modernbert", "f2llm"])
def test_stage_weights_copies_the_pinned_files_as_their_caches_lay_them_out(
    shape, caches, tmp_path
):
    (tmp_path / "method").mkdir()
    pin_weights(tmp_path / "method", caches, shape)

    copies = stage_weights(tmp_path / "method", tmp_path / "weights")

    assert [copy.relative_to(tmp_path / "weights").as_posix() for copy in copies] == list(STAGED)
    assert files(tmp_path / "weights") == STAGED


def test_stage_weights_refuses_a_cache_file_whose_hash_is_not_its_pin(caches, tmp_path):
    (tmp_path / "method").mkdir()
    pin_weights(tmp_path / "method", caches)
    (caches.trained / "weighted_fake" / "fold_0" / "weights.safetensors").write_bytes(b"other\n")

    with pytest.raises(PackagingError, match="weights.safetensors does not have the sha256"):
        stage_weights(tmp_path / "method", tmp_path / "weights")


def test_stage_weights_makes_an_empty_folder_for_a_method_without_weights(tmp_path):
    assert not stage_weights(model_dir("tfidf"), tmp_path / "weights")
    assert (tmp_path / "weights").is_dir()
    assert not list((tmp_path / "weights").iterdir())


def test_package_writes_the_build_context_of_the_image(commands):
    out = package_method("tfidf")

    assert sorted(path.name for path in out.iterdir()) == [
        "Dockerfile",
        "model",
        "requirements.lock.txt",
        "weights",
    ]
    assert (out / "Dockerfile").read_bytes() == DOCKERFILE.read_bytes()
    assert (out / "requirements.lock.txt").read_text(encoding="utf-8") == LOCKED
    assert not list((out / "weights").iterdir())
    assert commands == [(UV_EXPORT, PROJECT_ROOT)]


def test_package_stages_the_weights_that_a_method_folder_pins_beside_its_model(caches, tmp_path):
    folder = tmp_path / "assets" / "label_prior" / "model"
    shutil.copytree(model_dir("label_prior"), folder)
    pin_weights(folder, caches)

    out = package_method("label_prior", assets_dir=tmp_path / "assets")

    assert files(out / "weights") == STAGED
    assert files(out / "model" / "artifacts" / "model") == files(folder)


def test_package_keeps_the_previous_build_when_a_cache_file_is_not_its_pin(
    caches, tmp_path, previous_build, build_dir
):
    # missing_caches finds every file, so the hash is checked only while the weights are staged.
    folder = tmp_path / "assets" / "label_prior" / "model"
    shutil.copytree(model_dir("label_prior"), folder)
    pin_weights(folder, caches)
    (caches.trained / "weighted_fake" / "fold_0" / "weights.safetensors").write_bytes(b"other\n")
    before = previous_build("label_prior")

    with pytest.raises(PackagingError, match="weights.safetensors does not have the sha256"):
        package_method("label_prior", assets_dir=tmp_path / "assets")

    assert files(build_dir) == before
    assert [path.name for path in build_dir.iterdir()] == ["label_prior"]


def test_package_keeps_the_previous_build_when_uv_export_fails(
    monkeypatch, previous_build, build_dir
):
    def fail(command, *, cwd=None):  # pylint: disable=unused-argument
        raise subprocess.CalledProcessError(2, command, stderr="error: no group `serving`")

    monkeypatch.setattr(package, "run_command", fail)
    before = previous_build("tfidf")

    with pytest.raises(subprocess.CalledProcessError):
        package_method("tfidf")

    assert files(build_dir) == before
    assert [path.name for path in build_dir.iterdir()] == ["tfidf"]


@pytest.mark.parametrize("escaping", ESCAPING.values(), ids=ESCAPING)
def test_stage_weights_refuses_a_name_outside_its_folder_before_copying_anything(
    escaping, caches, tmp_path
):
    (tmp_path / "method").mkdir()
    pin_weights(tmp_path / "method", caches)
    name = escaping(tmp_path)
    name_fold_file(tmp_path / "method", name, tmp_path)
    before = files(tmp_path)

    with pytest.raises(PackagingError, match=re.escape(name)):
        stage_weights(tmp_path / "method", tmp_path / "context" / "weights")

    # Not even the pretrained files, which the weight lists name first, were copied.
    assert files(tmp_path) == before


@pytest.mark.parametrize("escaping", ESCAPING.values(), ids=ESCAPING)
def test_package_refuses_a_weight_name_outside_its_folder_and_keeps_the_previous_build(
    escaping, caches, tmp_path, previous_build, build_dir
):
    folder = tmp_path / "assets" / "label_prior" / "model"
    shutil.copytree(model_dir("label_prior"), folder)
    pin_weights(folder, caches)
    name = escaping(tmp_path)
    name_fold_file(folder, name, tmp_path)
    previous_build("label_prior")
    before = files(tmp_path)

    with pytest.raises(PackagingError, match="would land outside"):
        package_method("label_prior", assets_dir=tmp_path / "assets")

    assert files(tmp_path) == before
    assert [path.name for path in build_dir.iterdir()] == ["label_prior"]
