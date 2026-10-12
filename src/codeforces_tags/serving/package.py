"""Package a frozen tagging method as the build context of its image: model, weights, lock.

``package_method`` turns a committed method folder (``assets/methods/<name>/model``) into
``build/serving/<name>/`` (docs/adr/0012), the folder that ``docker build`` reads:

- ``model/``: an MLflow model saved from code (``mlflow.pyfunc.save_model``): the deployment
  entry file ``entry.py`` (``SERVING_ENTRY``), the method folder as its artifact ``method``, the
  package ``codeforces_tags`` under ``code/`` (MLflow puts that folder first on ``sys.path``
  when the model loads, so a server installs no wheel of the project), the signature of the
  served records (``served_signature``), and explicit requirements pinned from ``uv.lock``, with
  no ``uv.lock`` of its own;
- ``weights/``: the files that the folder's weight lists pin, copied from their caches and
  hashed again (``stage_weights``), laid out as the caches are, so that the image points
  ``$CODEFORCES_PRETRAINED_DIR`` and ``$CODEFORCES_TRAINED_DIR`` at its two subfolders; empty
  for a method without weights;
- ``requirements.lock.txt``: what the image installs, ``uv.lock``'s pins of the project's
  dependencies, of the ``serving`` group and of the method's groups, with their hashes
  (``lock_requirements``);
- ``Dockerfile``: a copy of ``deploy/Dockerfile`` (``copy_dockerfile``).

It refuses, before anything is deleted or written, a method that is not registered or not
frozen, whose method folder holds anything but data files (``check_method_folder``, as
``log_method`` does), whose weight caches miss a file, whose requirements are not installed at
the versions ``uv.lock`` pins, or whose inputs hold uncommitted changes (``--allow-dirty`` lets
them through, and the model's metadata then says ``code_dirty``). It writes only under
``SERVING_BUILD_DIR``: into a staging folder beside the build context, which replaces the
context only once it is complete, so that a failure leaves the previous build as it was.

Every ``uv`` and ``docker`` command of the serving commands, and ``short_sha``'s ``git
rev-parse``, runs through ``run_command``, the seam the tests fake; ``git_state`` and
``paths_dirty`` (``provenance``) run their own git, and the tests replace them by name. This
module imports mlflow at module level: the commands run it, never a prediction.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow
from mlflow.models import Model, ModelSignature
from mlflow.types import ColSpec, DataType, Schema

from codeforces_tags.cli_methods import package_command
from codeforces_tags.config import (
    FOCUS_TAGS,
    METHODS_ASSETS_DIR,
    PROJECT_ROOT,
    PYPROJECT_FILE,
    SERVING_BUILD_DIR,
    UV_LOCK_FILE,
)
from codeforces_tags.conformance import ConformanceError, check_method_folder
from codeforces_tags.methods import METHOD_FILE, METHODS, MODEL_FOLDER, MethodSpec
from codeforces_tags.methods.modernbert.settings import PRETRAINED_FILE, TRAINED_FILE
from codeforces_tags.methods.modernbert.weights import missing_caches, trained_root
from codeforces_tags.pretrained import pinned_files, pretrained_root, snapshot_folder
from codeforces_tags.provenance import canonical_name, file_sha256, git_state, paths_dirty
from codeforces_tags.serving.records import (
    COST_COLUMN,
    FAILED_COLUMN,
    METHOD_COLUMN,
    score_column,
    threshold_column,
)
from codeforces_tags.tracking.environment import mlflow_env
from codeforces_tags.tracking.methods import (
    CODE_MODEL_LOADER,
    method_requirements,
    raw_input_signature,
)

# The deployment entry file, by its path: MLflow copies it into the model and runs the copy when
# the model loads. Never imported here.
SERVING_ENTRY = Path(__file__).with_name("entry.py")
# The package that MLflow copies into the model's code/ folder, __pycache__ left out.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
# The image's recipe: an uncommitted change to it makes the build context dirty.
DOCKERFILE = PROJECT_ROOT / "deploy" / "Dockerfile"
# The folders and the file of a build context, beside its Dockerfile: the MLflow model, the
# weights (the pretrained snapshots and the trained weights, each laid out as its cache is) and
# the image's requirements.
MODEL_DIR = "model"
WEIGHTS_DIR = "weights"
PRETRAINED_WEIGHTS = "pretrained"
TRAINED_WEIGHTS = "trained"
LOCK_FILE = "requirements.lock.txt"
# The dependency group of the model server, exported into every image with the method's groups.
SERVING_GROUP = "serving"
# A bare distribution name at the start of a requirement such as "numpy>=2.0".
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class PackagingError(RuntimeError):
    """A method this machine cannot package: the command reports it on one line, status 1."""


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a command and return what it printed: the seam that the tests fake.

    Args:
        command: The program and its arguments, e.g. ``["git", "rev-parse", "HEAD"]``.
        cwd: The folder it runs in; the current one when None.

    Returns:
        Its standard output, as text.

    Raises:
        subprocess.CalledProcessError: If it exits with a status other than 0; the error's
            ``stderr`` holds what it printed there.
    """
    completed = subprocess.run(list(command), cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def short_sha(root: Path = PROJECT_ROOT) -> str:
    """Name the commit checked out, as the tag of a method's image names it.

    Args:
        root: The repository.

    Returns:
        ``git rev-parse --short HEAD``: the abbreviated sha of ``HEAD``.

    Raises:
        subprocess.CalledProcessError: If ``root`` is not in a git repository with a commit.
    """
    return run_command(["git", "rev-parse", "--short", "HEAD"], cwd=root).strip()


def served_signature(spec: MethodSpec) -> ModelSignature:
    """Return the signature of a served method: raw strings in, the served records out.

    Args:
        spec: The method's spec.

    Returns:
        The inputs of the logged models (``raw_input_signature``: one required string per input
        column of ``spec``), and the outputs ``method`` (string) and one long per focus tag;
        then, for a method of kind ``score``, a double ``score: <tag>`` per focus tag, then a
        double ``threshold: <tag>`` per focus tag; then, for a method that calls an API, the
        boolean ``failed`` and the double ``cost_usd``.
    """
    columns = [(DataType.string, METHOD_COLUMN)]
    columns += [(DataType.long, tag) for tag in FOCUS_TAGS]
    if spec.kind == "score":
        columns += [(DataType.double, score_column(tag)) for tag in FOCUS_TAGS]
        columns += [(DataType.double, threshold_column(tag)) for tag in FOCUS_TAGS]
    if spec.calls_api:
        columns += [(DataType.boolean, FAILED_COLUMN), (DataType.double, COST_COLUMN)]
    outputs = Schema([ColSpec(kind, column, required=True) for kind, column in columns])
    inputs = raw_input_signature(FOCUS_TAGS, spec.input_columns).inputs
    return ModelSignature(inputs=inputs, outputs=outputs)


def package_method(
    name: str,
    out: Path | None = None,
    *,
    allow_dirty: bool = False,
    registry: Mapping[str, MethodSpec] = METHODS,
    assets_dir: Path = METHODS_ASSETS_DIR,
) -> Path:
    """Package a frozen method's committed folder as the build context of its image.

    It refuses first, before anything is deleted or written: a method that is not registered,
    not frozen, whose folder holds anything but data files (``check_method_folder``), or whose
    weight caches miss a file (``missing_caches``); a folder ``out`` that is not inside
    ``SERVING_BUILD_DIR``; requirements that are not installed at a version ``uv.lock`` pins
    (``method_requirements``); and, unless ``allow_dirty``, uncommitted changes to the package
    ``src/codeforces_tags``, ``pyproject.toml``, ``uv.lock``, ``deploy/Dockerfile`` or the method
    folder, an untracked file included (``paths_dirty``). Then it writes the build context into
    a staging folder beside ``out``, ``.<name of out>.partial``, emptied first: ``model/``,
    ``requirements.lock.txt`` (``lock_requirements``), ``Dockerfile`` (``copy_dockerfile``) and
    ``weights/`` (``stage_weights``), in that order. Only once all four are written does it
    delete ``out`` and rename the staging folder to ``out``, so that no file of an earlier build
    survives. If one fails (e.g. a cache file that is not its pin, or ``uv export``), it removes
    the staging folder alone: the previous build is left as it was.

    The model is saved from code (``SERVING_ENTRY``) with MLflow's environment set first
    (``mlflow_env``): with MLflow's defaults, ``save_model`` would copy the working directory's
    ``uv.lock``, ``pyproject.toml`` and ``.python-version`` into the model, whatever its
    requirements say (docs/adr/0002, safeguard 3). Its requirements pin the project's
    dependencies, the method's and mlflow; its metadata holds ``method``, ``git_commit`` and
    ``code_dirty``, whether any input differs from that commit. Its ``MLmodel`` names its files
    relative to its folder, so that it loads once renamed.

    Args:
        name: The method's registry key.
        out: The build context to write; ``SERVING_BUILD_DIR / name`` when None.
        allow_dirty: True to package uncommitted inputs (``--allow-dirty``).
        registry: The registered methods.
        assets_dir: The folder of the frozen methods.

    Returns:
        ``out``, resolved.

    Raises:
        PackagingError: If any refusal above applies, if the saved model does not load from
            code (``CODE_MODEL_LOADER``), if a weight list names a file outside its folder, or
            if a weight file copied differs from its pin.
        ValueError: If the method folder's ``pretrained.json`` is in neither shape that
            ``pinned_files`` reads (``missing_caches``).
        subprocess.CalledProcessError: If ``uv export`` fails (``lock_requirements``).
    """
    spec = _registered(name, registry)
    folder = _servable_folder(name, spec, Path(assets_dir))
    out = _build_folder(name, out)
    requirements = _pinned_requirements(name, spec)
    metadata = {"method": name, **_committed_inputs(folder, allow_dirty)}
    staging = out.with_name(f".{out.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _save_model(staging / MODEL_DIR, spec, folder, requirements, metadata)
        lock_requirements(spec, staging / LOCK_FILE)
        copy_dockerfile(staging)
        stage_weights(folder, staging / WEIGHTS_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return out


def stage_weights(method_folder: Path, weights_dir: Path) -> list[Path]:
    """Copy the weights a method folder pins into a build context, checking each copy's hash.

    The files of ``pretrained.json`` (either shape that ``pinned_files`` reads) go to
    ``weights_dir/pretrained/<org>--<repo>/<revision>/``, the layout of the pretrained cache
    (``snapshot_folder``), and those of ``trained.json`` to ``weights_dir/trained/<root>/``, the
    layout of the trained cache (``trained_root``): pointed at those two folders, the method
    loads them as it loads its caches. Before it copies anything, it refuses a name whose copy
    would land outside ``weights_dir`` (a ``..`` or an absolute name), so that it writes there
    only.

    Args:
        method_folder: A method folder; one without weight lists pins nothing.
        weights_dir: The folder to copy into, created if missing.

    Returns:
        The copies, the pretrained files first, each in the order its weight list names it.

    Raises:
        PackagingError: If a weight list names a file whose copy would land outside
            ``weights_dir``, or if a copy's sha256 is not the one its weight list pins: the cache
            holds other weights than the frozen method's.
        FileNotFoundError: If a pinned file is missing from its cache (``missing_caches`` says
            which before ``package_method`` writes anything).
        ValueError: If ``pretrained.json`` is in neither shape (``pinned_files``).
    """
    pinned = _pinned_weights(method_folder, weights_dir)
    _refuse_outside(method_folder, weights_dir, [copy for _, copy, _ in pinned])
    weights_dir.mkdir(parents=True, exist_ok=True)
    copies = []
    for source, copy, sha256 in pinned:
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        if file_sha256(copy) != sha256:
            raise PackagingError(
                f"{source} does not have the sha256 {sha256} that {method_folder} pins: its "
                "cache holds other weights than the frozen method's."
            )
        copies.append(copy)
    return copies


def lock_requirements(spec: MethodSpec, path: Path) -> None:
    """Write what an image installs: ``uv.lock``'s pins of the server's and the method's groups.

    ``uv export --frozen`` reads ``uv.lock`` as it is, never resolving again: every requirement
    pinned with ``==`` and its hashes, for pip's ``--require-hashes``. It exports the project's
    dependencies, the ``serving`` group (mlflow) and the method's groups, without the default
    groups (the notebooks', the tests' and the studies') and without the project itself, which
    the model holds under ``code/``.

    Args:
        spec: The method's spec: its ``groups`` are exported after ``serving``.
        path: The file to write, ``<build context>/requirements.lock.txt``.

    Raises:
        subprocess.CalledProcessError: If ``uv export`` fails, e.g. on a ``uv.lock`` that lacks
            a group.
    """
    groups = [option for group in (SERVING_GROUP, *spec.groups) for option in ("--group", group)]
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-default-groups",
        "--no-emit-project",
        "--format",
        "requirements-txt",
        *groups,
    ]
    path.write_text(run_command(command, cwd=PROJECT_ROOT), encoding="utf-8")


def copy_dockerfile(out: Path, dockerfile: Path = DOCKERFILE) -> Path:
    """Copy the image's recipe into a build context.

    Args:
        out: The build context.
        dockerfile: The recipe, ``deploy/Dockerfile`` by default.

    Returns:
        The copy, ``out / "Dockerfile"``.
    """
    copy = out / DOCKERFILE.name
    shutil.copyfile(dockerfile, copy)
    return copy


def _pinned_weights(method_folder: Path, weights_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return each file a method folder pins: its cache's path, its staged path, its sha256."""
    pinned = []
    if (method_folder / PRETRAINED_FILE).is_file():
        weight_list = json.loads((method_folder / PRETRAINED_FILE).read_text(encoding="utf-8"))
        snapshot = snapshot_folder(weight_list["repo_id"], weight_list["revision"])
        staged = weights_dir / PRETRAINED_WEIGHTS / snapshot.relative_to(pretrained_root())
        hashes = _pretrained_hashes(weight_list)
        pinned += [(snapshot / name, staged / name, sha256) for name, sha256 in hashes.items()]
    if (method_folder / TRAINED_FILE).is_file():
        weight_list = json.loads((method_folder / TRAINED_FILE).read_text(encoding="utf-8"))
        root = weight_list["root"]
        staged = weights_dir / TRAINED_WEIGHTS / root
        pinned += [
            (trained_root() / root / fold["file"], staged / fold["file"], fold["sha256"])
            for fold in weight_list["folds"].values()
        ]
    return pinned


def _refuse_outside(method_folder: Path, weights_dir: Path, copies: list[Path]) -> None:
    """Refuse the copies that a weight list's names would put outside ``weights_dir``."""
    root = weights_dir.resolve()
    outside = [str(copy) for copy in copies if root not in copy.resolve().parents]
    if outside:
        raise PackagingError(
            f"{method_folder} names weights whose copies would land outside {root}: "
            f"{', '.join(outside)}. A weight list names files inside its cache's folder, with "
            "neither '..' nor an absolute path."
        )


def _pretrained_hashes(weight_list: Mapping) -> dict[str, str]:
    """Return the sha256 of each file a ``pretrained.json`` pins, by name, from either shape."""
    names = pinned_files(weight_list)
    files = weight_list["files"]
    hashes = files.values() if isinstance(files, Mapping) else [file["sha256"] for file in files]
    return dict(zip(names, hashes, strict=True))


def _registered(name: str, registry: Mapping[str, MethodSpec]) -> MethodSpec:
    """Return a method's spec, refusing a method that is not registered."""
    if name not in registry:
        raise PackagingError(
            f"{name!r} is not a registered method: the registry has {sorted(registry)}."
        )
    return registry[name]


def _servable_folder(name: str, spec: MethodSpec, assets_dir: Path) -> Path:
    """Return a frozen method's folder, refusing one that is not data only or lacks weights."""
    folder = assets_dir / name / MODEL_FOLDER
    if not (folder / METHOD_FILE).is_file():
        raise PackagingError(
            f"{name} is not frozen: {folder} holds no {METHOD_FILE}, so run its freeze first."
        )
    try:
        check_method_folder(folder)
    except ConformanceError as error:
        raise PackagingError(f"{name} cannot be packaged: {error}") from error
    missing = missing_caches(folder, name, command=package_command(spec))
    if missing:
        raise PackagingError(
            f"{name} cannot be packaged on this machine: it needs {' and '.join(missing)}."
        )
    return folder


def _build_folder(name: str, out: Path | None) -> Path:
    """Return the build context to write, refusing one outside ``SERVING_BUILD_DIR``."""
    root = SERVING_BUILD_DIR.resolve()
    folder = (root / name if out is None else Path(out)).resolve()
    if folder == root or not folder.is_relative_to(root):
        raise PackagingError(
            f"{folder} is not a folder of {root}: package deletes the folder it writes, so it "
            "writes only there."
        )
    return folder


def _pinned_requirements(name: str, spec: MethodSpec) -> list[str]:
    """Pin the project's dependencies, the method's requirements and mlflow, from uv.lock."""
    project = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))["project"]
    names: dict[str, str] = {}
    for entry in (*project["dependencies"], *spec.requirements, "mlflow"):
        bare = _DISTRIBUTION.match(entry).group(0)
        names.setdefault(canonical_name(bare), bare)
    served = dataclasses.replace(spec, requirements=tuple(names.values()))
    try:
        return method_requirements(name, registry={name: served})
    except ValueError as error:
        raise PackagingError(f"{name} cannot be packaged here: {error}") from error


def _committed_inputs(folder: Path, allow_dirty: bool) -> dict[str, object]:
    """Return the commit and whether an input differs from it; refuse that unless allowed.

    Every input is checked by ``paths_dirty``, whose ``--untracked-files=all`` lists an untracked
    file whatever the user's git config: ``git_state``'s own check would miss an untracked module
    of the package under ``status.showUntrackedFiles=no``, and ``model/code`` would ship it.
    """
    commit = git_state()["commit"]
    inputs = [PACKAGE_DIR, PYPROJECT_FILE, UV_LOCK_FILE, DOCKERFILE, folder]
    dirty = [line.strip() for line in paths_dirty(inputs)]
    if dirty and not allow_dirty:
        raise PackagingError(
            f"The image would hold uncommitted changes ({'; '.join(dirty)}) under a commit that "
            "lacks them: commit them first, or pass --allow-dirty (its tag then ends in -dirty)."
        )
    return {"git_commit": commit, "code_dirty": bool(dirty)}


def _save_model(
    path: Path,
    spec: MethodSpec,
    folder: Path,
    requirements: list[str],
    metadata: dict[str, object],
) -> None:
    """Save the MLflow model of a method from code, and check that it loads from code."""
    mlflow_env()
    mlflow.pyfunc.save_model(
        path,
        python_model=str(SERVING_ENTRY),
        artifacts={"method": str(folder)},
        code_paths=[str(PACKAGE_DIR)],
        signature=served_signature(spec),
        pip_requirements=requirements,
        metadata=metadata,
    )
    loader = Model.load(path).flavors.get("python_function", {}).get("loader_module")
    if loader != CODE_MODEL_LOADER:
        raise PackagingError(
            f"{path} loads with {loader}, not {CODE_MODEL_LOADER}: a served method is saved "
            "from code, and nothing else is loaded."
        )
