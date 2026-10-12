"""The executable contract of a tagging method (docs/adr/0004): every method's tests call it.

- ``check_tagging_method``: a method is registered; its tags follow the order rule; a method of
  kind "score" exposes its scores and the per-tag thresholds it applies to them; ``predict``
  returns int 0/1 tags with its input's index, leaves its input unchanged, reads only its
  ``MethodSpec.input_columns`` and predicts one problem as within a batch; the method survives
  ``save_method`` then ``load_method``; its method folder holds data only; and it loads and
  predicts in a fresh interpreter without mlflow, optuna, gepa or a provider SDK.
- ``check_method_folder``: a method folder holds data files only, within the size caps, and no
  NumPy file in it holds an object array (a pickle inside).
- ``check_import_isolation``: the fresh-interpreter half of ``check_tagging_method``.
- ``check_frozen_method``: a method's committed ``assets/methods/<name>/`` is complete and
  consistent with the frozen split, answers files included for a method that calls an API.

Like the rest of the core, it imports neither mlflow, optuna, gepa, openai nor any deep-learning
library: the isolation check runs the method in a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, NamedTuple

import numpy as np
import pandas as pd

from codeforces_tags.config import (
    ALL_TAGS,
    FOCUS_TAGS,
    METHOD_FILE_MAX_BYTES,
    METHOD_FOLDER_MAX_BYTES,
    METHODS_ASSETS_DIR,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.evaluation.predictions import (
    PREDICTIONS_FILE,
    RUN_RECORD_FILE,
    SCORES_FILE,
    load_predictions,
    load_run_record,
    load_scores,
)
from codeforces_tags.llm.providers import sdk_modules
from codeforces_tags.llm.store import answers_file_name, read_answers
from codeforces_tags.methods import (
    METHOD_FILE,
    METHODS,
    MODEL_FOLDER,
    MethodSpec,
    ScoreMethod,
    TaggingMethod,
    load_method,
    read_method_file,
    save_method,
)
from codeforces_tags.model.thresholds import apply_thresholds
from codeforces_tags.splitting import PARTS, load_split

# The tracking and search libraries, which a method's prediction path never imports either.
TRACKING_MODULES: tuple[str, ...] = ("gepa", "mlflow", "optuna")
# The files a method folder may hold: data formats only (docs/adr/0002).
DATA_SUFFIXES: tuple[str, ...] = (
    ".json",
    ".jsonl",
    ".csv",
    ".npy",
    ".npz",
    ".skops",
    ".safetensors",
    ".onnx",
    ".md",
    ".txt",
    ".j2",
)
# Pickle formats, refused by name: loading one runs the code it holds.
PICKLE_SUFFIXES: tuple[str, ...] = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt")
# The data files read as text, to find an absolute path in them.
_TEXT_SUFFIXES = (".json", ".jsonl", ".csv", ".md", ".txt", ".j2")
# NumPy's files, an array (.npy) or a zip archive of them (.npz): numpy stores an object array
# as a pickle inside, which np.load(..., allow_pickle=True) runs.
_NUMPY_SUFFIXES = (".npy", ".npz")
# A pickle of protocol 2 to 5 starts with the PROTO opcode, then its protocol.
_PICKLE_PROTOCOLS = (b"\x02", b"\x03", b"\x04", b"\x05")
# The raw problem the isolation subprocess predicts when it is given none: read_samples()'s
# columns, filled as the dataset fills them.
_SYNTHETIC_PROBLEM = {
    "prob_desc_description": "Given a tree of $$$n$$$ vertices, print the number of its leaves.",
    "prob_desc_input_spec": "The first line holds $$$n$$$, then $$$n - 1$$$ edges.",
    "prob_desc_output_spec": "Print one integer.",
    "prob_desc_notes": "NoteVertices $$$2$$$ and $$$3$$$ are leaves.",
    "prob_desc_sample_inputs": '["3\\n1 2\\n1 3"]',
    "prob_desc_sample_outputs": '["2"]',
    "source_code": "n = int(input())\nprint(n - 1)\n",
}
# Run by check_import_isolation in a fresh interpreter. argv[1] is a JSON payload: the method
# folder, the registry ({name: asdict(spec)}), sys_path, and the modules to look for. stdin is
# the raw problem to predict (DataFrame.to_json(orient="table")), or empty for none. It prints
# the modules it finds in sys.modules, as a JSON list.
_ISOLATION_SCRIPT = """\
import json
import sys
from io import StringIO
from pathlib import Path

payload = json.loads(sys.argv[1])
sys.path[:0] = payload["sys_path"]

import pandas as pd

from codeforces_tags.methods import MethodSpec, load_method

registry = {name: MethodSpec(**values) for name, values in payload["registry"].items()}
method = load_method(Path(payload["folder"]), registry=registry)
row = sys.stdin.read()
if row:
    method.predict(pd.read_json(StringIO(row), orient="table"))
print(json.dumps(sorted(set(payload["modules"]) & set(sys.modules))))
"""


class ConformanceError(AssertionError):
    """A tagging method, method folder or frozen method breaks the framework's contract."""


class _Outputs(NamedTuple):
    """What a method returned for some raw problems: its predictions, and scores if it has any."""

    predictions: pd.DataFrame
    scores: pd.DataFrame | None

    def row(self, position: int) -> _Outputs:
        """Return the outputs of one problem, by position."""
        scores = None if self.scores is None else self.scores.iloc[[position]]
        return _Outputs(self.predictions.iloc[[position]], scores)


def check_tagging_method(
    method: TaggingMethod,
    raw_problems: pd.DataFrame,
    folder: Path,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    sys_path: Sequence[str] = (),
) -> None:
    """Fail unless ``method`` follows the framework (ADR 0004); every method's tests call it.

    It checks, in order:

    - ``method`` is a ``TaggingMethod`` whose ``name`` is a key of ``registry``;
    - ``method.tags`` is a tuple of distinct tags of ``ALL_TAGS`` holding every focus tag, in
      ``FOCUS_TAGS`` order when they are the focus tags alone (as ``focus_labels`` gives them),
      in ``ALL_TAGS`` order otherwise;
    - a method of kind "score", or with ``score_atol > 0``, implements ``ScoreMethod``;
    - ``predict`` (and ``scores``) on the raw problems restricted to the spec's
      ``input_columns`` returns a DataFrame with their index, in order, and the columns of
      ``tags``: int 0/1 predictions, float scores without NaN; and leaves its input unchanged;
    - a ``ScoreMethod``'s ``predict_from_scores`` of those scores returns those predictions
      exactly, and leaves the scores unchanged;
    - a ``ScoreMethod``'s ``tag_thresholds()`` is a Series of floats without NaN, indexed by
      ``tags`` in order, and ``apply_thresholds`` of those scores and thresholds returns those
      predictions exactly;
    - it returns the same given the raw problems with all their columns (e.g. ``fold``,
      ``group``, ``tags``), and given one problem at a time;
    - ``save_method`` into ``folder`` then ``load_method(folder, registry=registry)`` gives a
      method that returns the same;
    - ``check_method_folder(folder)`` and ``check_import_isolation(folder, ...)`` pass.

    "The same" is exact for 0/1 predictions when the spec's ``score_atol`` is 0; scores agree
    within ``max(score_atol, TUNING_EXPORT_TOLERANCE)`` (rtol 0; equal infinities agree). With
    ``score_atol > 0``, a tag may flip only where its scores differ: to 1 where its score rose,
    to 0 where it fell. This is weaker than verify's rule (step 7 of CLAUDE.md's checklist), which
    allows a flip only where the frozen score is within ``score_atol`` of the method's
    threshold: this check does not compare the thresholds of the two runs, so it cannot tell a
    flip at the decision boundary from one under another threshold.

    Args:
        method: The fitted method.
        raw_problems: At least two raw problems, as ``read_samples()`` gives them (synthetic in
            a test), with every column of the spec's ``input_columns``.
        folder: A new or empty folder, e.g. ``tmp_path / "model"``, for the method folder.
        registry: The registered methods.
        sys_path: Folders the subprocess prepends to ``sys.path`` to import the method's module
            (e.g. a test's fakes folder).

    Raises:
        ConformanceError: If the method breaks a rule.
        ValueError: If ``raw_problems`` lacks an input column or holds fewer than two problems,
            or if ``folder`` is not empty.
    """
    spec = _check_contract(method, registry)
    inputs = _input_problems(raw_problems, spec)
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"{folder} is not empty: pass a new folder, e.g. tmp_path / 'model'.")
    batch = _outputs(method, inputs)
    _check_same(f"{method.name} given extra columns", batch, _outputs(method, raw_problems), spec)
    for position, sample_id in enumerate(inputs.index):
        single = _outputs(method, inputs.iloc[[position]])
        what = f"{method.name} given one problem at a time (sample_id {sample_id})"
        _check_same(what, batch.row(position), single, spec)
    save_method(method, folder)
    loaded = load_method(folder, registry=registry)
    if isinstance(method, ScoreMethod) and not isinstance(loaded, ScoreMethod):
        raise ConformanceError(
            f"{method.name} loses its scores through save_method and load_method: its load "
            f"returns a {type(loaded).__name__}, which is not a ScoreMethod: it lacks "
            "scores(raw_problems), predict_from_scores(scores) or tag_thresholds()."
        )
    what = f"{method.name} after save_method and load_method"
    _check_same(what, batch, _outputs(loaded, inputs), spec)
    check_method_folder(folder)
    check_import_isolation(folder, registry=registry, sys_path=sys_path, sample_row=inputs.iloc[:1])


def check_method_folder(folder: Path) -> None:
    """Fail unless a method folder holds data files only, within the size caps.

    Every file, in every sub-folder, has a suffix of ``DATA_SUFFIXES``, does not start with
    pickle header bytes (``starts_like_pickle``), holds at most ``METHOD_FILE_MAX_BYTES``, and,
    when it is text, does not hold the folder's absolute path (a moved folder must load). An
    ``.npy`` file, and each member of an ``.npz`` archive, is an array that numpy wrote, of a
    dtype without Python objects: numpy pickles an object array. The folder holds no symbolic
    link and at most ``METHOD_FOLDER_MAX_BYTES`` in all.

    Args:
        folder: The method folder.

    Raises:
        ConformanceError: If a file or the folder breaks a rule.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ConformanceError(f"{folder} is not a folder: save the method with save_method().")
    total = 0
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ConformanceError(
                f"{path.relative_to(folder)} in {folder} is a symbolic link: a method folder "
                "holds its own files, so it loads once moved."
            )
        if path.is_file():
            _check_method_file(path, folder)
            total += path.stat().st_size
    if total > METHOD_FOLDER_MAX_BYTES:
        raise ConformanceError(
            f"{folder} holds {total} bytes in all, above {METHOD_FOLDER_MAX_BYTES} "
            "(METHOD_FOLDER_MAX_BYTES): pretrained weights stay out of it (ADR 0004)."
        )


def starts_like_pickle(path: Path) -> bool:
    """Tell whether a file starts with the header of a pickle.

    Pickles of protocols 2 to 5, the ones Python 3 writes by default, start with the PROTO
    opcode and their protocol. A safetensors file can start with the same two bytes (its first 8
    bytes give the length of its JSON header, little endian, e.g. 640): it is told apart by its
    header, which starts with ``{`` and fits in the file.

    Args:
        path: The file.

    Returns:
        Whether it starts like a pickle.
    """
    with Path(path).open("rb") as file:
        head = file.read(9)
    if head[:1] != b"\x80" or head[1:2] not in _PICKLE_PROTOCOLS:
        return False
    header_length = int.from_bytes(head[:8], "little")
    return not (head[8:9] == b"{" and 8 + header_length <= Path(path).stat().st_size)


def check_import_isolation(
    folder: Path,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    sys_path: Sequence[str] = (),
    sample_row: pd.DataFrame | None = None,
) -> None:
    """Fail unless a saved method loads and predicts without mlflow, optuna, gepa or a provider SDK.

    A subprocess, which inherits the environment (so a test's ``$CODEFORCES_DATA_DIR`` or
    ``$CODEFORCES_PRETRAINED_DIR`` reaches it), receives the registry's specs and ``sys_path``
    as JSON on argv, prepends ``sys_path`` to ``sys.path``, and calls
    ``load_method(folder, registry=...)``; unless the method calls an API, it also predicts one
    raw problem, passed as JSON on stdin. It then lists which of ``TRACKING_MODULES`` and of
    the providers' SDKs (``llm.providers.sdk_modules()``) are in ``sys.modules``.

    Args:
        folder: A method folder written by ``save_method``.
        registry: The registered methods.
        sys_path: Folders to prepend to the subprocess's ``sys.path`` to import the method's
            module.
        sample_row: The raw problem to predict, indexed by ``sample_id``, with the spec's
            ``input_columns``; a synthetic problem when None.

    Raises:
        ConformanceError: If ``folder`` holds no registered method, if the subprocess fails, or
            if it imported a tracking or provider module.
    """
    folder = Path(folder)
    name = _stored_method(folder)["method"]
    spec = _registered(name, registry)
    row = sample_row if sample_row is not None else _synthetic_row(spec.input_columns)
    payload = {
        "folder": str(folder.resolve()),
        "registry": {key: asdict(value) for key, value in registry.items()},
        "sys_path": [str(entry) for entry in sys_path],
        "modules": [*TRACKING_MODULES, *sdk_modules()],
    }
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_SCRIPT, json.dumps(payload)],
        input="" if spec.calls_api else row.to_json(orient="table"),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ConformanceError(
            f"Loading {name} in a fresh interpreter failed (sys_path must hold the folders of "
            f"its modules):\n{result.stderr.strip()}"
        )
    found = json.loads(result.stdout.strip().splitlines()[-1])
    if found:
        raise ConformanceError(
            f"Loading {name} and predicting one problem imported {found}: a method's prediction "
            "path never imports mlflow, optuna, gepa or a provider SDK (an API method builds its "
            "client when it first asks)."
        )


def check_frozen_method(
    name: str,
    *,
    assets_dir: Path = METHODS_ASSETS_DIR,
    split: pd.DataFrame | None = None,
    registry: Mapping[str, MethodSpec] = METHODS,
) -> None:
    """Fail unless a method's frozen ``assets/methods/<name>/`` is complete and consistent.

    Read without importing the method, it checks that:

    - the method is registered, and ``model/method.json`` names it, with tags that follow the
      order rule of ``check_tagging_method``;
    - ``predictions.csv`` holds int 0/1 values, one row per dev and test problem of the split
      (``(part, sample_id)``), and the tags of ``method.json`` as columns, in order;
    - a method of kind "score" has a ``scores.csv``; when present, it has the rows and columns
      of ``predictions.csv``, in order;
    - ``run_record.json`` has every ``RunRecord`` key and names the method, and a local method's
      ``method_entries`` has the ``supersedes`` key (ADR 0004);
    - a method that calls an API through requests of its own (no ``depends_on``) has an
      ``answers_<part>.jsonl`` per part, which ``llm.store.read_answers`` reads, with exactly one
      line per problem of the part; a composite method over one has none;
    - ``check_method_folder`` passes on ``model/``.

    Args:
        name: The method's registry key.
        assets_dir: The folder of every frozen method.
        split: The frozen split (``load_split()`` when None).
        registry: The registered methods.

    Raises:
        ConformanceError: If a file is missing, malformed or inconsistent.
    """
    spec = _registered(name, registry)
    directory = Path(assets_dir) / name
    split = load_split() if split is None else split
    tags = _frozen_tags(directory / MODEL_FOLDER, name)
    predictions = _read_frozen(load_predictions, directory, PREDICTIONS_FILE)
    _check_frozen_rows(predictions, split)
    if tuple(predictions.columns) != tags:
        raise ConformanceError(
            f"The columns of {directory / PREDICTIONS_FILE} must be the tags of {METHOD_FILE}, "
            f"in order: {list(tags)}; they are {list(predictions.columns)}."
        )
    _check_frozen_scores(directory, predictions, spec)
    _check_frozen_record(directory, name, spec)
    if spec.calls_api and not spec.depends_on:
        _check_frozen_answers(directory, split)
    check_method_folder(directory / MODEL_FOLDER)


def _registered(name: str, registry: Mapping[str, MethodSpec]) -> MethodSpec:
    """Return the spec of a method, refusing an unregistered one."""
    if name not in registry:
        raise ConformanceError(
            f"{name!r} is not registered: add its MethodSpec to METHODS (the registry holds "
            f"{sorted(registry)})."
        )
    return registry[name]


def _check_contract(method: object, registry: Mapping[str, MethodSpec]) -> MethodSpec:
    """Check the protocol, the registration, the tags and the kind; return the method's spec."""
    if not isinstance(method, TaggingMethod):
        raise ConformanceError(
            f"A {type(method).__name__} is not a TaggingMethod: it needs name, tags, "
            "predict(raw_problems) and save(folder)."
        )
    spec = _registered(method.name, registry)
    _check_tags(method.tags, f"The tags of {method.name}")
    scored = isinstance(method, ScoreMethod)
    if spec.kind == "score" and not scored:
        raise ConformanceError(
            f"{method.name} is not a ScoreMethod: a method of kind 'score' implements "
            "ScoreMethod, with scores(raw_problems), predict_from_scores(scores) and "
            "tag_thresholds()."
        )
    if spec.score_atol > 0 and not scored:
        raise ConformanceError(
            f"{method.name} has score_atol {spec.score_atol}, which needs scores: it implements "
            "ScoreMethod, or has score_atol 0."
        )
    return spec


def _check_tags(tags: object, what: str) -> None:
    """Fail unless tags are distinct tags of ALL_TAGS with every focus tag, in the order rule."""
    _check_distinct_tags(tags, what)
    _check_vocabulary(tags, what)
    if set(tags) == set(FOCUS_TAGS):
        expected, rule = FOCUS_TAGS, "FOCUS_TAGS order (as focus_labels gives them)"
    else:
        expected, rule = tuple(tag for tag in ALL_TAGS if tag in tags), "ALL_TAGS order"
    if tags != expected:
        raise ConformanceError(f"{what} must be in {rule}: {list(expected)}, not {list(tags)}.")


def _check_distinct_tags(tags: object, what: str) -> None:
    """Fail unless tags are a tuple of str, each once."""
    if not isinstance(tags, tuple) or not all(isinstance(tag, str) for tag in tags):
        raise ConformanceError(f"{what} must be a tuple of str, not {tags!r}.")
    repeated = sorted({tag for tag in tags if tags.count(tag) > 1})
    if repeated:
        raise ConformanceError(f"{what} repeat {repeated}: a method predicts each tag once.")


def _check_vocabulary(tags: tuple[str, ...], what: str) -> None:
    """Fail unless tags are tags of ALL_TAGS that hold every focus tag."""
    unknown = [tag for tag in tags if tag not in ALL_TAGS]
    if unknown:
        raise ConformanceError(f"{what} hold {unknown}, which are not tags of ALL_TAGS.")
    missing = [tag for tag in FOCUS_TAGS if tag not in tags]
    if missing:
        raise ConformanceError(f"{what} lack the focus tags {missing}: every method predicts them.")


def _input_problems(raw_problems: pd.DataFrame, spec: MethodSpec) -> pd.DataFrame:
    """Return a copy of the raw problems restricted to the spec's input columns."""
    missing = [column for column in spec.input_columns if column not in raw_problems]
    if missing:
        raise ValueError(
            f"The raw problems lack the input columns {missing}: pass rows as read_samples() "
            "returns them."
        )
    if len(raw_problems) < 2:
        raise ValueError(
            "check_tagging_method needs at least two raw problems, to compare one problem with a "
            "batch."
        )
    return raw_problems[list(spec.input_columns)].copy()


def _call(method: TaggingMethod, function: Callable, raw_problems: pd.DataFrame) -> object:
    """Call ``function(raw_problems)``, turning any error into a ConformanceError."""
    try:
        return function(raw_problems)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ConformanceError(
            f"{method.name}.{function.__name__} raised {error!r} on raw problems with the columns "
            f"{list(raw_problems.columns)}: every raw column it reads must be in its "
            "MethodSpec.input_columns."
        ) from error


def _outputs(method: TaggingMethod, raw_problems: pd.DataFrame) -> _Outputs:
    """Call predict (and scores), checking their outputs and that the input is unchanged."""
    before = raw_problems.copy(deep=True)
    predictions = _call(method, method.predict, raw_problems)
    scores = _call(method, method.scores, raw_problems) if isinstance(method, ScoreMethod) else None
    if not raw_problems.equals(before):
        raise ConformanceError(
            f"{method.name} changed its input: predict and scores leave the raw problems as given."
        )
    _check_frame(f"{method.name}.predict", predictions, raw_problems, method.tags)
    integer = all(pd.api.types.is_integer_dtype(dtype) for dtype in predictions.dtypes)
    if not integer or not predictions.isin((0, 1)).all().all():
        raise ConformanceError(f"{method.name}.predict must return int 0/1 values only.")
    if scores is not None:
        _check_frame(f"{method.name}.scores", scores, raw_problems, method.tags)
        floats = all(pd.api.types.is_float_dtype(dtype) for dtype in scores.dtypes)
        if not floats or scores.isna().any().any():
            raise ConformanceError(f"{method.name}.scores must return float scores without NaN.")
        _check_predictions_from_scores(method, scores, predictions, raw_problems)
        _check_tag_thresholds(method, scores, predictions)
    return _Outputs(predictions, scores)


def _check_predictions_from_scores(
    method: ScoreMethod, scores: pd.DataFrame, predictions: pd.DataFrame, raw_problems: pd.DataFrame
) -> None:
    """Fail unless the tags a score method takes from its scores are its predictions, exactly."""
    before = scores.copy(deep=True)
    try:
        derived = method.predict_from_scores(scores)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ConformanceError(
            f"{method.name}.predict_from_scores raised {error!r} on the frame that "
            f"{method.name}.scores returned."
        ) from error
    if not scores.equals(before):
        raise ConformanceError(
            f"{method.name}.predict_from_scores changed the scores it was given: the command line "
            "prints them after taking the tags from them."
        )
    _check_frame(f"{method.name}.predict_from_scores", derived, raw_problems, method.tags)
    if not derived.equals(predictions):
        raise ConformanceError(
            f"{method.name}.predict_from_scores(scores) must return predict's 0/1 tags, exactly: "
            "predict(raw_problems) is predict_from_scores(scores(raw_problems)), the tags the "
            "command line prints beside the scores."
        )


def _check_tag_thresholds(
    method: ScoreMethod, scores: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Fail unless a score method's thresholds are floats by tag that give it its predictions."""
    thresholds = method.tag_thresholds()
    what = f"{method.name}.tag_thresholds()"
    if not isinstance(thresholds, pd.Series):
        raise ConformanceError(f"{what} must return a Series, not {type(thresholds).__name__}.")
    if list(thresholds.index) != list(method.tags):
        raise ConformanceError(
            f"{what} must be indexed by its tags, in order: {list(method.tags)}, not "
            f"{list(thresholds.index)}."
        )
    if not pd.api.types.is_float_dtype(thresholds.dtype) or thresholds.isna().any():
        raise ConformanceError(
            f"{what} must return float thresholds without NaN: inf for a tag it never predicts."
        )
    applied = apply_thresholds(scores, thresholds)
    if not np.array_equal(applied.to_numpy(), predictions.to_numpy()):
        raise ConformanceError(
            f"{what} must be the thresholds that predict_from_scores applies: "
            "apply_thresholds(scores, tag_thresholds()) must return its 0/1 tags, exactly, so a "
            "score shown beside its threshold explains the tag."
        )


def _check_frame(what: str, frame: object, raw_problems: pd.DataFrame, tags: Sequence[str]) -> None:
    """Fail unless ``frame`` is a DataFrame indexed like the raw problems, with the tags."""
    if not isinstance(frame, pd.DataFrame):
        raise ConformanceError(f"{what} must return a DataFrame, not {type(frame).__name__}.")
    index = raw_problems.index
    if not frame.index.equals(index) or list(frame.index.names) != list(index.names):
        raise ConformanceError(
            f"{what} must return the index of its input ({index.names}), in order."
        )
    if list(frame.columns) != list(tags):
        raise ConformanceError(
            f"{what} must return the columns of its tags, in order: {list(tags)}, not "
            f"{list(frame.columns)}."
        )


def _check_same(what: str, expected: _Outputs, actual: _Outputs, spec: MethodSpec) -> None:
    """Fail unless two outputs of the same problems agree, under the spec's score_atol."""
    if expected.scores is not None:
        atol = max(spec.score_atol, TUNING_EXPORT_TOLERANCE)
        _check_close_scores(what, expected.scores, actual.scores, atol)
    flips = expected.predictions.to_numpy() != actual.predictions.to_numpy()
    if not flips.any():
        return
    if spec.score_atol == 0:
        raise ConformanceError(
            f"{what}: the predictions differ on {int(flips.sum())} (problem, tag) pairs."
        )
    before, after = expected.scores.to_numpy(float), actual.scores.to_numpy(float)
    with_scores = np.where(actual.predictions.to_numpy() == 1, after > before, after < before)
    if not with_scores[flips].all():
        raise ConformanceError(
            f"{what}: a tag flips against its scores: within score_atol, a tag may flip only "
            "where its scores differ, to 1 where its score rose and to 0 where it fell."
        )


def _check_close_scores(
    what: str, expected: pd.DataFrame, actual: pd.DataFrame, atol: float
) -> None:
    """Fail unless two score frames agree within atol, rtol 0; equal infinities agree."""
    before, after = expected.to_numpy(float), actual.to_numpy(float)
    close = np.isclose(before, after, rtol=0, atol=atol)
    if not close.all():
        gap = float(np.abs(before[~close] - after[~close]).max())
        raise ConformanceError(
            f"{what}: the scores differ by up to {gap:.3g}, more than {atol:g} (score_atol, "
            "rtol 0)."
        )


def _synthetic_row(columns: Sequence[str]) -> pd.DataFrame:
    """Return one synthetic raw problem with the given columns, indexed by sample_id."""
    values = {column: _SYNTHETIC_PROBLEM.get(column, "") for column in columns}
    return pd.DataFrame([values], index=pd.Index([0], name="sample_id"))


def _stored_method(folder: Path) -> dict:
    """Read a method folder's ``method.json``, turning a missing or malformed one into an error."""
    try:
        return read_method_file(folder)
    except (FileNotFoundError, ValueError) as error:
        raise ConformanceError(str(error)) from error


def _frozen_tags(model: Path, name: str) -> tuple[str, ...]:
    """Return the tags of a frozen method folder, checking that it holds the method ``name``."""
    stored = _stored_method(model)
    if stored.get("method") != name:
        raise ConformanceError(
            f"{model / METHOD_FILE} names the method {stored.get('method')!r}, not {name!r}."
        )
    tags = tuple(stored.get("tags", ()))
    _check_tags(tags, f"The tags of {model / METHOD_FILE}")
    return tags


def _read_frozen(read: Callable, directory: Path, file_name: str) -> object:
    """Read a frozen file with its loader, turning a missing or malformed file into an error."""
    path = directory / file_name
    if not path.is_file():
        raise ConformanceError(f"{directory} has no {file_name}: run the method's freeze.")
    try:
        return read(directory)
    except ValueError as error:
        raise ConformanceError(f"{path} is malformed: {error}") from error


def _check_frozen_rows(predictions: pd.DataFrame, split: pd.DataFrame) -> None:
    """Fail unless the frozen rows are the split's dev and test problems, each once."""
    _check_each_once(
        f"{PREDICTIONS_FILE} must hold each dev and test problem of split.csv once",
        list(predictions.index),
        set(zip(split["part"], split.index, strict=True)),
        "the split",
    )


def _check_frozen_answers(directory: Path, split: pd.DataFrame) -> None:
    """Fail unless each answers file of an API method has one line per problem of its part."""
    for part in PARTS:
        path = directory / answers_file_name(part)
        if not path.is_file():
            raise ConformanceError(
                f"{directory} has no {path.name}: run `llm run --part {part}`, which writes a "
                "method's answers, and commit them."
            )
        try:
            answers = read_answers(path)
        except ValueError as error:
            raise ConformanceError(f"{path} is malformed: {error}") from error
        _check_each_once(
            f"{path} must hold one line per problem of the {part} part",
            [answer.sample_id for answer in answers],
            set(split.index[split["part"] == part]),
            f"the {part} part",
        )


def _check_each_once(what: str, rows: list, expected: set, where: str) -> None:
    """Fail unless ``rows`` holds each expected row once, and no other; ``what`` says why."""
    found = set(rows)
    if found != expected or len(rows) != len(found):
        raise ConformanceError(
            f"{what}: {len(expected - found)} missing, {len(found - expected)} not in {where}, "
            f"{len(rows) - len(found)} repeated."
        )


def _check_frozen_scores(directory: Path, predictions: pd.DataFrame, spec: MethodSpec) -> None:
    """Fail unless a score method has scores.csv, aligned with predictions.csv when present."""
    if not (directory / SCORES_FILE).is_file():
        if spec.kind == "score":
            raise ConformanceError(
                f"{directory} has no {SCORES_FILE}: a method of kind 'score' freezes its scores."
            )
        return
    scores = _read_frozen(load_scores, directory, SCORES_FILE)
    if not scores.index.equals(predictions.index) or list(scores.columns) != list(
        predictions.columns
    ):
        raise ConformanceError(
            f"{directory / SCORES_FILE} must have the rows and columns of {PREDICTIONS_FILE}, "
            "in order."
        )


def _check_frozen_record(directory: Path, name: str, spec: MethodSpec) -> None:
    """Fail unless the run record names the method and, for a local method, its supersedes."""
    record = _read_frozen(load_run_record, directory, RUN_RECORD_FILE)
    if record.method != name:
        raise ConformanceError(
            f"{directory / RUN_RECORD_FILE} names the method {record.method!r}, not {name!r}."
        )
    if not spec.calls_api and "supersedes" not in record.method_entries:
        raise ConformanceError(
            f"{directory / RUN_RECORD_FILE} lacks method_entries['supersedes']: a local "
            "method's freeze sets it (ADR 0004)."
        )


def _check_method_file(path: Path, folder: Path) -> None:
    """Fail unless one file of a method folder is a data file within the size cap."""
    name = path.relative_to(folder)
    suffix = path.suffix.lower()
    if suffix in PICKLE_SUFFIXES:
        raise ConformanceError(
            f"{name} in {folder} is a pickle format ({suffix}): loading it runs the code it "
            "holds. Save JSON, CSV, .npy/.npz, skops, safetensors or ONNX instead."
        )
    if suffix not in DATA_SUFFIXES:
        raise ConformanceError(
            f"{name} in {folder} is not a data file: a method folder holds only {DATA_SUFFIXES}."
        )
    size = path.stat().st_size
    if size > METHOD_FILE_MAX_BYTES:
        raise ConformanceError(
            f"{name} in {folder} holds {size} bytes, above {METHOD_FILE_MAX_BYTES} "
            "(METHOD_FILE_MAX_BYTES)."
        )
    if starts_like_pickle(path):
        raise ConformanceError(f"{name} in {folder} starts with pickle header bytes.")
    if suffix in _NUMPY_SUFFIXES:
        _check_numpy_file(path, name, folder)
    if suffix in _TEXT_SUFFIXES and _names_folder(path, folder):
        raise ConformanceError(
            f"{name} in {folder} holds the folder's absolute path: a method folder loads from "
            "anywhere, so it stores paths relative to itself."
        )


def _check_numpy_file(path: Path, name: Path, folder: Path) -> None:
    """Fail unless an .npy file, or each member of an .npz archive, holds no Python object."""
    try:
        dtypes = _numpy_dtypes(path)
    except (ValueError, zipfile.BadZipFile) as error:
        raise ConformanceError(
            f"{name} in {folder} is not a valid {path.suffix} file ({error}): save arrays with "
            "np.save or np.savez."
        ) from error
    if any(dtype.hasobject for dtype in dtypes):
        raise ConformanceError(
            f"{name} in {folder} holds an object array, which numpy stores as a pickle: loading "
            "it with allow_pickle=True runs the code it holds. Save numbers or strings only."
        )


def _numpy_dtypes(path: Path) -> list[np.dtype]:
    """Read the dtype of an .npy file, or of each member of an .npz archive, from its header."""
    if path.suffix.lower() == ".npy":
        with path.open("rb") as file:
            return [_npy_dtype(file)]
    dtypes = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            with archive.open(member) as file:
                dtypes.append(_npy_dtype(file))
    return dtypes


def _npy_dtype(file: IO[bytes]) -> np.dtype:
    """Read the dtype in the header of an .npy file, refusing what is not one."""
    version = np.lib.format.read_magic(file)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(file)[2]
    return np.lib.format.read_array_header_2_0(file)[2]


def _names_folder(path: Path, folder: Path) -> bool:
    """Tell whether a text file holds its folder's absolute path."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(str(marker) in text for marker in (folder.absolute(), folder.resolve()))
