"""Log tagging methods to MLflow once, and load them back (docs/adr/0002).

Every tagging method is logged through one models-from-code wrapper: the entry file
``pyfunc_model.py``, logged by its path (``PYFUNC_ENTRY``), with the method folder as its
artifact ``method``, an explicit signature of raw string columns (``raw_input_signature``) and
explicit requirements pinned from ``uv.lock`` (``method_requirements``). ``load_logged_method``
is the only sanctioned way to load one back.

``log_method`` gives each method one run of the ``tag-methods`` experiment, found again by its
``recipe_sha256`` (``method_run_id``), with one logged model whose parameters are the run record
(``flat_record``), and the method's dataset-level metrics (``standard_metrics``) logged once per
dataset on that model; the run's own metrics show the out-of-fold dev ones. A second call adds
only what is missing, so an empty store is rebuilt from the committed files and a full one gains
no duplicate. ``log_run_extras`` adds metrics, figures and tables to the run the same way.
"""

from __future__ import annotations

import importlib.metadata
import json
import posixpath
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.entities import LoggedModelStatus
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

from codeforces_tags.config import (
    ALL_TAGS,
    FOCUS_TAGS,
    MLFLOW_METRIC_DATASETS,
    PAPER_TAGS,
    UV_LOCK_FILE,
)
from codeforces_tags.conformance import check_method_folder
from codeforces_tags.evaluation.report import ALL_PROBLEMS, all_tag_macro_f1, evaluate
from codeforces_tags.methods import METHODS, MethodSpec, read_method_file
from codeforces_tags.provenance import RunRecord, canonical_name, locked_versions
from codeforces_tags.tracking.environment import use_store

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# The entry file every tagging method is logged through, by its path: MLflow copies it into the
# logged model and runs the copy on load.
PYFUNC_ENTRY = Path(__file__).with_name("pyfunc_model.py")
# The loader that MLflow names in the MLmodel of a model logged from code. Any other loader is
# refused: mlflow.pyfunc.model, for one, unpickles python_model.pkl.
CODE_MODEL_LOADER = "mlflow.pyfunc.loaders.code_model"
# A bare distribution name, such as "scikit-learn": the form of MethodSpec.requirements.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The name of a method's logged model in its run.
LOGGED_MODEL_NAME = "tagging_method"
# The tag that tells a method run (value "method") from the other runs of tag-methods, such as
# the evaluation runs, which carry the same recipe_sha256 tag.
RUN_KIND_TAG = "run_kind"
# The dataset of the selection score (ADR 0001): a method run's own metrics show its values.
_SELECTION_DATASET = MLFLOW_METRIC_DATASETS["dev"]
# A sha256 hex digest, the form of every recipe_sha256.
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A key flat_record extends with a dot: characters MLflow accepts in a parameter name on every
# platform, except the dot itself, so that the dotted keys stay unambiguous.
_PARAM_KEY = re.compile(r"[\w/ -]+")


def raw_input_signature(tags: Sequence[str], input_columns: Sequence[str]) -> ModelSignature:
    """Return the signature of a logged tagging method: raw string columns in, tags out.

    The raw sample inputs and outputs stay JSON strings: MLflow 3.16 returns twice the rows, or
    pairs rows wrongly, for list columns on a frame indexed by ``sample_id`` (safeguard 2).
    MLflow drops any input column outside the signature.

    Args:
        tags: The method's tags, in order.
        input_columns: The raw columns its ``predict`` reads (``MethodSpec.input_columns``).

    Returns:
        Required string inputs, one per input column, and one long output per tag.
    """
    inputs = Schema([ColSpec(DataType.string, column, required=True) for column in input_columns])
    outputs = Schema([ColSpec(DataType.long, tag, required=True) for tag in tags])
    return ModelSignature(inputs=inputs, outputs=outputs)


def method_requirements(
    name: str, *, registry: Mapping[str, MethodSpec] = METHODS, lock: Path = UV_LOCK_FILE
) -> list[str]:
    """Pin a method's requirements to their installed versions, which ``uv.lock`` must pin.

    ``uv.lock`` can pin a package at several versions, one per Python version (e.g. numpy): the
    installed one is kept. MLflow adds its own requirement to the logged model's.

    Args:
        name: The method's registry key.
        registry: The registered methods.
        lock: The ``uv.lock`` file.

    Returns:
        ``"<requirement>==<version>"`` for each of ``registry[name].requirements``, in order.

    Raises:
        ValueError: If the method is not registered, or a requirement is not a bare
            distribution name, is not installed, or is installed at a version ``uv.lock`` does
            not pin.
    """
    if name not in registry:
        raise ValueError(f"{name!r} is not registered: the registry has {sorted(registry)}.")
    locked = locked_versions(lock)
    return [_pinned(requirement, locked) for requirement in registry[name].requirements]


def _pinned(requirement: str, locked: Mapping[str, Sequence[str]]) -> str:
    """Pin one requirement to its installed version, refusing a version uv.lock does not pin."""
    if not _DISTRIBUTION.fullmatch(requirement):
        raise ValueError(
            f"{requirement!r} is not a distribution name: MethodSpec.requirements names the "
            "packages, and method_requirements pins them from uv.lock."
        )
    try:
        installed = importlib.metadata.version(requirement)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            f"{requirement} is not installed: sync the method's dependency groups first."
        ) from error
    versions = list(locked.get(canonical_name(requirement), ()))
    if installed not in versions:
        raise ValueError(
            f"{requirement} {installed} is installed, but uv.lock pins {versions}: run uv sync."
        )
    return f"{requirement}=={installed}"


def load_logged_method(model_uri: str, **store: Any) -> mlflow.pyfunc.PyFuncModel:
    """Load a logged tagging method: the only sanctioned way (docs/adr/0002, safeguard 1).

    It points MLflow at the project's store first (``use_store``, which sets MLflow's
    environment: no pickle is ever loaded, and no default ``./mlflow.db`` appears in the working
    directory), refuses a model that was not logged from code, and loads with
    ``sys.pycache_prefix`` pointed at a temporary folder: Python then neither runs bytecode
    cached in the store instead of the reviewed entry file, nor writes any there. The entry file
    loads the method through the loading code's ``METHODS``. Load only models this project
    logged: loading one runs its entry file.

    The model's signature makes every raw column a required string (safeguard 2): pass rows as
    ``read_samples()`` or ``select_part`` give them, a missing field NaN in a ``str`` column. A
    problem typed by hand with ``np.nan`` alone in a field has a float64 column, which MLflow's
    schema check refuses ("Can not safely convert float64 to <U0"): cast it first, with
    ``problems.astype({column: "str"})``.

    Args:
        model_uri: The logged model, e.g. ``models:/<model id>``.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``
            (config's store when omitted).

    Returns:
        The pyfunc model: its ``predict(raw_problems)`` is the method's.

    Raises:
        ValueError: If the model's MLmodel names a loader other than ``CODE_MODEL_LOADER``.
    """
    use_store(**store)
    flavor = mlflow.models.get_model_info(model_uri).flavors.get("python_function", {})
    loader = flavor.get("loader_module")
    if loader != CODE_MODEL_LOADER:
        raise ValueError(
            f"{model_uri} loads with {loader}, not {CODE_MODEL_LOADER}: tagging methods are "
            "logged from code (log_method), and nothing else is loaded."
        )
    previous = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="pycache-") as cache:
        sys.pycache_prefix = cache
        try:
            return mlflow.pyfunc.load_model(model_uri)
        finally:
            sys.pycache_prefix = previous


def flat_record(record: RunRecord) -> dict[str, str]:
    """Flatten a run record into the parameters of a logged model: dotted keys, JSON values.

    Nested mappings become dotted keys (``git.commit``, ``environment.packages.numpy``) while
    their keys are MLflow parameter names without a dot; a mapping with any other key (e.g.
    thresholds keyed by ``"*special"``) stays one JSON value, as do lists and empty mappings.
    Every value is JSON, so ``json.loads`` reads it back exactly.

    Args:
        record: The run record of a frozen method.

    Returns:
        The parameters, in the order of the record's fields.

    Raises:
        ValueError: If a value is longer than MLflow keeps (``MAX_PARAM_VAL_LENGTH``): MLflow
            would cut it silently.
    """
    flat: dict[str, str] = {}
    _flatten("", record.to_dict(), flat)
    too_long = [key for key, value in flat.items() if len(value) > MAX_PARAM_VAL_LENGTH]
    if too_long:
        raise ValueError(
            f"The run record's {too_long} hold more than {MAX_PARAM_VAL_LENGTH} characters, "
            "which MLflow would cut: keep long entries in a file of the method folder."
        )
    return flat


def _flatten(key: str, value: object, flat: dict[str, str]) -> None:
    """Add ``value`` to ``flat`` under ``key``, a mapping of parameter names key by key."""
    extends = isinstance(value, Mapping) and bool(value)
    if extends and all(isinstance(name, str) and _PARAM_KEY.fullmatch(name) for name in value):
        for name, item in value.items():
            _flatten(f"{key}.{name}" if key else name, item, flat)
    else:
        flat[key] = json.dumps(value, sort_keys=True, allow_nan=False)


def standard_metrics(
    truth: pd.DataFrame, predictions: pd.DataFrame, tags: Sequence[str]
) -> dict[str, float]:
    """Score a method's predictions as every method run and evaluation run reports them.

    Args:
        truth: 0/1 true tags, one row per problem, with a column per focus tag, and per tag of
            ``ALL_TAGS`` when ``tags`` covers them all.
        predictions: The method's 0/1 predictions, with the index of ``truth`` and the columns
            ``tags``, in order.
        tags: The method's tags.

    Returns:
        ``macro_f1``, ``micro_f1``, then ``f1/<tag>``, ``precision/<tag>`` and
        ``recall/<tag>`` for each focus tag (``evaluate``); then, when ``tags`` covers
        ``ALL_TAGS``, ``all_tag_macro_f1``, ``all_tag_macro_f1_zero_absent`` and
        ``paper_tag_macro_f1`` (``all_tag_macro_f1``).

    Raises:
        ValueError: If the predictions' columns are not ``tags``, if ``tags`` lacks a focus tag,
            or if ``evaluate`` or ``all_tag_macro_f1`` refuses the frames.
    """
    if list(predictions.columns) != list(tags) or not set(FOCUS_TAGS) <= set(tags):
        raise ValueError(
            f"The predictions must have the columns {list(tags)}, in order, holding every focus "
            "tag."
        )
    focus = list(FOCUS_TAGS)
    scores = evaluate(truth[focus], predictions[focus])
    per_tag = scores.per_tag.loc[ALL_PROBLEMS]
    metrics = {"macro_f1": scores.macro_f1, "micro_f1": scores.micro_f1}
    for metric in ("f1", "precision", "recall"):
        metrics |= {f"{metric}/{tag}": float(per_tag.at[tag, metric]) for tag in focus}
    if set(ALL_TAGS) <= set(tags):
        metrics |= {
            "all_tag_macro_f1": all_tag_macro_f1(truth, predictions, ALL_TAGS)[0],
            "all_tag_macro_f1_zero_absent": all_tag_macro_f1(
                truth, predictions, ALL_TAGS, absent="zero"
            )[0],
            "paper_tag_macro_f1": all_tag_macro_f1(truth, predictions, PAPER_TAGS)[0],
        }
    return metrics


def part_metrics(
    labels: pd.DataFrame, predictions: pd.DataFrame, tags: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Score a frozen method's predictions of each part, under the name of its MLflow dataset.

    Args:
        labels: True tags of every problem, indexed by ``sample_id``, as ``standard_metrics``
            reads them.
        predictions: The frozen predictions, indexed by ``(part, sample_id)``, one column per
            tag of ``tags``.
        tags: The method's tags.

    Returns:
        ``standard_metrics`` of each part, by dataset name (``MLFLOW_METRIC_DATASETS``).
    """
    metrics = {}
    for part, dataset in MLFLOW_METRIC_DATASETS.items():
        rows = predictions.loc[part]
        metrics[dataset] = standard_metrics(labels.loc[rows.index], rows, tags)
    return metrics


def method_run_id(recipe_sha256: str, **store: Any) -> str | None:
    """Find the method run of a recipe in the ``tag-methods`` experiment.

    Args:
        recipe_sha256: The method's ``recipe_sha256``.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``
            (config's store and ``tag-methods`` when omitted).

    Returns:
        The id of the run tagged ``run_kind=method`` with this recipe, or None.

    Raises:
        ValueError: If ``recipe_sha256`` is not a sha256 hex digest, or if two method runs
            carry it.
    """
    return _method_run_id(use_store(**store), recipe_sha256)


def _method_run_id(experiment_id: str, recipe_sha256: str) -> str | None:
    """Find the method run of a recipe in an experiment."""
    if not _SHA256.fullmatch(recipe_sha256):
        raise ValueError(f"{recipe_sha256!r} is not a sha256 hex digest: pass a recipe_sha256.")
    runs = mlflow.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.{RUN_KIND_TAG} = 'method' AND tags.recipe_sha256 = '{recipe_sha256}'"
        ),
        output_format="list",
    )
    if len(runs) > 1:
        ids = [run.info.run_id for run in runs]
        raise ValueError(
            f"{len(runs)} method runs have the recipe {recipe_sha256}: {ids}. Delete all but "
            "one in the MLflow UI."
        )
    return runs[0].info.run_id if runs else None


def log_method(
    folder: Path,
    record: RunRecord,
    metrics_by_dataset: Mapping[str, Mapping[str, float]],
    *,
    split: pd.DataFrame,
    registry: Mapping[str, MethodSpec] = METHODS,
    **store: Any,
) -> str:
    """Log a frozen tagging method to MLflow once, with its dataset-level metrics.

    When the recipe has no method run yet, it starts one in ``tag-methods``, named and tagged
    after the method, and logs the method folder through the models-from-code wrapper: the
    logged model ``tagging_method``, its parameters ``flat_record(record)``, its signature
    ``raw_input_signature``, its requirements ``method_requirements``, and the run record's
    pinned prompts linked. A method whose dev score is read in sample
    (``method_entries["dev_in_sample"]``, ADR 0004) has its run and model tagged
    ``dev_in_sample=true``, and one proposed after a test score was read
    (``method_entries["test_informed"]``, docs/adr/0011) ``test_informed=true``. Then, in every
    case, it logs each dataset's metrics that the logged model has none of yet, on the model,
    with the dataset of the part's sample ids in ``split``: a dataset's metrics are logged once,
    and a second call duplicates nothing.

    A logged model is not always self-contained: one whose folder holds a ``pretrained.json``
    needs the weight cache (``config.PRETRAINED_CACHE_DIR``) on the machine that loads it, one
    that reads example texts needs the dataset, and a composite method needs its bases'
    committed method folders under ``config.METHODS_ASSETS_DIR`` (ADR 0004).

    Args:
        folder: The method folder, written by ``save_method``; it must hold data files only.
        record: The method's run record (``load_run_record``).
        metrics_by_dataset: Metrics by dataset name, ``dev-oof`` or ``test``
            (``MLFLOW_METRIC_DATASETS``), e.g. ``standard_metrics``' of each part.
        split: The split the metrics were computed on (``load_split`` of the freeze's
            ``--split``): each dataset holds the sample ids of its part.
        registry: The registered methods.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The id of the method's logged model.

    Raises:
        ValueError: If a dataset name is unknown, if the folder holds another method, if the
            method or its requirements are refused (``method_requirements``), if two method
            runs carry the recipe, or if the method run holds more than one model, or a model
            whose logging did not finish (``logged_model_id``).
        ConformanceError: If the folder holds anything but data files (``check_method_folder``).
    """
    unknown = sorted(set(metrics_by_dataset) - set(MLFLOW_METRIC_DATASETS.values()))
    if unknown:
        raise ValueError(
            f"No dataset is named {unknown}: the datasets of the metrics are "
            f"{list(MLFLOW_METRIC_DATASETS.values())}."
        )
    folder = Path(folder)
    check_method_folder(folder)
    arguments = _model_arguments(folder, record, registry)
    experiment_id = use_store(**store)
    run_id = _method_run_id(experiment_id, record.recipe_sha256)
    if run_id is None:
        tags = {
            RUN_KIND_TAG: "method",
            "method": record.method,
            "recipe_sha256": record.recipe_sha256,
        }
        tags |= _disclosure_tags(record)
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=record.method, tags=tags
        ) as run:
            run_id = run.info.run_id
    model_id = logged_model_id(run_id)
    if model_id is None:
        with mlflow.start_run(run_id=run_id):
            model_id = mlflow.pyfunc.log_model(**arguments).model_id
    _log_new_metrics(run_id, model_id, metrics_by_dataset, split)
    return model_id


def _model_arguments(
    folder: Path, record: RunRecord, registry: Mapping[str, MethodSpec]
) -> dict[str, Any]:
    """Return the arguments of ``mlflow.pyfunc.log_model``, checked before anything is logged."""
    stored = read_method_file(folder)
    if stored["method"] != record.method:
        raise ValueError(
            f"{folder} holds the method {stored['method']!r}, not {record.method!r}, the method "
            "of the run record."
        )
    requirements = method_requirements(record.method, registry=registry)
    return {
        "name": LOGGED_MODEL_NAME,
        "python_model": str(PYFUNC_ENTRY),
        "artifacts": {"method": str(folder)},
        "signature": raw_input_signature(stored["tags"], registry[record.method].input_columns),
        "pip_requirements": requirements,
        "params": flat_record(record),
        "prompts": list(record.prompts) or None,
        "tags": _disclosure_tags(record) or None,
    }


def _disclosure_tags(record: RunRecord) -> dict[str, str]:
    """Return the tags that disclose how far a method's scores can be trusted, if any.

    ``dev_in_sample``: its dev score is read in sample (ADR 0004); ``test_informed``: it was
    proposed after a test score was read (docs/adr/0011).
    """
    entries = record.method_entries
    tags = {"dev_in_sample": "true"} if entries.get("dev_in_sample") is True else {}
    if entries.get("test_informed"):
        tags["test_informed"] = "true"
    return tags


def logged_model_id(run_id: str) -> str | None:
    """Return the id of the model logged in a method run.

    Args:
        run_id: The method run (``method_run_id``).

    Returns:
        The id of the run's logged model, or None before ``log_method`` logged it.

    Raises:
        ValueError: If the run holds more than one model, or a model whose logging did not
            finish (FAILED, or PENDING after an interruption): MLflow links a model to its run
            before it saves its artifacts, so such a model may hold none.
    """
    model_ids = [output.model_id for output in mlflow.get_run(run_id).outputs.model_outputs]
    if len(model_ids) > 1:
        raise ValueError(f"The method run {run_id} holds {len(model_ids)} models, not one.")
    if not model_ids:
        return None
    status = mlflow.get_logged_model(model_ids[0]).status
    if status != LoggedModelStatus.READY:
        raise ValueError(
            f"The logged model {model_ids[0]} of the method run {run_id} is {status}, not READY: "
            "its logging stopped before its artifacts were saved. Delete the run in the MLflow "
            "UI, then run the method's freeze again."
        )
    return model_ids[0]


def _log_new_metrics(
    run_id: str,
    model_id: str,
    metrics_by_dataset: Mapping[str, Mapping[str, float]],
    split: pd.DataFrame,
) -> None:
    """Log each dataset's metrics on the logged model, unless it has some of that dataset.

    ``mlflow.log_metrics`` also writes them to the run, whose view (the runs table,
    ``search_runs``) keeps each key's latest value, by step then timestamp. The selection
    dataset, dev-oof, is logged last and every other one at an earlier timestamp than its, so a
    method run shows its out-of-fold dev scores, never its test scores (ADR 0001).
    """
    metrics_so_far = mlflow.get_logged_model(model_id).metrics or ()  # None before the first
    logged = {metric.dataset_name for metric in metrics_so_far}
    now = time.time_ns() // 1_000_000
    selected = [m.timestamp for m in metrics_so_far if m.dataset_name == _SELECTION_DATASET]
    earlier = min([now, *selected]) - 1
    new = [name for name in metrics_by_dataset if name not in logged]
    for name in sorted(new, key=lambda name: name == _SELECTION_DATASET):
        mlflow.log_metrics(
            dict(metrics_by_dataset[name]),
            run_id=run_id,
            timestamp=now if name == _SELECTION_DATASET else earlier,
            model_id=model_id,
            dataset=_part_dataset(name, split),
        )


def _part_dataset(name: str, split: pd.DataFrame) -> PandasDataset:
    """Return the dataset of a metrics dataset name: the sample ids of its part in the split."""
    part = {dataset: part for part, dataset in MLFLOW_METRIC_DATASETS.items()}[name]
    sample_ids = pd.DataFrame({"sample_id": split.index[split["part"] == part]})
    return from_pandas(sample_ids, name=name)


def log_run_extras(
    recipe_sha256: str,
    *,
    metrics: Mapping[str, float] | None = None,
    figures: Mapping[str, Figure] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    **store: Any,
) -> None:
    """Add metrics, figures and tables to a method run, skipping those it already has.

    A metric whose key the run has, or a figure or table whose artifact path it has, is
    skipped: a second call logs nothing, and a fresh store is rebuilt from the committed files.

    Args:
        recipe_sha256: The method's ``recipe_sha256``.
        metrics: Run metrics by key, e.g. ``latency/median_s``.
        figures: Matplotlib figures by artifact path, e.g. ``figures/latency.png``.
        tables: Frames by artifact path, e.g. ``latency.csv``, written with ``to_csv()``.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Raises:
        ValueError: If the recipe has no method run.
    """
    run_id = method_run_id(recipe_sha256, **store)
    if run_id is None:
        raise ValueError(
            f"No method run has the recipe {recipe_sha256}: run the method's freeze first."
        )
    client = mlflow.MlflowClient()
    logged = client.get_run(run_id).data.metrics
    for key, value in (metrics or {}).items():
        if key not in logged:
            # The client, unlike mlflow.log_metrics, links no metric to the run's logged model.
            client.log_metric(run_id, key, value)
    _log_new_artifacts(client, run_id, figures or {}, client.log_figure)
    texts = {path: table.to_csv() for path, table in (tables or {}).items()}
    _log_new_artifacts(client, run_id, texts, client.log_text)


def _log_new_artifacts(
    client: mlflow.MlflowClient,
    run_id: str,
    artifacts: Mapping[str, Any],
    log: Callable[[str, Any, str], object],
) -> None:
    """Log each artifact with ``log(run_id, artifact, path)``, unless the run has its path."""
    for path, artifact in artifacts.items():
        if not _has_artifact(client, run_id, path):
            log(run_id, artifact, path)


def _has_artifact(client: mlflow.MlflowClient, run_id: str, path: str) -> bool:
    """Tell whether a run holds an artifact at ``path``."""
    folder = posixpath.dirname(path) or None
    return path in {item.path for item in client.list_artifacts(run_id, folder)}
