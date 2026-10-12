"""The per-problem view: every problem, and each frozen method's tags for it, in MLflow.

``python -m codeforces_tags.tracking datasets`` copies the problems of each part, once, into an
evaluation dataset of the ``tag-methods`` experiment (``ensure_datasets``): ``codeforces-dev``
and ``codeforces-test``, one record per problem (``evaluation_records``) with its inputs, its
expected tags and its place in the split. MLflow keys a record on a hash of its inputs, and a
dataset's digest changes on every merge, even an identical one: so a dataset is never merged
again. Its tags hold the sha256 of its records (``content_sha256``) and of the dataset and split
they came from, and a later run only checks the stored records against them.

``python -m codeforces_tags.tracking evaluate --part dev|test`` scores a frozen method's
predictions on a part's dataset (``evaluate_method``): one run ``eval-<name>-<part>``, through
``mlflow.genai.evaluate``, with a ``predict_fn`` that looks the frozen tags up by sample id
(``frozen_predict_fn``: nothing is predicted again) and code scorers only (``SCORERS``). Each
problem gets one trace with its inputs, its expected and predicted tags, its scores and, for an
API method, the tokens and cost its call recorded. MLflow can lose rows silently, so the run is
marked complete only once ``check_traces`` has counted every trace and assessment; a complete
run is only checked again, never redone, and the test part is evaluated once per method
(ADR 0001). An incomplete run is refused until ``discard_evaluation_run`` (``python -m
codeforces_tags.tracking discard --run RUN_ID``) deletes it with its traces. An evaluation run
needs the method's logged model: ``unlogged_methods`` names the frozen methods whose model the
store lacks, so that ``tracking evaluate`` without ``--method`` refuses before its first run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Feedback
from mlflow.genai.datasets import EvaluationDataset
from mlflow.genai.scorers import Scorer, scorer
from mlflow.tracing.constant import CostKey, SpanAttributeKey, TokenUsageKey

from codeforces_tags.config import (
    ALL_TAGS,
    EVALUATION_COST_REL_TOL,
    EVALUATION_INPUT_FIELDS,
    FOCUS_TAGS,
    MLFLOW_DATASETS,
)
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    RUN_RECORD_FILE,
    load_per_problem,
    load_predictions,
    load_run_record,
)
from codeforces_tags.methods import METHODS, MethodSpec, method_dir
from codeforces_tags.provenance import RunRecord, sha256_json
from codeforces_tags.splitting import SPLIT_COLUMNS
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import (
    RUN_KIND_TAG,
    logged_model_id,
    method_run_id,
    standard_metrics,
)

# The dataset tag that holds the sha256 of a dataset's records, in sample id order; its other
# tags, dataset_sha256 and split_sha256, name the dataset folder and the split they came from.
CONTENT_TAG = "content_sha256"
# Set on an evaluation run once check_traces has passed: only such a run is skipped later.
COMPLETE_TAG = "evaluation_complete"
# The run_kind of an evaluation run; a method run's is "method".
EVALUATION_RUN_KIND = "evaluation"
# The expectations of each record, which each trace holds as assessments.
EXPECTATIONS = ("tags", "focus_tags")
# What a method chooses its tags from, in the outputs of its traces: all_tags_jaccard scores
# only a method that chooses from every tag of ALL_TAGS.
ALL_TAGS_VOCABULARY, FOCUS_VOCABULARY = "all tags", "focus tags"
# A run of backticks: a Markdown code fence must be longer than any run in the code it holds.
_BACKTICKS = re.compile(r"`+")
# The rationale of correct_<tag>, by (the tag is predicted, the tag is true).
_OUTCOMES = {
    (True, True): "TP: predicted, and a true tag",
    (True, False): "FP: predicted, but not a true tag",
    (False, True): "FN: not predicted, but a true tag",
    (False, False): "TN: not predicted, and not a true tag",
}


def evaluation_records(problems: pd.DataFrame, split: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn problems into the records of the evaluation datasets, one per problem.

    A record's inputs are its sample id and its ``EVALUATION_INPUT_FIELDS`` as text, because the
    MLflow UI shows inputs as Markdown: a missing field is ``""`` (a NaN blanks the Datasets
    page), each sample input and output is a code fence, and the solution a ``python`` fence.
    The inputs are only shown: an evaluation run looks the predictions up by sample id. A
    record's expectations are its tags, sorted, and its focus tags; its tags are its part, fold
    and group, as strings.

    Args:
        problems: Preprocessed problems (``load_dataset()``), indexed by ``sample_id``, with the
            ``EVALUATION_INPUT_FIELDS`` and ``tags``.
        split: The frozen split (``load_split()``) of the same problems.

    Returns:
        The records, in sample id order, each a dict of ``inputs``, ``expectations`` and
        ``tags``.

    Raises:
        ValueError: If ``problems`` lacks a column, or if it and ``split`` hold different sample
            ids.
    """
    columns = (*EVALUATION_INPUT_FIELDS, "tags")
    missing = [column for column in columns if column not in problems.columns]
    if missing:
        raise ValueError(f"The problems lack the columns {missing}: pass load_dataset().")
    if set(problems.index) != set(split.index):
        raise ValueError(
            "The problems and the split hold different sample ids: pass load_dataset() and "
            "load_split()."
        )
    return [
        _record(int(sample_id), problems.loc[sample_id], split.loc[sample_id])
        for sample_id in sorted(problems.index)
    ]


def _record(sample_id: int, problem: pd.Series, place: pd.Series) -> dict[str, Any]:
    """Return the record of one problem."""
    tags = sorted(problem["tags"])
    inputs = {field: _field_text(field, problem[field]) for field in EVALUATION_INPUT_FIELDS}
    return {
        "inputs": {"sample_id": sample_id, **inputs},
        "expectations": {"tags": tags, "focus_tags": [tag for tag in tags if tag in FOCUS_TAGS]},
        "tags": {column: str(place[column]) for column in SPLIT_COLUMNS},
    }


def _field_text(field: str, value: object) -> str:
    """Return a field of a problem as Markdown: "" when missing, code in code fences."""
    if isinstance(value, list):
        return "\n\n".join(_fenced(str(sample)) for sample in value)
    if pd.isna(value):
        return ""
    if field == "source_code":
        return _fenced(str(value), "python")
    return str(value)


def _fenced(text: str, language: str = "") -> str:
    """Wrap text in a Markdown code fence longer than any run of backticks it holds."""
    longest = max((len(run) for run in _BACKTICKS.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    body = text.rstrip("\n")
    return f"{fence}{language}\n{body}\n{fence}"


def ensure_datasets(
    problems: pd.DataFrame,
    split: pd.DataFrame,
    *,
    dataset_sha256: str,
    split_sha256: str,
    **store: Any,
) -> dict[str, EvaluationDataset]:
    """Make the evaluation dataset of each part once; later, only check it.

    A missing dataset is created with the tags ``content_sha256`` (``sha256_json`` of its
    records), ``dataset_sha256`` and ``split_sha256``, then given its records in one merge. A
    dataset that exists is never merged again (MLflow would change its records' keys and its
    digest): its tags and its records, rebuilt from ``to_df()``, must equal the ones these
    problems give, or it is refused, with the way to rebuild it.

    Args:
        problems: Preprocessed problems (``load_dataset()``), indexed by ``sample_id``.
        split: The frozen split (``load_split()``) of the same problems.
        dataset_sha256: The fingerprint of the dataset folder the problems were read from.
        split_sha256: The fingerprint of the split file.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The datasets by part, named as ``MLFLOW_DATASETS`` says.

    Raises:
        ValueError: If the records cannot be built (``evaluation_records``), if two datasets of
            the experiment share a name, or if a dataset holds other tags or records.
    """
    records = evaluation_records(problems, split)
    experiment_id = use_store(**store)
    datasets = {}
    for part, name in MLFLOW_DATASETS.items():
        in_part = [record for record in records if record["tags"]["part"] == part]
        tags = {
            CONTENT_TAG: sha256_json(in_part),
            "dataset_sha256": dataset_sha256,
            "split_sha256": split_sha256,
        }
        datasets[part] = _ensure_dataset(experiment_id, name, in_part, tags)
    return datasets


def _ensure_dataset(
    experiment_id: str, name: str, records: list[dict[str, Any]], tags: dict[str, str]
) -> EvaluationDataset:
    """Create a dataset with its records and tags, or check the one that exists."""
    dataset = _find_dataset(experiment_id, name)
    if dataset is None:
        created = mlflow.genai.datasets.create_dataset(
            name=name, experiment_id=experiment_id, tags=tags
        )
        return created.merge_records(records)
    changed = [key for key, value in tags.items() if dataset.tags.get(key) != value]
    if changed:
        raise ValueError(
            f"The evaluation dataset {name} has other {changed} tags than these problems and "
            f"this split give. {_rebuild(dataset)}"
        )
    if _stored_records(dataset) != records:
        raise ValueError(
            f"The evaluation dataset {name} holds other records than these problems and this "
            f"split give (edited in the MLflow UI?). {_rebuild(dataset)}"
        )
    return dataset


def _rebuild(dataset: EvaluationDataset) -> str:
    """Say how to rebuild a dataset that is refused."""
    return (
        "A dataset is never merged again: delete it (in the MLflow UI, or with "
        f"mlflow.genai.datasets.delete_dataset(dataset_id='{dataset.dataset_id}')), then run "
        "python -m codeforces_tags.tracking datasets again."
    )


def _find_dataset(experiment_id: str, name: str) -> EvaluationDataset | None:
    """Return the experiment's dataset of a name, or None."""
    found = mlflow.genai.datasets.search_datasets(
        experiment_ids=[experiment_id], filter_string=f"name = '{name}'"
    )
    if len(found) > 1:
        ids = [dataset.dataset_id for dataset in found]
        raise ValueError(
            f"{len(found)} evaluation datasets are named {name}: {ids}. Delete them, then run "
            "python -m codeforces_tags.tracking datasets again."
        )
    return found[0] if found else None


def _stored_records(dataset: EvaluationDataset) -> list[dict[str, Any]]:
    """Rebuild a dataset's records from ``to_df()``, in sample id order, without MLflow's tags."""
    records = [_stored_record(row) for row in dataset.to_df().to_dict("records")]
    return sorted(records, key=_sample_id)


def _stored_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild one record; an output added in the MLflow UI is kept, so it shows as a change."""
    tags = {
        key: value for key, value in (row["tags"] or {}).items() if not key.startswith("mlflow.")
    }
    record = {"inputs": row["inputs"], "expectations": row["expectations"], "tags": tags}
    if row["outputs"]:
        record["outputs"] = row["outputs"]
    return record


def _sample_id(record: Mapping[str, Any]) -> int:
    """Return a record's sample id, or -1 for a record that has none (added in the MLflow UI)."""
    sample_id = record["inputs"].get("sample_id")
    return sample_id if isinstance(sample_id, int) else -1


def _focus(tags: Sequence[str]) -> set[str]:
    """Return the focus tags among ``tags``."""
    return set(tags) & set(FOCUS_TAGS)


@scorer
def focus_exact_match(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> Feedback:
    """Tell whether a problem got exactly its focus tags.

    Args:
        outputs: The trace's outputs: the predicted ``tags``.
        expectations: The record's ``focus_tags``.

    Returns:
        True or False, with the focus tags missed and the extra ones as the rationale.
    """
    predicted, expected = _focus(outputs["tags"]), set(expectations["focus_tags"])
    missed, extra = sorted(expected - predicted), sorted(predicted - expected)
    return Feedback(value=not missed and not extra, rationale=f"missed {missed}, extra {extra}")


@scorer
def focus_f1(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float:
    """Score a problem's predicted focus tags by their F1.

    Args:
        outputs: The trace's outputs: the predicted ``tags``.
        expectations: The record's ``focus_tags``.

    Returns:
        The F1 of the predicted focus tags against the true ones; 1.0 when a problem has no
        focus tag and none is predicted.
    """
    predicted, expected = _focus(outputs["tags"]), set(expectations["focus_tags"])
    if not predicted and not expected:
        return 1.0
    return 2 * len(predicted & expected) / (len(predicted) + len(expected))


@scorer
def all_tags_jaccard(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> float | None:
    """Score a problem's predicted tags, all of them, by their Jaccard index.

    Args:
        outputs: The trace's outputs: the predicted ``tags`` and the method's ``vocabulary``.
        expectations: The record's ``tags``.

    Returns:
        The intersection over the union of the predicted and the true tags, 1.0 when both are
        empty; None, which records no assessment, for a method that does not choose from every
        tag.
    """
    if outputs["vocabulary"] != ALL_TAGS_VOCABULARY:
        return None
    predicted, expected = set(outputs["tags"]), set(expectations["tags"])
    union = predicted | expected
    return len(predicted & expected) / len(union) if union else 1.0


def _tag_scorer(tag: str) -> Scorer:
    """Return the scorer ``correct_<tag>`` (spaces as underscores) of one focus tag."""

    def correct(outputs: Mapping[str, Any], expectations: Mapping[str, Any]) -> Feedback:
        """Tell whether the tag is right, with TP, FP, FN or TN as the rationale."""
        predicted, expected = tag in outputs["tags"], tag in expectations["focus_tags"]
        return Feedback(value=predicted == expected, rationale=_OUTCOMES[predicted, expected])

    return scorer(correct, name=f"correct_{tag.replace(' ', '_')}")


# The code scorers of every evaluation run; no LLM judge.
SCORERS: list[Scorer] = [
    focus_exact_match,
    focus_f1,
    all_tags_jaccard,
    *(_tag_scorer(tag) for tag in FOCUS_TAGS),
]


def _vocabulary(tags: Sequence[str]) -> str:
    """Return what a method of these tags chooses from: every tag, or the focus tags."""
    return ALL_TAGS_VOCABULARY if set(ALL_TAGS) <= set(tags) else FOCUS_VOCABULARY


def _assessments_per_trace(tags: Sequence[str]) -> int:
    """Return how many assessments each trace of a method of these tags holds."""
    # all_tags_jaccard records nothing for a method that does not choose from every tag.
    skipped = 0 if _vocabulary(tags) == ALL_TAGS_VOCABULARY else 1
    return len(EXPECTATIONS) + len(SCORERS) - skipped


def frozen_predict_fn(
    name: str, part: str, *, registry: Mapping[str, MethodSpec] = METHODS
) -> Callable[..., dict[str, Any]]:
    """Return the ``predict_fn`` of a frozen method's evaluation run: a lookup, nothing re-run.

    Its keyword-only parameters are the input keys of a record, because MLflow calls
    ``predict_fn(**inputs)``; each trace then shows the problem's fields. When the method has a
    ``per_problem.csv``, it sets each problem's recorded call on the trace's span: the served
    model, the provider, the response id, the tokens, the cost (a dict, so that the trace's cost
    is set), the recorded latency and the status. MLflow first calls ``predict_fn`` once with
    tracing off, to check it: there is no span then, and nothing is set.

    Args:
        name: The method's registry key; its frozen files are in ``method_dir(name)``.
        part: ``dev`` or ``test``.
        registry: The registered methods.

    Returns:
        The function, traced under the method's name, which returns a problem's frozen ``tags``
        (sorted) and the method's ``vocabulary``.

    Raises:
        ValueError: If the method is not registered.
    """
    directory = _frozen_dir(name, registry)
    rows = load_predictions(directory).loc[part]
    vocabulary = _vocabulary(rows.columns)
    has_calls = (directory / PER_PROBLEM_FILE).is_file()
    calls = load_per_problem(directory).loc[part] if has_calls else None

    @mlflow.trace(name=name)
    def predict_fn(  # pylint: disable=unused-argument,too-many-arguments
        *,
        sample_id: int,
        prob_desc_description: str,
        prob_desc_input_spec: str,
        prob_desc_output_spec: str,
        prob_desc_notes: str,
        prob_desc_sample_inputs: str,
        prob_desc_sample_outputs: str,
        source_code: str,
    ) -> dict[str, Any]:
        """Return a problem's frozen tags, and set its recorded call on the span, if any."""
        span = mlflow.get_current_active_span()
        if calls is not None and span is not None:
            span.set_attributes(_call_attributes(calls.loc[sample_id]))
        predicted = rows.loc[sample_id]
        return {"tags": sorted(predicted.index[predicted == 1]), "vocabulary": vocabulary}

    return predict_fn


def _call_attributes(call: pd.Series) -> dict[str, Any]:
    """Return the span attributes of one recorded API call, a row of ``per_problem.csv``."""
    input_tokens, output_tokens = int(call["input_tokens"]), int(call["output_tokens"])
    return {
        SpanAttributeKey.MODEL: str(call["served_model"]),
        SpanAttributeKey.MODEL_PROVIDER: str(call["provider"]),
        "response_id": str(call["response_id"]),
        SpanAttributeKey.CHAT_USAGE: {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
            TokenUsageKey.CACHE_READ_INPUT_TOKENS: int(call["cached_tokens"]),
        },
        # A dict of costs: a float would leave the trace's cost None.
        SpanAttributeKey.LLM_COST: {
            CostKey.INPUT_COST: float(call["input_cost_usd"]),
            CostKey.OUTPUT_COST: float(call["output_cost_usd"]),
            CostKey.TOTAL_COST: float(call["total_cost_usd"]),
        },
        "recorded_latency_s": float(call["latency_s"]),
        "status": str(call["status"]),
    }


def evaluate_method(
    name: str, part: str, *, registry: Mapping[str, MethodSpec] = METHODS, **store: Any
) -> str:
    """Score a frozen method's predictions on a part's evaluation dataset, one trace per problem.

    The run, ``eval-<name>-<part>``, is tagged ``run_kind=evaluation``, ``method``, ``part``,
    ``recipe_sha256`` and ``dataset_content_sha256``, and linked to the run record's pinned
    prompts. ``mlflow.genai.evaluate`` gives each problem a trace (``frozen_predict_fn``,
    ``SCORERS``) and logs each scorer's mean to the run and to the method's logged model; the
    run also gets ``standard_metrics`` as run metrics, linked to no model, so the model keeps
    one copy of the dataset-level metrics, ``log_method``'s. The run is tagged
    ``evaluation_complete=true`` only after ``check_traces`` passes.

    A complete run of the same method, part and recipe is not redone: its traces are checked
    again, and its id returned. A run of them that is not complete is refused until
    ``discard_evaluation_run`` deletes it, and so is a second complete run: re-scoring frozen
    predictions chooses nothing, and the test part is evaluated once per method (ADR 0001).

    Args:
        name: The method's registry key; its frozen files are in ``method_dir(name)``.
        part: ``dev`` or ``test``.
        registry: The registered methods: a name outside it is refused before any file is read,
            since the runs are looked up by name.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The id of the complete evaluation run.

    Raises:
        FileNotFoundError: If the method is not frozen.
        ValueError: If the part is unknown or the method unregistered; if an evaluation run of
            the method and part is incomplete, or a second one would be complete; if the method
            has no logged model (or one whose logging did not finish), the store no dataset of
            the part, or the dataset changed or holds other problems than the predictions; or
            if ``check_traces`` refuses the run.
    """
    if part not in MLFLOW_DATASETS:
        raise ValueError(f"Unknown part {part!r}: expected one of {list(MLFLOW_DATASETS)}.")
    record = _frozen_record(name, registry)
    predictions = load_predictions(method_dir(name)).loc[part]
    run_id = _checked_complete_run(use_store(**store), name, part, record.recipe_sha256)
    if run_id is None:
        run_id = _evaluation_run(name, part, record, predictions, registry=registry, **store)
    check_traces(run_id, len(predictions), _assessments_per_trace(predictions.columns), **store)
    mlflow.MlflowClient().set_tag(run_id, COMPLETE_TAG, "true")
    return run_id


def _frozen_dir(name: str, registry: Mapping[str, MethodSpec]) -> Path:
    """Return a registered method's directory, refusing a name the registry does not hold."""
    if name not in registry:
        raise ValueError(
            f"{name!r} is not registered: the registry has {sorted(registry)}. Evaluation runs "
            "are looked up by the method's name, so only a registry key names a method."
        )
    return method_dir(name)


def _frozen_record(name: str, registry: Mapping[str, MethodSpec]) -> RunRecord:
    """Return a registered method's run record, refusing a method that is not frozen."""
    directory = _frozen_dir(name, registry)
    if not (directory / RUN_RECORD_FILE).is_file():
        raise FileNotFoundError(
            f"{directory} holds no {RUN_RECORD_FILE}: run the method's freeze first."
        )
    return load_run_record(directory)


def unlogged_methods(
    names: Sequence[str], *, registry: Mapping[str, MethodSpec] = METHODS, **store: Any
) -> list[str]:
    """Name the frozen methods whose logged model the store lacks.

    ``evaluate_method`` links each trace to the method's logged model, which the method's freeze
    (``llm score`` for a prompted method) logs in its method run. ``tracking evaluate`` without
    ``--method`` calls this before its first run, so that a store lacking a method is refused at
    once, not after the methods sorted before it were evaluated. It reads the run records and
    the store, and creates no run.

    Args:
        names: Registry keys of frozen methods.
        registry: The registered methods: a name outside it is refused, as ``evaluate_method``
            refuses it.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        The names, in their order, whose recipe has no method run, or a method run that holds no
        logged model yet.

    Raises:
        FileNotFoundError: If a method is not frozen.
        ValueError: If a method is not registered, if two method runs carry its recipe, or if
            its logged model is not READY: such a store is refused, not taken for one that lacks
            the method, since the method's freeze would refuse it too.
    """
    unlogged = []
    for name in names:
        run_id = method_run_id(_frozen_record(name, registry).recipe_sha256, **store)
        if run_id is None or logged_model_id(run_id) is None:
            unlogged.append(name)
    return unlogged


def _complete(run: Any) -> bool:
    """Tell whether an evaluation run is complete: check_traces passed on it."""
    return run.data.tags.get(COMPLETE_TAG) == "true"


def _checked_complete_run(
    experiment_id: str, name: str, part: str, recipe_sha256: str
) -> str | None:
    """Return the complete evaluation run of a recipe on a part, or None; refuse the others.

    Refused: an incomplete run of the recipe, a second complete one, and, on the test part, a
    complete run of another recipe.
    """
    runs = mlflow.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.{RUN_KIND_TAG} = '{EVALUATION_RUN_KIND}' AND tags.method = '{name}' "
            f"AND tags.part = '{part}'"
        ),
        output_format="list",
    )
    same = [run for run in runs if run.data.tags.get("recipe_sha256") == recipe_sha256]
    _refuse_incomplete_runs(same, name, part)
    if len(same) > 1:
        raise ValueError(
            f"{len(same)} complete evaluation runs of {name} on {part} have its recipe: "
            f"{[run.info.run_id for run in same]}. Delete all but one in the MLflow UI: a "
            "complete run is never redone."
        )
    if part == "test":
        _refuse_another_test_run([run for run in runs if run not in same], name)
    return same[0].info.run_id if same else None


def _refuse_another_test_run(other_recipes: Sequence[Any], name: str) -> None:
    """Refuse a new test evaluation when a run of another recipe completed one (ADR 0001)."""
    complete = [run.info.run_id for run in other_recipes if _complete(run)]
    if complete:
        raise ValueError(
            f"{name} already has a complete test evaluation run of another recipe, {complete}: "
            "the test part is evaluated once per method (ADR 0001), and a new recipe is a new "
            "method."
        )


def _refuse_incomplete_runs(runs: Sequence[Any], name: str, part: str) -> None:
    """Refuse evaluation runs that are not complete, naming the command that discards them."""
    incomplete = [run.info.run_id for run in runs if not _complete(run)]
    if incomplete:
        commands = "; ".join(_discard_command(run_id) for run_id in incomplete)
        raise ValueError(
            f"The evaluation runs {incomplete} of {name} on {part} are incomplete (they stopped, "
            f"or check_traces refused them): discard them ({commands}), which deletes their "
            "traces too, then evaluate again."
        )


def _discard_command(run_id: str) -> str:
    """Return the command that discards an incomplete evaluation run."""
    return f"python -m codeforces_tags.tracking discard --run {run_id}"


def discard_evaluation_run(run_id: str, **store: Any) -> int:
    """Delete an incomplete evaluation run and its traces, so that its part is evaluated again.

    Deleting a run in the MLflow UI keeps its traces, linked to the method's logged model: the
    next evaluation would show each problem twice. This deletes the traces, then the run. The
    run's ``<scorer>/mean`` metrics stay on the method's logged model, under the run's id: MLflow
    has no call that deletes a logged model's metrics.

    Args:
        run_id: An evaluation run that is not complete.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Returns:
        How many traces were deleted.

    Raises:
        ValueError: If the run is not an evaluation run, or is complete: a complete run is
            never redone (ADR 0001).
    """
    experiment_id = use_store(**store)
    run = mlflow.get_run(run_id)
    if run.data.tags.get(RUN_KIND_TAG) != EVALUATION_RUN_KIND:
        raise ValueError(f"The run {run_id} is not an evaluation run: it is not discarded.")
    if _complete(run):
        raise ValueError(
            f"The evaluation run {run_id} is complete: a complete evaluation run is never redone "
            "(ADR 0001)."
        )
    traces = mlflow.search_traces(
        locations=[experiment_id], run_id=run_id, return_type="list", include_spans=False
    )
    client = mlflow.MlflowClient()
    if traces:
        client.delete_traces(experiment_id, trace_ids=[trace.info.trace_id for trace in traces])
    client.delete_run(run_id)
    return len(traces)


def _evaluation_run(
    name: str,
    part: str,
    record: RunRecord,
    predictions: pd.DataFrame,
    *,
    registry: Mapping[str, MethodSpec],
    **store: Any,
) -> str:
    """Evaluate a frozen method on a part in a new run, and return the run's id."""
    model_id = _method_model_id(record.recipe_sha256, store)
    experiment_id = use_store(**store)
    dataset, stored = _checked_dataset(experiment_id, part)
    metrics = standard_metrics(_truth(stored, predictions), predictions, list(predictions.columns))
    run_tags = {
        RUN_KIND_TAG: EVALUATION_RUN_KIND,
        "method": name,
        "part": part,
        "recipe_sha256": record.recipe_sha256,
        "dataset_content_sha256": dataset.tags[CONTENT_TAG],
    }
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"eval-{name}-{part}", tags=run_tags
    ) as run:
        for prompt in record.prompts:
            mlflow.MlflowClient().link_prompt_version_to_run(run.info.run_id, prompt)
        mlflow.genai.evaluate(
            data=dataset,
            predict_fn=frozen_predict_fn(name, part, registry=registry),
            scorers=SCORERS,
            model_id=model_id,
        )
        _log_run_metrics(run.info.run_id, metrics)
    return run.info.run_id


def _log_run_metrics(run_id: str, metrics: Mapping[str, float]) -> None:
    """Log metrics to a run only: the client, unlike mlflow.log_metrics, links them to no model."""
    client = mlflow.MlflowClient()
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)


def _method_model_id(recipe_sha256: str, store: Mapping[str, Any]) -> str:
    """Return the logged model of a recipe's method run."""
    run_id = method_run_id(recipe_sha256, **store)
    model_id = None if run_id is None else logged_model_id(run_id)
    if model_id is None:
        raise ValueError(
            f"No method run of the recipe {recipe_sha256} holds a logged model: run the "
            "method's freeze first."
        )
    return model_id


def _checked_dataset(
    experiment_id: str, part: str
) -> tuple[EvaluationDataset, list[dict[str, Any]]]:
    """Return a part's dataset and its records, refusing one that changed since it was made."""
    name = MLFLOW_DATASETS[part]
    dataset = _find_dataset(experiment_id, name)
    if dataset is None:
        raise ValueError(
            f"The store has no evaluation dataset {name}: run python -m "
            "codeforces_tags.tracking datasets first."
        )
    stored = _stored_records(dataset)
    if sha256_json(stored) != dataset.tags.get(CONTENT_TAG):
        raise ValueError(
            f"The evaluation dataset {name} changed since tracking datasets made it: its "
            f"records no longer match its {CONTENT_TAG} tag. {_rebuild(dataset)}"
        )
    return dataset, stored


def _truth(stored: list[dict[str, Any]], predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the true tags of the dataset's records, aligned with the predictions."""
    expected = pd.Series(
        {_sample_id(record): record["expectations"]["tags"] for record in stored}, dtype=object
    )
    if sorted(expected.index) != sorted(predictions.index):
        raise ValueError(
            "The evaluation dataset holds other problems than the frozen predictions of its "
            "part: the split changed since one of them was made."
        )
    return multi_hot(expected.reindex(predictions.index), ALL_TAGS)


def check_traces(run_id: str, rows: int, per_trace_assessments: int, **store: Any) -> None:
    """Fail unless an evaluation run holds every trace, assessment and recorded cost.

    MLflow can lose evaluation rows and traces without an error, and records a scorer's failure
    as an assessment, so ``mlflow.genai.evaluate`` reports success either way: this counts the
    run's traces and each one's assessments, refuses an assessment that holds an error, and,
    when the method has a ``per_problem.csv``, compares the run's summed trace cost with its
    part's summed ``total_cost_usd`` (``math.isclose``, ``EVALUATION_COST_REL_TOL``).

    Args:
        run_id: The evaluation run.
        rows: The problems of its part: one trace each.
        per_trace_assessments: The assessments of each trace: the 2 expectations plus 10
            feedbacks for a method over the focus tags, 11 for one over every tag.
        **store: ``tracking_uri``, ``experiment`` and ``artifact_dir``, passed to ``use_store``.

    Raises:
        ValueError: If a trace or an assessment is missing or extra, if a scorer failed, or if
            the costs differ.
    """
    traces = mlflow.search_traces(
        locations=[use_store(**store)], run_id=run_id, return_type="list", include_spans=False
    )
    redo = f"Discard the run ({_discard_command(run_id)}), then evaluate again."
    if len(traces) != rows:
        raise ValueError(
            f"The evaluation run {run_id} holds {len(traces):,} traces, not one per problem "
            f"({rows:,}): MLflow lost or added some. {redo}"
        )
    wrong = [t.info.trace_id for t in traces if len(t.info.assessments) != per_trace_assessments]
    if wrong:
        raise ValueError(
            f"{len(wrong)} traces of the evaluation run {run_id} do not hold "
            f"{per_trace_assessments} assessments, e.g. {wrong[0]}. {redo}"
        )
    failed = [
        trace.info.trace_id
        for trace in traces
        if any(assessment.error is not None for assessment in trace.info.assessments)
    ]
    if failed:
        raise ValueError(
            f"A scorer failed on {len(failed)} traces of the evaluation run {run_id}, e.g. "
            f"{failed[0]}: read the error in the MLflow UI and fix the scorer. {redo}"
        )
    _check_costs(run_id, traces)


def _check_costs(run_id: str, traces: Sequence[Any]) -> None:
    """Fail unless the run's traces cost what the method's ``per_problem.csv`` records."""
    tags = mlflow.get_run(run_id).data.tags
    directory = method_dir(tags["method"])
    if not (directory / PER_PROBLEM_FILE).is_file():
        return
    recorded = math.fsum(load_per_problem(directory).loc[tags["part"], "total_cost_usd"])
    traced = math.fsum((trace.info.cost or {}).get(CostKey.TOTAL_COST, 0.0) for trace in traces)
    if not math.isclose(traced, recorded, rel_tol=EVALUATION_COST_REL_TOL):
        raise ValueError(
            f"The traces of the evaluation run {run_id} cost {traced} USD in all, but "
            f"per_problem.csv records {recorded} USD for {tags['part']}: the frozen calls "
            "changed since the run, or MLflow lost a cost."
        )
