"""Tests for the per-problem view: evaluation datasets, scorers and evaluation runs in MLflow."""

import dataclasses
import inspect
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import mlflow
import pandas as pd
import pytest
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
)
from mlflow.genai import scorer
from mlflow.genai.datasets import EvaluationDataset

from codeforces_tags import methods
from codeforces_tags.config import ALL_TAGS, EVALUATION_INPUT_FIELDS, FOCUS_TAGS, MLFLOW_DATASETS
from codeforces_tags.conformance import PICKLE_SUFFIXES, starts_like_pickle
from codeforces_tags.data.labels import multi_hot
from codeforces_tags.data.loading import load_dataset
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    freeze_predictions,
    load_predictions,
    stack_parts,
)
from codeforces_tags.evaluation.report import evaluate
from codeforces_tags.methods import MODEL_FOLDER, save_method
from codeforces_tags.provenance import make_run_record, sha256_json
from codeforces_tags.tracking import evaluation
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.evaluation import (
    ALL_TAGS_VOCABULARY,
    COMPLETE_TAG,
    CONTENT_TAG,
    EXPECTATIONS,
    FOCUS_VOCABULARY,
    SCORERS,
    check_traces,
    discard_evaluation_run,
    ensure_datasets,
    evaluate_method,
    evaluation_records,
    frozen_predict_fn,
    unlogged_methods,
)
from codeforces_tags.tracking.methods import log_method, method_run_id, standard_metrics

# Five problems, by sample id: 1-3 dev, 4-5 test. Problem 1 has no notes, problem 2 no tag and no
# input specification, problem 3 a solution that holds a code fence, problem 4 no sample test.
PROBLEMS = {
    1: {
        "prob_desc_description": "Count the leaves of a tree.",
        "prob_desc_notes": None,
        "tags": ["trees", "dfs and similar"],
    },
    2: {"prob_desc_input_spec": None, "tags": []},
    3: {"source_code": 'print("```")\n', "tags": ["number theory", "math", "greedy"]},
    4: {"prob_desc_sample_inputs": "[]", "prob_desc_sample_outputs": "[]", "tags": ["strings"]},
    5: {"tags": ["graphs", "trees"]},
}
# What each fake predicted for each problem.
PREDICTED = {
    "code_rule_fake": {1: ["trees"], 2: [], 3: ["implementation", "math"], 4: [], 5: ["trees"]},
    "keyword_fake": {
        1: ["dfs and similar", "trees"],
        2: ["greedy"],
        3: ["math"],
        4: [],
        5: ["trees"],
    },
}
# The fingerprints of the dataset folder and the split file that ensure_datasets records.
FINGERPRINTS = {"dataset_sha256": "d" * 64, "split_sha256": "5" * 64}
# The dataset-level metrics of a frozen method, as its freeze logs them.
METRICS = {"dev-oof": {"macro_f1": 0.5}, "test": {"macro_f1": 0.25}}
# The scorers, by name.
SCORER = {each.name: each for each in SCORERS}
# The feedback of a method over the focus tags: every scorer but all_tags_jaccard.
FOCUS_FEEDBACK = [name for name in SCORER if name != "all_tags_jaccard"]
# Edits of a record that the MLflow UI allows: other expected tags, or an output added.
UI_EDITS = [
    {"expectations": {"tags": ["graphs"], "focus_tags": ["graphs"]}},
    {"outputs": {"tags": ["math"]}},
]


class Env(NamedTuple):
    """What an evaluation test works with: problems and split, fakes, assets and a store."""

    data_dir: Path
    problems: pd.DataFrame
    split: pd.DataFrame
    registry: dict
    assets: Path
    store: dict


@pytest.fixture(name="data_dir")
def data_dir_fixture(write_samples, make_raw_record):
    """Write the five problems as the challenge's JSON files, and return their folder."""
    return write_samples(
        {sample_id: make_raw_record(**fields) for sample_id, fields in PROBLEMS.items()}
    )


@pytest.fixture(name="assets")
def assets_fixture(tmp_path, monkeypatch):
    """Return the folder of the frozen methods, which ``method_dir`` reads while the test runs."""
    assets = tmp_path / "assets"
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", assets)
    return assets


@pytest.fixture(name="env")
def env_fixture(data_dir, registry, assets, store):
    """Bundle the problems as ``load_dataset`` reads them, their split, the fakes and a store."""
    split = pd.DataFrame(
        {"part": ["dev"] * 3 + ["test"] * 2, "fold": [0, 1, 2, -1, -1], "group": [1, 2, 3, 4, 5]},
        index=pd.Index([1, 2, 3, 4, 5], name="sample_id"),
    )
    return Env(data_dir, load_dataset(data_dir), split, registry, assets, store)


def make_datasets(env, *, problems=None, split=None):
    """Make the evaluation datasets of the problems and split of ``env``, or of others."""
    return ensure_datasets(
        env.problems if problems is None else problems,
        env.split if split is None else split,
        **FINGERPRINTS,
        **env.store,
    )


def edit_a_record(env, dataset, change):
    """Change the record of problem 1 in a dataset, as an edit in the MLflow UI does."""
    [record, *_] = evaluation_records(env.problems, env.split)
    dataset.merge_records([{"inputs": record["inputs"], **change}])


def recorded_calls(split):
    """Return the API call of each problem, as an API method's per_problem.csv records it."""
    ids = list(split.index)
    calls = pd.DataFrame({"sample_id": ids, "part": split["part"].tolist(), "provider": "openai"})
    calls["input_tokens"] = [100 * sample_id for sample_id in ids]
    calls["cached_tokens"] = 10
    calls["output_tokens"] = [20 + sample_id for sample_id in ids]
    calls["input_cost_usd"] = [0.001 * sample_id for sample_id in ids]
    calls["output_cost_usd"] = [0.0003 * sample_id for sample_id in ids]
    calls["total_cost_usd"] = calls["input_cost_usd"] + calls["output_cost_usd"]
    calls["latency_s"] = [0.5 + sample_id for sample_id in ids]
    calls["served_model"] = "gpt-6-luna-2026-09-01"
    calls["response_id"] = [f"resp_{sample_id}" for sample_id in ids]
    calls["status"] = "ok"
    return calls


def write_frozen(env, fake, *, prompts=(), calls=False):
    """Write a fake's frozen files, as a freeze does: predictions, run record and method folder.

    With ``calls``, the fake also has the ``per_problem.csv`` of an API method. Returns the record.
    """
    directory = env.assets / fake.name
    rows = pd.DataFrame(
        [[int(tag in PREDICTED[fake.name][sid]) for tag in fake.tags] for sid in env.split.index],
        index=env.split.index,
        columns=list(fake.tags),
    )
    part = env.split["part"]
    record = make_run_record(
        fake.name,
        sha256_json([fake.name, "frozen"]),
        prompts=prompts,
        allow_dirty=True,
        method_entries={"supersedes": []},
        data_dir=env.data_dir,
    )
    stacked = stack_parts(dev=rows[part == "dev"], test=rows[part == "test"])
    freeze_predictions(stacked, directory, record)
    save_method(fake, directory / MODEL_FOLDER)
    if calls:
        recorded_calls(env.split).to_csv(directory / PER_PROBLEM_FILE, index=False)
    return record


def log_frozen(env, record):
    """Log a fake whose files ``write_frozen`` wrote, as its freeze would."""
    folder = env.assets / record.method / MODEL_FOLDER
    log_method(folder, record, METRICS, split=env.split, registry=env.registry, **env.store)


def freeze(env, fake, **frozen):
    """Write a fake's frozen files (``write_frozen``), log its method run, return its record."""
    record = write_frozen(env, fake, **frozen)
    log_frozen(env, record)
    return record


def run_traces(env, run_id):
    """Return the traces of a run, with their spans."""
    return mlflow.search_traces(
        locations=[use_store(**env.store)], run_id=run_id, return_type="list"
    )


def evaluation_runs(env):
    """Return the evaluation runs of the store."""
    return mlflow.search_runs(
        [use_store(**env.store)],
        filter_string="tags.run_kind = 'evaluation'",
        output_format="list",
    )


def experiment_traces(env, **where):
    """Return the traces of the store's experiment, of a run or a model when given one."""
    return mlflow.search_traces(
        locations=[use_store(**env.store)], return_type="list", include_spans=False, **where
    )


def evaluation_run_tags(record, part, **tags):
    """Return the tags of an evaluation run of ``code_rule_fake`` on a part, made by hand."""
    return {
        "run_kind": "evaluation",
        "method": "code_rule_fake",
        "part": part,
        "recipe_sha256": record.recipe_sha256,
    } | tags


def sample_id_of(trace):
    """Return the sample id of the problem a trace holds."""
    return trace.data.spans[0].inputs["sample_id"]


def test_evaluation_records_give_each_problem_its_inputs_expectations_and_place(env):
    records = evaluation_records(env.problems, env.split.iloc[::-1])

    assert [record["inputs"]["sample_id"] for record in records] == [1, 2, 3, 4, 5]
    problem = env.problems.loc[3]
    assert records[2] == {
        "inputs": {
            "sample_id": 3,
            "prob_desc_description": "Find $$$n$$$.",
            "prob_desc_input_spec": "One integer $$$n$$$.",
            "prob_desc_output_spec": "Print one integer.",
            "prob_desc_notes": problem["prob_desc_notes"],
            "prob_desc_sample_inputs": "```\n1\n4\n```",
            "prob_desc_sample_outputs": "```\n2\n```",
            "source_code": '````python\nprint("```")\n````',
        },
        "expectations": {
            "tags": ["greedy", "math", "number theory"],
            "focus_tags": ["math", "number theory"],
        },
        "tags": {"part": "dev", "fold": "2", "group": "3"},
    }
    assert tuple(records[0]["inputs"]) == ("sample_id", *EVALUATION_INPUT_FIELDS)


def test_evaluation_records_turn_missing_fields_into_empty_strings(env):
    records = {
        record["inputs"]["sample_id"]: record
        for record in evaluation_records(env.problems, env.split)
    }

    assert pd.isna(env.problems.loc[1, "prob_desc_notes"])
    assert records[1]["inputs"]["prob_desc_notes"] == ""
    assert records[2]["inputs"]["prob_desc_input_spec"] == ""
    assert records[4]["inputs"]["prob_desc_sample_inputs"] == ""
    assert records[2]["expectations"] == {"tags": [], "focus_tags": []}


def test_evaluation_records_refuse_a_split_of_other_problems(env):
    with pytest.raises(ValueError, match="different sample ids"):
        evaluation_records(env.problems, env.split.drop(index=5))


def test_evaluation_records_name_the_missing_columns(env):
    with pytest.raises(ValueError, match=r"\['source_code'\]"):
        evaluation_records(env.problems.drop(columns="source_code"), env.split)


def test_ensure_datasets_makes_a_dataset_per_part_holding_its_records(env):
    records = evaluation_records(env.problems, env.split)

    datasets = make_datasets(env)

    assert {part: dataset.name for part, dataset in datasets.items()} == MLFLOW_DATASETS
    for part, dataset in datasets.items():
        in_part = [record for record in records if record["tags"]["part"] == part]
        stored = mlflow.genai.datasets.get_dataset(dataset_id=dataset.dataset_id)
        assert {key: stored.tags[key] for key in (CONTENT_TAG, *FINGERPRINTS)} == {
            CONTENT_TAG: sha256_json(in_part),
            **FINGERPRINTS,
        }
        rows = sorted(stored.to_df().to_dict("records"), key=lambda row: row["inputs"]["sample_id"])
        assert [row["inputs"] for row in rows] == [record["inputs"] for record in in_part]
        assert [row["expectations"] for row in rows] == [
            record["expectations"] for record in in_part
        ]
        assert [row["tags"] | {"mlflow.user": None} for row in rows] == [
            record["tags"] | {"mlflow.user": None} for record in in_part
        ]


def test_ensure_datasets_twice_neither_merges_nor_changes_the_digest(env, monkeypatch):
    first = make_datasets(env)
    digests = {
        part: mlflow.genai.datasets.get_dataset(dataset_id=dataset.dataset_id).digest
        for part, dataset in first.items()
    }
    monkeypatch.setattr(EvaluationDataset, "merge_records", None)  # a merge fails the test
    monkeypatch.setattr(mlflow.genai.datasets, "create_dataset", None)

    second = make_datasets(env)

    assert {part: dataset.dataset_id for part, dataset in second.items()} == {
        part: dataset.dataset_id for part, dataset in first.items()
    }
    assert {
        part: mlflow.genai.datasets.get_dataset(dataset_id=dataset.dataset_id).digest
        for part, dataset in second.items()
    } == digests
    found = mlflow.genai.datasets.search_datasets(experiment_ids=[use_store(**env.store)])
    assert len(found) == 2


def test_ensure_datasets_refuses_problems_that_changed(env):
    make_datasets(env)
    changed = env.problems.copy()
    changed.loc[4, "prob_desc_description"] = "Reverse the string twice."

    with pytest.raises(ValueError, match=r"codeforces-test .*\['content_sha256'\].*delete_dataset"):
        make_datasets(env, problems=changed)


@pytest.mark.parametrize("change", UI_EDITS)
def test_ensure_datasets_refuses_a_record_edited_in_the_mlflow_ui(env, change):
    datasets = make_datasets(env)
    edit_a_record(env, datasets["dev"], change)

    with pytest.raises(ValueError, match="codeforces-dev .* other records"):
        make_datasets(env)


def test_ensure_datasets_refuses_records_of_another_split_file(env):
    make_datasets(env)

    with pytest.raises(ValueError, match=r"\['split_sha256'\]"):
        ensure_datasets(
            env.problems, env.split, **(FINGERPRINTS | {"split_sha256": "6" * 64}), **env.store
        )


def test_ensure_datasets_refuses_two_datasets_of_one_name(env):
    experiment_id = use_store(**env.store)
    for _ in range(2):
        mlflow.genai.datasets.create_dataset(name="codeforces-dev", experiment_id=experiment_id)

    with pytest.raises(ValueError, match="2 evaluation datasets are named codeforces-dev"):
        make_datasets(env)


def test_scorers_score_the_focus_tags_every_tag_and_each_focus_tag():
    assert list(SCORER) == [
        "focus_exact_match",
        "focus_f1",
        "all_tags_jaccard",
        "correct_math",
        "correct_graphs",
        "correct_strings",
        "correct_number_theory",
        "correct_trees",
        "correct_geometry",
        "correct_games",
        "correct_probabilities",
    ]


def test_focus_exact_match_names_the_missed_and_the_extra_focus_tags():
    wrong = SCORER["focus_exact_match"](
        outputs={"tags": ["dp", "math", "trees"], "vocabulary": ALL_TAGS_VOCABULARY},
        expectations={"tags": ["graphs", "math"], "focus_tags": ["graphs", "math"]},
    )
    right = SCORER["focus_exact_match"](
        outputs={"tags": ["dp"], "vocabulary": ALL_TAGS_VOCABULARY},
        expectations={"tags": ["greedy"], "focus_tags": []},
    )

    assert (wrong.value, wrong.rationale) == (False, "missed ['graphs'], extra ['trees']")
    assert (right.value, right.rationale) == (True, "missed [], extra []")


@pytest.mark.parametrize(
    ("predicted", "expected", "f1"),
    [
        (["graphs", "math"], ["math", "trees"], 0.5),
        ([], [], 1.0),
        (["dp"], [], 1.0),
        ([], ["math"], 0.0),
    ],
)
def test_focus_f1_scores_the_focus_tags_of_one_problem(predicted, expected, f1):
    outputs = {"tags": predicted, "vocabulary": ALL_TAGS_VOCABULARY}

    assert SCORER["focus_f1"](outputs=outputs, expectations={"focus_tags": expected}) == f1


def test_all_tags_jaccard_scores_only_a_method_over_every_tag():
    expectations = {"tags": ["math", "trees"], "focus_tags": ["math", "trees"]}
    jaccard = SCORER["all_tags_jaccard"]

    assert jaccard(
        outputs={"tags": ["dp", "math"], "vocabulary": ALL_TAGS_VOCABULARY},
        expectations=expectations,
    ) == pytest.approx(1 / 3)
    assert (
        jaccard(
            outputs={"tags": ["dp", "math"], "vocabulary": FOCUS_VOCABULARY},
            expectations=expectations,
        )
        is None
    )
    assert jaccard(
        outputs={"tags": [], "vocabulary": ALL_TAGS_VOCABULARY},
        expectations={"tags": [], "focus_tags": []},
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("predicted", "expected", "value", "rationale"),
    [
        (["number theory"], ["number theory"], True, "TP: predicted, and a true tag"),
        (["number theory"], [], False, "FP: predicted, but not a true tag"),
        ([], ["number theory"], False, "FN: not predicted, but a true tag"),
        ([], [], True, "TN: not predicted, and not a true tag"),
    ],
)
def test_correct_tag_says_whether_a_focus_tag_is_right_and_why(
    predicted, expected, value, rationale
):
    feedback = SCORER["correct_number_theory"](
        outputs={"tags": predicted, "vocabulary": FOCUS_VOCABULARY},
        expectations={"tags": expected, "focus_tags": expected},
    )

    assert (feedback.value, feedback.rationale) == (value, rationale)


def test_frozen_predict_fn_takes_the_inputs_of_a_record_as_keyword_only_parameters(fakes, env):
    write_frozen(env, fakes.CodeRuleFake())

    parameters = inspect.signature(
        frozen_predict_fn("code_rule_fake", "dev", registry=env.registry)
    ).parameters

    [record, *_] = evaluation_records(env.problems, env.split)
    assert tuple(parameters) == tuple(record["inputs"]) == ("sample_id", *EVALUATION_INPUT_FIELDS)
    assert {parameter.kind for parameter in parameters.values()} == {inspect.Parameter.KEYWORD_ONLY}


def test_frozen_predict_fn_looks_the_frozen_tags_up_with_tracing_off(fakes, env):
    # As in MLflow's first call, which checks predict_fn: no span, so no recorded call is set.
    write_frozen(env, fakes.KeywordFake(), calls=True)
    predict_fn = frozen_predict_fn("keyword_fake", "dev", registry=env.registry)
    records = evaluation_records(env.problems, env.split)

    mlflow.tracing.disable()
    try:
        outputs = [predict_fn(**record["inputs"]) for record in records[:3]]
    finally:
        mlflow.tracing.enable()

    assert outputs == [
        {"tags": ["dfs and similar", "trees"], "vocabulary": ALL_TAGS_VOCABULARY},
        {"tags": ["greedy"], "vocabulary": ALL_TAGS_VOCABULARY},
        {"tags": ["math"], "vocabulary": ALL_TAGS_VOCABULARY},
    ]


def test_evaluate_method_logs_one_trace_per_problem_with_its_expectations_and_feedback(fakes, env):
    datasets = make_datasets(env)
    record = freeze(env, fakes.CodeRuleFake())

    run_id = evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "eval-code_rule_fake-dev"
    assert {key: run.data.tags.get(key) for key in ("run_kind", "method", "part")} == {
        "run_kind": "evaluation",
        "method": "code_rule_fake",
        "part": "dev",
    }
    assert run.data.tags["recipe_sha256"] == record.recipe_sha256
    content = mlflow.genai.datasets.get_dataset(dataset_id=datasets["dev"].dataset_id).tags
    assert run.data.tags["dataset_content_sha256"] == content[CONTENT_TAG]
    assert run.data.tags[COMPLETE_TAG] == "true"
    assert run.data.tags["mlflow.runType"] == "genai_evaluate"
    traces = {sample_id_of(trace): trace for trace in run_traces(env, run_id)}
    assert sorted(traces) == [1, 2, 3]
    for trace in traces.values():
        names = sorted(assessment.name for assessment in trace.info.assessments)
        assert names == sorted([*EXPECTATIONS, *FOCUS_FEEDBACK])
    [record_3] = [
        r for r in evaluation_records(env.problems, env.split) if r["inputs"]["sample_id"] == 3
    ]
    assert traces[3].data.spans[0].inputs == record_3["inputs"]
    assert traces[3].data.spans[0].outputs == {
        "tags": ["implementation", "math"],
        "vocabulary": FOCUS_VOCABULARY,
    }
    truth = multi_hot(env.problems["tags"], ALL_TAGS).loc[[1, 2, 3]]
    predictions = load_predictions(env.assets / "code_rule_fake").loc["dev"]
    expected = standard_metrics(truth, predictions, list(predictions.columns))
    assert {key: run.data.metrics[key] for key in expected} == expected
    focus = list(FOCUS_TAGS)
    assert run.data.metrics["macro_f1"] == evaluate(truth[focus], predictions[focus]).macro_f1
    assert (MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()) == (
        1,
        False,
    )


def test_evaluate_method_traces_the_api_calls_prompts_and_jaccard_of_an_all_tag_method(fakes, env):
    make_datasets(env)
    use_store(**env.store)
    mlflow.genai.register_prompt(name="descode2tag", template="Tag it: {{ problem }}")
    freeze(env, fakes.KeywordFake(), prompts=("prompts:/descode2tag/1",), calls=True)

    run_id = evaluate_method("keyword_fake", "test", registry=env.registry, **env.store)

    calls = recorded_calls(env.split).set_index("sample_id")
    traces = run_traces(env, run_id)
    assert sorted(sample_id_of(trace) for trace in traces) == [4, 5]
    for trace in traces:
        call = calls.loc[sample_id_of(trace)]
        names = sorted(assessment.name for assessment in trace.info.assessments)
        assert names == sorted([*EXPECTATIONS, *SCORER])
        assert trace.info.cost == {
            "input_cost": call["input_cost_usd"],
            "output_cost": call["output_cost_usd"],
            "total_cost": call["total_cost_usd"],
        }
        assert trace.info.token_usage == {
            "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "total_tokens": call["input_tokens"] + call["output_tokens"],
            "cache_read_input_tokens": 10,
        }
        attributes = trace.data.spans[0].attributes
        assert attributes["mlflow.llm.model"] == "gpt-6-luna-2026-09-01"
        assert attributes["mlflow.llm.provider"] == "openai"
        assert attributes["response_id"] == call["response_id"]
        assert attributes["recorded_latency_s"] == call["latency_s"]
        assert attributes["status"] == "ok"
    run = mlflow.get_run(run_id)
    assert json.loads(run.data.tags["mlflow.linkedPrompts"]) == [
        {"name": "descode2tag", "version": "1"}
    ]
    assert "all_tag_macro_f1" in run.data.metrics


def test_evaluate_method_traces_the_copied_calls_of_a_composite_method_over_an_api_method(
    fakes, env
):
    # docs/adr/0011: a calibrated twin calls the API through its base and has no answers file;
    # its freeze copies its base's per_problem.csv, so each trace carries the base's call.
    make_datasets(env)
    freeze(env, fakes.KeywordFake(), calls=True)
    composite = dataclasses.replace(
        env.registry["code_rule_fake"], calls_api=True, depends_on=("keyword_fake",)
    )
    registry = env.registry | {"code_rule_fake": composite}
    record = write_frozen(env, fakes.CodeRuleFake())
    shutil.copyfile(
        env.assets / "keyword_fake" / PER_PROBLEM_FILE,
        env.assets / "code_rule_fake" / PER_PROBLEM_FILE,
    )
    folder = env.assets / "code_rule_fake" / MODEL_FOLDER
    log_method(folder, record, METRICS, split=env.split, registry=registry, **env.store)

    runs = {
        name: evaluate_method(name, "dev", registry=registry, **env.store)
        for name in ("keyword_fake", "code_rule_fake")
    }

    costs = {
        name: {sample_id_of(trace): trace.info.cost for trace in run_traces(env, run_id)}
        for name, run_id in runs.items()
    }
    assert sorted(costs["code_rule_fake"]) == [1, 2, 3]
    assert costs["code_rule_fake"] == costs["keyword_fake"]
    assert mlflow.get_run(runs["code_rule_fake"]).data.tags[COMPLETE_TAG] == "true"
    check_traces(runs["code_rule_fake"], 3, 2 + len(FOCUS_FEEDBACK), **env.store)


def test_evaluate_twice_adds_no_trace(fakes, env, monkeypatch):
    # Evaluating a part again re-checks its complete run, and logs nothing.
    make_datasets(env)
    freeze(env, fakes.CodeRuleFake())
    runs = [
        evaluate_method("code_rule_fake", part, registry=env.registry, **env.store)
        for part in ("dev", "test")
    ]
    checked = []
    check = evaluation.check_traces

    def spy(run_id, *args, **kwargs):
        checked.append(run_id)
        check(run_id, *args, **kwargs)

    monkeypatch.setattr(evaluation, "check_traces", spy)
    monkeypatch.setattr(mlflow.genai, "evaluate", None)  # evaluating again fails the test

    again = [
        evaluate_method("code_rule_fake", part, registry=env.registry, **env.store)
        for part in ("dev", "test")
    ]

    assert again == checked == runs
    assert len(evaluation_runs(env)) == 2
    assert [len(run_traces(env, run_id)) for run_id in runs] == [3, 2]
    client = mlflow.MlflowClient()
    assert [len(client.get_metric_history(run_id, "macro_f1")) for run_id in runs] == [1, 1]
    # The method's dataset-level metrics stay one per dataset, on its logged model.
    [model] = mlflow.search_logged_models([use_store(**env.store)], output_format="list")
    logged = mlflow.get_logged_model(model.model_id).metrics
    assert sorted((m.dataset_name, m.value) for m in logged if m.key == "macro_f1") == [
        ("dev-oof", 0.5),
        ("test", 0.25),
    ]
    stored = [path for path in env.store["artifact_dir"].rglob("*") if path.is_file()]
    assert not any(path.suffix in PICKLE_SUFFIXES or starts_like_pickle(path) for path in stored)


def test_evaluate_method_never_completes_a_run_whose_scorer_failed_and_then_refuses_it(
    fakes, env, monkeypatch
):
    make_datasets(env)
    freeze(env, fakes.CodeRuleFake())

    def broken(outputs):
        raise KeyError(outputs["missing"])

    monkeypatch.setattr(evaluation, "SCORERS", [*SCORERS[:-1], scorer(broken, name="broken")])

    with pytest.raises(ValueError, match="A scorer failed on 3 traces"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    [run] = evaluation_runs(env)
    assert COMPLETE_TAG not in run.data.tags
    monkeypatch.setattr(evaluation, "SCORERS", SCORERS)
    with pytest.raises(ValueError, match=rf"\['{run.info.run_id}'\] .* incomplete"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)


def test_discard_evaluation_run_lets_a_part_be_evaluated_again_with_one_trace_per_problem(
    fakes, env, monkeypatch
):
    # Deleting a run in the MLflow UI keeps its traces, linked to the method's logged model:
    # the next evaluation would show every problem twice.
    make_datasets(env)
    freeze(env, fakes.CodeRuleFake())
    broken = scorer(lambda outputs: outputs["missing"], name="broken")
    monkeypatch.setattr(evaluation, "SCORERS", [*SCORERS[:-1], broken])
    with pytest.raises(ValueError, match="A scorer failed"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)
    monkeypatch.setattr(evaluation, "SCORERS", SCORERS)
    [dead] = evaluation_runs(env)
    with pytest.raises(ValueError, match=f"discard --run {dead.info.run_id}"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    assert discard_evaluation_run(dead.info.run_id, **env.store) == 3

    run_id = evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)
    assert [run.info.run_id for run in evaluation_runs(env)] == [run_id]
    [model] = mlflow.search_logged_models([use_store(**env.store)], output_format="list")
    assert len(experiment_traces(env)) == len(experiment_traces(env, model_id=model.model_id)) == 3


def test_discard_evaluation_run_deletes_a_run_that_stopped_before_its_first_trace(fakes, env):
    record = write_frozen(env, fakes.CodeRuleFake())
    with mlflow.start_run(
        experiment_id=use_store(**env.store), tags=evaluation_run_tags(record, "test")
    ) as run:
        pass

    assert discard_evaluation_run(run.info.run_id, **env.store) == 0
    assert evaluation_runs(env) == []


def test_discard_evaluation_run_refuses_a_complete_run_and_a_method_run(fakes, env):
    make_datasets(env)
    record = freeze(env, fakes.CodeRuleFake())
    run_id = evaluate_method("code_rule_fake", "test", registry=env.registry, **env.store)

    with pytest.raises(ValueError, match="is complete: a complete evaluation run is never redone"):
        discard_evaluation_run(run_id, **env.store)
    with pytest.raises(ValueError, match="is not an evaluation run"):
        discard_evaluation_run(method_run_id(record.recipe_sha256, **env.store), **env.store)
    assert len(experiment_traces(env, run_id=run_id)) == 2


@pytest.mark.parametrize("name", ["/nonexistent/x", "code_rule_fake/", "../assets/code_rule_fake"])
def test_evaluate_method_refuses_a_name_that_is_not_registered(fakes, env, name):
    # A path-like alias of a frozen method would read its files, then miss its evaluation runs,
    # which are looked up by name: a second test evaluation (ADR 0001).
    write_frozen(env, fakes.CodeRuleFake())

    with pytest.raises(ValueError, match="is not registered"):
        evaluate_method(name, "test", registry=env.registry, **env.store)
    with pytest.raises(ValueError, match="is not registered"):
        frozen_predict_fn(name, "test", registry=env.registry)


def test_evaluate_method_scores_dev_again_for_a_new_recipe(fakes, env):
    # Only the test part is evaluated once per method: re-scoring dev after a recipe changed
    # chooses nothing on test.
    make_datasets(env)
    record = freeze(env, fakes.CodeRuleFake())
    other = evaluation_run_tags(record, "dev", recipe_sha256="0" * 64, **{COMPLETE_TAG: "true"})
    with mlflow.start_run(experiment_id=use_store(**env.store), tags=other):
        pass

    run_id = evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    assert mlflow.get_run(run_id).data.tags[COMPLETE_TAG] == "true"
    assert len(evaluation_runs(env)) == 2


@pytest.mark.parametrize(
    ("same_recipe", "message"),
    [
        (True, "2 complete evaluation runs of code_rule_fake on test .* Delete all but one"),
        (False, r"evaluated once per method \(ADR 0001\)"),
    ],
)
def test_evaluate_method_refuses_a_second_complete_test_run(fakes, env, same_recipe, message):
    # Two complete runs of the frozen recipe, or one of another recipe, made by hand.
    record = write_frozen(env, fakes.CodeRuleFake())
    experiment_id = use_store(**env.store)
    tags = {"run_kind": "evaluation", "method": "code_rule_fake", "part": "test"}
    recipes = [record.recipe_sha256] * 2 if same_recipe else ["0" * 64]
    for recipe_sha256 in recipes:
        complete = tags | {"recipe_sha256": recipe_sha256, COMPLETE_TAG: "true"}
        with mlflow.start_run(experiment_id=experiment_id, tags=complete):
            pass

    with pytest.raises(ValueError, match=message):
        evaluate_method("code_rule_fake", "test", registry=env.registry, **env.store)


def test_evaluate_method_refuses_a_method_with_no_method_run_then_with_no_dataset(fakes, env):
    record = write_frozen(env, fakes.CodeRuleFake())

    with pytest.raises(ValueError, match="run the method's freeze first"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    folder = env.assets / "code_rule_fake" / MODEL_FOLDER
    log_method(folder, record, METRICS, split=env.split, registry=env.registry, **env.store)
    with pytest.raises(ValueError, match="run python -m codeforces_tags.tracking datasets first"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)
    assert evaluation_runs(env) == []


def test_evaluate_method_refuses_a_dataset_edited_since_it_was_made(fakes, env):
    datasets = make_datasets(env)
    freeze(env, fakes.CodeRuleFake())
    edit_a_record(env, datasets["dev"], UI_EDITS[0])

    with pytest.raises(ValueError, match="codeforces-dev changed since"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    assert evaluation_runs(env) == []


def test_evaluate_method_refuses_a_dataset_of_other_problems(fakes, env):
    # Problem 3 is a test problem in the datasets, and a dev problem in the frozen predictions.
    make_datasets(env, split=env.split.assign(part=["dev", "dev", "test", "test", "test"]))
    freeze(env, fakes.CodeRuleFake())

    with pytest.raises(ValueError, match="other problems than the frozen predictions"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)

    assert evaluation_runs(env) == []


def test_evaluate_method_refuses_a_method_that_is_not_frozen(env):
    with pytest.raises(FileNotFoundError, match="run the method's freeze first"):
        evaluate_method("code_rule_fake", "dev", registry=env.registry, **env.store)


def test_evaluate_method_refuses_a_part_that_is_not_one(env):
    with pytest.raises(ValueError, match="Unknown part 'train'"):
        evaluate_method("code_rule_fake", "train", registry=env.registry, **env.store)


def test_check_traces_refuses_other_costs_lost_assessments_and_lost_traces(fakes, env):
    make_datasets(env)
    freeze(env, fakes.KeywordFake(), calls=True)
    run_id = evaluate_method("keyword_fake", "dev", registry=env.registry, **env.store)
    path = env.assets / "keyword_fake" / PER_PROBLEM_FILE
    calls = recorded_calls(env.split)

    calls.assign(total_cost_usd=calls["total_cost_usd"] * 2).to_csv(path, index=False)
    with pytest.raises(ValueError, match="per_problem.csv records"):
        check_traces(run_id, 3, 13, **env.store)
    calls.to_csv(path, index=False)
    check_traces(run_id, 3, 13, **env.store)

    [trace, *_] = run_traces(env, run_id)
    mlflow.delete_assessment(trace.info.trace_id, trace.info.assessments[0].assessment_id)
    with pytest.raises(ValueError, match="1 traces of the evaluation run .* not hold 13"):
        check_traces(run_id, 3, 13, **env.store)

    experiment_id = use_store(**env.store)
    mlflow.MlflowClient().delete_traces(experiment_id, trace_ids=[trace.info.trace_id])
    with pytest.raises(ValueError, match="holds 2 traces, not one per problem .3."):
        check_traces(run_id, 3, 13, **env.store)


def method_run_tags(record):
    """Return the tags of a fake's method run, as log_method tags it."""
    return {"run_kind": "method", "method": record.method, "recipe_sha256": record.recipe_sha256}


def test_unlogged_methods_names_the_frozen_methods_whose_recipe_has_no_logged_model(fakes, env):
    names = ["keyword_fake", "code_rule_fake"]
    records = [write_frozen(env, fakes.KeywordFake()), write_frozen(env, fakes.CodeRuleFake())]
    assert unlogged_methods(names, registry=env.registry, **env.store) == names

    log_frozen(env, records[1])
    assert unlogged_methods(names, registry=env.registry, **env.store) == ["keyword_fake"]
    # A method run that stopped before its model was logged holds none.
    mlflow.MlflowClient().create_run(use_store(**env.store), tags=method_run_tags(records[0]))
    assert unlogged_methods(names, registry=env.registry, **env.store) == ["keyword_fake"]

    log_frozen(env, records[0])
    assert not unlogged_methods(names, registry=env.registry, **env.store)
    assert evaluation_runs(env) == []


def test_unlogged_methods_lets_an_unfinished_model_and_two_method_runs_of_a_recipe_refuse(
    fakes, env, monkeypatch
):
    # Neither store is one that lacks the method: it is broken, and taking it for unlogged would
    # send the user to a freeze that refuses it too.
    record = write_frozen(env, fakes.CodeRuleFake())

    def full_disk(*args, **kwargs):
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(mlflow.MlflowClient, "log_model_artifacts", full_disk)
        with pytest.raises(OSError, match="No space left"):
            log_frozen(env, record)

    with pytest.raises(ValueError, match="is FAILED, not READY"):
        unlogged_methods(["code_rule_fake"], registry=env.registry, **env.store)

    mlflow.MlflowClient().create_run(use_store(**env.store), tags=method_run_tags(record))
    with pytest.raises(ValueError, match="2 method runs have the recipe"):
        unlogged_methods(["code_rule_fake"], registry=env.registry, **env.store)


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        ("code_rule_fake/", ValueError, "is not registered"),
        ("code_rule_fake", FileNotFoundError, "run the method's freeze first"),
    ],
)
def test_unlogged_methods_refuses_a_method_that_is_not_registered_or_not_frozen(
    env, name, error, message
):
    with pytest.raises(error, match=message):
        unlogged_methods([name], registry=env.registry, **env.store)
