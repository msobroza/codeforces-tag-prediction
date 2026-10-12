"""Tests for the command line: its commands, how it reads samples, predicts and evaluates.

The commands, the samples they read, the API runs and the evaluation's rows are one module, and
their tests share these fakes and fixtures, so this one file is longer than pylint's limit on a
module, as ``tests/llm/test_main.py`` is.

Every test runs offline: an API method is the frozen ``jev-statement-code`` behind a fake Jev
session (``cli_jev``) or a fake method object, its dependency group counts as installed through a
stand-in ``missing_groups``, and its key is set to a sentinel or removed with ``monkeypatch``; no
request is sent and no key is read. ``cli_challenge`` stands in for the challenge dataset.
"""

# pylint: disable=too-many-lines

import json
import math
import os
import subprocess
import sys
import types
from dataclasses import replace

import pandas as pd
import pytest

from codeforces_tags import cli, cli_methods
from codeforces_tags.config import (
    API_KEY_ENV_VARS,
    CLI_DEFAULT_METHOD,
    CLI_JSON_LIST_FIELDS,
    DATA_DIR_ENV_VAR,
    FOCUS_TAGS,
    LLM_PRICES,
    METHODS_ASSETS_DIR,
    RAW_INPUT_COLUMNS,
)
from codeforces_tags.data.loading import read_samples
from codeforces_tags.evaluation.report import (
    REPORT_FILE,
    REPORT_TABLES,
    evaluate,
    report_headline,
    report_text,
)
from codeforces_tags.evaluation.timing import load_latency
from codeforces_tags.llm.answers import answer_cost_usd
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.methods import METHODS, MODEL_FOLDER, MethodSpec, load_method
from codeforces_tags.methods.typesafe_jev.settings import MODEL as JEV_MODEL
from codeforces_tags.splitting import load_split

# What no fresh import of the command line may load: the libraries of the methods and their
# tracking, those of the notebooks and the audit, and the packages that set MLflow's environment
# or import it (codeforces_tags.llm imports tracking.environment).
METHOD_LIBRARIES = ("mlflow", "optuna", "gepa", "torch", "transformers", "openai", "typesafe_sdk")
OTHER_LIBRARIES = ("matplotlib", "seaborn", "claude_agent_sdk")
MLFLOW_PACKAGES = ("codeforces_tags.llm", "codeforces_tags.tracking", "codeforces_tags.tuning")
BANNED_MODULES = METHOD_LIBRARIES + OTHER_LIBRARIES + MLFLOW_PACKAGES
# Runs the code argv[2] in a fresh interpreter, then prints on its last line, as JSON, the names of
# the JSON list argv[1] that it loaded: a package counts when it or any submodule is loaded.
LOADED_BANNED_MODULES = """
import json, sys
exec(sys.argv[2])
loaded = {name.split(".")[0] for name in sys.modules} | set(sys.modules)
packages = {".".join(name.split(".")[:2]) for name in sys.modules}
print(json.dumps(sorted(set(json.loads(sys.argv[1])) & (loaded | packages))))
"""
# A module that no test may import: a spec that names it proves that nothing was imported.
NEVER_IMPORTED = "codeforces_tags.no_such_module"
# The key of the TypeSafe methods, set to this sentinel or removed; never a real key.
KEY = API_KEY_ENV_VARS["llm-typesafe"]
SENTINEL = "sentinel-key-never-sent"


def test_challenge_labels_refuse_a_missing_directory_naming_the_option_and_the_variable(tmp_path):
    with pytest.raises(cli.CliError) as error:
        cli.challenge_labels(tmp_path / "missing")

    assert str(error.value) == (
        f"{tmp_path / 'missing'} does not exist: --data DIR must be a directory of "
        "sample_<id>.json files (default: $CODEFORCES_DATA_DIR)."
    )


def test_challenge_labels_refuse_a_directory_that_is_not_the_challenge_dataset(
    write_samples, make_raw_record
):
    data_dir = write_samples({1: make_raw_record()})

    with pytest.raises(cli.CliError) as error:
        cli.challenge_labels(data_dir)

    assert str(error.value).startswith(
        f"{data_dir} is not the challenge dataset: the dataset has 1 problem files"
    )
    assert str(error.value).endswith(
        "Point $CODEFORCES_DATA_DIR at the challenge dataset, unchanged, or give it as --data DIR."
    )


def test_challenge_labels_are_the_focus_labels_of_every_problem_of_the_dataset(cli_challenge):
    labels = cli.challenge_labels(cli_challenge["data_dir"])

    pd.testing.assert_frame_equal(labels, cli_challenge["labels"], check_names=False)


def test_main_reports_a_cli_error_on_one_line_of_stderr_with_status_1(
    tmp_path, monkeypatch, capsys
):
    def refuse(*args, **kwargs):
        raise cli.CliError("no table today")

    monkeypatch.setattr(cli, "method_table", refuse)
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path / "missing"))

    status = cli.main(["methods"])

    assert status == 1
    assert capsys.readouterr().err.splitlines()[-1] == "codeforces-tags: error: no table today"


def test_a_command_line_without_a_command_is_a_usage_error_with_status_2(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    errors = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert errors.startswith("usage: codeforces-tags [-h] COMMAND ...")
    assert errors.endswith("error: the following arguments are required: COMMAND\n")


@pytest.mark.parametrize(
    ("command", "usage"),
    [
        ("methods", "usage: codeforces-tags methods [-h] [--data DIR]"),
        ("predict", "usage: codeforces-tags predict [-h] [--method M] [--max-cost-usd USD] INPUT"),
        (
            "evaluate",
            "usage: codeforces-tags evaluate [-h] [--part {test,dev,all}] [--out DIR] "
            "[--method M] [--max-cost-usd USD] [DATA]",
        ),
    ],
)
def test_each_command_shows_its_usage_and_says_what_it_does(capsys, command, usage):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])

    shown = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert " ".join(shown.split("\n\n")[0].split()) == usage
    assert shown.split("\n\n")[1].strip()  # a description, before the options


def test_the_help_of_the_command_line_is_plain_text():
    shown = cli.build_parser().format_help()

    assert "``" not in shown
    assert "--method tfidf" in shown


def test_a_reader_that_stops_early_ends_the_command_without_a_traceback(monkeypatch, capsys):
    # predict | head -1: the next print raises BrokenPipeError; stdout then goes to /dev/null,
    # so that Python's flush at exit does not print the error again.
    def broken(_):
        raise BrokenPipeError(32, "Broken pipe")

    redirected = []
    monkeypatch.setattr(cli, "_methods", broken)
    monkeypatch.setattr(cli.os, "open", lambda path, flags: (path, flags))
    monkeypatch.setattr(cli.os, "dup2", lambda source, target: redirected.append((source, target)))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(fileno=lambda: 99))

    status = cli.main(["methods"])

    assert (status, redirected) == (1, [((os.devnull, os.O_WRONLY), 99)])
    assert capsys.readouterr().err == ""


def test_predict_into_a_closed_pipe_exits_1_with_nothing_on_stderr(sample_7):
    command = [sys.executable, "-m", "codeforces_tags", "predict", str(sample_7), "--method"]
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        [*command, "label_prior"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.stdout.close()  # the reader is gone before the first line

    errors = process.stderr.read().decode()
    process.stderr.close()

    assert (process.wait(), errors) == (1, "")


def loaded_banned_modules(code):
    """Run Python code in a fresh interpreter; return the banned modules it loaded."""
    run = subprocess.run(
        [sys.executable, "-c", LOADED_BANNED_MODULES, json.dumps(BANNED_MODULES), code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout.splitlines()[-1])


def test_a_fresh_import_of_the_command_line_loads_no_heavy_library_and_no_mlflow_package():
    assert loaded_banned_modules("import codeforces_tags.cli, codeforces_tags.__main__") == []


# A field a test leaves out of a sample file, rather than writing it as null.
MISSING = object()
# What an SDK error's text holds in the tests: an echo of the state, which no message may show.
SECRET = "state-echoed-by-the-sdk"


class RecordingMethod:  # pylint: disable=too-few-public-methods
    """A local method that predicts no tag and records the columns it was handed."""

    name = "recording"
    tags = FOCUS_TAGS

    def __init__(self):
        """Start with no call."""
        self.columns = []

    def predict(self, raw_problems):
        """Record the columns, and predict no tag."""
        self.columns.append(list(raw_problems.columns))
        return pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags))


class WideMethod:
    """A local score method with a tag besides the focus tags: every tag, scored 1.0."""

    name = "wide"
    tags = (*FOCUS_TAGS, "greedy")

    def predict(self, raw_problems):
        """Predict from the scores, as every score method does: every tag."""
        return self.predict_from_scores(self.scores(raw_problems))

    def scores(self, raw_problems):
        """Score every tag 1.0."""
        return pd.DataFrame(1.0, index=raw_problems.index, columns=list(self.tags))

    def predict_from_scores(self, scores):
        """Predict the tags scored at least 0.5."""
        return (scores >= 0.5).astype(int)

    def tag_thresholds(self):
        """Stand for the threshold of every tag, 0.5: the command line never asks for it."""

    def save(self, _folder):
        """Save nothing: the method is never frozen."""


class CountingScoreMethod(WideMethod):
    """A local score method that records its calls: math scores a hundredth of the sample id."""

    name = "counting"
    tags = FOCUS_TAGS

    def __init__(self):
        """Start with no call."""
        self.calls = []

    def predict(self, raw_problems):
        """Record the call, then predict from the scores, as every score method does."""
        self.calls.append("predict")
        return self.predict_from_scores(self.scores(raw_problems))

    def scores(self, raw_problems):
        """Record the call; score math with a hundredth of each sample id, every other tag 0."""
        self.calls.append("scores")
        scores = pd.DataFrame(0.0, index=raw_problems.index, columns=list(self.tags))
        scores["math"] = raw_problems.index.to_numpy(dtype=float) / 100
        return scores

    def predict_from_scores(self, scores):
        """Predict the tags scored at least 0.15."""
        return (scores >= 0.15).astype(int)

    def tag_thresholds(self):
        """Stand for the threshold of every tag, 0.15: the command line never asks for it."""


class FakeApiMethod:  # pylint: disable=too-few-public-methods
    """A method that calls an API, faked: math for every problem, one ``make_answer`` each.

    Each answer bills ``tokens[sample_id]`` input tokens, or ``make_answer``'s 100 for a problem
    that ``tokens`` lacks. It has no ``scores_from_answers``, so the command line prints no score
    for it.
    """

    name = "fake-api"
    tags = FOCUS_TAGS

    def __init__(self, build, tokens=None):
        """Bind the answer builder and the tokens of each problem; no batch asked yet."""
        self.build = build
        self.tokens = tokens or {}
        self.batches = []

    async def apredict_with_answers(self, raw_problems):
        """Record the batch, predict math for each problem, and answer each."""
        self.batches.append(raw_problems.index.tolist())
        tags = pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags)).assign(math=1)
        return tags, [self.answer(int(sample_id)) for sample_id in raw_problems.index]

    def answer(self, sample_id):
        """Build the answer to one problem, billed its tokens."""
        usage = {"input_tokens": self.tokens.get(sample_id, 100), "cached_tokens": 0}
        usage |= {"output_tokens": 5, "reasoning_tokens": 0}
        return self.build(sample_id=sample_id, usage=usage)


class ScoringApiMethod(FakeApiMethod):  # pylint: disable=too-few-public-methods
    """A ``FakeApiMethod`` that scores its answers too: a problem's math score is its sample id."""

    def scores_from_answers(self, answers, index):
        """Score math with each answer's sample id, every other tag 0.0, in the order of index."""
        math_scores = {answer.sample_id: float(answer.sample_id) for answer in answers}
        scores = pd.DataFrame(0.0, index=index, columns=list(self.tags))
        scores["math"] = [math_scores[sample_id] for sample_id in index]
        return scores


def predict(capsys, *args):
    """Run ``predict`` with the arguments; return its status, JSON lines and lines of stderr."""
    status = cli.main(["predict", *map(str, args)])
    output, errors = capsys.readouterr()
    return status, [json.loads(line) for line in output.splitlines()], errors.splitlines()


@pytest.fixture(name="loaded")
def loaded_fixture(monkeypatch):
    """Record the name of every method the command line loads, then load it."""
    names, load_cli_method = [], cli.load_cli_method

    def load(name, **kwargs):
        names.append(name)
        return load_cli_method(name, **kwargs)

    monkeypatch.setattr(cli, "load_cli_method", load)
    return names


def no_group_missing(*_, **__):
    """Stand in for ``missing_groups``: every group of every method counts as installed."""
    return []


@pytest.fixture(name="jev_runs_here")
def jev_runs_here_fixture(monkeypatch):
    """Let the Jev methods run: every group counts as installed, and the key is a sentinel."""
    monkeypatch.setattr(cli_methods, "missing_groups", no_group_missing)
    monkeypatch.setenv(KEY, SENTINEL)


@pytest.fixture(name="sample_7")
def sample_7_fixture(write_samples, make_raw_record):
    """Return the path of a sample file, ``sample_7.json``, with every raw field."""
    return write_samples({7: make_raw_record()}) / "sample_7.json"


def test_predict_with_tfidf_prints_each_samples_tags_and_scores_as_a_json_line(
    write_samples, make_raw_record, loaded, capsys
):
    data_dir = write_samples(
        {
            3: make_raw_record(),
            1: make_raw_record(prob_desc_description="Reverse the string $$$s$$$ of letters."),
        }
    )

    status, lines, errors = predict(capsys, data_dir, "--method", "tfidf")

    method = load_method(METHODS_ASSETS_DIR / "tfidf" / MODEL_FOLDER)
    raw = read_samples(data_dir)[list(RAW_INPUT_COLUMNS)]
    tags, scores = method.predict(raw), method.scores(raw)
    assert (status, errors, loaded) == (0, [], ["tfidf"])
    assert [line["sample_id"] for line in lines] == [1, 3]
    for line in lines:
        sample_id = line["sample_id"]
        assert list(line) == ["sample_id", "tags", "scores"]
        assert line["tags"] == [tag for tag in FOCUS_TAGS if tags.loc[sample_id, tag] == 1]
        assert line["scores"] == scores.loc[sample_id].to_dict()


def test_predict_with_a_method_without_scores_prints_its_tags_only(
    write_samples, make_raw_record, capsys
):
    data_dir = write_samples({4: make_raw_record()})

    status, [line], _ = predict(capsys, data_dir / "sample_4.json", "--method", "label_prior")

    assert status == 0
    assert list(line) == ["sample_id", "tags"]


def test_predict_reads_a_missing_optional_field_as_the_datasets_null(write_samples, capsys):
    # A hand-written sample often holds the description and the solution only.
    fields = {
        "prob_desc_description": "Given a tree with $$$n$$$ vertices, count its leaves.",
        "source_code": "n = int(input())\nprint(n - 1)",
    }
    nulls = {column: None for column in RAW_INPUT_COLUMNS if column not in fields}
    data_dir = write_samples({1: fields, 2: fields | nulls})

    status_missing, [missing], _ = predict(capsys, data_dir / "sample_1.json", "--method", "tfidf")
    status_null, [null], _ = predict(capsys, data_dir / "sample_2.json", "--method", "tfidf")

    assert (status_missing, status_null) == (0, 0)
    assert missing | {"sample_id": 2} == null
    assert list(cli.read_input(data_dir / "sample_1.json").columns) == [*fields, *nulls]


@pytest.mark.parametrize("value", [MISSING, None, 42, " \n\t"])
def test_predict_refuses_a_sample_without_a_required_field(
    write_samples, make_raw_record, loaded, capsys, value
):
    # Refused with the file and the field named, before any method loads.
    record = make_raw_record(source_code=value)
    if value is MISSING:
        del record["source_code"]
    path = write_samples({5: record}) / "sample_5.json"

    status, lines, errors = predict(capsys, path, "--method", "tfidf")

    assert (status, lines, loaded) == (1, [], [])
    assert errors == [
        "codeforces-tags: error: every sample needs a text in prob_desc_description and "
        "source_code, but source_code holds none in sample_5.json."
    ]


@pytest.mark.parametrize(
    "field_value",
    [
        {"prob_desc_notes": 42},
        {"prob_desc_input_spec": ["The first line holds n."]},
        {"prob_desc_sample_inputs": "not a JSON list"},
        {"prob_desc_sample_inputs": "[1, 2]"},
        {"prob_desc_sample_outputs": ["2", 3]},
    ],
)
def test_predict_refuses_an_optional_field_that_no_method_can_read(
    write_samples, make_raw_record, loaded, capsys, field_value
):
    # Refused with the file and the field named, rather than a traceback from preprocessing.
    path = write_samples({5: make_raw_record(**field_value)}) / "sample_5.json"
    [field] = field_value
    kind = "a list of texts (or its JSON text)" if field in CLI_JSON_LIST_FIELDS else "a text"

    status, lines, errors = predict(capsys, path, "--method", "tfidf")

    assert (status, lines, loaded) == (1, [], [])
    assert errors == [
        "codeforces-tags: error: a sample holds a field that no method can read: "
        f"{field} holds neither {kind} nor null in sample_5.json."
    ]


def test_predict_reads_a_list_of_sample_inputs_as_the_json_text_that_the_dataset_holds(
    write_samples, make_raw_record, capsys
):
    io = {"prob_desc_sample_inputs": ["3 2\n1 3", "4 0"], "prob_desc_sample_outputs": ["2", "0"]}
    as_json = {field: json.dumps(texts) for field, texts in io.items()}
    data_dir = write_samples({1: make_raw_record(**io), 2: make_raw_record(**as_json)})

    status, [listed, written], errors = predict(capsys, data_dir, "--method", "tfidf")

    raw = cli.read_input(data_dir)
    assert (status, errors) == (0, [])
    assert listed | {"sample_id": 2} == written
    assert raw.loc[1, list(CLI_JSON_LIST_FIELDS)].tolist() == list(as_json.values())


def test_check_fields_names_every_sample_and_field_that_lacks_a_text(raw_problems):
    raw = raw_problems.assign(prob_desc_description=["", "Reverse it.", None])
    raw.loc[10, "source_code"] = "   "

    with pytest.raises(cli.CliError) as error:
        cli.check_fields(raw.drop(columns="prob_desc_notes"))

    assert str(error.value).endswith(
        "but prob_desc_description holds none in sample_30.json, sample_20.json; source_code "
        "holds none in sample_10.json."
    )


def test_check_fields_names_at_most_ten_files_per_field_and_counts_the_others():
    raw = pd.DataFrame(
        {"prob_desc_description": ["Count the leaves."] * 12, "source_code": [""] * 12},
        index=pd.Index(range(1, 13), name="sample_id"),
    )

    with pytest.raises(cli.CliError) as error:
        cli.check_fields(raw)

    named = ", ".join(f"sample_{sample_id}.json" for sample_id in range(1, 11))
    assert str(error.value).endswith(f"but source_code holds none in {named} and 2 more.")


def test_read_input_adds_each_raw_column_that_no_sample_holds_as_null_and_keeps_the_others(
    write_samples, make_raw_record
):
    records = {1: make_raw_record(), 2: make_raw_record()}
    for record in records.values():
        del record["prob_desc_notes"]
    del records[2]["prob_desc_input_spec"]
    data_dir = write_samples(records)

    raw = cli.read_input(data_dir)

    assert raw["prob_desc_notes"].tolist() == [None, None]
    assert raw["prob_desc_notes"].dtype == object
    assert raw["prob_desc_input_spec"].isna().tolist() == [False, True]
    pd.testing.assert_frame_equal(raw.drop(columns="prob_desc_notes"), read_samples(data_dir))


def test_predict_names_the_files_of_a_directory_that_are_not_samples(
    write_samples, make_raw_record, capsys
):
    # Read_samples skips them without a word; stdout stays pure JSON lines.
    data_dir = write_samples({1: make_raw_record(), 2: make_raw_record()})
    for name in ("README.md", "problem.json"):
        (data_dir / name).write_text("{}", encoding="utf-8")

    status, lines, errors = predict(capsys, data_dir, "--method", "label_prior")

    assert (status, [line["sample_id"] for line in lines]) == (0, [1, 2])
    assert errors == [
        "codeforces-tags: note: skipped 2 entries that are not sample_<id>.json files: "
        "README.md, problem.json"
    ]


def test_predict_skips_a_folder_named_like_a_sample_file_and_names_it(
    write_samples, make_raw_record, capsys
):
    data_dir = write_samples({14: make_raw_record()})
    (data_dir / "sample_99.json").mkdir()

    status, lines, errors = predict(capsys, data_dir, "--method", "label_prior")

    assert (status, [line["sample_id"] for line in lines]) == (0, [14])
    assert errors == [
        "codeforces-tags: note: skipped 1 entry that is not a sample_<id>.json file: sample_99.json"
    ]


def test_predict_refuses_two_sample_files_of_one_sample_id(
    write_samples, make_raw_record, loaded, capsys
):
    data_dir = write_samples({7: make_raw_record(), 29: make_raw_record()})
    (data_dir / "sample_07.json").write_text(json.dumps(make_raw_record()), encoding="utf-8")

    status, lines, errors = predict(capsys, data_dir, "--method", "tfidf")

    assert (status, lines, loaded) == (1, [], [])
    assert errors == [
        f"codeforces-tags: error: sample_07.json and sample_7.json in {data_dir} both give the "
        "sample id 7: keep one."
    ]


def test_predict_names_a_sample_file_that_cannot_be_read(sample_7, monkeypatch, capsys):
    read_text = type(sample_7).read_text

    def refuse_the_sample(self, *args, **kwargs):
        if self == sample_7:
            raise PermissionError(13, "Permission denied", str(self))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(sample_7), "read_text", refuse_the_sample)

    status, lines, errors = predict(capsys, sample_7, "--method", "tfidf")

    assert (status, lines) == (1, [])
    assert errors == [f"codeforces-tags: error: {sample_7} cannot be read: Permission denied."]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing", None, "missing does not exist: give a sample_<id>.json file or a directory"),
        ("empty", "", "No sample_<id>.json files in "),
        ("problem.json", "{}", "problem.json is not named sample_<id>.json"),
        ("sample_4.json", "[]", "sample_4.json does not hold a JSON object of the exercise's"),
    ],
)
def test_predict_refuses_an_input_without_a_readable_sample(
    tmp_path, capsys, name, content, message
):
    path = tmp_path / name
    if content == "":
        path.mkdir()
        (path / "README.md").write_text("No samples here.", encoding="utf-8")
    elif content is not None:
        path.write_text(content, encoding="utf-8")

    status, lines, errors = predict(capsys, path, "--method", "tfidf")

    assert (status, lines) == (1, [])
    assert errors[-1].startswith("codeforces-tags: error: ")
    assert message in errors[-1]


def test_run_method_hands_a_method_its_input_columns_only(raw_problems):
    method = RecordingMethod()
    spec = MethodSpec(
        module=NEVER_IMPORTED,
        kind="other",
        requirements=(),
        input_columns=("prob_desc_description", "source_code"),
    )

    predictions = cli.run_method(method, spec, raw_problems)

    assert method.columns == [["prob_desc_description", "source_code"]]
    assert predictions.tags.index.tolist() == [30, 10, 20]
    assert (predictions.scores, predictions.cost_usd) == (None, None)


def test_run_method_scores_a_local_score_method_once_and_tags_from_those_scores(raw_problems):
    # Never scores, then predict, which scores again: one forward pass per problem, as
    # tracking latency times a prediction.
    method = CountingScoreMethod()
    spec = MethodSpec(module=NEVER_IMPORTED, kind="score", requirements=())

    predictions = cli.run_method(method, spec, raw_problems)

    assert method.calls == ["scores"]
    assert predictions.scores["math"].tolist() == [0.3, 0.1, 0.2]  # problems 30, 10 and 20
    assert predictions.tags["math"].tolist() == [1, 0, 1]
    pd.testing.assert_frame_equal(predictions.tags, method.predict(raw_problems))  # the same tags


@pytest.fixture(name="counting")
def counting_fixture(monkeypatch):
    """Load a ``CountingScoreMethod`` in place of any method the command line loads."""
    method = CountingScoreMethod()
    monkeypatch.setattr(cli, "load_cli_method", lambda name, **_: method)
    return method


def test_predict_scores_a_local_score_method_once(counting, sample_7, capsys):
    status, [line], errors = predict(capsys, sample_7, "--method", "tfidf")

    assert (status, errors, counting.calls) == (0, [], ["scores"])
    assert (line["tags"], line["scores"]["math"]) == ([], 0.07)


def test_evaluate_scores_a_local_score_method_once(counting, cli_challenge, tmp_path, capsys):
    out = tmp_path / "report"

    status, _, _ = evaluate_command(
        capsys, cli_challenge["data_dir"], "--method", "tfidf", "--out", out
    )

    assert (status, counting.calls) == (0, ["scores"])  # the 8 test problems in one call
    predictions = pd.read_csv(out / "predictions.csv", index_col="sample_id")
    assert predictions.index[predictions["math"] == 1].tolist() == [15, 16]


def test_run_method_keeps_the_focus_tags_of_a_method_that_predicts_more(raw_problems):
    # The three commands agree: predict prints, and evaluate scores, the focus tags only.
    spec = MethodSpec(module=NEVER_IMPORTED, kind="score", requirements=())

    predictions = cli.run_method(WideMethod(), spec, raw_problems)

    assert list(predictions.tags.columns) == list(FOCUS_TAGS)
    assert list(predictions.scores.columns) == list(FOCUS_TAGS)
    assert WideMethod().predict(raw_problems).eq(1).all().all()
    assert json.loads(cli.prediction_lines(predictions)[0])["tags"] == list(FOCUS_TAGS)


def test_prediction_lines_write_the_tags_then_the_scores_then_the_cost_of_each_problem():
    index = pd.Index([4, 2], name="sample_id")
    predictions = cli.Predictions(
        tags=pd.DataFrame({"math": [1, 0], "graphs": [1, 0]}, index=index),
        scores=pd.DataFrame({"math": [0.5, -1.25], "graphs": [2.0, -0.1]}, index=index),
        cost_usd=pd.Series([0.001, 0.0005], index=index, name="cost_usd"),
    )

    assert cli.prediction_lines(predictions) == [
        '{"sample_id": 4, "tags": ["math", "graphs"], "scores": {"math": 0.5, "graphs": 2.0}, '
        '"cost_usd": 0.001}',
        '{"sample_id": 2, "tags": [], "scores": {"math": -1.25, "graphs": -0.1}, '
        '"cost_usd": 0.0005}',
    ]


def test_prediction_lines_mark_a_failed_answer_and_round_its_cost():
    index = pd.Index([4, 2], name="sample_id")
    predictions = cli.Predictions(
        tags=pd.DataFrame({"math": [1, 0]}, index=index),
        cost_usd=pd.Series([0.00012600000000000003, 0.0], index=index, name="cost_usd"),
        failed=pd.Series([False, True], index=index, name="failed"),
    )

    assert cli.prediction_lines(predictions) == [
        '{"sample_id": 4, "tags": ["math"], "cost_usd": 0.000126}',
        '{"sample_id": 2, "tags": [], "cost_usd": 0.0, "failed": true}',
    ]


def test_prediction_lines_refuse_a_score_that_is_not_a_number_rather_than_print_invalid_json():
    index = pd.Index([4], name="sample_id")
    predictions = cli.Predictions(
        tags=pd.DataFrame({"math": [0]}, index=index),
        scores=pd.DataFrame({"math": [math.nan]}, index=index),
    )

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        cli.prediction_lines(predictions)


def test_estimate_cost_usd_prices_the_problems_at_the_mean_cost_of_the_timed_requests(tmp_path):
    calls = load_latency(METHODS_ASSETS_DIR / CLI_DEFAULT_METHOD).calls

    estimate = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 10)

    assert estimate == 10 * calls["cost_usd"].mean()
    assert cli.estimate_cost_usd("tfidf", 10) is None  # a local method's latency.csv has no cost
    assert cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 10, assets_dir=tmp_path) is None


def test_check_cost_cap_refuses_a_cap_for_a_method_that_sends_no_paid_request():
    with pytest.raises(cli.CliError) as error:
        cli.check_cost_cap("tfidf", METHODS["tfidf"], 1, 0.05)

    assert str(error.value) == (
        "tfidf sends no paid request: --max-cost-usd is for a method that calls an API."
    )


@pytest.mark.parametrize("cap", [0.0, -1.0, math.nan])
def test_check_cost_cap_refuses_a_cap_not_above_0(cap):
    with pytest.raises(cli.CliError, match=r"cost cap \(--max-cost-usd\) must be above 0, not"):
        cli.check_cost_cap(CLI_DEFAULT_METHOD, METHODS[CLI_DEFAULT_METHOD], 1, cap)


def test_check_cost_cap_lets_one_paid_request_go_without_a_cap_and_more_within_their_cap():
    spec = METHODS[CLI_DEFAULT_METHOD]

    cli.check_cost_cap(CLI_DEFAULT_METHOD, spec, 1, None)
    cli.check_cost_cap(CLI_DEFAULT_METHOD, spec, 994, 0.2)
    cli.check_cost_cap("tfidf", METHODS["tfidf"], 994, None)


def test_check_cost_cap_refuses_more_requests_than_the_cap_pays_for_before_any_request():
    estimate = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 994)

    with pytest.raises(cli.CliError) as error:
        cli.check_cost_cap(CLI_DEFAULT_METHOD, METHODS[CLI_DEFAULT_METHOD], 994, 0.05)

    assert str(error.value) == (
        f"jev-statement-code: 994 requests would cost about ${estimate:.4f}, above the cost cap "
        "of $0.05: raise --max-cost-usd (--method tfidf runs offline, for free)."
    )


def test_check_cost_cap_counts_one_request_in_the_singular():
    estimate = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 1)

    with pytest.raises(cli.CliError) as error:
        cli.check_cost_cap(CLI_DEFAULT_METHOD, METHODS[CLI_DEFAULT_METHOD], 1, 0.00001)

    assert str(error.value).startswith(
        f"jev-statement-code: 1 request would cost about ${estimate:.4f}, above the cost cap of"
    )


def test_check_cost_cap_needs_a_cap_for_more_than_one_paid_request(tmp_path):
    spec = METHODS[CLI_DEFAULT_METHOD]
    estimate = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 2)

    with pytest.raises(cli.CliError) as known:
        cli.check_cost_cap(CLI_DEFAULT_METHOD, spec, 2, None)
    with pytest.raises(cli.CliError) as unknown:
        cli.check_cost_cap(CLI_DEFAULT_METHOD, spec, 2, None, assets_dir=tmp_path)

    assert str(known.value) == (
        f"jev-statement-code sends one paid request per problem: 2 requests, about "
        f"${estimate:.4f}, need a cost cap: pass --max-cost-usd X (--method tfidf runs offline, "
        "for free)."
    )
    assert "2 requests, at an unknown cost, need a cost cap" in str(unknown.value)


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_of_a_directory_with_an_api_method_needs_a_cap(
    cli_jev, write_samples, make_raw_record, capsys
):
    # A directory sends one paid request per file.
    data_dir = write_samples({1: make_raw_record(), 2: make_raw_record()})

    status, lines, errors = predict(capsys, data_dir)

    assert (status, lines, cli_jev.asked) == (1, [], [])
    assert "2 requests, about $" in errors[-1]
    assert "pass --max-cost-usd X" in errors[-1]


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_of_a_directory_with_an_api_method_runs_under_a_cap(
    cli_jev, write_samples, make_raw_record, capsys
):
    data_dir = write_samples({1: make_raw_record(), 2: make_raw_record()})

    status, lines, errors = predict(capsys, data_dir, "--max-cost-usd", "0.01")

    cost = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    assert (status, [line["sample_id"] for line in lines], cli_jev.asked) == (0, [1, 2], [1, 2])
    assert errors == [f"codeforces-tags: note: 2 paid requests cost ${2 * cost:.6f}"]


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_stops_between_batches_once_the_answers_cost_the_cap(
    cli_jev, write_samples, make_raw_record, monkeypatch, capsys
):
    # Through predict: the answers cost twice the estimate.
    monkeypatch.setattr(cli, "CLI_API_BATCH", 1)
    cli_jev.input_tokens *= 2
    data_dir = write_samples({i: make_raw_record() for i in range(1, 5)})
    cap = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, 4)

    status, lines, errors = predict(capsys, data_dir, "--max-cost-usd", cap)

    assert (status, lines, cli_jev.asked) == (1, [], [1, 2, 3])
    assert "jev-statement-code reached the cost cap of $" in errors[-1]


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_reports_a_request_the_api_refused_rather_than_hide_it_as_no_tag(
    cli_jev, write_samples, make_raw_record, capsys
):
    # Regression: a refused request scored 0.0 on every tag and printed as a prediction of no tag.
    data_dir = write_samples({1: make_raw_record(), 2: make_raw_record()})
    cli_jev.refused = {2}

    status, [answered, refused], errors = predict(capsys, data_dir, "--max-cost-usd", "0.01")

    cost = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    assert (status, "failed" in answered) == (0, False)
    assert refused == {
        "sample_id": 2,
        "tags": [],
        "scores": dict.fromkeys(FOCUS_TAGS, 0.0),
        "cost_usd": 0.0,
        "failed": True,
    }
    assert errors == [
        f"codeforces-tags: note: 2 paid requests cost ${cost:.6f}",
        "codeforces-tags: note: 1 of 2 answers failed (the API refused the request, or its answer "
        "was not one the method reads): sample_2.json got no tag",
    ]


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_says_what_the_answers_before_a_failure_cost(
    cli_jev, write_samples, make_raw_record, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "CLI_API_BATCH", 1)
    data_dir = write_samples({i: make_raw_record() for i in (1, 2, 3)})
    cli_jev.failure, cli_jev.failing = TransientApiError("no answer"), {3}

    status, lines, errors = predict(capsys, data_dir, "--max-cost-usd", "0.01")

    cost = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    assert (status, lines, cli_jev.asked) == (1, [], [1, 2, 3])
    assert errors[0] == (
        f"codeforces-tags: note: 2 of 3 problems were answered before this failure, for "
        f"${2 * cost:.6f}; nothing is reported"
    )
    assert errors[1].startswith("codeforces-tags: error: jev-statement-code got no answer")


@pytest.mark.usefixtures("jev_runs_here")
def test_predict_with_the_default_method_asks_jev_once_and_never_shows_the_key(
    cli_jev, sample_7, capsys
):
    status = cli.main(["predict", str(sample_7)])

    output, errors = capsys.readouterr()
    [line] = [json.loads(text) for text in output.splitlines()]
    cost = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    assert (status, cli_jev.asked) == (0, [7])
    assert line == {
        "sample_id": 7,
        "tags": ["math", "number theory"],
        "scores": cli_jev.probabilities,
        "cost_usd": pytest.approx(cost),
    }
    assert errors == f"codeforces-tags: note: 1 paid request cost ${line['cost_usd']:.6f}\n"
    assert SENTINEL not in output + errors


def test_predict_with_the_default_method_without_its_key_prints_one_line_and_loads_nothing(
    cli_jev, sample_7, loaded, monkeypatch, capsys
):
    # Through the command: status 1, one line on stderr, nothing on stdout.
    monkeypatch.setattr(cli_methods, "missing_groups", no_group_missing)
    monkeypatch.delenv(KEY, raising=False)

    status = cli.main(["predict", str(sample_7)])

    output, errors = capsys.readouterr()
    assert (status, output, cli_jev.asked, loaded) == (1, "", [], [])
    assert errors.splitlines() == [
        "codeforces-tags: error: jev-statement-code cannot run here: it needs TYPESAFE_API_KEY "
        "set. Run it as uv run --env-file FILE codeforces-tags ..., FILE holding "
        "TYPESAFE_API_KEY=... (--method tfidf runs offline, for free)."
    ]


@pytest.mark.parametrize("status", [401, None])
@pytest.mark.usefixtures("jev_runs_here")
def test_predict_reports_an_sdk_error_by_its_class_never_its_text(
    cli_jev, sample_7, monkeypatch, capsys, status
):
    # A refused key passes the presence check; the SDK's text can echo the state.
    base = type("TypeSafeError", (Exception,), {})
    refused = type("TypeSafeAuthenticationError", (base,), {})(f"401 {SECRET}")
    named = "TypeSafeAuthenticationError"
    if status is not None:
        refused.status = status
        named += f" {status}"
    sdk = types.ModuleType("typesafe_sdk")
    sdk.TypeSafeError = base
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)
    cli_jev.failure = refused

    exit_status, lines, errors = predict(capsys, sample_7)

    assert (exit_status, lines) == (1, [])
    assert errors == [
        f"codeforces-tags: error: jev-statement-code: its API refused the request ({named}): "
        "check that TYPESAFE_API_KEY holds a valid key, with credit (--method tfidf runs "
        "offline, for free)."
    ]
    assert SECRET not in "".join(errors)


@pytest.mark.parametrize(
    ("status", "named"), [(None, "TransientApiError"), (429, "RateLimitError 429")]
)
@pytest.mark.usefixtures("jev_runs_here")
def test_predict_without_an_answer_names_its_cause_by_class_never_its_text(
    cli_jev, sample_7, capsys, status, named
):
    # An adapter's TransientApiError may quote its SDK's error, whose text can echo the state.
    cli_jev.failure = TransientApiError(f"OpenAI did not answer: {SECRET}")
    if status is not None:
        cause = type("RateLimitError", (Exception,), {"status": status})(SECRET)
        cli_jev.failure.__cause__ = cause

    exit_status, lines, errors = predict(capsys, sample_7)

    assert (exit_status, lines) == (1, [])
    assert errors == [
        "codeforces-tags: error: jev-statement-code got no answer from its API (no network, a "
        f"rate limit or a server error): {named} (--method tfidf runs offline, for free)."
    ]


@pytest.mark.usefixtures("jev_runs_here")
def test_any_other_error_of_an_api_method_keeps_its_traceback(cli_jev, sample_7):
    cli_jev.failure = RuntimeError("a bug, not the API")

    with pytest.raises(RuntimeError, match="a bug, not the API"):
        cli.main(["predict", str(sample_7)])


def test_run_method_asks_an_api_method_in_batches_and_prices_each_answer(
    raw_problems, make_answer, monkeypatch
):
    monkeypatch.setattr(cli, "CLI_API_BATCH", 2)
    method = FakeApiMethod(make_answer)
    spec = replace(METHODS[CLI_DEFAULT_METHOD], module=NEVER_IMPORTED)

    predictions = cli.run_method(method, spec, raw_problems, max_cost_usd=1.0)

    cost = answer_cost_usd(make_answer())["total_cost"]
    assert method.batches == [[30, 10], [20]]
    assert predictions.tags["math"].tolist() == [1, 1, 1]
    assert predictions.scores is None
    assert predictions.cost_usd.tolist() == [cost] * 3
    assert predictions.cost_usd.index.tolist() == [30, 10, 20]


def test_run_method_stops_once_the_spend_reaches_the_cap_and_reports_nothing(
    raw_problems, make_answer, monkeypatch
):
    # The cap is checked between batches, so a run overshoots it by one at most.
    monkeypatch.setattr(cli, "CLI_API_BATCH", 1)
    method = FakeApiMethod(make_answer)
    spec = replace(METHODS[CLI_DEFAULT_METHOD], module=NEVER_IMPORTED)
    cost = answer_cost_usd(make_answer())["total_cost"]

    with pytest.raises(cli.CliError) as error:
        cli.run_method(method, spec, raw_problems, max_cost_usd=1.5 * cost)

    assert method.batches == [[30], [10]]
    assert str(error.value) == (
        f"fake-api reached the cost cap of ${1.5 * cost}: it spent ${2 * cost:.6f} on 2 of 3 "
        "problems, and nothing is reported (--method tfidf runs offline, for free)."
    )


def test_run_method_feeds_the_cap_what_each_batch_cost(raw_problems, make_answer, monkeypatch):
    # The second problem costs a thousand times the first: the cap stops the run after it.
    monkeypatch.setattr(cli, "CLI_API_BATCH", 1)
    method = FakeApiMethod(make_answer, tokens={30: 1_000, 10: 1_000_000, 20: 1_000})
    spec = replace(METHODS[CLI_DEFAULT_METHOD], module=NEVER_IMPORTED)
    costs = {i: answer_cost_usd(method.answer(i))["total_cost"] for i in (30, 10, 20)}

    with pytest.raises(cli.CliError, match="reached the cost cap"):
        cli.run_method(method, spec, raw_problems, max_cost_usd=costs[30] + costs[10] / 2)

    assert method.batches == [[30], [10]]


def test_run_method_joins_the_batches_with_each_problems_cost_and_scores(
    raw_problems, make_answer, monkeypatch
):
    monkeypatch.setattr(cli, "CLI_API_BATCH", 2)
    method = ScoringApiMethod(make_answer, tokens={30: 1_000, 10: 2_000, 20: 3_000})
    spec = replace(METHODS[CLI_DEFAULT_METHOD], module=NEVER_IMPORTED)

    predictions = cli.run_method(method, spec, raw_problems, max_cost_usd=1.0)

    costs = {i: answer_cost_usd(method.answer(i))["total_cost"] for i in (30, 10, 20)}
    assert method.batches == [[30, 10], [20]]
    assert predictions.tags.index.tolist() == [30, 10, 20]
    assert predictions.cost_usd.to_dict() == costs
    assert predictions.scores["math"].to_dict() == {30: 30.0, 10: 10.0, 20: 20.0}
    assert not predictions.failed.any()


def test_run_method_needs_apredict_with_answers_for_a_method_that_calls_an_api(raw_problems):
    spec = replace(METHODS[CLI_DEFAULT_METHOD], module=NEVER_IMPORTED)

    with pytest.raises(TypeError, match="recording calls an API but has no apredict_with_answers"):
        cli.run_method(RecordingMethod(), spec, raw_problems)


def test_a_fresh_predict_with_tfidf_loads_no_heavy_library_and_no_mlflow_package(sample_7):
    run = f"main(['predict', {str(sample_7)!r}, '--method', 'tfidf'])"

    loaded = loaded_banned_modules(f"from codeforces_tags.cli import main\n{run}")

    assert loaded == []


def evaluate_command(capsys, *args):
    """Run ``evaluate`` with the arguments; return its status, stdout and lines of stderr."""
    status = cli.main(["evaluate", *map(str, args)])
    output, errors = capsys.readouterr()
    return status, output, errors.splitlines()


@pytest.mark.parametrize(
    ("part", "read", "in_dev"), [(None, "test", 0), ("test", "test", 0), ("dev", "dev", 8)]
)
def test_evaluation_rows_are_a_part_of_the_frozen_split_on_the_challenge_dataset(
    cli_challenge, part, read, in_dev
):
    ids = cli_challenge[read]

    rows = cli.evaluation_rows(
        cli_challenge["data_dir"], part, split_path=cli_challenge["split_path"]
    )

    assert (rows.part, rows.data, rows.in_dev) == (read, cli_challenge["data_dir"], in_dev)
    assert rows.raw.index.tolist() == ids
    pd.testing.assert_frame_equal(rows.labels, cli_challenge["labels"].loc[ids], check_names=False)
    assert rows.groups.tolist() == [sample_id - 1 for sample_id in ids]


def test_evaluation_rows_of_every_problem_count_those_with_a_dev_problems_id(cli_challenge):
    rows = cli.evaluation_rows(
        cli_challenge["data_dir"], "all", split_path=cli_challenge["split_path"]
    )

    assert (rows.part, rows.groups, rows.in_dev) == ("all", None, len(cli_challenge["dev"]))
    pd.testing.assert_frame_equal(rows.labels, cli_challenge["labels"], check_names=False)


def test_evaluation_rows_of_another_directory_are_all_its_problems_with_a_note(
    write_samples, make_raw_record, capsys
):
    data_dir = write_samples({3: make_raw_record(), 40: make_raw_record(tags=["trees"])})

    rows = cli.evaluation_rows(data_dir, None)

    assert (rows.part, rows.raw.index.tolist()) == ("all", [3, 40])
    assert rows.labels[["math", "trees"]].to_numpy().tolist() == [[1, 0], [0, 1]]
    assert rows.in_dev == int((load_split().loc[[3, 40], "part"] == "dev").sum())
    assert capsys.readouterr().err == (
        f"codeforces-tags: note: {data_dir} is not the challenge dataset: evaluating its 2 "
        "problems (--part all)\n"
    )


@pytest.mark.parametrize("part", ["test", "dev"])
def test_evaluation_rows_refuse_a_part_of_the_frozen_split_on_another_directory(
    write_samples, make_raw_record, part
):
    data_dir = write_samples({3: make_raw_record()})

    with pytest.raises(cli.CliError) as error:
        cli.evaluation_rows(data_dir, part)

    message = str(error.value)
    assert message.startswith(
        f"--part {part} selects problems by the frozen split, on the challenge dataset only, "
        f"and {data_dir} is not it: the dataset has 1 problem files"
    )
    assert message.endswith(
        "Point $CODEFORCES_DATA_DIR at the challenge dataset, unchanged, or give it as DATA; or "
        f"pass --part all to evaluate every problem of {data_dir}."
    )


@pytest.mark.parametrize("tags", [MISSING, None, "math"])
def test_evaluation_rows_refuse_a_problem_without_a_list_of_tags(
    write_samples, make_raw_record, tags
):
    records = {5: make_raw_record(tags=tags)}
    if tags is MISSING:
        del records[5]["tags"]  # the only sample: the frame has no tags column at all
    else:
        records[1] = make_raw_record()

    with pytest.raises(cli.CliError) as error:
        cli.evaluation_rows(write_samples(records), "all")

    assert str(error.value) == (
        "every problem to evaluate needs its tags, as a list, but tags holds none in sample_5.json."
    )


@pytest.mark.parametrize("data", ["missing", "a file"])
def test_evaluate_names_data_that_is_not_a_directory_and_the_variable(
    loaded, tmp_path, capsys, data
):
    path = tmp_path / "sample_1.json"
    if data == "a file":
        path.write_text("{}", encoding="utf-8")

    status, output, errors = evaluate_command(capsys, path, "--method", "tfidf")

    state = "does not exist" if data == "missing" else "is a file"
    assert (status, output, loaded) == (1, "", [])
    assert errors == [
        f"codeforces-tags: error: {path} {state}: DATA must be a directory of sample_<id>.json "
        "files (default: $CODEFORCES_DATA_DIR)."
    ]


def test_evaluate_scores_a_method_on_the_test_part_and_writes_its_report(
    cli_challenge, loaded, tmp_path, capsys
):
    out = tmp_path / "report"

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--method", "tfidf", "--out", out
    )

    rows = cli.evaluation_rows(
        cli_challenge["data_dir"], "test", split_path=cli_challenge["split_path"]
    )
    method = load_method(METHODS_ASSETS_DIR / "tfidf" / MODEL_FOLDER)
    tags = method.predict(rows.raw[list(RAW_INPUT_COLUMNS)])
    expected = evaluate(rows.labels, tags, groups=rows.groups)
    report = json.loads((out / REPORT_FILE).read_text(encoding="utf-8"))
    assert (status, loaded) == (0, ["tfidf"])
    assert output == f"tfidf on test: 8 problems\n{report_text(expected)}\n"
    assert errors == [f"codeforces-tags: note: wrote the report in {out}"]
    assert report == {
        "method": "tfidf",
        "part": "test",
        "data": str(cli_challenge["data_dir"]),
        "problems": 8,
        "cost_usd": None,
        "failed": None,
    } | report_headline(expected)
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [REPORT_FILE, *REPORT_TABLES, "predictions.csv", "scores.csv"]
    )
    predictions = pd.read_csv(out / "predictions.csv", index_col="sample_id")
    pd.testing.assert_frame_equal(predictions, tags, check_names=False)


def test_evaluate_on_the_dev_part_notes_that_its_scores_are_in_sample(
    cli_challenge, tmp_path, capsys
):
    (tmp_path / "empty").mkdir()

    status, output, errors = evaluate_command(
        capsys,
        cli_challenge["data_dir"],
        "--method",
        "label_prior",
        "--part",
        "dev",
        "--out",
        tmp_path / "empty",
    )

    assert (status, output.splitlines()[0]) == (0, "label_prior on dev: 8 problems")
    assert errors[0] == (
        "codeforces-tags: note: 8 of these problems are dev problems of the frozen split, by "
        "sample id: every frozen method was fitted or thresholded on the dev part, so their scores "
        "are in sample (codeforces-tags methods lists the out-of-fold dev macro F1)"
    )
    assert not (tmp_path / "empty" / "scores.csv").exists()  # label_prior has no scores


def test_evaluate_of_one_dev_problem_says_so_in_the_singular(
    write_samples, make_raw_record, capsys
):
    dev_id = int(load_split().query("part == 'dev'").index[0])
    data_dir = write_samples({dev_id: make_raw_record()})

    status, _, errors = evaluate_command(capsys, data_dir, "--method", "label_prior")

    assert status == 0
    assert errors[1].startswith(
        "codeforces-tags: note: 1 of these problems is a dev problem of the frozen split, by "
        "sample id: "
    )


def test_evaluate_notes_the_focus_tags_that_no_problem_has_since_their_f1_is_0(
    write_samples, make_raw_record, capsys
):
    tags = [["math"], ["graphs", "trees"], ["strings", "games", "geometry"]]
    data_dir = write_samples({i: make_raw_record(tags=tag) for i, tag in enumerate(tags, 1)})

    status, output, errors = evaluate_command(capsys, data_dir, "--method", "label_prior")

    assert (status, output.splitlines()[0]) == (0, "label_prior on all: 3 problems")
    assert errors[-1] == (
        "codeforces-tags: note: no problem here has the focus tags number theory, probabilities: "
        "the F1 of each is 0 (as with scikit-learn's zero_division=0), which lowers the macro F1"
    )


def test_evaluate_refuses_a_report_folder_it_cannot_create_before_loading(
    cli_challenge, loaded, tmp_path, capsys
):
    # Regression: the folder was created after the run, so a path under a file failed with a
    # traceback once every request had been paid for.
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    out = tmp_path / "a_file" / "report"

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--method", "tfidf", "--out", out
    )

    assert (status, output, loaded) == (1, "", [])
    assert errors == [
        f"codeforces-tags: error: cannot create the report folder {out}: Not a directory."
    ]


@pytest.mark.parametrize("part", ["test", "all"])
def test_evaluate_reads_the_part_it_is_given(cli_challenge, tmp_path, capsys, part):
    out = tmp_path / "new" / "report"

    status, output, _ = evaluate_command(
        capsys, cli_challenge["data_dir"], "--method", "label_prior", "--part", part, "--out", out
    )

    problems = len(cli_challenge["test"]) if part == "test" else len(cli_challenge["labels"])
    assert (status, output.splitlines()[0]) == (0, f"label_prior on {part}: {problems} problems")
    assert (out / REPORT_FILE).is_file()  # its parent folder is created too


@pytest.mark.parametrize("folder", ["a file", "a folder with a file"])
def test_evaluate_refuses_a_report_folder_that_holds_something_before_loading(
    cli_challenge, loaded, tmp_path, capsys, folder
):
    out = tmp_path / "out"
    if folder == "a file":
        out.write_text("", encoding="utf-8")
    else:
        out.mkdir()
        (out / REPORT_FILE).write_text("{}", encoding="utf-8")

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--method", "tfidf", "--out", out
    )

    assert (status, output, loaded) == (1, "", [])
    assert errors == [
        f"codeforces-tags: error: {out} is not an empty folder: give --out a new or empty "
        "folder, so that a report is never mixed with other files."
    ]


@pytest.mark.usefixtures("jev_runs_here")
def test_evaluate_with_the_default_method_asks_each_problem_once_under_its_cap(
    cli_challenge, cli_jev, tmp_path, capsys
):
    out = tmp_path / "jev"

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--max-cost-usd", "0.01", "--out", out
    )

    cost = len(cli_challenge["test"]) * cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    report = json.loads((out / REPORT_FILE).read_text(encoding="utf-8"))
    assert (status, sorted(cli_jev.asked)) == (0, cli_challenge["test"])
    assert output.startswith("jev-statement-code on test: 8 problems\n")
    assert errors[0] == f"codeforces-tags: note: 8 paid requests cost ${cost:.6f}"
    assert (report["cost_usd"], report["failed"]) == (pytest.approx(cost), 0)
    scores = pd.read_csv(out / "scores.csv", index_col="sample_id")
    assert scores.loc[9].to_dict() == pytest.approx(cli_jev.probabilities)
    assert SENTINEL not in output + "".join(errors)


@pytest.mark.usefixtures("jev_runs_here")
def test_evaluate_refuses_an_api_method_whose_estimated_cost_exceeds_the_cap(
    cli_challenge, cli_jev, loaded, capsys
):
    # Refused before the method loads, so no request is sent.
    estimate = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, len(cli_challenge["test"]))

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--max-cost-usd", "0.0005"
    )

    assert (status, output, cli_jev.asked, loaded) == (1, "", [], [])
    assert errors == [
        f"codeforces-tags: error: jev-statement-code: 8 requests would cost about ${estimate:.4f},"
        " above the cost cap of $0.0005: raise --max-cost-usd (--method tfidf runs offline, for "
        "free)."
    ]


@pytest.mark.usefixtures("jev_runs_here")
def test_evaluate_stops_at_the_cost_cap_and_writes_no_report(
    cli_challenge, cli_jev, tmp_path, monkeypatch, capsys
):
    # The answers cost twice the estimate, so the cap stops the run midway.
    monkeypatch.setattr(cli, "CLI_API_BATCH", 1)
    cli_jev.input_tokens *= 2
    cap = cli.estimate_cost_usd(CLI_DEFAULT_METHOD, len(cli_challenge["test"]))
    answer = cli_jev.input_tokens * LLM_PRICES[JEV_MODEL]["input"] / 1e6
    answered = math.ceil(cap / answer)

    status, output, errors = evaluate_command(
        capsys, cli_challenge["data_dir"], "--max-cost-usd", cap, "--out", tmp_path / "jev"
    )

    assert (status, output, len(cli_jev.asked)) == (1, "", answered)
    assert answered < len(cli_challenge["test"])
    assert errors == [
        f"codeforces-tags: error: jev-statement-code reached the cost cap of ${cap}: it spent "
        f"${answered * answer:.6f} on {answered} of 8 problems, and nothing is reported "
        "(--method tfidf runs offline, for free)."
    ]
    assert not any((tmp_path / "jev").iterdir())  # created empty, before the run
