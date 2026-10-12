"""Tests for the LLM toolkit's commands, on fake prompted packages and a tiny dataset.

The four commands are one module, and their tests share these fixtures, so this one file is
longer than pylint's limit on a module.

No test sends a request: ``client.ask_for`` gives a fake ``Ask`` that answers the tags whose
keyword a prompt holds. git is faked too (``FakeGit``), since the answers files, the split and
the slice live under ``tmp_path``; so are the dataset check, which only the challenge's dataset
passes, and the prompt registry and MLflow log, which the MLflow tests put back.
"""

# pylint: disable=too-many-lines

import dataclasses
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from codeforces_tags import methods, provenance
from codeforces_tags.config import (
    ALL_TAGS,
    FOCUS_TAGS,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    TEXT_PARSER_VERSION,
)
from codeforces_tags.conformance import ConformanceError, check_tagging_method
from codeforces_tags.data.loading import load_dataset, read_samples
from codeforces_tags.evaluation.contamination import freeze_slice, self_identifying_solutions
from codeforces_tags.evaluation.predictions import (
    PER_PROBLEM_FILE,
    PREDICTIONS_FILE,
    load_per_problem,
    load_predictions,
    load_run_record,
)
from codeforces_tags.llm import __main__ as cli
from codeforces_tags.llm import client
from codeforces_tags.llm.client import TransientApiError
from codeforces_tags.llm.client import ask_for as real_ask_for
from codeforces_tags.llm.prompted import PROMPTED_FILE, PromptedLlm
from codeforces_tags.llm.prompts import field_sha256s, prompt_snapshot
from codeforces_tags.llm.store import answers_file, read_answers
from codeforces_tags.methods import METHODS, MODEL_FOLDER, MethodSpec, load_method
from codeforces_tags.preprocessing.pipeline import raw_to_problems
from codeforces_tags.provenance import file_sha256, sha256_json
from codeforces_tags.splitting import load_split, select_part
from codeforces_tags.tracking.environment import use_store
from codeforces_tags.tracking.methods import log_method
from codeforces_tags.tracking.prompts import register_prompts, set_prompt_aliases

FAKES = Path(__file__).parent / "fakes"
# The fake prompted methods, registered beside the real ones.
PROMPTED = {
    "fake-few-shot": MethodSpec(
        module="prompted_fakes.few_shot",
        kind="fitted",
        requirements=("numpy", "pandas"),
        calls_api=True,
    ),
    "fake-json": MethodSpec(
        module="prompted_fakes.zero_shot",
        kind="zero_shot",
        requirements=("numpy", "pandas"),
        calls_api=True,
    ),
    "fake-text": MethodSpec(
        module="prompted_fakes.zero_shot",
        kind="zero_shot",
        requirements=("numpy", "pandas"),
        calls_api=True,
    ),
}
REGISTRY = dict(METHODS) | PROMPTED
# sample_id -> (part, fold, tags, description). The fake Ask answers the tags whose keyword
# (KEYWORDS) a prompt holds; problem 6's solution names its Codeforces problem (the slice), and
# problem 12 has no notes.
PROBLEMS = {
    1: ("dev", 0, ["trees", "graphs"], "Count the leaves."),
    2: ("dev", 0, ["strings"], "Reverse the letters."),
    3: ("dev", 1, ["math", "number theory"], "Print the gcd."),
    4: ("dev", 1, ["graphs"], "Link the vertices."),
    5: ("dev", 2, ["math"], "Print the sum."),
    6: ("dev", 2, [], "Sort the array."),
    7: ("dev", 3, ["strings", "math"], "Count the letters, then their sum."),
    8: ("dev", 3, ["trees"], "Remove the leaves."),
    9: ("dev", 4, ["games"], "Who wins?"),
    10: ("dev", 4, ["math"], "Print the sum of two numbers."),
    11: ("test", -1, ["number theory"], "Print the gcd of three numbers."),
    12: ("test", -1, ["graphs", "trees"], "Count the vertices and the leaves."),
    13: ("test", -1, ["strings"], "Print the letters twice."),
}
KEYWORDS = {
    "leaves": "trees",
    "letters": "strings",
    "gcd": "number theory",
    "vertices": "graphs",
    "sum": "math",
}
DEV = [sample_id for sample_id, (part, *_) in PROBLEMS.items() if part == "dev"]
TEST = [sample_id for sample_id, (part, *_) in PROBLEMS.items() if part == "test"]


class FakeGit:
    """Stands for git: each commit records the bytes of the files it is given.

    Attributes:
        history: Each committed file's ``(sha, bytes)`` at each commit, oldest first; a test
            rewrites it to stand for a rewritten git history (e.g. an amended commit).
    """

    def __init__(self):
        """Start with nothing committed."""
        self.commits = 0
        self.history = {}

    def commit(self, *paths):
        """Commit the files as they are now, in one new commit."""
        self.commits += 1
        sha = f"{self.commits:040x}"
        for path in paths:
            self.history.setdefault(Path(path), []).append((sha, Path(path).read_bytes()))

    def committed_versions(self, path):
        """Return a file's ``(sha, bytes)`` at each commit that changed it, oldest first."""
        return list(self.history.get(Path(path), []))

    def committed_unchanged(self, path):
        """Tell whether a file is committed and holds the bytes it was last committed with."""
        versions = self.committed_versions(path)
        return (
            bool(versions) and Path(path).is_file() and Path(path).read_bytes() == versions[-1][1]
        )

    def last_commit(self, path):
        """Name the commit that last changed a file."""
        return self.history[Path(path)][-1][0]


@pytest.fixture(name="registry", autouse=True)
def registry_fixture(monkeypatch):
    """Make the fake packages importable and register them in the commands' registry."""
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setattr(cli, "METHODS", REGISTRY)
    return REGISTRY


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(monkeypatch, tmp_path):
    """Point the methods' frozen files, the answers files among them, at a folder of tmp_path."""
    assets = tmp_path / "assets"
    monkeypatch.setattr(methods, "METHODS_ASSETS_DIR", assets)
    return assets


@pytest.fixture(name="dataset")
def dataset_fixture(write_samples, make_raw_record, monkeypatch):
    """Write the problems as sample files, read through $CODEFORCES_DATA_DIR; return the folder."""
    records = {
        sample_id: make_raw_record(
            prob_desc_description=description, tags=tags, source_code=f"print({sample_id})"
        )
        for sample_id, (_, _, tags, description) in PROBLEMS.items()
    }
    records[6]["source_code"] = "# codeforces.com/contest/6/problem/A\nprint(sorted(input()))"
    records[12]["prob_desc_notes"] = None
    folder = write_samples(records)
    monkeypatch.setenv("CODEFORCES_DATA_DIR", str(folder))
    return folder


@pytest.fixture(name="split_file")
def split_file_fixture(monkeypatch, tmp_path):
    """Write the frozen split of the problems, each its own group, where the commands read it."""
    rows = [(sample_id, part, fold, sample_id) for sample_id, (part, fold, *_) in PROBLEMS.items()]
    path = tmp_path / "split.csv"
    pd.DataFrame(rows, columns=["sample_id", "part", "fold", "group"]).to_csv(path, index=False)
    monkeypatch.setattr(cli, "SPLIT_FILE", path)
    return path


@pytest.fixture(name="slice_file")
def slice_file_fixture(monkeypatch, dataset, split_file, tmp_path):
    """Freeze the self-identifying slice of the problems, where the commands read it."""
    path = tmp_path / "slices" / "self_identifying_solutions.csv"
    freeze_slice(self_identifying_solutions(load_dataset(dataset), load_split(split_file)), path)
    monkeypatch.setattr(cli, "SELF_IDENTIFYING_FILE", path)
    return path


@pytest.fixture(name="git", autouse=True)
def git_fixture(monkeypatch, split_file, slice_file):
    """Fake git in the commands, with the split and the slice committed; return the fake."""
    git = FakeGit()
    git.commit(split_file, slice_file)
    monkeypatch.setattr(cli, "committed_unchanged", git.committed_unchanged)
    monkeypatch.setattr(cli, "committed_versions", git.committed_versions)
    monkeypatch.setattr(cli, "last_commit", git.last_commit)
    return git


@pytest.fixture(name="events", autouse=True)
def events_fixture(monkeypatch):
    """Replace the dataset check, the prompt registry and the MLflow log by recorders.

    Returns the events they record, in order, with the fake Ask's. The registry's recorder
    numbers each method's version by its place in the family.
    """
    events = []

    def register(prompt_name, family, **store):
        events.append(("register_prompts", prompt_name, [spec.name for spec, _ in family], store))
        return {
            spec.name: SimpleNamespace(
                name=prompt_name, version=number, uri=f"prompts:/{prompt_name}/{number}"
            )
            for number, (spec, _) in enumerate(family, start=1)
        }

    def aliases(prompt_name, versions, specs, **_store):
        numbers = {name: version.version for name, version in versions.items()}
        events.append(("set_prompt_aliases", prompt_name, numbers, sorted(specs)))

    def log(folder, record, metrics, *, split, registry, **_store):
        events.append(("log_method", folder, record, metrics, split, registry))
        return "model-id"

    monkeypatch.setattr(cli, "check_dataset", lambda: events.append(("check_dataset",)))
    monkeypatch.setattr(cli, "register_prompts", register)
    monkeypatch.setattr(cli, "set_prompt_aliases", aliases)
    monkeypatch.setattr(cli, "log_method", log)
    return events


def keyword_tags(request):
    """Return the tags whose keyword a request's prompt holds."""
    return [tag for word, tag in KEYWORDS.items() if word in request.input]


@pytest.fixture(name="sent", autouse=True)
def sent_fixture(monkeypatch, make_answer, events):
    """Make ``client.ask_for`` give a fake Ask that answers keyword_tags; return what it is sent."""
    sent = []

    async def ask(request):
        events.append(("ask", request.sample_id))
        sent.append(request)
        tags = keyword_tags(request)
        json_format = request.settings.output_format == "json"
        return make_answer(
            request_sha256=request.sha256(),
            sample_id=request.sample_id,
            method=request.method,
            output_text=json.dumps({"tags": tags}) if json_format else ", ".join(tags),
        )

    monkeypatch.setattr(client, "ask_for", lambda settings: ask)
    return sent


@pytest.fixture(name="llm")
def llm_fixture(store):
    """Return a runner of one command on a method, which reads the test's split.

    run and score also get ``--allow-dirty`` (the tree may be dirty while the tests run) and the
    test's MLflow store.
    """

    def main(command, method, *options):
        arguments = [command, "--method", method, *options]
        if command in ("run", "score"):
            arguments += ["--allow-dirty", "--tracking-uri", store["tracking_uri"]]
            arguments += ["--artifact-dir", str(store["artifact_dir"])]
        return cli.main(arguments)

    return main


def answer_both_parts(llm, git, method):
    """Answer the dev part and commit its answers file; then the test part, and commit it."""
    assert llm("run", method, "--part", "dev") == 0
    git.commit(answers_file(method, "dev"))
    assert llm("run", method, "--part", "test") == 0
    git.commit(answers_file(method, "test"))


def snapshot(directory):
    """Map each file under ``directory`` to its bytes and modification time."""
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def examples_of(request):
    """Return the sample ids that a few-shot request names as its examples."""
    line = request.input.splitlines()[0]
    return [
        int(sample_id)
        for sample_id in line.removeprefix("Examples: problems ").rstrip(".").split(", ")
    ]


def test_parse_args_reads_each_command_and_its_defaults():
    render = cli.parse_args(
        ["render", "--method", "m", "--part", "dev", "--limit", "2", "--out", "x"]
    )
    run = cli.parse_args(["run", "--method", "m", "--part", "test"])
    score = cli.parse_args(["score", "--method", "m"])
    verify = cli.parse_args(["verify", "--method", "m"])

    assert (render.part, render.limit, render.out) == ("dev", 2, Path("x"))
    assert not hasattr(render, "data")  # the dataset is $CODEFORCES_DATA_DIR's, as load reads it
    assert not hasattr(render, "split")  # the split is the frozen one, config.SPLIT_FILE
    assert (run.part, run.max_cost_usd, run.allow_dirty) == ("test", None, False)
    assert (score.tracking_uri, score.artifact_dir) == (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR)
    assert not hasattr(verify, "tracking_uri")
    assert not hasattr(verify, "allow_dirty")


@pytest.mark.parametrize("limit", ["0", "-3"])
def test_parse_args_refuses_a_limit_below_one(limit, capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["render", "--method", "m", "--part", "dev", "--limit", limit, "--out", "x"])

    assert "expected a count of at least 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [
        ["render", "--method", "m", "--part", "dev", "--limit", "1", "--out", "x"],
        ["run", "--method", "m", "--part", "test"],
        ["score", "--method", "m"],
        ["verify", "--method", "m"],
    ],
)
def test_parse_args_takes_no_split_which_could_lift_the_gates(command, capsys):
    # A split that parks an unanswered dev problem elsewhere would let the test part through.
    with pytest.raises(SystemExit):
        cli.parse_args([*command, "--split", "other.csv"])

    assert "unrecognized arguments: --split other.csv" in capsys.readouterr().err


@pytest.mark.parametrize("cap", ["nan", "inf", "0", "-1"])
def test_parse_args_refuses_a_cost_cap_that_is_not_a_finite_amount_above_zero(cap, capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["run", "--method", "m", "--part", "dev", "--max-cost-usd", cap])

    assert f"expected a finite cost above 0 USD, not {cap}" in capsys.readouterr().err


def test_prompted_package_returns_the_package_of_a_prompted_method():
    zero_shot = cli.prompted_package("fake-json", registry=REGISTRY)
    few_shot = cli.prompted_package("fake-few-shot", registry=REGISTRY)

    assert zero_shot.__name__ == "prompted_fakes.zero_shot"
    assert few_shot.__name__ == "prompted_fakes.few_shot"


def test_prompted_package_refuses_an_unregistered_method():
    with pytest.raises(ValueError, match="'fake-xml' is not registered"):
        cli.prompted_package("fake-xml", registry=REGISTRY)


def test_prompted_package_refuses_a_method_that_asks_no_llm():
    with pytest.raises(ValueError, match="label_prior is not a prompted method"):
        cli.prompted_package("label_prior", registry=REGISTRY)


def test_prompted_package_refuses_a_package_without_the_functions_of_the_contract():
    registry = REGISTRY | {
        "fake-bare": MethodSpec("prompted_fakes", kind="fitted", requirements=(), calls_api=True)
    }

    with pytest.raises(
        ValueError, match=r"lacks \['load', 'llm_spec', 'prompt_family', 'make_method'\]"
    ):
        cli.prompted_package("fake-bare", registry=registry)


def test_a_package_whose_spec_or_family_leaves_out_the_method_is_refused(
    llm, monkeypatch, tmp_path, sent
):
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    llm_spec = zero_shot.llm_spec
    monkeypatch.setattr(zero_shot, "llm_spec", lambda name: llm_spec("fake-text"))

    with pytest.raises(ValueError, match="gives the spec of 'fake-text'"):
        llm("render", "fake-json", "--part", "dev", "--limit", "1", "--out", str(tmp_path))

    monkeypatch.setattr(zero_shot, "llm_spec", llm_spec)
    monkeypatch.setattr(zero_shot, "prompt_family", lambda prompt_name: (llm_spec("fake-text"),))
    with pytest.raises(
        ValueError, match=r"of prompted_fakes.zero_shot lacks the spec of fake-json"
    ):
        llm("run", "fake-json", "--part", "dev")
    assert sent == []


def test_the_fake_packages_are_conforming_prompted_methods(dataset, split_file, tmp_path):
    # The fitted fake's load reads its examples from $CODEFORCES_DATA_DIR, which the dataset
    # fixture sets and the isolation subprocess inherits: it fits on the dataset's problems.
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    few_shot = importlib.import_module("prompted_fakes.few_shot")
    dev = select_part(read_samples(dataset), load_split(split_file), "dev")
    labels = pd.DataFrame(1, index=dev.index, columns=list(FOCUS_TAGS))
    fitted = few_shot.make_method("fake-few-shot").fit(dev, labels)
    spec = zero_shot.llm_spec("fake-text")
    text = PromptedLlm(spec, prompt_snapshot(spec, registry=REGISTRY))

    for method in (text, fitted):
        check_tagging_method(
            method, dev, tmp_path / "out" / method.name, registry=REGISTRY, sys_path=[str(FAKES)]
        )


def test_the_fake_verify_extra_refuses_a_description_on_two_lines(raw_problems):
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    raw_problems.loc[10, "prob_desc_description"] = "Reverse each\nof the strings."

    zero_shot.verify_extra(raw_problems.drop(index=10))
    with pytest.raises(ValueError, match=r"problems \[10\] hold a line break"):
        zero_shot.verify_extra(raw_problems)


def test_render_writes_the_prompt_and_the_body_of_the_first_problems_and_sends_nothing(
    llm, tmp_path, sent, capsys
):
    assert (
        llm("render", "fake-json", "--part", "test", "--limit", "2", "--out", str(tmp_path / "out"))
        == 0
    )

    folder = tmp_path / "out" / "fake-json" / "test"
    assert sorted(path.name for path in folder.iterdir()) == [
        "11.json",
        "11.txt",
        "12.json",
        "12.txt",
    ]
    prompt = (folder / "12.txt").read_text(encoding="utf-8")
    assert prompt == (
        "Tag the problem that this Python program solves.\n"
        "Description: Count the vertices and the leaves.\n"
        "Notes: None\n"
        "\n"
        "print(12)\n"
        "\n"
        f"Choose from: {', '.join(ALL_TAGS)}"
    )
    body = json.loads((folder / "12.json").read_text(encoding="utf-8"))
    assert body["input"] == [{"role": "user", "content": prompt}]
    assert body["text"]["format"]["name"] == "fake_tags"
    assert body["store"] is False
    assert sent == []
    assert [line.split(":")[0] for line in capsys.readouterr().out.splitlines()] == [
        str(folder / "11.txt"),
        str(folder / "12.txt"),
    ]


def test_render_of_a_fitted_method_shows_each_dev_problem_examples_from_other_folds(llm, tmp_path):
    llm("render", "fake-few-shot", "--part", "dev", "--limit", "10", "--out", str(tmp_path / "out"))

    folder = tmp_path / "out" / "fake-few-shot" / "dev"
    first_lines = {
        sample_id: (folder / f"{sample_id}.txt").read_text(encoding="utf-8").splitlines()[0]
        for sample_id in DEV
    }
    assert first_lines == {
        1: "Examples: problems 3, 4.",
        2: "Examples: problems 3, 4.",
        3: "Examples: problems 1, 2.",
        4: "Examples: problems 2, 5.",
        5: "Examples: problems 3, 4.",
        6: "Examples: problems 4, 7.",
        7: "Examples: problems 5, 9.",
        8: "Examples: problems 9, 10.",
        9: "Examples: problems 7, 8.",
        10: "Examples: problems 7, 8.",
    }
    assert (folder / "1.txt").read_text(encoding="utf-8").splitlines()[1:3] == [
        "Problem 3: Print the gcd.",
        "Problem 4: Link the vertices.",
    ]


def test_run_sends_each_request_of_the_part_and_writes_its_answer(llm, sent, events, capsys):
    assert llm("run", "fake-text", "--part", "dev") == 0

    answers = read_answers(answers_file("fake-text", "dev"))
    assert [answer.sample_id for answer in answers] == DEV
    assert answers[0].tags == ("trees",)
    assert answers[6].tags == ("strings", "math")
    assert {answer.request_sha256 for answer in answers} == {request.sha256() for request in sent}
    assert events[0] == ("check_dataset",)
    assert "10 of 10 answers (0 failed)" in capsys.readouterr().out


def test_run_registers_the_prompts_of_the_family_before_its_first_request(llm, events, store):
    llm("run", "fake-json", "--part", "dev")

    kinds = [event[0] for event in events]
    assert kinds.index("register_prompts") < kinds.index("ask")
    [registered] = [event for event in events if event[0] == "register_prompts"]
    assert registered[1:3] == ("fake", ["fake-text", "fake-json"])
    assert registered[3] == store


def test_run_with_nothing_to_send_needs_no_api_key(llm, git, monkeypatch, capsys):
    answer_both_parts(llm, git, "fake-text")
    path = answers_file("fake-text", "dev")
    before = path.read_bytes()
    capsys.readouterr()
    # No openai, no key: the real ask_for would fail to import the SDK if it were called.
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(client, "ask_for", real_ask_for)

    assert llm("run", "fake-text", "--part", "dev") == 0
    assert llm("score", "fake-text") == 0
    assert llm("verify", "fake-text") == 0

    assert path.read_bytes() == before
    assert f"{path}: every request has an answer: nothing to send." in capsys.readouterr().out
    with pytest.raises(ImportError):
        real_ask_for(
            cli.prompted_package("fake-text", registry=REGISTRY).llm_spec("fake-text").settings
        )


def test_run_resumes_a_part_after_a_transient_error_and_exits_with_1_until_done(
    llm, make_answer, monkeypatch, capsys
):
    async def flaky(request):
        if request.sample_id == 3:
            raise TransientApiError("The API timed out.")
        return make_answer(request_sha256=request.sha256(), sample_id=request.sample_id)

    monkeypatch.setattr(client, "ask_for", lambda settings: flaky)
    assert llm("run", "fake-text", "--part", "dev") == 1
    assert "1 of 10 requests have no answer yet: run it again" in capsys.readouterr().err

    sent = []

    async def ask(request):
        sent.append(request.sample_id)
        return make_answer(request_sha256=request.sha256(), sample_id=request.sample_id)

    monkeypatch.setattr(client, "ask_for", lambda settings: ask)
    assert llm("run", "fake-text", "--part", "dev") == 0
    assert sent == [3]
    assert len(read_answers(answers_file("fake-text", "dev"))) == 10


def test_run_exits_with_1_at_the_cost_cap_and_sends_the_rest_next_time(
    llm, sent, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "LLM_CONCURRENCY", 1)

    assert llm("run", "fake-text", "--part", "dev", "--max-cost-usd", "0.00001") == 1
    assert len(sent) == 1
    assert "9 of 10 requests have no answer yet" in capsys.readouterr().err

    assert llm("run", "fake-text", "--part", "dev") == 0
    assert len(sent) == 10


def test_run_refuses_a_slice_that_is_not_committed(llm, slice_file, sent):
    slice_file.write_text(slice_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is not committed, or has changed since"):
        llm("run", "fake-text", "--part", "dev")
    assert sent == []


def test_run_refuses_a_slice_that_is_not_the_one_of_the_data(llm, slice_file, git, sent):
    slice_file.write_text("sample_id,part,rules\n", encoding="utf-8")
    git.commit(slice_file)

    with pytest.raises(ValueError, match="is not the slice that self_identifying_solutions finds"):
        llm("run", "fake-text", "--part", "dev")
    assert sent == []


@pytest.mark.parametrize("command", [["run", "--part", "dev"], ["score"]])
def test_run_and_score_check_the_dataset_then_refuse_a_dirty_tree(
    command, events, sent, monkeypatch
):
    monkeypatch.setattr(
        provenance, "git_state", lambda root: {"commit": "0" * 40, "code_dirty": True}
    )

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        cli.main([*command, "--method", "fake-text"])
    assert events == [("check_dataset",)]
    assert sent == []


@pytest.mark.parametrize("command", [["run", "--part", "test"], ["score"]])
def test_run_and_score_refuse_a_split_that_is_not_committed_and_unchanged(
    llm, split_file, sent, command
):
    # Parking a dev problem that has no answer outside dev would let the test part through.
    split = pd.read_csv(split_file)
    split.loc[split["sample_id"] == 9, "part"] = "parked"
    split.to_csv(split_file, index=False)

    with pytest.raises(ValueError, match="split.csv is not committed, or has changed since"):
        llm(command[0], "fake-text", *command[1:])
    assert sent == []


def test_run_refuses_answers_that_are_not_to_the_current_requests_in_either_file(
    llm, sent, events, monkeypatch
):
    llm("run", "fake-text", "--part", "dev")
    dev = answers_file("fake-text", "dev")
    answers_file("fake-text", "test").write_bytes(dev.read_bytes())
    sent.clear()

    with pytest.raises(
        ValueError, match=r"answers for the problems \[1, 2, .*\], which have no request"
    ):
        llm("run", "fake-text", "--part", "dev")

    answers_file("fake-text", "test").unlink()
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    changed = dataclasses.replace(
        zero_shot.llm_spec("fake-text"), constants={"language": "Python 3"}
    )
    monkeypatch.setattr(zero_shot, "llm_spec", lambda name: changed)
    with pytest.raises(
        ValueError, match="A prompted method never changes after its first dev request"
    ):
        llm("run", "fake-text", "--part", "dev")
    assert sent == []
    assert [event[0] for event in events].count("register_prompts") == 1


def test_run_refuses_an_answers_file_that_lost_a_committed_line_even_once_committed(
    llm, git, make_answer, monkeypatch
):
    # A failed answer is final: removing its line would have its request asked again.
    sent = []

    async def ask(request):
        sent.append(request.sample_id)
        status = "incomplete" if request.sample_id == 3 else "completed"
        return make_answer(
            request_sha256=request.sha256(), sample_id=request.sample_id, status=status
        )

    monkeypatch.setattr(client, "ask_for", lambda settings: ask)
    assert llm("run", "fake-text", "--part", "dev") == 0
    path = answers_file("fake-text", "dev")
    git.commit(path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:2] + lines[3:]))  # the line of problem 3
    sent.clear()

    with pytest.raises(ValueError, match="the file on disk lacks or changes a line of its commit"):
        llm("run", "fake-text", "--part", "dev")
    git.commit(path)
    with pytest.raises(
        ValueError, match=r"its commit \w{12} lacks or changes a line of its commit"
    ):
        llm("run", "fake-text", "--part", "dev")
    assert not sent


def test_run_sends_no_dev_request_once_the_test_part_has_answers(llm, git, sent):
    answer_both_parts(llm, git, "fake-text")
    path = answers_file("fake-text", "dev")
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
    git.history[path] = []  # a rewritten history (e.g. an amended commit) shows no removal
    git.commit(path)
    sent.clear()

    with pytest.raises(
        ValueError, match="answers_test.jsonl has answers, yet the dev problems \\[10\\] have none"
    ):
        llm("run", "fake-text", "--part", "dev")
    assert sent == []


def test_run_asks_the_test_part_only_once_the_dev_answers_are_committed_and_complete(
    llm, git, make_answer, monkeypatch
):
    with pytest.raises(ValueError, match="answers_dev.jsonl is not committed"):
        llm("run", "fake-text", "--part", "test")

    async def flaky(request):
        if request.sample_id == 9:
            raise TransientApiError("The API timed out.")
        return make_answer(request_sha256=request.sha256(), sample_id=request.sample_id)

    monkeypatch.setattr(client, "ask_for", lambda settings: flaky)
    llm("run", "fake-text", "--part", "dev")
    with pytest.raises(ValueError, match="answers_dev.jsonl is not committed"):
        llm("run", "fake-text", "--part", "test")  # --allow-dirty concerns the code only
    git.commit(answers_file("fake-text", "dev"))
    with pytest.raises(ValueError, match=r"no answer yet for the problems \[9\]"):
        llm("run", "fake-text", "--part", "test")
    assert not answers_file("fake-text", "test").exists()


def test_run_of_the_test_part_sends_the_test_requests_once_dev_is_frozen(llm, git, sent):
    llm("run", "fake-text", "--part", "dev")
    git.commit(answers_file("fake-text", "dev"))
    sent.clear()

    assert llm("run", "fake-text", "--part", "test") == 0

    assert [request.sample_id for request in sent] == TEST
    assert [answer.sample_id for answer in read_answers(answers_file("fake-text", "test"))] == TEST


def test_run_of_a_fitted_method_asks_dev_out_of_fold_and_test_with_every_dev_example(
    llm, git, sent
):
    answer_both_parts(llm, git, "fake-few-shot")

    folds = {sample_id: fold for sample_id, (_, fold, *_) in PROBLEMS.items()}
    examples = {request.sample_id: examples_of(request) for request in sent}
    for sample_id in DEV:
        assert all(folds[example] != folds[sample_id] for example in examples[sample_id])
    assert {sample_id: examples[sample_id] for sample_id in TEST} == {
        11: [9, 10],
        12: [9, 10],
        13: [9, 10],
    }


def test_score_freezes_the_predictions_the_calls_and_the_method_folder(llm, git, assets, capsys):
    answer_both_parts(llm, git, "fake-json")
    capsys.readouterr()

    assert llm("score", "fake-json") == 0

    directory = assets / "fake-json"
    assert sorted(path.name for path in directory.iterdir()) == [
        "answers_dev.jsonl",
        "answers_test.jsonl",
        MODEL_FOLDER,
        PER_PROBLEM_FILE,
        PREDICTIONS_FILE,
        "run_record.json",
    ]
    predictions = load_predictions(directory)
    assert list(predictions.columns) == list(ALL_TAGS)
    assert list(predictions.index) == [("dev", sid) for sid in DEV] + [
        ("test", sid) for sid in TEST
    ]
    assert predictions.loc[("test", 12), ["graphs", "trees"]].tolist() == [1, 1]
    assert int(predictions.to_numpy().sum()) == 13
    per_problem = load_per_problem(directory)
    assert list(per_problem.index) == list(predictions.index)
    assert per_problem["total_cost_usd"].sum() == pytest.approx(13 * 1.25e-5)
    stored = json.loads((directory / MODEL_FOLDER / PROMPTED_FILE).read_text(encoding="utf-8"))
    assert (stored["prompt"]["version"], stored["prompt"]["uri"]) == (2, "prompts:/fake/2")
    assert capsys.readouterr().out.splitlines() == [
        f"saved 13 predictions to {directory}",
        f"saved the calls of 13 problems to {directory / PER_PROBLEM_FILE}",
        f"saved the method folder {directory / MODEL_FOLDER}",
        "logged fake-json to MLflow: model model-id, prompt prompts:/fake/2",
    ]


def test_score_records_the_recipe_the_pinned_prompt_and_the_answers(llm, git, assets):
    answer_both_parts(llm, git, "fake-text")

    llm("score", "fake-text")

    record = load_run_record(assets / "fake-text")
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    spec = zero_shot.llm_spec("fake-text")
    prompt = prompt_snapshot(spec, registry=REGISTRY)
    assert record.recipe_sha256 == cli.recipe_sha256(spec, prompt)
    assert record.prompts == ("prompts:/fake/1",)
    assert record.allow_dirty is True
    entries = record.method_entries
    assert entries["prompt"] == {
        "name": "fake",
        "version": 1,
        "uri": "prompts:/fake/1",
        "content_sha256": prompt.content_sha256,
    }
    assert entries["llm_settings"] == spec.settings.to_dict()
    assert (entries["fields"], entries["constants"], entries["extra"]) == (
        list(spec.fields),
        {"language": "Python"},
        {},
    )
    assert entries["text_parser_version"] == TEXT_PARSER_VERSION
    assert entries["answers_sha256"] == {
        part: file_sha256(answers_file("fake-text", part)) for part in ("dev", "test")
    }
    assert entries["answers_commit"] == {
        part: git.last_commit(answers_file("fake-text", part)) for part in ("dev", "test")
    }
    assert entries["answers"]["dev"]["answers"] == 10
    assert entries["answers"]["test"]["statuses"] == {"completed": 3}
    assert entries["answers"]["dev"]["empty_answers"] == 2  # problems 6 and 9
    assert entries["dev_in_sample"] is False


def test_the_recipe_hashes_the_settings_the_prompt_the_inputs_the_tags_and_the_parser(monkeypatch):
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    spec = zero_shot.llm_spec("fake-json")
    prompt = prompt_snapshot(spec, registry=REGISTRY)

    recipe = cli.recipe_sha256(spec, prompt)

    assert recipe == sha256_json(
        {
            "method": "fake-json",
            "llm_settings": spec.settings.to_dict(),
            "prompt_content_sha256": prompt.content_sha256,
            "schema_name": "fake_tags",
            "fields": ["prob_desc_description", "prob_desc_notes", "source_code"],
            "constants": {"language": "Python"},
            "extra": {},
            "tags": list(ALL_TAGS),
            "text_parser_version": TEXT_PARSER_VERSION,
        }
    )
    monkeypatch.setattr(cli, "TEXT_PARSER_VERSION", TEXT_PARSER_VERSION + 1)
    assert cli.recipe_sha256(spec, prompt) != recipe


def test_score_logs_the_frozen_record_with_each_parts_metrics_and_sets_the_alias(
    llm, git, assets, events, split_file
):
    answer_both_parts(llm, git, "fake-json")

    llm("score", "fake-json")

    [aliases] = [event for event in events if event[0] == "set_prompt_aliases"]
    assert aliases[1:] == ("fake", {"fake-json": 2}, ["fake-json"])
    [(_, folder, record, metrics, split, registry)] = [
        event for event in events if event[0] == "log_method"
    ]
    assert folder == assets / "fake-json" / MODEL_FOLDER
    assert record == load_run_record(assets / "fake-json")
    assert registry is REGISTRY
    pd.testing.assert_frame_equal(split, load_split(split_file))
    assert sorted(metrics) == ["dev-oof", "test"]
    assert metrics["test"]["f1/trees"] == 1.0
    assert metrics["test"]["all_tag_macro_f1"] == 1.0
    assert metrics["dev-oof"]["recall/games"] == 0.0


def test_score_refuses_answers_files_that_are_not_committed_or_complete(llm, git):
    llm("run", "fake-text", "--part", "dev")
    git.commit(answers_file("fake-text", "dev"))
    llm("run", "fake-text", "--part", "test")

    with pytest.raises(ValueError, match="answers_test.jsonl is not committed"):
        llm("score", "fake-text")

    path = answers_file("fake-text", "test")
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:2]))
    git.commit(path)
    with pytest.raises(ValueError, match=r"no answer yet for the problems \[13\]"):
        llm("score", "fake-text")


def test_score_a_second_time_rewrites_nothing_and_logs_again(llm, git, assets, events):
    answer_both_parts(llm, git, "fake-few-shot")
    llm("score", "fake-few-shot")
    before = snapshot(assets / "fake-few-shot")

    assert llm("score", "fake-few-shot") == 0

    assert snapshot(assets / "fake-few-shot") == before
    logged = [event for event in events if event[0] == "log_method"]
    assert len(logged) == 2
    assert logged[0][2] == logged[1][2]  # the frozen record, not the one made by the second run


def test_score_saves_no_method_folder_that_fails_the_parity_check(llm, git, assets, monkeypatch):
    answer_both_parts(llm, git, "fake-text")
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    load = zero_shot.load
    # A load that loses a constant: the folder holds the prompt, but builds other requests.
    other = dataclasses.replace(zero_shot.llm_spec("fake-text"), constants={"language": "C"})
    broken = PromptedLlm(other, prompt_snapshot(other, registry=REGISTRY))
    monkeypatch.setattr(zero_shot, "load", lambda folder: broken)

    with pytest.raises(ValueError, match="builds other dev requests than the answered ones"):
        llm("score", "fake-text")

    directory = assets / "fake-text"
    assert not (directory / MODEL_FOLDER).exists()
    assert not list(directory.glob(".saving-*"))  # nor the temporary folder it was saved in
    monkeypatch.setattr(zero_shot, "load", load)
    assert llm("score", "fake-text") == 0
    assert (directory / MODEL_FOLDER / PROMPTED_FILE).is_file()


@pytest.mark.parametrize("command", ["score", "verify"])
def test_score_and_verify_refuse_a_per_problem_file_that_is_not_the_answers(
    llm, git, assets, command
):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    path = assets / "fake-text" / PER_PROBLEM_FILE
    path.write_text(
        path.read_text(encoding="utf-8").replace("completed", "incomplete", 1), encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="is missing, or is not the rows derived from the answers files"
    ):
        llm(command, "fake-text")


def test_score_mirrors_the_method_and_its_prompt_to_mlflow(llm, git, assets, store, monkeypatch):
    monkeypatch.setattr(cli, "register_prompts", register_prompts)
    monkeypatch.setattr(cli, "set_prompt_aliases", set_prompt_aliases)
    monkeypatch.setattr(cli, "log_method", log_method)
    answer_both_parts(llm, git, "fake-json")

    llm("score", "fake-json")
    llm("score", "fake-json")

    experiment_id = use_store(**store)
    [model] = mlflow.search_logged_models([experiment_id], output_format="list")
    assert json.loads(model.tags["mlflow.linkedPrompts"]) == [{"name": "fake", "version": "2"}]
    client_ = mlflow.MlflowClient()
    assert client_.get_prompt_version_by_alias("fake", "json").version == 2
    assert sorted(version.version for version in client_.search_prompt_versions("fake")) == [1, 2]
    record = load_run_record(assets / "fake-json")
    assert model.params["prompts"] == '["prompts:/fake/2"]'
    assert record.prompts == ("prompts:/fake/2",)


def test_verify_re_derives_the_frozen_files_and_reports_the_scores(
    llm, git, assets, monkeypatch, capsys
):
    answer_both_parts(llm, git, "fake-json")
    llm("score", "fake-json")
    before = snapshot(assets)
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")
    verified = []
    monkeypatch.setattr(
        zero_shot, "verify_extra", lambda raw_problems: verified.append(list(raw_problems.index))
    )
    capsys.readouterr()

    assert llm("verify", "fake-json") == 0

    assert snapshot(assets) == before
    assert verified == [DEV + TEST]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "the frozen files of fake-json are complete and consistent with the split"
    assert (
        out[1] == "re-rendered 13 requests: each has its answer, and each answer its stored parse"
    )
    assert out[2:4] == [
        "the run record holds the hashes and the summaries of the answers files",
        "each answers file holds every line of its committed versions, then new ones",
    ]
    assert out[-2].startswith("dev-oof: macro F1 ")
    assert out[-1] == "test: macro F1 0.5000, micro F1 1.0000"


def test_verify_refuses_a_stored_parse_that_is_not_the_parsers(llm, git):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    path = answers_file("fake-text", "dev")
    path.write_text(
        path.read_text(encoding="utf-8").replace('"unknown_pieces": 0', '"unknown_pieces": 1', 1),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"stored tags, failed or unknown_pieces of the answers to the problems \[1\]",
    ):
        llm("verify", "fake-text")


def test_score_and_verify_refuse_a_run_record_that_holds_other_answers_files(llm, git):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    # An answer's extra changes neither a prediction nor per_problem.csv: only its file's hash.
    path = answers_file("fake-text", "test")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"321"', '"322"', 1), encoding="utf-8")
    git.history[path] = []  # a rewritten history (e.g. an amended commit) shows no edit
    git.commit(path)

    for command in ("score", "verify"):
        with pytest.raises(ValueError, match="holds other answers_sha256 than the answers files"):
            llm(command, "fake-text")


def test_score_and_verify_refuse_an_answers_file_whose_committed_line_was_edited(llm, git):
    answer_both_parts(llm, git, "fake-text")
    path = answers_file("fake-text", "dev")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"321"', '"322"', 1), encoding="utf-8")
    git.commit(path)  # committed before the first score, whose run record would hold it
    edited = r"answers_dev.jsonl: its commit \w{12} lacks or changes a line of its commit"

    with pytest.raises(ValueError, match=edited):
        llm("score", "fake-text")

    # Scored from a rewritten history that shows no edit, then verified against the true one.
    history = git.history[path]
    git.history[path] = history[-1:]
    assert llm("score", "fake-text") == 0
    git.history[path] = history
    with pytest.raises(ValueError, match=edited):
        llm("verify", "fake-text")


def test_verify_refuses_predictions_that_are_not_the_answers(llm, git, assets):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    path = assets / "fake-text" / PREDICTIONS_FILE
    predictions = load_predictions(assets / "fake-text")
    predictions.loc[("dev", 6), "dp"] = 1
    predictions.to_csv(path)

    with pytest.raises(ValueError, match="differ from the predictions encoded from the answers"):
        llm("verify", "fake-text")


@pytest.mark.parametrize("command", ["score", "verify"])
def test_score_and_verify_refuse_a_method_folder_that_holds_another_prompt(
    llm, git, assets, events, command
):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    path = assets / "fake-text" / MODEL_FOLDER / PROMPTED_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["prompt"]["content_sha256"] = "0" * 64
    path.write_text(json.dumps(stored), encoding="utf-8")
    events.clear()

    with pytest.raises(ValueError, match="holds the prompt 000000000000"):
        llm(command, "fake-text")
    assert "log_method" not in [event[0] for event in events]


def test_verify_checks_the_dataset_then_refuses_a_method_folder_holding_a_file_not_data(
    llm, git, assets, events
):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    (assets / "fake-text" / MODEL_FOLDER / "extra.pkl").write_bytes(b"\x80\x04N.")
    events.clear()

    with pytest.raises(ConformanceError, match="extra.pkl"):
        llm("verify", "fake-text")
    assert events == [("check_dataset",)]


def test_verify_refuses_a_fitted_method_folder_whose_test_requests_were_not_answered(
    llm, git, assets
):
    answer_both_parts(llm, git, "fake-few-shot")
    llm("score", "fake-few-shot")
    assert llm("verify", "fake-few-shot") == 0
    path = assets / "fake-few-shot" / MODEL_FOLDER / "examples.json"
    hashes = field_sha256s(raw_to_problems(read_samples()).loc[[1, 2]], ("prob_desc_description",))
    examples = [{"sample_id": sample_id, "sha256": sha256} for sample_id, sha256 in hashes.items()]
    path.write_text(json.dumps({"examples": examples}), encoding="utf-8")

    with pytest.raises(ValueError, match="builds other test requests than the answered ones"):
        llm("verify", "fake-few-shot")


def test_a_fitted_method_folder_reads_its_examples_from_the_dataset_and_checks_them(
    llm, git, assets, dataset
):
    answer_both_parts(llm, git, "fake-few-shot")
    llm("score", "fake-few-shot")
    assert llm("verify", "fake-few-shot") == 0
    sample = dataset / "sample_9.json"  # an example of every test problem
    record = json.loads(sample.read_text(encoding="utf-8"))
    sample.write_text(json.dumps(record | {"prob_desc_description": "Who?"}), encoding="utf-8")

    with pytest.raises(ValueError, match="their fields changed since the method was saved"):
        load_method(assets / "fake-few-shot" / MODEL_FOLDER, registry=REGISTRY)


def test_verify_refuses_another_recipe(llm, git, monkeypatch):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    monkeypatch.setattr(cli, "TEXT_PARSER_VERSION", TEXT_PARSER_VERSION + 1)

    with pytest.raises(ValueError, match="was frozen with another recipe"):
        llm("verify", "fake-text")


def test_verify_lets_the_error_of_verify_extra_through(llm, git, monkeypatch):
    answer_both_parts(llm, git, "fake-text")
    llm("score", "fake-text")
    zero_shot = importlib.import_module("prompted_fakes.zero_shot")

    def refuse(raw_problems):
        raise ValueError(f"{len(raw_problems)} problems refused")

    monkeypatch.setattr(zero_shot, "verify_extra", refuse)
    with pytest.raises(ValueError, match="13 problems refused"):
        llm("verify", "fake-text")


def test_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.llm", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.llm.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.llm", run_name="__main__")

    assert exit_info.value.code == 0
