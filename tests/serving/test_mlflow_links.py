"""Tests for the links from a served tagging method to its MLflow experiment, run and prompts.

The UI is the stub server of ``tests/serving/conftest.py`` on 127.0.0.1, answering the two routes
of MLflow's REST API that ``method_links`` reads as ``mlflow server`` 3.16.1 answers them.
"""

import json
import os
import subprocess
import sys

import pytest

from codeforces_tags.config import MLFLOW_ENV, MLFLOW_TRACKING_URI, SERVING_MLFLOW_UI_URL
from codeforces_tags.evaluation.predictions import RUN_RECORD_FILE
from codeforces_tags.serving import mlflow_links
from codeforces_tags.serving.client import ServingError
from codeforces_tags.serving.mlflow_links import (
    MlflowLinks,
    method_links,
    mlflow_ui_command,
    mlflow_ui_environment,
    mlflow_ui_url,
    ui_alive,
)
from codeforces_tags.tracking.methods import RUN_KIND_TAG
from codeforces_tags.tracking.prompts import LINKED_PROMPTS_TAG

# The two routes of MLflow's REST API that method_links reads, as the stub server sees them.
EXPERIMENT_ROUTE = "/api/2.0/mlflow/experiments/get-by-name?experiment_name=tag-methods"
RUNS_ROUTE = "/api/2.0/mlflow/runs/search"
# The tag-methods experiment and the method run of the stub's store, as MLflow answers them.
EXPERIMENT = {"experiment": {"experiment_id": "3", "name": "tag-methods"}}
RUN_ID = "53772b68e10c4b94b30796f1799dd0f4"
RUN_INFO = {"run_id": RUN_ID, "experiment_id": "3", "status": "FINISHED"}
METHOD_TAG = {"key": RUN_KIND_TAG, "value": "method"}
RUNS = {"runs": [{"info": RUN_INFO, "data": {"tags": [METHOD_TAG]}}]}
# What MLflow answers for a store without the experiment (with 404), and when no run matches.
NO_EXPERIMENT = {
    "error_code": "RESOURCE_DOES_NOT_EXIST",
    "message": "Could not find experiment with name 'tag-methods'",
}
NO_RUN = {}
# A method run whose tag of linked prompts is not the JSON text that MLflow writes there.
UNREADABLE_RUNS = {
    "runs": [{"info": RUN_INFO, "data": {"tags": [{"key": LINKED_PROMPTS_TAG, "value": "v1"}]}}]
}
# The prompt versions that the prompted method's run record pins.
PINNED = ("prompts:/descode2tag/2", "prompts:/jev-tags/1")
# Variables a local mlflow server never gets: credentials of LLM providers, clouds and databases,
# proxies.
LEFT_OUT = (
    "OPENAI_API_KEY",
    "TYPESAFE_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "DATABASE_PASSWORD",
    "mlflow_tracking_password",
    "AZURE_CREDENTIALS",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "google_application_credentials",
    "HTTPS_PROXY",
    "https_proxy",
)
# URLs that urllib refuses before it opens a socket: a bracket never closed, and no scheme at all.
MALFORMED_URLS = ["http://[::1", ""]
# Asks the UI at argv[1] for its health from a fresh interpreter, whose environment names a proxy.
THROUGH_NO_PROXY = (
    "import sys; from codeforces_tags.serving.mlflow_links import ui_alive; "
    "print(ui_alive(sys.argv[1]))"
)


def write_run_record(folder, record):
    """Write a run record into a method's folder of frozen files, as a freeze writes it."""
    folder.mkdir(parents=True)
    (folder / RUN_RECORD_FILE).write_text(json.dumps(record.to_dict()), encoding="utf-8")


def linking(*versions):
    """Return MLflow's answer to the search of a method run linked to these prompt versions.

    MLflow lists them in the run's ``mlflow.linkedPrompts`` tag, as ``log_method`` and
    ``link_prompt_version`` leave it: the JSON text of a list of ``{"name", "version"}``.
    """
    linked = [{"name": name, "version": version} for name, version in versions]
    tags = [METHOD_TAG, {"key": LINKED_PROMPTS_TAG, "value": json.dumps(linked)}]
    return {"runs": [{"info": RUN_INFO, "data": {"tags": tags}}]}


def serve_store(stub_server, *, experiment=True, runs=None):
    """Answer method_links' two routes as an MLflow UI on a store that may lack either.

    ``runs`` is the answer to the search of the method run, ``RUNS`` when None: ``NO_RUN`` for a
    store that lacks the method run.
    """
    if experiment:
        stub_server.answer("GET", EXPERIMENT_ROUTE, 200, EXPERIMENT)
    else:
        stub_server.answer("GET", EXPERIMENT_ROUTE, 404, NO_EXPERIMENT)
    stub_server.answer("POST", RUNS_ROUTE, 200, RUNS if runs is None else runs)


@pytest.fixture(name="assets_dir")
def assets_dir_fixture(tmp_path, placeholder_record):
    """Return a folder of frozen methods, each with the run record of its freeze.

    The run record of ``descode2tag-json`` pins two prompt versions; those of ``tfidf`` and of
    ``jev-statement-code`` pin none, as the frozen Jev records pin none.
    """
    assets = tmp_path / "methods"
    prompted = placeholder_record("descode2tag-json", prompts=PINNED)
    write_run_record(assets / "descode2tag-json", prompted)
    for method in ("tfidf", "jev-statement-code"):
        write_run_record(assets / method, placeholder_record(method))
    return assets


def test_mlflow_ui_url_is_the_argument_then_the_variable_then_the_setting(monkeypatch):
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", "http://127.0.0.1:5001")

    assert mlflow_ui_url("http://127.0.0.1:5002") == "http://127.0.0.1:5002"
    assert mlflow_ui_url() == "http://127.0.0.1:5001"
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", "")  # empty counts as unset
    assert mlflow_ui_url() == SERVING_MLFLOW_UI_URL
    monkeypatch.delenv("CODEFORCES_MLFLOW_UI_URL")
    assert mlflow_ui_url() == SERVING_MLFLOW_UI_URL == "http://127.0.0.1:5000"


def test_method_links_open_the_experiment_the_method_run_and_each_pinned_prompt_version(
    stub_server, assets_dir
):
    serve_store(stub_server)
    ui = stub_server.url

    links = method_links(f"{ui}/", "descode2tag-json", assets_dir=assets_dir)

    assert links == MlflowLinks(
        experiment=f"{ui}/#/experiments/3",
        run=f"{ui}/#/experiments/3/runs/{RUN_ID}",
        prompts=(
            ("descode2tag v2", f"{ui}/#/prompts/descode2tag?promptVersion=2"),
            ("jev-tags v1", f"{ui}/#/prompts/jev-tags?promptVersion=1"),
        ),
        registry=f"{ui}/#/prompts",
    )
    # The experiment by name, then the run that log_method made: never an evaluation run.
    get, search = stub_server.requests
    assert [(get.method, get.path), (search.method, search.path)] == [
        ("GET", EXPERIMENT_ROUTE),
        ("POST", RUNS_ROUTE),
    ]
    assert json.loads(search.body) == {
        "experiment_ids": ["3"],
        "filter": "tags.run_kind = 'method' AND attributes.run_name = 'descode2tag-json'",
        "max_results": 1,
    }


@pytest.mark.parametrize(
    ("method", "labels"),
    [
        ("descode2tag-json", ["descode2tag v2", "jev-tags v1"]),
        ("tfidf", []),  # its run record pins no prompt
        ("f2llm_setfit", []),  # registered but not frozen: no run record
    ],
)
def test_method_links_take_the_prompt_versions_from_the_committed_run_record(
    stub_server, assets_dir, method, labels
):
    serve_store(stub_server)

    links = method_links(stub_server.url, method, assets_dir=assets_dir)

    assert [label for label, _ in links.prompts] == labels
    assert links.run == f"{stub_server.url}/#/experiments/3/runs/{RUN_ID}"


@pytest.mark.parametrize(
    ("method", "linked", "labels"),
    [
        # A frozen Jev record pins no prompt: its freeze links jev-tags 1 to its method run.
        ("jev-statement-code", [("jev-tags", "1")], ["jev-tags v1"]),
        # The record's versions first, then the run's others, each version once.
        (
            "descode2tag-json",
            [("descode2tag", "2"), ("code2tag", "1")],
            ["descode2tag v2", "jev-tags v1", "code2tag v1"],
        ),
    ],
)
def test_method_links_add_the_prompt_versions_linked_to_the_method_run_after_the_pinned_ones(
    stub_server, assets_dir, method, linked, labels
):
    serve_store(stub_server, runs=linking(*linked))

    links = method_links(stub_server.url, method, assets_dir=assets_dir)

    assert [label for label, _ in links.prompts] == labels
    jev_tags = f"{stub_server.url}/#/prompts/jev-tags?promptVersion=1"
    assert dict(links.prompts)["jev-tags v1"] == jev_tags


def test_method_links_link_no_experiment_and_no_run_of_a_store_without_the_experiment(
    stub_server, assets_dir
):
    serve_store(stub_server, experiment=False)

    links = method_links(stub_server.url, "descode2tag-json", assets_dir=assets_dir)

    assert (links.experiment, links.run, len(links.prompts)) == (None, None, 2)
    assert links.registry == f"{stub_server.url}/#/prompts"
    assert [request.method for request in stub_server.requests] == ["GET"]  # no run searched


def test_method_links_link_no_run_of_a_store_without_the_method_run(stub_server, assets_dir):
    serve_store(stub_server, runs=NO_RUN)

    links = method_links(stub_server.url, "tfidf", assets_dir=assets_dir)

    assert links == MlflowLinks(
        f"{stub_server.url}/#/experiments/3", None, (), f"{stub_server.url}/#/prompts"
    )


def test_method_links_raise_a_serving_error_naming_mlflow_ui_when_the_ui_does_not_answer(
    assets_dir, refused_url
):
    with pytest.raises(ServingError, match="serving mlflow-ui") as raised:
        method_links(refused_url, "tfidf", assets_dir=assets_dir)

    assert raised.value.status is None


@pytest.mark.parametrize("url", MALFORMED_URLS)
def test_method_links_raise_a_serving_error_naming_mlflow_ui_for_a_url_that_is_not_one(
    assets_dir, url
):
    with pytest.raises(ServingError, match="serving mlflow-ui") as raised:
        method_links(url, "tfidf", assets_dir=assets_dir)

    assert raised.value.status is None


@pytest.mark.parametrize(
    "error",
    [
        ("GET", EXPERIMENT_ROUTE, 500, "the store is locked"),
        ("POST", RUNS_ROUTE, 400, "Invalid clause(s) in filter string"),
    ],
)
def test_method_links_raise_the_status_and_mlflows_message_of_an_error_answer(
    stub_server, assets_dir, error
):
    method, route, status, message = error
    serve_store(stub_server)
    stub_server.answer(method, route, status, {"error_code": "X", "message": message})

    with pytest.raises(ServingError) as raised:
        method_links(stub_server.url, "tfidf", assets_dir=assets_dir)

    assert (raised.value.status, raised.value.message) == (status, message)


def test_method_links_raise_the_status_line_of_an_error_answer_without_json(
    stub_server, assets_dir
):
    serve_store(stub_server)
    page = b"<html><h1>502 Bad Gateway</h1></html>"
    stub_server.answer("POST", RUNS_ROUTE, 502, page, "text/html")

    with pytest.raises(ServingError) as raised:
        method_links(stub_server.url, "tfidf", assets_dir=assets_dir)

    assert (raised.value.status, raised.value.message) == (502, "502 Bad Gateway")


@pytest.mark.parametrize(
    ("experiment", "runs", "match"),
    [
        (b"<html>a login page</html>", RUNS, "with no JSON"),
        ({"experiments": []}, RUNS, "is it an MLflow UI"),
        (EXPERIMENT, {"runs": [{}]}, "is it an MLflow UI"),
        (EXPERIMENT, UNREADABLE_RUNS, "is it an MLflow UI"),
    ],
)
def test_method_links_raise_a_serving_error_on_an_answer_that_mlflow_does_not_write(
    stub_server, assets_dir, experiment, runs, match
):
    stub_server.answer("GET", EXPERIMENT_ROUTE, 200, experiment)
    stub_server.answer("POST", RUNS_ROUTE, 200, runs)

    with pytest.raises(ServingError, match=match) as raised:
        method_links(stub_server.url, "tfidf", assets_dir=assets_dir)

    assert raised.value.status == 200


@pytest.mark.parametrize("method", ["tfidf' OR attributes.run_name != '", "../tfidf", "TFIDF"])
def test_method_links_refuse_a_name_that_is_not_a_method_name_before_any_request(
    stub_server, assets_dir, method
):
    with pytest.raises(ValueError, match="is not a method name"):
        method_links(stub_server.url, method, assets_dir=assets_dir)

    assert stub_server.requests == []


def test_method_links_refuse_a_run_record_that_pins_a_prompt_alias(
    stub_server, tmp_path, placeholder_record
):
    record = placeholder_record("descode2tag-json", prompts=("prompts:/descode2tag@champion",))
    write_run_record(tmp_path / "methods" / "descode2tag-json", record)

    with pytest.raises(ValueError, match="not a prompt version"):
        method_links(stub_server.url, "descode2tag-json", assets_dir=tmp_path / "methods")

    assert stub_server.requests == []


def test_ui_alive_tells_whether_the_ui_answers_health_with_200(stub_server, refused_url):
    stub_server.answer("GET", "/health", 503, b"starting", "text/plain")
    assert not ui_alive(stub_server.url)

    stub_server.answer("GET", "/health", 200, b"OK", "text/html; charset=utf-8")
    assert ui_alive(f"{stub_server.url}/")

    assert not ui_alive(refused_url)


@pytest.mark.parametrize("url", MALFORMED_URLS)
def test_ui_alive_is_false_for_a_url_that_is_not_one(url):
    assert not ui_alive(url)


def test_the_ui_is_asked_through_no_proxy_even_when_the_environment_names_one(
    stub_server, refused_url
):
    stub_server.answer("GET", "/health", 200, b"OK", "text/html; charset=utf-8")
    env = {name: value for name, value in os.environ.items() if not name.lower().endswith("proxy")}
    # A proxy that listens nowhere: a request sent through it would be refused.
    env["HTTP_PROXY"] = refused_url

    result = subprocess.run(
        [sys.executable, "-c", THROUGH_NO_PROXY, stub_server.url],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "True"


def test_mlflow_ui_command_serves_the_project_store_on_127_0_0_1_with_one_worker():
    serve = [sys.executable, "-m", "mlflow", "server", "--backend-store-uri", MLFLOW_TRACKING_URI]
    on_loopback = ["--host", "127.0.0.1", "--port", "5000", "--workers", "1"]

    assert mlflow_ui_command(5000) == [*serve, *on_loopback]


def test_mlflow_ui_environment_sets_mlflows_variables_and_leaves_out_credentials_and_proxies(
    monkeypatch,
):
    for name in LEFT_OUT:
        monkeypatch.setenv(name, "placeholder")
    monkeypatch.setenv("CODEFORCES_DATA_DIR", "/data")

    env = mlflow_ui_environment()

    assert {name: env[name] for name in MLFLOW_ENV} == MLFLOW_ENV
    assert env["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] == "false"
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert sorted(set(LEFT_OUT) & set(env)) == []
    assert env["CODEFORCES_DATA_DIR"] == "/data"
    assert os.environ["OPENAI_API_KEY"] == "placeholder"  # this process keeps them


def test_the_tags_read_are_those_that_log_method_and_mlflow_write():
    assert mlflow_links.RUN_KIND_TAG == RUN_KIND_TAG
    assert mlflow_links.LINKED_PROMPTS_TAG == LINKED_PROMPTS_TAG


def test_importing_mlflow_links_loads_no_heavy_library(heavy_after_import):
    assert heavy_after_import("codeforces_tags.serving.mlflow_links") == []
