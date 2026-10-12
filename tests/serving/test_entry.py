"""Tests for the deployment entry file that MLflow's model server runs.

MLflow runs a copy of the entry file, kept in each served model, which coverage does not see:
these tests also import the file and call its wrapper directly. The method it serves is the score
fake ``edge_fake`` of ``tests/methods/fakes/conformance_fakes.py``, registered in ``METHODS``
while a test runs, local methods that watch the wrapper's lock, or the frozen
``jev-statement-code``, which no test asks: the command line's fake Jev session (``cli_jev``)
stands in for TypeSafe's. MLflow's scoring server, called in process, shows what a client reads
of the wrapper's error; and ruff lets only this file of the package, beside its packaging and its
command, import mlflow at module level.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import scoring_server

from codeforces_tags import methods
from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import FOCUS_TAGS, PROJECT_ROOT, RAW_INPUT_COLUMNS
from codeforces_tags.methods import MethodSpec, model_dir, save_method
from codeforces_tags.serving import entry
from codeforces_tags.serving.records import served_records
from codeforces_tags.tracking.environment import mlflow_env

# The score fake that the wrapper loads from a method folder, registered in METHODS.
REGISTRY = {
    "edge_fake": MethodSpec(module="conformance_fakes", kind="score", requirements=("pandas",))
}
# The spec of the local methods that watch the lock: a module that no test may import.
LOCAL_SPEC = MethodSpec(module="codeforces_tags.no_such_module", kind="other", requirements=())
# The method that calls an API, frozen: two problems in one request are refused.
JEV = "jev-statement-code"
# The modules of the package, and those that a per-file-ignore of pyproject.toml lets import mlflow
# at module level (ruff's TID253): this entry file, the packaging and the command.
SERVING = PROJECT_ROOT / "src" / "codeforces_tags" / "serving"
MLFLOW_IMPORTERS = {"entry.py", "package.py", "__main__.py"}


@pytest.fixture(name="edge_fake")
def edge_fake_fixture(fakes):
    """Return a score fake that scores a problem asked alone 1.0, above its threshold of 0.5."""
    return fakes.EdgeFake(offset=1.0, threshold=0.5)


@pytest.fixture(name="edge_folder")
def edge_folder_fixture(edge_fake, monkeypatch, tmp_path):
    """Register ``edge_fake`` in ``METHODS``, save the fake to a method folder, and return it."""
    monkeypatch.setattr(methods, "METHODS", REGISTRY)
    save_method(edge_fake, tmp_path / "method")
    return tmp_path / "method"


def loaded_model(folder):
    """Return the wrapper with a method folder loaded, as MLflow's model server loads it."""
    model = entry.ServedMethodModel()
    model.load_context(SimpleNamespace(artifacts={"method": str(folder)}))
    return model


def saved_model(folder, path, monkeypatch):
    """Save a method folder as a served model from code at ``path``, and load it as MLflow does."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # loading a model may add to sys.path
    mlflow_env()  # no uv file in the model, no pickle loaded; the fixtures restore it
    mlflow.pyfunc.save_model(
        path,
        python_model=entry.__file__,
        artifacts={"method": str(folder)},
        pip_requirements=["mlflow"],
    )
    return mlflow.pyfunc.load_model(str(path))


def mlflow_import_codes(module):
    """Return the codes ruff reports for a module-level ``import mlflow`` in a serving module."""
    lint = [sys.executable, "-m", "ruff", "check", "--select=TID253", "--output-format=json"]
    linted = subprocess.run(
        [*lint, "--stdin-filename", module.relative_to(PROJECT_ROOT).as_posix(), "-"],
        cwd=PROJECT_ROOT,
        input="import mlflow\n",
        text=True,
        capture_output=True,
        check=False,  # ruff exits with 1 when it reports a violation
    )
    return [violation["code"] for violation in json.loads(linted.stdout)]


class LockWatcher:  # pylint: disable=too-few-public-methods
    """A local method that predicts no tag and records whether the wrapper's lock is held."""

    name = "lock_watcher"
    tags = FOCUS_TAGS

    def __init__(self, lock):
        """Watch ``lock``; no prediction yet."""
        self.lock = lock
        self.held = []

    def predict(self, raw_problems):
        """Record whether the lock is held, then predict no tag."""
        self.held.append(self.lock.locked())
        return pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags))


class BlockingMethod:  # pylint: disable=too-few-public-methods
    """A local method whose prediction waits until the test releases it; it counts its calls."""

    name = "blocking"
    tags = FOCUS_TAGS

    def __init__(self):
        """Start with no call, and not released."""
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, raw_problems):
        """Count the call and say so, then wait for the release and predict no tag."""
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=10)
        return pd.DataFrame(0, index=raw_problems.index, columns=list(self.tags))


def test_the_wrapper_answers_the_served_records_of_the_method_it_loaded(
    edge_fake, edge_folder, raw_problems
):
    model = loaded_model(edge_folder)

    served = model.predict(None, raw_problems.loc[[10]])

    assert model.spec == REGISTRY["edge_fake"]
    expected = served_records(edge_fake, REGISTRY["edge_fake"], raw_problems.loc[[10]])
    pd.testing.assert_frame_equal(served, expected)


def test_the_wrapper_loads_through_the_registry_of_the_loading_code(edge_folder, monkeypatch):
    monkeypatch.setattr(methods, "METHODS", {})

    with pytest.raises(ValueError, match="not registered"):
        loaded_model(edge_folder)


def test_the_wrapper_holds_its_lock_while_the_method_predicts(raw_problems):
    model = entry.ServedMethodModel()
    model.method, model.spec = LockWatcher(model.lock), LOCAL_SPEC

    served = model.predict(None, raw_problems)

    assert model.method.held == [True]
    assert not model.lock.locked()
    assert served[list(FOCUS_TAGS)].eq(0).all().all()


def test_two_requests_never_predict_at_once(raw_problems):
    # The scoring server runs each request in a thread: the second waits for the first.
    model = entry.ServedMethodModel()
    model.method, model.spec = BlockingMethod(), LOCAL_SPEC
    requests = [threading.Thread(target=model.predict, args=(None, raw_problems)) for _ in range(2)]

    requests[0].start()
    assert model.method.entered.wait(timeout=10)
    requests[1].start()
    requests[1].join(timeout=0.5)  # time enough to start predicting, were it not for the lock
    calls_while_the_first_predicts = model.method.calls
    model.method.release.set()
    for request in requests:
        request.join(timeout=10)

    assert calls_while_the_first_predicts == 1
    assert model.method.calls == 2


def test_a_cli_error_reaches_the_client_as_one_mlflow_error_without_its_cause(
    cli_jev, raw_problems
):
    model = loaded_model(model_dir(JEV))
    with pytest.raises(CliError) as refused:
        served_records(model.method, model.spec, raw_problems)

    with pytest.raises(MlflowException) as raised:
        model.predict(None, raw_problems)

    assert raised.value.message == str(refused.value)
    assert raised.value.get_http_status_code() == 500
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert cli_jev.asked == []
    assert not model.lock.locked()


def test_the_scoring_server_answers_a_cli_error_with_a_500_whose_message_quotes_the_input_first(
    cli_jev, raw_problems, monkeypatch, tmp_path
):
    # What /invocations sends back for the wrapper's MlflowException: its JSON, at its status.
    model = saved_model(model_dir(JEV), tmp_path / "served", monkeypatch)
    problem = raw_problems.loc[10, list(RAW_INPUT_COLUMNS)].tolist()
    body = {"dataframe_split": {"columns": list(RAW_INPUT_COLUMNS), "data": [problem, problem]}}

    with pytest.raises(MlflowException) as raised:
        scoring_server.invocations(json.dumps(body).encode(), "application/json", model, None)

    answer = json.loads(raised.value.serialize_as_json())
    quote, error = answer["message"].rsplit("'. \nError: ", 1)
    assert raised.value.get_http_status_code() == 500
    assert (answer["error_code"], "stack_trace" in answer) == ("INTERNAL_ERROR", False)
    assert quote.startswith("Failed to predict data '")
    assert error.startswith(f"{JEV} sends a paid request per problem")
    assert error.endswith("this request holds 2.")
    assert cli_jev.asked == []


def test_the_entry_file_ends_by_handing_mlflow_its_wrapper():
    last_line = Path(entry.__file__).read_text(encoding="utf-8").rstrip().splitlines()[-1]

    assert last_line == "mlflow.models.set_model(ServedMethodModel())"


def test_mlflow_loads_the_entry_file_as_a_model_from_code_and_serves_the_records(
    edge_fake, edge_folder, raw_problems, monkeypatch, tmp_path
):
    model = saved_model(edge_folder, tmp_path / "served", monkeypatch)

    served = model.predict(raw_problems.loc[[10]])

    expected = served_records(edge_fake, REGISTRY["edge_fake"], raw_problems.loc[[10]])
    pd.testing.assert_frame_equal(served, expected)


def test_only_the_entry_file_packaging_and_command_of_serving_may_import_mlflow_at_module_level():
    # A per-file-ignore widened to the package (serving/*) would let each of its modules import it.
    modules = sorted(SERVING.glob("*.py"))

    exempt = [module.name for module in modules if not mlflow_import_codes(module)]

    assert exempt == sorted(MLFLOW_IMPORTERS.intersection(module.name for module in modules))


def test_importing_the_entry_file_lets_mlflow_start_neither_telemetry_nor_its_hint(fresh_import):
    # The package sets MLflow's import-time variables before the entry file imports mlflow.
    assert fresh_import("codeforces_tags.serving.entry") == {
        "telemetry": False,
        "agent_hint": False,
    }
