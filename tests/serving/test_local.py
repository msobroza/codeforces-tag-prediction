"""Tests for the local model server: its environment, its command, its life, and tfidf served.

The unit tests fake ``subprocess.Popen`` or start small stand-in processes, each in a session of
its own, as the server's ``mlflow models serve`` runs; the end-to-end test packages the committed
``tfidf`` folder and serves it with MLflow's real command line on 127.0.0.1.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from codeforces_tags.config import (
    FOCUS_TAGS,
    MLFLOW_ENV,
    RAW_INPUT_COLUMNS,
    SERVING_START_TIMEOUT_S,
    TUNING_EXPORT_TOLERANCE,
)
from codeforces_tags.methods import load_method, model_dir
from codeforces_tags.serving import local, package
from codeforces_tags.serving.client import predict
from codeforces_tags.serving.local import (
    LocalServer,
    free_port,
    server_environment,
    start_local_server,
    stop,
    wait_for_ping,
)

# The variables the image sets besides MLflow's and the device.
OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")
# The most seconds a stand-in server may take to write a line to its log.
LINE_TIMEOUT_S = 30
# A stand-in server that ignores SIGTERM and starts a child process, as mlflow models serve
# starts uvicorn: only a SIGKILL to its whole process group ends both.
STUBBORN = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # its child leaves no zombie
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print("serving", flush=True)
time.sleep(60)
"""


def started(tmp_path, script):
    """Start a stand-in for a model server, in a session of its own, its output in a log."""
    log = tmp_path / "server.log"
    with log.open("wb") as output:
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", script],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return LocalServer(process, f"http://127.0.0.1:{free_port()}", log)


def wait_for_line(server, line):
    """Wait until a stand-in server has written a line to its log; fail after LINE_TIMEOUT_S."""
    deadline = time.monotonic() + LINE_TIMEOUT_S
    while line not in server.log.read_text(encoding="utf-8").splitlines():
        assert time.monotonic() < deadline, server.log.read_text(encoding="utf-8")
        time.sleep(0.05)


@pytest.fixture(name="model_folder")
def model_folder_fixture(tmp_path):
    """Return a stand-in for a packaged MLflow model: ``<build folder>/model``, with an MLmodel."""
    folder = tmp_path / "build" / "model"
    folder.mkdir(parents=True)
    (folder / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    return folder


@pytest.fixture(name="popen_calls")
def popen_calls_fixture(monkeypatch):
    """Record each ``subprocess.Popen`` call in the list returned, starting nothing.

    Each call is recorded as its command and its keyword arguments, the log file's name in place
    of the open file.
    """
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs | {"stdout": kwargs["stdout"].name}))
        return "the process"

    monkeypatch.setattr(local.subprocess, "Popen", fake_popen)
    return calls


def test_free_port_is_a_port_of_127_0_0_1_that_nothing_listens_on():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", free_port()))  # raises if something holds the port


def test_server_environment_is_this_one_with_the_variables_the_image_sets(
    model_folder, monkeypatch
):
    monkeypatch.setenv("CODEFORCES_DATA_DIR", "/data")
    monkeypatch.setenv("PATH", "/usr/bin")
    for name in ("CODEFORCES_DEVICE", *OFFLINE):
        monkeypatch.delenv(name, raising=False)

    env = server_environment(model_folder, "cuda")

    weights = model_folder.resolve().parent / "weights"
    assert env["CODEFORCES_DEVICE"] == "cuda"
    assert env["CODEFORCES_PRETRAINED_DIR"] == str(weights / "pretrained")
    assert env["CODEFORCES_TRAINED_DIR"] == str(weights / "trained")
    assert [env[name] for name in OFFLINE] == ["1", "1", "1"]
    # This interpreter's bin/ first: bash -c runs its uvicorn, never a shim's.
    assert env["PATH"] == f"{Path(sys.executable).parent}{os.pathsep}/usr/bin"
    assert {name: env[name] for name in MLFLOW_ENV} == MLFLOW_ENV
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert env["CODEFORCES_DATA_DIR"] == "/data"
    # The environment of this process gets none of the server's own variables.
    assert sorted({"CODEFORCES_DEVICE", *OFFLINE} & set(os.environ)) == []


def test_server_environment_extends_a_given_base_with_mlflows_variables(model_folder, monkeypatch):
    monkeypatch.setenv("CODEFORCES_DATA_DIR", "/data")

    env = server_environment(model_folder, "cpu", base={"HOME": "/home/someone"})

    assert env["HOME"] == "/home/someone"
    assert env["PATH"] == str(Path(sys.executable).parent)  # the base has no PATH
    assert {name: env[name] for name in MLFLOW_ENV} == MLFLOW_ENV
    assert "CODEFORCES_DATA_DIR" not in env  # nothing else of this process's environment


@pytest.mark.parametrize("device", ["gpu", "CUDA", ""])
def test_server_environment_refuses_a_device_outside_devices(model_folder, device):
    with pytest.raises(ValueError, match="is not a device: CODEFORCES_DEVICE takes one of"):
        server_environment(model_folder, device)


def test_start_local_server_runs_mlflow_models_serve_on_127_0_0_1_with_one_worker(
    model_folder, popen_calls, monkeypatch
):
    monkeypatch.setattr(local, "free_port", lambda: 43210)

    server = start_local_server(model_folder, device="cuda")

    build = model_folder.resolve().parent
    assert server == LocalServer("the process", "http://127.0.0.1:43210", build / "server.log")
    [(command, kwargs)] = popen_calls
    serve = [sys.executable, "-m", "mlflow", "models", "serve", "-m", str(build / "model")]
    options = ["--env-manager", "local", "-h", "127.0.0.1", "-p", "43210", "-w", "1"]
    assert command == [*serve, *options]
    # From the build folder, in a session of its own, with no stdin: bash -c then sources no
    # ~/.bashrc.
    assert kwargs == {
        "cwd": build,
        "env": server_environment(model_folder, "cuda"),
        "stdin": subprocess.DEVNULL,
        "stdout": str(build / "server.log"),
        "stderr": subprocess.STDOUT,
        "start_new_session": True,
    }


def test_start_local_server_serves_on_a_given_free_port(model_folder, popen_calls):
    port = free_port()

    server = start_local_server(model_folder, port=port)

    assert server.url == f"http://127.0.0.1:{port}"
    assert popen_calls[0][0][-4:] == ["-p", str(port), "-w", "1"]
    assert popen_calls[0][1]["env"]["CODEFORCES_DEVICE"] == "cpu"


def test_start_local_server_refuses_a_device_outside_devices(model_folder, popen_calls):
    with pytest.raises(ValueError, match="is not a device"):
        start_local_server(model_folder, device="gpu")

    assert popen_calls == []


def test_start_local_server_refuses_a_folder_without_an_mlmodel(tmp_path, popen_calls):
    with pytest.raises(FileNotFoundError, match="holds no MLmodel: package the method first"):
        start_local_server(tmp_path)

    assert popen_calls == []


def test_start_local_server_refuses_a_port_in_use(model_folder, popen_calls):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"127.0.0.1:{port} is in use"):
            start_local_server(model_folder, port=port)

    assert popen_calls == []


def test_wait_for_ping_returns_once_the_server_answers(tmp_path, stub_server):
    stub_server.answer("GET", "/ping", 200, b"\n")
    server = started(tmp_path, "import time; time.sleep(60)")._replace(url=stub_server.url)

    try:
        wait_for_ping(server, timeout=30)
    finally:
        stop(server)

    assert [(request.method, request.path) for request in stub_server.requests] == [
        ("GET", "/ping")
    ]


def test_wait_for_ping_raises_the_last_lines_of_the_log_when_the_server_exits_first(tmp_path):
    script = "for i in range(1, 26):\n    print(f'line {i}')\nraise SystemExit(3)"
    server = started(tmp_path, script)

    with pytest.raises(RuntimeError, match="exited with 3 before it answered") as raised:
        wait_for_ping(server, timeout=30)

    assert str(raised.value).splitlines()[1:] == [f"line {i}" for i in range(6, 26)]


def test_wait_for_ping_raises_a_timeout_error_when_the_server_never_answers(tmp_path):
    server = started(
        tmp_path, "print('loading the method', flush=True)\nimport time\ntime.sleep(60)"
    )
    wait_for_line(server, "loading the method")

    try:
        with pytest.raises(TimeoutError, match=r"/ping in 1 s:\nloading the method$"):
            wait_for_ping(server, timeout=1)
    finally:
        stop(server)

    assert server.process.returncode == -signal.SIGTERM


def test_stop_kills_a_server_group_that_ignores_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "STOP_TIMEOUT_S", 1)
    server = started(tmp_path, STUBBORN)
    wait_for_line(server, "serving")

    stop(server)

    assert server.process.returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.killpg(server.process.pid, 0)


def test_importing_local_loads_no_heavy_library(heavy_after_import):
    assert heavy_after_import("codeforces_tags.serving.local") == []


def test_a_packaged_tfidf_served_locally_predicts_as_the_method_does(
    tmp_path, monkeypatch, raw_problems
):
    monkeypatch.setattr(package, "SERVING_BUILD_DIR", tmp_path)
    monkeypatch.setattr(package, "run_command", lambda command, *, cwd=None: "")  # no uv export
    build = package.package_method("tfidf", tmp_path / "tfidf", allow_dirty=True)
    # Both problems have a null optional field, which the request sends as JSON null.
    problems = raw_problems.loc[[30, 20], list(RAW_INPUT_COLUMNS)]

    server = start_local_server(build / "model")
    try:
        wait_for_ping(server, timeout=SERVING_START_TIMEOUT_S)
        served = predict(server.url, problems)
    finally:
        stop(server)

    method = load_method(model_dir("tfidf"))
    tags, scores = method.predict(problems), method.scores(problems)
    for prediction, sample_id in zip(served, problems.index, strict=True):
        assert prediction.method == "tfidf"
        assert set(prediction.tags) == {tag for tag in FOCUS_TAGS if tags.loc[sample_id, tag]}
        expected = scores.loc[sample_id, list(FOCUS_TAGS)].to_dict()
        assert prediction.scores == pytest.approx(expected, rel=0, abs=TUNING_EXPORT_TOLERANCE)
        assert prediction.thresholds == method.tag_thresholds()[list(FOCUS_TAGS)].to_dict()
        assert (prediction.failed, prediction.cost_usd) == (None, None)
    # The server wrote no bytecode into model/code, which docker build copies.
    assert not list((build / "model" / "code").rglob("__pycache__"))
