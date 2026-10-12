"""A local model server of a packaged tagging method, for notebook 12 and the tests.

``start_local_server`` runs MLflow's own command line, ``mlflow models serve``, on the MLflow
model that ``package_method`` wrote (``build/serving/<method>/model``), in a subprocess on a port
of 127.0.0.1 with one worker: the scoring server the image runs behind nginx, without the
container. This module imports no mlflow.

The server's environment (``server_environment``) is the image's, so that a local answer is the
container's: MLflow imports ``codeforces_tags`` from ``model/code``, where ``config.PROJECT_ROOT``
is the MLflow model folder (``build/serving/<method>/model``) and the default weight caches do
not exist, so the weights come from the build folder's staged ``weights/`` through
``CODEFORCES_PRETRAINED_DIR`` and ``CODEFORCES_TRAINED_DIR``. Its standard input is
``/dev/null``: ``mlflow models serve`` starts uvicorn through ``bash -c``, which, given a socket
for stdin, sources ``~/.bashrc`` (here pyenv's shims) and could run a uvicorn without mlflow;
this interpreter's ``bin/`` comes first on ``PATH`` for the same reason. ``wait_for_ping`` waits
until the packaged method has loaded, and ``stop`` ends the server's whole process group.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import NamedTuple

from codeforces_tags.config import (
    DEVICE_ENV_VAR,
    DEVICES,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    PRETRAINED_DIR_ENV_VAR,
    SERVING_START_TIMEOUT_S,
    SERVING_STOP_TIMEOUT_S,
    TRAINED_DIR_ENV_VAR,
)
from codeforces_tags.serving.client import ping
from codeforces_tags.tracking.environment import mlflow_env

# The server's log, beside its model folder: build/serving/<method>/server.log.
SERVER_LOG = "server.log"
# How many of the log's last lines an error quotes.
LOG_TAIL_LINES = 20
# stop sends SIGTERM to the server's process group, then SIGKILL to what is left of it after
# this many seconds.
STOP_TIMEOUT_S = SERVING_STOP_TIMEOUT_S
_HOST = "127.0.0.1"
# The variables the image sets besides MLflow's and the device: nothing is downloaded, and no
# bytecode is written into model/code, which docker build copies.
_OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


class LocalServer(NamedTuple):
    """A model server started by ``start_local_server``.

    Attributes:
        process: The ``mlflow models serve`` process, the leader of the server's process group.
        url: The server's URL, ``http://127.0.0.1:<port>``.
        log: The file of its standard output and error, ``<build folder>/server.log``.
    """

    process: subprocess.Popen
    url: str
    log: Path


def free_port() -> int:
    """Return a port of 127.0.0.1 that nothing listens on.

    Returns:
        The port the system gave a socket bound to port 0, closed again.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((_HOST, 0))
        return probe.getsockname()[1]


def server_environment(
    model_dir: Path, device: str, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the environment of a local model server: the image's, for a packaged method.

    MLflow's variables come from ``os.environ`` once ``mlflow_env()`` set them (no pickle loaded,
    no telemetry, no hint to coding agents, explicit requirements), whatever ``base`` holds.

    Args:
        model_dir: The MLflow model, ``build/serving/<method>/model``; its parent holds the
            staged ``weights/``.
        device: The device of every torch method, ``cpu``, ``cuda`` or ``auto``
            (``config.DEVICES``).
        base: The environment to extend; ``os.environ`` when None.

    Returns:
        ``base`` with MLflow's variables, ``CODEFORCES_DEVICE``, ``CODEFORCES_PRETRAINED_DIR``
        and ``CODEFORCES_TRAINED_DIR`` (the build folder's ``weights/pretrained`` and
        ``weights/trained``), ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``,
        ``PYTHONDONTWRITEBYTECODE=1``, and this interpreter's ``bin/`` first on ``PATH``.

    Raises:
        ValueError: If ``device`` is not one of ``config.DEVICES``.
    """
    if device not in DEVICES:
        raise ValueError(f"{device!r} is not a device: {DEVICE_ENV_VAR} takes one of {DEVICES}.")
    mlflow_env()
    env = dict(os.environ if base is None else base)
    weights = Path(model_dir).resolve().parent / "weights"
    # Not resolved: a virtual environment's python is a link to an interpreter elsewhere, whose
    # bin/ holds no uvicorn.
    interpreter_bin = str(Path(sys.executable).parent)
    path = env.get("PATH")
    return (
        env
        | {name: os.environ[name] for name in (*MLFLOW_ENV, *MLFLOW_ENV_DEFAULTS)}
        | _OFFLINE
        | {
            DEVICE_ENV_VAR: device,
            PRETRAINED_DIR_ENV_VAR: str(weights / "pretrained"),
            TRAINED_DIR_ENV_VAR: str(weights / "trained"),
            "PATH": os.pathsep.join([interpreter_bin, path]) if path else interpreter_bin,
        }
    )


def start_local_server(
    model_dir: Path, *, port: int | None = None, device: str = "cpu"
) -> LocalServer:
    """Start ``mlflow models serve`` on a packaged method, on 127.0.0.1 with one worker.

    The server runs in its own session, from the build folder, and writes its output to
    ``<build folder>/server.log``; it keeps running after this returns. Wait for it with
    ``wait_for_ping`` and end it with ``stop``.

    Args:
        model_dir: The MLflow model that ``package_method`` wrote,
            ``build/serving/<method>/model``.
        port: The port on 127.0.0.1; a free one when None.
        device: The device of every torch method (``config.DEVICES``); the CPU by default.

    Returns:
        The server: its process, URL and log.

    Raises:
        ValueError: If ``device`` is not one of ``config.DEVICES``.
        FileNotFoundError: If ``model_dir`` holds no ``MLmodel``.
        RuntimeError: If something already listens on ``port``: it would answer instead.
    """
    model_dir = Path(model_dir).resolve()
    env = server_environment(model_dir, device)
    if not (model_dir / "MLmodel").is_file():
        raise FileNotFoundError(
            f"{model_dir} holds no MLmodel: package the method first (python -m "
            "codeforces_tags.serving package --method <name>)."
        )
    if port is None:
        port = free_port()
    elif _listening(port):
        raise RuntimeError(f"{_HOST}:{port} is in use: stop what listens there, or pass no port.")
    command = [sys.executable, "-m", "mlflow", "models", "serve", "-m", str(model_dir)]
    command += ["--env-manager", "local", "-h", _HOST, "-p", str(port), "-w", "1"]
    log = model_dir.parent / SERVER_LOG
    process = spawn(command, cwd=model_dir.parent, env=env, log=log)
    return LocalServer(process, f"http://{_HOST}:{port}", log)


def spawn(command: list[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> subprocess.Popen:
    """Start a server that outlives the call: in its own session, stdin ``/dev/null``, logged.

    Its own session keeps the terminal's Ctrl-C from reaching it, and lets ``stop`` end its whole
    process group.

    Args:
        command: The program and its arguments.
        cwd: The folder it runs in.
        env: Its whole environment.
        log: The file of its standard output and error, written afresh.

    Returns:
        The process.
    """
    with log.open("wb") as output:
        # The server outlives this call: stop() ends its process group.
        return subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_for_ping(server: LocalServer, timeout: float = SERVING_START_TIMEOUT_S) -> None:
    """Wait until a local server answers ``/ping``: once the packaged method has loaded.

    The process is checked after each ping: when it has exited, another server answered.

    Args:
        server: The server, from ``start_local_server``.
        timeout: Seconds to wait.

    Raises:
        RuntimeError: If the server exits first, with its status and the last lines of its log.
        TimeoutError: If it does not answer within ``timeout``, with the last lines of its log.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        answered = ping(server.url, timeout=2.0)
        if server.process.poll() is not None:
            raise RuntimeError(
                f"The model server exited with {server.process.returncode} before it answered "
                f"{server.url}/ping:\n{_tail(server.log)}"
            )
        if answered:
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"The model server did not answer {server.url}/ping in {timeout} s:\n{_tail(server.log)}"
    )


def stop(server: LocalServer) -> None:
    """End a local server's process group: SIGTERM, then SIGKILL to what is left of it.

    ``mlflow models serve`` runs uvicorn as a child process, which would otherwise keep the port.

    Args:
        server: The server, from ``start_local_server``.
    """
    for signum in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(server.process.pid, signum)
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while _group_alive(server.process) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not _group_alive(server.process):
            return


def _listening(port: int) -> bool:
    """Tell whether something listens on a port of 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex((_HOST, port)) == 0


def _group_alive(process: subprocess.Popen) -> bool:
    """Tell whether a process of the server's group is left, after reaping the server itself."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _tail(log: Path) -> str:
    """Return the last lines of a server's log."""
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return "\n".join(text.splitlines()[-LOG_TAIL_LINES:])
