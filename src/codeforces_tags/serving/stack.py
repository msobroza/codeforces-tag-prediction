"""Everything needed to interact with a frozen tagging method, in one terminal: ``serving up``.

``up`` checks first, before it starts anything: the method is registered and frozen, the ports
of 127.0.0.1 it would serve on are free, Streamlit (and, with ``--local``, the method's
dependency groups) is installed here, the MLflow store exists, and the model server's runtime is
ready: the method's image and, for ``--device gpu``, Docker's ``nvidia`` runtime; or, with
``--local``, a packaged build context, which it packages first when it lacks one. Then it starts
three parts, each on 127.0.0.1, each logged: the model server (``docker compose`` on the image
``serving build`` made, or ``serving.local``'s ``mlflow models serve`` on the build context), the
MLflow UI of the project's store (as ``serving mlflow-ui`` runs it) and the Streamlit app, which
asks the one and links to the other.

It then supervises them in the foreground. Every ``config.SERVING_UP_CHECK_INTERVAL_S`` seconds
it checks that each part still runs: a process it started that exited, or a container that
``docker compose ps --status running`` no longer lists, is a part that died (a container whose
method stopped answering is not checked: Docker's own healthcheck marks it, and the app says so).
Ctrl-C (SIGINT), SIGTERM or SIGHUP (the terminal closed) ends the supervision with status 0; a
part that died ends it with its log's last lines and a ``StackError``. Either way, and on any
failure midway, every part that started is stopped, in reverse order: the app, the MLflow UI
(each process group sent SIGTERM, then SIGKILL after ``config.SERVING_STOP_TIMEOUT_S``, by
``local.stop``), then the model server (``docker compose down``, or ``local.stop``). The three
signals are ignored while the parts stop, so that a second Ctrl-C cannot leave one running; only
a SIGKILL of up (or the kernel's out-of-memory killer) leaves the parts running, each in a
session of its own, and the container. The model server's container is in a compose project of
its own (``compose_command``), so that up never recreates or removes another one's container.

This module imports neither mlflow nor torch: it runs MLflow's and Streamlit's command lines in
subprocesses, as ``local`` does, each through ``_spawn``. The docker commands, packaging and the
image's build come from its caller (``Hooks``): ``serving/__main__.py`` passes
``package.run_command``, ``package.package_method`` and its own ``build``, which import mlflow.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from codeforces_tags.cli_methods import api_keys
from codeforces_tags.config import (
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    SERVING_APP_PORT,
    SERVING_APP_SCRIPT,
    SERVING_BUILD_DIR,
    SERVING_IMAGE_PREFIX,
    SERVING_MLFLOW_UI_URL,
    SERVING_MLFLOW_UI_URL_ENV_VAR,
    SERVING_PORT,
    SERVING_START_TIMEOUT_S,
    SERVING_UP_CHECK_INTERVAL_S,
    SERVING_UP_COMPOSE_PROJECT,
    SERVING_UP_LOG_DIR,
    SERVING_UP_UI_TIMEOUT_S,
    SERVING_URL_ENV_VAR,
)
from codeforces_tags.methods import METHOD_FILE, METHODS, MethodSpec, missing_groups, model_dir
from codeforces_tags.serving import local
from codeforces_tags.serving.client import ping
from codeforces_tags.serving.mlflow_links import (
    mlflow_ui_command,
    mlflow_ui_environment,
    ui_alive,
)

COMMAND = "python -m codeforces_tags.serving"
# The devices of up: the compose profiles, which set CODEFORCES_DEVICE to cpu and cuda.
DEVICES = ("cpu", "gpu")
# The port of the MLflow UI that the app links to by default.
DEFAULT_UI_PORT = urlsplit(SERVING_MLFLOW_UI_URL).port
# The compose file of the model server, which reads METHOD, TAG and PORT from its environment.
COMPOSE_FILE = PROJECT_ROOT / "deploy" / "docker-compose.yml"
# The logs of the parts that up starts as processes, in build/serving/<method>/up/.
UI_LOG = "mlflow-ui.log"
APP_LOG = "app.log"
# The names of the three parts, as the messages say them.
SERVER = "the model server"
UI = "the MLflow UI"
APP = "the Streamlit app"
_HOST = "127.0.0.1"
# The seconds between two checks of a starting MLflow UI.
_UI_POLL_S = 0.5
# The signals that end the supervision as Ctrl-C does: `kill`'s, and the terminal's hangup.
_ENDING_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
# What docker says of an image it does not hold, as opposed to a daemon it cannot reach.
_NO_IMAGE = "No such image"


class StackError(RuntimeError):
    """A check that failed, a part that did not start or died: one line on stderr, status 1."""


@dataclass(frozen=True)
class UpOptions:  # pylint: disable=too-many-instance-attributes
    """What ``serving up`` starts, as its options say.

    Attributes:
        method: The tagging method's registry key.
        device: ``cpu`` or ``gpu``: the compose profile, or ``CODEFORCES_DEVICE`` (``cpu`` or
            ``cuda``) of a ``--local`` server.
        local: Serve the build context with ``serving.local`` rather than the Docker image.
        tag: The image's tag; None for ``latest`` (Docker only).
        build: Build the image first, as ``serving build`` does (Docker only).
        port: The model server's port of 127.0.0.1.
        app_port: The Streamlit app's port of 127.0.0.1.
        mlflow_port: The MLflow UI's port of 127.0.0.1; None to start no UI.
        browser: Let Streamlit open the app in a browser; False runs it headless.
        allow_dirty: Package, or build, from uncommitted inputs, as ``package`` and ``build``
            allow them with ``--allow-dirty``.
    """

    method: str
    device: str = "cpu"
    local: bool = False
    tag: str | None = None
    build: bool = False
    port: int = SERVING_PORT
    app_port: int = SERVING_APP_PORT
    mlflow_port: int | None = DEFAULT_UI_PORT
    browser: bool = True
    allow_dirty: bool = False


class Hooks(NamedTuple):
    """What ``up`` asks its caller to run: the commands that import mlflow or run docker.

    Attributes:
        run_command: Runs a command and returns its standard output (``package.run_command``),
            raising ``subprocess.CalledProcessError`` when it fails.
        package_method: Packages a method's build context and returns its folder
            (``package.package_method``), called with the method and ``allow_dirty``.
        build_image: Builds a method's image as ``serving build`` does, older images removed,
            called with the method and ``allow_dirty``.
    """

    run_command: Callable[[Sequence[str]], str]
    package_method: Callable[..., Path]
    build_image: Callable[..., object]


@dataclass(frozen=True)
class _Part:
    """A part that up started: how to tell it died, how to stop it, the end of its log."""

    name: str
    stopped: Callable[[], str | None]
    stop: Callable[[], object]
    tail: Callable[[], str]


def up(options: UpOptions, hooks: Hooks) -> int:
    """Check, start the model server, the MLflow UI and the app, then supervise them until Ctrl-C.

    While it runs, SIGTERM and SIGHUP end it as Ctrl-C does; while the parts stop, all three are
    ignored; their previous handlers come back on return.

    Args:
        options: What to start.
        hooks: The docker commands, packaging and the image's build.

    Returns:
        0, after Ctrl-C, SIGTERM or SIGHUP: every part that started was stopped.

    Raises:
        StackError: If a check fails, a part does not start, or a part dies; every part that
            started is stopped first, and a part that died prints its log's last lines.
        subprocess.CalledProcessError: If a docker command fails, e.g. ``docker info``.
        OSError: If a program cannot be run (no ``docker`` on the PATH) or a log written.
    """
    started: list[_Part] = []
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, *_ENDING_SIGNALS)}
    for signum in _ENDING_SIGNALS:
        signal.signal(signum, _interrupt)
    try:
        context = check(options, hooks)
        logs = _log_folder(options.method)
        server_url, server_log = _start_server(options, hooks, context, started)
        ui_url = _start_ui(options.mlflow_port, logs, started)
        app_url = _start_app(options, server_url, ui_url, logs, started)
        _print_parts(options, (app_url, server_url, ui_url), logs, server_log)
        part, how = _supervise(started)
        raise _death(part, how)
    except KeyboardInterrupt:
        _note("stopping every part, which a second Ctrl-C does not interrupt")
        return 0
    finally:
        _stop_ignoring_signals(started, previous)


def check(options: UpOptions, hooks: Hooks) -> Path | None:
    """Check everything ``up`` needs, before it starts anything; build or package when due.

    Args:
        options: What to start.
        hooks: The docker commands, packaging and the image's build.

    Returns:
        The packaged build context, resolved, with ``--local``; None for Docker.

    Raises:
        StackError: If the method is not registered or not frozen; if ``--build`` or ``--tag``
            come with ``--local``, ``--build`` with ``--tag``, or the device is not one of
            ``DEVICES``; if two ports are
            the same, or one is in use; if Streamlit, or a dependency group of a ``--local``
            method, is not installed; if the MLflow store does not exist (unless no UI); if
            Docker lacks the method's image, or the ``nvidia`` runtime for ``--device gpu``, or
            does not answer.
    """
    spec = _frozen_spec(options.method)
    _check_options(options)
    _check_ports(options)
    _check_installed(options, spec)
    if options.mlflow_port is not None:
        _check_store()
    context = _build_context(options, hooks) if options.local else None
    if not options.local:
        _check_image(options, hooks)
    if spec.calls_api:
        _note_api(options, spec)
    return context


def port_in_use(port: int) -> bool:
    """Tell whether something listens on a port of 127.0.0.1: a connection to it succeeds.

    Args:
        port: The port.

    Returns:
        True if a connection to ``127.0.0.1:<port>`` succeeds within a second.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex((_HOST, port)) == 0


def image_reference(options: UpOptions) -> str:
    """Name the image the model server runs: ``codeforces-tags/<method>:<tag>``.

    Args:
        options: What to start; a tag of None is ``latest``.

    Returns:
        The image's reference.
    """
    return f"{SERVING_IMAGE_PREFIX}/{options.method}:{options.tag or 'latest'}"


def compose_command(options: UpOptions) -> list[str]:
    """Return the start of every ``docker compose`` command of the model server.

    The compose file reads METHOD, TAG and PORT from its environment, which ``env`` sets, since
    ``package.run_command`` passes none of its own. The project is up's own, named after the
    method and the port (``config.SERVING_UP_COMPOSE_PROJECT``), not the compose file's: two
    ``up`` runs on two ports, or a compose run by hand, never share a container.

    Args:
        options: What to start: the method, the tag, the port and the device (the profile).

    Returns:
        ``env METHOD=... TAG=... PORT=... docker compose -f deploy/docker-compose.yml -p
        codeforces-tags-up-<method>-<port> --profile <device>``.
    """
    variables = [f"METHOD={options.method}", f"TAG={options.tag or 'latest'}"]
    project = f"{SERVING_UP_COMPOSE_PROJECT}-{options.method}-{options.port}"
    compose = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project]
    return ["env", *variables, f"PORT={options.port}", *compose, "--profile", options.device]


def _frozen_spec(method: str) -> MethodSpec:
    """Return a method's spec, refusing one that is not registered or not frozen."""
    if method not in METHODS:
        raise StackError(f"{method!r} is not a registered method: the registry has {list(METHODS)}")
    if not (model_dir(method) / METHOD_FILE).is_file():
        raise StackError(
            f"{method} is not frozen: {model_dir(method)} holds no {METHOD_FILE}, so run its "
            "freeze first"
        )
    return METHODS[method]


def _check_options(options: UpOptions) -> None:
    """Refuse a device up does not know, Docker's options with --local, and --build with --tag."""
    if options.device not in DEVICES:
        raise StackError(f"{options.device!r} is not a device of up: {DEVICES}")
    if options.local and (options.build or options.tag is not None):
        raise StackError(
            "--build and --tag name a Docker image, which --local does not run: --local serves "
            f"the build context of {options.method}, packaged first when it has none"
        )
    if options.build and options.tag is not None:
        raise StackError(
            f"--build builds {options.method}'s image of HEAD and serves it as latest, removing "
            "older tags, so --tag cannot name the image served: pass one of the two"
        )


def _check_ports(options: UpOptions) -> None:
    """Refuse two parts on one port, and a port of 127.0.0.1 that something listens on."""
    ports = {"--port": options.port, "--app-port": options.app_port}
    if options.mlflow_port is not None:
        ports["--mlflow-port"] = options.mlflow_port
    if len(set(ports.values())) < len(ports):
        named = ", ".join(f"{option} {port}" for option, port in ports.items())
        raise StackError(f"each part needs a port of its own: {named}")
    busy = [f"{_HOST}:{port} ({option})" for option, port in ports.items() if port_in_use(port)]
    if busy:
        raise StackError(f"{', '.join(busy)} in use: stop what listens there, or pass another port")


def _check_installed(options: UpOptions, spec: MethodSpec) -> None:
    """Refuse an environment without Streamlit, or without a --local method's groups."""
    groups = [] if find_spec("streamlit") else ["app"]
    if options.local:
        groups += missing_groups(options.method, spec)
    if groups:
        flags = " ".join(f"--group {group}" for group in dict.fromkeys(["app", *groups]))
        raise StackError(
            f"this environment lacks the dependency group {', '.join(groups)}: run "
            f"`uv run {flags} {COMMAND} up --method {options.method} ...`"
        )


def _check_store() -> None:
    """Refuse a store that does not exist: the MLflow UI would create it empty."""
    database = Path(MLFLOW_TRACKING_URI.removeprefix("sqlite:///"))
    if not database.is_file():
        raise StackError(
            f"{database} does not exist: rebuild the store from committed files "
            "(notebooks/README.md, Figures), or pass --no-mlflow-ui"
        )


def _build_context(options: UpOptions, hooks: Hooks) -> Path:
    """Return the method's packaged build context, packaging it first when it has none."""
    folder = SERVING_BUILD_DIR.resolve() / options.method
    if (folder / "model" / "MLmodel").is_file() and (folder / "weights").is_dir():
        return folder
    _note(f"{folder} holds no packaged model with its weights: packaging {options.method} first")
    packaged = hooks.package_method(options.method, allow_dirty=options.allow_dirty)
    return Path(packaged).resolve()


def _check_image(options: UpOptions, hooks: Hooks) -> None:
    """Refuse a GPU without Docker's nvidia runtime; build the image if asked; check it exists."""
    if options.device == "gpu":
        output = hooks.run_command(["docker", "info", "--format", "{{json .Runtimes}}"])
        runtimes = json.loads(output) or {}
        if "nvidia" not in runtimes:
            raise StackError(
                "Docker lists no nvidia runtime: --device gpu needs the NVIDIA Container Toolkit "
                "and an NVIDIA driver of 580 or later; pass --device cpu, or --local"
            )
    if options.build:
        hooks.build_image(options.method, allow_dirty=options.allow_dirty)
    reference = image_reference(options)
    try:
        hooks.run_command(["docker", "image", "inspect", "--format", "{{.Id}}", reference])
    except subprocess.CalledProcessError as error:
        said = _last_line(error.stderr)
        if _NO_IMAGE not in said:
            raise StackError(
                f"Docker did not answer `docker image inspect {reference}` ({said}): is its "
                "daemon running, and may this user use it?"
            ) from None
        raise StackError(
            f"Docker cannot run {reference} ({said}): build it with "
            f"`{COMMAND} build --method {options.method}`, or pass --build"
        ) from None


def _note_api(options: UpOptions, spec: MethodSpec) -> None:
    """Say that every prediction of a method that calls an API is paid, and where its key is."""
    where = (
        "this shell's environment"
        if options.local
        else "deploy/.env.serving (gitignored: copy deploy/.env.serving.example)"
    )
    _note(
        f"{options.method} calls an API: every prediction the app asks for is a paid request, "
        f"one problem each; its key, {' and '.join(api_keys(spec))}, is read from {where}"
    )


def _log_folder(method: str) -> Path:
    """Create and return the folder of up's logs, ``build/serving/<method>/up/``, resolved."""
    folder = SERVING_BUILD_DIR.resolve() / method / SERVING_UP_LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _start_server(
    options: UpOptions, hooks: Hooks, context: Path | None, started: list[_Part]
) -> tuple[str, str]:
    """Start the model server, wait until it answers /ping; return its URL and where its log is."""
    if context is None:
        url, log = _start_docker(options, hooks, started)
    else:
        url, log = _start_local(options, context, started)
    if not ping(url):
        raise StackError(f"{SERVER} at {url} does not answer /ping with 200")
    return url, log


def _start_docker(options: UpOptions, hooks: Hooks, started: list[_Part]) -> tuple[str, str]:
    """Start the model server's container with compose, and wait until it is healthy."""
    compose = compose_command(options)

    def stopped() -> str | None:
        return None if _container_running(hooks, compose) else "is no longer running"

    def stop() -> object:
        return hooks.run_command([*compose, "down"])

    def tail() -> str:
        return _compose_logs(hooks, compose)

    started.append(_Part(SERVER, stopped, stop, tail))  # a failed up leaves a container to remove
    _note(f"starting {image_reference(options)} with docker compose, until its method has loaded")
    try:
        hooks.run_command(
            [*compose, "up", "-d", "--wait", "--wait-timeout", str(SERVING_START_TIMEOUT_S)]
        )
    except subprocess.CalledProcessError as error:
        _print_tail(SERVER, tail())
        raise StackError(
            f"docker compose did not start {image_reference(options)}: {_last_line(error.stderr)}"
        ) from None
    return f"http://{_HOST}:{options.port}", f"`{' '.join(compose[4:])} logs`"


def _start_local(options: UpOptions, context: Path, started: list[_Part]) -> tuple[str, str]:
    """Start ``mlflow models serve`` on the build context, and wait until it answers /ping."""
    device = "cuda" if options.device == "gpu" else "cpu"
    try:
        server = local.start_local_server(context / "model", port=options.port, device=device)
        started.append(_process_part(SERVER, server))
        local.wait_for_ping(server)
    except (RuntimeError, TimeoutError) as error:
        first, _, rest = str(error).partition("\n")
        _print_tail(SERVER, rest)
        raise StackError(first.rstrip(":")) from None
    return server.url, str(server.log)


def _start_ui(port: int | None, logs: Path, started: list[_Part]) -> str | None:
    """Start the MLflow UI as ``serving mlflow-ui`` runs it, and wait until it answers /health."""
    if port is None:
        return None
    url, log = f"http://{_HOST}:{port}", logs / UI_LOG
    server = local.LocalServer(
        _spawn(mlflow_ui_command(port), mlflow_ui_environment(), log), url, log
    )
    started.append(_process_part(UI, server))
    deadline = time.monotonic() + SERVING_UP_UI_TIMEOUT_S
    while time.monotonic() < deadline:
        if ui_alive(url):
            return url
        if server.process.poll() is not None:
            _print_tail(UI, _tail(log))
            raise StackError(
                f"{UI} exited with status {server.process.returncode} before it answered "
                f"{url}/health"
            )
        _pause(_UI_POLL_S)
    _print_tail(UI, _tail(log))
    raise StackError(f"{UI} did not answer {url}/health in {SERVING_UP_UI_TIMEOUT_S} s")


def _start_app(
    options: UpOptions, server_url: str, ui_url: str | None, logs: Path, started: list[_Part]
) -> str:
    """Start the Streamlit app, which asks the model server and links to the MLflow UI."""
    command = [sys.executable, "-m", "streamlit", "run", str(SERVING_APP_SCRIPT)]
    command += ["--server.address", _HOST, "--server.port", str(options.app_port)]
    # Without a credentials file, a Streamlit that is not headless asks for an email on its
    # stdin, /dev/null, and exits: the prompt is skipped, whether or not it opens a browser.
    command += ["--server.showEmailPrompt", "false"]
    if not options.browser:
        command += ["--server.headless", "true"]
    env = dict(os.environ) | {SERVING_URL_ENV_VAR: server_url}
    if ui_url is not None:
        env[SERVING_MLFLOW_UI_URL_ENV_VAR] = ui_url
    url, log = f"http://{_HOST}:{options.app_port}", logs / APP_LOG
    started.append(_process_part(APP, local.LocalServer(_spawn(command, env, log), url, log)))
    return url


def _spawn(command: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    """Start a long-running part from the repository root, as ``local.spawn`` starts a server.

    Its own session keeps Ctrl-C from reaching it: up stops it, whole process group and all.
    """
    return local.spawn(command, cwd=PROJECT_ROOT, env=env, log=log)


def _process_part(name: str, server: local.LocalServer) -> _Part:
    """Return the part of a process up started: dead once it exited, stopped by ``local.stop``."""

    def stopped() -> str | None:
        status = server.process.poll()
        return None if status is None else f"exited with status {status}"

    return _Part(name, stopped, lambda: local.stop(server), lambda: _tail(server.log))


def _print_parts(
    options: UpOptions, urls: tuple[str, str, str | None], logs: Path, server_log: str
) -> None:
    """Say where each part is, where the logs are, and how to stop everything."""
    app_url, server_url, ui_url = urls
    runtime = "--local" if options.local else image_reference(options)
    print(f"{APP}: {app_url}")
    print(f"{SERVER}: {server_url} ({options.method} on the {options.device}, {runtime})")
    print(f"{UI}: {ui_url}" if ui_url else f"{UI}: not started (--no-mlflow-ui)")
    print(f"the logs: {logs}; {SERVER}'s: {server_log}")
    print("Ctrl-C stops everything", flush=True)


def _supervise(parts: list[_Part]) -> tuple[_Part, str]:
    """Check each part every ``SERVING_UP_CHECK_INTERVAL_S`` seconds; return the first that died."""
    while True:
        for part in parts:
            how = part.stopped()
            if how is not None:
                return part, how
        _pause(SERVING_UP_CHECK_INTERVAL_S)


def _death(part: _Part, how: str) -> StackError:
    """Print the end of a dead part's log, read before it is stopped; return the error to raise."""
    _print_tail(part.name, part.tail())
    return StackError(f"{part.name} {how}: every part that up started was stopped")


def _stop_ignoring_signals(parts: list[_Part], previous: dict[int, object]) -> None:
    """Stop every part with Ctrl-C, SIGTERM and SIGHUP ignored, then restore their handlers.

    A child process inherits the ignored signals, so the terminal's second Ctrl-C reaches
    neither this loop nor ``docker compose down``.
    """
    for signum in previous:
        signal.signal(signum, signal.SIG_IGN)
    try:
        _stop_all(parts)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _stop_all(parts: list[_Part]) -> None:
    """Stop every part that started, the last first; a part that cannot be stopped is named."""
    for part in reversed(parts):
        try:
            part.stop()
        except (OSError, subprocess.CalledProcessError) as error:
            _note(f"{part.name} could not be stopped: {error}")
        else:
            _note(f"stopped {part.name}")


def _container_running(hooks: Hooks, compose: list[str]) -> bool:
    """Tell whether compose lists a running container of the model server."""
    try:
        return bool(hooks.run_command([*compose, "ps", "--status", "running", "--quiet"]).strip())
    except subprocess.CalledProcessError:  # e.g. the docker daemon stopped
        return False


def _compose_logs(hooks: Hooks, compose: list[str]) -> str:
    """Return the last lines of the model server's container log, or nothing."""
    try:
        output = hooks.run_command(
            [*compose, "logs", "--no-color", "--tail", str(local.LOG_TAIL_LINES)]
        )
    except subprocess.CalledProcessError:
        return ""
    return output.rstrip("\n")


def _print_tail(name: str, tail: str) -> None:
    """Print the last lines of a part's log on stderr, if it has any."""
    if tail.strip():
        print(f"the last lines of {name}'s log:\n{tail}", file=sys.stderr)


def _tail(log: Path) -> str:
    """Return the last lines of a log, or nothing when it does not exist."""
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return "\n".join(text.splitlines()[-local.LOG_TAIL_LINES :])


def _interrupt(_signum: int, _frame: object) -> None:
    """End up's supervision on SIGTERM as Ctrl-C ends it."""
    raise KeyboardInterrupt


def _pause(seconds: float) -> None:
    """Wait between two checks: the seam the tests replace."""
    time.sleep(seconds)


def _last_line(text: str | None) -> str:
    """Return the last line of a command's output, e.g. docker's own error."""
    return (text or "").strip().rpartition("\n")[2]


def _note(message: str) -> None:
    """Print a note on stderr, which keeps stdout for where each part is.

    A terminal that closed (SIGHUP) makes the write fail: the note is lost, and the stop goes on.
    """
    with contextlib.suppress(OSError):
        print(f"{COMMAND}: note: {message}", file=sys.stderr)
