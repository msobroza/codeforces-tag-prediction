"""Tests for ``serving up``: its checks, the three parts it starts, its supervision and its stop.

Nothing here runs docker, a model server, the MLflow UI or Streamlit: docker runs through
``package.run_command``, the processes through ``stack._spawn`` and the local model server
through ``local.start_local_server``, all of which ``Rig`` replaces; ``local.stop`` records the
part it would stop, and ``stack._pause``, the wait between two checks, plays Ctrl-C or a part
that dies. Two tests start real processes: ``_spawn``'s, and a port that listens.
"""

import argparse
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codeforces_tags.config import (
    PROJECT_ROOT,
    SERVING_APP_SCRIPT,
    SERVING_START_TIMEOUT_S,
    SERVING_UP_CHECK_INTERVAL_S,
)
from codeforces_tags.serving import __main__ as serving_main
from codeforces_tags.serving import local, package, stack
from codeforces_tags.serving.local import LocalServer
from codeforces_tags.serving.mlflow_links import mlflow_ui_command, mlflow_ui_environment
from codeforces_tags.serving.stack import COMPOSE_FILE, Hooks, StackError, UpOptions

PROG = "python -m codeforces_tags.serving"
NOTE = f"{PROG}: note: "
ERROR = f"{PROG}: error: "
# The compose command of tfidf's model server on the CPU, before its subcommand.
COMPOSE = [
    "env",
    "METHOD=tfidf",
    "TAG=latest",
    "PORT=8080",
    *["docker", "compose", "-f", str(COMPOSE_FILE), "-p", "codeforces-tags-up-tfidf-8080"],
    *["--profile", "cpu"],
]
INSPECT = ["docker", "image", "inspect", "--format", "{{.Id}}", "codeforces-tags/tfidf:latest"]
# What docker says of an image it does not hold.
NO_IMAGE = "Error response from daemon: No such image: codeforces-tags/tfidf:latest\n"
# The container's log, as docker compose logs prints it.
SERVER_LOG = "model-server-cpu-1  | loading tfidf\nmodel-server-cpu-1  | serving"
# The wait between two checks, before the rig replaces it.
PAUSE = stack._pause  # pylint: disable=protected-access,invalid-name
# The signals ignored while the parts stop: Ctrl-C, and the two that end the supervision as it.
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# A key that no line of up's output may hold.
SECRET = "sk-up-test-secret-0123456789"


@dataclass
class FakeProcess:
    """Stands in for a process that up started: running until a test gives it a status.

    Attributes:
        label: The part it stands for: ``ui``, ``app`` or ``server``.
        returncode: Its status once it exited; None while it runs.
    """

    label: str
    returncode: int | None = None

    def poll(self):
        """Return the process's status, None while it runs."""
        return self.returncode


def interrupt(rig):
    """Play Ctrl-C at the first check, once every part has started."""
    raise KeyboardInterrupt(rig)


@dataclass
class Rig:  # pylint: disable=too-many-instance-attributes
    """Stands in for docker, the processes, the local model server and the waits of up.

    Attributes:
        root: The stand-in of ``SERVING_BUILD_DIR``.
        answers: What each docker command prints (or raises), by ``key``.
        events: What happened, in order: each docker command's key, ``start <part>`` and
            ``stop <part>``, ``build`` and ``package``.
        commands: Every docker command run, in order.
        spawned: The command, environment and log of each process started.
        processes: The fake processes, by part.
        local_calls: The model folder, port and device of each local model server.
        local_failure: What waiting for the local model server raises; None to answer.
        ui_answers: What ``ui_alive`` answers in turn; the last one answers from then on.
        pings: What ``ping`` answers.
        busy: The ports in use.
        streamlit: What ``find_spec("streamlit")`` returns.
        missing: The dependency groups that the method lacks here.
        on_check: What happens at each check of the supervision: Ctrl-C by default.
        handlers: The SIGTERM handler at each wait.
    """

    root: Path
    answers: dict[str, object] = field(
        default_factory=lambda: {
            "image": "sha256:abc\n",
            "info": '{"runc": {}, "nvidia": {}}\n',
            "compose ps": "0123456789ab\n",
            "compose logs": SERVER_LOG + "\n",
        }
    )
    events: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    spawned: list[tuple[list[str], dict[str, str], Path]] = field(default_factory=list)
    processes: dict[str, FakeProcess] = field(default_factory=dict)
    local_calls: list[tuple[Path, int, str]] = field(default_factory=list)
    local_failure: BaseException | None = None
    ui_answers: list[bool] = field(default_factory=lambda: [False, True])
    pings: bool = True
    busy: tuple[int, ...] = ()
    streamlit: object = "a module spec"
    missing: tuple[str, ...] = ()
    on_check: object = interrupt
    handlers: list[object] = field(default_factory=list)

    def docker(self, command):
        """Record a docker command, then print what docker would, or fail as it would."""
        command = list(command)
        key = f"compose {command[len(COMPOSE)]}" if command[0] == "env" else command[1]
        self.commands.append(command)
        self.events.append(key)
        answer = self.answers.get(key, "")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def spawn(self, command, env, log):
        """Start no process: write its log's first lines and return a fake process."""
        label = "app" if "streamlit" in command else "ui"
        log.write_text(f"{label} line 1\n{label} line 2\n", encoding="utf-8")
        self.spawned.append((command, env, log))
        self.events.append(f"start {label}")
        self.processes[label] = FakeProcess(label)
        return self.processes[label]

    def start_local_server(self, model_dir, *, port, device):
        """Start no local model server: return a fake one beside its model folder."""
        self.local_calls.append((model_dir, port, device))
        self.events.append("start server")
        self.processes["server"] = FakeProcess("server")
        url, log = f"http://127.0.0.1:{port}", model_dir.parent / "server.log"
        return LocalServer(self.processes["server"], url, log)

    def wait_for_ping(self, server):
        """Answer at once, or raise ``local_failure``."""
        assert server.process is self.processes["server"]
        if self.local_failure is not None:
            raise self.local_failure

    def stop(self, server):
        """Record the part that ``local.stop`` would end."""
        self.events.append(f"stop {server.process.label}")

    def ui_alive(self, url):
        """Answer as ``ui_answers`` says, in turn."""
        assert url == "http://127.0.0.1:5000"
        return self.ui_answers.pop(0) if len(self.ui_answers) > 1 else self.ui_answers[0]

    def pause(self, seconds):
        """Record the SIGTERM handler, then play ``on_check`` at a check of the supervision."""
        self.handlers.append(signal.getsignal(signal.SIGTERM))
        if seconds == SERVING_UP_CHECK_INTERVAL_S:
            self.on_check(self)

    def package_method(self, name, *, allow_dirty=False):
        """Write a build context as packaging leaves it: the model and the staged weights."""
        self.events.append("package" + (" --allow-dirty" if allow_dirty else ""))
        folder = self.root / name
        (folder / "model").mkdir(parents=True)
        (folder / "weights").mkdir()
        (folder / "model" / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
        return folder

    def stops(self):
        """Return the stops, in order: the docker command down and each part's stop."""
        return [
            event for event in self.events if event.startswith("stop ") or event.endswith(" down")
        ]


@pytest.fixture(name="rig")
def rig_fixture(monkeypatch, tmp_path):
    """Replace docker, the processes, the local server and the waits of up by a ``Rig``.

    The store is a file of ``tmp_path``, no port is in use, Streamlit is installed, and the build
    folder is ``tmp_path / "build" / "serving"``.
    """
    rig = Rig(tmp_path / "build" / "serving")
    database = tmp_path / "mlflow.db"
    database.write_bytes(b"")
    monkeypatch.setattr(stack, "MLFLOW_TRACKING_URI", f"sqlite:///{database}")
    monkeypatch.setattr(stack, "SERVING_BUILD_DIR", rig.root)
    monkeypatch.setattr(package, "run_command", rig.docker)
    monkeypatch.setattr(package, "package_method", rig.package_method)
    monkeypatch.setattr(stack, "_spawn", rig.spawn)
    monkeypatch.setattr(stack, "_pause", rig.pause)
    monkeypatch.setattr(stack, "ping", lambda url: rig.pings)
    monkeypatch.setattr(stack, "ui_alive", rig.ui_alive)
    monkeypatch.setattr(stack, "port_in_use", lambda port: port in rig.busy)
    monkeypatch.setattr(stack, "find_spec", lambda name: rig.streamlit)
    monkeypatch.setattr(stack, "missing_groups", lambda name, spec: list(rig.missing))
    monkeypatch.setattr(local, "start_local_server", rig.start_local_server)
    monkeypatch.setattr(local, "wait_for_ping", rig.wait_for_ping)
    monkeypatch.setattr(local, "stop", rig.stop)
    return rig


def logs_of(rig, method="tfidf"):
    """Return the folder of up's logs of a method, resolved."""
    return rig.root.resolve() / method / "up"


def test_the_parser_gives_up_its_defaults():
    args = serving_main.build_parser().parse_args(["up", "--method", "tfidf"])

    assert (args.device, args.local, args.tag, args.build) == ("cpu", False, None, False)
    assert (args.port, args.app_port, args.mlflow_port) == (8080, 8501, 5000)
    assert (args.no_mlflow_ui, args.no_browser, args.allow_dirty) == (False, False, False)


@pytest.mark.parametrize(
    "argv",
    [["--device", "cuda"], ["--app-port", "0"], ["--mlflow-port", "70000"], ["--port", "x"]],
)
def test_the_parser_refuses_a_wrong_use_of_up_with_status_2(argv):
    with pytest.raises(SystemExit) as exit_info:
        serving_main.build_parser().parse_args(["up", "--method", "tfidf", *argv])

    assert exit_info.value.code == 2


def test_up_with_docker_starts_the_three_parts_then_stops_them_in_reverse_order_on_ctrl_c(
    rig, capsys
):
    status = serving_main.main(["up", "--method", "tfidf", "--no-browser"])

    out, err = capsys.readouterr()
    logs = logs_of(rig)
    up = [*COMPOSE, "up", "-d", "--wait", "--wait-timeout", str(SERVING_START_TIMEOUT_S)]
    assert status == 0
    ps = [*COMPOSE, "ps", "--status", "running", "--quiet"]
    assert rig.commands == [INSPECT, up, ps, [*COMPOSE, "down"]]
    assert rig.events == [
        "image",
        "compose up",
        "start ui",
        "start app",
        "compose ps",
        "stop app",
        "stop ui",
        "compose down",
    ]
    [(ui, ui_env, ui_log), (app, app_env, app_log)] = rig.spawned
    assert (ui, ui_env, ui_log) == (
        mlflow_ui_command(5000),
        mlflow_ui_environment(),
        logs / "mlflow-ui.log",
    )
    assert app == [
        sys.executable,
        *["-m", "streamlit", "run", str(SERVING_APP_SCRIPT)],
        *["--server.address", "127.0.0.1", "--server.port", "8501"],
        *["--server.showEmailPrompt", "false", "--server.headless", "true"],
    ]
    assert app_log == logs / "app.log"
    assert app_env["CODEFORCES_SERVING_URL"] == "http://127.0.0.1:8080"
    assert app_env["CODEFORCES_MLFLOW_UI_URL"] == "http://127.0.0.1:5000"
    assert out.splitlines() == [
        "the Streamlit app: http://127.0.0.1:8501",
        "the model server: http://127.0.0.1:8080 (tfidf on the cpu, codeforces-tags/tfidf:latest)",
        "the MLflow UI: http://127.0.0.1:5000",
        f"the logs: {logs}; the model server's: "
        f"`docker compose -f {COMPOSE_FILE} -p codeforces-tags-up-tfidf-8080 --profile cpu logs`",
        "Ctrl-C stops everything",
    ]
    assert err.splitlines() == [
        f"{NOTE}starting codeforces-tags/tfidf:latest with docker compose, until its method has "
        "loaded",
        f"{NOTE}stopping every part, which a second Ctrl-C does not interrupt",
        f"{NOTE}stopped the Streamlit app",
        f"{NOTE}stopped the MLflow UI",
        f"{NOTE}stopped the model server",
    ]
    # The UI was asked twice, a wait between; then the supervision's first check was Ctrl-C.
    assert rig.handlers == [stack._interrupt] * 2  # pylint: disable=protected-access


def test_up_lets_streamlit_open_the_browser_without_its_email_prompt_and_restores_handlers(rig):
    before = [signal.getsignal(signum) for signum in SIGNALS]

    serving_main.main(["up", "--method", "tfidf"])

    command = rig.spawned[1][0]
    assert "--server.headless" not in command
    # Not headless, Streamlit would ask for an email on its stdin, /dev/null, and exit.
    assert command[-2:] == ["--server.showEmailPrompt", "false"]
    assert [signal.getsignal(signum) for signum in SIGNALS] == before


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_or_sighup_during_the_supervisions_wait_stops_every_part_with_status_0(
    rig, monkeypatch, signum
):
    def terminated(seconds):
        # The signal comes from elsewhere while the supervision waits: `kill`, or the terminal
        # that closed.
        threading.Timer(0.2, os.kill, (os.getpid(), signum)).start()
        PAUSE(seconds * 10)

    monkeypatch.setattr(stack, "_pause", terminated)
    rig.ui_answers = [True]

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 0
    assert rig.stops() == ["stop app", "stop ui", "compose down"]


def test_signals_while_the_parts_stop_are_ignored_so_every_part_still_stops(rig, monkeypatch):
    before = [signal.getsignal(signum) for signum in SIGNALS]
    during = []

    def stop_under_signals(server):
        # A second Ctrl-C, a SIGTERM and a hangup reach up while it stops the app.
        during.append([signal.getsignal(signum) for signum in SIGNALS])
        for signum in (signal.SIGTERM, signal.SIGHUP):
            os.kill(os.getpid(), signum)
        PAUSE(0.05)
        rig.stop(server)

    monkeypatch.setattr(local, "stop", stop_under_signals)

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 0
    assert rig.stops() == ["stop app", "stop ui", "compose down"]
    assert during == [[signal.SIG_IGN] * 3] * 2
    assert [signal.getsignal(signum) for signum in SIGNALS] == before


def test_a_note_to_a_terminal_that_closed_is_lost_and_up_goes_on(monkeypatch):
    class ClosedTerminal(io.StringIO):
        """Stands in for the stderr of a terminal that hung up: every write fails."""

        def write(self, text):
            """Fail as a write to a hung-up terminal does."""
            raise OSError(5, "Input/output error", text)

    monkeypatch.setattr(sys, "stderr", ClosedTerminal())

    stack._note("stopped the Streamlit app")  # pylint: disable=protected-access


def test_up_without_the_mlflow_ui_needs_no_store_and_leaves_the_apps_link_alone(
    rig, monkeypatch, capsys
):
    monkeypatch.setattr(stack, "MLFLOW_TRACKING_URI", "sqlite:////nowhere/mlflow.db")
    monkeypatch.delenv("CODEFORCES_MLFLOW_UI_URL", raising=False)
    rig.busy = (5000,)  # the UI's port is not checked either

    status = serving_main.main(["up", "--method", "tfidf", "--no-mlflow-ui"])

    out = capsys.readouterr().out
    [(command, env, _)] = rig.spawned
    assert status == 0
    assert "streamlit" in command
    assert "CODEFORCES_MLFLOW_UI_URL" not in env
    assert "the MLflow UI: not started (--no-mlflow-ui)" in out.splitlines()
    assert rig.stops() == ["stop app", "compose down"]


def test_up_with_a_tag_and_a_gpu_runs_that_image_in_a_compose_project_of_its_own_port(rig):
    argv = ["--tag", "7972494", "--device", "gpu", "--port", "9090"]

    status = serving_main.main(["up", "--method", "tfidf", *argv])

    assert status == 0
    assert rig.commands[0] == ["docker", "info", "--format", "{{json .Runtimes}}"]
    assert rig.commands[1][-1] == "codeforces-tags/tfidf:7972494"
    assert rig.commands[2][:12] == [
        "env",
        "METHOD=tfidf",
        "TAG=7972494",
        "PORT=9090",
        *["docker", "compose", "-f", str(COMPOSE_FILE), "-p", "codeforces-tags-up-tfidf-9090"],
        *["--profile", "gpu"],
    ]


@pytest.mark.parametrize("dirty", [False, True])
def test_up_with_build_builds_the_image_as_build_does_before_it_checks_it(rig, monkeypatch, dirty):
    builds = []

    def build(args):
        builds.append(args)
        rig.events.append("build")
        return 0

    monkeypatch.setattr(serving_main, "_build", build)

    option = ["--allow-dirty"] if dirty else []

    status = serving_main.main(["up", "--method", "tfidf", "--build", *option])

    assert status == 0
    assert builds == [argparse.Namespace(method="tfidf", allow_dirty=dirty, keep_previous=False)]
    assert rig.events[:2] == ["build", "image"]


def test_up_local_packages_a_build_context_without_weights_then_serves_it_on_the_gpu(
    rig, monkeypatch, tmp_path, capsys
):
    # The build folder is a link to another disk: up names the resolved folders.
    (tmp_path / "disk").mkdir()
    disk = (tmp_path / "disk").resolve()
    rig.root.parent.mkdir(parents=True)
    rig.root.symlink_to(disk)
    (disk / "tfidf" / "model").mkdir(parents=True)  # as build leaves it: no staged weights
    (disk / "tfidf" / "model" / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")

    def repackage(name, *, allow_dirty):
        rig.events.append(f"package allow_dirty={allow_dirty}")
        (disk / name / "weights").mkdir()
        return rig.root / name

    monkeypatch.setattr(package, "package_method", repackage)
    argv = ["--local", "--device", "gpu", "--allow-dirty"]

    status = serving_main.main(["up", "--method", "tfidf", *argv])

    out, err = capsys.readouterr()
    assert status == 0
    assert rig.commands == []
    assert rig.local_calls == [(disk / "tfidf" / "model", 8080, "cuda")]
    assert rig.events == [
        "package allow_dirty=True",
        "start server",
        "start ui",
        "start app",
        "stop app",
        "stop ui",
        "stop server",
    ]
    assert f"{NOTE}{disk / 'tfidf'} holds no packaged model with its weights: packaging " in err
    assert "the model server: http://127.0.0.1:8080 (tfidf on the gpu, --local)" in out
    server_log = disk / "tfidf" / "server.log"
    assert f"the logs: {disk / 'tfidf' / 'up'}; the model server's: {server_log}" in out


def test_up_local_serves_a_packaged_build_context_as_it_is(rig):
    rig.package_method("tfidf")
    rig.events.clear()

    status = serving_main.main(["up", "--method", "tfidf", "--local", "--port", "18080"])

    assert status == 0
    assert "package" not in rig.events
    assert rig.local_calls == [(rig.root.resolve() / "tfidf" / "model", 18080, "cpu")]
    assert rig.spawned[1][1]["CODEFORCES_SERVING_URL"] == "http://127.0.0.1:18080"


def test_a_local_server_that_does_not_start_is_stopped_and_named_with_its_log(rig, capsys):
    rig.package_method("tfidf")
    rig.local_failure = RuntimeError(
        "The model server exited with 3 before it answered http://127.0.0.1:8080/ping:\n"
        "Traceback (most recent call last):\nImportError: no module named torch"
    )

    status = serving_main.main(["up", "--method", "tfidf", "--local"])

    err = capsys.readouterr().err.splitlines()
    assert status == 1
    assert rig.spawned == []
    assert rig.stops() == ["stop server"]
    assert err[-5:] == [
        "the last lines of the model server's log:",
        "Traceback (most recent call last):",
        "ImportError: no module named torch",
        f"{NOTE}stopped the model server",
        f"{ERROR}The model server exited with 3 before it answered http://127.0.0.1:8080/ping",
    ]


def test_a_container_that_compose_does_not_start_is_removed_and_named_with_its_log(rig, capsys):
    rig.answers["compose up"] = subprocess.CalledProcessError(
        1,
        ["docker"],
        stderr="dependency failed to start: container model-server-cpu-1 is unhealthy\n",
    )

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err
    assert status == 1
    assert rig.spawned == []
    assert rig.events == ["image", "compose up", "compose logs", "compose down"]
    assert f"the last lines of the model server's log:\n{SERVER_LOG}\n" in err
    assert err.endswith(
        f"{ERROR}docker compose did not start codeforces-tags/tfidf:latest: dependency failed to "
        "start: container model-server-cpu-1 is unhealthy\n"
    )


def test_a_model_server_that_does_not_answer_ping_is_stopped_before_anything_else_starts(
    rig, capsys
):
    rig.pings = False

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert rig.spawned == []
    assert rig.stops() == ["compose down"]
    assert capsys.readouterr().err.endswith(
        f"{ERROR}the model server at http://127.0.0.1:8080 does not answer /ping with 200\n"
    )


def test_an_mlflow_ui_that_never_answers_stops_what_started_in_reverse_order(
    rig, monkeypatch, capsys
):
    monkeypatch.setattr(stack, "SERVING_UP_UI_TIMEOUT_S", 0)

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err
    assert status == 1
    assert [event for event in rig.events if event.startswith("start ")] == ["start ui"]
    assert rig.stops() == ["stop ui", "compose down"]
    assert "the last lines of the MLflow UI's log:\nui line 1\nui line 2\n" in err
    assert err.endswith(
        f"{ERROR}the MLflow UI did not answer http://127.0.0.1:5000/health in 0 s\n"
    )


def test_an_mlflow_ui_that_exits_before_it_answers_is_named_with_its_status(
    rig, monkeypatch, capsys
):
    rig.ui_answers = [False]

    def exit_at_once(command, env, log):
        process = rig.spawn(command, env, log)
        process.returncode = 2
        return process

    monkeypatch.setattr(stack, "_spawn", exit_at_once)

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert rig.stops() == ["stop ui", "compose down"]
    assert capsys.readouterr().err.endswith(
        f"{ERROR}the MLflow UI exited with status 2 before it answered "
        "http://127.0.0.1:5000/health\n"
    )


def test_a_part_that_dies_stops_the_others_and_prints_the_end_of_its_log(rig, capsys):
    def app_dies(rig):
        rig.processes["app"].returncode = 1
        (logs_of(rig) / "app.log").write_text(
            "".join(f"app line {i}\n" for i in range(1, 31)), encoding="utf-8"
        )

    rig.on_check = app_dies

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err.splitlines()
    assert status == 1
    assert rig.stops() == ["stop app", "stop ui", "compose down"]
    start = err.index("the last lines of the Streamlit app's log:")
    assert err[start + 1 : start + 21] == [f"app line {i}" for i in range(11, 31)]
    assert err[-1] == (
        f"{ERROR}the Streamlit app exited with status 1: every part that up started was stopped"
    )


def test_a_container_that_stopped_running_is_named_with_its_log_read_before_it_is_removed(
    rig, capsys
):
    def container_stops(rig):
        rig.answers["compose ps"] = ""

    rig.on_check = container_stops

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err
    assert status == 1
    assert rig.events[-5:] == ["compose ps", "compose logs", "stop app", "stop ui", "compose down"]
    assert f"the last lines of the model server's log:\n{SERVER_LOG}\n" in err
    assert err.endswith(
        f"{ERROR}the model server is no longer running: every part that up started was stopped\n"
    )


def test_a_docker_that_answers_neither_ps_nor_logs_counts_as_a_container_stopped(rig, capsys):
    def docker_stops(rig):
        failure = subprocess.CalledProcessError(1, ["docker"], stderr="Cannot connect\n")
        rig.answers |= {"compose ps": failure, "compose logs": failure}

    rig.on_check = docker_stops

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err
    assert status == 1
    assert "the last lines of the model server's log" not in err
    assert "the model server is no longer running" in err


def test_a_part_that_cannot_be_stopped_is_named_and_the_others_still_stop(rig, capsys):
    rig.answers["compose down"] = subprocess.CalledProcessError(1, ["docker", "compose"])

    status = serving_main.main(["up", "--method", "tfidf"])

    err = capsys.readouterr().err
    assert status == 0
    assert rig.stops() == ["stop app", "stop ui", "compose down"]
    assert f"{NOTE}the model server could not be stopped: Command '['docker', 'compose']'" in err


@pytest.mark.parametrize("local_run", [False, True])
def test_up_notes_that_a_method_that_calls_an_api_pays_each_prediction_and_prints_no_key(
    rig, monkeypatch, capsys, local_run
):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    rig.package_method("jev-statement-code")
    option = ["--local"] if local_run else []

    status = serving_main.main(["up", "--method", "jev-statement-code", *option])

    out, err = capsys.readouterr()
    where = "this shell's environment" if local_run else "deploy/.env.serving (gitignored"
    assert status == 0
    assert (
        f"{NOTE}jev-statement-code calls an API: every prediction the app asks for is a paid "
        f"request, one problem each; its key, TYPESAFE_API_KEY, is read from {where}"
    ) in err
    assert SECRET not in out + err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--method", "f2llm_setfit"], "f2llm_setfit is not frozen: "),
        (["--method", "tfidf", "--local", "--build"], "--build and --tag name a Docker image"),
        (["--method", "tfidf", "--local", "--tag", "x"], "--build and --tag name a Docker image"),
        (
            ["--method", "tfidf", "--build", "--tag", "x"],
            "--build builds tfidf's image of HEAD and serves it as latest, removing older tags, "
            "so --tag cannot name the image served: pass one of the two",
        ),
        (
            ["--method", "tfidf", "--app-port", "8080"],
            "each part needs a port of its own: --port 8080, --app-port 8080, --mlflow-port 5000",
        ),
    ],
)
def test_up_refuses_a_method_or_options_it_cannot_serve_before_it_starts_anything(
    rig, capsys, argv, message
):
    status = serving_main.main(["up", *argv])

    err = capsys.readouterr().err
    assert status == 1
    assert err.startswith(f"{ERROR}{message}")
    assert len(err.splitlines()) == 1
    assert (rig.events, rig.spawned) == ([], [])


def test_up_refuses_a_port_in_use_and_names_its_option(rig, capsys):
    rig.busy = (8501, 5000)

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{ERROR}127.0.0.1:8501 (--app-port), 127.0.0.1:5000 (--mlflow-port) in use: stop what "
        "listens there, or pass another port\n"
    )


@pytest.mark.parametrize(
    ("streamlit", "missing", "expected"),
    [
        (None, (), "the dependency group app: run `uv run --group app " + PROG),
        ("spec", ("f2llm",), "group f2llm: run `uv run --group app --group f2llm "),
        (None, ("f2llm",), "group app, f2llm: run `uv run --group app --group f2llm"),
    ],
)
def test_up_names_the_groups_this_environment_lacks_and_the_command_that_brings_them(
    rig, capsys, streamlit, missing, expected
):
    # Only --local needs the method's groups.
    rig.streamlit, rig.missing = streamlit, missing
    option = ["--local"] if missing else []

    status = serving_main.main(["up", "--method", "tfidf", *option])

    err = capsys.readouterr().err
    assert status == 1
    assert expected in err
    assert err.endswith(" up --method tfidf ...`\n")
    assert rig.events == []


def test_up_refuses_a_store_that_does_not_exist_unless_it_starts_no_ui(rig, monkeypatch, capsys):
    monkeypatch.setattr(stack, "MLFLOW_TRACKING_URI", "sqlite:////nowhere/mlflow.db")

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{ERROR}/nowhere/mlflow.db does not exist: rebuild the store from committed files "
        "(notebooks/README.md, Figures), or pass --no-mlflow-ui\n"
    )
    assert rig.events == []


def test_up_on_the_gpu_refuses_a_docker_without_the_nvidia_runtime(rig, capsys):
    rig.answers["info"] = '{"io.containerd.runc.v2": {}, "runc": {}}\n'

    status = serving_main.main(["up", "--method", "tfidf", "--device", "gpu"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{ERROR}Docker lists no nvidia runtime: --device gpu needs the NVIDIA Container Toolkit "
        "and an NVIDIA driver of 580 or later; pass --device cpu, or --local\n"
    )
    assert rig.events == ["info"]


def test_up_refuses_an_image_docker_does_not_hold_and_names_the_build(rig, capsys):
    rig.answers["image"] = subprocess.CalledProcessError(1, INSPECT, stderr=NO_IMAGE)

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{ERROR}Docker cannot run codeforces-tags/tfidf:latest ({NO_IMAGE.strip()}): build it "
        f"with `{PROG} build --method tfidf`, or pass --build\n"
    )
    assert rig.events == ["image"]


def test_up_tells_a_docker_that_does_not_answer_from_an_image_it_lacks(rig, capsys):
    daemon = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.\n"
    rig.answers["image"] = subprocess.CalledProcessError(1, INSPECT, stderr=daemon)

    status = serving_main.main(["up", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{ERROR}Docker did not answer `docker image inspect codeforces-tags/tfidf:latest` "
        f"({daemon.strip()}): is its daemon running, and may this user use it?\n"
    )
    assert rig.events == ["image"]


def test_check_refuses_a_method_that_is_not_registered_and_a_device_up_does_not_know(rig):
    hooks = Hooks(rig.docker, rig.package_method, lambda method: None)

    with pytest.raises(StackError, match="'nope' is not a registered method"):
        stack.up(UpOptions("nope"), hooks)
    with pytest.raises(StackError, match="'tpu' is not a device of up"):
        stack.check(UpOptions("tfidf", device="tpu"), hooks)


def test_port_in_use_tells_whether_something_listens_on_a_port_of_127_0_0_1(refused_url):
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen()

        assert stack.port_in_use(listening.getsockname()[1])
    assert not stack.port_in_use(int(refused_url.rpartition(":")[2]))


def test_spawn_starts_a_part_from_the_repository_root_in_a_session_of_its_own(tmp_path):
    log = tmp_path / "part.log"
    script = (
        "import os, time\n"
        "print(os.getcwd(), os.getsid(0) == os.getpid(), os.environ['PART'], flush=True)\n"
        "time.sleep(60)"
    )
    env = {"PART": "the app", "PATH": os.environ["PATH"]}

    process = stack._spawn([sys.executable, "-c", script], env, log)  # pylint: disable=protected-access
    try:
        deadline = time.monotonic() + 30
        while not log.read_text(encoding="utf-8") and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        local.stop(LocalServer(process, "", log))

    assert log.read_text(encoding="utf-8") == f"{PROJECT_ROOT} True the app\n"
    assert process.returncode == -signal.SIGTERM


def test_importing_stack_loads_no_heavy_library(heavy_after_import):
    assert heavy_after_import("codeforces_tags.serving.stack") == []
