"""Tests for the shared fixtures of tests/conftest.py."""

import gc
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pytest
from mlflow.telemetry.client import get_telemetry_client

from codeforces_tags.config import MLFLOW_ENV, MLFLOW_ENV_DEFAULTS
from codeforces_tags.tracking.environment import mlflow_env


def mlflow_variables():
    """Return MLflow's environment variables and their values."""
    return {name: value for name, value in os.environ.items() if name.startswith("MLFLOW_")}


@pytest.fixture(name="environment_before")
def environment_before_fixture():
    """Snapshot MLflow's variables; once the test's other fixtures are torn down, check them."""
    before = mlflow_variables()
    yield
    assert mlflow_variables() == before


@pytest.mark.usefixtures("environment_before", "mlflow_environment")
def test_mlflow_environment_restores_what_mlflow_env_sets():
    # A variable that was absent must be absent again: otherwise every later test, and every
    # subprocess it starts, runs with MLflow's environment as mlflow_env() left it.
    mlflow_env()

    assert {name: os.environ[name] for name in MLFLOW_ENV} == MLFLOW_ENV
    assert set(MLFLOW_ENV_DEFAULTS) <= set(os.environ)


def test_the_session_imported_mlflow_with_its_telemetry_off():
    assert get_telemetry_client() is None


def test_a_figure_the_session_draws_and_closes_leaves_no_tk_object_to_collect():
    # On TkAgg, matplotlib's default with a display, a closed figure leaves Tk objects that the
    # garbage collector may delete on a worker thread, where Tk raises (tests/conftest.py).
    figure = plt.figure()
    figure.canvas.draw()
    plt.close(figure)

    tk_objects = [obj for obj in gc.get_objects() if type(obj).__module__ == "tkinter"]

    assert (matplotlib.get_backend(), tk_objects) == ("Agg", [])


@pytest.mark.parametrize(
    ("event", "args", "host"),
    [
        (
            "socket.getaddrinfo",
            ("config.mlflow-telemetry.io", 443, 0, 1, 6, 0),
            "config.mlflow-telemetry.io",
        ),
        ("socket.connect", (None, ("192.0.2.1", 443)), "192.0.2.1"),
    ],
)
def test_remote_connections_records_a_connection_outside_localhost(
    remote_connections, event, args, host
):
    sys.audit(event, *args)

    opened = remote_connections.copy()
    remote_connections.clear()

    assert opened == [(event, host, "MainThread")]


@pytest.mark.parametrize(
    ("event", "args"),
    [
        ("socket.getaddrinfo", (b"localhost", 5000, 0, 1, 6, 0)),
        ("socket.getaddrinfo", (None, 5000, 0, 1, 6, 0)),
        ("socket.connect", (None, ("127.0.0.1", 5000))),
        ("socket.connect", (None, ("::1", 5000, 0, 0))),
        ("socket.connect", (None, "/tmp/mlflow.sock")),
        ("socket.bind", (None, ("192.0.2.1", 443))),
    ],
)
def test_remote_connections_ignores_local_sockets_and_other_events(remote_connections, event, args):
    sys.audit(event, *args)

    assert not remote_connections
