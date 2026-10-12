"""Tests for the serving commands: package, build, clean-images, manifests, smoke and mlflow-ui.

Nothing here runs docker, git or uv: they run through ``package.run_command``, which
``FakeDocker`` replaces, and packaging is ``FakePackaging``'s. ``smoke`` asks the stub server of
``tests/serving/conftest.py`` on 127.0.0.1, set up as a model server (``model_server``), and
``mlflow-ui`` starts no server: ``subprocess.run`` is ``FakeRun``, and the port it is given is
``refused_url``'s, bound and never listened on (``ui_port``).
"""

import json
import os
import re
import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import mlflow
import pytest
import yaml

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import FOCUS_TAGS, PROJECT_ROOT, SERVING_SMOKE_COUNT
from codeforces_tags.data.labels import focus_labels
from codeforces_tags.methods import METHODS
from codeforces_tags.serving import __main__ as serving_main
from codeforces_tags.serving import mlflow_links, package
from codeforces_tags.serving.client import request_payload
from codeforces_tags.serving.mlflow_links import mlflow_ui_command, mlflow_ui_environment
from codeforces_tags.serving.problems import FrozenRow
from codeforces_tags.serving.records import METHOD_COLUMN, score_column, threshold_column
from codeforces_tags.tracking.environment import use_store

PROG = "python -m codeforces_tags.serving"
# HEAD's short sha in these tests, and the commit a build context was packaged at: all digits,
# as a short sha may be, so that a test sees the overlay keep the tag a string.
SHA = "7972494"
COMMIT = SHA + "0" * 33
# The image built last (two tags) and an older one, as docker 28 lists them, the older first.
NEW_ID = "sha256:" + "b" * 64
OLD_ID = "sha256:" + "a" * 64
NEWER = "2026-10-11 11:52:03 +0000 UTC"
OLDER = "2026-10-10 07:00:00 +0000 UTC"
# The line of a removed tag: docker image ls gives the size of its whole image, shared layers too.
REMOVED = "removed {} (an image of 1.02GB; layers it shares with other images stay)"
# The end of a failed docker build's log, which BuildKit writes on stderr.
BUILD_LOG = (
    "#9 [5/6] RUN pip install --no-deps --require-hashes -r /tmp/requirements.lock.txt\n"
    "#9 ERROR: process did not complete successfully: exit code: 1\n"
    "ERROR: failed to build: failed to solve: exit code: 1\n"
)
# What docker says when it refuses to remove the last tag of an image that a container uses.
CONFLICT = (
    'Error response from daemon: conflict: unable to remove repository reference "{}" (must '
    "force) - container 0123456789ab is using its referenced image aaaaaaaaaaaa\n"
)
# The served method of the smoke tests, a score method whose score_atol (9e-05) allows a flip
# next to a per-tag threshold, and the per-tag threshold of every focus tag there.
SERVED = "f2llm_late_fusion"
THRESHOLD = 0.5
# nginx's answer to /ping while the model server behind it loads its tagging method.
NGINX_502 = b"<html><head><title>502 Bad Gateway</title></head></html>\r\n"
# What load_test_set says when $CODEFORCES_DATA_DIR names no dataset.
NO_DATASET = "the test set of /nowhere cannot be read: [Errno 2] No such file or directory"


def image_row(image_id, tag, created, repository="codeforces-tags/tfidf"):
    """Return one line of ``docker image ls --no-trunc --format '{{json .}}'``."""
    values = {"CreatedAt": created, "ID": image_id, "Repository": repository, "Tag": tag}
    return json.dumps(values | {"Size": "1.02GB"})


LISTING = [
    image_row(OLD_ID, "1111111", OLDER),
    image_row(NEW_ID, SHA, NEWER),
    image_row(NEW_ID, "latest", NEWER),
]


def scores_of(chosen):
    """Return a score per focus tag: those of ``chosen``, 0.1 for the others."""
    return {tag: chosen.get(tag, 0.1) for tag in FOCUS_TAGS}


# The scores of the model server's answer, and of each frozen test row unless a test changes it.
SCORES = scores_of({"trees": 0.9})


def served_record(scores, method=SERVED):
    """Return a model server's record of one problem: its tags, scores and per-tag thresholds."""
    record = {METHOD_COLUMN: method}
    record |= {tag: int(score >= THRESHOLD) for tag, score in scores.items()}
    record |= {score_column(tag): score for tag, score in scores.items()}
    return record | {threshold_column(tag): THRESHOLD for tag in scores}


def frozen_row(scores):
    """Return the frozen test row of some scores, its tags compared with the same thresholds."""
    tags = {tag: int(score >= THRESHOLD) for tag, score in scores.items()}
    return FrozenRow(tags=tags, scores=dict(scores))


@dataclass
class FakeDocker:
    """Stands in for ``package.run_command``: records each command and answers as docker would.

    Attributes:
        listing: The lines that ``docker image ls`` prints.
        in_use: The references whose ``docker image rm`` docker refuses: a container uses them.
        build_fails: Whether ``docker build`` fails.
        commands: Every command run, in order.
    """

    listing: list[str] = field(default_factory=lambda: list(LISTING))
    in_use: tuple[str, ...] = ()
    build_fails: bool = False
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, command):
        """Record the command, then print what docker would, or fail as it would."""
        command = list(command)
        self.commands.append(command)
        if command[1] == "build" and self.build_fails:
            raise subprocess.CalledProcessError(1, command, output="", stderr=BUILD_LOG)
        if command[1:3] == ["image", "rm"] and command[3] in self.in_use:
            raise subprocess.CalledProcessError(
                1, command, output="", stderr=CONFLICT.format(command[3])
            )
        outputs = {
            ("image", "inspect"): f"{NEW_ID} 1020304050\n",
            ("image", "ls"): "".join(f"{line}\n" for line in self.listing),
            ("image", "prune"): "Total reclaimed space: 0B\n",
            ("builder", "prune"): "ID\tRECLAIMABLE\tSIZE\tLAST ACCESSED\nTotal:\t1.21GB\n",
        }
        return outputs.get(tuple(command[1:3]), "")


@dataclass
class FakePackaging:
    """Stands in for ``package.package_method``: writes a build context as packaging leaves it.

    The context, ``<root>/<method>`` unless ``out`` is given, holds ``model/MLmodel``, whose
    metadata says it was packaged at ``COMMIT`` (with uncommitted inputs when ``dirty``), and a
    staged weight file.

    Attributes:
        root: The stand-in of ``SERVING_BUILD_DIR``.
        dirty: The ``code_dirty`` of the next build context.
        calls: The method, ``out`` and ``allow_dirty`` of each call.
    """

    root: Path
    dirty: bool = False
    calls: list[tuple[str, Path | None, bool]] = field(default_factory=list)

    def package_method(self, name, out=None, *, allow_dirty=False):
        """Write the build context of method ``name``; return its folder."""
        self.calls.append((name, out, allow_dirty))
        folder = out or self.root / name
        (folder / "model").mkdir(parents=True)
        (folder / "weights" / "pretrained").mkdir(parents=True)
        (folder / "weights" / "pretrained" / "model.safetensors").write_bytes(b"weights")
        metadata = {"method": name, "git_commit": COMMIT, "code_dirty": self.dirty}
        mlmodel = yaml.safe_dump({"metadata": metadata})
        (folder / "model" / "MLmodel").write_text(mlmodel, encoding="utf-8")
        return folder


@dataclass
class FakeRun:
    """Stands in for ``subprocess.run`` under mlflow-ui: records the server's command, starts none.

    Attributes:
        failure: What the run raises, e.g. the ``KeyboardInterrupt`` of Ctrl-C; None to return.
        returncode: The status the server exits with.
        calls: The command, the environment and ``check`` of each run.
    """

    failure: BaseException | None = None
    returncode: int = 0
    calls: list[tuple[list[str], dict[str, str], bool]] = field(default_factory=list)

    def __call__(self, command, *, env, check):
        """Record the run, then raise ``failure`` or return the server's status."""
        self.calls.append((command, env, check))
        if self.failure is not None:
            raise self.failure
        return subprocess.CompletedProcess(command, self.returncode)


@pytest.fixture(name="docker")
def docker_fixture(monkeypatch):
    """Replace ``package.run_command`` by a ``FakeDocker``, and return it."""
    fake = FakeDocker()
    monkeypatch.setattr(package, "run_command", fake)
    return fake


@pytest.fixture(name="packaging")
def packaging_fixture(monkeypatch, tmp_path):
    """Replace packaging by a ``FakePackaging`` under ``tmp_path``, HEAD's short sha by ``SHA``."""
    fake = FakePackaging(tmp_path / "build" / "serving")
    monkeypatch.setattr(package, "package_method", fake.package_method)
    monkeypatch.setattr(package, "short_sha", lambda: SHA)
    monkeypatch.setattr(serving_main, "SERVING_BUILD_DIR", fake.root)
    return fake


@pytest.fixture(name="model_server")
def model_server_fixture(stub_server):
    """Set the stub server up as a model server of SERVED that has loaded its tagging method.

    ``/ping`` answers 200, and each ``/invocations`` the record of ``SCORES``, as MLflow's scoring
    server answers about one problem: ``{"predictions": [record]}``. A test may answer
    ``/invocations`` otherwise.
    """
    stub_server.answer("GET", "/ping", body=b"\n")
    stub_server.answer("POST", "/invocations", body={"predictions": [served_record(SCORES)]})
    return stub_server


@pytest.fixture(name="frozen_rows")
def frozen_rows_fixture(monkeypatch, raw_problems):
    """Stand in for the test set, the three raw problems, and for their frozen test rows of SERVED.

    Returns:
        The frozen rows by sample id, each that of ``SCORES``, as ``frozen_prediction`` reads
        them: a test may change them.
    """
    rows = {sample_id: frozen_row(SCORES) for sample_id in raw_problems.index}
    test_set = (raw_problems, focus_labels(raw_problems))
    monkeypatch.setattr(serving_main, "load_test_set", lambda: test_set)
    monkeypatch.setattr(
        serving_main, "frozen_prediction", lambda name, sample_id: rows.get(sample_id)
    )
    return rows


@pytest.fixture(name="ui_port")
def ui_port_fixture(refused_url):
    """Return the port of ``refused_url``: bound, never listened on, so mlflow-ui may take it."""
    return urlsplit(refused_url).port


@pytest.fixture(name="ui_server")
def ui_server_fixture(monkeypatch, store):
    """Point mlflow-ui and its server at the test's store; make ``subprocess.run`` a FakeRun."""
    monkeypatch.setattr(serving_main, "MLFLOW_TRACKING_URI", store["tracking_uri"])
    monkeypatch.setattr(mlflow_links, "MLFLOW_TRACKING_URI", store["tracking_uri"])
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def test_the_parser_gives_each_command_its_defaults():
    parser = serving_main.build_parser()

    smoke = parser.parse_args(["smoke", "--method", "tfidf"])
    build = parser.parse_args(["build", "--method", "tfidf"])
    clean = parser.parse_args(["clean-images", "--method", "tfidf"])
    manifests = parser.parse_args(["manifests", "--method", "tfidf", "--device", "cpu"])

    assert (smoke.url, smoke.count) == (None, SERVING_SMOKE_COUNT)
    assert (build.allow_dirty, build.keep_previous) == (False, False)
    assert (clean.keep, clean.all) == (None, False)
    assert (manifests.tag, manifests.registry) == (None, None)
    assert parser.parse_args(["mlflow-ui"]).port == 5000


@pytest.mark.parametrize(
    "argv",
    [
        ["smoke", "--method", "tfidf", "--count", "0"],
        ["package", "--method", "no-such-method"],
        ["clean-images", "--method", "tfidf", "--keep", "aaaa", "--all"],
        ["manifests", "--method", "tfidf", "--device", "cuda"],
        ["mlflow-ui", "--port", "70000"],
        ["mlflow-ui", "--port", "0"],
    ],
)
def test_the_parser_refuses_a_wrong_use_with_status_2(argv):
    # The parser alone: should it let a wrong use through, no command runs (no mlflow server).
    with pytest.raises(SystemExit) as exit_info:
        serving_main.build_parser().parse_args(argv)

    assert exit_info.value.code == 2


def test_the_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["codeforces_tags.serving", "--help"])
    # Run the module afresh, as `python -m` would.
    monkeypatch.delitem(sys.modules, "codeforces_tags.serving.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("codeforces_tags.serving", run_name="__main__")

    assert exit_info.value.code == 0


def test_package_writes_the_build_context_where_out_says_and_prints_it(packaging, tmp_path, capsys):
    out = tmp_path / "context"

    status = serving_main.main(["package", "--method", "tfidf", "--out", str(out), "--allow-dirty"])

    assert status == 0
    assert packaging.calls == [("tfidf", out, True)]
    assert capsys.readouterr().out == f"packaged tfidf in {out}\n"


def test_a_refused_package_fails_on_one_line(monkeypatch, capsys):
    def refuse(name, *_, **__):
        raise package.PackagingError(f"{name} is not frozen")

    monkeypatch.setattr(package, "package_method", refuse)

    status = serving_main.main(["package", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == f"{PROG}: error: tfidf is not frozen\n"


def test_build_tags_the_short_sha_and_latest_then_removes_older_images_cache_and_weights(
    packaging, docker, capsys
):
    status = serving_main.main(["build", "--method", "tfidf"])

    folder = packaging.root / "tfidf"
    out = capsys.readouterr().out
    assert status == 0
    assert packaging.calls == [("tfidf", None, False)]
    assert docker.commands == [
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--build-arg",
            "METHOD=tfidf",
            "-t",
            f"codeforces-tags/tfidf:{SHA}",
            "-t",
            "codeforces-tags/tfidf:latest",
            str(folder),
        ],
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}} {{.Size}}",
            f"codeforces-tags/tfidf:{SHA}",
        ],
        ["docker", "image", "ls", "--no-trunc", "--format", "{{json .}}", "codeforces-tags/tfidf"],
        ["docker", "image", "rm", "codeforces-tags/tfidf:1111111"],
        ["docker", "image", "prune", "--force", "--filter", "label=codeforces-tags.method=tfidf"],
        ["docker", "builder", "prune", "--force", "--filter", "type=source.local"],
    ]
    assert not (folder / "weights").exists()
    assert (folder / "model" / "MLmodel").is_file()
    assert f"built codeforces-tags/tfidf:{SHA} and codeforces-tags/tfidf:latest: image " in out
    assert REMOVED.format("codeforces-tags/tfidf:1111111") in out
    assert "dropped BuildKit's cached build contexts: Total:\t1.21GB" in out


def test_build_tags_an_image_packaged_with_uncommitted_changes_sha_dirty(packaging, docker):
    packaging.dirty = True

    status = serving_main.main(["build", "--method", "tfidf", "--allow-dirty"])

    assert status == 0
    assert packaging.calls == [("tfidf", None, True)]
    assert f"codeforces-tags/tfidf:{SHA}-dirty" in docker.commands[0]


def test_build_with_keep_previous_removes_nothing(packaging, docker):
    status = serving_main.main(["build", "--method", "tfidf", "--keep-previous"])

    assert status == 0
    assert [command[1] for command in docker.commands] == ["build", "image"]
    assert (packaging.root / "tfidf" / "weights" / "pretrained" / "model.safetensors").is_file()


def test_a_failed_build_removes_nothing_and_prints_the_end_of_its_log_then_one_error_line(
    packaging, docker, capsys
):
    docker.build_fails = True

    status = serving_main.main(["build", "--method", "tfidf"])

    err = capsys.readouterr().err.splitlines()
    assert status == 1
    assert [command[1] for command in docker.commands] == ["build"]
    assert (packaging.root / "tfidf" / "weights" / "pretrained" / "model.safetensors").is_file()
    assert err[-2] == "ERROR: failed to build: failed to solve: exit code: 1"
    assert err[-1] == (
        f"{PROG}: error: docker build of codeforces-tags/tfidf:{SHA} failed with status 1: "
        "nothing was removed, and the previous image of tfidf keeps serving"
    )


@pytest.mark.usefixtures("docker")
def test_clean_images_keeps_the_newest_image_and_removes_the_others_tag_by_tag(capsys):
    removed = serving_main.clean_images("tfidf")

    assert removed == ["codeforces-tags/tfidf:1111111"]
    assert capsys.readouterr().out.splitlines() == [
        f"kept codeforces-tags/tfidf:{SHA}, codeforces-tags/tfidf:latest (image {'b' * 12})",
        REMOVED.format("codeforces-tags/tfidf:1111111"),
        "pruned the dangling images labelled codeforces-tags.method=tfidf: "
        "Total reclaimed space: 0B",
    ]


def test_clean_images_keeps_the_image_that_keep_names_by_a_prefix_of_its_id(docker):
    status = serving_main.main(["clean-images", "--method", "tfidf", "--keep", "sha256:aaaa"])

    removed = [command[3] for command in docker.commands if command[1:3] == ["image", "rm"]]
    assert status == 0
    assert removed == [f"codeforces-tags/tfidf:{SHA}", "codeforces-tags/tfidf:latest"]


@pytest.mark.usefixtures("docker")
def test_clean_images_with_all_keeps_no_image():
    removed = serving_main.clean_images("tfidf", remove_all=True)

    assert removed == [
        "codeforces-tags/tfidf:1111111",
        f"codeforces-tags/tfidf:{SHA}",
        "codeforces-tags/tfidf:latest",
    ]


def test_clean_images_leaves_an_image_that_a_container_uses_in_place_and_names_it(docker, capsys):
    docker.in_use = ("codeforces-tags/tfidf:1111111",)

    removed = serving_main.clean_images("tfidf")

    out = capsys.readouterr().out
    assert not removed
    assert (
        "left codeforces-tags/tfidf:1111111 in place: Error response from daemon: conflict: "
        'unable to remove repository reference "codeforces-tags/tfidf:1111111"'
    ) in out
    assert docker.commands[-1][1:3] == ["image", "prune"]


def test_clean_images_never_touches_another_repository_or_an_untagged_image(docker):
    docker.listing += [
        image_row(OLD_ID, SHA, OLDER, repository="registry.example/codeforces-tags/tfidf"),
        image_row(OLD_ID, "<none>", OLDER),
    ]

    serving_main.clean_images("tfidf", remove_all=True)

    removed = [command[3] for command in docker.commands if command[1:3] == ["image", "rm"]]
    assert removed == [
        "codeforces-tags/tfidf:1111111",
        f"codeforces-tags/tfidf:{SHA}",
        "codeforces-tags/tfidf:latest",
    ]
    assert docker.commands[0][-1] == "codeforces-tags/tfidf"
    assert docker.commands[-1][-1] == "label=codeforces-tags.method=tfidf"


def test_clean_images_of_a_method_without_images_only_prunes(docker, capsys):
    docker.listing = []

    assert not serving_main.clean_images("tfidf")
    assert capsys.readouterr().out.startswith("pruned the dangling images labelled")


@pytest.mark.parametrize("keep", ["cccc", "sha256:"])
def test_clean_images_refuses_a_keep_that_names_not_one_image_of_the_repository(docker, keep):
    with pytest.raises(ValueError, match="images of codeforces-tags/tfidf, not one"):
        serving_main.clean_images("tfidf", keep=keep)

    assert [command[1:3] for command in docker.commands] == [["image", "ls"]]


def test_clean_images_refuses_keep_and_all_together(docker):
    with pytest.raises(ValueError, match="exclude each other"):
        serving_main.clean_images("tfidf", keep=OLD_ID, remove_all=True)

    assert not docker.commands


def test_the_label_that_clean_images_prunes_by_is_the_one_the_dockerfile_sets():
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")

    assert f"\nLABEL {serving_main.IMAGE_LABEL}=${{METHOD}}\n" in dockerfile


def test_a_docker_command_that_fails_is_named_on_one_line_with_dockers_last_line(
    monkeypatch, capsys
):
    def no_daemon(command):
        message = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.\n"
        raise subprocess.CalledProcessError(1, list(command), output="", stderr=message)

    monkeypatch.setattr(package, "run_command", no_daemon)

    status = serving_main.main(["clean-images", "--method", "tfidf"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{PROG}: error: `docker image ls --no-trunc --format {{{{json .}}}} "
        "codeforces-tags/tfidf` exited with status 1: Cannot connect to the Docker daemon at "
        "unix:///var/run/docker.sock.\n"
    )


def test_a_folder_that_cannot_be_written_fails_on_one_line(monkeypatch, capsys):
    def read_only(*_, **__):
        raise PermissionError(13, "Permission denied", "build/k8s/tfidf-cpu")

    monkeypatch.setattr(serving_main, "write_overlay", read_only)

    status = serving_main.main(["manifests", "--method", "tfidf", "--device", "cpu", "--tag", "v1"])

    assert status == 1
    assert capsys.readouterr().err == (
        f"{PROG}: error: [Errno 13] Permission denied: 'build/k8s/tfidf-cpu'\n"
    )


def test_a_reader_that_stops_early_ends_the_command_without_a_traceback(monkeypatch, capsys):
    # smoke ... | head -3: the next print raises BrokenPipeError, an OSError; stdout then goes to
    # /dev/null, so that Python's flush at exit does not print the error again.
    def broken(*_, **__):
        raise BrokenPipeError(32, "Broken pipe")

    redirected = []
    monkeypatch.setattr(package, "package_method", broken)
    monkeypatch.setattr(serving_main.os, "open", lambda path, flags: (path, flags))
    monkeypatch.setattr(
        serving_main.os, "dup2", lambda source, target: redirected.append((source, target))
    )
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(fileno=lambda: 99))

    status = serving_main.main(["package", "--method", "tfidf"])

    assert (status, redirected) == (1, [((os.devnull, os.O_WRONLY), 99)])
    assert capsys.readouterr().err == ""


def test_image_tags_name_the_short_sha_dirty_for_uncommitted_changes_and_latest():
    assert serving_main.image_tags("tfidf", SHA, False) == [
        f"codeforces-tags/tfidf:{SHA}",
        "codeforces-tags/tfidf:latest",
    ]
    assert serving_main.image_tags("tfidf", SHA, True)[0] == f"codeforces-tags/tfidf:{SHA}-dirty"


def test_dns_name_turns_each_underscore_into_a_hyphen_for_every_registered_method():
    names = {serving_main.dns_name(method) for method in METHODS}

    assert serving_main.dns_name("f2llm_late_fusion") == "f2llm-late-fusion"
    assert names == {method.replace("_", "-") for method in METHODS}


@pytest.mark.parametrize("method", ["2fast", "Tfidf", "tf.idf", "tfidf-", "m" * 51])
def test_dns_name_refuses_a_name_that_kubernetes_refuses(method):
    with pytest.raises(ValueError, match="gives no Kubernetes name"):
        serving_main.dns_name(method)


def test_write_overlay_prefixes_and_labels_the_methods_resources_and_points_them_at_its_image(
    tmp_path,
):
    path = serving_main.write_overlay("f2llm_late_fusion", "gpu", SHA, "ghcr.io/me/", tmp_path)

    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    resources = overlay.pop("resources")
    assert path == tmp_path / "f2llm_late_fusion-gpu" / "kustomization.yaml"
    assert overlay == {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namePrefix": "f2llm-late-fusion-",
        "labels": [
            {"pairs": {"app.kubernetes.io/instance": "f2llm-late-fusion"}, "includeSelectors": True}
        ],
        "images": [
            {
                "name": "codeforces-tags/model-server",
                "newName": "ghcr.io/me/codeforces-tags/f2llm_late_fusion",
                "newTag": SHA,
            }
        ],
    }
    assert [(path.parent / resource).resolve() for resource in resources] == [
        (PROJECT_ROOT / "deploy" / "k8s" / "overlays" / "gpu").resolve()
    ]


@pytest.mark.parametrize("tag", ["1234e56", "0128456", SHA])
def test_write_overlay_quotes_the_texts_that_kustomize_could_read_as_numbers(tmp_path, tag):
    # kustomize reads YAML as go-yaml does, to which a plain 1234e56 or 0128456 is a number (and
    # it then refuses the overlay), though PyYAML writes both plain: read the text, not the YAML.
    path = serving_main.write_overlay("tfidf", "cpu", tag, out_dir=tmp_path)

    text = path.read_text(encoding="utf-8")
    [image] = yaml.safe_load(text)["images"]
    assert re.search(rf"^  newTag: '{tag}'$", text, flags=re.MULTILINE)
    assert re.search(r"^  newName: 'codeforces-tags/tfidf'$", text, flags=re.MULTILINE)
    assert re.search(r"^    app\.kubernetes\.io/instance: 'tfidf'$", text, flags=re.MULTILINE)
    assert image["newTag"] == tag


@pytest.mark.parametrize(
    ("device", "tag", "match"),
    [("cpu", "latest", "never pulls it again"), ("cuda", SHA, "not a device")],
)
def test_write_overlay_refuses_the_tag_latest_and_a_device_without_an_overlay(
    tmp_path, device, tag, match
):
    with pytest.raises(ValueError, match=match):
        serving_main.write_overlay("tfidf", device, tag, out_dir=tmp_path)

    assert not list(tmp_path.iterdir())


def test_manifests_tags_the_image_as_build_did_by_default(packaging, monkeypatch, tmp_path, capsys):
    packaging.dirty = True
    packaging.package_method("tfidf")
    monkeypatch.setattr(serving_main, "SERVING_K8S_DIR", tmp_path / "k8s")

    status = serving_main.main(["manifests", "--method", "tfidf", "--device", "cpu"])

    path = tmp_path / "k8s" / "tfidf-cpu" / "kustomization.yaml"
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert status == 0
    assert capsys.readouterr().out == f"wrote {path}: kubectl apply -k {path.parent}\n"
    assert overlay["namePrefix"] == "tfidf-"
    assert overlay["images"] == [
        {
            "name": "codeforces-tags/model-server",
            "newName": "codeforces-tags/tfidf",
            "newTag": f"{SHA}-dirty",
        }
    ]


def test_manifests_takes_the_tag_and_the_registry_given_without_a_build(
    packaging, monkeypatch, tmp_path
):
    monkeypatch.setattr(serving_main, "SERVING_K8S_DIR", tmp_path / "k8s")

    status = serving_main.main(
        [
            "manifests",
            "--method",
            "tfidf",
            "--device",
            "gpu",
            "--tag",
            "v1",
            "--registry",
            "registry.example:5000",
        ]
    )

    path = tmp_path / "k8s" / "tfidf-gpu" / "kustomization.yaml"
    [image] = yaml.safe_load(path.read_text(encoding="utf-8"))["images"]
    assert status == 0
    assert (image["newName"], image["newTag"]) == (
        "registry.example:5000/codeforces-tags/tfidf",
        "v1",
    )
    assert not packaging.calls


def test_manifests_without_a_build_or_a_tag_says_to_build_first(packaging, capsys):
    status = serving_main.main(["manifests", "--method", "tfidf", "--device", "cpu"])

    mlmodel = packaging.root / "tfidf" / "model" / "MLmodel"
    assert status == 1
    assert capsys.readouterr().err == (
        f"{PROG}: error: {mlmodel} does not exist: build the image first ({PROG} build "
        "--method tfidf), or name its tag with --tag\n"
    )


def test_manifests_refuses_a_build_packaged_at_another_commit_than_head(
    packaging, monkeypatch, capsys
):
    packaging.package_method("tfidf")
    monkeypatch.setattr(package, "short_sha", lambda: "1111111")

    status = serving_main.main(["manifests", "--method", "tfidf", "--device", "cpu"])

    assert status == 1
    assert f"was packaged at commit {COMMIT}, not at HEAD (1111111)" in capsys.readouterr().err


def test_smoke_compares_each_answer_with_its_frozen_test_row_one_problem_per_request(
    model_server, frozen_rows, raw_problems, capsys
):
    # The server answers SCORES about every problem: problem 30's frozen row, but not 10's.
    frozen_rows[10] = frozen_row(scores_of({"trees": 0.9, "strings": 0.8}))

    status = serving_main.main(
        ["smoke", "--url", model_server.url, "--method", SERVED, "--count", "2"]
    )

    out = capsys.readouterr().out
    paths = [request.path for request in model_server.requests]
    posted = [json.loads(request.body) for request in model_server.requests[1:]]
    assert status == 1
    assert paths == ["/ping", "/invocations", "/invocations"]
    assert posted == [
        request_payload(raw_problems.loc[[30]]),
        request_payload(raw_problems.loc[[10]]),
    ]
    assert re.search(r"^problem 30: \d+\.\d{3} s, trees: agrees$", out, flags=re.MULTILINE)
    assert re.search(
        r"^problem 10: \d+\.\d{3} s, trees: differs\n    strings: score 0\.1, frozen 0\.8",
        out,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^2 test problems, one request each: median \d+\.\d{3} s, slowest \d+\.\d{3} s; 1 differ "
        r"from their frozen test rows$",
        out,
        flags=re.MULTILINE,
    )


def test_smoke_lets_a_tag_flip_where_the_frozen_score_lies_within_score_atol_of_its_threshold(
    model_server, frozen_rows, monkeypatch, capsys
):
    # 4e-05 from the threshold on each side: the scores differ by 8e-05, within score_atol.
    frozen_rows[30] = frozen_row(scores_of({"trees": 0.9, "math": THRESHOLD - 4e-5}))
    flipped = served_record(scores_of({"trees": 0.9, "math": THRESHOLD + 4e-5}))
    model_server.answer("POST", "/invocations", body={"predictions": [flipped]})
    monkeypatch.setenv("CODEFORCES_SERVING_URL", model_server.url)

    status = serving_main.main(["smoke", "--method", SERVED, "--count", "1"])

    out = capsys.readouterr().out
    assert status == 0
    assert re.search(
        r"^problem 30: \d+\.\d{3} s, math, trees: agrees, math flipped within score_atol of "
        r"its threshold$",
        out,
        flags=re.MULTILINE,
    )
    assert "; 0 differ from their frozen test rows" in out


@pytest.mark.usefixtures("frozen_rows")
def test_smoke_exits_1_on_a_score_that_moved_beyond_score_atol(model_server, capsys):
    moved = served_record(scores_of({"trees": 0.9, "math": 0.2}))
    model_server.answer("POST", "/invocations", body={"predictions": [moved]})

    status = serving_main.main(
        ["smoke", "--url", model_server.url, "--method", SERVED, "--count", "1"]
    )

    out = capsys.readouterr().out
    assert status == 1
    assert re.search(r"^problem 30: \d+\.\d{3} s, trees: differs\n    .*math", out, re.MULTILINE)
    assert "; 1 differ from their frozen test rows" in out


def test_smoke_counts_a_problem_without_a_frozen_test_row_as_a_difference(
    model_server, frozen_rows, capsys
):
    del frozen_rows[30]

    status = serving_main.main(
        ["smoke", "--url", model_server.url, "--method", SERVED, "--count", "1"]
    )

    assert status == 1
    assert "trees: differs\n    no frozen test row\n" in capsys.readouterr().out


@pytest.mark.usefixtures("frozen_rows")
def test_smoke_refuses_a_server_of_another_method(model_server, capsys):
    tfidf = served_record(SCORES, method="tfidf")
    model_server.answer("POST", "/invocations", body={"predictions": [tfidf]})

    status = serving_main.main(["smoke", "--url", model_server.url, "--method", SERVED])

    assert status == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{PROG}: error: {model_server.url} serves tfidf, not {SERVED}: pass --method tfidf, or "
        f"the URL of a server of {SERVED}"
    ]


def test_smoke_refuses_a_method_that_calls_an_api_before_it_asks_the_server(model_server, capsys):
    status = serving_main.main(
        ["smoke", "--url", model_server.url, "--method", "jev-statement-code"]
    )

    assert status == 1
    assert not model_server.requests
    assert "jev-statement-code calls an API" in capsys.readouterr().err


@pytest.mark.parametrize("server", ["refused", "loading"])
def test_smoke_fails_on_one_line_when_no_model_server_answers(
    stub_server, refused_url, server, capsys
):
    # Nothing listens at refused_url; the stub answers /ping as nginx does while the model
    # server behind it loads its tagging method.
    stub_server.answer("GET", "/ping", 502, NGINX_502, "text/html")
    url = {"refused": refused_url, "loading": stub_server.url}[server]

    status = serving_main.main(["smoke", "--url", url, "--method", SERVED])

    assert status == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{PROG}: error: no model server at {url} answers /ping with 200: start one, wait until "
        "it has loaded its tagging method (nginx answers 502 until then), or pass its URL as "
        "--url or $CODEFORCES_SERVING_URL"
    ]


def test_smoke_fails_on_one_line_when_the_test_set_cannot_be_read(
    model_server, monkeypatch, capsys
):
    def no_dataset():
        raise CliError(NO_DATASET)

    monkeypatch.setattr(serving_main, "load_test_set", no_dataset)

    status = serving_main.main(["smoke", "--url", model_server.url, "--method", SERVED])

    assert status == 1
    assert [request.path for request in model_server.requests] == ["/ping"]
    assert capsys.readouterr().err.splitlines() == [f"{PROG}: error: {NO_DATASET}"]


def test_mlflow_ui_prints_the_experiments_url_then_serves_the_store_until_the_server_exits(
    ui_server, ui_port, store, monkeypatch, capsys
):
    experiment_id = use_store(**store)
    ui = f"http://127.0.0.1:{ui_port}"
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", ui)

    status = serving_main.main(["mlflow-ui", "--port", str(ui_port)])

    [(command, env, check)] = ui_server.calls
    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == (
        f"the tag-methods experiment: {ui}/#/experiments/{experiment_id}\n"
        f"the prompt registry: {ui}/#/prompts\n"
    )
    assert not captured.err
    assert command[command.index("--backend-store-uri") + 1] == store["tracking_uri"]
    assert (command, env, check) == (mlflow_ui_command(ui_port), mlflow_ui_environment(), False)


@pytest.mark.parametrize(
    ("failure", "returncode", "status"), [(KeyboardInterrupt(), 0, 0), (None, 3, 3)]
)
def test_mlflow_ui_returns_0_after_ctrl_c_and_the_servers_own_status_otherwise(
    ui_server, ui_port, failure, returncode, status
):
    ui_server.failure, ui_server.returncode = failure, returncode

    assert serving_main.main(["mlflow-ui", "--port", str(ui_port)]) == status


@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.usefixtures("ui_server")
def test_mlflow_ui_links_the_experiments_page_of_a_store_without_the_experiment_and_says_so(
    store, ui_port, monkeypatch, capsys, deleted
):
    if deleted:
        mlflow.delete_experiment(use_store(**store))
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", "http://127.0.0.1:5000")

    serving_main.main(["mlflow-ui", "--port", str(ui_port)])

    captured = capsys.readouterr()
    assert captured.out.startswith(
        f"the tag-methods experiment: http://127.0.0.1:{ui_port}/#/experiments\n"
    )
    assert captured.err == (
        f"{PROG}: note: the store holds no tag-methods experiment: rebuild it from committed "
        "files (notebooks/README.md, Figures)\n"
        f"{PROG}: note: the Streamlit app links to http://127.0.0.1:5000: start it with "
        f"CODEFORCES_MLFLOW_UI_URL=http://127.0.0.1:{ui_port} to link here\n"
    )


@pytest.mark.parametrize(
    ("linked", "noted"),
    [
        ("http://127.0.0.1:{port}/", False),
        ("http://localhost:{port}", False),
        ("https://127.0.0.1:{port}", True),
        ("http://127.0.0.1:{port}/mlflow", True),
    ],
)
@pytest.mark.usefixtures("ui_server")
def test_mlflow_ui_notes_the_apps_link_only_when_it_names_another_server(
    ui_port, monkeypatch, capsys, linked, noted
):
    monkeypatch.setenv("CODEFORCES_MLFLOW_UI_URL", linked.format(port=ui_port))

    serving_main.main(["mlflow-ui", "--port", str(ui_port)])

    assert ("note: the Streamlit app links to" in capsys.readouterr().err) is noted


def test_mlflow_ui_refuses_a_port_in_use_before_it_starts_anything(ui_server, stub_server, capsys):
    port = urlsplit(stub_server.url).port

    status = serving_main.main(["mlflow-ui", "--port", str(port)])

    assert status == 1
    assert not ui_server.calls
    assert capsys.readouterr().err == (
        f"{PROG}: error: 127.0.0.1:{port} is in use: stop what listens there, or pass another "
        "--port\n"
    )


def test_mlflow_ui_refuses_a_store_that_does_not_exist(
    ui_server, ui_port, monkeypatch, tmp_path, capsys
):
    database = tmp_path / "mlflow.db"
    monkeypatch.setattr(serving_main, "MLFLOW_TRACKING_URI", f"sqlite:///{database}")

    status = serving_main.main(["mlflow-ui", "--port", str(ui_port)])

    assert status == 1
    assert not ui_server.calls
    assert not database.exists()
    assert capsys.readouterr().err == (
        f"{PROG}: error: {database} does not exist: rebuild the store from committed files "
        "(notebooks/README.md, Figures)\n"
    )
