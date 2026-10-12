"""Command line of the serving package: package, build, clean up, deploy, check and use a server.

    python -m codeforces_tags.serving package --method M [--out DIR] [--allow-dirty]
    python -m codeforces_tags.serving build --method M [--allow-dirty] [--keep-previous]
    python -m codeforces_tags.serving clean-images --method M [--keep IMAGE_ID | --all]
    python -m codeforces_tags.serving manifests --method M --device cpu|gpu [--tag T]
        [--registry R]
    python -m codeforces_tags.serving smoke --method M [--url U] [--count N]
    python -m codeforces_tags.serving mlflow-ui [--port 5000]
    python -m codeforces_tags.serving up --method M [--device cpu|gpu] [--local] [--tag T]
        [--build] [--port 8080] [--app-port 8501] [--mlflow-port 5000] [--no-mlflow-ui]
        [--no-browser] [--allow-dirty]

- package turns the committed folder of a frozen tagging method into a self-contained MLflow
  model and its Docker build context, build/serving/M/ (``package.package_method``): the model,
  the weights its folder pins, requirements.lock.txt and the Dockerfile. Run it with the method's
  dependency groups (uv run --group ...).
- build packages, then builds the image codeforces-tags/M:<git short sha> and
  codeforces-tags/M:latest; the sha tag ends with -dirty when a packaged input (src/,
  pyproject.toml, uv.lock, deploy/Dockerfile or the method folder) had uncommitted changes, so a
  tag never names a commit that lacks the image's content. Once the build has succeeded, and
  unless --keep-previous, it removes the method's older images (clean-images), BuildKit's cached
  build contexts (every one on the host, not only this method's: they hold the weights again,
  and another build only sends its context again) and the staged weights, which the image now
  holds. A failed build removes nothing: the previous image keeps serving.
- clean-images removes every tag of the repository codeforces-tags/M whose image is not the kept
  one (--keep, else the newest; none with --all), then the dangling images labelled
  codeforces-tags.method=M: never an image of another repository. An image that a container
  still uses is left in place and named. It prints each tag it removed with the size of its
  image, of which docker frees only the layers that no other image shares, and the space that
  the prune reclaimed.
- manifests writes build/k8s/M-<device>/kustomization.yaml, an overlay over
  deploy/k8s/overlays/<device> that prefixes the method's resource names, labels their pods and
  selectors, and points them at the image codeforces-tags/M:<tag> ([--registry R/] before it);
  the tag is by default the one build gave, never latest.
- smoke asks a running model server (--url, else $CODEFORCES_SERVING_URL, else
  http://127.0.0.1:8080) about the first test problems, one request each, and compares each answer
  with the method's frozen test row: scores within max(score_atol, TUNING_EXPORT_TOLERANCE), a
  focus tag flipped only where the frozen score lies within score_atol of its per-tag threshold.
  It prints each problem's line as its answer comes (its latency, its focus tags, whether it
  agrees), then the median latency, and exits with 1 on any other difference.
  It needs the dataset ($CODEFORCES_DATA_DIR), and refuses a method that calls an API before it
  sends anything: a new answer is a new sample (docs/adr/0003), and every request is paid.
- mlflow-ui serves the project's MLflow store on 127.0.0.1 until it exits or Ctrl-C: the UI that
  the Streamlit app's links open (the tag-methods experiment, each method's run, the prompt
  registry). It refuses a port in use, then prints the URLs of the experiment and of the prompt
  registry, and notes when the app's links ($CODEFORCES_MLFLOW_UI_URL) name another server.
- up starts, in one terminal, everything needed to interact with a frozen method, and stops all
  of it on Ctrl-C (``stack.up``): its model server (docker compose on the image
  codeforces-tags/M:<tag>, latest by default, which --build builds first as build does; or, with
  --local, mlflow models serve on build/serving/M/, packaged first when it has none), the MLflow
  UI (unless --no-mlflow-ui) and the Streamlit app, each on 127.0.0.1, which opens in a browser
  unless --no-browser. It checks everything before it starts anything: the method, the ports,
  Streamlit (uv run --group app ...), the store, the image and, for --device gpu, Docker's nvidia
  runtime. --allow-dirty lets it package or build from uncommitted inputs, as package and build
  do. A part that dies stops the others, prints the end of its log and exits with 1; the logs are
  in build/serving/M/up/.

A failure prints one line on stderr and exits with status 1 (a failed build first prints the end
of docker's log there); a usage error, a --port outside 1-65535 among them, exits with status 2.
A reader of stdout that stops early (smoke ... | head -3) ends the command with status 1 and
nothing on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import mlflow
import yaml

from codeforces_tags.cli_methods import CliError
from codeforces_tags.config import (
    FOCUS_TAGS,
    MLFLOW_METHODS_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    SERVING_APP_PORT,
    SERVING_BUILD_DIR,
    SERVING_DEFAULT_URL,
    SERVING_IMAGE_PREFIX,
    SERVING_K8S_DIR,
    SERVING_MLFLOW_UI_URL,
    SERVING_MLFLOW_UI_URL_ENV_VAR,
    SERVING_PORT,
    SERVING_SMOKE_COUNT,
    SERVING_URL_ENV_VAR,
)
from codeforces_tags.methods import METHODS
from codeforces_tags.serving import package, stack
from codeforces_tags.serving.client import (
    ServedPrediction,
    ServingError,
    ping,
    predict,
    server_url,
)
from codeforces_tags.serving.mlflow_links import (
    mlflow_ui_command,
    mlflow_ui_environment,
    mlflow_ui_url,
)
from codeforces_tags.serving.problems import (
    FrozenRow,
    frozen_prediction,
    live_differences,
    load_test_set,
)
from codeforces_tags.tracking.environment import mlflow_env

PROG = "python -m codeforces_tags.serving"
# The label of every image of a method (deploy/Dockerfile: LABEL codeforces-tags.method=${METHOD}),
# by which clean-images prunes the method's dangling images, and nothing else on the host.
IMAGE_LABEL = f"{SERVING_IMAGE_PREFIX}.method"
# The Kubernetes overlays a method's overlay builds on, one per device (deploy/k8s/overlays/), the
# image placeholder of deploy/k8s/base/ that it replaces, and the name of the base's Deployment
# and Service, which its name prefix extends.
OVERLAYS_DIR = PROJECT_ROOT / "deploy" / "k8s" / "overlays"
OVERLAY_DEVICES = ("cpu", "gpu")
BASE_IMAGE = f"{SERVING_IMAGE_PREFIX}/model-server"
BASE_NAME = "model-server"
# A Service's name must be a DNS-1035 label: lower-case letters, digits and "-", starting with a
# letter and ending with a letter or a digit, at most 63 characters.
_DNS_LABEL = re.compile(r"[a-z]([-a-z0-9]*[a-z0-9])?")
_DNS_LABEL_MAX = 63
# The lines of a failed docker build's log that build prints before its error.
_LOG_TAIL = 20
# The highest TCP port.
_MAX_PORT = 65535
# The hosts by which a browser of this machine reaches a server that listens on 127.0.0.1.
_LOCAL_HOSTS = ("127.0.0.1", "localhost")
# The failures that a command reports on one line of stderr, with status 1. OSError covers no
# docker on the PATH and a folder that cannot be written; main catches BrokenPipeError, an
# OSError too, before them.
_FAILURES = (
    package.PackagingError,
    stack.StackError,
    ServingError,
    CliError,
    ValueError,
    OSError,
    subprocess.CalledProcessError,
)
# What --help says of each command.
_COMMANDS = {
    "package": "package a frozen method as an MLflow model and a Docker build context",
    "build": "package, build the method's image, then remove its older images",
    "clean-images": "remove the method's older images, never another repository's",
    "manifests": "write the Kubernetes overlay of the method's image",
    "smoke": "compare a model server's answers with the method's frozen test rows",
    "mlflow-ui": "serve the project's MLflow store on 127.0.0.1, which the app's links open",
    "up": "start the method's model server, the MLflow UI and the Streamlit app until Ctrl-C",
}


class _Quoted(str):
    """A text that an overlay writes between single quotes, whatever it looks like.

    kustomize reads YAML as go-yaml does, to which a plain ``1234e56`` or ``0128456`` (a git
    short sha may be either) is a number, and it refuses a number where a text is due; PyYAML
    writes both plain, since to PyYAML they are texts.
    """


class _OverlayDumper(yaml.SafeDumper):
    """PyYAML's safe dumper, which writes a ``_Quoted`` text between single quotes."""


def _represent_quoted(dumper: yaml.SafeDumper, text: _Quoted) -> yaml.ScalarNode:
    """Represent a ``_Quoted`` text as a YAML string between single quotes."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(text), style="'")


_OverlayDumper.add_representer(_Quoted, _represent_quoted)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser of the commands, one subcommand each.

    Returns:
        The parser: ``package``, ``build``, ``clean-images``, ``manifests``, ``smoke``,
        ``mlflow-ui`` and ``up``, with the options the module's docstring lists.
    """
    parser = argparse.ArgumentParser(
        prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    added = {name: commands.add_parser(name, help=text) for name, text in _COMMANDS.items()}
    for name in ("package", "build", "clean-images", "manifests", "smoke", "up"):
        added[name].add_argument(
            "--method",
            required=True,
            choices=sorted(METHODS),
            metavar="M",
            help="the tagging method's registry key",
        )
    added["package"].add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help=f"the build context, under {SERVING_BUILD_DIR} (default: {SERVING_BUILD_DIR}/M)",
    )
    for name in ("package", "build", "up"):
        added[name].add_argument(
            "--allow-dirty",
            action="store_true",
            help="package uncommitted code, Dockerfile or method folder",
        )
    added["build"].add_argument(
        "--keep-previous",
        action="store_true",
        help="keep the method's older images, BuildKit's cached contexts and the staged weights",
    )
    kept = added["clean-images"].add_mutually_exclusive_group()
    kept.add_argument("--keep", metavar="IMAGE_ID", help="the image to keep (default: the newest)")
    kept.add_argument("--all", action="store_true", help="keep no image of the method")
    _add_manifests_options(added["manifests"])
    _add_smoke_options(added["smoke"])
    added["mlflow-ui"].add_argument(
        "--port",
        type=_port,
        default=urlsplit(SERVING_MLFLOW_UI_URL).port,
        help="the port of 127.0.0.1 to serve the UI on (default: %(default)s)",
    )
    _add_up_options(added["up"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The command's exit status: 0; 1 when smoke finds a difference; for mlflow-ui, the
        server's own, and 0 after Ctrl-C; for up, 0 after Ctrl-C. A failure prints one line on
        stderr and returns 1 (a failed docker build first prints the end of its log there, and
        a part of up that died the end of its own): a ``PackagingError``, a ``StackError`` (a
        check of up, or a part that did not start or died), a ``ServingError`` (its message), a
        ``CliError`` (e.g. no dataset), a ``ValueError``, an ``OSError`` (no ``docker`` on the
        PATH, a folder that cannot be written) or a command that failed
        (``subprocess.CalledProcessError``: the command and the last line of its error output).
        Once the reader of stdout has gone (``smoke ... | head -3``), it returns 1 with nothing
        on stderr. A usage error exits with status 2, from argparse.
    """
    args = build_parser().parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "package": _package,
        "build": _build,
        "clean-images": _clean_images,
        "manifests": _manifests,
        "smoke": _smoke,
        "mlflow-ui": _mlflow_ui,
        "up": _up,
    }
    try:
        status = commands[args.command](args)
        sys.stdout.flush()  # a reader that has gone fails the flush here, rather than at exit
        return status
    except BrokenPipeError:
        # Python's recipe: what is left of stdout goes to /dev/null, so that the flush at exit
        # does not fail again with a traceback. It comes before _FAILURES, whose OSError it is.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except _FAILURES as error:
        print(f"{PROG}: error: {_message(error)}", file=sys.stderr)
        return 1


def dns_name(method: str) -> str:
    """Name a method's Kubernetes resources: its name with each ``_`` turned into ``-``.

    Args:
        method: The method's registry key, e.g. ``f2llm_late_fusion``.

    Returns:
        The prefix of its resources' names and the value of its instance label, e.g.
        ``f2llm-late-fusion``: its Service is ``f2llm-late-fusion-model-server``.

    Raises:
        ValueError: If that Service name would not be a DNS-1035 label (lower-case letters,
            digits and ``-``, starting with a letter, at most 63 characters).
    """
    name = method.replace("_", "-")
    if not _DNS_LABEL.fullmatch(name) or len(f"{name}-{BASE_NAME}") > _DNS_LABEL_MAX:
        raise ValueError(
            f"{method!r} gives no Kubernetes name: {name}-{BASE_NAME} must be lower-case letters, "
            f"digits and '-', start with a letter and hold at most {_DNS_LABEL_MAX} characters"
        )
    return name


def image_tags(name: str, sha: str, dirty: bool) -> list[str]:
    """Name the two tags of a method's image.

    Args:
        name: The method's registry key.
        sha: The git short sha of the commit it was packaged at.
        dirty: Whether a packaged input (``src/``, ``pyproject.toml``, ``uv.lock``,
            ``deploy/Dockerfile`` or the method folder) had uncommitted changes: the packaged
            model's ``code_dirty``.

    Returns:
        ``codeforces-tags/<name>:<sha>`` (``<sha>-dirty`` when ``dirty``), then
        ``codeforces-tags/<name>:latest``.
    """
    repository = f"{SERVING_IMAGE_PREFIX}/{name}"
    return [f"{repository}:{_sha_tag(sha, dirty)}", f"{repository}:latest"]


def write_overlay(
    method: str,
    device: str,
    tag: str,
    registry: str | None = None,
    out_dir: Path = SERVING_K8S_DIR,
) -> Path:
    """Write the Kubernetes overlay of a method's image, ``<out_dir>/<method>-<device>/``.

    The overlay holds inline fields only, since kustomize refuses a patch file outside an
    overlay's folder: it builds on ``deploy/k8s/overlays/<device>``, prefixes every name with
    ``dns_name(method)``, labels every pod and selector with ``app.kubernetes.io/instance``, so
    that each method's Service selects only its own pods, and points the base's image
    placeholder at the method's image. It is written afresh each time. Its instance label, the
    image's name and its tag are written between single quotes: kustomize reads a plain
    ``1234e56`` or ``0128456``, which a git short sha may be, as a number, and refuses it.

    Args:
        method: The method's registry key.
        device: ``cpu`` or ``gpu``, the overlay it builds on.
        tag: The image's tag; never ``latest``, which a node that holds it never pulls again
            (``imagePullPolicy: IfNotPresent``).
        registry: The registry that the cluster pulls from, e.g. ``ghcr.io/me``; None for an
            image loaded into the nodes (``kind load docker-image``).
        out_dir: The folder of the overlays.

    Returns:
        The overlay's ``kustomization.yaml``.

    Raises:
        ValueError: If ``device`` is not one of ``OVERLAY_DEVICES``, if ``tag`` is ``latest``,
            or as ``dns_name``.
    """
    if device not in OVERLAY_DEVICES:
        raise ValueError(f"{device!r} is not a device of deploy/k8s/overlays/: {OVERLAY_DEVICES}")
    if tag == "latest":
        raise ValueError(
            "a node that holds the tag latest never pulls it again (imagePullPolicy: "
            "IfNotPresent): name the image by the sha tag that build printed"
        )
    name = dns_name(method)
    folder = out_dir / f"{method}-{device}"
    folder.mkdir(parents=True, exist_ok=True)
    image = f"{SERVING_IMAGE_PREFIX}/{method}"
    overlay = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": [Path(os.path.relpath(OVERLAYS_DIR / device, folder)).as_posix()],
        "namePrefix": f"{name}-",
        "labels": [
            {"pairs": {"app.kubernetes.io/instance": _Quoted(name)}, "includeSelectors": True}
        ],
        "images": [
            {
                "name": BASE_IMAGE,
                "newName": _Quoted(f"{registry.rstrip('/')}/{image}" if registry else image),
                "newTag": _Quoted(tag),
            }
        ],
    }
    path = folder / "kustomization.yaml"
    header = f"# Written by `{PROG} manifests`: write it again rather than edit it.\n"
    written = yaml.dump(overlay, Dumper=_OverlayDumper, sort_keys=False)
    path.write_text(header + written, encoding="utf-8")
    return path


def clean_images(method: str, *, keep: str | None = None, remove_all: bool = False) -> list[str]:
    """Remove a method's older images, then its dangling ones, never another repository's.

    Every tag of the repository ``codeforces-tags/<method>`` whose image is not the kept one is
    removed by its reference (``docker image rm``), so a tag of another repository on the same
    image stays; then the dangling images labelled ``codeforces-tags.method=<method>`` are
    pruned. Docker refuses to remove the last tag of an image that a container uses: that tag is
    left in place and named. Every docker command runs through ``package.run_command``. It
    prints what it kept, each tag it removed with the size of its image (docker frees only the
    layers that no other image shares, and says nothing of the space a removal frees), and the
    space that the prune reclaimed.

    Args:
        method: The method's registry key.
        keep: The id of the image to keep, or a prefix of it (with or without ``sha256:``);
            None keeps the newest image of the repository.
        remove_all: Keep no image of the repository.

    Returns:
        The references removed, in the order ``docker image ls`` lists them.

    Raises:
        ValueError: If ``keep`` and ``remove_all`` are both given, or if ``keep`` names no image
            of the repository, or more than one.
    """
    if keep is not None and remove_all:
        raise ValueError("--keep and --all exclude each other: keep one image, or none")
    repository = f"{SERVING_IMAGE_PREFIX}/{method}"
    images = _listed_images(repository)
    kept = None if remove_all else _kept_image(images, keep, repository)
    removed = _remove_tags(repository, images, kept)
    output = package.run_command(
        ["docker", "image", "prune", "--force", "--filter", f"label={IMAGE_LABEL}={method}"]
    )
    print(f"pruned the dangling images labelled {IMAGE_LABEL}={method}: {_last_line(output)}")
    return removed


def _package(args: argparse.Namespace) -> int:
    """Write the method's build context from its committed folder."""
    folder = package.package_method(args.method, args.out, allow_dirty=args.allow_dirty)
    print(f"packaged {args.method} in {folder}")
    return 0


def _build(args: argparse.Namespace) -> int:
    """Package, build the image, then remove what it makes redundant, unless --keep-previous."""
    folder = package.package_method(args.method, allow_dirty=args.allow_dirty)
    print(f"packaged {args.method} in {folder}", flush=True)  # the build takes minutes
    references = image_tags(args.method, *_packaged_version(folder))
    _docker_build(args.method, references, folder)
    image_id, size = _built_image(references[0])
    print(
        f"built {references[0]} and {references[1]}: image {_short_id(image_id)}, "
        f"{size / 1e9:.2f} GB"
    )
    if not args.keep_previous:
        clean_images(args.method, keep=image_id)
        output = package.run_command(
            ["docker", "builder", "prune", "--force", "--filter", "type=source.local"]
        )
        print(f"dropped BuildKit's cached build contexts: {_last_line(output)}")
        shutil.rmtree(folder / "weights")
        print(f"deleted the staged weights, which the image holds: {folder / 'weights'}")
    return 0


def _clean_images(args: argparse.Namespace) -> int:
    """Remove the method's older images and its dangling ones."""
    clean_images(args.method, keep=args.keep, remove_all=args.all)
    return 0


def _manifests(args: argparse.Namespace) -> int:
    """Write the method's overlay, tagged by --tag or as build tagged its image."""
    tag = args.tag or _sha_tag(*_packaged_version(SERVING_BUILD_DIR / args.method))
    path = write_overlay(args.method, args.device, tag, args.registry, out_dir=SERVING_K8S_DIR)
    print(f"wrote {path}: kubectl apply -k {path.parent}")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """Compare a model server's answers on the first test problems with the frozen test rows."""
    spec = METHODS[args.method]
    if spec.calls_api:
        raise ValueError(
            f"{args.method} calls an API: asking it again gives a new sample (docs/adr/0003), so "
            "there is no frozen row to compare with, and every request is paid; smoke checks "
            "local methods only"
        )
    url = server_url(args.url)
    if not ping(url):
        raise ServingError(
            None,
            f"no model server at {url} answers /ping with 200: start one, wait until it has "
            "loaded its tagging method (nginx answers 502 until then), or pass its URL as --url "
            f"or ${SERVING_URL_ENV_VAR}",
        )
    problems, _ = load_test_set()
    seconds, differing = [], 0
    for sample_id in problems.index[: args.count]:
        [prediction] = predict(url, problems.loc[[sample_id]])
        if prediction.method != args.method:
            raise ValueError(
                f"{url} serves {prediction.method}, not {args.method}: pass --method "
                f"{prediction.method}, or the URL of a server of {args.method}"
            )
        frozen = frozen_prediction(args.method, sample_id)
        differences = (
            ["no frozen test row"]
            if frozen is None
            else live_differences(prediction, frozen, score_atol=spec.score_atol)
        )
        print(_smoke_line(sample_id, prediction, frozen, differences), flush=True)
        seconds.append(prediction.seconds)
        differing += bool(differences)
    print(
        f"{len(seconds)} test problems, one request each: median {statistics.median(seconds):.3f}"
        f" s, slowest {max(seconds):.3f} s; {differing} differ from their frozen test rows"
    )
    return 1 if differing else 0


def _mlflow_ui(args: argparse.Namespace) -> int:
    """Serve the project's MLflow store on 127.0.0.1 until the server exits or Ctrl-C."""
    if stack.port_in_use(args.port):
        raise ValueError(
            f"127.0.0.1:{args.port} is in use: stop what listens there, or pass another --port"
        )
    ui = f"http://127.0.0.1:{args.port}"
    print(f"the {MLFLOW_METHODS_EXPERIMENT} experiment: {_experiment_url(ui)}")
    print(f"the prompt registry: {ui}/#/prompts", flush=True)
    linked = mlflow_ui_url()
    if not _links_here(linked, args.port):
        _note(
            f"the Streamlit app links to {linked}: start it with "
            f"{SERVING_MLFLOW_UI_URL_ENV_VAR}={ui} to link here"
        )
    try:
        server = subprocess.run(
            mlflow_ui_command(args.port), env=mlflow_ui_environment(), check=False
        )
    except KeyboardInterrupt:
        return 0
    return server.returncode


def _up(args: argparse.Namespace) -> int:
    """Start the method's model server, the MLflow UI and the app, and stop them on Ctrl-C."""
    options = stack.UpOptions(
        method=args.method,
        device=args.device,
        local=args.local,
        tag=args.tag,
        build=args.build,
        port=args.port,
        app_port=args.app_port,
        mlflow_port=None if args.no_mlflow_ui else args.mlflow_port,
        browser=not args.no_browser,
        allow_dirty=args.allow_dirty,
    )
    hooks = stack.Hooks(package.run_command, package.package_method, _build_image)
    return stack.up(options, hooks)


def _build_image(method: str, *, allow_dirty: bool) -> int:
    """Build a method's image as build does: packaged first, its older images removed after."""
    return _build(argparse.Namespace(method=method, allow_dirty=allow_dirty, keep_previous=False))


def _add_up_options(command: argparse.ArgumentParser) -> None:
    """Add the options of up: the device, the runtime, the image and the three ports."""
    command.add_argument(
        "--device",
        choices=stack.DEVICES,
        default="cpu",
        help="the model server's device (default: %(default)s)",
    )
    command.add_argument(
        "--local",
        action="store_true",
        help="serve build/serving/M/ with mlflow models serve, without Docker",
    )
    command.add_argument("--tag", metavar="T", help="the image's tag (default: latest)")
    command.add_argument(
        "--build", action="store_true", help="build the image first, as build does"
    )
    ports = (
        ("--port", SERVING_PORT, "the model server"),
        ("--app-port", SERVING_APP_PORT, "the Streamlit app"),
        ("--mlflow-port", stack.DEFAULT_UI_PORT, "the MLflow UI"),
    )
    for option, default, part in ports:
        command.add_argument(
            option,
            type=_port,
            default=default,
            metavar="P",
            help=f"the port of 127.0.0.1 of {part} (default: %(default)s)",
        )
    command.add_argument(
        "--no-mlflow-ui", action="store_true", help="start no MLflow UI: the app's links then fail"
    )
    command.add_argument(
        "--no-browser", action="store_true", help="run the app headless, opening no browser"
    )


def _add_manifests_options(command: argparse.ArgumentParser) -> None:
    """Add the options of manifests: the device, the image's tag and the registry."""
    command.add_argument(
        "--device", required=True, choices=OVERLAY_DEVICES, help="the overlay to build on"
    )
    command.add_argument(
        "--tag",
        metavar="T",
        help="the image's tag (default: the one build gave it, the git short sha)",
    )
    command.add_argument(
        "--registry",
        metavar="R",
        help="the registry the cluster pulls from (default: none, for an image loaded into kind)",
    )


def _add_smoke_options(command: argparse.ArgumentParser) -> None:
    """Add the options of smoke: the server's URL and how many test problems to ask about."""
    command.add_argument(
        "--url",
        metavar="U",
        help=f"the model server (default: ${SERVING_URL_ENV_VAR}, else {SERVING_DEFAULT_URL})",
    )
    command.add_argument(
        "--count",
        type=_positive,
        default=SERVING_SMOKE_COUNT,
        metavar="N",
        help="how many of the first test problems to ask about (default: %(default)s)",
    )


def _sha_tag(sha: str, dirty: bool) -> str:
    """Return the sha tag of an image: the git short sha, with ``-dirty`` for uncommitted inputs."""
    return f"{sha}-dirty" if dirty else sha


def _packaged_version(folder: Path) -> tuple[str, bool]:
    """Read the commit a build context was packaged at, checked on HEAD, and its ``code_dirty``.

    Raises:
        ValueError: If the folder holds no packaged model, or one packaged at another commit
            than HEAD: its image's tag would not be HEAD's.
    """
    mlmodel = folder / "model" / "MLmodel"
    if not mlmodel.is_file():
        raise ValueError(
            f"{mlmodel} does not exist: build the image first ({PROG} build --method "
            f"{folder.name}), or name its tag with --tag"
        )
    metadata = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))["metadata"]
    sha = package.short_sha()
    if not str(metadata["git_commit"]).startswith(sha):
        raise ValueError(
            f"{folder} was packaged at commit {metadata['git_commit']}, not at HEAD ({sha}): "
            "build the image again, or name its tag with --tag"
        )
    return sha, bool(metadata["code_dirty"])


def _docker_build(name: str, references: list[str], folder: Path) -> None:
    """Build the image of a build context; on a failure, print the end of docker's log."""
    command = ["docker", "build", "--platform", "linux/amd64", "--build-arg", f"METHOD={name}"]
    for reference in references:
        command += ["-t", reference]
    _note(f"building {references[0]}: its log is printed only if the build fails")
    try:
        package.run_command([*command, str(folder)])
    except subprocess.CalledProcessError as error:
        print("\n".join((error.stderr or "").splitlines()[-_LOG_TAIL:]), file=sys.stderr)
        raise package.PackagingError(
            f"docker build of {references[0]} failed with status {error.returncode}: nothing was "
            f"removed, and the previous image of {name} keeps serving"
        ) from error


def _built_image(reference: str) -> tuple[str, int]:
    """Return the id and the size in bytes of the image that a reference names."""
    output = package.run_command(
        ["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", reference]
    )
    image_id, size = output.split()
    return image_id, int(size)


def _listed_images(repository: str) -> list[dict[str, str]]:
    """List the tagged images of one repository, one row per tag, as docker lists them.

    A row of another repository, or without a tag, is never kept, whatever docker printed.
    """
    output = package.run_command(
        ["docker", "image", "ls", "--no-trunc", "--format", "{{json .}}", repository]
    )
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    return [row for row in rows if row["Repository"] == repository and row["Tag"] != "<none>"]


def _kept_image(images: list[dict[str, str]], keep: str | None, repository: str) -> str | None:
    """Return the id of the image to keep: the one ``keep`` names, else the newest, if any."""
    if keep is None:
        return max(images, key=_created)["ID"] if images else None
    wanted = keep.removeprefix("sha256:")
    named = {row["ID"] for row in images if row["ID"].removeprefix("sha256:").startswith(wanted)}
    if len(named) != 1:
        raise ValueError(
            f"--keep {keep!r} names {len(named)} images of {repository}, not one: give an id "
            f"that `docker image ls {repository}` lists"
        )
    return named.pop()


def _remove_tags(repository: str, images: list[dict[str, str]], kept: str | None) -> list[str]:
    """Remove each tag of the repository whose image is not the kept one; say what it kept."""
    if kept is not None:
        tags = ", ".join(f"{repository}:{row['Tag']}" for row in images if row["ID"] == kept)
        print(f"kept {tags} (image {_short_id(kept)})")
    removed = []
    for row in images:
        reference = f"{repository}:{row['Tag']}"
        if row["ID"] != kept and _remove(reference, row["Size"]):
            removed.append(reference)
    return removed


def _created(row: dict[str, str]) -> datetime:
    """Read when an image was created, from docker's ``2026-10-11 11:52:03 +0000 UTC``."""
    return datetime.strptime(" ".join(row["CreatedAt"].split()[:3]), "%Y-%m-%d %H:%M:%S %z")


def _remove(reference: str, size: str) -> bool:
    """Remove one tag; leave it in place, and name it, when docker refuses to remove it."""
    try:
        package.run_command(["docker", "image", "rm", reference])
    except subprocess.CalledProcessError as error:
        print(f"left {reference} in place: {_last_line(error.stderr)}")
        return False
    print(f"removed {reference} (an image of {size}; layers it shares with other images stay)")
    return True


def _smoke_line(
    sample_id: int,
    prediction: ServedPrediction,
    frozen: FrozenRow | None,
    differences: list[str],
) -> str:
    """Say how one test problem's answer compares with its frozen row, with its latency."""
    tags = ", ".join(prediction.tags) or "no tag"
    head = f"problem {sample_id}: {prediction.seconds:.3f} s, {tags}"
    if differences:
        return "\n".join([f"{head}: differs", *(f"    {line}" for line in differences)])
    flipped = [tag for tag in FOCUS_TAGS if (tag in prediction.tags) != bool(frozen.tags[tag])]
    if flipped:
        return f"{head}: agrees, {', '.join(flipped)} flipped within score_atol of its threshold"
    return f"{head}: agrees"


def _experiment_url(ui: str) -> str:
    """Return the UI's URL of the tag-methods experiment, read from the store before it is served.

    Raises:
        ValueError: If the store's sqlite file does not exist: the server would create it empty.
    """
    database = Path(MLFLOW_TRACKING_URI.removeprefix("sqlite:///"))
    if not database.is_file():
        raise ValueError(
            f"{database} does not exist: rebuild the store from committed files "
            "(notebooks/README.md, Figures)"
        )
    mlflow_env()
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_METHODS_EXPERIMENT)
    if experiment is None or experiment.lifecycle_stage != "active":
        _note(
            f"the store holds no {MLFLOW_METHODS_EXPERIMENT} experiment: rebuild it from "
            "committed files (notebooks/README.md, Figures)"
        )
        return f"{ui}/#/experiments"
    return f"{ui}/#/experiments/{experiment.experiment_id}"


def _short_id(image_id: str) -> str:
    """Shorten an image id as docker shows it: its first 12 hex digits."""
    return image_id.removeprefix("sha256:")[:12]


def _links_here(url: str, port: int) -> bool:
    """Tell whether the app's MLflow UI URL names mlflow-ui's server: http, this machine, the port.

    A trailing ``/`` and the name ``localhost`` name it too, as they do in a browser.
    """
    parts = urlsplit(url.rstrip("/"))
    here = {f"{host}:{port}" for host in _LOCAL_HOSTS}
    return parts.scheme == "http" and parts.netloc.lower() in here and not parts.path


def _positive(text: str) -> int:
    """Read a count of at least 1, for argparse."""
    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a count of at least 1")
    return count


def _port(text: str) -> int:
    """Read a port of 127.0.0.1 to serve on, for argparse: 0 would let the system choose one."""
    port = int(text)
    if not 1 <= port <= _MAX_PORT:
        raise argparse.ArgumentTypeError(f"{text} is not a port from 1 to {_MAX_PORT}")
    return port


def _message(error: Exception) -> str:
    """Say in one line what failed: a model server's message, a command's, or the error's own."""
    if isinstance(error, ServingError):
        return error.message
    if isinstance(error, subprocess.CalledProcessError):
        command = " ".join(map(str, error.cmd))
        return f"`{command}` exited with status {error.returncode}: {_last_line(error.stderr)}"
    return str(error)


def _last_line(text: str | None) -> str:
    """Return the last line of a command's output, e.g. docker's own error or its total."""
    return (text or "").strip().rpartition("\n")[2]


def _note(message: str) -> None:
    """Print a note on stderr, which keeps stdout for the command's results."""
    print(f"{PROG}: note: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
