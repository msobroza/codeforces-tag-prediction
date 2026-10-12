"""Tests for the deployment files: the image, compose, Kubernetes and the SageMaker scripts.

They parse ``deploy/`` and check that its files agree with each other and with the project: the
served port, the ``/ping`` probes, each compose profile's and Kubernetes overlay's device, the
image's environment (MLflow's safeguards of ``config.MLFLOW_ENV``) and MLflow's own container
entrypoint. The SageMaker scripts run against fake ``aws``, ``docker`` and ``uv`` commands, so
nothing leaves the machine and nothing is pushed.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
import yaml
from click.testing import CliRunner
from mlflow.models.cli import commands as mlflow_models

from codeforces_tags.config import (
    API_KEY_ENV_VARS,
    DEVICE_ENV_VAR,
    MLFLOW_ENV,
    MLFLOW_ENV_DEFAULTS,
    PRETRAINED_DIR_ENV_VAR,
    PROJECT_ROOT,
    SERVING_IMAGE_PREFIX,
    SERVING_PORT,
    TRAINED_DIR_ENV_VAR,
)
from codeforces_tags.serving.__main__ import write_overlay
from codeforces_tags.tracking.environment import mlflow_env

DEPLOY = PROJECT_ROOT / "deploy"
DOCKERFILE = DEPLOY / "Dockerfile"
COMPOSE = DEPLOY / "docker-compose.yml"
ENV_EXAMPLE = DEPLOY / ".env.serving.example"
K8S = DEPLOY / "k8s"
SAGEMAKER = DEPLOY / "sagemaker"
SCRIPTS = ["deploy.sh", "invoke.sh", "delete.sh"]
# The variables the image sets beyond MLflow's safeguards (MLFLOW_ENV_DEFAULTS | MLFLOW_ENV): no
# bytecode or pip cache, MLflow's container settings (one worker, no environment built at start),
# no download, and where the baked weights are.
IMAGE_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PIP_NO_CACHE_DIR": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "MLFLOW_MODELS_WORKERS": "1",
    "MLFLOW_DISABLE_ENV_CREATION": "True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    PRETRAINED_DIR_ENV_VAR: "/opt/ml/weights/pretrained",
    TRAINED_DIR_ENV_VAR: "/opt/ml/weights/trained",
    DEVICE_ENV_VAR: "auto",
}
# The image of the Kubernetes base, a placeholder that each method's overlay replaces.
PLACEHOLDER = f"{SERVING_IMAGE_PREFIX}/model-server"
# The smallest models-from-code entry file: a model for `generate-dockerfile` to write a
# Dockerfile for.
ECHO_MODEL = """\
import mlflow


class Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        return model_input


mlflow.models.set_model(Echo())
"""
# The fake aws, docker and uv: each call appends [command, *arguments] to $FAKE_LOG. uv runs its
# Python program with this interpreter, so the registry and the config answer for real, and only
# from $FAKE_ROOT, the repository's root, where the real uv finds the project. aws fails unless
# AWS_PAGER is empty (a pager can wait for q). aws and docker fail (status 254) on an argument
# equal to $FAKE_FAIL; otherwise they print what the scripts read or send to stderr (a CLI that
# knows InferenceAmiVersion), and write $FAKE_ANSWER as the endpoint's answer. aws prints
# $FAKE_PASSWORD as the ECR password, and docker login writes what it reads on stdin to
# $FAKE_LOGIN: where the password went.
FAKE_COMMAND = """\
import json
import os
import sys

command, arguments = sys.argv[1], sys.argv[2:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps([command, *arguments]) + "\\n")
if command == "uv":
    if os.path.realpath(os.getcwd()) != os.path.realpath(os.environ["FAKE_ROOT"]):
        sys.exit("uv: not run from the repository root")
    program = arguments[arguments.index("python") + 1 :]
    os.execv(sys.executable, [sys.executable, *program])
if command == "aws" and os.environ.get("AWS_PAGER") != "":
    sys.exit("aws: AWS_PAGER is not empty, so a pager could wait for q")
if os.environ.get("FAKE_FAIL") in arguments:
    sys.exit(254)
if "--generate-cli-skeleton" in arguments:
    print('{"ProductionVariants": [{"InferenceAmiVersion": ""}]}')
if arguments[:2] == ["sts", "get-caller-identity"]:
    print("123456789012")
if arguments[:2] == ["ecr", "get-login-password"]:
    print(os.environ["FAKE_PASSWORD"])
if command == "docker" and arguments[:1] == ["login"]:
    with open(os.environ["FAKE_LOGIN"], "w", encoding="utf-8") as stdin_record:
        stdin_record.write(sys.stdin.read())
if arguments[:2] == ["sagemaker-runtime", "invoke-endpoint"]:
    print('{"ContentType": "application/json", "InvokedProductionVariant": "AllTraffic"}')
    with open(arguments[-1], "w", encoding="utf-8") as answer:
        answer.write(os.environ["FAKE_ANSWER"])
"""
ANSWER = '{"predictions": [{"method": "tfidf", "math": 1}]}'
# The fake ECR password, and the file in which the fake docker keeps what `docker login` read.
PASSWORD = "ecr-password-0123456789"
LOGIN_STDIN = "docker_login_stdin.txt"
# The inputs of the SageMaker scripts: a test's shell may hold some (compose reads METHOD and TAG
# too), so the fake runs start without them.
SCRIPT_INPUTS = ("METHOD", "TAG", "DEVICE", "AWS_REGION", "ROLE_ARN", "INSTANCE_TYPE", "EXTRA_ENV")
DEPLOY_INPUTS = {
    "METHOD": "tfidf",
    "DEVICE": "cpu",
    "AWS_REGION": "eu-west-1",
    "ROLE_ARN": "arn:aws:iam::123456789012:role/sagemaker-execution",
}
REGISTRY = "123456789012.dkr.ecr.eu-west-1.amazonaws.com"
# The aws call that prints the request the CLI knows, offline: it reaches no AWS service.
SKELETON = ["aws", "sagemaker", "create-endpoint-config", "--generate-cli-skeleton"]


def load_yaml(path: Path) -> dict:
    """Return a YAML file of the repository, parsed.

    Args:
        path: The file.

    Returns:
        Its content.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dockerfile_lines(text: str) -> list[str]:
    """Return a Dockerfile's instructions, one per line, its continued lines joined.

    Args:
        text: The Dockerfile's text.

    Returns:
        Each line that is neither blank nor a comment, stripped.
    """
    lines = (line.strip() for line in text.replace("\\\n", " ").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def env_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """Return the (name, value) pairs that a Dockerfile's ENV instructions set, in order.

    Args:
        lines: The Dockerfile's instructions, from ``dockerfile_lines``.

    Returns:
        One pair per ``name=value`` word: a name set twice comes twice.
    """
    words = [
        word
        for line in lines
        if line.startswith("ENV ")
        for word in shlex.split(line.removeprefix("ENV "))
    ]
    return [tuple(word.split("=", 1)) for word in words]


@pytest.fixture(name="run_script")
def run_script_fixture(tmp_path):
    """Return a runner of a SageMaker script, with fake aws, docker and uv first on PATH.

    The runner takes the script's name, its arguments, and its inputs as keywords, starts the
    script from ``tmp_path``, outside the repository, and returns the finished process and the
    commands it ran, each as ``[command, *arguments]``.
    """
    fake = tmp_path / "fake_command.py"
    fake.write_text(FAKE_COMMAND, encoding="utf-8")
    runs_fake = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(fake))}"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("aws", "docker", "uv"):
        wrapper = bin_dir / command
        wrapper.write_text(f'#!/bin/sh\n{runs_fake} {command} "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    kept = {name: value for name, value in os.environ.items() if name not in SCRIPT_INPUTS}
    env = kept | {
        "PATH": f"{bin_dir}{os.pathsep}{kept['PATH']}",
        "AWS_PAGER": "less",  # a user's pager, which every script turns off
        "FAKE_LOG": str(log),
        "FAKE_ROOT": str(PROJECT_ROOT),
        "FAKE_ANSWER": ANSWER,
        "FAKE_PASSWORD": PASSWORD,
        "FAKE_LOGIN": str(tmp_path / LOGIN_STDIN),
    }

    def run(script, *args, **inputs):
        log.write_text("", encoding="utf-8")
        result = subprocess.run(
            [str(SAGEMAKER / script), *args],
            cwd=tmp_path,
            env=env | inputs,
            capture_output=True,
            text=True,
            check=False,
        )
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        return result, calls

    return run


def test_the_image_labels_its_method_and_exposes_the_served_port():
    lines = dockerfile_lines(DOCKERFILE.read_text(encoding="utf-8"))
    [install] = [i for i, line in enumerate(lines) if line.startswith("RUN pip install")]
    label = "LABEL codeforces-tags.method=${METHOD}"

    assert "ARG METHOD" in lines
    # clean-images prunes the method's dangling images by this label, and nothing else.
    assert label in lines
    # After the dependencies and before the weights: an ARG changes the cache key of every layer
    # after it, so methods with one lock share the dependency layers (ADR 0012).
    assert install < lines.index("ARG METHOD") < lines.index(label)
    assert lines.index(label) < lines.index("COPY weights /opt/ml/weights")
    assert f"EXPOSE {SERVING_PORT}" in lines


def test_the_image_sets_mlflows_safeguards_and_the_projects_variables_once_each():
    pairs = env_pairs(dockerfile_lines(DOCKERFILE.read_text(encoding="utf-8")))

    assert len({name for name, _ in pairs}) == len(pairs)
    assert dict(pairs) == MLFLOW_ENV_DEFAULTS | MLFLOW_ENV | IMAGE_ENV


@pytest.mark.usefixtures("mlflow_environment")
def test_the_image_runs_the_entrypoint_and_the_env_creation_flag_that_mlflow_generates(tmp_path):
    # MLflow's container entrypoint is a private API, pinned by uv.lock: the image must run the
    # line `mlflow models generate-dockerfile --env-manager local` writes, and set what it sets
    # (without MLFLOW_DISABLE_ENV_CREATION the entrypoint pip-installs at every start).
    mlflow_env()  # explicit requirements: no uv.lock copied into the model
    code = tmp_path / "echo_model.py"
    code.write_text(ECHO_MODEL, encoding="utf-8")
    model = tmp_path / "model"
    mlflow.pyfunc.save_model(
        str(model), python_model=str(code), pip_requirements=[f"mlflow=={mlflow.__version__}"]
    )

    result = CliRunner().invoke(
        mlflow_models,
        [
            "generate-dockerfile",
            "--model-uri",
            str(model),
            "--output-directory",
            str(tmp_path / "generated"),
            "--env-manager",
            "local",
        ],
    )
    assert result.exit_code == 0, result.output  # before a missing Dockerfile hides MLflow's error
    generated = dockerfile_lines(
        (tmp_path / "generated" / "Dockerfile").read_text(encoding="utf-8")
    )
    ours = dockerfile_lines(DOCKERFILE.read_text(encoding="utf-8"))

    assert [line for line in ours if line.startswith("ENTRYPOINT ")] == [
        line for line in generated if line.startswith("ENTRYPOINT ")
    ]
    assert dict(env_pairs(generated)) == {
        "MLFLOW_DISABLE_ENV_CREATION": dict(env_pairs(ours))["MLFLOW_DISABLE_ENV_CREATION"]
    }


def test_compose_runs_an_image_built_here_on_this_machine_only():
    services = load_yaml(COMPOSE)["services"]

    assert sorted(service["profiles"] for service in services.values()) == [["cpu"], ["gpu"]]
    for service in services.values():
        assert (
            service["image"]
            == f"{SERVING_IMAGE_PREFIX}/${{METHOD:-f2llm_late_fusion}}:${{TAG:-latest}}"
        )
        assert service["pull_policy"] == "never"
        assert service["ports"] == [f"127.0.0.1:${{PORT:-8080}}:{SERVING_PORT}"]
        assert service["env_file"] == [{"path": ".env.serving", "required": False}]
        # MLflow's entrypoint exits with status 0 when the server dies: on-failure never restarts.
        assert "restart" not in service


def test_compose_waits_two_minutes_for_ping_through_python():
    for service in load_yaml(COMPOSE)["services"].values():
        healthcheck = service["healthcheck"]

        assert healthcheck["test"][:3] == ["CMD", "python", "-c"]  # the slim image has no curl
        assert f"urlopen('http://127.0.0.1:{SERVING_PORT}/ping'" in healthcheck["test"][3]
        assert healthcheck["start_period"] == "120s"


@pytest.mark.parametrize(
    ("profile", "device", "deploy"),
    [
        ("cpu", "cpu", None),
        (
            "gpu",
            "cuda",
            {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            },
        ),
    ],
    ids=["cpu", "gpu"],
)
def test_each_compose_profile_sets_its_device_and_only_gpu_reserves_a_gpu(profile, device, deploy):
    services = load_yaml(COMPOSE)["services"].values()
    [service] = [service for service in services if service["profiles"] == [profile]]

    assert service["environment"] == {DEVICE_ENV_VAR: device}
    assert service.get("deploy") == deploy


def test_the_example_lists_each_variable_of_the_server_commented_out():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    # A commented-out assignment: an unquoted value, with no `export`.
    assignments = [re.fullmatch(r"# ([A-Z][A-Z0-9_]*)=([^\s\"']*)", line) for line in lines]

    assert [line for line in lines if line and not line.startswith("#")] == []
    assert [match.group(1) for match in assignments if match] == [
        "MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT",
        *API_KEY_ENV_VARS.values(),
    ]


def test_the_base_runs_one_model_server_with_its_probes_secret_and_resources():
    deployment = load_yaml(K8S / "base" / "deployment.yaml")
    [container] = deployment["spec"]["template"]["spec"]["containers"]
    probes = [container[probe] for probe in ("startupProbe", "readinessProbe", "livenessProbe")]

    assert load_yaml(K8S / "base" / "kustomization.yaml")["resources"] == [
        "deployment.yaml",
        "service.yaml",
    ]
    assert (deployment["metadata"]["name"], deployment["spec"]["replicas"]) == ("model-server", 1)
    assert (container["name"], container["image"]) == ("model-server", PLACEHOLDER)
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["ports"] == [{"containerPort": SERVING_PORT}]
    assert [probe["httpGet"] for probe in probes] == [{"path": "/ping", "port": SERVING_PORT}] * 3
    assert probes[0]["periodSeconds"] * probes[0]["failureThreshold"] == 300
    assert container["envFrom"] == [
        {"secretRef": {"name": "codeforces-tags-env", "optional": True}}
    ]
    assert container["resources"] == {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {"memory": "8Gi"},
    }


def test_the_base_service_selects_the_pods_of_the_deployment_on_the_served_port():
    deployment = load_yaml(K8S / "base" / "deployment.yaml")
    service = load_yaml(K8S / "base" / "service.yaml")
    selected = deployment["spec"]["selector"]["matchLabels"]

    assert (service["metadata"]["name"], service["spec"]["type"]) == ("model-server", "ClusterIP")
    assert service["spec"]["selector"] == selected
    assert selected.items() <= deployment["spec"]["template"]["metadata"]["labels"].items()
    assert service["spec"]["ports"] == [
        {"name": "http", "port": SERVING_PORT, "targetPort": SERVING_PORT}
    ]


@pytest.mark.parametrize(
    ("overlay", "device", "limits", "tolerations"),
    [
        ("cpu", "cpu", None, None),
        (
            "gpu",
            "cuda",
            {"nvidia.com/gpu": 1},
            [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        ),
    ],
    ids=["cpu", "gpu"],
)
def test_each_overlay_sets_its_device_and_only_gpu_asks_for_a_gpu(
    overlay, device, limits, tolerations
):
    kustomization = load_yaml(K8S / "overlays" / overlay / "kustomization.yaml")
    [patch] = kustomization["patches"]
    pod = yaml.safe_load(patch["patch"])["spec"]["template"]["spec"]
    [container] = pod["containers"]

    assert kustomization["resources"] == ["../../base"]
    assert patch["target"] == {"kind": "Deployment", "name": "model-server"}
    assert container["name"] == "model-server"
    assert container["env"] == [{"name": DEVICE_ENV_VAR, "value": device}]
    assert container.get("resources", {}).get("limits") == limits
    assert pod.get("tolerations") == tolerations


def test_the_overlay_that_serving_manifests_writes_replaces_the_image_of_the_base(tmp_path):
    # Another name would leave the placeholder in place: a pod that never pulls its image.
    write_overlay("tfidf", "cpu", "abc1234", out_dir=tmp_path)
    [kustomization] = tmp_path.rglob("kustomization.yaml")

    assert [image["name"] for image in load_yaml(kustomization)["images"]] == [PLACEHOLDER]


@pytest.mark.parametrize("script", SCRIPTS)
def test_each_sagemaker_script_parses_runs_and_reads_no_file_of_variables(script):
    path = SAGEMAKER / script
    syntax = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)

    assert syntax.returncode == 0, syntax.stderr
    assert os.access(path, os.X_OK)
    # The model server's variables file holds API keys, and a SageMaker model's Environment is
    # returned by DescribeModel: no script reads it.
    assert ".env.serving" not in path.read_text(encoding="utf-8")


def test_deploy_pushes_the_image_and_creates_a_cpu_endpoint_without_the_gpu_ami(
    run_script, tmp_path
):
    name = "codeforces-tags-tfidf-cpu"
    image = f"{REGISTRY}/codeforces-tags/tfidf:abc1234"

    result, calls = run_script(
        "deploy.sh", **DEPLOY_INPUTS, TAG="abc1234", FAKE_FAIL="describe-repositories"
    )

    assert result.returncode == 0, result.stderr
    assert calls == [
        ["uv", "run", "--no-sync", "python", "-", "tfidf"],
        ["docker", "image", "inspect", "codeforces-tags/tfidf:abc1234"],
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        ["aws", "ecr", "describe-repositories", "--region", "eu-west-1"]
        + ["--repository-names", "codeforces-tags/tfidf"],
        ["aws", "ecr", "create-repository", "--region", "eu-west-1"]
        + ["--repository-name", "codeforces-tags/tfidf"],
        ["aws", "ecr", "get-login-password", "--region", "eu-west-1"],
        ["docker", "login", "--username", "AWS", "--password-stdin", REGISTRY],
        ["docker", "tag", "codeforces-tags/tfidf:abc1234", image],
        ["docker", "push", image],
        ["aws", "sagemaker", "create-model", "--region", "eu-west-1", "--model-name", name]
        + ["--execution-role-arn", DEPLOY_INPUTS["ROLE_ARN"]]
        + ["--primary-container", f"Image={image},Environment={{CODEFORCES_DEVICE=cpu}}"],
        ["aws", "sagemaker", "create-endpoint-config", "--region", "eu-west-1"]
        + ["--endpoint-config-name", name, "--production-variants"]
        + [
            f"VariantName=AllTraffic,ModelName={name},InstanceType=ml.m5.xlarge,"
            "InitialInstanceCount=1,ContainerStartupHealthCheckTimeoutInSeconds=600"
        ],
        ["aws", "sagemaker", "create-endpoint", "--region", "eu-west-1", "--endpoint-name", name]
        + ["--endpoint-config-name", name],
        ["aws", "sagemaker", "wait", "endpoint-in-service", "--region", "eu-west-1"]
        + ["--endpoint-name", name],
    ]
    # The ECR password reaches `docker login` on its stdin: on no command line (the calls above
    # hold none) and in no output of the script.
    assert (tmp_path / LOGIN_STDIN).read_text(encoding="utf-8") == f"{PASSWORD}\n"
    assert PASSWORD not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("inputs", "instance_type"),
    [({}, "ml.g4dn.xlarge"), ({"INSTANCE_TYPE": "ml.p6-b200.48xlarge"}, "ml.p6-b200.48xlarge")],
    ids=["default_t4", "hyphenated_family"],
)
def test_deploy_runs_a_gpu_endpoint_on_the_inference_ami_of_the_cu130_torch(
    run_script, inputs, instance_type
):
    name = "codeforces-tags-f2llm-late-fusion-gpu"
    image = f"{REGISTRY}/codeforces-tags/f2llm_late_fusion:latest"
    extra_env = "MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT=30"

    result, calls = run_script(
        "deploy.sh",
        **(
            DEPLOY_INPUTS | {"METHOD": "f2llm_late_fusion", "DEVICE": "gpu", "EXTRA_ENV": extra_env}
        ),
        **inputs,
    )
    sagemaker = {
        call[2]: call[3:] for call in calls if call[:2] == ["aws", "sagemaker"] and call != SKELETON
    }

    assert result.returncode == 0, result.stderr
    assert calls[0][-2:] == ["f2llm_late_fusion", "MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT"]
    assert calls[1] == SKELETON  # this CLI can set InferenceAmiVersion: checked before the push
    assert ["ecr", "create-repository"] not in [call[1:3] for call in calls]  # it exists
    assert sagemaker["create-model"][-1] == (
        f"Image={image},Environment={{CODEFORCES_DEVICE=cuda,{extra_env}}}"
    )
    assert sagemaker["create-endpoint-config"][-1] == (
        f"VariantName=AllTraffic,ModelName={name},InstanceType={instance_type},"
        "InitialInstanceCount=1,ContainerStartupHealthCheckTimeoutInSeconds=600,"
        "InferenceAmiVersion=al2023-ami-sagemaker-inference-gpu-4-1"
    )


def test_deploy_takes_sixteen_variables_with_codeforces_device(run_script):
    # SageMaker's limit, reached: one more is refused (the seventeen_variables case below).
    extra_env = [f"V{i}=1" for i in range(15)]
    environment = ",".join(["CODEFORCES_DEVICE=cpu", *extra_env])

    result, calls = run_script("deploy.sh", **DEPLOY_INPUTS, EXTRA_ENV=" ".join(extra_env))

    assert result.returncode == 0, result.stderr
    assert [call[-1] for call in calls if call[:3] == ["aws", "sagemaker", "create-model"]] == [
        f"Image={REGISTRY}/codeforces-tags/tfidf:latest,Environment={{{environment}}}"
    ]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"METHOD": "jev-statement-code"}, "jev-statement-code calls an API"),
        ({"METHOD": "nothing"}, "nothing is not a registered tagging method"),
        ({"ROLE_ARN": ""}, "set ROLE_ARN"),
        ({"ROLE_ARN": "sagemaker-execution"}, "ROLE_ARN is not the ARN of an IAM role"),
        (
            {"ROLE_ARN": "arn:aws:iam::123456789012:user/me"},
            "ROLE_ARN is not the ARN of an IAM role",
        ),
        (
            {"ROLE_ARN": "arn:aws:iam::12345678901:role/sagemaker-execution"},
            "ROLE_ARN is not the ARN of an IAM role",
        ),
        ({"DEVICE": "tpu"}, "DEVICE must be cpu or gpu, not 'tpu'"),
        ({"INSTANCE_TYPE": "ml.m5.xlarge,X=1"}, "is not an instance type such as ml.m5.xlarge"),
        (
            {"DEVICE": "gpu", "INSTANCE_TYPE": "ml.p3.2xlarge"},
            "ml.p3.2xlarge is not an instance type of al2023-ami-sagemaker-inference-gpu-4-1",
        ),
        ({"EXTRA_ENV": "A=1,B=2"}, "EXTRA_ENV entry 1 is not KEY=value"),
        ({"EXTRA_ENV": "A='1'"}, "EXTRA_ENV entry 1 is not KEY=value"),
        ({"EXTRA_ENV": "CODEFORCES_DEVICE=cuda"}, "sets CODEFORCES_DEVICE, which DEVICE sets"),
        ({"EXTRA_ENV": "A=1 A=2"}, "EXTRA_ENV sets A twice"),
        ({"EXTRA_ENV": " ".join(f"V{i}=1" for i in range(16))}, "at most 16 variables"),
        (
            {"EXTRA_ENV": "OPENAI_API_KEY=sk-0 HTTPS_PROXY=x"},
            "never takes: OPENAI_API_KEY HTTPS_PROXY",
        ),
        (
            {"EXTRA_ENV": "DB_PASSWORD=x gcp_credentials=y"},
            "never takes: DB_PASSWORD gcp_credentials",
        ),
        (
            {"DEVICE": "gpu", "FAKE_FAIL": "--generate-cli-skeleton"},
            "this aws CLI cannot set InferenceAmiVersion",
        ),
        (
            {"TAG": "missing", "FAKE_FAIL": "inspect"},
            "python -m codeforces_tags.serving build --method tfidf",
        ),
    ],
    ids=[
        "api_method",
        "unregistered",
        "no_role",
        "role_that_is_no_arn",
        "arn_of_a_user",
        "account_of_eleven_digits",
        "no_such_device",
        "comma_in_instance_type",
        "gpu_outside_the_ami",
        "comma",
        "quotes",
        "device",
        "key_twice",
        "seventeen_variables",
        "credentials",
        "password_and_credentials_file",
        "old_aws_cli",
        "no_image",
    ],
)
def test_deploy_refuses_before_anything_reaches_aws(run_script, inputs, message):
    result, calls = run_script("deploy.sh", **(DEPLOY_INPUTS | inputs))

    assert result.returncode == 1
    assert message in result.stderr
    assert [call for call in calls if call[0] == "aws" and call != SKELETON] == []


@pytest.mark.parametrize(
    ("extra_env", "message", "secret"),
    [
        ("A=1 TYPESAFE_API_KEY=abc==", "EXTRA_ENV entry 2 is not KEY=value", "abc=="),
        ("OPENAI_API_KEY=sk-a@b!", "EXTRA_ENV entry 1 is not KEY=value", "sk-a@b!"),
        ("ghp_0123456789abcdef", "EXTRA_ENV entry 1 is not KEY=value", "ghp_0123456789abcdef"),
        ("sk-live-0123=@x", "EXTRA_ENV entry 1 is not KEY=value", "sk-live-0123"),
        ("dGVzdHNlY3JldA=", "EXTRA_ENV entry 1 is not KEY=value", "dGVzdHNlY3JldA"),
    ],
    ids=["padding", "symbols", "no_key", "key_that_is_no_name", "one_pad_character"],
)
def test_deploy_names_a_malformed_extra_env_entry_by_position_and_never_by_value(
    run_script, extra_env, message, secret
):
    # A value outside the safe characters may be a credential: it is printed nowhere, and
    # neither is the text before its "=", which may be a credential's prefix.
    result, calls = run_script("deploy.sh", **(DEPLOY_INPUTS | {"EXTRA_ENV": extra_env}))

    assert result.returncode == 1
    assert message in result.stderr
    assert secret not in result.stdout + result.stderr
    assert extra_env.split()[-1].split("=")[0] not in result.stdout + result.stderr
    assert calls == []


def test_deploy_prints_no_part_of_a_padded_base64_secret_that_reads_as_a_key(run_script):
    # A bare credential in base64 ends in "==": its text before the first "=" reads as a key, so
    # no part of an entry's text is printed, only its position.
    secret = "dGVzdHNlY3JldA"
    pieces = [secret[start : start + 8] for start in range(len(secret) - 7)]

    result, calls = run_script("deploy.sh", **(DEPLOY_INPUTS | {"EXTRA_ENV": f"{secret}=="}))

    printed = result.stdout + result.stderr
    assert result.returncode == 1
    assert "EXTRA_ENV entry 1 is not KEY=value" in result.stderr
    assert secret not in printed
    assert [piece for piece in pieces if piece in printed] == []
    assert calls == []


def test_invoke_sends_the_payload_file_as_bytes_and_prints_the_answer(run_script, tmp_path):
    payload, answer = tmp_path / "payload.json", tmp_path / "answer.json"
    payload.write_text('{"dataframe_split": {}}', encoding="utf-8")

    result, calls = run_script(
        "invoke.sh", str(payload), str(answer), METHOD="tfidf", DEVICE="cpu", AWS_REGION="eu-west-1"
    )

    assert result.returncode == 0, result.stderr
    assert calls == [
        ["aws", "sagemaker-runtime", "invoke-endpoint", "--region", "eu-west-1"]
        + ["--endpoint-name", "codeforces-tags-tfidf-cpu", "--content-type", "application/json"]
        + ["--body", f"fileb://{payload}", str(answer)]
    ]
    assert "InvokedProductionVariant" in result.stderr  # aws's own reply, kept off stdout
    assert result.stdout == f"{ANSWER}\n"


@pytest.mark.parametrize(
    ("script", "args", "device", "message"),
    [
        ("invoke.sh", ["payload.json"], "cpu", "usage: invoke.sh PAYLOAD_FILE OUT_FILE"),
        ("invoke.sh", ["/nowhere/payload.json", "answer.json"], "cpu", "no payload file"),
        ("delete.sh", [], "tpu", "DEVICE must be cpu or gpu, not 'tpu'"),
    ],
    ids=["invoke_usage", "invoke_no_payload", "delete_no_such_device"],
)
def test_invoke_and_delete_refuse_before_anything_reaches_aws(
    run_script, script, args, device, message
):
    result, calls = run_script(script, *args, METHOD="tfidf", DEVICE=device, AWS_REGION="eu-west-1")

    assert result.returncode == 1
    assert message in result.stderr
    assert calls == []


def test_delete_removes_the_endpoint_its_configuration_and_its_model(run_script):
    name = "codeforces-tags-tfidf-cpu"

    result, calls = run_script("delete.sh", METHOD="tfidf", DEVICE="cpu", AWS_REGION="eu-west-1")

    assert result.returncode == 0, result.stderr
    assert calls == [
        ["aws", "sagemaker", "delete-endpoint", "--region", "eu-west-1", "--endpoint-name", name],
        ["aws", "sagemaker", "wait", "endpoint-deleted", "--region", "eu-west-1"]
        + ["--endpoint-name", name],
        ["aws", "sagemaker", "delete-endpoint-config", "--region", "eu-west-1"]
        + ["--endpoint-config-name", name],
        ["aws", "sagemaker", "delete-model", "--region", "eu-west-1", "--model-name", name],
    ]


@pytest.mark.parametrize(
    "failing", ["delete-endpoint", "endpoint-deleted", "delete-endpoint-config", "delete-model"]
)
def test_delete_tries_every_deletion_and_fails_when_one_failed(run_script, failing):
    result, calls = run_script(
        "delete.sh", METHOD="tfidf", DEVICE="cpu", AWS_REGION="eu-west-1", FAKE_FAIL=failing
    )

    assert result.returncode == 1
    assert [call[2] for call in calls] == [
        "delete-endpoint",
        "wait",
        "delete-endpoint-config",
        "delete-model",
    ]
