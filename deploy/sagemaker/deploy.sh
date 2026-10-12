#!/usr/bin/env bash
# Deploy a tagging method's image to a SageMaker real-time endpoint (docs/adr/0012).
#
# Run it only on the user's explicit go: it pushes the image to Amazon ECR (about 8 GB for
# f2llm_late_fusion), and the endpoint is billed by the hour until delete.sh removes it.
#
#   METHOD=f2llm_late_fusion DEVICE=gpu AWS_REGION=eu-west-1 \
#     ROLE_ARN=arn:aws:iam::123456789012:role/sagemaker-execution deploy/sagemaker/deploy.sh
#
# Inputs, from the environment:
#   METHOD         a tagging method whose image `python -m codeforces_tags.serving build` made here
#   TAG            the image's tag (latest by default)
#   DEVICE         cpu or gpu
#   AWS_REGION     the region of the ECR repository and of the endpoint
#   ROLE_ARN       the SageMaker execution role, which must be able to pull from ECR
#   INSTANCE_TYPE  optional: ml.m5.xlarge (cpu) or ml.g4dn.xlarge (gpu) by default
#   EXTRA_ENV      optional: space-separated KEY=value pairs added to the model's Environment
#
# A method that calls an API is refused, and no credential is ever sent: a SageMaker model's
# Environment is not for secrets (DescribeModel returns it), so the script reads no file of
# variables and refuses a credential in EXTRA_ENV.
set -euf -o pipefail
export AWS_PAGER="" # no pager: aws CLI v2 would wait for q when $LESS lacks F

fail() {
  echo "deploy.sh: $*" >&2
  exit 1
}

: "${METHOD:?set METHOD to a tagging method}"
: "${DEVICE:?set DEVICE to cpu or gpu}"
: "${AWS_REGION:?set AWS_REGION}"
: "${ROLE_ARN:?set ROLE_ARN to the SageMaker execution role}"
# An IAM role of any partition and path. A typo would fail only at create-model, after the push.
[[ $ROLE_ARN =~ ^arn:aws[a-z-]*:iam::[0-9]{12}:role/.+$ ]] ||
  fail "ROLE_ARN is not the ARN of an IAM role such as arn:aws:iam::123456789012:role/name"
TAG=${TAG:-latest}
EXTRA_ENV=${EXTRA_ENV:-}
# SageMaker's default GPU image has an NVIDIA driver of 470, on which the cu130 torch sees no GPU;
# this one has driver 580 and CUDA 13.0, on the instance families listed.
GPU_AMI=al2023-ami-sagemaker-inference-gpu-4-1
GPU_FAMILIES="g4dn g5 g6 g6e p4d p4de p5 p5e p5en g7 g7e p6"

case $DEVICE in
  cpu)
    device=cpu
    INSTANCE_TYPE=${INSTANCE_TYPE:-ml.m5.xlarge}
    ;;
  gpu)
    device=cuda
    INSTANCE_TYPE=${INSTANCE_TYPE:-ml.g4dn.xlarge}
    ;;
  *) fail "DEVICE must be cpu or gpu, not '$DEVICE'" ;;
esac
[[ $INSTANCE_TYPE =~ ^ml\.[a-z0-9-]+\.[a-z0-9]+$ ]] ||
  fail "INSTANCE_TYPE '$INSTANCE_TYPE' is not an instance type such as ml.m5.xlarge"
family=${INSTANCE_TYPE#ml.}
family=${family%%.*}
family=${family%%-*} # ml.p6-b200.48xlarge is of the p6 family
if [[ $DEVICE == gpu && " $GPU_FAMILIES " != *" $family "* ]]; then
  fail "$INSTANCE_TYPE is not an instance type of $GPU_AMI, whose families are $GPU_FAMILIES"
fi

# The aws shorthand below splits at commas and strips quotes, so keys and values keep to safe
# characters; SageMaker takes at most 16 variables, CODEFORCES_DEVICE included. A refused entry
# is named by its position only, never by any part of its text: no rule can tell a key from a
# credential's prefix (a padded base64 secret, ending in ==, reads as a key and then "="), and
# stderr may be logged. A value is never empty: with one pad character (=), such a secret would
# read as a key with an empty value, and reach the model's Environment as a variable's name.
pair_pattern='^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9._:/+-]+$'
safe_characters='letters, digits and . _ : / + -'
environment="CODEFORCES_DEVICE=$device"
extra_keys=" "
count=1
position=0
for pair in $EXTRA_ENV; do # no glob expansion: set -f
  position=$((position + 1))
  [[ $pair =~ $pair_pattern ]] ||
    fail "EXTRA_ENV entry $position is not KEY=value, with a non-empty value of $safe_characters"
  key=${pair%%=*}
  [[ $key != CODEFORCES_DEVICE ]] || fail "EXTRA_ENV sets CODEFORCES_DEVICE, which DEVICE sets"
  [[ $extra_keys != *" $key "* ]] || fail "EXTRA_ENV sets $key twice"
  extra_keys+="$key "
  environment+=",$pair"
  count=$((count + 1))
done
((count <= 16)) || fail "SageMaker takes at most 16 variables with CODEFORCES_DEVICE, not $count"

# The registry says whether the method calls an API, and config.MLFLOW_SERVER_LEFT_OUT which
# variable names hold a credential (API keys, tokens, secrets, passwords, credentials, cloud keys,
# proxies); uv finds the project from the repository's root.
cd "$(dirname "$0")/../.."
# shellcheck disable=SC2086 # one argument per key of EXTRA_ENV
uv run --no-sync python - "$METHOD" $extra_keys <<'PYTHON'
import re
import sys

from codeforces_tags.config import MLFLOW_SERVER_LEFT_OUT
from codeforces_tags.methods import METHODS

method, keys = sys.argv[1], sys.argv[2:]
spec = METHODS.get(method)
if spec is None:
    sys.exit(f"deploy.sh: {method} is not a registered tagging method")
if spec.calls_api:
    sys.exit(f"deploy.sh: {method} calls an API, and API methods are not deployed to SageMaker")
credentials = [key for key in keys if re.search(MLFLOW_SERVER_LEFT_OUT, key, re.IGNORECASE)]
if credentials:
    sys.exit(
        "deploy.sh: EXTRA_ENV holds credentials, which a model's Environment never takes: "
        + " ".join(credentials)
    )
PYTHON

# --generate-cli-skeleton prints the request this aws CLI knows, offline: an older CLI would refuse
# InferenceAmiVersion only after the push, with the model already created.
if [[ $DEVICE == gpu ]]; then
  skeleton=$(aws sagemaker create-endpoint-config --generate-cli-skeleton 2>/dev/null || true)
  [[ $skeleton == *InferenceAmiVersion* ]] ||
    fail "this aws CLI cannot set InferenceAmiVersion, which a gpu endpoint needs: upgrade it first"
fi

local_image=codeforces-tags/$METHOD:$TAG
docker image inspect "$local_image" >/dev/null ||
  fail "no image $local_image: python -m codeforces_tags.serving build --method $METHOD makes it"

# One name for the model, its endpoint configuration and its endpoint; SageMaker names allow no
# underscore.
name=codeforces-tags-${METHOD//_/-}-$DEVICE
account=$(aws sts get-caller-identity --query Account --output text)
registry=$account.dkr.ecr.$AWS_REGION.amazonaws.com
repository=codeforces-tags/$METHOD
image=$registry/$repository:$TAG

if ! aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repository" \
  >/dev/null 2>&1; then
  aws ecr create-repository --region "$AWS_REGION" --repository-name "$repository" >/dev/null
fi
password=$(aws ecr get-login-password --region "$AWS_REGION")
docker login --username AWS --password-stdin "$registry" <<<"$password"
docker tag "$local_image" "$image"
docker push "$image"

aws sagemaker create-model --region "$AWS_REGION" --model-name "$name" \
  --execution-role-arn "$ROLE_ARN" --primary-container "Image=$image,Environment={$environment}"
variant="VariantName=AllTraffic,ModelName=$name,InstanceType=$INSTANCE_TYPE,InitialInstanceCount=1"
variant+=",ContainerStartupHealthCheckTimeoutInSeconds=600"
if [[ $DEVICE == gpu ]]; then
  variant+=",InferenceAmiVersion=$GPU_AMI"
fi
aws sagemaker create-endpoint-config --region "$AWS_REGION" --endpoint-config-name "$name" \
  --production-variants "$variant"
aws sagemaker create-endpoint --region "$AWS_REGION" --endpoint-name "$name" \
  --endpoint-config-name "$name"
echo "Waiting for $name to be in service (up to an hour)..."
aws sagemaker wait endpoint-in-service --region "$AWS_REGION" --endpoint-name "$name" ||
  fail "$name is not in service: describe-endpoint gives the reason, and delete.sh removes it"
echo "$name is in service: invoke.sh asks it, and delete.sh removes it."
