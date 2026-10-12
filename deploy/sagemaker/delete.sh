#!/usr/bin/env bash
# Delete the SageMaker endpoint that deploy.sh created, its endpoint configuration and its model,
# which ends the hourly bill (docs/adr/0012). The image stays in ECR.
#
#   METHOD=f2llm_late_fusion DEVICE=gpu AWS_REGION=eu-west-1 deploy/sagemaker/delete.sh
#
# Every deletion is tried, even after one failed (an endpoint already deleted, say); the script
# then exits with 1.
set -euf -o pipefail
export AWS_PAGER="" # no pager: aws CLI v2 would wait for q when $LESS lacks F

fail() {
  echo "delete.sh: $*" >&2
  exit 1
}

: "${METHOD:?set METHOD to the method deploy.sh deployed}"
: "${DEVICE:?set DEVICE to cpu or gpu, as for deploy.sh}"
: "${AWS_REGION:?set AWS_REGION, as for deploy.sh}"
[[ $DEVICE == cpu || $DEVICE == gpu ]] || fail "DEVICE must be cpu or gpu, not '$DEVICE'"

name=codeforces-tags-${METHOD//_/-}-$DEVICE
status=0
aws sagemaker delete-endpoint --region "$AWS_REGION" --endpoint-name "$name" || status=1
aws sagemaker wait endpoint-deleted --region "$AWS_REGION" --endpoint-name "$name" || status=1
aws sagemaker delete-endpoint-config --region "$AWS_REGION" --endpoint-config-name "$name" ||
  status=1
aws sagemaker delete-model --region "$AWS_REGION" --model-name "$name" || status=1
echo "The image stays in ECR, in the repository codeforces-tags/$METHOD."
exit "$status"
