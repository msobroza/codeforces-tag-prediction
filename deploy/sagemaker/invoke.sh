#!/usr/bin/env bash
# Ask the SageMaker endpoint that deploy.sh created about the problems of a payload file, and
# print its answer (docs/adr/0012). Each request is billed.
#
#   METHOD=f2llm_late_fusion DEVICE=gpu AWS_REGION=eu-west-1 \
#     deploy/sagemaker/invoke.sh payload.json answer.json
#
# The payload is a request of the model server: a dataframe_split of the raw input columns, as
# codeforces_tags.serving.client.request_payload makes it. fileb:// sends the file's bytes as they
# are: AWS CLI v2 would base64-decode a file:// body.
set -euf -o pipefail
export AWS_PAGER="" # no pager: aws CLI v2 would wait for q when $LESS lacks F

fail() {
  echo "invoke.sh: $*" >&2
  exit 1
}

(($# == 2)) || fail "usage: invoke.sh PAYLOAD_FILE OUT_FILE"
: "${METHOD:?set METHOD to the method deploy.sh deployed}"
: "${DEVICE:?set DEVICE to cpu or gpu, as for deploy.sh}"
: "${AWS_REGION:?set AWS_REGION, as for deploy.sh}"
[[ $DEVICE == cpu || $DEVICE == gpu ]] || fail "DEVICE must be cpu or gpu, not '$DEVICE'"
[[ -f $1 ]] || fail "no payload file $1"

name=codeforces-tags-${METHOD//_/-}-$DEVICE
aws sagemaker-runtime invoke-endpoint --region "$AWS_REGION" --endpoint-name "$name" \
  --content-type application/json --body "fileb://$1" "$2" >&2
cat "$2"
echo
