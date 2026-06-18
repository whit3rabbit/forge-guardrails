#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RUN_LOCAL_EVAL="$SCRIPT_DIR/run_local_eval.sh"

[[ -x "$RUN_LOCAL_EVAL" ]] || {
  printf 'error: expected executable eval runner at %s\n' "$RUN_LOCAL_EVAL" >&2
  exit 1
}

cd "$REPO_ROOT"

MODEL="openrouter/owl-alpha"
BASE="target/local-eval/or-owl-alpha-classify"

common=(
  --suite release
  --runs 30
  --upstream-backend openrouter
  --model "$MODEL"
  --skip-published-compare
  --tool-output-compression aggressive
  --tool-output-compression-method auto
)

"$RUN_LOCAL_EVAL" "${common[@]}" \
  --output-dir "$BASE/compression-only"

"$RUN_LOCAL_EVAL" "${common[@]}" \
  --classify \
  --classifier-model full \
  --classifier-mode advisory \
  --verify-final-response \
  --final-response-classifier-mode advisory \
  --output-dir "$BASE/advisory"

"$RUN_LOCAL_EVAL" "${common[@]}" \
  --classify \
  --classifier-model full \
  --classifier-mode enforce \
  --verify-final-response \
  --final-response-classifier-mode enforce \
  --output-dir "$BASE/enforce"
