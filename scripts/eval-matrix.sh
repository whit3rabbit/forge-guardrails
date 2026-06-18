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
BASE="target/local-eval/or-owl-alpha"

common=(
  --suite release
  --runs 10
  --upstream-backend openrouter
  --model "$MODEL"
  --skip-published-compare
)

# 1. Baseline: no classifier, no final-response verifier, no compression
"$RUN_LOCAL_EVAL" "${common[@]}" \
  --tool-output-compression disabled \
  --output-dir "$BASE/baseline"

# 2. Tool-call classifier in shadow mode
"$RUN_LOCAL_EVAL" "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --tool-output-compression disabled \
  --output-dir "$BASE/toolcall-shadow"

# 3. Tool-call classifier + final-response verifier, both shadow
"$RUN_LOCAL_EVAL" "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --tool-output-compression disabled \
  --output-dir "$BASE/toolcall-final-shadow"

# 4. Compression only, no classifier/verifier
"$RUN_LOCAL_EVAL" "${common[@]}" \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto \
  --output-dir "$BASE/compression-aggressive"

# 5. Full stack: classifier + final verifier + aggressive compression
"$RUN_LOCAL_EVAL" "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto \
  --output-dir "$BASE/full-shadow-aggressive"
