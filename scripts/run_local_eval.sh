#!/usr/bin/env bash
set -euo pipefail

# Bash reads scripts incrementally; run from a temp copy so long evals keep
# using the version they started with even if this file is edited mid-run.
if [[ -z "${RUN_LOCAL_EVAL_SCRIPT_COPY:-}" ]]; then
  script_original="${BASH_SOURCE[0]}"
  if [[ "$script_original" != */* ]]; then
    script_original="$(command -v "$script_original")"
  elif [[ "$script_original" != /* ]]; then
    script_original="$(pwd -P)/$script_original"
  fi
  script_original="$(cd "$(dirname "$script_original")" && pwd -P)/$(basename "$script_original")"
  script_copy="$(mktemp "${TMPDIR:-/tmp}/run_local_eval.XXXXXX")"
  cp "$script_original" "$script_copy"
  chmod +x "$script_copy"
  export RUN_LOCAL_EVAL_SCRIPT_COPY="$script_copy"
  export RUN_LOCAL_EVAL_ORIGINAL="$script_original"
  exec /usr/bin/env bash "$script_copy" "$@"
fi

trap 'rm -f "${RUN_LOCAL_EVAL_SCRIPT_COPY:-}"' EXIT

DEFAULT_GGUF="mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf"
DEFAULT_PROXY_PORT="8081"
DEFAULT_BACKEND_PORT="8080"
DEFAULT_BUDGET_TOKENS="8192"
DEFAULT_MODEL="Ministral-3-8B-Instruct-2512-Q8_0"
DEFAULT_PROXY_BACKEND_MODE="native"
DEFAULT_UPSTREAM_BACKEND="llamaserver"
DEFAULT_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

SCRIPT_SOURCE="${RUN_LOCAL_EVAL_ORIGINAL:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PROGRAM_NAME="$(basename "$SCRIPT_SOURCE")"

SUITE="smoke"
RUNS="1"
GGUF="$DEFAULT_GGUF"
MODEL="$DEFAULT_MODEL"
MODEL_EXPLICIT=0
PUBLISHED_MODEL=""
PUBLISHED_BACKEND_MODE=""
PUBLISHED_BACKEND_MODE_SET=0
PROXY_BACKEND_MODE="$DEFAULT_PROXY_BACKEND_MODE"
UPSTREAM_BACKEND="${FORGE_EVAL_UPSTREAM_BACKEND:-${UPSTREAM_BACKEND:-$DEFAULT_UPSTREAM_BACKEND}}"
OPENROUTER_BASE_URL="${FORGE_EVAL_OPENROUTER_BASE_URL:-${OPENROUTER_BASE_URL:-$DEFAULT_OPENROUTER_BASE_URL}}"
SKIP_PUBLISHED_COMPARE=0
FORCE_PUBLISHED_COMPARE=0
AUTO_SKIP_PUBLISHED_COMPARE=0
INCLUDE_COMPACTION_CHAIN=0
PROXY_PORT="${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
BACKEND_PORT="${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"
HEALTH_TIMEOUT="180"
STREAM=1
OUTPUT_DIR=""
CLASSIFIER_DIR="${FORGE_CLASSIFIER_DIR:-}"
CLASSIFIER_MODE="${FORGE_CLASSIFIER_MODE:-shadow}"
if [[ -n "${FORGE_CLASSIFIER_MODE:-}" ]]; then
  CLASSIFIER_MODE_EXPLICIT=1
else
  CLASSIFIER_MODE_EXPLICIT=0
fi
if [[ -n "${FORGE_CLASSIFIER_MODEL:-}" ]]; then
  CLASSIFIER_MODEL="$FORGE_CLASSIFIER_MODEL"
elif [[ "${FORGE_CLASSIFIER_USE_QUANTIZED:-}" =~ ^(0|false|no|off)$ ]]; then
  CLASSIFIER_MODEL="full"
else
  CLASSIFIER_MODEL="quantized"
fi
CLASSIFY=0
DOWNLOAD_CLASSIFIER=0
FINAL_RESPONSE_CLASSIFIER_DIR="${FORGE_FINAL_RESPONSE_CLASSIFIER_DIR:-}"
FINAL_RESPONSE_CLASSIFIER_MODE="${FORGE_FINAL_RESPONSE_CLASSIFIER_MODE:-shadow}"
FINAL_RESPONSE_CLASSIFIER_MODEL="${FORGE_FINAL_RESPONSE_CLASSIFIER_MODEL:-quantized}"
VERIFY_FINAL_RESPONSE=0
DOWNLOAD_FINAL_RESPONSE_CLASSIFIER=0
if [[ "${FORGE_RESOURCE_BASELINE:-}" =~ ^(1|true|yes|on)$ ]]; then
  RESOURCE_BASELINE=1
else
  RESOURCE_BASELINE=0
fi
RESOURCE_INTERVAL="${FORGE_RESOURCE_INTERVAL:-1.0}"
TOOL_OUTPUT_COMPRESSION="${FORGE_TOOL_OUTPUT_COMPRESSION:-disabled}"
TOOL_OUTPUT_COMPRESSION_METHOD="${FORGE_TOOL_OUTPUT_COMPRESSION_METHOD:-lzw}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
PROXY_PID=""
CURRENT_PROXY_LOG=""
RESOURCE_SAMPLER_PID=""
RESOURCE_SAMPLER_LABEL=""

usage() {
  cat <<EOF
Usage: $PROGRAM_NAME [options]

Runs Forge evals against the local Rust proxy with a managed or remote upstream.

Options:
  --suite smoke|release     Eval suite to run (default: smoke)
  --runs N                  Runs per scenario (default: 1)
  --gguf PATH               GGUF path for --upstream-backend llamaserver (default: $DEFAULT_GGUF)
  --output-dir DIR          Output directory (default: target/local-eval/<timestamp>)
  --model MODEL             Model name sent to the proxy (default: $DEFAULT_MODEL)
  --upstream-backend llamaserver|openrouter
                            Upstream used behind the local Forge proxy (default: $DEFAULT_UPSTREAM_BACKEND)
  --openrouter-base-url URL OpenRouter-compatible upstream URL (default: $DEFAULT_OPENROUTER_BASE_URL)
  --published-model MODEL   Published baseline model (default: --model)
  --published-mode LS/N|LS/P Published baseline row (default: LS/N for native, LS/P for prompt)
  --proxy-backend-mode native|prompt
                            Backend tool-call mode for managed proxy (default: $DEFAULT_PROXY_BACKEND_MODE)
  --skip-published-compare  Do not compare release results to published results
  --force-published-compare Compare proxy rows to direct published rows anyway
  --include-compaction-chain Also run compaction-chain scenarios after published scenarios
  --proxy-port PORT         Proxy port (default: $DEFAULT_PROXY_PORT)
  --backend-port PORT       Managed llama-server port (default: $DEFAULT_BACKEND_PORT)
  --health-timeout SECONDS  Seconds to wait for /health (default: 180)
  --no-stream               Disable streaming eval requests
  --classifier-dir DIR      Enable local ONNX classifier artifact directory
  --classifier-mode MODE    disabled|shadow|advisory|enforce (default: $CLASSIFIER_MODE)
  --classifier-model MODEL  quantized|full (default: $CLASSIFIER_MODEL)
  --classify                Enable tool-call classifier shortcut; download if missing
  --download-classifier     Download classifier artifacts before running
  --final-response-classifier-dir DIR
                            Enable local final-response classifier artifact directory
  --final-response-classifier-mode MODE
                            disabled|shadow|advisory|enforce (default: $FINAL_RESPONSE_CLASSIFIER_MODE)
  --final-response-classifier-model MODEL
                            quantized|full (default: $FINAL_RESPONSE_CLASSIFIER_MODEL)
  --verify-final-response   Enable final-response verifier shortcut; download if missing
  --download-final-response-classifier
                            Download final-response classifier artifacts before running
  --resource-baseline       Capture proxy/backend CPU and RSS stats during eval windows
  --resource-interval SECONDS
                            Resource sampling interval (default: 1.0)
  --tool-output-compression disabled|safe|standard|aggressive
                            Enable proxy tool-output compression (default: $TOOL_OUTPUT_COMPRESSION)
  --tool-output-compression-method lzw|repair|auto
                            Aggressive compression method (default: $TOOL_OUTPUT_COMPRESSION_METHOD)
  -h, --help                Show this help

Examples:
  $PROGRAM_NAME --suite smoke --runs 1
  $PROGRAM_NAME --suite release --runs 10
  $PROGRAM_NAME --suite release --runs 10 --classify
  $PROGRAM_NAME --suite release --runs 10 --classify --classifier-mode shadow
  $PROGRAM_NAME --suite release --runs 10 --classify --classifier-mode shadow --verify-final-response
  $PROGRAM_NAME --suite release --runs 10 --classifier-dir target/classifier-artifacts/onnx
  $PROGRAM_NAME --suite release --runs 10 --download-classifier
  $PROGRAM_NAME --suite release --runs 10 --tool-output-compression standard
  OPENROUTER_API_KEY=... $PROGRAM_NAME --suite smoke --runs 1 --upstream-backend openrouter --model openrouter/free
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '%s\n' "$*" >&2
}

phase() {
  log ""
  log "==> $*"
}

valid_positive_int() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
  esac
  (( 10#$1 > 0 ))
}

valid_positive_decimal() {
  case "$1" in
    ''|*[!0-9.]*|*.*.*|.)
      return 1
      ;;
  esac
  local digits
  digits="${1//./}"
  [[ "$digits" == *[1-9]* ]]
}

expand_path() {
  case "$1" in
    \~)
      printf '%s\n' "$HOME"
      ;;
    \~/*)
      printf '%s/%s\n' "$HOME" "${1#~/}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

canonical_dir() {
  local path
  path="$(resolve_local_path "$1")"
  [[ -d "$path" ]] || die "directory not found: $path"
  (cd "$path" && pwd -P)
}

resolve_local_path() {
  local path
  path="$(expand_path "$1")"
  case "$path" in
    /*)
      printf '%s\n' "$path"
      ;;
    *)
      printf '%s/%s\n' "$REPO_ROOT" "$path"
      ;;
  esac
}

classifier_enabled() {
  [[ -n "$CLASSIFIER_DIR" && "$CLASSIFIER_MODE" != "disabled" ]]
}

final_response_classifier_enabled() {
  [[ -n "$FINAL_RESPONSE_CLASSIFIER_DIR" && "$FINAL_RESPONSE_CLASSIFIER_MODE" != "disabled" ]]
}

classifier_feature_enabled() {
  classifier_enabled || final_response_classifier_enabled
}

openrouter_api_key_source() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    printf '%s\n' "OPENAI_API_KEY"
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    printf '%s\n' "OPENROUTER_API_KEY"
  else
    printf '%s\n' "missing"
  fi
}

next_arg() {
  local flag value
  flag="$1"
  value="${2:-}"
  [[ -n "$value" ]] || die "$flag requires a value"
  [[ "$value" != --* ]] || die "$flag requires a value"
  printf '%s\n' "$value"
}

have_python_runner() {
  command -v uv >/dev/null 2>&1 \
    || [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]] \
    || command -v "$PYTHON_BIN" >/dev/null 2>&1
}

run_python_stdin() {
  if command -v uv >/dev/null 2>&1; then
    (cd "$REPO_ROOT" && env -u VIRTUAL_ENV uv run --project forge python - "$@")
  elif [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]]; then
    (cd "$REPO_ROOT" && "$REPO_ROOT/forge/.venv/bin/python" - "$@")
  else
    (cd "$REPO_ROOT" && "$PYTHON_BIN" - "$@")
  fi
}

run_python_script() {
  if command -v uv >/dev/null 2>&1; then
    (cd "$REPO_ROOT" && env -u VIRTUAL_ENV uv run --project forge python "$@")
  elif [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]]; then
    (cd "$REPO_ROOT" && "$REPO_ROOT/forge/.venv/bin/python" "$@")
  else
    (cd "$REPO_ROOT" && "$PYTHON_BIN" "$@")
  fi
}

scenario_names() {
  local kind="$1"
  run_python_stdin "$REPO_ROOT" "$kind" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
kind = sys.argv[2]
sys.path.insert(0, str(root / "forge" / "src"))
sys.path.insert(0, str(root / "forge"))

from tests.eval.scenarios import ALL_SCENARIOS

smoke = {"basic_2step", "sequential_3step", "error_recovery"}
process_budget_compaction = {
    "compaction_chain_p1",
    "compaction_chain_p2",
    "compaction_chain_p3",
}

if kind == "smoke":
    names = [scenario.name for scenario in ALL_SCENARIOS if scenario.name in smoke]
elif kind == "release-normal":
    names = [
        scenario.name
        for scenario in ALL_SCENARIOS
        if not scenario.name.startswith("compaction_chain_")
    ]
else:
    raise SystemExit(f"unknown scenario set: {kind}")

for name in names:
    print(name)
PY
}

cleanup() {
  local status="$?"
  trap - EXIT INT TERM HUP
  stop_resource_sampler
  stop_proxy
  rm -f "${RUN_LOCAL_EVAL_SCRIPT_COPY:-}"
  exit "$status"
}

stop_proxy() {
  stop_resource_sampler
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    log "Stopping proxy pid $PROXY_PID"
    kill -TERM "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  PROXY_PID=""
}

start_resource_sampler() {
  local label sampler_log
  label="$1"
  [[ "$RESOURCE_BASELINE" == "1" ]] || return 0
  [[ -n "$PROXY_PID" ]] || die "cannot start resource sampler without a proxy pid"

  RESOURCE_SAMPLER_LABEL="$label"
  sampler_log="$OUTPUT_DIR/resource_sampler_${label}.log"
  log "Resource baseline: sampling label=$label, interval=${RESOURCE_INTERVAL}s"
  (
    cd "$REPO_ROOT"
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      exec "$PYTHON_BIN" scripts/proxy_resource_baseline.py sample \
        --root-pid "$PROXY_PID" \
        --label "$label" \
        --output-dir "$OUTPUT_DIR" \
        --interval "$RESOURCE_INTERVAL"
    elif [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]]; then
      exec "$REPO_ROOT/forge/.venv/bin/python" scripts/proxy_resource_baseline.py sample \
        --root-pid "$PROXY_PID" \
        --label "$label" \
        --output-dir "$OUTPUT_DIR" \
        --interval "$RESOURCE_INTERVAL"
    elif command -v uv >/dev/null 2>&1; then
      exec env -u VIRTUAL_ENV uv run --project forge python scripts/proxy_resource_baseline.py sample \
        --root-pid "$PROXY_PID" \
        --label "$label" \
        --output-dir "$OUTPUT_DIR" \
        --interval "$RESOURCE_INTERVAL"
    else
      die "python runner not found for resource sampler"
    fi
  ) >"$sampler_log" 2>&1 &
  RESOURCE_SAMPLER_PID="$!"
}

stop_resource_sampler() {
  local status
  if [[ -z "$RESOURCE_SAMPLER_PID" ]]; then
    return 0
  fi

  if kill -0 "$RESOURCE_SAMPLER_PID" 2>/dev/null; then
    log "Stopping resource sampler pid $RESOURCE_SAMPLER_PID"
    kill -TERM "$RESOURCE_SAMPLER_PID" 2>/dev/null || true
  fi

  set +e
  wait "$RESOURCE_SAMPLER_PID" 2>/dev/null
  status="$?"
  set -e
  if [[ "$status" != "0" && "$status" != "130" && "$status" != "143" ]]; then
    log "warning: resource sampler for $RESOURCE_SAMPLER_LABEL exited with status $status"
  fi
  RESOURCE_SAMPLER_PID=""
  RESOURCE_SAMPLER_LABEL=""
}

run_resource_report() {
  local report
  [[ "$RESOURCE_BASELINE" == "1" ]] || return 0
  report="$OUTPUT_DIR/resource_baseline_report.txt"

  phase "Resource baseline report"
  run_python_script scripts/proxy_resource_baseline.py report \
    --output-dir "$OUTPUT_DIR" \
    --report "$report"
  log "Resource report: $report"
}

wait_for_health() {
  local url pid elapsed
  url="http://127.0.0.1:${PROXY_PORT}/health"
  pid="$1"
  elapsed=0

  log "Waiting for proxy health at $url"
  while (( elapsed < HEALTH_TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 80 "$CURRENT_PROXY_LOG" >&2 || true
      die "proxy exited before becoming healthy"
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "Proxy healthy at $url"
      return 0
    fi
    if (( elapsed > 0 && elapsed % 10 == 0 )); then
      log "Still waiting for proxy health: ${elapsed}s elapsed"
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  tail -n 80 "$CURRENT_PROXY_LOG" >&2 || true
  die "proxy did not become healthy within ${HEALTH_TIMEOUT}s"
}

download_classifier_artifacts() {
  local output_dir
  command -v cargo >/dev/null 2>&1 || die "cargo is required for --download-classifier"

  if [[ -z "$CLASSIFIER_DIR" ]]; then
    CLASSIFIER_DIR="$REPO_ROOT/target/classifier-artifacts/onnx"
  fi

  CLASSIFIER_DIR="$(resolve_local_path "$CLASSIFIER_DIR")"
  if [[ "$(basename "$CLASSIFIER_DIR")" == "onnx" ]]; then
    output_dir="$(dirname "$CLASSIFIER_DIR")"
  else
    output_dir="$CLASSIFIER_DIR"
    CLASSIFIER_DIR="$CLASSIFIER_DIR/onnx"
  fi

  phase "Download classifier artifacts"
  log "Downloading classifier artifacts -> $CLASSIFIER_DIR"
  log "Classifier model: $CLASSIFIER_MODEL"
  (cd "$REPO_ROOT" && cargo run --features classifier --bin download-classifier -- \
    --artifact tool-call \
    --output-dir "$output_dir" \
    --classifier-model "$CLASSIFIER_MODEL")
}

download_classifier_shortcut_artifact() {
  local output status
  command -v cargo >/dev/null 2>&1 || die "cargo is required for --classify"

  phase "Download classifier shortcut artifact"
  log "Classifier model: $CLASSIFIER_MODEL"
  if [[ -n "$CLASSIFIER_DIR" ]]; then
    CLASSIFIER_DIR="$(resolve_local_path "$CLASSIFIER_DIR")"
    log "Classifier dir: $CLASSIFIER_DIR"
  else
    log "Classifier dir: user cache"
  fi

  local cmd=(
    cargo run --quiet --features classifier --bin forge-guardrails-proxy --
    --classify-download
    --classifier-model "$CLASSIFIER_MODEL"
  )
  if [[ -n "$CLASSIFIER_DIR" ]]; then
    cmd+=(--classifier-dir "$CLASSIFIER_DIR")
  fi

  set +e
  output="$(cd "$REPO_ROOT" && "${cmd[@]}")"
  status="$?"
  set -e
  printf '%s\n' "$output" >&2
  [[ "$status" == "0" ]] || die "failed to download classifier shortcut artifact"

  CLASSIFIER_DIR="$(printf '%s\n' "$output" | awk -F= '$1 == "classifier_dir" { sub(/^classifier_dir=/, ""); print; exit }')"
  [[ -n "$CLASSIFIER_DIR" ]] || die "classifier_dir missing from --classify-download output"
}

download_final_response_classifier_artifacts() {
  local output_dir
  command -v cargo >/dev/null 2>&1 || die "cargo is required for --download-final-response-classifier"

  if [[ -z "$FINAL_RESPONSE_CLASSIFIER_DIR" ]]; then
    FINAL_RESPONSE_CLASSIFIER_DIR="$REPO_ROOT/target/final-response-classifier-artifacts/onnx"
  fi

  FINAL_RESPONSE_CLASSIFIER_DIR="$(resolve_local_path "$FINAL_RESPONSE_CLASSIFIER_DIR")"
  if [[ "$(basename "$FINAL_RESPONSE_CLASSIFIER_DIR")" == "onnx" ]]; then
    output_dir="$(dirname "$FINAL_RESPONSE_CLASSIFIER_DIR")"
  else
    output_dir="$FINAL_RESPONSE_CLASSIFIER_DIR"
    FINAL_RESPONSE_CLASSIFIER_DIR="$FINAL_RESPONSE_CLASSIFIER_DIR/onnx"
  fi

  phase "Download final-response classifier artifacts"
  log "Downloading final-response classifier artifacts -> $FINAL_RESPONSE_CLASSIFIER_DIR"
  log "Final-response classifier model: $FINAL_RESPONSE_CLASSIFIER_MODEL"
  (cd "$REPO_ROOT" && cargo run --features classifier --bin download-classifier -- \
    --artifact final-response \
    --output-dir "$output_dir" \
    --classifier-model "$FINAL_RESPONSE_CLASSIFIER_MODEL")
}

prepare_classifier_binaries() {
  classifier_feature_enabled || return 0
  command -v cargo >/dev/null 2>&1 || die "cargo is required when --classifier-dir is set"

  if classifier_enabled; then
    CLASSIFIER_DIR="$(canonical_dir "$CLASSIFIER_DIR")"
    [[ -f "$CLASSIFIER_DIR/artifact_manifest.json" ]] || die "classifier artifact_manifest.json missing in $CLASSIFIER_DIR"
    [[ -f "$CLASSIFIER_DIR/tokenizer.json" ]] || die "classifier tokenizer.json missing in $CLASSIFIER_DIR"
    case "$CLASSIFIER_MODEL" in
      quantized)
        [[ -f "$CLASSIFIER_DIR/model_quantized.onnx" ]] || die "classifier model_quantized.onnx missing in $CLASSIFIER_DIR"
        ;;
      full)
        [[ -f "$CLASSIFIER_DIR/model.onnx" ]] || die "classifier model.onnx missing in $CLASSIFIER_DIR"
        ;;
    esac
  fi

  if final_response_classifier_enabled; then
    FINAL_RESPONSE_CLASSIFIER_DIR="$(canonical_dir "$FINAL_RESPONSE_CLASSIFIER_DIR")"
    [[ -f "$FINAL_RESPONSE_CLASSIFIER_DIR/artifact_manifest.json" ]] || die "final-response classifier artifact_manifest.json missing in $FINAL_RESPONSE_CLASSIFIER_DIR"
    [[ -f "$FINAL_RESPONSE_CLASSIFIER_DIR/tokenizer.json" ]] || die "final-response classifier tokenizer.json missing in $FINAL_RESPONSE_CLASSIFIER_DIR"
    case "$FINAL_RESPONSE_CLASSIFIER_MODEL" in
      quantized)
        [[ -f "$FINAL_RESPONSE_CLASSIFIER_DIR/model_quantized.onnx" ]] || die "final-response classifier model_quantized.onnx missing in $FINAL_RESPONSE_CLASSIFIER_DIR"
        ;;
      full)
        [[ -f "$FINAL_RESPONSE_CLASSIFIER_DIR/model.onnx" ]] || die "final-response classifier model.onnx missing in $FINAL_RESPONSE_CLASSIFIER_DIR"
        ;;
    esac
  fi

  phase "Build classifier-enabled binaries"
  log "Building forge-guardrails-proxy and forge-eval with classifier feature"
  (cd "$REPO_ROOT" && cargo build --features classifier --bin forge-guardrails-proxy --bin forge-eval)
}

start_proxy() {
  local budget classifier_log compression_log label
  budget="$1"
  label="$2"
  CURRENT_PROXY_LOG="$OUTPUT_DIR/proxy_${label}.log"
  classifier_log="$OUTPUT_DIR/proxy_classifier_${label}.jsonl"
  compression_log="$OUTPUT_DIR/proxy_tool_output_compression_${label}.jsonl"

  phase "Start proxy: $label"
  log "Budget tokens: $budget"
  log "Upstream backend: $UPSTREAM_BACKEND"
  log "Proxy backend mode: $PROXY_BACKEND_MODE"
  log "Proxy log: $CURRENT_PROXY_LOG"
  local env_args=(
    "FORGE_PROXY_PORT=$PROXY_PORT"
    "FORGE_BACKEND_PORT=$BACKEND_PORT"
  )
  local proxy_args=(
    --mode "$PROXY_BACKEND_MODE"
    --budget-mode manual
    --budget-tokens "$budget"
  )
  if classifier_feature_enabled; then
    env_args+=("FORGE_PROXY_BIN=$REPO_ROOT/target/debug/forge-guardrails-proxy")
    env_args+=("FORGE_CLASSIFIER_LOG=$classifier_log")
  fi
  if classifier_enabled; then
    proxy_args+=(
      --classifier-dir "$CLASSIFIER_DIR"
      --classifier-mode "$CLASSIFIER_MODE"
      --classifier-model "$CLASSIFIER_MODEL"
    )
    log "Classifier: enabled, mode=$CLASSIFIER_MODE, model=$CLASSIFIER_MODEL"
  else
    log "Classifier: disabled"
  fi
  if final_response_classifier_enabled; then
    proxy_args+=(
      --final-response-classifier-dir "$FINAL_RESPONSE_CLASSIFIER_DIR"
      --final-response-classifier-mode "$FINAL_RESPONSE_CLASSIFIER_MODE"
      --final-response-classifier-model "$FINAL_RESPONSE_CLASSIFIER_MODEL"
    )
    log "Final-response classifier: enabled, mode=$FINAL_RESPONSE_CLASSIFIER_MODE, model=$FINAL_RESPONSE_CLASSIFIER_MODEL"
  else
    log "Final-response classifier: disabled"
  fi
  proxy_args+=(--tool-output-compression "$TOOL_OUTPUT_COMPRESSION")
  proxy_args+=(--tool-output-compression-method "$TOOL_OUTPUT_COMPRESSION_METHOD")
  log "Tool-output compression: mode=$TOOL_OUTPUT_COMPRESSION, method=$TOOL_OUTPUT_COMPRESSION_METHOD"
  if [[ "$TOOL_OUTPUT_COMPRESSION" != "disabled" ]]; then
    env_args+=("FORGE_TOOL_OUTPUT_COMPRESSION_LOG=$compression_log")
    log "Tool-output compression JSONL: $compression_log"
  fi
  if classifier_feature_enabled; then
    log "Classifier JSONL: $classifier_log"
  fi
  case "$UPSTREAM_BACKEND" in
    llamaserver)
      env "${env_args[@]}" "$SCRIPT_DIR/start_llamaserver_proxy.sh" "$GGUF" "${proxy_args[@]}" \
        >"$CURRENT_PROXY_LOG" 2>&1 &
      ;;
    openrouter)
      log "OpenRouter base URL: $OPENROUTER_BASE_URL"
      log "OpenRouter API key source: $(openrouter_api_key_source)"
      proxy_args+=(
        --backend-url "$OPENROUTER_BASE_URL"
        --model "$MODEL"
        --port "$PROXY_PORT"
      )
      env_args+=("OPENAI_API_KEY=${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}")
      if classifier_feature_enabled; then
        env "${env_args[@]}" "$REPO_ROOT/target/debug/forge-guardrails-proxy" "${proxy_args[@]}" \
          >"$CURRENT_PROXY_LOG" 2>&1 &
      else
        (
          cd "$REPO_ROOT"
          env "${env_args[@]}" cargo run --bin forge-guardrails-proxy -- "${proxy_args[@]}"
        ) >"$CURRENT_PROXY_LOG" 2>&1 &
      fi
      ;;
  esac
  PROXY_PID="$!"
  wait_for_health "$PROXY_PID"
  start_resource_sampler "$label"
}

run_rust_smoke() {
  local count elapsed eval_pid expected_rows initial_rows new_rows output started status
  output="$OUTPUT_DIR/rust_smoke.jsonl"
  local scenarios
  scenarios=($(scenario_names smoke))
  expected_rows=$(( RUNS * ${#scenarios[@]} ))
  if [[ -f "$output" ]]; then
    initial_rows="$(wc -l <"$output" | tr -d '[:space:]')"
  else
    initial_rows=0
  fi

  phase "Rust smoke eval"
  log "Scenarios: ${#scenarios[@]}, runs: $RUNS, expected rows: $expected_rows"
  log "Output: $output"
  if classifier_enabled; then
    log "Classifier capture: enabled"
  else
    log "Classifier capture: disabled"
  fi
  if final_response_classifier_enabled; then
    log "Final-response classifier capture: enabled"
  else
    log "Final-response classifier capture: disabled"
  fi
  if [[ "$initial_rows" -gt 0 ]]; then
    log "Rust smoke output already has ${initial_rows} rows; progress reports new rows"
  fi
  local cmd=(cargo run)
  if classifier_feature_enabled; then
    cmd+=(--features classifier)
  fi
  cmd+=(--bin forge-eval --)
  cmd+=(
    --backend openai-proxy
    --base-url "http://127.0.0.1:${PROXY_PORT}/v1"
    --model "$MODEL"
    --runs "$RUNS"
    --num-ctx "$DEFAULT_BUDGET_TOKENS"
    --scenario "${scenarios[@]}"
    --output "$output"
  )
  if [[ "$STREAM" == "1" ]]; then
    cmd+=(--stream)
  fi
  if classifier_enabled; then
    cmd+=(
      --classifier-dir "$CLASSIFIER_DIR"
      --classifier-mode "$CLASSIFIER_MODE"
      --classifier-model "$CLASSIFIER_MODEL"
    )
  fi
  if final_response_classifier_enabled; then
    cmd+=(
      --final-response-classifier-dir "$FINAL_RESPONSE_CLASSIFIER_DIR"
      --final-response-classifier-mode "$FINAL_RESPONSE_CLASSIFIER_MODE"
      --final-response-classifier-model "$FINAL_RESPONSE_CLASSIFIER_MODEL"
    )
  fi

  (cd "$REPO_ROOT" && "${cmd[@]}") &
  eval_pid="$!"
  started="$(date +%s)"
  while kill -0 "$eval_pid" 2>/dev/null; do
    if [[ -f "$output" ]]; then
      count="$(wc -l <"$output" | tr -d '[:space:]')"
    else
      count=0
    fi
    new_rows=$(( count - initial_rows ))
    elapsed=$(( $(date +%s) - started ))
    log "Rust smoke progress: ${new_rows}/${expected_rows} new rows, ${elapsed}s elapsed"
    sleep 10
  done

  set +e
  wait "$eval_pid"
  status="$?"
  set -e

  if [[ -f "$output" ]]; then
    count="$(wc -l <"$output" | tr -d '[:space:]')"
  else
    count=0
  fi
  new_rows=$(( count - initial_rows ))
  elapsed=$(( $(date +%s) - started ))
  log "Rust smoke complete: ${new_rows}/${expected_rows} new rows, ${elapsed}s elapsed"
  return "$status"
}

run_python_oracle() {
  local budget expected_rows output
  budget="$1"
  shift
  output="$OUTPUT_DIR/python_oracle.jsonl"

  local scenarios=("$@")
  [[ "${#scenarios[@]}" -gt 0 ]] || die "no Python oracle scenarios selected"
  expected_rows=$(( RUNS * ${#scenarios[@]} ))

  phase "Python oracle eval"
  log "Budget tokens: $budget"
  log "Scenarios: ${#scenarios[@]}, runs: $RUNS, expected rows: $expected_rows"
  log "Output: $output"
  local cmd=(
    scripts/eval_openai_proxy.py
    --base-url "http://127.0.0.1:${PROXY_PORT}/v1"
    --model "$MODEL"
    --runs "$RUNS"
    --scenario "${scenarios[@]}"
    --budget-tokens "$budget"
    --backend-label "$UPSTREAM_BACKEND"
    --mode-label "$PROXY_BACKEND_MODE"
    --proxy-backend-mode "$PROXY_BACKEND_MODE"
    --eval-target-backend openai-proxy
    --output "$output"
  )
  if [[ "$STREAM" == "1" ]]; then
    cmd+=(--stream)
  fi
  run_python_script "${cmd[@]}"
}

run_python_report() {
  local output report summary
  output="$OUTPUT_DIR/python_oracle.jsonl"
  report="$OUTPUT_DIR/python_report.txt"
  summary="$OUTPUT_DIR/proxy_summary.txt"
  if [[ ! -f "$output" ]]; then
    log "Skipping Python report; oracle output not found: $output"
    return 0
  fi

  phase "Python reports"
  log "Generating Python report -> $report"
  if command -v uv >/dev/null 2>&1; then
    (cd "$REPO_ROOT/forge" && env -u VIRTUAL_ENV uv run python -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  elif [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]]; then
    (cd "$REPO_ROOT/forge" && "$REPO_ROOT/forge/.venv/bin/python" -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  else
    (cd "$REPO_ROOT/forge" && "$PYTHON_BIN" -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  fi

  log "Generating proxy summary -> $summary"
  local summary_cmd classifier_logs
  summary_cmd=(scripts/summarize_proxy_eval.py "$output")
  classifier_logs=("$OUTPUT_DIR"/proxy_classifier_*.jsonl)
  if [[ -e "${classifier_logs[0]:-}" ]]; then
    summary_cmd+=(--classifier-jsonl "${classifier_logs[@]}")
  fi
  run_python_script "${summary_cmd[@]}" 2>&1 | tee "$summary"
}

run_published_compare() {
  local output compare_report published_model
  output="$OUTPUT_DIR/python_oracle.jsonl"
  compare_report="$OUTPUT_DIR/published_compare.txt"
  published_model="${PUBLISHED_MODEL:-$MODEL}"
  if [[ ! -f "$output" ]]; then
    log "Skipping published compare; oracle output not found: $output"
    return 0
  fi

  phase "Published baseline compare"
  log "Comparing against published baseline -> $compare_report"
  local cmd=(
    scripts/compare_published_eval.py "$output"
    --model "$published_model"
    --backend-mode "$PUBLISHED_BACKEND_MODE"
    --local-model "$MODEL"
  )
  if [[ "$FORCE_PUBLISHED_COMPARE" == "1" ]]; then
    cmd+=(--force-proxy-compare)
  fi
  set +e
  run_python_script "${cmd[@]}" 2>&1 | tee "$compare_report"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

write_metadata() {
  local classifier_enabled_value final_response_classifier_enabled_value metadata
  local api_key_source managed_backend_url openrouter_base_url_metadata upstream_backend_url
  metadata="$OUTPUT_DIR/local_eval_metadata.txt"
  if classifier_enabled; then
    classifier_enabled_value=1
  else
    classifier_enabled_value=0
  fi
  if final_response_classifier_enabled; then
    final_response_classifier_enabled_value=1
  else
    final_response_classifier_enabled_value=0
  fi
  if [[ "$UPSTREAM_BACKEND" == "openrouter" ]]; then
    api_key_source="$(openrouter_api_key_source)"
    managed_backend_url=""
    openrouter_base_url_metadata="$OPENROUTER_BASE_URL"
    upstream_backend_url="$OPENROUTER_BASE_URL"
  else
    api_key_source="not_used"
    managed_backend_url="http://127.0.0.1:${BACKEND_PORT}/v1"
    openrouter_base_url_metadata=""
    upstream_backend_url="$managed_backend_url"
  fi
  cat >"$metadata" <<EOF
suite=$SUITE
runs=$RUNS
stream=$STREAM
gguf=$GGUF
model=$MODEL
upstream_backend=$UPSTREAM_BACKEND
upstream_backend_url=$upstream_backend_url
openrouter_base_url=$openrouter_base_url_metadata
openrouter_api_key_source=$api_key_source
published_model=${PUBLISHED_MODEL:-$MODEL}
published_backend_mode=$PUBLISHED_BACKEND_MODE
proxy_backend_mode=$PROXY_BACKEND_MODE
proxy_url=http://127.0.0.1:${PROXY_PORT}/v1
managed_backend_url=$managed_backend_url
managed_backend=$UPSTREAM_BACKEND
eval_target_backend=openai-proxy
mode=proxy
recommended_sampling=1
force_published_compare=$FORCE_PUBLISHED_COMPARE
auto_skip_published_compare=$AUTO_SKIP_PUBLISHED_COMPARE
normal_budget_tokens=$DEFAULT_BUDGET_TOKENS
compaction_chain_p1_budget_tokens=3600
compaction_chain_p2_budget_tokens=2200
compaction_chain_p3_budget_tokens=1536
include_compaction_chain=$INCLUDE_COMPACTION_CHAIN
classifier_enabled=$classifier_enabled_value
classify_shortcut=$CLASSIFY
classifier_dir=$CLASSIFIER_DIR
classifier_mode=$CLASSIFIER_MODE
classifier_model=$CLASSIFIER_MODEL
final_response_classifier_enabled=$final_response_classifier_enabled_value
verify_final_response_shortcut=$VERIFY_FINAL_RESPONSE
final_response_classifier_dir=$FINAL_RESPONSE_CLASSIFIER_DIR
final_response_classifier_mode=$FINAL_RESPONSE_CLASSIFIER_MODE
final_response_classifier_model=$FINAL_RESPONSE_CLASSIFIER_MODEL
classifier_jsonl_pattern=$OUTPUT_DIR/proxy_classifier_*.jsonl
resource_baseline=$RESOURCE_BASELINE
resource_interval=$RESOURCE_INTERVAL
resource_samples_pattern=$OUTPUT_DIR/resource_samples_*.jsonl
resource_summary_pattern=$OUTPUT_DIR/resource_summary_*.json
resource_report=$OUTPUT_DIR/resource_baseline_report.txt
tool_output_compression=$TOOL_OUTPUT_COMPRESSION
tool_output_compression_method=$TOOL_OUTPUT_COMPRESSION_METHOD
tool_output_compression_jsonl_pattern=$OUTPUT_DIR/proxy_tool_output_compression_*.jsonl
EOF
  log "Metadata: $metadata"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --runs)
      RUNS="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --gguf)
      GGUF="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --model)
      MODEL="$(next_arg "$1" "${2:-}")"
      MODEL_EXPLICIT=1
      shift 2
      ;;
    --upstream-backend)
      UPSTREAM_BACKEND="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --openrouter-base-url)
      OPENROUTER_BASE_URL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --published-model)
      PUBLISHED_MODEL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --published-mode)
      PUBLISHED_BACKEND_MODE="$(next_arg "$1" "${2:-}")"
      PUBLISHED_BACKEND_MODE_SET=1
      shift 2
      ;;
    --proxy-backend-mode)
      PROXY_BACKEND_MODE="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --skip-published-compare)
      SKIP_PUBLISHED_COMPARE=1
      shift
      ;;
    --force-published-compare)
      FORCE_PUBLISHED_COMPARE=1
      shift
      ;;
    --include-compaction-chain)
      INCLUDE_COMPACTION_CHAIN=1
      shift
      ;;
    --proxy-port)
      PROXY_PORT="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --health-timeout)
      HEALTH_TIMEOUT="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --no-stream)
      STREAM=0
      shift
      ;;
    --classifier-dir)
      CLASSIFIER_DIR="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --classifier-mode)
      CLASSIFIER_MODE="$(next_arg "$1" "${2:-}")"
      CLASSIFIER_MODE_EXPLICIT=1
      shift 2
      ;;
    --classifier-model)
      CLASSIFIER_MODEL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --classify)
      CLASSIFY=1
      if [[ "$CLASSIFIER_MODE_EXPLICIT" != "1" ]]; then
        CLASSIFIER_MODE="advisory"
      fi
      shift
      ;;
    --download-classifier)
      DOWNLOAD_CLASSIFIER=1
      shift
      ;;
    --final-response-classifier-dir)
      FINAL_RESPONSE_CLASSIFIER_DIR="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --final-response-classifier-mode)
      FINAL_RESPONSE_CLASSIFIER_MODE="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --final-response-classifier-model)
      FINAL_RESPONSE_CLASSIFIER_MODEL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --verify-final-response)
      VERIFY_FINAL_RESPONSE=1
      DOWNLOAD_FINAL_RESPONSE_CLASSIFIER=1
      shift
      ;;
    --download-final-response-classifier)
      DOWNLOAD_FINAL_RESPONSE_CLASSIFIER=1
      shift
      ;;
    --resource-baseline)
      RESOURCE_BASELINE=1
      shift
      ;;
    --resource-interval)
      RESOURCE_INTERVAL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --tool-output-compression)
      TOOL_OUTPUT_COMPRESSION="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --tool-output-compression-method)
      TOOL_OUTPUT_COMPRESSION_METHOD="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$SUITE" in
  smoke|release)
    ;;
  *)
    die "--suite must be smoke or release"
    ;;
esac
case "$PROXY_BACKEND_MODE" in
  native|prompt)
    ;;
  *)
    die "--proxy-backend-mode must be native or prompt"
    ;;
esac
case "$UPSTREAM_BACKEND" in
  llamaserver|openrouter)
    ;;
  *)
    die "--upstream-backend must be llamaserver or openrouter"
    ;;
esac
if [[ "$PUBLISHED_BACKEND_MODE_SET" != "1" ]]; then
  if [[ "$PROXY_BACKEND_MODE" == "native" ]]; then
    PUBLISHED_BACKEND_MODE="LS/N"
  else
    PUBLISHED_BACKEND_MODE="LS/P"
  fi
fi
case "$PUBLISHED_BACKEND_MODE" in
  LS/N|LS/P)
    ;;
  *)
    die "--published-mode must be LS/N or LS/P"
    ;;
esac
case "$CLASSIFIER_MODE" in
  disabled|shadow|advisory|enforce)
    ;;
  *)
    die "--classifier-mode must be disabled, shadow, advisory, or enforce"
    ;;
esac
case "$CLASSIFIER_MODEL" in
  quantized|full)
    ;;
  *)
    die "--classifier-model must be quantized or full"
    ;;
esac
if [[ "$CLASSIFY" == "1" && "$CLASSIFIER_MODE" == "disabled" ]]; then
  die "--classify cannot be used with --classifier-mode disabled"
fi
if [[ "$CLASSIFY" == "1" && "$DOWNLOAD_CLASSIFIER" == "1" ]]; then
  die "--classify uses the user cache; use --download-classifier for target/ artifacts"
fi
case "$FINAL_RESPONSE_CLASSIFIER_MODE" in
  disabled|shadow|advisory|enforce)
    ;;
  *)
    die "--final-response-classifier-mode must be disabled, shadow, advisory, or enforce"
    ;;
esac
case "$FINAL_RESPONSE_CLASSIFIER_MODEL" in
  quantized|full)
    ;;
  *)
    die "--final-response-classifier-model must be quantized or full"
    ;;
esac
case "$TOOL_OUTPUT_COMPRESSION" in
  disabled|safe|standard|aggressive)
    ;;
  *)
    die "--tool-output-compression must be disabled, safe, standard, or aggressive"
    ;;
esac
case "$TOOL_OUTPUT_COMPRESSION_METHOD" in
  lzw|repair|auto)
    ;;
  *)
    die "--tool-output-compression-method must be lzw, repair, or auto"
    ;;
esac
if [[ "$VERIFY_FINAL_RESPONSE" == "1" && "$FINAL_RESPONSE_CLASSIFIER_MODE" == "disabled" ]]; then
  die "--verify-final-response cannot be used with --final-response-classifier-mode disabled"
fi
if [[ "$UPSTREAM_BACKEND" == "openrouter" ]]; then
  [[ "$MODEL_EXPLICIT" == "1" ]] || die "--upstream-backend openrouter requires explicit --model"
  [[ -n "${OPENROUTER_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" ]] || die "OPENROUTER_API_KEY or OPENAI_API_KEY is required for --upstream-backend openrouter"
  if [[ "$SUITE" == "release" && "$FORCE_PUBLISHED_COMPARE" != "1" && "$SKIP_PUBLISHED_COMPARE" != "1" ]]; then
    SKIP_PUBLISHED_COMPARE=1
    AUTO_SKIP_PUBLISHED_COMPARE=1
  fi
fi
valid_positive_int "$RUNS" || die "--runs must be a positive integer"
valid_positive_int "$PROXY_PORT" || die "--proxy-port must be a positive integer"
valid_positive_int "$BACKEND_PORT" || die "--backend-port must be a positive integer"
valid_positive_int "$HEALTH_TIMEOUT" || die "--health-timeout must be a positive integer"
valid_positive_decimal "$RESOURCE_INTERVAL" || die "--resource-interval must be a positive number"
[[ "$PROXY_PORT" != "$BACKEND_PORT" ]] || die "proxy and backend ports must differ"
command -v curl >/dev/null 2>&1 || die "curl is required for health checks"
have_python_runner || die "python runner not found; install uv, create forge/.venv, or set PYTHON"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/target/local-eval/$(date +%Y%m%dT%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

if [[ "$CLASSIFY" == "1" ]]; then
  download_classifier_shortcut_artifact
elif [[ "$DOWNLOAD_CLASSIFIER" == "1" ]]; then
  download_classifier_artifacts
fi
if [[ "$DOWNLOAD_FINAL_RESPONSE_CLASSIFIER" == "1" ]]; then
  download_final_response_classifier_artifacts
fi
prepare_classifier_binaries

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

phase "Eval setup"
log "Output directory: $OUTPUT_DIR"
log "Suite: $SUITE, runs: $RUNS, stream: $STREAM, upstream_backend: $UPSTREAM_BACKEND, proxy_backend_mode: $PROXY_BACKEND_MODE"
if [[ "$UPSTREAM_BACKEND" == "openrouter" ]]; then
  log "OpenRouter base URL: $OPENROUTER_BASE_URL"
  log "OpenRouter API key source: $(openrouter_api_key_source)"
fi
log "Tool-output compression: mode=$TOOL_OUTPUT_COMPRESSION, method=$TOOL_OUTPUT_COMPRESSION_METHOD"
if [[ "$RESOURCE_BASELINE" == "1" ]]; then
  log "Resource baseline: enabled, interval=${RESOURCE_INTERVAL}s"
else
  log "Resource baseline: disabled"
fi
if classifier_enabled; then
  log "Classifier: dir=$CLASSIFIER_DIR, mode=$CLASSIFIER_MODE, model=$CLASSIFIER_MODEL"
else
  log "Classifier: disabled"
fi
if final_response_classifier_enabled; then
  log "Final-response classifier: dir=$FINAL_RESPONSE_CLASSIFIER_DIR, mode=$FINAL_RESPONSE_CLASSIFIER_MODE, model=$FINAL_RESPONSE_CLASSIFIER_MODEL"
else
  log "Final-response classifier: disabled"
fi
write_metadata

start_proxy "$DEFAULT_BUDGET_TOKENS" "budget_${DEFAULT_BUDGET_TOKENS}"
run_rust_smoke

if [[ "$SUITE" == "smoke" ]]; then
  smoke_scenarios=($(scenario_names smoke))
  run_python_oracle "$DEFAULT_BUDGET_TOKENS" "${smoke_scenarios[@]}"
else
  normal_scenarios=($(scenario_names release-normal))
  run_python_oracle "$DEFAULT_BUDGET_TOKENS" "${normal_scenarios[@]}"

  if [[ "$INCLUDE_COMPACTION_CHAIN" == "1" ]]; then
    stop_proxy
    start_proxy "3600" "compaction_chain_p1"
    run_python_oracle "3600" "compaction_chain_p1"

    stop_proxy
    start_proxy "2200" "compaction_chain_p2"
    run_python_oracle "2200" "compaction_chain_p2"

    stop_proxy
    start_proxy "1536" "compaction_chain_p3"
    run_python_oracle "1536" "compaction_chain_p3"
  fi
fi

stop_resource_sampler
run_resource_report
run_python_report
if [[ "$SUITE" == "release" && "$SKIP_PUBLISHED_COMPARE" != "1" ]]; then
  run_published_compare
elif [[ "$SUITE" == "release" ]]; then
  if [[ "$AUTO_SKIP_PUBLISHED_COMPARE" == "1" ]]; then
    log "Skipping published compare because OpenRouter runs do not match local published baselines; pass --force-published-compare to override"
  else
    log "Skipping published compare because --skip-published-compare was set"
  fi
fi

phase "Complete"
log "Local eval complete"
log "Rust smoke:    $OUTPUT_DIR/rust_smoke.jsonl"
log "Python oracle: $OUTPUT_DIR/python_oracle.jsonl"
log "Python report: $OUTPUT_DIR/python_report.txt"
if [[ "$RESOURCE_BASELINE" == "1" ]]; then
  log "Resource rpt:  $OUTPUT_DIR/resource_baseline_report.txt"
fi
if [[ "$SUITE" == "release" && "$SKIP_PUBLISHED_COMPARE" != "1" ]]; then
  log "Published cmp: $OUTPUT_DIR/published_compare.txt"
fi
