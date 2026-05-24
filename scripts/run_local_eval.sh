#!/usr/bin/env bash
set -euo pipefail

DEFAULT_GGUF="mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf"
DEFAULT_PROXY_PORT="8081"
DEFAULT_BACKEND_PORT="8080"
DEFAULT_BUDGET_TOKENS="8192"
DEFAULT_MODEL="Ministral-3-8B-Instruct-2512-Q8_0"
DEFAULT_PUBLISHED_BACKEND_MODE="LS/N"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

SUITE="smoke"
RUNS="1"
GGUF="$DEFAULT_GGUF"
MODEL="$DEFAULT_MODEL"
PUBLISHED_MODEL=""
PUBLISHED_BACKEND_MODE="$DEFAULT_PUBLISHED_BACKEND_MODE"
SKIP_PUBLISHED_COMPARE=0
INCLUDE_COMPACTION_CHAIN=0
PROXY_PORT="${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
BACKEND_PORT="${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"
HEALTH_TIMEOUT="180"
STREAM=1
OUTPUT_DIR=""
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
PROXY_PID=""
CURRENT_PROXY_LOG=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs local Forge evals against the Rust proxy using managed llama-server.

Options:
  --suite smoke|release     Eval suite to run (default: smoke)
  --runs N                  Runs per scenario (default: 1)
  --gguf PATH               GGUF path (default: $DEFAULT_GGUF)
  --output-dir DIR          Output directory (default: target/local-eval/<timestamp>)
  --model MODEL             Model name sent to the proxy (default: $DEFAULT_MODEL)
  --published-model MODEL   Published baseline model (default: --model)
  --published-mode LS/N|LS/P Published baseline row (default: $DEFAULT_PUBLISHED_BACKEND_MODE)
  --skip-published-compare  Do not compare release results to published results
  --include-compaction-chain Also run compaction-chain scenarios after published scenarios
  --proxy-port PORT         Proxy port (default: $DEFAULT_PROXY_PORT)
  --backend-port PORT       Managed llama-server port (default: $DEFAULT_BACKEND_PORT)
  --health-timeout SECONDS  Seconds to wait for /health (default: 180)
  --no-stream               Disable streaming eval requests
  -h, --help                Show this help

Examples:
  $(basename "$0") --suite smoke --runs 1
  $(basename "$0") --suite release --runs 10
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '%s\n' "$*" >&2
}

valid_positive_int() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
  esac
  (( 10#$1 > 0 ))
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
  stop_proxy
  exit "$status"
}

stop_proxy() {
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    log "Stopping proxy pid $PROXY_PID"
    kill -TERM "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  PROXY_PID=""
}

wait_for_health() {
  local url pid elapsed
  url="http://127.0.0.1:${PROXY_PORT}/health"
  pid="$1"
  elapsed=0

  while (( elapsed < HEALTH_TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 80 "$CURRENT_PROXY_LOG" >&2 || true
      die "proxy exited before becoming healthy"
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "Proxy healthy at $url"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  tail -n 80 "$CURRENT_PROXY_LOG" >&2 || true
  die "proxy did not become healthy within ${HEALTH_TIMEOUT}s"
}

start_proxy() {
  local budget label
  budget="$1"
  label="$2"
  CURRENT_PROXY_LOG="$OUTPUT_DIR/proxy_${label}.log"

  log "Starting proxy with budget_tokens=$budget"
  FORGE_PROXY_PORT="$PROXY_PORT" \
    FORGE_BACKEND_PORT="$BACKEND_PORT" \
    "$SCRIPT_DIR/start_llamaserver_proxy.sh" "$GGUF" \
      --budget-mode manual \
      --budget-tokens "$budget" \
      >"$CURRENT_PROXY_LOG" 2>&1 &
  PROXY_PID="$!"
  wait_for_health "$PROXY_PID"
}

run_rust_smoke() {
  local output
  output="$OUTPUT_DIR/rust_smoke.jsonl"
  local scenarios
  scenarios=($(scenario_names smoke))

  log "Running Rust smoke eval -> $output"
  local cmd=(
    cargo run --bin forge-eval --
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
  (cd "$REPO_ROOT" && "${cmd[@]}")
}

run_python_oracle() {
  local budget output
  budget="$1"
  shift
  output="$OUTPUT_DIR/python_oracle.jsonl"

  local scenarios=("$@")
  [[ "${#scenarios[@]}" -gt 0 ]] || die "no Python oracle scenarios selected"

  log "Running Python oracle with budget_tokens=$budget -> $output"
  local cmd=(
    scripts/eval_openai_proxy.py
    --base-url "http://127.0.0.1:${PROXY_PORT}/v1"
    --model "$MODEL"
    --runs "$RUNS"
    --scenario "${scenarios[@]}"
    --budget-tokens "$budget"
    --output "$output"
  )
  if [[ "$STREAM" == "1" ]]; then
    cmd+=(--stream)
  fi
  run_python_script "${cmd[@]}"
}

run_python_report() {
  local output report
  output="$OUTPUT_DIR/python_oracle.jsonl"
  report="$OUTPUT_DIR/python_report.txt"
  [[ -f "$output" ]] || return 0

  log "Generating Python report -> $report"
  if command -v uv >/dev/null 2>&1; then
    (cd "$REPO_ROOT/forge" && env -u VIRTUAL_ENV uv run python -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  elif [[ -x "$REPO_ROOT/forge/.venv/bin/python" ]]; then
    (cd "$REPO_ROOT/forge" && "$REPO_ROOT/forge/.venv/bin/python" -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  else
    (cd "$REPO_ROOT/forge" && "$PYTHON_BIN" -m tests.eval.report "$output" --include-partial 2>&1) | tee "$report"
  fi
}

run_published_compare() {
  local output compare_report published_model
  output="$OUTPUT_DIR/python_oracle.jsonl"
  compare_report="$OUTPUT_DIR/published_compare.txt"
  published_model="${PUBLISHED_MODEL:-$MODEL}"
  [[ -f "$output" ]] || return 0

  log "Comparing against published baseline -> $compare_report"
  local cmd=(
    scripts/compare_published_eval.py "$output"
    --model "$published_model"
    --backend-mode "$PUBLISHED_BACKEND_MODE"
    --local-model "$MODEL"
  )
  set +e
  run_python_script "${cmd[@]}" 2>&1 | tee "$compare_report"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

write_metadata() {
  local metadata
  metadata="$OUTPUT_DIR/local_eval_metadata.txt"
  cat >"$metadata" <<EOF
suite=$SUITE
runs=$RUNS
stream=$STREAM
gguf=$GGUF
model=$MODEL
published_model=${PUBLISHED_MODEL:-$MODEL}
published_backend_mode=$PUBLISHED_BACKEND_MODE
proxy_url=http://127.0.0.1:${PROXY_PORT}/v1
managed_backend_url=http://127.0.0.1:${BACKEND_PORT}/v1
managed_backend=llamaserver
eval_target_backend=openai-proxy
normal_budget_tokens=$DEFAULT_BUDGET_TOKENS
compaction_chain_p1_budget_tokens=3600
compaction_chain_p2_budget_tokens=2200
compaction_chain_p3_budget_tokens=1536
include_compaction_chain=$INCLUDE_COMPACTION_CHAIN
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
      shift 2
      ;;
    --published-model)
      PUBLISHED_MODEL="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --published-mode)
      PUBLISHED_BACKEND_MODE="$(next_arg "$1" "${2:-}")"
      shift 2
      ;;
    --skip-published-compare)
      SKIP_PUBLISHED_COMPARE=1
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
case "$PUBLISHED_BACKEND_MODE" in
  LS/N|LS/P)
    ;;
  *)
    die "--published-mode must be LS/N or LS/P"
    ;;
esac
valid_positive_int "$RUNS" || die "--runs must be a positive integer"
valid_positive_int "$PROXY_PORT" || die "--proxy-port must be a positive integer"
valid_positive_int "$BACKEND_PORT" || die "--backend-port must be a positive integer"
valid_positive_int "$HEALTH_TIMEOUT" || die "--health-timeout must be a positive integer"
[[ "$PROXY_PORT" != "$BACKEND_PORT" ]] || die "proxy and backend ports must differ"
command -v curl >/dev/null 2>&1 || die "curl is required for health checks"
have_python_runner || die "python runner not found; install uv, create forge/.venv, or set PYTHON"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/target/local-eval/$(date +%Y%m%dT%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

log "Output directory: $OUTPUT_DIR"
log "Suite: $SUITE, runs: $RUNS, stream: $STREAM"
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

run_python_report
if [[ "$SUITE" == "release" && "$SKIP_PUBLISHED_COMPARE" != "1" ]]; then
  run_published_compare
fi

log "Local eval complete"
log "Rust smoke:    $OUTPUT_DIR/rust_smoke.jsonl"
log "Python oracle: $OUTPUT_DIR/python_oracle.jsonl"
log "Python report: $OUTPUT_DIR/python_report.txt"
if [[ "$SUITE" == "release" && "$SKIP_PUBLISHED_COMPARE" != "1" ]]; then
  log "Published cmp: $OUTPUT_DIR/published_compare.txt"
fi
