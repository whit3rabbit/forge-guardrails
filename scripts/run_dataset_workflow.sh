#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OUT_DIR="target/dataset/run"
DEFAULT_PROXY_PORT="8081"
DEFAULT_BACKEND_PORT="8080"
DEFAULT_MODEL="test-model"
DEFAULT_PROVIDER="auto"
DEFAULT_VERIFIER_PROVIDER="same"
DEFAULT_MAX_TURNS="4"
DEFAULT_RUNS="1"
DEFAULT_DOMAINS="repo_docs,shopping,calendar,support"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

OUT_DIR="${FORGE_DATASET_OUT_DIR:-$DEFAULT_OUT_DIR}"
PROXY_PORT="${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
BACKEND_PORT="${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"
MODEL="${FORGE_DATASET_MODEL:-$DEFAULT_MODEL}"
PROVIDER="${FORGE_DATASET_REVIEW_PROVIDER:-$DEFAULT_PROVIDER}"
VERIFIER_PROVIDER="${FORGE_DATASET_VERIFIER_PROVIDER:-$DEFAULT_VERIFIER_PROVIDER}"
MAX_TURNS="${FORGE_DATASET_MAX_TURNS:-$DEFAULT_MAX_TURNS}"
RUNS="${FORGE_DATASET_RUNS:-$DEFAULT_RUNS}"
DOMAINS="${FORGE_DATASET_DOMAINS:-$DEFAULT_DOMAINS}"
GGUF_PATH_ARG="${GGUF_PATH:-${FORGE_GGUF:-${GGUF:-}}}"
CAPTURE_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [GGUF_PATH] [extra proxy args...]

Starts managed llama-server + forge proxy, generates tool-call prompts, captures
real model tool calls against harmless stub tools, then reviews rows with
MiniMax/OpenRouter unless --capture-only or --provider none is used.

Options:
  --out-dir DIR                 Output directory (default: $DEFAULT_OUT_DIR)
  --model MODEL                 Chat model name sent to the proxy (default: $DEFAULT_MODEL)
  --provider auto|minimax|openrouter|none
                                Review provider (default: auto)
  --verifier-provider same|auto|minimax|openrouter
                                Verifier provider (default: same)
  --max-turns N                 Capture turns per scenario (default: $DEFAULT_MAX_TURNS)
  --runs N                      Scenario repetitions (default: $DEFAULT_RUNS)
  --domains CSV                 Dataset domains (default: $DEFAULT_DOMAINS; also supports forge_eval)
  --proxy-port PORT             Forge proxy port (default: $DEFAULT_PROXY_PORT)
  --backend-port PORT           Managed llama-server port (default: $DEFAULT_BACKEND_PORT)
  --capture-only                Skip review
  -h, --help                    Show this help

Environment:
  MINIMAX_API_KEY, OPENROUTER_API_KEY
  GENERATETD_MINIMAX_MODEL, GENERATETD_OPENROUTER_MODEL
  FORGE_DATASET_* equivalents for options above

Examples:
  $(basename "$0") /models/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf
  OPENROUTER_API_KEY=... $(basename "$0") --provider openrouter --out-dir target/dataset/openrouter
  MINIMAX_API_KEY=... $(basename "$0") --provider minimax --verifier-provider openrouter
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

valid_port() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
  esac
  (( 10#$1 >= 1 && 10#$1 <= 65535 ))
}

PROXY_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --out-dir)
      OUT_DIR="${2:?--out-dir requires a value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --provider)
      PROVIDER="${2:?--provider requires a value}"
      shift 2
      ;;
    --verifier-provider)
      VERIFIER_PROVIDER="${2:?--verifier-provider requires a value}"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="${2:?--max-turns requires a value}"
      shift 2
      ;;
    --runs)
      RUNS="${2:?--runs requires a value}"
      shift 2
      ;;
    --domains)
      DOMAINS="${2:?--domains requires a value}"
      shift 2
      ;;
    --proxy-port)
      PROXY_PORT="${2:?--proxy-port requires a value}"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="${2:?--backend-port requires a value}"
      shift 2
      ;;
    --capture-only)
      CAPTURE_ONLY=1
      shift
      ;;
    --)
      shift
      PROXY_EXTRA_ARGS+=("$@")
      break
      ;;
    --*)
      PROXY_EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$GGUF_PATH_ARG" ]]; then
        GGUF_PATH_ARG="$1"
      else
        PROXY_EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

valid_port "$PROXY_PORT" || die "invalid proxy port: $PROXY_PORT"
valid_port "$BACKEND_PORT" || die "invalid backend port: $BACKEND_PORT"
[[ "$PROVIDER" =~ ^(auto|minimax|openrouter|none)$ ]] || die "--provider must be auto|minimax|openrouter|none"
[[ "$VERIFIER_PROVIDER" =~ ^(same|auto|minimax|openrouter)$ ]] || die "--verifier-provider must be same|auto|minimax|openrouter"
[[ "$MAX_TURNS" =~ ^[0-9]+$ ]] && (( MAX_TURNS > 0 )) || die "--max-turns must be a positive integer"
[[ "$RUNS" =~ ^[0-9]+$ ]] && (( RUNS > 0 )) || die "--runs must be a positive integer"

mkdir -p "$OUT_DIR"
PROMPTS_JSONL="$OUT_DIR/tool_prompts.jsonl"
CAPTURE_JSONL="$OUT_DIR/capture.jsonl"
PROXY_CAPTURE_JSONL="$OUT_DIR/proxy_training_capture.jsonl"
TRAINING_JSONL="$OUT_DIR/training.toolcall.jsonl"
TRAINING_REJECTS_JSONL="$OUT_DIR/training.toolcall.rejects.jsonl"

pid_listening_on_port() {
  local port
  port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1
  fi
}

wait_for_pid_exit() {
  local pid
  pid="$1"
  for _ in $(seq 1 200); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

stop_pid() {
  local pid label
  pid="$1"
  label="$2"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0

  kill -TERM "$pid" 2>/dev/null || true
  if ! wait_for_pid_exit "$pid"; then
    printf 'warning: %s pid %s did not stop after SIGTERM; sending SIGKILL\n' "$label" "$pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
    wait_for_pid_exit "$pid" || true
  fi
}

cleanup() {
  local status
  status="$?"
  trap - EXIT INT TERM HUP

  if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    stop_pid "$PROXY_PID" "launcher"
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      wait "$PROXY_PID" 2>/dev/null || true
    fi
  fi
  stop_pid "${MANAGED_PROXY_PID:-}" "forge proxy"
  stop_pid "${MANAGED_BACKEND_PID:-}" "llama-server"

  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

cd "$REPO_ROOT"

printf 'Output: %s\n' "$OUT_DIR"
printf 'Prompt payloads: %s\n' "$PROMPTS_JSONL"
printf 'Capture rows: %s\n' "$CAPTURE_JSONL"
printf 'Proxy capture rows: %s\n' "$PROXY_CAPTURE_JSONL"

env \
  FORGE_PROXY_PORT="$PROXY_PORT" \
  FORGE_BACKEND_PORT="$BACKEND_PORT" \
  FORGE_TRAINING_CAPTURE_LOG="$PROXY_CAPTURE_JSONL" \
  "$SCRIPT_DIR/start_llamaserver_proxy.sh" ${GGUF_PATH_ARG:+"$GGUF_PATH_ARG"} "${PROXY_EXTRA_ARGS[@]}" &
PROXY_PID="$!"

printf 'Waiting for proxy health on http://127.0.0.1:%s/health' "$PROXY_PORT"
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then
    printf '\n'
    break
  fi
  printf '.'
  sleep 1
done
printf '\n'
curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null || die "proxy did not become healthy"
MANAGED_PROXY_PID="$(pid_listening_on_port "$PROXY_PORT" || true)"
MANAGED_BACKEND_PID="$(pid_listening_on_port "$BACKEND_PORT" || true)"

cargo run --bin forge-dataset -- prompts \
  --model "$MODEL" \
  --domains "$DOMAINS" \
  --runs "$RUNS" \
  --output "$PROMPTS_JSONL"

cargo run --bin forge-dataset -- capture \
  --proxy-base-url "http://127.0.0.1:$PROXY_PORT/v1" \
  --model "$MODEL" \
  --domains "$DOMAINS" \
  --max-turns "$MAX_TURNS" \
  --runs "$RUNS" \
  --output "$CAPTURE_JSONL"

cargo run --bin forge-dataset -- validate \
  --input "$PROMPTS_JSONL" \
  --input "$CAPTURE_JSONL"

if [[ "$CAPTURE_ONLY" == "1" || "$PROVIDER" == "none" ]]; then
  printf 'Skipping review. Capture-only workflow complete.\n'
  exit 0
fi

cargo run --bin forge-dataset -- review \
  --input "$CAPTURE_JSONL" \
  --output "$TRAINING_JSONL" \
  --provider "$PROVIDER" \
  --verifier-provider "$VERIFIER_PROVIDER"

cargo run --bin forge-dataset -- validate --input "$TRAINING_JSONL"
if [[ -f "$TRAINING_REJECTS_JSONL" ]]; then
  cargo run --bin forge-dataset -- validate --input "$TRAINING_REJECTS_JSONL"
fi

printf 'Training rows: %s\n' "$TRAINING_JSONL"
