#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OUT_DIR="target/dataset/run"
DEFAULT_PROXY_PORT="8081"
DEFAULT_BACKEND_PORT="8080"
DEFAULT_MODEL="test-model"
DEFAULT_CAPTURE_BACKEND="llamaserver"
DEFAULT_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_PROVIDER="auto"
DEFAULT_VERIFIER_PROVIDER="same"
DEFAULT_REVIEW_CONCURRENCY="1"
DEFAULT_MAX_TURNS="4"
DEFAULT_CAPTURE_MAX_ERRORS="25"
DEFAULT_RUNS="1"
DEFAULT_DOMAINS="repo_docs,shopping,calendar,support"
DEFAULT_COMBINED_OUTPUT="training.toolcall.combined.jsonl"
DEFAULT_SPLIT_SEED="forge-dataset-v1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

OUT_DIR="${FORGE_DATASET_OUT_DIR:-$DEFAULT_OUT_DIR}"
PROXY_PORT="${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
BACKEND_PORT="${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"
MODEL="${FORGE_DATASET_MODEL:-$DEFAULT_MODEL}"
CAPTURE_BACKEND="${FORGE_DATASET_CAPTURE_BACKEND:-$DEFAULT_CAPTURE_BACKEND}"
OPENROUTER_BASE_URL="${FORGE_DATASET_OPENROUTER_BASE_URL:-$DEFAULT_OPENROUTER_BASE_URL}"
PROVIDER="${FORGE_DATASET_REVIEW_PROVIDER:-$DEFAULT_PROVIDER}"
VERIFIER_PROVIDER="${FORGE_DATASET_VERIFIER_PROVIDER:-$DEFAULT_VERIFIER_PROVIDER}"
REVIEW_CONCURRENCY="${FORGE_DATASET_REVIEW_CONCURRENCY:-$DEFAULT_REVIEW_CONCURRENCY}"
REVIEW_CHUNK_SIZE="${FORGE_DATASET_REVIEW_CHUNK_SIZE:-}"
REVIEW_MAX_ALTERNATIVES_PER_GROUP="${FORGE_DATASET_REVIEW_MAX_ALTERNATIVES_PER_GROUP:-}"
REVIEW_MAX_ALTERNATIVE_RATIO="${FORGE_DATASET_REVIEW_MAX_ALTERNATIVE_RATIO:-}"
MAX_TURNS="${FORGE_DATASET_MAX_TURNS:-$DEFAULT_MAX_TURNS}"
CAPTURE_MAX_ERRORS="${FORGE_DATASET_CAPTURE_MAX_ERRORS:-$DEFAULT_CAPTURE_MAX_ERRORS}"
RUNS="${FORGE_DATASET_RUNS:-$DEFAULT_RUNS}"
DOMAINS="${FORGE_DATASET_DOMAINS:-$DEFAULT_DOMAINS}"
OPENROUTER_MODEL="${FORGE_DATASET_OPENROUTER_MODEL:-}"
AGENT_LOG_LIMIT="${FORGE_DATASET_AGENT_LOG_LIMIT:-}"
AGENT_LOG_SYNTHETIC_BALANCED="${FORGE_DATASET_AGENT_LOG_SYNTHETIC_BALANCED:-0}"
COMBINED_OUTPUT="${FORGE_DATASET_COMBINED_OUTPUT:-$DEFAULT_COMBINED_OUTPUT}"
SPLIT_VALIDATION_RATIO="${FORGE_DATASET_SPLIT_VALIDATION_RATIO:-}"
SPLIT_SEED="${FORGE_DATASET_SPLIT_SEED:-$DEFAULT_SPLIT_SEED}"
GGUF_PATH_ARG="${GGUF_PATH:-${FORGE_GGUF:-${GGUF:-}}}"
CAPTURE_ONLY=0
INCLUDE_AGENT_LOGS=0
RESUME_REVIEW=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [GGUF_PATH] [extra proxy args...]

Starts managed llama-server + forge proxy, generates tool-call prompts, captures
real model tool calls against harmless stub tools, then reviews rows with
MiniMax/OpenRouter unless --capture-only or --provider none is used.

Options:
  --out-dir DIR                 Output directory (default: $DEFAULT_OUT_DIR)
  --model MODEL                 Chat model name sent to the proxy (default: $DEFAULT_MODEL)
  --capture-backend llamaserver|openrouter
                                Capture backend (default: $DEFAULT_CAPTURE_BACKEND)
  --capture-model MODEL         Alias for --model; useful with --capture-backend openrouter
  --openrouter-base-url URL     OpenRouter-compatible capture base URL (default: $DEFAULT_OPENROUTER_BASE_URL)
  --provider auto|minimax|openrouter|none
                                Review provider (default: auto)
  --verifier-provider same|auto|minimax|openrouter
                                Verifier provider (default: same)
  --openrouter-model MODEL      OpenRouter review/verifier model
  --review-concurrency N        Parallel capture reviews (default: $DEFAULT_REVIEW_CONCURRENCY)
  --review-chunk-size N         Client-side review chunk size
  --review-max-alternatives-per-group N
                                Cap targeted hard negatives accepted per group
  --review-max-alternative-ratio R
                                Cap targeted hard negatives as a ratio of real accepted rows
  --max-turns N                 Capture turns per scenario (default: $DEFAULT_MAX_TURNS)
  --capture-max-errors N        Failed capture scenarios tolerated (default: $DEFAULT_CAPTURE_MAX_ERRORS; 0 fails fast)
  --runs N                      Scenario repetitions (default: $DEFAULT_RUNS)
  --domains CSV                 Dataset domains (default: $DEFAULT_DOMAINS; also supports forge_eval)
  --proxy-port PORT             Forge proxy port (default: $DEFAULT_PROXY_PORT)
  --backend-port PORT           Managed llama-server port (default: $DEFAULT_BACKEND_PORT)
  --capture-only                Skip review
  --resume-review               Resume review/downstream steps from existing capture rows
  --include-agent-logs          Also mine sanitized Codex/Claude logs through forge-dataset
  --agent-log-limit N           Limit agent-log candidates
  --agent-log-synthetic-balanced N
                                Add bounded synthetic agent-log negatives
  --combined-output NAME        Combined output file name (default: $DEFAULT_COMBINED_OUTPUT)
  --split-validation-ratio R    Split reviewed rows into train/validation outputs
  --split-seed TEXT             Deterministic split seed (default: $DEFAULT_SPLIT_SEED)
  -h, --help                    Show this help

Environment:
  MINIMAX_API_KEY, OPENROUTER_API_KEY
  GENERATETD_MINIMAX_MODEL, GENERATETD_OPENROUTER_MODEL
  FORGE_DATASET_* equivalents for options above

Examples:
  $(basename "$0") /models/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf
  OPENROUTER_API_KEY=... $(basename "$0") --provider openrouter --out-dir target/dataset/openrouter
  OPENROUTER_API_KEY=... $(basename "$0") --capture-backend openrouter --capture-model mistralai/mixtral-8x22b-instruct
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

# check_python_env BACKEND_DIR PYTHON_BIN
# Verifies that PYTHON_BIN can import all packages required by generatetd.
# Prints a clear remediation hint and exits 1 on failure.
check_python_env() {
  local backend_dir="$1"
  local python_bin="$2"

  printf 'Checking Python environment for --include-agent-logs (%s)...\n' "$python_bin"

  # 1. Confirm the interpreter exists at all.
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'error: Python interpreter not found: %s\n' "$python_bin" >&2
    printf 'hint:  Install Python 3.11+ or set PYTHON=/path/to/python3\n' >&2
    exit 1
  fi

  local python_version
  python_version="$("$python_bin" -c 'import sys; print(sys.version)' 2>&1)"
  printf '       interpreter: %s %s\n' "$python_bin" "$python_version"

  # 2. Check that every required package is importable.
  # Packages come from notebook/generatetd/pyproject.toml [project.dependencies].
  local -a required_packages=(requests jsonschema generatetd)
  local missing=()
  for pkg in "${required_packages[@]}"; do
    if ! "$python_bin" -c "import $pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'error: The following Python packages are missing from %s:\n' "$python_bin" >&2
    for pkg in "${missing[@]}"; do
      printf '         - %s\n' "$pkg" >&2
    done
    printf '\n' >&2
    printf 'hint:  Activate your venv or install the missing packages before running this script.\n' >&2
    printf '       Option 1 – install in your active Python environment:\n' >&2
    printf '         pip install %s\n' "${missing[*]}" >&2
    printf '       Option 2 – install generatetd in editable mode (installs all deps):\n' >&2
    printf '         pip install -e %s\n' "$backend_dir" >&2
    printf '       Option 3 – use uv run (no manual venv activation needed):\n' >&2
    printf '         uv run --project %s <your workflow command>\n' "$backend_dir" >&2
    printf '       Option 4 – set PYTHON to a venv interpreter that has the packages:\n' >&2
    printf '         PYTHON=%s/.venv/bin/python3 <this script> ...\n' "$backend_dir" >&2
    exit 1
  fi

  printf '       all required packages available: %s\n' "${required_packages[*]}"
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
    --capture-model)
      MODEL="${2:?--capture-model requires a value}"
      shift 2
      ;;
    --capture-backend)
      CAPTURE_BACKEND="${2:?--capture-backend requires a value}"
      shift 2
      ;;
    --openrouter-base-url)
      OPENROUTER_BASE_URL="${2:?--openrouter-base-url requires a value}"
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
    --openrouter-model)
      OPENROUTER_MODEL="${2:?--openrouter-model requires a value}"
      shift 2
      ;;
    --review-concurrency)
      REVIEW_CONCURRENCY="${2:?--review-concurrency requires a value}"
      shift 2
      ;;
    --review-chunk-size)
      REVIEW_CHUNK_SIZE="${2:?--review-chunk-size requires a value}"
      shift 2
      ;;
    --review-max-alternatives-per-group)
      REVIEW_MAX_ALTERNATIVES_PER_GROUP="${2:?--review-max-alternatives-per-group requires a value}"
      shift 2
      ;;
    --review-max-alternative-ratio)
      REVIEW_MAX_ALTERNATIVE_RATIO="${2:?--review-max-alternative-ratio requires a value}"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="${2:?--max-turns requires a value}"
      shift 2
      ;;
    --capture-max-errors)
      CAPTURE_MAX_ERRORS="${2:?--capture-max-errors requires a value}"
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
    --resume-review)
      RESUME_REVIEW=1
      shift
      ;;
    --include-agent-logs)
      INCLUDE_AGENT_LOGS=1
      shift
      ;;
    --agent-log-limit)
      AGENT_LOG_LIMIT="${2:?--agent-log-limit requires a value}"
      shift 2
      ;;
    --agent-log-synthetic-balanced)
      AGENT_LOG_SYNTHETIC_BALANCED="${2:?--agent-log-synthetic-balanced requires a value}"
      shift 2
      ;;
    --combined-output)
      COMBINED_OUTPUT="${2:?--combined-output requires a value}"
      shift 2
      ;;
    --split-validation-ratio)
      SPLIT_VALIDATION_RATIO="${2:?--split-validation-ratio requires a value}"
      shift 2
      ;;
    --split-seed)
      SPLIT_SEED="${2:?--split-seed requires a value}"
      shift 2
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
[[ "$CAPTURE_BACKEND" =~ ^(llamaserver|openrouter)$ ]] || die "--capture-backend must be llamaserver|openrouter"
[[ "$PROVIDER" =~ ^(auto|minimax|openrouter|none)$ ]] || die "--provider must be auto|minimax|openrouter|none"
[[ "$VERIFIER_PROVIDER" =~ ^(same|auto|minimax|openrouter)$ ]] || die "--verifier-provider must be same|auto|minimax|openrouter"
[[ "$REVIEW_CONCURRENCY" =~ ^[0-9]+$ ]] && (( REVIEW_CONCURRENCY >= 1 && REVIEW_CONCURRENCY <= 32 )) || die "--review-concurrency must be an integer from 1 to 32"
if [[ -n "$REVIEW_CHUNK_SIZE" ]]; then
  [[ "$REVIEW_CHUNK_SIZE" =~ ^[0-9]+$ ]] && (( REVIEW_CHUNK_SIZE > 0 )) || die "--review-chunk-size must be a positive integer"
fi
if [[ -n "$REVIEW_MAX_ALTERNATIVES_PER_GROUP" ]]; then
  [[ "$REVIEW_MAX_ALTERNATIVES_PER_GROUP" =~ ^[0-9]+$ ]] || die "--review-max-alternatives-per-group must be a non-negative integer"
fi
if [[ -n "$REVIEW_MAX_ALTERNATIVE_RATIO" ]]; then
  [[ "$REVIEW_MAX_ALTERNATIVE_RATIO" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--review-max-alternative-ratio must be a number"
fi
[[ "$MAX_TURNS" =~ ^[0-9]+$ ]] && (( MAX_TURNS > 0 )) || die "--max-turns must be a positive integer"
[[ "$CAPTURE_MAX_ERRORS" =~ ^[0-9]+$ ]] || die "--capture-max-errors must be a non-negative integer"
[[ "$RUNS" =~ ^[0-9]+$ ]] && (( RUNS > 0 )) || die "--runs must be a positive integer"
if [[ "$RESUME_REVIEW" == "1" && ( "$CAPTURE_ONLY" == "1" || "$PROVIDER" == "none" ) ]]; then
  die "--resume-review requires review to be enabled"
fi
if [[ -n "$AGENT_LOG_LIMIT" ]]; then
  [[ "$AGENT_LOG_LIMIT" =~ ^[0-9]+$ ]] || die "--agent-log-limit must be a non-negative integer"
fi
[[ "$AGENT_LOG_SYNTHETIC_BALANCED" =~ ^[0-9]+$ ]] || die "--agent-log-synthetic-balanced must be a non-negative integer"
if [[ -n "$SPLIT_VALIDATION_RATIO" ]]; then
  [[ "$SPLIT_VALIDATION_RATIO" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--split-validation-ratio must be a number"
fi
[[ -n "$SPLIT_SEED" ]] || die "--split-seed must not be empty"
PROMPTS_JSONL="$OUT_DIR/tool_prompts.jsonl"
CAPTURE_JSONL="$OUT_DIR/capture.jsonl"
PROXY_CAPTURE_JSONL="$OUT_DIR/proxy_training_capture.jsonl"
TRAINING_JSONL="$OUT_DIR/training.toolcall.jsonl"
TRAINING_REJECTS_JSONL="$OUT_DIR/training.toolcall.rejects.jsonl"
AGENT_LOGS_DIR="$OUT_DIR/agent_logs"
AGENT_LOGS_TOOL_JSONL="$AGENT_LOGS_DIR/tool_call_training.jsonl"
COMBINED_JSONL="$OUT_DIR/$COMBINED_OUTPUT"
TRAIN_JSONL="$OUT_DIR/train.jsonl"
VALIDATION_JSONL="$OUT_DIR/validation.jsonl"

if [[ "$CAPTURE_BACKEND" == "openrouter" && "$RESUME_REVIEW" != "1" ]]; then
  [[ -n "${OPENROUTER_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" ]] || die "OPENROUTER_API_KEY or OPENAI_API_KEY is required for --capture-backend openrouter"
fi
if [[ "$RESUME_REVIEW" == "1" ]]; then
  [[ -d "$OUT_DIR" ]] || die "--resume-review requires an existing output directory: $OUT_DIR"
else
  mkdir -p "$OUT_DIR"
fi

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
if [[ "$INCLUDE_AGENT_LOGS" == "1" ]]; then
  printf 'Agent log rows: %s\n' "$AGENT_LOGS_TOOL_JSONL"
  printf 'Combined rows: %s\n' "$COMBINED_JSONL"
fi
if [[ -n "$SPLIT_VALIDATION_RATIO" ]]; then
  printf 'Train split rows: %s\n' "$TRAIN_JSONL"
  printf 'Validation split rows: %s\n' "$VALIDATION_JSONL"
fi

# Preflight: validate the Python environment before starting any long-running
# proxy/capture/review work so that a missing venv or package is caught early.
if [[ "$INCLUDE_AGENT_LOGS" == "1" ]]; then
  _GENERATETD_DIR="$REPO_ROOT/notebook/generatetd"
  _PYTHON_BIN="${PYTHON:-python3}"
  check_python_env "$_GENERATETD_DIR" "$_PYTHON_BIN"
  # Resolve to absolute path so Rust's ProcessCommand::new() finds the
  # interpreter correctly even when it runs with a different CWD
  # (notebook/generatetd/). A relative path like
  # "notebook/generatetd/.venv/bin/python3" would be double-nested otherwise.
  # Use sys.executable: Python always reports its own real absolute path,
  # so relative paths and symlinks both resolve correctly.
  export PYTHON
  PYTHON="$("$_PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  printf '       resolved interpreter: %s\n' "$PYTHON"
fi

if [[ "$RESUME_REVIEW" == "1" ]]; then
  [[ -s "$CAPTURE_JSONL" ]] || die "--resume-review requires existing non-empty capture rows: $CAPTURE_JSONL; check --out-dir or rerun capture"
  printf 'Resuming review from existing capture rows: %s\n' "$CAPTURE_JSONL"
elif [[ "$CAPTURE_BACKEND" == "openrouter" ]]; then
  PROXY_CMD=(cargo run --bin forge-guardrails-proxy --)
  PROXY_CMD+=(
    --backend-url "$OPENROUTER_BASE_URL"
    --model "$MODEL"
    --port "$PROXY_PORT"
  )
  PROXY_CMD+=("${PROXY_EXTRA_ARGS[@]}")
  printf 'Capture backend: OpenRouter-compatible %s\n' "$OPENROUTER_BASE_URL"
  printf 'Capture model: %s\n' "$MODEL"
  env \
    FORGE_TRAINING_CAPTURE_LOG="$PROXY_CAPTURE_JSONL" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}" \
    "${PROXY_CMD[@]}" &
else
  env \
    FORGE_PROXY_PORT="$PROXY_PORT" \
    FORGE_BACKEND_PORT="$BACKEND_PORT" \
    FORGE_TRAINING_CAPTURE_LOG="$PROXY_CAPTURE_JSONL" \
    "$SCRIPT_DIR/start_llamaserver_proxy.sh" ${GGUF_PATH_ARG:+"$GGUF_PATH_ARG"} "${PROXY_EXTRA_ARGS[@]}" &
fi
if [[ "$RESUME_REVIEW" != "1" ]]; then
  PROXY_PID="$!"

  printf 'Waiting for proxy health on http://127.0.0.1:%s/health' "$PROXY_PORT"
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then
      printf '\n'
      break
    fi
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      printf '\n'
      wait "$PROXY_PID" 2>/dev/null || true
      die "proxy launcher exited before health check succeeded"
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
    --max-scenario-errors "$CAPTURE_MAX_ERRORS" \
    --runs "$RUNS" \
    --output "$CAPTURE_JSONL"
fi

if [[ "$RESUME_REVIEW" == "1" ]]; then
  cargo run --bin forge-dataset -- validate --input "$CAPTURE_JSONL"
else
  cargo run --bin forge-dataset -- validate \
    --input "$PROMPTS_JSONL" \
    --input "$CAPTURE_JSONL"
fi

if [[ "$CAPTURE_ONLY" == "1" || "$PROVIDER" == "none" ]]; then
  printf 'Skipping review. Capture-only workflow complete.\n'
  exit 0
fi

REVIEW_ARGS=(
  --input "$CAPTURE_JSONL"
  --output "$TRAINING_JSONL"
  --provider "$PROVIDER"
  --verifier-provider "$VERIFIER_PROVIDER"
  --concurrency "$REVIEW_CONCURRENCY"
)
if [[ -n "$OPENROUTER_MODEL" ]]; then
  REVIEW_ARGS+=(--openrouter-model "$OPENROUTER_MODEL")
fi
if [[ -n "$REVIEW_CHUNK_SIZE" ]]; then
  REVIEW_ARGS+=(--chunk-size "$REVIEW_CHUNK_SIZE")
fi
if [[ -n "$REVIEW_MAX_ALTERNATIVES_PER_GROUP" ]]; then
  REVIEW_ARGS+=(--max-alternatives-per-group "$REVIEW_MAX_ALTERNATIVES_PER_GROUP")
fi
if [[ -n "$REVIEW_MAX_ALTERNATIVE_RATIO" ]]; then
  REVIEW_ARGS+=(--max-alternative-ratio "$REVIEW_MAX_ALTERNATIVE_RATIO")
fi
if [[ "$RESUME_REVIEW" == "1" ]]; then
  REVIEW_ARGS+=(--resume)
fi
cargo run --bin forge-dataset -- review "${REVIEW_ARGS[@]}"

cargo run --bin forge-dataset -- validate --input "$TRAINING_JSONL"
if [[ -f "$TRAINING_REJECTS_JSONL" ]]; then
  cargo run --bin forge-dataset -- validate --input "$TRAINING_REJECTS_JSONL"
fi

if [[ "$INCLUDE_AGENT_LOGS" == "1" ]]; then
  AGENT_LOG_ARGS=(
    --out "$AGENT_LOGS_DIR"
    --provider "$PROVIDER"
    --verifier-provider "$VERIFIER_PROVIDER"
  )
  if [[ -n "$OPENROUTER_MODEL" ]]; then
    AGENT_LOG_ARGS+=(--openrouter-model "$OPENROUTER_MODEL")
  fi
  if [[ -n "$AGENT_LOG_LIMIT" ]]; then
    AGENT_LOG_ARGS+=(--limit "$AGENT_LOG_LIMIT")
  fi
  if (( AGENT_LOG_SYNTHETIC_BALANCED > 0 )); then
    AGENT_LOG_ARGS+=(--synthetic-balanced "$AGENT_LOG_SYNTHETIC_BALANCED")
  fi
  if [[ "$RESUME_REVIEW" == "1" && -s "$AGENT_LOGS_TOOL_JSONL" ]]; then
    printf 'Reusing existing agent log rows: %s\n' "$AGENT_LOGS_TOOL_JSONL"
  else
    cargo run --bin forge-dataset -- agent-logs "${AGENT_LOG_ARGS[@]}"
  fi
  cargo run --bin forge-dataset -- assemble \
    --input "$TRAINING_JSONL" \
    --input "$AGENT_LOGS_TOOL_JSONL" \
    --out-dir "$OUT_DIR" \
    --combined-output "$COMBINED_OUTPUT" \
    --drop-conflicts
  cargo run --bin forge-dataset -- validate --input "$COMBINED_JSONL"
fi

if [[ -n "$SPLIT_VALIDATION_RATIO" ]]; then
  SPLIT_INPUT="$TRAINING_JSONL"
  if [[ "$INCLUDE_AGENT_LOGS" == "1" ]]; then
    SPLIT_INPUT="$COMBINED_JSONL"
  fi
  cargo run --bin forge-dataset -- split \
    --input "$SPLIT_INPUT" \
    --out-dir "$OUT_DIR" \
    --validation-ratio "$SPLIT_VALIDATION_RATIO" \
    --seed "$SPLIT_SEED"
  cargo run --bin forge-dataset -- validate \
    --input "$TRAIN_JSONL" \
    --input "$VALIDATION_JSONL"
fi

printf 'Training rows: %s\n' "$TRAINING_JSONL"
if [[ "$INCLUDE_AGENT_LOGS" == "1" ]]; then
  printf 'Combined training rows: %s\n' "$COMBINED_JSONL"
fi
if [[ -n "$SPLIT_VALIDATION_RATIO" ]]; then
  printf 'Train split rows: %s\n' "$TRAIN_JSONL"
  printf 'Validation split rows: %s\n' "$VALIDATION_JSONL"
fi
