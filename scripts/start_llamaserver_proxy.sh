#!/usr/bin/env bash
set -euo pipefail

DEFAULT_GGUF_FILENAME="mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf"
DEFAULT_PROXY_PORT="8081"
DEFAULT_BACKEND_PORT="8080"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

DRY_RUN=0
GGUF_ARG="${GGUF_PATH:-${FORGE_GGUF:-${GGUF:-}}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [GGUF_PATH] [proxy args...]

Starts forge-guardrails-proxy in managed llama-server mode.

Defaults:
  GGUF filename: $DEFAULT_GGUF_FILENAME
  proxy port:    ${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}
  backend port:  ${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}

Environment:
  GGUF_PATH, FORGE_GGUF, or GGUF       Explicit GGUF path
  FORGE_MODELS_DIR or MODELS_DIR       Directory to search first
  FORGE_PROXY_PORT or PROXY_PORT       Proxy listen port
  FORGE_BACKEND_PORT or BACKEND_PORT   Managed llama-server port
  FORGE_PROXY_BIN                      forge-guardrails-proxy binary path/name

Examples:
  $(basename "$0")
  $(basename "$0") /models/$DEFAULT_GGUF_FILENAME
  FORGE_MODELS_DIR=/models $(basename "$0") --verbose
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'warning: %s\n' "$*" >&2
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

canonical_file() {
  local path dir base
  path="$(expand_path "$1")"
  dir="$(dirname "$path")"
  base="$(basename "$path")"
  (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
}

valid_port() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
  esac
  (( 10#$1 >= 1 && 10#$1 <= 65535 ))
}

check_port_free() {
  local port label
  port="$1"
  label="$2"

  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
      die "$label port $port is already in use"
    fi
    return 0
  fi

  if command -v nc >/dev/null 2>&1; then
    if nc -z 127.0.0.1 "$port" >/dev/null 2>&1; then
      die "$label port $port is already accepting TCP connections"
    fi
    return 0
  fi

  warn "cannot verify $label port $port; neither lsof nor nc is available"
}

add_candidate() {
  local candidate existing
  candidate="$(expand_path "$1")"
  [[ -f "$candidate" ]] || return 0
  candidate="$(canonical_file "$candidate")"
  for existing in "${GGUF_CANDIDATES[@]}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  GGUF_CANDIDATES+=("$candidate")
}

search_dir() {
  local dir found
  dir="$(expand_path "$1")"
  [[ -d "$dir" ]] || return 0
  while IFS= read -r found; do
    add_candidate "$found"
  done < <(find "$dir" -maxdepth 6 \( -type f -o -type l \) -name "$DEFAULT_GGUF_FILENAME" 2>/dev/null)
}

resolve_gguf() {
  local explicit
  explicit="$(expand_path "$GGUF_ARG")"

  if [[ -n "$explicit" ]]; then
    [[ -f "$explicit" ]] || die "GGUF file not found: $explicit"
    [[ "$explicit" == *.gguf ]] || die "expected a .gguf file: $explicit"
    [[ -r "$explicit" ]] || die "GGUF file is not readable: $explicit"
    canonical_file "$explicit"
    return 0
  fi

  GGUF_CANDIDATES=()

  [[ -n "${FORGE_MODELS_DIR:-}" ]] && search_dir "$FORGE_MODELS_DIR"
  [[ -n "${MODELS_DIR:-}" ]] && search_dir "$MODELS_DIR"
  search_dir "$REPO_ROOT"
  search_dir "$REPO_ROOT/models"
  search_dir "$REPO_ROOT/../models"
  search_dir "$HOME/Models"
  search_dir "$HOME/models"
  search_dir "$HOME/.cache/huggingface/hub"

  case "${#GGUF_CANDIDATES[@]}" in
    0)
      die "could not find $DEFAULT_GGUF_FILENAME; pass GGUF_PATH or set FORGE_MODELS_DIR"
      ;;
    1)
      printf '%s\n' "${GGUF_CANDIDATES[0]}"
      ;;
    *)
      printf 'error: multiple matching GGUF files found:\n' >&2
      printf '  %s\n' "${GGUF_CANDIDATES[@]}" >&2
      die "pass the intended GGUF path explicitly"
      ;;
  esac
}

resolve_proxy_command() {
  local bin target_dir candidate
  if [[ -n "${FORGE_PROXY_BIN:-}" ]]; then
    bin="$(expand_path "$FORGE_PROXY_BIN")"
    if [[ "$bin" == */* ]]; then
      [[ -x "$bin" ]] || die "FORGE_PROXY_BIN is not executable: $bin"
    else
      command -v "$bin" >/dev/null 2>&1 || die "FORGE_PROXY_BIN not found on PATH: $bin"
    fi
    PROXY_CMD=("$bin")
    return 0
  fi

  if command -v forge-guardrails-proxy >/dev/null 2>&1; then
    PROXY_CMD=("forge-guardrails-proxy")
    return 0
  fi

  for target_dir in "${CARGO_TARGET_DIR:-}" "$REPO_ROOT/target"; do
    [[ -n "$target_dir" ]] || continue
    target_dir="$(expand_path "$target_dir")"
    for candidate in \
      "$target_dir/debug/forge-guardrails-proxy" \
      "$target_dir/release/forge-guardrails-proxy" \
      "$target_dir"/*/debug/forge-guardrails-proxy \
      "$target_dir"/*/release/forge-guardrails-proxy
    do
      [[ -x "$candidate" ]] || continue
      PROXY_CMD=("$candidate")
      return 0
    done
  done

  command -v cargo >/dev/null 2>&1 || die "forge-guardrails-proxy not found and cargo is unavailable"
  if [[ "$DRY_RUN" == "1" ]]; then
    PROXY_CMD=("$REPO_ROOT/target/debug/forge-guardrails-proxy")
    return 0
  fi

  printf 'building forge-guardrails-proxy...\n' >&2
  (cd "$REPO_ROOT" && cargo build --bin forge-guardrails-proxy)
  PROXY_CMD=("$REPO_ROOT/target/debug/forge-guardrails-proxy")
}

CHILD_PID=""
SHUTTING_DOWN=0

stop_child_and_exit() {
  local signal exit_code
  signal="$1"
  exit_code="$2"

  trap - INT TERM HUP
  if [[ "$SHUTTING_DOWN" == "1" ]]; then
    exit "$exit_code"
  fi
  SHUTTING_DOWN=1

  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    printf '\nreceived SIG%s, stopping forge proxy...\n' "$signal" >&2
    kill -s "$signal" "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

run_proxy() {
  local status
  trap 'stop_child_and_exit INT 130' INT
  trap 'stop_child_and_exit TERM 143' TERM
  trap 'stop_child_and_exit HUP 129' HUP

  (trap - INT TERM HUP; exec "${CMD[@]}") &
  CHILD_PID="$!"

  set +e
  wait "$CHILD_PID"
  status="$?"
  set -e

  trap - INT TERM HUP
  CHILD_PID=""
  return "$status"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      break
      ;;
    --*)
      break
      ;;
    *)
      if [[ -z "$GGUF_ARG" ]]; then
        GGUF_ARG="$1"
        shift
      else
        break
      fi
      ;;
  esac
done

EXTRA_PROXY_ARGS=("$@")
PROXY_PORT="${FORGE_PROXY_PORT:-${PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
BACKEND_PORT="${FORGE_BACKEND_PORT:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"

valid_port "$PROXY_PORT" || die "invalid proxy port: $PROXY_PORT"
valid_port "$BACKEND_PORT" || die "invalid backend port: $BACKEND_PORT"
[[ "$PROXY_PORT" != "$BACKEND_PORT" ]] || die "proxy and backend ports must differ"

command -v llama-server >/dev/null 2>&1 || die "llama-server is not on PATH"

GGUF_PATH_RESOLVED="$(resolve_gguf)"
check_port_free "$PROXY_PORT" "proxy"
check_port_free "$BACKEND_PORT" "backend"
resolve_proxy_command

CMD=(
  "${PROXY_CMD[@]}"
  "--backend" "llamaserver"
  "--gguf" "$GGUF_PATH_RESOLVED"
  "--backend-port" "$BACKEND_PORT"
  "--port" "$PROXY_PORT"
  "${EXTRA_PROXY_ARGS[@]}"
)

cd "$REPO_ROOT"

printf 'GGUF: %s\n' "$GGUF_PATH_RESOLVED"
printf 'Proxy: http://127.0.0.1:%s/v1\n' "$PROXY_PORT"
printf 'Backend: http://127.0.0.1:%s/v1\n' "$BACKEND_PORT"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

run_proxy
