#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec forge-guardrails-proxy "$@"
fi

case "${1:-}" in
    forge-guardrails-proxy|anyllm_proxy|bash|sh|/bin/*|/usr/bin/*)
        exec "$@"
        ;;
esac

args=("$@")
sidecar_pid=""
forge_pid=""

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

is_falsey() {
    case "${1,,}" in
        0|false|no|off) return 0 ;;
        *) return 1 ;;
    esac
}

has_flag() {
    local flag="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
            return 0
        fi
    done
    return 1
}

has_backend_mode() {
    has_flag "--backend-url" "${args[@]}" || has_flag "--backend" "${args[@]}"
}

append_default() {
    local flag="$1"
    local value="$2"
    if ! has_flag "$flag" "${args[@]}"; then
        args+=("$flag" "$value")
    fi
}

random_key() {
    od -An -tx1 -N32 /dev/urandom | tr -d ' \n'
}

terminate() {
    local status="${1:-0}"
    set +e
    trap - EXIT INT TERM
    if [[ -n "$forge_pid" ]] && kill -0 "$forge_pid" 2>/dev/null; then
        kill "$forge_pid" 2>/dev/null
        wait "$forge_pid" 2>/dev/null
    fi
    if [[ -n "$sidecar_pid" ]] && kill -0 "$sidecar_pid" 2>/dev/null; then
        kill "$sidecar_pid" 2>/dev/null
        wait "$sidecar_pid" 2>/dev/null
    fi
    exit "$status"
}

trap 'terminate $?' EXIT
trap 'terminate 130' INT
trap 'terminate 143' TERM

start_sidecar="${FORGE_START_SIDECAR:-}"
if [[ -z "$start_sidecar" ]]; then
    if has_backend_mode; then
        start_sidecar="false"
    else
        start_sidecar="true"
    fi
fi

using_sidecar="false"
sidecar_key=""
sidecar_port="${ANYLLM_LISTEN_PORT:-3000}"

if is_truthy "$start_sidecar"; then
    using_sidecar="true"
    if [[ -n "${FORGE_SIDECAR_API_KEY:-}" ]]; then
        sidecar_key="$FORGE_SIDECAR_API_KEY"
        sidecar_keys="${PROXY_API_KEYS:-$sidecar_key}"
    elif [[ -n "${PROXY_API_KEYS:-}" ]]; then
        sidecar_keys="$PROXY_API_KEYS"
        sidecar_key="${PROXY_API_KEYS%%,*}"
    else
        sidecar_key="$(random_key)"
        sidecar_keys="$sidecar_key"
    fi

    export ANYLLM_HOME="${ANYLLM_HOME:-/var/lib/forge/anyllm}"
    mkdir -p "$ANYLLM_HOME"

    eprintln_prefix="forge-docker-entrypoint"
    echo "${eprintln_prefix}: starting anyllm sidecar on port ${sidecar_port} (not exposed by this image)" >&2
    LISTEN_PORT="$sidecar_port" \
        DISABLE_ADMIN=1 \
        PROXY_API_KEYS="$sidecar_keys" \
        anyllm_proxy &
    sidecar_pid="$!"

    ready_retries="${FORGE_SIDECAR_READY_RETRIES:-30}"
    ready_sleep="${FORGE_SIDECAR_READY_SLEEP:-1}"
    ready_try=0
    until curl -fsS "http://127.0.0.1:${sidecar_port}/health" >/dev/null 2>&1; do
        if ! kill -0 "$sidecar_pid" 2>/dev/null; then
            set +e
            wait "$sidecar_pid"
            status="$?"
            echo "${eprintln_prefix}: anyllm sidecar exited before readiness" >&2
            terminate "$status"
        fi
        ready_try=$((ready_try + 1))
        if [[ "$ready_try" -ge "$ready_retries" ]]; then
            echo "${eprintln_prefix}: anyllm sidecar did not become ready" >&2
            terminate 1
        fi
        sleep "$ready_sleep"
    done

    if ! has_backend_mode; then
        append_default "--backend-url" "http://127.0.0.1:${sidecar_port}"
    fi
fi

append_default "--host" "${FORGE_HOST:-0.0.0.0}"
append_default "--port" "${FORGE_PORT:-${PORT:-${LISTEN_PORT:-8081}}}"
append_default "--model" "${FORGE_MODEL:-${SMALL_MODEL:-gpt-4o-mini}}"

if [[ "$using_sidecar" == "true" || -n "${FORGE_CONTEXT_TOKENS:-}" ]]; then
    append_default "--budget-tokens" "${FORGE_CONTEXT_TOKENS:-128000}"
fi

if [[ -n "${FORGE_MAX_RETRIES:-}" ]]; then
    append_default "--max-retries" "$FORGE_MAX_RETRIES"
fi

if [[ -n "${FORGE_RESCUE_ENABLED:-}" ]] && is_falsey "$FORGE_RESCUE_ENABLED"; then
    if ! has_flag "--no-rescue" "${args[@]}"; then
        args+=("--no-rescue")
    fi
fi

if [[ -n "${FORGE_SERIALIZE_REQUESTS:-}" ]] && is_truthy "$FORGE_SERIALIZE_REQUESTS"; then
    if ! has_flag "--serialize" "${args[@]}" && ! has_flag "--no-serialize" "${args[@]}"; then
        args+=("--serialize")
    fi
fi

if [[ -n "${FORGE_CLASSIFIER_DIR:-}" ]]; then
    append_default "--classifier-dir" "$FORGE_CLASSIFIER_DIR"
fi

if [[ -n "${FORGE_CLASSIFIER_MODE:-}" ]]; then
    append_default "--classifier-mode" "$FORGE_CLASSIFIER_MODE"
fi

if [[ -n "${FORGE_CLASSIFIER_MODEL:-}" ]]; then
    append_default "--classifier-model" "$FORGE_CLASSIFIER_MODEL"
fi

echo "forge-docker-entrypoint: starting Forge proxy on ${FORGE_HOST:-0.0.0.0}:${FORGE_PORT:-${PORT:-${LISTEN_PORT:-8081}}}" >&2
if [[ "$using_sidecar" == "true" ]]; then
    OPENAI_API_KEY="$sidecar_key" forge-guardrails-proxy "${args[@]}" &
else
    forge-guardrails-proxy "${args[@]}" &
fi
forge_pid="$!"

while true; do
    if ! kill -0 "$forge_pid" 2>/dev/null; then
        set +e
        wait "$forge_pid"
        status="$?"
        terminate "$status"
    fi
    if [[ -n "$sidecar_pid" ]] && ! kill -0 "$sidecar_pid" 2>/dev/null; then
        set +e
        wait "$sidecar_pid"
        status="$?"
        echo "forge-docker-entrypoint: anyllm sidecar exited" >&2
        terminate "$status"
    fi
    sleep 1
done
