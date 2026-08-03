#!/usr/bin/env bash
# Read-only checks performed before run_server.sh imports tt-metal or opens the NPU.

tt_server_preflight() {
    local port="${TT_PORT:-8100}"
    local health_host="${TT_HEALTH_HOST:-127.0.0.1}"
    local health_url="http://${health_host}:${port}/health"
    local lock_file="${TT_SERVER_LOCK_FILE:-/tmp/remembr-tt-npu-server.lock}"
    local device_node="${BOS_DEV_NODE:-/dev/bos/0}"
    local health existing_process holders

    if ! command -v flock >/dev/null 2>&1; then
        echo "[tt-server] startup blocked: 'flock' is required for exclusive NPU startup." >&2
        return 1
    fi

    # FD 9 is intentionally inherited by the final `exec python` so this global
    # lock remains held for the server's entire lifetime. It also closes the race
    # where two launchers both observe an idle port before either opens the board.
    exec 9>"$lock_file"
    if ! flock -n 9; then
        echo "[tt-server] startup skipped: another NPU server is running or starting (lock: $lock_file)." >&2
        return 10
    fi

    if command -v curl >/dev/null 2>&1 && \
       health="$(curl -fsS --max-time "${TT_HEALTH_TIMEOUT:-2}" "$health_url" 2>/dev/null)"; then
        echo "[tt-server] startup skipped: server already ready at $health_url" >&2
        echo "[tt-server] health: $health" >&2
        return 10
    fi

    if command -v pgrep >/dev/null 2>&1 && \
       existing_process="$(pgrep -af '[t]t_infer_server.py' 2>/dev/null)"; then
        echo "[tt-server] startup blocked: an inference server process exists but $health_url is not ready." >&2
        echo "$existing_process" >&2
        return 1
    fi

    if [[ -e "$device_node" ]] && command -v fuser >/dev/null 2>&1 && \
       holders="$(fuser "$device_node" 2>/dev/null)"; then
        echo "[tt-server] startup blocked: NPU device $device_node is already held by PID(s):$holders" >&2
        return 1
    fi

    return 0
}
