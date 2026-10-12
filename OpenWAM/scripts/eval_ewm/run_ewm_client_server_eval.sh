#!/usr/bin/env bash
# One command: reserve one GPU, start the EWM WebSocket server, run the
# training-window client, collect logs, and stop the server.
set -euo pipefail

project_root=/projects/hdd/myproject/CVPR2027
python_bin="$project_root/envs/openwam/bin/python"
checkpoint_dir="${EWM_CHECKPOINT_DIR:-}"
output_dir=""
client_args=()
while (($#)); do
    case "$1" in
        --checkpoint-dir)
            checkpoint_dir="$2"; shift 2 ;;
        --output-dir)
            output_dir="$2"; shift 2 ;;
        *)
            client_args+=("$1"); shift ;;
    esac
done
if [[ -z "$checkpoint_dir" || ! -f "$checkpoint_dir/config.yaml" ]]; then
    echo "Pass --checkpoint-dir (or EWM_CHECKPOINT_DIR) pointing to an existing EWM run." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" && "${EWM_EVAL_DIRECT:-0}" != 1 ]]; then
    exec srun --partition="${EWM_EVAL_PARTITION:-cluster02}" \
        --gres="${EWM_EVAL_GRES:-gpu:l40:1}" --ntasks=1 \
        --cpus-per-task=4 --time="${EWM_EVAL_TIME:-02:00:00}" \
        --job-name=ewm-client-server-eval bash "$0" "$@"
fi
checkpoint_dir="$(realpath "$checkpoint_dir")"
if [[ -z "$output_dir" ]]; then
    output_dir="$checkpoint_dir/eval_client_server/$(date -u +%Y%m%d_%H%M%S)"
fi
output_dir="$(realpath -m "$output_dir")"
if [[ -e "$output_dir/complete.json" ]]; then
    echo "Refusing to overwrite completed evaluation: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir" "$project_root/logs"
run_dir="$(mktemp -d /tmp/ewm_eval.XXXXXX)"
ready_file="$run_dir/ready.json"
log_tag="$(basename "$checkpoint_dir")_$(basename "$output_dir")_${SLURM_JOB_ID:-local}"
server_log="$project_root/logs/ewm_eval_server_${log_tag}.log"
client_log="$project_root/logs/ewm_eval_client_${log_tag}.log"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf -- "$run_dir"
}
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$python_bin" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_server.py" \
    --checkpoint-dir "$checkpoint_dir" --ready-file "$ready_file" \
    >"$server_log" 2>&1 &
server_pid=$!
echo "[ewm-eval] checkpoint: $checkpoint_dir"
echo "[ewm-eval] output: $output_dir"
echo "[ewm-eval] server log: $server_log"
echo "[ewm-eval] client log: $client_log"
for ((attempt=0; attempt<600; attempt++)); do
    [[ -s "$ready_file" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[ewm-eval] server exited before readiness; see $server_log" >&2
        tail -n 30 "$server_log" >&2 || true
        exit 1
    fi
    sleep 0.5
done
if [[ ! -s "$ready_file" ]]; then
    echo "[ewm-eval] server readiness timeout; see $server_log" >&2
    exit 1
fi
"$python_bin" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_client.py" \
    --checkpoint-dir "$checkpoint_dir" --output-dir "$output_dir" \
    --ready-file "$ready_file" "${client_args[@]}" 2>&1 | tee "$client_log"
