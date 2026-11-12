#!/usr/bin/env bash
#SBATCH --job-name=ewm-piper-sim-eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_piper_sim_eval_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_piper_sim_eval_%j.log
# Submit the default 50-trial Piper/EVIS evaluation on one GPU.
# RGB-only OOD: --rgb-ood enables lighting + motion blur; default is clean RGB.
# Individual switches: --rgb-lighting-ood / --rgb-motion-blur-ood.
# See OpenWAM/docs/ewm_rgb_ood.md for severity, seeds, and event isolation.
set -euo pipefail

project_root=/projects/hdd/myproject/CVPR2027
checkpoint_dir="${EWM_CHECKPOINT_DIR:-}"
original_args=("$@")
output_dir=""
client_args=()
while (($#)); do
    case "$1" in
        --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        *) client_args+=("$1"); shift ;;
    esac
done
if [[ -z "$checkpoint_dir" || ! -f "$checkpoint_dir/config.yaml" ]]; then
    echo "Pass --checkpoint-dir (or EWM_CHECKPOINT_DIR) pointing to an existing EWM run." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch "$0" "${original_args[@]}"
fi
checkpoint_dir="$(realpath "$checkpoint_dir")"
if [[ -z "$output_dir" ]]; then
    output_dir="$checkpoint_dir/eval_piper_sim/job${SLURM_JOB_ID}"
fi
output_dir="$(realpath -m "$output_dir")"
if [[ -e "$output_dir/complete.json" ]]; then
    echo "Refusing to overwrite completed evaluation: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir" "$project_root/logs"
run_dir="$(mktemp -d /tmp/ewm_piper_eval.XXXXXX)"
ready_file="$run_dir/ready.json"
server_log="$project_root/logs/ewm_piper_server_${SLURM_JOB_ID}.log"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf -- "$run_dir"
}
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="$project_root/isaacsim601/cache"
export OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
compile_core="${COMPILE_CORE:-false}"
cache_context_kv="${CACHE_CONTEXT_KV:-true}"
# Preserve the established closed-loop Piper evaluation protocol by default.
warmup_iters="${WARMUP_ITERS:-0}"
benchmark_default=0
if [[ "$compile_core" == "true" ]]; then benchmark_default=1; fi
benchmark_iters="${BENCHMARK_ITERS:-$benchmark_default}"
server_args=(--warmup-iters "$warmup_iters" --benchmark-iters "$benchmark_iters")
if [[ "$compile_core" == "true" ]]; then
    server_args+=(--compile-core)
else
    server_args+=(--no-compile-core)
fi
if [[ "$cache_context_kv" == "true" ]]; then
    server_args+=(--cache-context-kv)
else
    server_args+=(--no-cache-context-kv)
fi
echo "[ewm-sim] shared GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ewm-sim] checkpoint: $checkpoint_dir"
echo "[ewm-sim] output: $output_dir"
echo "[ewm-sim] model server log: $server_log"
"$project_root/envs/openwam/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_server.py" \
    --checkpoint-dir "$checkpoint_dir" --ready-file "$ready_file" "${server_args[@]}" \
    >"$server_log" 2>&1 &
server_pid=$!
for ((attempt=0; attempt<3600; attempt++)); do
    [[ -s "$ready_file" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[ewm-sim] model server exited before readiness" >&2
        tail -n 50 "$server_log" >&2 || true
        exit 1
    fi
    sleep 0.5
done
if [[ ! -s "$ready_file" ]]; then
    echo "[ewm-sim] model server readiness timeout" >&2
    exit 1
fi
"$project_root/envs/openwam/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_watchdog.py" \
    --output-dir "$output_dir" \
    --stall-timeout "${CLIENT_STALL_TIMEOUT:-720}" \
    --startup-timeout "${CLIENT_STARTUP_TIMEOUT:-900}" \
    --max-restarts "${CLIENT_MAX_RESTARTS:-3}" -- \
    "$project_root/isaacsim601/env/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/eval_ewm_piper_sim.py" \
    --ready-file "$ready_file" --output-dir "$output_dir" \
    --trials 50 --max-steps 500 --sampling-steps 4 --execution-horizon 16 --profile-timing \
    --decode-predicted-events --seed 10000 --scene-seed 20260930 --scene-seed-mode stream \
    --headless --enable_cameras "${client_args[@]}"
if [[ ! -s "$output_dir/complete.json" ]]; then
    echo "[ewm-sim] client exited without complete.json" >&2
    [[ -s "$output_dir/failure.json" ]] && cat "$output_dir/failure.json" >&2
    exit 1
fi
