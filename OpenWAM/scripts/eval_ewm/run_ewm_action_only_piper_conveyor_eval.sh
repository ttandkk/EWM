#!/usr/bin/env bash
#SBATCH --job-name=ewm-action-conveyor-eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_action_conveyor_eval_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_action_conveyor_eval_%j.log
# Submit 50 RGB/state-only trials in the training Piper conveyor scene.
# Observed event panels are for rendering/diagnostics only and are never policy inputs.
# bash run_ewm_action_only_piper_conveyor_eval.sh --checkpoint-dir <training-run>
# Alternatively: sbatch run_ewm_action_only_piper_conveyor_eval.sh --checkpoint-dir <training-run>
# Add --latency-aware for buffered offline latency evaluation; default is paused.
# --latency-controller serial retains the observe/wait/execute comparison controller.
# --speed-ood slow|fast changes only belt speed; omitted keeps the training scene.
# --conveyor-speed 0.07 sets an exact speed instead of a preset.
# No image corruptions are applied.
set -euo pipefail

project_root=/projects/hdd/myproject/CVPR2027
checkpoint_dir="${EWM_CHECKPOINT_DIR:-}"
original_args=("$@")
output_dir=""
speed_ood="none"
conveyor_speed=""
client_args=()
while (($#)); do
    case "$1" in
        --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --speed-ood) speed_ood="$2"; client_args+=("$1" "$2"); shift 2 ;;
        --conveyor-speed) conveyor_speed="$2"; client_args+=("$1" "$2"); shift 2 ;;
        --conveyor-speed=*) conveyor_speed="${1#*=}"; client_args+=("$1"); shift ;;
        --speed-ood=*) speed_ood="${1#*=}"; client_args+=("$1"); shift ;;
        --decode-predicted-events|--event-*|--no-event-*)
            echo "Action-only conveyor policy does not accept event prediction/input flags." >&2
            exit 2 ;;
        *) client_args+=("$1"); shift ;;
    esac
done
case "$speed_ood" in
    none|slow|fast) ;;
    *) echo "--speed-ood must be none, slow or fast" >&2; exit 2 ;;
esac
speed_folder=""
if [[ -n "$conveyor_speed" ]]; then
    if [[ "$speed_ood" != "none" ]]; then
        echo "--conveyor-speed cannot be combined with a --speed-ood preset" >&2
        exit 2
    fi
    speed_folder="$(python3 -c 'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and 0<v<=.2, "invalid conveyor speed"; print(format(v,".12g").replace(".","p"))' "$conveyor_speed")" || exit 2
fi
if [[ -z "$checkpoint_dir" || ! -f "$checkpoint_dir/config.yaml" ]]; then
    echo "Pass --checkpoint-dir (or EWM_CHECKPOINT_DIR) pointing to an existing EWM run." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch "$0" "${original_args[@]}"
fi
checkpoint_dir="$(realpath "$checkpoint_dir")"
if [[ -z "$output_dir" ]]; then
    eval_group="eval_piper_conveyor_action_only"
    if [[ -n "$speed_folder" ]]; then
        eval_group="$eval_group/speed_$speed_folder"
    elif [[ "$speed_ood" != "none" ]]; then
        eval_group="$eval_group/speed_$speed_ood"
    fi
    output_dir="$checkpoint_dir/$eval_group/job${SLURM_JOB_ID}"
fi
output_dir="$(realpath -m "$output_dir")"
if [[ -e "$output_dir/complete.json" ]]; then
    echo "Refusing to overwrite completed evaluation: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir" "$project_root/logs"
run_dir="$(mktemp -d /tmp/ewm_conveyor_eval.XXXXXX)"
ready_file="$run_dir/ready.json"
server_log="$project_root/logs/ewm_action_conveyor_server_${SLURM_JOB_ID}.log"
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
warmup_default=0
for argument in "${client_args[@]}"; do
    if [[ "$argument" == "--latency-aware" ]]; then
        warmup_default=1
    fi
done
warmup_iters="${WARMUP_ITERS:-$warmup_default}"
benchmark_iters="${BENCHMARK_ITERS:-0}"
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
echo "[ewm-conveyor] shared GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ewm-conveyor] checkpoint: $checkpoint_dir"
echo "[ewm-conveyor] output: $output_dir"
echo "[ewm-conveyor] model server log: $server_log"
"$project_root/envs/openwam/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/ewm_action_only_eval_server.py" \
    --checkpoint-dir "$checkpoint_dir" --ready-file "$ready_file" \
    --warmup-prompt 'Pick up both objects arriving on the conveyor belt and place them in the basket.' \
    "${server_args[@]}" \
    >"$server_log" 2>&1 &
server_pid=$!
for ((attempt=0; attempt<3600; attempt++)); do
    [[ -s "$ready_file" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[ewm-conveyor] model server exited before readiness" >&2
        tail -n 50 "$server_log" >&2 || true
        exit 1
    fi
    sleep 0.5
done
if [[ ! -s "$ready_file" ]]; then
    echo "[ewm-conveyor] model server readiness timeout" >&2
    exit 1
fi
"$project_root/envs/openwam/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_watchdog.py" \
    --output-dir "$output_dir" \
    --stall-timeout "${CLIENT_STALL_TIMEOUT:-720}" \
    --startup-timeout "${CLIENT_STARTUP_TIMEOUT:-900}" \
    --max-restarts "${CLIENT_MAX_RESTARTS:-3}" -- \
    "$project_root/isaacsim601/env/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/eval_ewm_action_only_piper_conveyor_sim.py" \
    --checkpoint-dir "$checkpoint_dir" --ready-file "$ready_file" --output-dir "$output_dir" \
    --trials 50 --max-steps 1050 --sampling-steps 4 --execution-horizon 16 --profile-timing \
    --seed 10000 --scene-seed 20280002 \
    --device cpu --headless --enable_cameras "${client_args[@]}"
if [[ ! -s "$output_dir/complete.json" ]]; then
    echo "[ewm-conveyor] client exited without complete.json" >&2
    [[ -s "$output_dir/failure.json" ]] && cat "$output_dir/failure.json" >&2
    exit 1
fi
