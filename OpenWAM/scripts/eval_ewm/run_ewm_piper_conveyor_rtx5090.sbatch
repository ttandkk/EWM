#!/usr/bin/env bash
#SBATCH --job-name=ewm-conveyor-eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_eval_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_eval_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
checkpoint_dir="${EWM_CHECKPOINT_DIR:-$project_root/OpenWAM/outputs/ewm_lean16_p4_d768_a512_dual_ada2_80k/job179463}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache" OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
export LD_LIBRARY_PATH="$project_root/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dir="$(mktemp -d /tmp/ewm_conveyor.XXXXXX)"
ready_file="$run_dir/ready.json"
output_dir="$project_root/tmp/piper_conveyor_v1/ewm_interface/job${SLURM_JOB_ID}"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$run_dir"
}
trap cleanup EXIT INT TERM
"$project_root/envs/openwam/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/ewm_eval_server.py" \
  --checkpoint-dir "$checkpoint_dir" --ready-file "$ready_file" --no-compile-core --cache-context-kv \
  >"$project_root/logs/ewm_conveyor_server_${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
for ((attempt=0; attempt<1800; attempt++)); do
  [[ -s "$ready_file" ]] && break
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 40 "$project_root/logs/ewm_conveyor_server_${SLURM_JOB_ID}.log" >&2
    exit 1
  fi
  sleep 0.5
done
[[ -s "$ready_file" ]] || { echo 'EWM server did not become ready' >&2; exit 1; }
"$project_root/isaacsim601/env/bin/python" "$project_root/OpenWAM/scripts/eval_ewm/eval_ewm_piper_conveyor_sim.py" \
  --ready-file "$ready_file" --output-dir "$output_dir" --device cpu --headless --enable_cameras "$@"
"$project_root/envs/openwam/bin/python" - "$output_dir/complete.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['interface_passed'] and r['video_decoded']
PY
