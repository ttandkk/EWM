#!/usr/bin/env bash
#SBATCH --job-name=piper-teleop-pro6000
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_teleop_pro6000_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_teleop_pro6000_%j.log

set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
SCRIPT="$ROOT/scripts/start_piper_teleop_pro6000.sh"

# `bash SCRIPT` is the one-command entry point, including from an interactive
# srun shell.  Direct `sbatch SCRIPT` also works without submitting twice.
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NAME:-}" != piper-teleop-pro6000 ]]; then
    umask 077
    job_id=$(sbatch --parsable "$SCRIPT")
    job_id=${job_id%%;*}
    echo "Submitted Piper teleoperation job $job_id (Pro 6000)."
    echo "Status: squeue -j $job_id -o '%.18i %.8T %.10M %.20R'"
    echo "Log and private browser token: $ROOT/logs/piper_teleop_pro6000_${job_id}.log"
    echo "Dataset: $ROOT/datasets/piper_stack_teleop_pathtracing_pro6000/job${job_id}"
    exit 0
fi

umask 077
LOG="$ROOT/logs/piper_teleop_pro6000_${SLURM_JOB_ID}.log"
chmod 600 "$LOG"
OUTPUT="$ROOT/datasets/piper_stack_teleop_pathtracing_pro6000/job${SLURM_JOB_ID}"
mkdir -p "$ROOT/datasets/piper_stack_teleop_pathtracing_pro6000"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROOT/isaacsim601/cache"
export OMNI_USER_CONFIG_PATH="$ROOT/isaacsim601/config"

echo "[teleop] job=$SLURM_JOB_ID node=$(hostname) output=$OUTPUT"
echo "[teleop] SSH-tunnel destination: $(hostname):127.0.0.1:8765"
cd "$ROOT/Event-DataCollection"
exec "$ROOT/isaacsim601/env/bin/python" \
    scripts/collect_piper_stack_teleop.py \
    --output "$OUTPUT" \
    --episodes 1 --max_steps 2400 --device cuda:0 \
    --renderer_mode pathtracing --pathtracing_spp 64 \
    --wrist-eye-local -0.12 0.0 -0.07 \
    --wrist-target-local 0.0 0.0 0.17 \
    --wrist-focal-length 8.5 \
    --teleop-host 127.0.0.1 --teleop-port 8765
