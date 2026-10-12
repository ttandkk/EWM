#!/usr/bin/env bash
#SBATCH --job-name=piper-stack-50-v3
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_stack_50_v3_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_stack_50_v3_%j.log

set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
OUTPUT="$ROOT/datasets/piper_stack_fronttight_v3_50/job${SLURM_JOB_ID}"
mkdir -p "$ROOT/datasets/piper_stack_fronttight_v3_50"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROOT/isaacsim601/cache"
export OMNI_USER_CONFIG_PATH="$ROOT/isaacsim601/config"

echo "[production] job=$SLURM_JOB_ID node=$(hostname) episodes=50 seed=20260930 output=$OUTPUT"
cd "$ROOT/Event-DataCollection"
"$ROOT/isaacsim601/env/bin/python" scripts/collect_piper_stack_evis.py \
    --output "$OUTPUT" \
    --repo-id local/piper_stack_fronttight_v3_50 \
    --episodes 50 --max_steps 600 --seed 20260930 \
    --renderer_mode pathtracing --pathtracing_spp 64 \
    --device cuda:0 --headless --enable_cameras

"$ROOT/envs/lerobot/bin/python" scripts/validate_piper_stack_lerobot.py \
    "$OUTPUT" --episodes 50 --require-collection-config
echo "[production] validated dataset: $OUTPUT"
