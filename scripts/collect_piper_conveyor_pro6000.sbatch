#!/usr/bin/env bash
#SBATCH --job-name=piper-conveyor-collect
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_collect_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_collect_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache" OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
export LD_LIBRARY_PATH="$project_root/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
dataset_root="${PIPER_CONVEYOR_DATASET_ROOT:-$project_root/datasets/piper_conveyor_front_wrist_v3_50}"
output_dir="${PIPER_CONVEYOR_OUTPUT:-$dataset_root/job${SLURM_JOB_ID}}"
episodes="${PIPER_CONVEYOR_EPISODES:-50}"
cd "$project_root/Event-DataCollection"
"$project_root/isaacsim601/env/bin/python" scripts/collect_piper_conveyor_evis.py \
  --output "$output_dir" --episodes "$episodes" --max-attempts 200 --seed 20263002 \
  --device cpu --headless --enable_cameras "$@"
"$project_root/envs/lerobot/bin/python" scripts/validate_piper_conveyor_lerobot.py "$output_dir" --episodes "$episodes"
