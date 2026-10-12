#!/usr/bin/env bash
#SBATCH --job-name=piper-conveyor-events
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_events_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_events_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache" OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
export LD_LIBRARY_PATH="$project_root/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
output_dir="$project_root/tmp/piper_conveyor_v1/event_probe/job${SLURM_JOB_ID}"
cd "$project_root/Event-DataCollection"
"$project_root/isaacsim601/env/bin/python" scripts/collect_piper_conveyor_evis.py \
  --output "$output_dir" --episodes 1 --max-attempts 1 --probe-steps 25 --repo-id local/piper_conveyor_interface_probe \
  --device cpu --headless --enable_cameras "$@"
"$project_root/envs/lerobot/bin/python" scripts/validate_piper_conveyor_lerobot.py "$output_dir" --episodes 1 --allow-probe
