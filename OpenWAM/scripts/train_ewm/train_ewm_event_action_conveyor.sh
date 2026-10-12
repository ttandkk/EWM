#!/usr/bin/env bash
#SBATCH --job-name=ewm-conveyor-ea
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_ea_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_ea_%j.log

# Conveyor dataset overrides; all other settings come from the original launcher.
# sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_event_action_conveyor.sh
# To change GPU count, override both sbatch --gres and the NUM_GPUS environment variable.
set -euo pipefail

project_root=/projects/hdd/myproject/CVPR2027
conveyor_run_suffix=${SLURM_JOB_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}

export DATASET_DIR="${DATASET_DIR:-$project_root/datasets/piper_conveyor_front_wrist_v3_random_50/shards/job181776}"
export NORMALIZATION_STATS="${NORMALIZATION_STATS:-$DATASET_DIR/meta/ewm_normalization_stats.json}"
# Literal null makes the loader read event PNGs without requiring a frame cache.
export EVENT_FRAME_CACHE_DIR="${EVENT_FRAME_CACHE_DIR:-null}"
export COMPILE_CORE="${COMPILE_CORE:-false}"
export RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
export PURE_NOISE_EVAL_ENABLED="${PURE_NOISE_EVAL_ENABLED:-false}"
export NUM_GPUS="${NUM_GPUS:-2}"
export RUN_NAME="${RUN_NAME:-conveyor_event_action_$conveyor_run_suffix}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_root/OpenWAM/outputs/$RUN_NAME}"

exec bash "$project_root/OpenWAM/scripts/train_ewm/train_ewm_event_action.sh"
