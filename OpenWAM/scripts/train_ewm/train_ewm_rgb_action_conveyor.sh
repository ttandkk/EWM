#!/usr/bin/env bash
#SBATCH --job-name=ewm-conveyor-rgb-ada2-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_rgb_action_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_rgb_action_%j.log

# Current dual RGB/state/text -> future dual RGB[k+1:k+25] + action[k:k+24].
# sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_rgb_action_conveyor.sh
# DRY_RUN=true bash <this-script> prints the command without starting training.
# RGB-action reads no event payloads; an event-frame cache is not used.
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
run_suffix=${SLURM_JOB_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
unset WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED WANDB_NAME
export TRAINING_MODE=rgb_action
export DATASET_DIR="${DATASET_DIR:-$project_root/datasets/piper_conveyor_front_wrist_v3_random_50/shards/job181776}"
export NORMALIZATION_STATS="${NORMALIZATION_STATS:-$DATASET_DIR/meta/ewm_normalization_stats.json}"
export NUM_GPUS="${NUM_GPUS:-2}" COMPILE_CORE="${COMPILE_CORE:-true}"
export RGB_CAMERA_MODE="${RGB_CAMERA_MODE:-dual}"
export BATCH_SIZE="${BATCH_SIZE:-32}" MAX_STEPS="${MAX_STEPS:-80000}"
export LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-80000}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-true}" SAVE_STEPS="${SAVE_STEPS:-10000}"
export SAVE_FULL_STATES_FOR_RESUME="${SAVE_FULL_STATES_FOR_RESUME:-true}"
export PURE_NOISE_EVAL_ENABLED=false RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
export RUN_NAME="${RUN_NAME:-conveyor_rgb_action_ada2_$run_suffix}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_root/OpenWAM/outputs/$RUN_NAME}"
export WANDB_MODE=online WANDB_PROJECT="${WANDB_PROJECT:-CVPR2027-EWM}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec bash "$project_root/OpenWAM/scripts/train_ewm/train_ewm_rgb_action.sh"
