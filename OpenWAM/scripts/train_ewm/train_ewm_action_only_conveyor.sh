#!/usr/bin/env bash
#SBATCH --job-name=ewm-conveyor-ao-l40-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_action_only_l40_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_conveyor_action_only_l40_%j.log

# Conveyor action-only: current dual RGB/state/text -> action[k:k+24].
# sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_action_only_conveyor.sh
# ACTION_ONLY_MMAP_CACHE_DIR optionally maps predecoded RGB/state/actions.
# Event payloads are not read in action-only, so event mmap caching is inapplicable.
# The common launcher clears EVENT_FRAME_CACHE_DIR for this mode.
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
run_suffix=${SLURM_JOB_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
unset WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED WANDB_NAME
export TRAINING_MODE=action_only
export DATASET_DIR="${DATASET_DIR:-$project_root/datasets/piper_conveyor_front_wrist_v3_random_50/shards/job181776}"
export NORMALIZATION_STATS="${NORMALIZATION_STATS:-$DATASET_DIR/meta/ewm_normalization_stats.json}"
export NUM_GPUS="${NUM_GPUS:-2}"
export COMPILE_CORE="${COMPILE_CORE:-true}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export DIM="${DIM:-768}" ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-512}"
export HEADS="${HEADS:-8}" LAYERS="${LAYERS:-12}" CHUNK_SIZE="${CHUNK_SIZE:-24}"
export BATCH_SIZE="${BATCH_SIZE:-32}" MAX_STEPS="${MAX_STEPS:-80000}"
export LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-80000}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-true}" SAVE_STEPS="${SAVE_STEPS:-10000}"
export SAVE_FULL_STATES_FOR_RESUME="${SAVE_FULL_STATES_FOR_RESUME:-true}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
export PURE_NOISE_EVAL_ENABLED=false
export RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
export RUN_NAME="${RUN_NAME:-conveyor_action_only_l40_$run_suffix}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_root/OpenWAM/outputs/$RUN_NAME}"
export WANDB_MODE=online WANDB_PROJECT="${WANDB_PROJECT:-CVPR2027-EWM}"
export EWM_DINO_PATH="${EWM_DINO_PATH:-$project_root/pretrained/dinov3-vitl16}"
export EWM_T5_PATH="${EWM_T5_PATH:-$project_root/pretrained/t5-small}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LD_LIBRARY_PATH="$project_root/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec bash "$project_root/OpenWAM/scripts/train_ewm/train_ewm.sh"
