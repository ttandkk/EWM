#!/usr/bin/env bash
#SBATCH --job-name=ewm-p8-profile
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_p8_profile_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_p8_profile_%j.log
set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
PROFILE_CODEC_BATCH_SIZE=${PROFILE_CODEC_BATCH_SIZE:-8}
if [[ "$PROFILE_CODEC_BATCH_SIZE" != 8 && "$PROFILE_CODEC_BATCH_SIZE" != 16 ]]; then
  echo 'PROFILE_CODEC_BATCH_SIZE must be 8 or 16.' >&2
  exit 2
fi

export WANDB_MODE=offline PYTHONUNBUFFERED=1
export NUM_GPUS=2 DIM=768 ACTION_HIDDEN_DIM=512 HEADS=8 LAYERS=12
export EVENT_CODEC_IMPL=leanvae16_joint EVENT_PATCH_SIZE=8 EVENT_SPATIAL_COMPRESSION=64
export EVENT_CAMERA_MODE=dual
export DATASET_DIR="$ROOT/datasets/piper_stack_fronttight_v3_50/job177683"
export BATCH_SIZE=32 CODEC_BATCH_SIZE="$PROFILE_CODEC_BATCH_SIZE" DATASET_NUM_WORKERS=4
export COMPILE_CORE=true GRADIENT_CHECKPOINTING=false
export MAX_STEPS=61 LR_SCHEDULE_STEPS=20000 SAVE_STEPS=0 SAVE_CHECKPOINTS=false
export LOG_EVERY=10 DIAGNOSTIC_EVERY=1000 PURE_NOISE_EVAL_ENABLED=false
export PROFILE_SEGMENTS=true PROFILE_WARMUP_STEPS=20 PROFILE_LOG_EVERY=20
export RUN_NAME="ewm-p8-dual-profile-c${PROFILE_CODEC_BATCH_SIZE}-${SLURM_JOB_ID}"
export OUTPUT_DIR="$ROOT/tmp/ewm_p8_dual_profile/results/c${PROFILE_CODEC_BATCH_SIZE}_job${SLURM_JOB_ID}"

exec bash "$ROOT/OpenWAM/scripts/train_ewm/train_ewm_d512_l12_k24_t2_wrist_ada2_80k.sh"
