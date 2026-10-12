#!/usr/bin/env bash
#SBATCH --job-name=ewm-d512-l12-1bin-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_d512_l12_bin1_80k_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_d512_l12_bin1_80k_%j.log

# Submit from any directory:
#   sbatch /projects/hdd/myproject/CVPR2027/scripts/train_ewm_d768_l12_bin1_80k.sh
# Override a parameter without editing, for example:
#   sbatch --export=ALL,BATCH_SIZE=32,LEARNING_RATE=5e-5 scripts/train_ewm_d768_l12_bin1_80k.sh
set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
PYTHON="$ROOT/envs/openwam/bin/python"
CONFIG=${CONFIG:-train_ewm_d768_l12}
RUN_NAME=${RUN_NAME:-ewm-d512-l12-1bin-dual-scale10-80k-${SLURM_JOB_ID}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/OpenWAM/outputs/ewm_d512_l12_bin1_80k/job${SLURM_JOB_ID}}

# Model and event representation.
DIM=${DIM:-512}
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-12}
BASE_DIM=${BASE_DIM:-32}
VOXEL_BINS=${VOXEL_BINS:-1}
EVENT_CAMERA_MODE=${EVENT_CAMERA_MODE:-wrist_only} # dual | wrist_only
EVENT_SCALE=${EVENT_SCALE:-10.0}
CODEC_BATCH_SIZE=${CODEC_BATCH_SIZE:-16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DINO_INJECTION_INTERVAL=${DINO_INJECTION_INTERVAL:-1} # 1 = every block; T5 remains every block.
EVENT_SPATIAL_COMPRESSION=${EVENT_SPATIAL_COMPRESSION:-32} # 16 -> 16x16, 32 -> 8x8
EVENT_TEMPORAL_COMPRESSION=${EVENT_TEMPORAL_COMPRESSION:-1} # Future event steps only; must divide CHUNK_SIZE.
EVENT_CODEC_IMPL=${EVENT_CODEC_IMPL:-patch_only_2d}

# Full Piper training split: info.json currently defines episodes 0:50 (50 episodes).
DATASET_DIR=${DATASET_DIR:-$ROOT/datasets/lerobot_piper_stack_evis_uv_patterned_50_v3}
NORMALIZATION_STATS=${NORMALIZATION_STATS:-$ROOT/OpenWAM/outputs/ewm_validation/d768_l12_bin1/ewm_normalization_stats_bin1.json}
CHUNK_SIZE=${CHUNK_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}

# Optimizer, schedule, and durable checkpoints.
MAX_STEPS=${MAX_STEPS:-80000}
SAVE_STEPS=${SAVE_STEPS:-40000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.01}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-CVPR2027-EWM}

mkdir -p "$OUTPUT_DIR"
cd "$ROOT/OpenWAM"

ARGS=(
  --config-name "$CONFIG"
  model.dim="$DIM"
  model.heads="$HEADS"
  model.layers="$LAYERS"
  model.base_dim="$BASE_DIM"
  model.voxel_bins="$VOXEL_BINS"
  model.event_camera_mode="$EVENT_CAMERA_MODE"
  model.event_scale="$EVENT_SCALE"
  model.codec_batch_size="$CODEC_BATCH_SIZE"
  model.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
  model.dino_injection_interval="$DINO_INJECTION_INTERVAL"
  model.event_codec_impl="$EVENT_CODEC_IMPL"
  model.event_temporal_compression="$EVENT_TEMPORAL_COMPRESSION"
  dataloader.dataset_dir="$DATASET_DIR"
  dataloader.voxel_bins="$VOXEL_BINS"
  dataloader.event_camera_mode="$EVENT_CAMERA_MODE"
  dataloader.normalization_stats_path="$NORMALIZATION_STATS"
  dataloader.chunk_size="$CHUNK_SIZE"
  training.max_steps="$MAX_STEPS"
  model.event_spatial_compression="$EVENT_SPATIAL_COMPRESSION"
  training.save_steps="$SAVE_STEPS"
  training.save_checkpoints=true
  training.batch_size="$BATCH_SIZE"
  training.dataset_num_workers="$NUM_WORKERS"
  training.learning_rate="$LEARNING_RATE"
  training.weight_decay="$WEIGHT_DECAY"
  training.max_grad_norm="$MAX_GRAD_NORM"
  training.warmup_ratio="$WARMUP_RATIO"
  training.lr_min_ratio="$LR_MIN_RATIO"
  training.mixed_precision="$MIXED_PRECISION"
  training.output_path="$OUTPUT_DIR"
  project.seed="$SEED"
  project.wandb.project="$WANDB_PROJECT"
  project.wandb.run_name="$RUN_NAME"
)

# Optional: export WANDB_ENTITY=<team-or-user> at submission time.
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ARGS+=(project.wandb.entity="$WANDB_ENTITY")
fi

exec "$PYTHON" -u scripts/train.py "${ARGS[@]}"
