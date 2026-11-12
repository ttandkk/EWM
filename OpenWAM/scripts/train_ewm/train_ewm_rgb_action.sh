#!/usr/bin/env bash
#SBATCH --job-name=ewm-rgb-p4-dual-ada2-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_rgb_p4_dual_ada2_80k_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_rgb_p4_dual_ada2_80k_%j.log

# Independent RGB-action counterpart of train_ewm.sh; event/action-only scripts are unchanged.
#   sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_rgb_action.sh
#   DRY_RUN=true bash <this-script>  # print resolved command without creating outputs/training
# Optional: RGB_CAMERA_MODE=wrist_only (default dual), RGB_DECODE_BACKEND=torchcodec.
# All optimizer, schedule, model geometry, BF16, W&B and checkpoint defaults match joint.
set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
OPENWAM="$ROOT/OpenWAM"
ENV_BIN="$ROOT/envs/openwam/bin"
TORCHRUN="$ENV_BIN/torchrun"

# Distributed launch and run identity.
if [[ "${TRAINING_MODE:-rgb_action}" != rgb_action ]]; then
  echo "This independent launcher requires TRAINING_MODE=rgb_action." >&2
  exit 2
fi
TRAINING_MODE=rgb_action
TRAIN_CONFIG=train_ewm_rgb_action_piper
DEFAULT_RUN_NAME=ewm-rgb-lean16-p4-d768-a512-dual-ada2-80k
DEFAULT_OUTPUT_NAME=ewm_rgb_action_lean16_p4_d768_a512_dual_ada2_80k
DEFAULT_COMPILE_CORE=true
NUM_GPUS=${NUM_GPUS:-2}
RUN_SUFFIX=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-${DEFAULT_RUN_NAME}-${RUN_SUFFIX}}
OUTPUT_DIR=${OUTPUT_DIR:-$OPENWAM/outputs/${DEFAULT_OUTPUT_NAME}/job${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-CVPR2027-EWM}
WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-online}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Default: the former P4 / D768 / dual-view 80k Ada2 production launch.
# Model and RGB-video token representation (legacy event_* geometry field names).
DIM=${DIM:-768}
ACTION_HIDDEN_DIM=${ACTION_HIDDEN_DIM:-512}
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-12}
VOXEL_BINS=${VOXEL_BINS:-1}
EVENT_CAMERA_MODE=${RGB_CAMERA_MODE:-${EVENT_CAMERA_MODE:-dual}}  # wrist_only | dual
EVENT_CODEC_IMPL=${EVENT_CODEC_IMPL:-leanvae16_rgb_joint}
EVENT_PATCH_SIZE=${EVENT_PATCH_SIZE:-4}
case "$EVENT_CODEC_IMPL" in
  leanvae16_rgb_joint) CODEC_SPATIAL_STRIDE=8; DEFAULT_EVENT_STRIDE=32; DEFAULT_VAE_PATH=$ROOT/pretrained/leanvae ;;
  *) echo "RGB-action supports only leanvae16_rgb_joint; got: $EVENT_CODEC_IMPL" >&2; exit 2 ;;
esac
if [[ -n "${EVENT_PATCH_SIZE:-}" ]]; then
  EVENT_SPATIAL_COMPRESSION=${EVENT_SPATIAL_COMPRESSION:-$((CODEC_SPATIAL_STRIDE * EVENT_PATCH_SIZE))}
else
  EVENT_SPATIAL_COMPRESSION=${EVENT_SPATIAL_COMPRESSION:-$DEFAULT_EVENT_STRIDE}
  EVENT_PATCH_SIZE=$((EVENT_SPATIAL_COMPRESSION / CODEC_SPATIAL_STRIDE))
fi
EVENT_TEMPORAL_COMPRESSION=${EVENT_TEMPORAL_COMPRESSION:-4}
EVENT_TEMPORAL_DOWNSAMPLE=${EVENT_TEMPORAL_DOWNSAMPLE:-1}
CODEC_BATCH_SIZE=${CODEC_BATCH_SIZE:-8}
DINO_INJECTION_INTERVAL=${DINO_INJECTION_INTERVAL:-1}

# Memory and input pipeline.
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
GRADIENT_CHECKPOINTING_SKIP_FIRST_BLOCKS=${GRADIENT_CHECKPOINTING_SKIP_FIRST_BLOCKS:-0}
COMPILE_CORE=${COMPILE_CORE:-$DEFAULT_COMPILE_CORE}
PROFILE_SEGMENTS=${PROFILE_SEGMENTS:-false}
PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-20}
PROFILE_LOG_EVERY=${PROFILE_LOG_EVERY:-20}
GPU_VOXELIZATION=false
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-4}
EVENT_CACHE_SIZE=${EVENT_CACHE_SIZE:-8}
EVENT_FRAME_CACHE_DIR=
RGB_DECODE_BACKEND=${RGB_DECODE_BACKEND:-pyav}
FFMPEG7_LIB="$ROOT/envs/torchcodec_ffmpeg7/lib"
# Also supplies libGL required by OpenCV imported by the shared dataset registry.
if [[ -d "$FFMPEG7_LIB" ]]; then
  export LD_LIBRARY_PATH="$FFMPEG7_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "$RGB_DECODE_BACKEND" == torchcodec ]]; then
  if [[ ! -f "$FFMPEG7_LIB/libavutil.so.59" ]]; then
    echo "Missing FFmpeg 7 library for TorchCodec: $FFMPEG7_LIB" >&2
    exit 2
  fi
fi

# Dataset, optimizer, schedule, and checkpointing.
DATASET_DIR=${DATASET_DIR:-${EWM_DATASET_DIR:-$ROOT/datasets/piper_stack_fronttight_v3_50/job177683}}
if [[ -z "$DATASET_DIR" || ! -f "$DATASET_DIR/meta/ewm_normalization_stats.json" ]]; then
  echo 'Set DATASET_DIR to an aligned event-frame LeRobot export with meta/ewm_normalization_stats.json.' >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/meta/piper_temporal_contract.json" ]] || ! grep -q 'aligned_event_frame_pre_action_v1' "$DATASET_DIR/meta/piper_temporal_contract.json"; then
  echo "Dataset lacks the aligned event-frame temporal contract: $DATASET_DIR" >&2
  exit 2
fi
NORMALIZATION_STATS=${NORMALIZATION_STATS:-$DATASET_DIR/meta/ewm_normalization_stats.json}
if [[ ! -f "$NORMALIZATION_STATS" ]]; then
  echo "Missing normalization stats: $NORMALIZATION_STATS" >&2
  exit 2
fi
CHUNK_SIZE=${CHUNK_SIZE:-24}
BATCH_SIZE=${BATCH_SIZE:-32}  # Per GPU; global batch is BATCH_SIZE * NUM_GPUS.
MAX_STEPS=${MAX_STEPS:-80000}
LR_SCHEDULE_STEPS=${LR_SCHEDULE_STEPS:-$MAX_STEPS}
SAVE_STEPS=${SAVE_STEPS:-10000}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-true}
SAVE_FULL_STATES_FOR_RESUME=${SAVE_FULL_STATES_FOR_RESUME:-true}
RESUME_CKPT_PATH=${RESUME_CKPT_PATH:-}
KEEP_LAST_K_RESUME_STATES=${KEEP_LAST_K_RESUME_STATES:-4}
KEEP_LAST_K_POLICY_CHECKPOINTS=${KEEP_LAST_K_POLICY_CHECKPOINTS:-4}
if [[ -n "$RESUME_CKPT_PATH" ]]; then
  OUTPUT_DIR="$RESUME_CKPT_PATH"
fi
PURE_NOISE_EVAL_ENABLED=${PURE_NOISE_EVAL_ENABLED:-false}
PURE_NOISE_EVAL_EVERY=${PURE_NOISE_EVAL_EVERY:-10000}
PURE_NOISE_EVAL_PARTITION=${PURE_NOISE_EVAL_PARTITION:-cluster02}
PURE_NOISE_EVAL_GRES=${PURE_NOISE_EVAL_GRES:-gpu:l40:1}
PURE_NOISE_EVAL_TIME=${PURE_NOISE_EVAL_TIME:-01:00:00}
LOG_EVERY=${LOG_EVERY:-10}
DIAGNOSTIC_EVERY=${DIAGNOSTIC_EVERY:-200}
DIAGNOSTIC_SAMPLES=${DIAGNOSTIC_SAMPLES:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.01}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-42}

if (( NUM_GPUS <= 0 || CHUNK_SIZE <= 0 || EVENT_TEMPORAL_COMPRESSION <= 0 )); then
  echo 'NUM_GPUS, CHUNK_SIZE, and EVENT_TEMPORAL_COMPRESSION must be positive.' >&2
  exit 2
fi
if (( CHUNK_SIZE % EVENT_TEMPORAL_COMPRESSION != 0 )); then
  echo "CHUNK_SIZE=$CHUNK_SIZE must be divisible by EVENT_TEMPORAL_COMPRESSION=$EVENT_TEMPORAL_COMPRESSION." >&2
  exit 2
fi
if [[ ! "$EVENT_TEMPORAL_DOWNSAMPLE" =~ ^[123]$ ]]; then
  echo 'EVENT_TEMPORAL_DOWNSAMPLE must be 1/2/3.' >&2
  exit 2
fi
if (( CHUNK_SIZE % (EVENT_TEMPORAL_COMPRESSION * EVENT_TEMPORAL_DOWNSAMPLE) != 0 )); then
  echo 'CHUNK_SIZE must be divisible by codec compression times temporal downsample; no frame padding.' >&2
  exit 2
fi
if [[ "$VOXEL_BINS" != 1 || "$EVENT_TEMPORAL_COMPRESSION" != 4 ]] ||
   (( EVENT_PATCH_SIZE != 2 && EVENT_PATCH_SIZE != 4 && EVENT_PATCH_SIZE != 8 )) ||
   (( EVENT_SPATIAL_COMPRESSION != CODEC_SPATIAL_STRIDE * EVENT_PATCH_SIZE )); then
  echo 'RGB joint codec requires bins=1 metadata, temporal stride=4 and a matching latent patch2/4/8.' >&2
  exit 2
fi
EVENT_VAE_PATH=${EWM_VAE_PATH:-$DEFAULT_VAE_PATH}
if [[ ! -d "$EVENT_VAE_PATH" ]]; then
  echo "Missing pretrained VAE directory: $EVENT_VAE_PATH" >&2
  exit 2
fi

ARGS=(
  training.max_steps="$MAX_STEPS"
  training.lr_schedule_steps="$LR_SCHEDULE_STEPS"
  training.batch_size="$BATCH_SIZE"
  training.save_steps="$SAVE_STEPS"
  training.save_checkpoints="$SAVE_CHECKPOINTS"
  training.save_full_states_for_resume="$SAVE_FULL_STATES_FOR_RESUME"
  training.resume_ckpt_path="${RESUME_CKPT_PATH:-null}"
  training.keep_last_k_resume_states="$KEEP_LAST_K_RESUME_STATES"
  training.keep_last_k_policy_checkpoints="$KEEP_LAST_K_POLICY_CHECKPOINTS"
  training.pure_noise_eval.enabled="$PURE_NOISE_EVAL_ENABLED"
  training.pure_noise_eval.every_steps="$PURE_NOISE_EVAL_EVERY"
  training.pure_noise_eval.partition="$PURE_NOISE_EVAL_PARTITION"
  training.pure_noise_eval.gres="$PURE_NOISE_EVAL_GRES"
  training.pure_noise_eval.time="$PURE_NOISE_EVAL_TIME"
  training.log_every="$LOG_EVERY"
  training.diagnostic_every="$DIAGNOSTIC_EVERY"
  training.learning_rate="$LEARNING_RATE"
  training.weight_decay="$WEIGHT_DECAY"
  training.max_grad_norm="$MAX_GRAD_NORM"
  training.warmup_ratio="$WARMUP_RATIO"
  training.lr_min_ratio="$LR_MIN_RATIO"
  training.mixed_precision="$MIXED_PRECISION"
  training.compile_core="$COMPILE_CORE"
  training.profile_segments="$PROFILE_SEGMENTS"
  training.profile_warmup_steps="$PROFILE_WARMUP_STEPS"
  training.profile_log_every="$PROFILE_LOG_EVERY"
  training.dataset_num_workers="$DATASET_NUM_WORKERS"
  training.output_path="$OUTPUT_DIR"
  model.training_mode=rgb_action
  model.dim="$DIM"
  model.action_hidden_dim="$ACTION_HIDDEN_DIM"
  model.heads="$HEADS"
  model.layers="$LAYERS"
  model.voxel_bins="$VOXEL_BINS"
  model.event_camera_mode="$EVENT_CAMERA_MODE"
  model.event_codec_impl="$EVENT_CODEC_IMPL"
  model.event_vae_path="$EVENT_VAE_PATH"
  model.event_spatial_compression="$EVENT_SPATIAL_COMPRESSION"
  model.event_patch_size="$EVENT_PATCH_SIZE"
  model.event_temporal_compression="$EVENT_TEMPORAL_COMPRESSION"
  model.event_temporal_downsample="$EVENT_TEMPORAL_DOWNSAMPLE"
  model.codec_batch_size="$CODEC_BATCH_SIZE"
  model.diagnostic_samples="$DIAGNOSTIC_SAMPLES"
  model.dino_injection_interval="$DINO_INJECTION_INTERVAL"
  model.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
  model.gradient_checkpointing_skip_first_blocks="$GRADIENT_CHECKPOINTING_SKIP_FIRST_BLOCKS"
  dataloader.observation_row_offset=0
  dataloader.dataset_dir="$DATASET_DIR"
  dataloader.voxel_bins="$VOXEL_BINS"
  dataloader.event_camera_mode="$EVENT_CAMERA_MODE"
  dataloader.chunk_size="$CHUNK_SIZE"
  dataloader.gpu_voxelization="$GPU_VOXELIZATION"
  dataloader.event_cache_size="$EVENT_CACHE_SIZE"
  dataloader.event_frame_cache_dir="${EVENT_FRAME_CACHE_DIR:-null}"
  dataloader.rgb_decode_backend="$RGB_DECODE_BACKEND"
  dataloader.normalization_stats_path="$NORMALIZATION_STATS"
  project.seed="$SEED"
  project.wandb.project="$WANDB_PROJECT"
  project.wandb.run_name="$RUN_NAME"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  ARGS+=(project.wandb.entity="$WANDB_ENTITY")
fi

if [[ "$PURE_NOISE_EVAL_ENABLED" != false ]]; then
  echo 'RGB-action does not support the event-only pure_noise_eval callback.' >&2
  exit 2
fi
if [[ "$EVENT_CAMERA_MODE" != dual && "$EVENT_CAMERA_MODE" != wrist_only ]]; then
  echo 'RGB_CAMERA_MODE must be dual or wrist_only.' >&2
  exit 2
fi
if [[ "${DRY_RUN:-false}" == true ]]; then
  printf '%q ' "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" scripts/train_ewm/train_ewm.py --config-name "$TRAIN_CONFIG" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
cd "$OPENWAM"
exec "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" scripts/train_ewm/train_ewm.py --config-name "$TRAIN_CONFIG" "${ARGS[@]}"
