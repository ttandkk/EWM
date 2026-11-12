#!/usr/bin/env bash
#SBATCH --job-name=ewm-simple-formal
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/ewm_simple_formal_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/ewm_simple_formal_%j.log
# Usage: sbatch BATCH/source/scripts/train_ewm_simple_a6000.sbatch BATCH MODE [resume]
# Submit only after worker cleanup and three-mode short-run acceptance.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'usage: sbatch <frozen launcher> <batch> <action_only|joint|rgb_action> [resume]' >&2
    exit 2
fi
batch=$(realpath "$1")
mode=$2
resume=${3:-fresh}
case "$mode" in
    action_only) config=train_ewm_eight_action_only ;;
    joint) config=train_ewm_eight ;;
    rgb_action) config=train_ewm_eight_rgb_action ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
case "$resume" in fresh|resume) ;; *) echo "Unknown run mode: $resume" >&2; exit 2 ;; esac
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EWM_EIGHT_CACHE="$root/tmp/training_data_cache/piper_dynamic_v1"
export EWM_CURRENT_DINO_CACHE="$EWM_EIGHT_CACHE/current_dino_fixed32"
export EWM_DINO_PATH="$root/pretrained/dinov3-vitl16" EWM_T5_PATH="$root/pretrained/t5-small"
export EWM_VAE_PATH="$root/pretrained/leanvae"
export EWM_EIGHT_STATS="$EWM_EIGHT_CACHE/shared_normalization.json" EWM_ACTION_STATS="$EWM_EIGHT_CACHE/shared_normalization.json"
export PYTHONPATH="$batch/source/OpenWAM"
export TMPDIR="/tmp/ewm-formal-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$batch/runs"
out="$batch/runs/$mode"
# One job per mode; lock is released even if the process dies.
exec 9>"$batch/runs/$mode.lock"
flock -n 9 || { echo "Concurrent run for $mode" >&2; exit 1; }
if [[ "$resume" == fresh ]]; then
    [[ ! -e "$out" ]] || { echo "Run exists; use explicit resume or a new batch: $out" >&2; exit 1; }
else
    [[ -d "$out" ]] || { echo "Missing original run: $out" >&2; exit 1; }
fi
"$OWPY" -B - "$batch/manifest.json" <<'CHECK_CACHE'
import json, sys
if not json.load(open(sys.argv[1])).get('cache_content_sha256'):
    raise RuntimeError('formal run requires --cache-content-sha256 at snapshot preparation')
CHECK_CACHE
"$OWPY" -B "$batch/source/scripts/prepare_ewm_simple_formal.py" verify --batch "$batch"
"$OWPY" -B - <<'CHECK'
import torch
if torch.cuda.device_count() != 2 or not all('rtx a6000' in torch.cuda.get_device_name(i).lower() for i in range(2)):
    raise RuntimeError('formal training requires two NVIDIA RTX A6000 GPUs')
print('FORMAL_GPUS_OK', flush=True)
CHECK
nvidia-smi --query-gpu=name,memory.total --format=csv
if [[ "$resume" == fresh ]]; then mkdir "$out"; fi
cd "$batch/source/OpenWAM"
options=("hydra.run.dir=$out/hydra_${SLURM_JOB_ID}" "training.output_path=$out"
    "dataloader.current_dino_cache_root=$EWM_CURRENT_DINO_CACHE"
    training.max_steps=80000 training.lr_schedule_steps=80000 training.batch_size=32
    training.gradient_accumulation_steps=1 training.mixed_precision=bf16 training.compile_core=false
    training.save_checkpoints=true training.save_steps=20000 training.save_full_states_for_resume=true
    training.keep_last_k_resume_states=2 training.keep_last_k_policy_checkpoints=4
    training.init_policy_checkpoint=null)
if [[ "$mode" == joint ]]; then options+=(dataloader.compact_events=true); fi
if [[ "$resume" == resume ]]; then options+=("training.resume_ckpt_path=$out"); fi
printf 'FORMAL_MODE=%s RUN=%s OUTPUT=%s JOB=%s\n' "$mode" "$resume" "$out" "$SLURM_JOB_ID"
"$OWPY" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train.py --config-name "$config" "${options[@]}"
