#!/usr/bin/env bash
#SBATCH --job-name=ewm-action-only-ada2-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_action_only_ada2_80k_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_action_only_ada2_80k_%j.log

# Reproduce job180957, originally submitted through
# tmp/ewm_action_only_formal/train_179463_match.sbatch.
# Uses job179463's saved config, preserving all non-ablation training settings.
# sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_action_only.sh
# DRY_RUN=true bash <this-script>  # validate/print only; no outputs or training
# Historical resource: two RTX 6000 Ada. Override --gres when submitting elsewhere.
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
cd "$project_root/OpenWAM"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
unset WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED WANDB_NAME
export WANDB_MODE=online WANDB_PROJECT=CVPR2027-EWM
export WANDB_ENTITY=1909005872-nanyang-technological-university-singapore
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export EWM_DINO_PATH="$project_root/pretrained/dinov3-vitl16"
export EWM_T5_PATH="$project_root/pretrained/t5-small"
export LD_LIBRARY_PATH="$project_root/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
helper="$project_root/OpenWAM/scripts/train_ewm/train_ewm_action_only.py"
if [[ "${DRY_RUN:-false}" == true ]]; then
    exec "$project_root/envs/openwam/bin/python" "$helper"
fi
exec "$project_root/envs/openwam/bin/torchrun" --standalone --nproc_per_node=2 "$helper"
