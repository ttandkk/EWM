#!/usr/bin/env bash
#SBATCH --job-name=ewm-p4-dual-ada2-80k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_p4_dual_ada2_80k_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_p4_dual_ada2_80k_%j.log

# Explicit event-action entry for the launcher used by job179463.
# Keeps the established train_ewm.sh implementation and its override interface.
# sbatch /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm_event_action.sh
# Historical resource: two RTX 6000 Ada. Override --gres when submitting elsewhere.
set -euo pipefail
if [[ "${TRAINING_MODE:-joint}" != joint ]]; then
    echo 'This event-action entry requires TRAINING_MODE=joint.' >&2
    exit 2
fi
export TRAINING_MODE=joint
export WANDB_MODE="${WANDB_MODE:-online}"
export PURE_NOISE_EVAL_ENABLED="${PURE_NOISE_EVAL_ENABLED:-false}"
exec bash /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm.sh
