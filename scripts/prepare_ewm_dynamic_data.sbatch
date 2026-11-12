#!/usr/bin/env bash
#SBATCH --job-name=ewm-dynamic-cache
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/ewm_dynamic_cache_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/ewm_dynamic_cache_%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Cache generation and loader verification only; this script never starts training.
"$OWPY" "$root/OpenWAM/scripts/train_ewm/prepare_piper_dynamic_data.py" \
    --stage all --cache-root "${EWM_EIGHT_CACHE:-$root/tmp/training_data_cache/piper_dynamic_v1}" "$@"
