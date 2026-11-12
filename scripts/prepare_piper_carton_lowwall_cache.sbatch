#!/usr/bin/env bash
#SBATCH --job-name=piper-carton-lowwall-cache
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=33G
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_carton_lowwall_cache_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_carton_lowwall_cache_%j.log
set -euo pipefail
source deployment/runtime_env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$NEW_PROJECT_ROOT/envs/torchcodec_ffmpeg7/bin:$PATH"
export PYTHONPATH="$NEW_PROJECT_ROOT/OpenWAM${PYTHONPATH:+:$PYTHONPATH}"
run="${1:?replacement audit directory required}"
exec "$OWPY" "$run/rebuild_cache.py" --run "$run"
