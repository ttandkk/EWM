#!/usr/bin/env bash
#SBATCH --job-name=piper-narrow-collect
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_narrow_collect_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_narrow_collect_%j.log
set -euo pipefail
source /projects/_hdd/isaachdd/EWM/deployment/runtime_env.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$NEW_PROJECT_ROOT/envs/torchcodec_ffmpeg7/bin:$PATH"
run_dir="${1:?prepared run directory required}"
ulimit -c 0
exec "$OWPY" "$run_dir/launcher.py" --run-dir "$run_dir" --stage collect
