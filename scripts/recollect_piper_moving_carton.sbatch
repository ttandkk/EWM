#!/usr/bin/env bash
#SBATCH --job-name=piper-carton-collect
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_carton_collect_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_carton_collect_%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
ulimit -c 0
run_dir="${1:?prepared run directory required}"
exec "$OWPY" "$run_dir/launcher.py" --run-dir "$run_dir"
