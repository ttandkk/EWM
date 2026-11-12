#!/usr/bin/env bash
#SBATCH --job-name=piper-lighting
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_lighting_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_lighting_%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=4
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
export PYTHONPATH="$root/Event-DataCollection/scripts"
output="${1:?output directory required}"
shift
tasks=("$@")
if (( ${#tasks[@]} == 0 )); then
    tasks=(conveyor moving_carton turntable_cups rotating_tube tennis_rolling tennis_falling tennis_parabolic tennis_container)
fi
mkdir -p "$output"
for task in "${tasks[@]}"; do
    srun --ntasks=1 "$SIMPY" "$root/Event-DataCollection/scripts/render_piper_lighting.py" \
        --task "$task" --output-dir "$output/$task" --headless --enable_cameras --device cpu
done
