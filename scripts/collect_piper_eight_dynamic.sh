#!/usr/bin/env bash
# Run inside one allocated GPU; this entry point does not submit Slurm jobs.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=4
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
ulimit -c 0
exec "$SIMPY" "$root/scripts/collect_piper_eight_dynamic.py" "$@"
