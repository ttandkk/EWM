#!/usr/bin/env bash
#SBATCH --job-name=piper-six-collect
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_six_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_six_%j.log
# Submit via collect_piper_six.sh so all tasks use the same immutable source snapshot.
set -euo pipefail
parallel="${PIPER_COLLECTION_PARALLEL:-2}"
group="${PIPER_COLLECTION_GROUP_INDEX:-}"
if [[ -n "$group" && ( "$parallel" != 2 || ( "$group" != 0 && "$group" != 1 ) ) ]]; then
    echo 'Split jobs require parallel=2 and group index 0 or 1' >&2
    exit 2
fi
if [[ "${1:-}" != --group ]]; then
    # Split jobs schedule one group independently. Never share a GPU between Isaac apps.
    workers="$parallel"
    if [[ -n "$group" ]]; then workers=1; fi
    # BASH_SOURCE is Slurm's node-local spool file here, unavailable on node 2.
    exec srun --kill-on-bad-exit=0 --ntasks="$workers" --ntasks-per-node=1 bash "${2:?output root}/source/collect_piper_six.sbatch" --group "$@"
fi
shift
root="${1:?project root}"
run="${2:?output root}"
shift 2
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT=4
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
export PYTHONPATH="$run/source/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$run/source/benchmark/src/piper_bench" ]]; then
    export PYTHONPATH="$run/source/benchmark/src:$run/source/scripts"
fi
tasks=(push_t drawer_cup square_nut moving_carton turntable_cups rotating_tube)
status=0
for (( index=${group:-${SLURM_PROCID:?rank required}}; index<${#tasks[@]}; index+=parallel )); do
    task="${tasks[$index]}"
    # Keep the original group IDs when selecting a subset, including afterok mappings.
    if [[ -n "${PIPER_COLLECTION_TASKS:-}" && ",${PIPER_COLLECTION_TASKS}," != *",${task},"* ]]; then
        continue
    fi
    output="$run/$task"
    if [[ -n "${PIPER_COLLECTION_DATASET_ROOT:-}" ]]; then
        output="$PIPER_COLLECTION_DATASET_ROOT/piper_$task"
    fi
    (
        # Kit performs many small cache/config writes during startup. Keep
        # disposable files node-local instead of contending on Ceph.
        runtime_dir="$(mktemp -d "/tmp/piper_${SLURM_JOB_ID}_${task}_XXXXXX")"
        trap 'rm -rf -- "$runtime_dir"' EXIT
        export OMNI_USER_CONFIG_PATH="$runtime_dir/config" XDG_CACHE_HOME="$runtime_dir/cache"
        export CUDA_CACHE_PATH="$runtime_dir/cuda" TMPDIR="$runtime_dir/tmp"
        export TRITON_CACHE_DIR="$runtime_dir/triton" TORCHINDUCTOR_CACHE_DIR="$runtime_dir/torchinductor"
        mkdir -p "$OMNI_USER_CONFIG_PATH" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
        "$SIMPY" "$run/source/scripts/collect_piper_tasks_evis.py" \
            --task "$task" --output "$output" --audit-dir "$run/${task}_collection_audit" --writer-python "$LRPY" \
            --device cpu --headless --enable_cameras \
            --kit_args="--/plugins/carb.tasking.plugin/threadCount=4" "$@" || exit "$?"
        "$LRPY" "$run/source/scripts/validate_piper_tasks_lerobot.py" "$output" \
            --episodes "$PIPER_COLLECTION_EPISODES"
    ) > "$run/logs/${task}_${SLURM_JOB_ID}.log" 2>&1 &
    # Wait before the next task on this GPU; other ranks keep running on failure.
    if ! wait "$!"; then status=1; fi
done
exit "$status"
