#!/usr/bin/env bash
# One allocation, serial tasks/conditions. Planning never requests a GPU.
set -euo pipefail
script_path="$(realpath "${BASH_SOURCE[0]}")"
project_root="$(cd -- "$(dirname -- "$script_path")/.." && pwd -P)"
source "$project_root/deployment/runtime_env.sh"
client="$project_root/scripts/eval_piper_eight_dynamic.py"
for arg in "$@"; do
    if [[ "$arg" == --plan-only || "$arg" == --preflight || "$arg" == --help || "$arg" == -h ]]; then
        exec "$OWPY" "$client" "$@"
    fi
done
# Full manifest preflight before allocating a GPU.
"$OWPY" "$client" "$@" --preflight
if [[ -z "${SLURM_JOB_ID:-}" && "${EWM_EVAL_DIRECT:-0}" != 1 ]]; then
    mkdir -p "$project_root/logs"
    exec sbatch --partition="${EWM_EVAL_PARTITION:-cluster02}" \
        --gres="${EWM_EVAL_GRES:-gpu:6000ada:1}" --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --mem=64G --time="${EWM_EVAL_TIME:-08:00:00}" --job-name=piper-eight-eval \
        --output="$project_root/logs/piper_eight_eval_%j.log" \
        --error="$project_root/logs/piper_eight_eval_%j.log" "$script_path" "$@"
fi
exec "$OWPY" "$client" "$@"
