#!/usr/bin/env bash
# Submit selected new-task datasets; one Isaac process per GPU.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
episodes=50
attempts=300
acceptance=10
split=train
run=""
dataset_root=""
resume=0
dry=0
smoke=0
parallel=2
split_jobs=0
after_jobs=""
all_tasks=(push_t drawer_cup square_nut moving_carton turntable_cups rotating_tube)
task_list="$(IFS=,; printf '%s' "${all_tasks[*]}")"
while (( $# )); do
    case "$1" in
        --output-root) run="${2:?output path required}"; shift 2 ;;
        --dataset-root) dataset_root="${2:?dataset parent required}"; shift 2 ;;
        --episodes) episodes="${2:?episode count required}"; shift 2 ;;
        --max-attempts) attempts="${2:?attempt count required}"; shift 2 ;;
        --acceptance-trials) acceptance="${2:?trial count required}"; shift 2 ;;
        --split) split="${2:?split required}"; shift 2 ;;
        --parallel-tasks) parallel="${2:?parallel task count required}"; shift 2 ;;
        --split-jobs) split_jobs=1; shift ;;
        --after-jobs) after_jobs="${2:?two parent job IDs required}"; shift 2 ;;
        --tasks) task_list="${2:?comma-separated task names required}"; shift 2 ;;
        --resume) resume=1; shift ;;
        --dry-run) dry=1; shift ;;
        --smoke) smoke=1; episodes=1; attempts=10; acceptance=3; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/collect_piper_six.sh [--output-root DIR] [--resume] [--dry-run] [--smoke]'
            echo 'Defaults: submit all 6 tasks, run 2 at once on 2 Pro6000 GPUs, each 50 successes; first 10 acceptance trials, >=90% required.'
            echo 'Options: --episodes N --max-attempts N --acceptance-trials N --split train|validation'
            echo '--output-root DIR: run logs/snapshot; --dataset-root DIR: final data parent (default datasets, or datasets/validation).'
            echo 'Formal data: <dataset-root>/piper_<task>/{data,meta,videos}; smoke data stays in the run directory.'
            echo '--parallel-tasks 6: one GPU per task, six tasks at once, using the preemptible override-limits-but-killable QOS.'
            echo '--split-jobs: submit two independent single-GPU jobs, each running its selected tasks in sequence.'
            echo '--after-jobs JOB0,JOB1: with --split-jobs, start each group only after its matching smoke job succeeds.'
            echo '--tasks NAME,...: select new tasks, preserving their original group assignment; defaults to all six.'
            echo 'GPU/partition overrides: PIPER_GPU_GRES=gpu:pro6000:1 PIPER_PARTITION=cluster02'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$episodes" =~ ^[1-9][0-9]*$ && "$attempts" =~ ^[1-9][0-9]*$ && "$acceptance" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Counts must be positive integers' >&2; exit 2;
}
(( attempts >= episodes )) || { echo 'max-attempts must be >= episodes' >&2; exit 2; }
[[ "$split" == train || "$split" == validation ]] || { echo 'Invalid split' >&2; exit 2; }
[[ "$parallel" == 2 || "$parallel" == 6 ]] || { echo '--parallel-tasks must be 2 or 6' >&2; exit 2; }
(( split_jobs == 0 || parallel == 2 )) || { echo '--split-jobs requires --parallel-tasks 2' >&2; exit 2; }
[[ "$task_list" != ,* && "$task_list" != *, && "$task_list" != *,,* ]] || { echo 'Task names must be nonempty' >&2; exit 2; }
IFS=, read -r -a requested_tasks <<< "$task_list"
declare -A selected=()
for task in "${requested_tasks[@]}"; do
    case "$task" in
        push_t|drawer_cup|square_nut|moving_carton|turntable_cups|rotating_tube) ;;
        *) echo "Unknown task: $task" >&2; exit 2 ;;
    esac
    [[ -z "${selected[$task]+present}" ]] || { echo "Duplicate task: $task" >&2; exit 2; }
    selected[$task]=1
done
ordered_tasks=()
group_tasks=("" "")
for index in "${!all_tasks[@]}"; do
    task="${all_tasks[$index]}"
    if [[ -n "${selected[$task]+present}" ]]; then
        ordered_tasks+=("$task")
        group_tasks[$((index % 2))]+=" $task"
    fi
done
(( parallel != 6 || ${#ordered_tasks[@]} == 6 )) || { echo 'Task subsets require --parallel-tasks 2' >&2; exit 2; }
[[ -n "${group_tasks[0]}" && -n "${group_tasks[1]}" ]] || { echo 'Select at least one task from each GPU group' >&2; exit 2; }
task_list="$(IFS=,; printf '%s' "${ordered_tasks[*]}")"
parents=()
if [[ -n "$after_jobs" ]]; then
    (( split_jobs == 1 && smoke == 0 )) || { echo '--after-jobs requires --split-jobs without --smoke' >&2; exit 2; }
    [[ "$after_jobs" =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] || { echo '--after-jobs needs two positive job IDs: JOB0,JOB1' >&2; exit 2; }
    IFS=, read -r -a parents <<< "$after_jobs"
    [[ "${parents[0]}" != "${parents[1]}" ]] || { echo 'Each group must depend on a different smoke job' >&2; exit 2; }
fi
if [[ -z "$run" ]]; then
    (( resume == 0 )) || { echo '--resume requires --output-root' >&2; exit 2; }
    parent="$root/tmp/piper_collection_runs"
    if (( smoke )); then parent="$root/tmp/piper_six_collection_smoke"; fi
    run="$parent/${split}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
run="$(realpath -m -- "$run")"
if (( smoke )); then
    [[ -z "$dataset_root" ]] || { echo '--dataset-root is for formal collection, not smoke' >&2; exit 2; }
else
    if [[ -z "$dataset_root" ]]; then
        dataset_root="$root/datasets"
        if [[ "$split" == validation ]]; then dataset_root+="/validation"; fi
    fi
    dataset_root="$(realpath -m -- "$dataset_root")"
fi
export PIPER_COLLECTION_DATASET_ROOT="$dataset_root"
args=(--episodes "$episodes" --max-attempts "$attempts" --acceptance-trials "$acceptance" --split "$split")
if (( resume )); then args+=(--resume); fi
export PIPER_COLLECTION_EPISODES="$episodes"
export PIPER_COLLECTION_PARALLEL="$parallel"
export PIPER_COLLECTION_TASKS="$task_list"
unset PIPER_COLLECTION_GROUP_INDEX
nodes="$parallel"
groups=("")
if (( split_jobs )); then nodes=1; groups=(0 1); fi
submit=(sbatch --parsable --nodes="$nodes" --ntasks-per-node=1 --partition="${PIPER_PARTITION:-cluster02}"
    --gres="${PIPER_GPU_GRES:-gpu:pro6000:1}" --chdir="$root"
    --output="$run/logs/slurm_%j.log" --error="$run/logs/slurm_%j.log")
if (( smoke )); then submit+=(--time=01:00:00); fi
if (( parallel == 6 )); then submit+=(--qos=override-limits-but-killable); fi
printf 'Tasks: %s\n' "${ordered_tasks[*]}"
printf 'Output: %s\nEach: %s successful episodes, %s acceptance trials\n' "$run" "$episodes" "$acceptance"
printf 'Dataset parent: %s\n' "${dataset_root:-$run (smoke only)}"
printf 'Concurrency: %s tasks, one Isaac process per GPU\n' "$parallel"
if (( split_jobs )); then printf 'Two independent jobs: group 0 (%s), group 1 (%s)\n' "${group_tasks[0]# }" "${group_tasks[1]# }"; fi
if (( dry )); then
    for group in "${groups[@]}"; do
        command=("${submit[@]}")
        if [[ -n "$group" ]]; then command+=(--job-name="piper-six-g$group" --export="ALL,PIPER_COLLECTION_GROUP_INDEX=$group"); fi
        if [[ -n "$after_jobs" ]]; then command+=(--dependency="afterok:${parents[$group]}" --kill-on-invalid-dep=yes); fi
        printf '%q ' "${command[@]}" "$run/source/collect_piper_six.sbatch" "$root" "$run" "${args[@]}"
        printf '\n'
    done
    exit 0
fi
source "$root/deployment/runtime_env.sh"
"$SIMPY" - "$root" "$run" "$resume" "$episodes" "$attempts" "$acceptance" "$split" "$parallel" "$split_jobs" "$task_list" "$dataset_root" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root, run = map(Path, sys.argv[1:3])
resume = int(sys.argv[3])
settings = dict(episodes=int(sys.argv[4]), acceptance_trials=int(sys.argv[6]), split=sys.argv[7], parallel_tasks=int(sys.argv[8]))
if int(sys.argv[9]):
    settings['split_jobs'] = True
selected_tasks = sys.argv[10].split(',')
dataset_root = Path(sys.argv[11]) if sys.argv[11] else None
if dataset_root is not None:
    settings['dataset_root'] = str(dataset_root)
if len(selected_tasks) != 6:
    settings['tasks'] = selected_tasks
if resume:
    if json.loads((run/'launch.json').read_text()) != settings:
        raise SystemExit('Resume must keep tasks, episodes, acceptance trials, split and dispatch mode unchanged')
    hashes = json.loads((run/'source_sha256.json').read_text())
    if any(hashlib.sha256((run/'source'/p).read_bytes()).hexdigest() != h for p,h in hashes.items()):
        raise SystemExit('Source snapshot changed; resume refused')
else:
    if dataset_root is not None:
        existing = [str(dataset_root/('piper_'+t)) for t in selected_tasks
                    if (dataset_root/('piper_'+t)).exists()]
        if existing:
            raise SystemExit('Datasets already exist; choose another --dataset-root or resume the original run: '+', '.join(existing))
    run.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'benchmark/src'))
    from piper_bench.collection.freeze import freeze_collection_source
    freeze_collection_source(root, run/'source')
    shutil.copy2(root/'scripts/collect_piper_six.sbatch', run/'source/collect_piper_six.sbatch')
    shutil.copy2(root/'scripts/collect_piper_six.sh', run/'source/collect_piper_six.sh')
    shutil.copy2(root/'docs/piper_eight_task_collection_eval_protocol_v1.md', run/'source/protocol.md')
    files = sorted(p for p in (run/'source').rglob('*') if p.is_file())
    (run/'source_sha256.json').write_text(json.dumps({str(p.relative_to(run/'source')): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    (run/'launch.json').write_text(json.dumps(settings, indent=2)+'\n')
(run/'logs').mkdir(exist_ok=True)
PY
for group in "${groups[@]}"; do
    command=("${submit[@]}")
    if [[ -n "$group" ]]; then command+=(--job-name="piper-six-g$group" --export="ALL,PIPER_COLLECTION_GROUP_INDEX=$group"); fi
    if [[ -n "$after_jobs" ]]; then command+=(--dependency="afterok:${parents[$group]}" --kill-on-invalid-dep=yes); fi
    if ! job="$("${command[@]}" "$run/source/collect_piper_six.sbatch" "$root" "$run" "${args[@]}")"; then
        printf 'Submission failed. Any earlier submissions are recorded in %s/submitted_jobs.txt\n' "$run" >&2
        exit 1
    fi
    printf '%s\n' "$job" >> "$run/submitted_jobs.txt"
    if [[ -n "$group" ]]; then printf '%s\t%s\n' "$group" "$job" >> "$run/submitted_groups.tsv"; fi
    if [[ -n "$after_jobs" ]]; then
        printf '%s\t%s\t%s\n' "$group" "$job" "${parents[$group]}" >> "$run/submitted_dependencies.tsv"
        printf 'Group %s waits for successful smoke job %s; a failed dependency cancels this collection job.\n' "$group" "${parents[$group]}"
    fi
    printf 'Submitted job %s (%s GPUs, group %s). Logs: %s/logs/\n' "$job" "$nodes" "${group:-all}" "$run"
done
