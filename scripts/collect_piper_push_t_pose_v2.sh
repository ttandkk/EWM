#!/usr/bin/env bash
# Freeze one Push-T revision, gate 50 demonstrations on boundary/RGB/event checks.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
run=""
dataset_root="$root/datasets"
seed=20261016
dry=0
while (( $# )); do
    case "$1" in
        --output-root) run="${2:?output directory required}"; shift 2 ;;
        --dataset-root) dataset_root="${2:?dataset parent required}"; shift 2 ;;
        --seed) seed="${2:?seed required}"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/collect_piper_push_t_pose_v2.sh [--output-root DIR] [--seed N] [--dry-run]'
            echo 'L40: 35/35 boundary cases, >=9/10 ID cases, 1 RGB/event export; then 50 successful episodes.'
            echo 'Override hardware with PIPER_GPU_GRES and PIPER_PARTITION. Existing output is never overwritten.'
            echo '--output-root stores logs/snapshots/manifests; --dataset-root stores piper_push_t (default datasets).'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'seed must be nonnegative' >&2; exit 2; }
if [[ -z "$run" ]]; then run="$root/tmp/piper_push_t_pose_v2/train_$(date -u +%Y%m%dT%H%M%SZ)"; fi
run="$(realpath -m -- "$run")"
dataset_root="$(realpath -m -- "$dataset_root")"
source "$root/deployment/runtime_env.sh"
export OPENBLAS_NUM_THREADS=1
"$SIMPY" - "$root" "$run" "$seed" "$dry" "$dataset_root" <<'PY'
import hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path
root, run = map(Path, sys.argv[1:3]); seed, dry = map(int, sys.argv[3:5])
dataset_root = Path(sys.argv[5])

def command(directory, episodes, dependency=None):
    result = ['sbatch', '--parsable', '--nodes=1', '--ntasks-per-node=1',
        '--partition='+os.environ.get('PIPER_PARTITION', 'cluster02'),
        '--gres='+os.environ.get('PIPER_GPU_GRES', 'gpu:l40:1'),
        '--time='+('01:00:00' if episodes == 1 else '24:00:00'),
        '--job-name='+('pusht-pose-v2-smoke' if episodes == 1 else 'pusht-pose-v2-50'),
        '--chdir='+str(root), '--output='+str(directory/'logs/slurm_%j.log'),
        '--error='+str(directory/'logs/slurm_%j.log'),
        '--export=ALL,PIPER_COLLECTION_GROUP_INDEX=0,PIPER_COLLECTION_PARALLEL=2,'
        'PIPER_COLLECTION_TASKS=push_t,PIPER_COLLECTION_EPISODES='+str(episodes)+
        ',PIPER_COLLECTION_DATASET_ROOT='+(str(dataset_root) if episodes != 1 else '')]
    if dependency:
        result += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
    return result + [str(directory/'source/collect_piper_six.sbatch'), str(root), str(directory),
        '--episodes', str(episodes), '--max-attempts', str(10 if episodes == 1 else 300),
        '--acceptance-trials', '10', '--split', 'train', '--seed', str(seed), '--boundary-preflight']

if dry:
    print(shlex.join(command(run/'smoke', 1)))
    print(shlex.join(command(run/'train', 50, '<SMOKE_JOB_ID>')))
    sys.exit(0)
if (dataset_root/'piper_push_t').exists():
    raise SystemExit('Dataset already exists: '+str(dataset_root/'piper_push_t'))
run.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root/'benchmark/src'))
from piper_bench.tasks.push_t.protocol import write_manifests
write_manifests(run/'manifests', seed)
source = run/'smoke/source'
sys.path.insert(0, str(root/'benchmark/src'))
from piper_bench.collection.freeze import freeze_collection_source
freeze_collection_source(root, source)
for name in ('collect_piper_six.sbatch', 'collect_piper_push_t_pose_v2.sh'):
    shutil.copy2(root/'scripts'/name, source/name)
shutil.copy2(root/'docs/piper_eight_task_collection_eval_protocol_v1.md', source/'protocol.md')
shutil.copytree(source, run/'train/source')
hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(source.rglob('*')) if p.is_file()}
for name, count in [('smoke', 1), ('train', 50)]:
    directory = run/name; (directory/'logs').mkdir()
    assert all(hashlib.sha256((directory/'source'/p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    (directory/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (directory/'launch.json').write_text(json.dumps(dict(task='push_t', episodes=count,
        acceptance_trials=10, boundary_preflight=35, seed=seed, split='train', group_index=0,
        dataset_root=str(dataset_root) if count != 1 else None), indent=2)+'\n')
parent = None
for name, count in [('smoke', 1), ('train', 50)]:
    directory = run/name
    result = subprocess.run(command(directory, count, parent), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{name} submission failed; previous job is recorded at {run}: {result.stderr}')
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('unexpected sbatch reply: '+result.stdout)
    (directory/'submitted_jobs.txt').write_text(job+'\n')
    (directory/'submitted_groups.tsv').write_text('0\t'+job+'\n')
    if parent:
        (directory/'submitted_dependencies.tsv').write_text('0\t'+job+'\t'+parent+'\n')
    with (run/'submitted_jobs.jsonl').open('a') as f:
        f.write(json.dumps(dict(phase=name, job_id=job, afterok=parent))+'\n')
    print(f'{name}: job {job}, afterok={parent}, output={directory}', flush=True)
    parent = job
print('Shared evaluation manifest:', run/'manifests/evaluation_50.json')
PY
