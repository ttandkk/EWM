#!/usr/bin/env bash
#SBATCH --job-name=ewm-simple-ada-serial
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/ewm_simple_ada_serial_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/ewm_simple_ada_serial_%j.log
# Usage: sbatch scripts/train_ewm_simple_ada_serial.sbatch BATCH
# Re-submit with the same batch after a timeout to resume complete full states.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo 'usage: sbatch train_ewm_simple_ada_serial.sbatch <frozen-batch>' >&2
    exit 2
fi
batch=$(realpath "$1")
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EWM_EIGHT_CACHE="$root/tmp/training_data_cache/piper_dynamic_v1"
export EWM_CURRENT_DINO_CACHE="$EWM_EIGHT_CACHE/current_dino_fixed32"
export EWM_DINO_PATH="$root/pretrained/dinov3-vitl16" EWM_T5_PATH="$root/pretrained/t5-small"
export EWM_VAE_PATH="$root/pretrained/leanvae"
export EWM_EIGHT_STATS="$EWM_EIGHT_CACHE/shared_normalization.json" EWM_ACTION_STATS="$EWM_EIGHT_CACHE/shared_normalization.json"
export PYTHONPATH="$batch/source/OpenWAM"
export TMPDIR="/tmp/ewm-ada-serial-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$batch/runs"
exec 8>"$batch/runs/ada_serial.lock"
flock -n 8 || { echo 'Another serial job is using this batch' >&2; exit 1; }
cmp -s "$0" "$batch/source/scripts/train_ewm_simple_ada_serial.sbatch" || {
    echo 'Submitted launcher differs from the frozen batch launcher' >&2; exit 1;
}

"$OWPY" -B - "$batch/manifest.json" <<'CHECK_CACHE'
import json, sys
if not json.load(open(sys.argv[1])).get('cache_content_sha256'):
    raise RuntimeError('formal run requires full cache-content SHA256')
CHECK_CACHE
"$OWPY" -B "$batch/source/scripts/prepare_ewm_simple_formal.py" verify --batch "$batch"
"$OWPY" -B - <<'CHECK_GPU'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 2 or not all('6000 ada' in name.lower() for name in names):
    raise RuntimeError(f'formal training requires two RTX 6000 Ada GPUs, found {names}')
print('FORMAL_GPUS_OK', names, flush=True)
CHECK_GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Prevent mixing another hardware/compile protocol or a different frozen snapshot.
"$OWPY" -B - "$batch" <<'CHECK_SPEC'
import hashlib, json, sys
from pathlib import Path
batch = Path(sys.argv[1])
spec = {'format': 'ewm_ada_serial_v1', 'source_manifest_sha256': hashlib.sha256(
    (batch / 'manifest.json').read_bytes()).hexdigest(), 'gpu': '6000ada',
    'world_size': 2, 'compile_core': True, 'max_steps': 80000}
path = batch / 'runs' / 'ada_serial_spec.json'
if path.exists():
    if json.loads(path.read_text()) != spec:
        raise RuntimeError('serial run spec does not match the existing batch')
else:
    path.write_text(json.dumps(spec, sort_keys=True, indent=2) + '\n')
CHECK_SPEC

cd "$batch/source/OpenWAM"
for entry in action_only:train_ewm_eight_action_only joint:train_ewm_eight rgb_action:train_ewm_eight_rgb_action; do
    mode=${entry%%:*}
    config=${entry#*:}
    out="$batch/runs/$mode"
    exec 9>"$batch/runs/$mode.lock"
    flock -n 9 || { echo "Concurrent run for $mode" >&2; exit 1; }
    run=fresh
    if [[ -d "$out" ]]; then
        [[ -f "$out/.ada_serial_owned" ]] || { echo "Unrecognized existing output: $out" >&2; exit 1; }
        # A partial save without optimizer/RNG state cannot be resumed safely.
        step=$("$OWPY" -B - "$out" <<'FIND_STATE'
from pathlib import Path
import sys
from openwam.train.utils.ewm_resume import find_latest_resume_state
try:
    _, metadata = find_latest_resume_state(Path(sys.argv[1]), 2)
except FileNotFoundError:
    print(0)
else:
    print(int(metadata['step']))
FIND_STATE
)
        if [[ "$step" == 80000 ]]; then
            "$OWPY" -B - "$batch" "$out" "$mode" <<'CHECK_COMPLETE'
import hashlib, json, sys
from pathlib import Path
batch, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
marker = output / 'formal_complete.json'
if not marker.is_file():
    raise RuntimeError(f'final state has no validated completion marker: {output}')
saved = json.loads(marker.read_text())
spec_sha = hashlib.sha256((batch / 'runs' / 'ada_serial_spec.json').read_bytes()).hexdigest()
if saved.get('mode') != mode or saved.get('step') != 80000 or saved.get('spec_sha256') != spec_sha:
    raise RuntimeError(f'invalid completion marker: {marker}')
CHECK_COMPLETE
            echo "FORMAL_SKIP_COMPLETE mode=$mode step=$step"
            flock -u 9
            continue
        elif [[ "$step" == 0 ]]; then
            echo "No complete optimizer/RNG checkpoint for $mode; preserved $out for inspection" >&2
            exit 1
        else
            (( step > 0 && step < 80000 )) || { echo "Invalid resume step: $step" >&2; exit 1; }
            run=resume
        fi
    elif [[ -e "$out" ]]; then
        echo "Output is not a directory: $out" >&2; exit 1
    fi
    if [[ "$run" == fresh ]]; then mkdir "$out"; touch "$out/.ada_serial_owned"; fi
    options=("hydra.run.dir=$out/hydra_${SLURM_JOB_ID}" "training.output_path=$out"
        "dataloader.current_dino_cache_root=$EWM_CURRENT_DINO_CACHE"
        training.max_steps=80000 training.lr_schedule_steps=80000 training.batch_size=32
        training.gradient_accumulation_steps=1 training.mixed_precision=bf16 training.compile_core=true
        training.save_checkpoints=true training.save_steps=20000 training.save_full_states_for_resume=true
        training.keep_last_k_resume_states=2 training.keep_last_k_policy_checkpoints=4
        training.init_policy_checkpoint=null)
    if [[ "$mode" == joint ]]; then options+=(dataloader.compact_events=true); fi
    if [[ "$run" == resume ]]; then options+=("training.resume_ckpt_path=$out"); fi
    log="$out/train_${SLURM_JOB_ID}.log"
    printf 'FORMAL_MODE=%s RUN=%s OUTPUT=%s JOB=%s COMPILE=true\n' "$mode" "$run" "$out" "$SLURM_JOB_ID"
    "$OWPY" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/train.py --config-name "$config" "${options[@]}" 2>&1 | tee "$log"
    if ! rg -q 'Enabled training core compile:' "$log"; then
        echo "Compiled training was not enabled: $log" >&2; exit 1
    fi
    if rg -q 'terminate called without an active exception|DataLoader worker.*killed by signal|Fatal Python error: Aborted' "$log"; then
        echo "Worker termination detected: $log" >&2; exit 1
    fi
    "$OWPY" -B - "$batch" "$out" "$mode" "$SLURM_JOB_ID" <<'CHECK_DONE'
import hashlib, json
from pathlib import Path
import sys
from openwam.train.utils.ewm_resume import find_latest_resume_state
batch, output, mode, job_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
_, metadata = find_latest_resume_state(output, 2)
if int(metadata['step']) != 80000:
    raise RuntimeError(f"training returned without complete step 80000: {metadata['step']}")
spec_sha = hashlib.sha256((batch / 'runs' / 'ada_serial_spec.json').read_bytes()).hexdigest()
marker = output / 'formal_complete.json'
temporary = output / f'formal_complete.{job_id}.tmp'
temporary.write_text(json.dumps({'mode': mode, 'step': 80000, 'job_id': job_id,
                                 'spec_sha256': spec_sha}, indent=2) + '\n')
temporary.replace(marker)
print('FORMAL_COMPLETE step=80000', flush=True)
CHECK_DONE
    flock -u 9
done
