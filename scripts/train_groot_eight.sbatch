#!/usr/bin/env bash
#SBATCH --job-name=groot-eight-rgb
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/groot_eight_train/%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/groot_eight_train/%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/activate_training.sh" groot
# The system Python venv has no development headers on compute nodes. Keep
# matching Ubuntu headers in this environment; export for all compiler workers.
python_dev_root="$root/envs/groot/python-dev"
export CPATH="$python_dev_root/usr/include/python3.12:$python_dev_root/usr/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export WANDB_MODE=online WANDB_INIT_TIMEOUT=120 WANDB_LOG_MODEL=false
export GROOT_EIGHT_OUTPUT="$root/Isaac-GR00T/outputs/groot_eight_rgb_${SLURM_JOB_ID:-preflight}"
export WANDB_RUN_ID="groot8-${SLURM_JOB_ID:-preflight}"
export WANDB_DIR="$GROOT_EIGHT_OUTPUT"
# W&B stages full checkpoints: keep that storage on disk, not in compiler tmpfs.
export WANDB_CACHE_DIR="$GROOT_EIGHT_OUTPUT/wandb_cache"
export WANDB_DATA_DIR="$GROOT_EIGHT_OUTPUT/wandb_data"
export GROOT_COMMIT_HASH="$(git -C "$root/Isaac-GR00T" rev-parse HEAD)"
data_root="$root/tmp/training_data_cache/piper_dynamic_rgb_v1"
dataset_paths="$(python - "$data_root" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
tasks = json.loads((root / 'manifest.json').read_text())['tasks']
assert len(tasks) == 8 and all(t['episodes'] == 50 for t in tasks)
assert [t['task'] for t in tasks] == ['piper_'+t for t in ('conveyor','moving_carton','turntable_cups','rotating_tube','tennis_rolling','tennis_falling','tennis_parabolic','tennis_container')]
ewm = json.loads((root.parent / 'piper_dynamic_v1/complete.json').read_text())
assert ewm['passed'] and sum(t['frames'] for t in tasks) == ewm['frames']
print(os.pathsep.join(str(root / t['task']) for t in tasks))
PY
)"
command=(python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1
    gr00t/experiment/launch_finetune.py
    --base-model-path "$GROOT_BASE_MODEL_PATH" --dataset-path "$dataset_paths"
    --backbone-model-path "$GROOT_BACKBONE_PATH"
    --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/Piper/piper_config.py
    --num-gpus 1 --global-batch-size 64 --gradient-accumulation-steps 1
    --dataloader-num-workers 4 --max-steps 80000 --save-steps 20000 --save-total-limit 4
    --learning-rate 0.0001 --warmup-ratio 0.05 --weight-decay 0.01
    --adam-beta1 0.9 --adam-beta2 0.95 --lr-min-ratio 0.01 --no-torch-compile
    --use-wandb --wandb-project CVPR2027-EWM
    --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model
    --shortest-image-edge 256 --crop-fraction 1.0 --no-use-percentiles
    --state-dropout-prob 0.0 --color-jitter-params brightness 0.0 contrast 0.0 saturation 0.0 hue 0.0
    --ds-weights-alpha 0 --episode-sampling-rate 1.0 --shard-size 1024
    --no-save-only-model --no-resume-from-checkpoint
    --output-dir "$GROOT_EIGHT_OUTPUT")
if [[ "${1:-}" == --print-command ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
: "${SLURM_JOB_ID:?Run this script with sbatch}"
test -x "${CUDA_HOME:?CUDA toolkit must be available}/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
# multiprocessing adds /pymp-XXXXXXXX/listener-XXXXXXXX; a deep output path
# exceeds AF_UNIX's 107-byte pathname limit and silently stalls batch delivery.
# Keep general temporary files on disk; W&B checkpoint staging has its own path.
TMPDIR="$(mktemp -d "/tmp/groot8_${SLURM_JOB_ID}_XXXXXX")"
export TMPDIR
runtime_dir=""
cleanup() {
    local code=$?
    trap - EXIT
    if [[ "$runtime_dir" == /dev/shm/groot8_compile_"${SLURM_JOB_ID}"_* && -O "$runtime_dir" ]]; then
        rm -rf -- "$runtime_dir"
    fi
    if [[ "$TMPDIR" == /tmp/groot8_"${SLURM_JOB_ID}"_* && -O "$TMPDIR" ]]; then
        rm -rf -- "$TMPDIR"
    fi
    exit "$code"
}
trap cleanup EXIT
mkdir -p "$GROOT_EIGHT_OUTPUT/source" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
python - <<'PY_IPC' | tee "$GROOT_EIGHT_OUTPUT/ipc_preflight.log"
from multiprocessing.connection import Listener
import os
with Listener(family='AF_UNIX') as listener:
    print('IPC_SOCKET_OK', len(os.fsencode(listener.address)), flush=True)
PY_IPC
runtime_dir="$(mktemp -d "/dev/shm/groot8_compile_${SLURM_JOB_ID}_XXXXXX")"
export TORCHINDUCTOR_CACHE_DIR="$runtime_dir/inductor"
export TRITON_CACHE_DIR="$runtime_dir/triton" CUDA_CACHE_PATH="$runtime_dir/cuda"
export TORCH_EXTENSIONS_DIR="$runtime_dir/extensions"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Eager run: validate the real model below without a compiler probe.
python - <<'PY'
import hashlib, json, os, pathlib, shutil, torch, wandb
assert torch.cuda.device_count() == 1
assert all('pro 6000' in torch.cuda.get_device_name(i).lower() for i in range(1))
root = pathlib.Path(os.environ['TRAINING_PROJECT_ROOT'])
out = pathlib.Path(os.environ['GROOT_EIGHT_OUTPUT'])
api = wandb.Api(timeout=30)
assert api.viewer, 'W&B online authentication failed'
url = f"https://wandb.ai/{api.default_entity}/CVPR2027-EWM/runs/{os.environ['WANDB_RUN_ID']}"
(out / 'wandb_run.json').write_text(json.dumps(dict(entity=api.default_entity,
    project='CVPR2027-EWM', id=os.environ['WANDB_RUN_ID'], url=url, mode='online'), indent=2))
paths = ['scripts/train_groot_eight.sbatch', 'scripts/verify_piper_rgb_baselines.py',
    'scripts/check_groot_compile.py', 'deployment/activate_training.sh',
    'scripts/smoke_groot_eight.py', 'scripts/prepare_piper_rgb_baselines.py',
    'Isaac-GR00T/examples/Piper/piper_config.py',
    'Isaac-GR00T/gr00t/configs/finetune_config.py',
    'Isaac-GR00T/gr00t/configs/training/training_config.py',
    'Isaac-GR00T/gr00t/experiment/launch_finetune.py',
    'Isaac-GR00T/gr00t/model/gr00t_n1d7/setup.py',
    'Isaac-GR00T/gr00t/experiment/experiment.py']
for name in paths:
    dst = out / 'source' / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, dst)
(out / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}, indent=2))
shutil.copyfile(root / 'tmp/training_data_cache/piper_dynamic_rgb_v1/manifest.json', out / 'data_manifest.json')
shutil.copyfile(root / 'envs/groot/python-dev/manifest.json', out / 'python_dev_manifest.json')
print('WANDB_AUTH_PASSED', url, flush=True)
PY
python "$root/scripts/verify_piper_rgb_baselines.py" export --data-root "$data_root" --report-dir "$GROOT_EIGHT_OUTPUT/preflight"
python "$root/scripts/verify_piper_rgb_baselines.py" groot --data-root "$data_root" --report-dir "$GROOT_EIGHT_OUTPUT/preflight"
printf '%q ' "${command[@]}" > "$GROOT_EIGHT_OUTPUT/command.sh"
printf '\n' >> "$GROOT_EIGHT_OUTPUT/command.sh"
cd "$root/Isaac-GR00T"
# Use the same real weights/data/batch/compile settings. Failure stops this job.
smoke_command=("${command[@]}")
smoke_command[6]="$root/scripts/smoke_groot_eight.py"
timeout --signal=TERM --kill-after=60s 45m "${smoke_command[@]}"
test -s "$GROOT_EIGHT_OUTPUT/gpu_preflight/rank_0_passed.json"
printf '%s\n' 'SINGLE_GPU_BATCH64_EAGER_PREFLIGHT_PASSED: starting formal 80000-step run'
"${command[@]}"
