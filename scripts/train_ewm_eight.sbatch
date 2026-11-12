#!/usr/bin/env bash
#SBATCH --job-name=ewm-eight
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/ewm_eight_train/%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/ewm_eight_train/%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export CUDA_HOME=/cluster/apps/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 WANDB_MODE=online WANDB_INIT_TIMEOUT=120
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export EWM_EIGHT_CONFIG="${EWM_EIGHT_CONFIG:-train_ewm_eight}"
default_event_stride=2
export EWM_EVENT_TEMPORAL_DOWNSAMPLE="${EWM_EVENT_TEMPORAL_DOWNSAMPLE:-$default_event_stride}"
case "$EWM_EVENT_TEMPORAL_DOWNSAMPLE" in
    1|2|3) ;;
    *) echo "EWM_EVENT_TEMPORAL_DOWNSAMPLE must be 1, 2 or 3" >&2; exit 1 ;;
esac
export EWM_EIGHT_OUTPUT="${EWM_EIGHT_OUTPUT:-$root/OpenWAM/outputs/ewm_eight_${SLURM_JOB_ID}}"
export EWM_EIGHT_STATS="$EWM_EIGHT_OUTPUT/shared_normalization.json"
export EWM_EIGHT_CACHE="${EWM_EIGHT_CACHE:-$root/tmp/training_data_cache/piper_dynamic_v1}"
if [[ ! -s "$EWM_EIGHT_CACHE/complete.json" ]]; then
    echo "Current data caches are not ready. Run scripts/prepare_ewm_dynamic_data.sbatch first." >&2
    exit 1
fi
"$OWPY" - <<'PY_CHECK'
import json, os
from pathlib import Path
stats = json.loads((Path(os.environ['EWM_EIGHT_CACHE']) / 'shared_normalization.json').read_text())
assert stats.get('action', {}).get('type') == 'minmax', 'Regenerate full-range action statistics before new training'
PY_CHECK
export EWM_CURRENT_DINO_CACHE="$EWM_EIGHT_CACHE/current_dino_fixed32"
export EWM_EIGHT_RUN_NAME="${EWM_EIGHT_RUN_NAME:-ewm-eight-${SLURM_JOB_ID}}"
export WANDB_RUN_ID="ewm8-${SLURM_JOB_ID}"
export WANDB_DIR="$EWM_EIGHT_OUTPUT"
mkdir -p "$EWM_EIGHT_OUTPUT/source"
# These are small compiler/runtime files, not a copy of the training data.
runtime_dir="$(mktemp -d "/dev/shm/ewm8_runtime_${SLURM_JOB_ID}_XXXXXX")"
cleanup() {
    local code=$?
    trap - EXIT
    if [[ "$runtime_dir" == /dev/shm/ewm8_runtime_"${SLURM_JOB_ID}"_* && -O "$runtime_dir" ]]; then
        rm -rf -- "$runtime_dir"
    fi
    exit "$code"
}
trap cleanup EXIT
export TMPDIR="$runtime_dir/tmp"
export TORCHINDUCTOR_CACHE_DIR="$runtime_dir/inductor"
export TRITON_CACHE_DIR="$runtime_dir/triton" CUDA_CACHE_PATH="$runtime_dir/cuda"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$OWPY" - <<'PY'
import json, os, pathlib, torch, wandb
assert torch.cuda.device_count() == 2
expected_gpu = os.environ.get('EWM_EIGHT_EXPECTED_GPU', '6000 ada').lower()
assert all(expected_gpu in torch.cuda.get_device_name(i).lower() for i in range(2))
api = wandb.Api(timeout=30)
assert api.viewer, 'W&B online authentication failed'
entity = api.default_entity
url = f"https://wandb.ai/{entity}/CVPR2027-EWM/runs/{os.environ['WANDB_RUN_ID']}"
pathlib.Path(os.environ['EWM_EIGHT_OUTPUT'], 'wandb_run.json').write_text(json.dumps(dict(
    entity=entity, project='CVPR2027-EWM', id=os.environ['WANDB_RUN_ID'], url=url, mode='online'), indent=2))
print('WANDB_AUTH_PASSED', url, flush=True)
PY
# Reuse the verified 400-episode cache; never regenerate it in this training job.
"$OWPY" - <<'PY_CACHE'
import json, os
from pathlib import Path
base = Path(os.environ['EWM_CURRENT_DINO_CACHE'])
tasks = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube',
         'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')
for task in tasks:
    path = base / f'piper_{task}' / 'dino_current16' / 'manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['format'] == 'ewm_dino_current_bf16_v1'
    assert manifest['encoding_batch_size'] == 32
    assert len(manifest['episodes']) == 50
    assert all((path.parent / row['file']).is_file() for row in manifest['episodes'].values())
print('EXISTING_DINO_CACHE_READY: 8 tasks, 400 episodes', flush=True)
PY_CACHE

# Verify the formal sampling/resume path before consuming a long training run.
(cd "$root/OpenWAM" && "$OWPY" -m pytest -q \
    tests/test_ewm_piper_multitask.py tests/test_ewm_resume.py tests/test_ewm_source_batch_sampler.py \
    tests/test_ewm_compact_inputs.py tests/test_ewm_asymmetric_horizon.py tests/test_ewm_model.py \
    --basetemp="$runtime_dir/pytest")
"$OWPY" "$root/OpenWAM/scripts/train_ewm/prepare_piper_eight.py" --root "$root" \
    --config-name "$EWM_EIGHT_CONFIG" \
    --event-temporal-downsample "$EWM_EVENT_TEMPORAL_DOWNSAMPLE" \
    --output "$EWM_EIGHT_OUTPUT" \
    --cache-root "$EWM_EIGHT_CACHE" --compact-events --current-dino-cache-root "$EWM_CURRENT_DINO_CACHE"
"$OWPY" - <<'PY'
import hashlib, json, os, pathlib, shutil
root = pathlib.Path(os.environ['NEW_PROJECT_ROOT'])
out = pathlib.Path(os.environ['EWM_EIGHT_OUTPUT'])
paths = ['scripts/train_ewm_eight.sbatch',
         'OpenWAM/configs/train_ewm.yaml', 'OpenWAM/configs/train_ewm_eight.yaml',
         'OpenWAM/configs/model/ewm.yaml', 'OpenWAM/configs/dataloader/ewm_dataset.yaml',
         'OpenWAM/scripts/train.py', 'OpenWAM/scripts/train_ewm/train_ewm.py',
         'OpenWAM/scripts/train_ewm/prepare_piper_eight.py',
         'OpenWAM/scripts/train_ewm/prepare_piper_current_dino.py']
paths += [str(p.relative_to(root)) for folder in ('OpenWAM/openwam/dataloader',
          'OpenWAM/openwam/model/architectures', 'OpenWAM/openwam/train/utils')
          for p in (root/folder).glob('*.py')]
if os.environ.get('EWM_EIGHT_LAUNCHER'):
    paths.append(os.environ['EWM_EIGHT_LAUNCHER'])
for name in dict.fromkeys(paths):
    dst = out / 'source' / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, dst)
(out / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}, indent=2))
PY
cd "$root/OpenWAM"
"$OWPY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train.py --config-name "$EWM_EIGHT_CONFIG" \
    "hydra.run.dir=$EWM_EIGHT_OUTPUT/hydra" \
    "model.event_temporal_downsample=$EWM_EVENT_TEMPORAL_DOWNSAMPLE" \
    dataloader.compact_events=true "dataloader.current_dino_cache_root=$EWM_CURRENT_DINO_CACHE" \
    training.compile_core=true
