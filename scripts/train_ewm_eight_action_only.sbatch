#!/usr/bin/env bash
#SBATCH --job-name=ewm-eight-action-only
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/ewm_eight_action_only/%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/ewm_eight_action_only/%j.log
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
source "$root/deployment/runtime_env.sh"
export CUDA_HOME=/cluster/apps/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 WANDB_MODE=online WANDB_INIT_TIMEOUT=120
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export EWM_ACTION_OUTPUT="$root/OpenWAM/outputs/ewm_eight_action_only_${SLURM_JOB_ID}"
export EWM_ACTION_STATS="$EWM_ACTION_OUTPUT/shared_normalization.json"
export EWM_ACTION_RUN_NAME="ewm-eight-action-only-${SLURM_JOB_ID}"
export WANDB_RUN_ID="ewm8-action-${SLURM_JOB_ID}" WANDB_DIR="$EWM_ACTION_OUTPUT"
mkdir -p "$EWM_ACTION_OUTPUT/source"
runtime_dir="$(mktemp -d "/dev/shm/ewm8_action_${SLURM_JOB_ID}_XXXXXX")"
cleanup() {
    local code=$?
    trap - EXIT
    if [[ "$runtime_dir" == /dev/shm/ewm8_action_"${SLURM_JOB_ID}"_* && -O "$runtime_dir" ]]; then
        rm -rf -- "$runtime_dir"
    fi
    exit "$code"
}
trap cleanup EXIT
export TMPDIR="$runtime_dir/tmp" TORCHINDUCTOR_CACHE_DIR="$runtime_dir/inductor"
export TRITON_CACHE_DIR="$runtime_dir/triton" CUDA_CACHE_PATH="$runtime_dir/cuda"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$OWPY" - <<'PY'
import json, os, pathlib, torch, wandb
assert torch.cuda.device_count() == 2
expected_gpu = os.environ.get('EWM_ACTION_EXPECTED_GPU', '6000 ada').lower()
assert all(expected_gpu in torch.cuda.get_device_name(i).lower() for i in range(2))
api = wandb.Api(timeout=30)
assert api.viewer, 'W&B online authentication failed'
entity = api.default_entity
url = f"https://wandb.ai/{entity}/CVPR2027-EWM/runs/{os.environ['WANDB_RUN_ID']}"
pathlib.Path(os.environ['EWM_ACTION_OUTPUT'], 'wandb_run.json').write_text(json.dumps(dict(
    entity=entity, project='CVPR2027-EWM', id=os.environ['WANDB_RUN_ID'], url=url,
    mode='online', upload_checkpoints=False), indent=2))
print('WANDB_AUTH_PASSED', url, flush=True)
PY
"$OWPY" "$root/OpenWAM/scripts/train_ewm/prepare_piper_action_eight.py" \
    --root "$root" --output "$EWM_ACTION_OUTPUT"
"$OWPY" - <<'PY'
import hashlib, json, os, pathlib, shutil
root = pathlib.Path(os.environ['NEW_PROJECT_ROOT'])
out = pathlib.Path(os.environ['EWM_ACTION_OUTPUT'])
paths = ['scripts/train_ewm_eight_action_only.sbatch',
         'OpenWAM/configs/train_ewm_eight_action_only.yaml', 'OpenWAM/configs/train_ewm_action_only_piper.yaml',
         'OpenWAM/configs/train_ewm.yaml', 'OpenWAM/configs/model/ewm.yaml', 'OpenWAM/configs/dataloader/ewm_dataset.yaml',
         'OpenWAM/openwam/dataloader/ewm_action_only.py', 'OpenWAM/openwam/dataloader/ewm_piper_action_multitask.py',
         'OpenWAM/openwam/dataloader/current_dino_cache.py', 'OpenWAM/openwam/dataloader/ewm_dataset.py',
         'OpenWAM/openwam/dataloader/ewm_normalizer.py', 'OpenWAM/openwam/dataloader/ewm_piper_multitask.py',
         'OpenWAM/openwam/dataloader/registry.py',
         'OpenWAM/openwam/dataloader/piper_training_data.py', 'OpenWAM/openwam/model/architectures/ewm.py',
         'OpenWAM/openwam/model/architectures/temporal_supervision.py',
         'OpenWAM/openwam/model/architectures/ewm_checkpoint.py',
         'OpenWAM/openwam/model/architectures/temporal_sampling.py',
         'OpenWAM/openwam/train/utils/ewm_resume.py', 'OpenWAM/scripts/train_ewm/train_ewm.py',
         'OpenWAM/scripts/train_ewm/prepare_piper_action_eight.py']
for name in paths:
    dst = out / 'source' / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, dst)
(out / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}, indent=2))
PY
cd "$root/OpenWAM"
"$OWPY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train.py --config-name train_ewm_eight_action_only \
    "hydra.run.dir=$EWM_ACTION_OUTPUT/hydra"
