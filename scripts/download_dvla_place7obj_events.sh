#!/usr/bin/env bash
#SBATCH --job-name=dvla-place7obj-download
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/dvla_place7obj_download_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/dvla_place7obj_download_%j.log

set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
REPO=mickeykang/dvla-place7obj-roll-250hz-events-250fps-delta-trim
REVISION=8ded7c2eded1e3b68a3ea2de99999da55e468f14
DEST="$ROOT/datasets/dvla-place7obj-roll-250hz-events-250fps-delta-trim"
PYTHON="$ROOT/envs/lerobot/bin/python"

mkdir -p "$DEST" "$ROOT/tmp/dvla_place7obj_hf_cache"
export HF_HOME="$ROOT/tmp/dvla_place7obj_hf_cache"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
# Anonymous Xet read tokens hit HTTP 429; regular HTTP was tested on parquet and MP4.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export PYTHONUNBUFFERED=1

echo "Repository: $REPO"
echo "Revision: $REVISION"
echo "Destination: $DEST"
echo "Started: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"

for attempt in $(seq 1 12); do
    echo "Download attempt $attempt/12 at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    if REPO="$REPO" REVISION="$REVISION" DEST="$DEST" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo = os.environ['REPO']
revision = os.environ['REVISION']
dest = Path(os.environ['DEST'])
info = HfApi().repo_info(repo, repo_type='dataset', revision=revision, files_metadata=True)
if info.sha != revision:
    raise RuntimeError(f'Unexpected revision: {info.sha} != {revision}')
files = [entry for entry in info.siblings if entry.rfilename]
expected_bytes = sum(entry.size or 0 for entry in files)
print(f'Expected: {len(files)} files, {expected_bytes} bytes', flush=True)

snapshot_download(
    repo_id=repo,
    repo_type='dataset',
    revision=revision,
    local_dir=dest,
    max_workers=16,
)

bad = []
for index, entry in enumerate(files, 1):
    path = dest / entry.rfilename
    if entry.size is None or not path.is_file() or path.stat().st_size != entry.size:
        bad.append(entry.rfilename)
    if index % 5000 == 0:
        print(f'Checked {index}/{len(files)} files; mismatches so far: {len(bad)}', flush=True)
if bad:
    for name in bad:
        path = dest / name
        if path.is_file():
            path.unlink()
    raise RuntimeError(f'Missing or wrong-size files ({len(bad)}): {bad[:20]}')

(dest / '.download_complete').write_text(
    f'repo={repo}\nrevision={revision}\nfiles={len(files)}\nbytes={expected_bytes}\n'
)
print(f'Complete: {len(files)} files, {expected_bytes} bytes', flush=True)
PY
    then
        echo "Finished: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
        exit 0
    fi
    if (( attempt < 12 )); then
        delay=$(( attempt * 300 ))
        if (( delay > 1800 )); then delay=1800; fi
        echo "Attempt failed; retrying with existing files preserved in $delay seconds"
        sleep "$delay"
    fi
done

echo 'Download failed after 12 attempts' >&2
exit 1
