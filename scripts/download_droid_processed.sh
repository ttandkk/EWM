#!/usr/bin/env bash
#SBATCH --job-name=droid-processed-download
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/droid_processed_download_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/droid_processed_download_%j.log

set -euo pipefail

ROOT=/projects/hdd/myproject/CVPR2027
DEST="$ROOT/datasets/droid-processed"
REPO=AivexRoboticsGroup/droid-processed
# Pinned to the repository HEAD checked on 2026-09-29.
REVISION=7bad681fc20752682dcd1e0ef497efbb7745f1a8

mkdir -p "$DEST" "$ROOT/tmp/droid_processed_hf_cache" "$ROOT/logs"
export HF_HOME="$ROOT/tmp/droid_processed_hf_cache"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONUNBUFFERED=1

echo "Downloading $REPO at $REVISION to $DEST"
echo "Started: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"

for attempt in 1 2 3 4 5; do
    echo "Attempt $attempt/5"
    if REPO="$REPO" REVISION="$REVISION" DEST="$DEST" python3 - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo = os.environ["REPO"]
revision = os.environ["REVISION"]
dest = Path(os.environ["DEST"])
api = HfApi()
info = api.repo_info(repo, repo_type="dataset", revision=revision, files_metadata=True)
assert info.sha == revision, (info.sha, revision)
files = info.siblings
expected_bytes = sum(f.size or 0 for f in files)
print(f"Expected: {len(files)} files, {expected_bytes} bytes", flush=True)

snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                  local_dir=dest, max_workers=8)

bad = []
for index, file in enumerate(files, 1):
    path = dest / file.rfilename
    if not path.is_file() or path.stat().st_size != file.size:
        bad.append(file.rfilename)
        continue
    if file.lfs and file.lfs.sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != file.lfs.sha256:
            bad.append(file.rfilename)
    if index % 50 == 0:
        print(f"Verified {index}/{len(files)} files", flush=True)

if bad:
    for name in bad:
        path = dest / name
        if path.is_file():
            path.unlink()
    raise RuntimeError(f"Missing or corrupt files ({len(bad)}): {bad[:10]}")

(dest / ".download_complete").write_text(
    f"repo={repo}\nrevision={revision}\nfiles={len(files)}\nbytes={expected_bytes}\n"
)
print(f"Complete: verified {len(files)} files and {expected_bytes} bytes", flush=True)
PY
    then
        echo "Finished: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
        exit 0
    fi
    if (( attempt < 5 )); then
        echo "Download or verification failed; retrying in 60 seconds"
        sleep 60
    fi
done

echo "Download failed after 5 attempts" >&2
exit 1
