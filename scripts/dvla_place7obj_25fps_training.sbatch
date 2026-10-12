#!/usr/bin/env bash
#SBATCH --job-name=dvla25-train-export
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/dvla25_training_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/dvla25_training_%j.log
set -euo pipefail

root=/projects/hdd/myproject/CVPR2027
source_dir="$root/datasets/dvla-place7obj-roll-250hz-events-250fps-delta-trim"
stage="$root/datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim.partial"
python_bin="$root/envs/lerobot/bin/python"
script="$root/scripts/export_dvla_25hz_training.py"

test -f "$source_dir/.download_complete"
"$python_bin" "$script" prepare
for chunk in $(seq 0 14); do
    shard="$stage/meta/stats_shards/$(printf 'chunk-%03d.jsonl' "$chunk")"
    if [[ -f "$shard" ]]; then
        echo "chunk $chunk already complete; final validation will recheck it" >&2
        continue
    fi
    echo "starting chunk $chunk of 14 at $(date -u +%FT%TZ)" >&2
    "$python_bin" "$script" convert --chunk "$chunk"
    echo "finished chunk $chunk at $(date -u +%FT%TZ)" >&2
done
"$python_bin" "$script" finalize
