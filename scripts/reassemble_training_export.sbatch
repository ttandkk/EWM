#!/usr/bin/env bash
#SBATCH --job-name=reasm10-export
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/reassemble_export_single_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/reassemble_export_single_%j.log
set -euo pipefail

root=/projects/hdd/myproject/CVPR2027
python_bin="$root/envs/openwam/bin/python"

for index in $(seq 0 148); do
    echo "processing episode $index" >&2
    manifest="$root/datasets/reassemble_training_eventframes_10ms.partial/event_frames/$(printf 'episode_%06d' "$index")/manifest.json"
    timing="$root/tmp/reassemble_export_timing/$(printf 'episode_%06d' "$index")/timing.json"
    pilot_timing="$root/tmp/reassemble_alignment/$(printf 'episode_%06d' "$index")/timing.json"

    if [[ -f "$manifest" ]]; then
        "$python_bin" "$root/scripts/export_reassemble_training.py" episode --index "$index" --timing-json "$timing"
        continue
    fi

    if [[ ! -f "$timing" ]]; then
        if [[ -f "$pilot_timing" ]]; then
            timing="$pilot_timing"
        else
            for attempt in 1 2 3; do
                if "$python_bin" "$root/scripts/extract_reassemble_timestamps.py" \
                    --episode "$index" \
                    --output-dir "$root/tmp/reassemble_export_timing" \
                    --max-compressed-mib 1024; then
                    break
                fi
                if [[ "$attempt" -eq 3 ]]; then
                    echo "timestamp extraction failed after 3 attempts for episode $index" >&2
                    exit 1
                fi
                sleep $((attempt * 15))
            done
        fi
    fi

    "$python_bin" "$root/scripts/export_reassemble_training.py" episode --index "$index" --timing-json "$timing"
done

"$python_bin" "$root/scripts/export_reassemble_training.py" finalize
