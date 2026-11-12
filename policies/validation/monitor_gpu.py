"""Sample only allocated GPUs; allocator peaks are recorded by the model process."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def select_device_uuids(rows, environment):
    """Resolve Slurm global IDs against the allocation-visible NVML namespace."""
    indices = {str(row['index']): row['uuid'] for row in rows}
    uuids = set(indices.values())
    for key in ('SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES'):
        tokens = [token.strip() for token in environment.get(key, '').split(',') if token.strip()]
        if not tokens:
            continue
        selected = []
        for token in tokens:
            if token in indices:
                selected.append(indices[token])
            else:
                matches = [uuid for uuid in uuids if uuid.startswith(token)]
                if len(matches) != 1:
                    break
                selected.append(matches[0])
        else:
            if len(set(selected)) == len(tokens):
                return selected
    raise ValueError('Allocation GPU IDs cannot be mapped to visible NVML UUIDs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-file', type=Path, required=True)
    args = parser.parse_args()
    discovered = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=5)
    rows = []
    for line in discovered.stdout.splitlines():
        index, uuid = [value.strip() for value in line.split(',')]
        rows.append(dict(index=index, uuid=uuid))
    selected = select_device_uuids(rows, os.environ)
    devices = ','.join(selected)
    args.output.with_suffix('.devices.json').write_text(json.dumps(dict(
        visible_nvml_devices=rows, selected_uuids=selected,
        slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2)+'\n')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    peaks = {}
    with args.output.open('x') as stream:
        while not stopped:
            record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          stage=args.stage_file.read_text().strip() if args.stage_file.exists() else 'setup')
            try:
                result = subprocess.run(['nvidia-smi', '--id='+devices,
                    '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=5)
                rows = []
                for line in result.stdout.splitlines():
                    uuid, used, total = [value.strip() for value in line.split(',')]
                    used = int(used); total = int(total)
                    rows.append(dict(uuid=uuid, used_mib=used, total_mib=total))
                    peaks[uuid] = max(peaks.get(uuid, 0), used)
                record['gpus'] = rows
            except Exception as exc:
                record['sample_error'] = str(exc)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            time.sleep(1)
    args.output.with_suffix('.summary.json').write_text(json.dumps(dict(
        sampled_device_memory_peak_mib=peaks, interval_seconds=1,
        scope='sampled total device use; exact PyTorch allocator peaks are separate server_resources.json'), indent=2)+'\n')


if __name__ == '__main__':
    main()
