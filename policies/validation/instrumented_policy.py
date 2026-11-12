"""Measure a native policy process; do not change model loading or inference."""
import json
import logging
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    from piper_policies.cli import main as policy_main
    serving = len(sys.argv) > 1 and sys.argv[1] == 'serve'
    started = time.monotonic()
    if serving:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
        logging.info('Native policy startup pid=%s requested_device=cuda:0', os.getpid())
        def terminate(signum, frame):
            raise InterruptedError(f'Policy terminated by signal {signum}')
        previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        return policy_main()
    finally:
        if serving:
            signal.signal(signal.SIGTERM, previous_handler)
        if serving and os.environ.get('PROBE_RESOURCE_METRICS'):
            metrics = dict(process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           process_seconds=time.monotonic()-started, pid=os.getpid(),
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           requested_device='cuda:0', cuda_initialized=False)
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                metrics['cuda_initialized'] = True
                try:
                    metrics.update(cuda0_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                   cuda0_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                                   cuda0_name=torch.cuda.get_device_name(0))
                except Exception as exc:
                    metrics['cuda_metrics_error'] = str(exc)
            with Path(os.environ['PROBE_RESOURCE_METRICS']).open('x') as stream:
                json.dump(metrics, stream, indent=2)
                stream.write('\n')


if __name__ == '__main__':
    main()
