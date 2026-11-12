"""Run one owned probe with local caches and observed host/storage budgets."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def cache_environment(root):
    root = Path(root).absolute()
    locations = {
        'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
        'XDG_CACHE_HOME': 'xdg', 'XDG_CONFIG_HOME': 'config', 'XDG_STATE_HOME': 'state',
        'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'HF_ASSETS_CACHE': 'huggingface/assets',
        'HF_DATASETS_CACHE': 'huggingface/datasets', 'TRANSFORMERS_CACHE': 'huggingface/transformers',
        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch/extensions',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
        'CUDA_CACHE_PATH': 'cuda', 'NUMBA_CACHE_DIR': 'numba',
        'MODELSCOPE_CACHE': 'modelscope', 'MPLCONFIGDIR': 'matplotlib', 'WANDB_DIR': 'wandb',
    }
    values = {key: str(root / suffix) for key, suffix in locations.items()}
    for directory in set(values.values()):
        Path(directory).mkdir(parents=True, exist_ok=True)
    values.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', DIFFSYNTH_SKIP_DOWNLOAD='true',
                  WANDB_MODE='disabled', CUDA_CACHE_MAXSIZE=str(64 * 1024**2))
    return values


def storage_usage(roots):
    seen, logical, allocated, count = set(), 0, 0, 0
    for root in roots:
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory) / name
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                logical += stat.st_size
                allocated += stat.st_blocks * 512
                count += 1
    return dict(logical_bytes=logical, allocated_bytes=allocated, files=count)


def process_tree(root_pid):
    processes = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            pid = int(path.parent.name)
            processes[pid] = dict(pid=pid, ppid=int(fields['PPid']),
                                  rss_kib=int(fields.get('VmRSS', '0 kB').split()[0]))
        except (OSError, KeyError, ValueError):
            continue
    owned = {root_pid}
    while True:
        children = {pid for pid, value in processes.items() if value['ppid'] in owned}
        if children <= owned:
            break
        owned.update(children)
    return [processes[pid] for pid in sorted(owned) if pid in processes]


def cgroup_memory():
    result = []
    for entry in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, suffix = entry.split(':', 2)
        if controllers == '':
            roots = [Path('/sys/fs/cgroup') / suffix.lstrip('/'), Path('/sys/fs/cgroup')]
            names = ('memory.current', 'memory.peak', 'memory.max', 'memory.events')
        elif 'memory' in controllers.split(','):
            roots = [Path('/sys/fs/cgroup/memory') / suffix.lstrip('/')]
            names = ('memory.usage_in_bytes', 'memory.max_usage_in_bytes',
                     'memory.limit_in_bytes', 'memory.failcnt', 'memory.oom_control')
        else:
            continue
        for root in roots:
            values = {}
            for name in names:
                try:
                    values[name] = (root / name).read_text().strip()
                except OSError:
                    pass
            if values:
                result.append(dict(path=str(root), values=values))
                break
    return result


def stop_owned(process, hard=False):
    # Services start their own sessions; signalling only the client's process
    # group would leave such children alive if the client itself was killed.
    records = process_tree(process.pid)
    children = [record['pid'] for record in records if record['pid'] != process.pid]
    signum = signal.SIGKILL if hard else signal.SIGTERM
    for pid in children + [process.pid]:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for pid in children + [process.pid]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
    # A parent can exit before a child that ignores TERM. These PIDs were
    # captured from our own live tree, including service sessions.
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--extra-cache-root', type=Path, action='append', default=[])
    parser.add_argument('--storage-root', type=Path, action='append', required=True)
    parser.add_argument('--stop-bytes', type=int, default=512 * 1024**2)
    parser.add_argument('--file-limit-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--cache-stop-bytes', type=int)
    parser.add_argument('--timeout', type=float, default=1500)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or min(args.stop_bytes, args.file_limit_bytes, args.timeout, args.interval) <= 0
            or (args.cache_stop_bytes is not None and args.cache_stop_bytes <= 0)):
        parser.error('command and positive resource limits are required')
    args.output.mkdir(parents=True, exist_ok=False)
    caches = cache_environment(args.cache_root)
    (args.output / 'environment.json').write_text(json.dumps(caches, indent=2) + '\n')
    environment = dict(os.environ, **caches)
    initial_cgroup = cgroup_memory()
    def limit_files():
        resource.setrlimit(resource.RLIMIT_FSIZE, (args.file_limit_bytes, args.file_limit_bytes))
    stopped = False
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    previous = signal.signal(signal.SIGTERM, interrupt)
    started, reason, peak_rss = time.monotonic(), None, 0
    process = None
    try:
        with (args.output / 'command.log').open('x') as log, (args.output / 'resources.jsonl').open('x') as samples:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=limit_files, start_new_session=True)
            while True:
                tree = process_tree(process.pid)
                rss = sum(record['rss_kib'] for record in tree)
                peak_rss = max(peak_rss, rss)
                storage = storage_usage(args.storage_root)
                cache_storage = storage_usage([args.cache_root, *args.extra_cache_root])
                samples.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
                    processes=tree, summed_rss_kib=rss, cgroup=cgroup_memory(), storage=storage,
                    cache_storage=cache_storage)) + '\n')
                samples.flush()
                if max(storage['logical_bytes'], storage['allocated_bytes']) > args.stop_bytes:
                    reason = 'storage_stop_threshold'
                    stop_owned(process, hard=True)
                    break
                if args.cache_stop_bytes is not None and max(cache_storage['logical_bytes'], cache_storage['allocated_bytes']) > args.cache_stop_bytes:
                    reason = 'cache_stop_threshold'
                    stop_owned(process, hard=True)
                    break
                if stopped or time.monotonic() - started > args.timeout:
                    reason = 'signal' if stopped else 'timeout'
                    stop_owned(process)
                    break
                if process.poll() is not None:
                    break
                time.sleep(args.interval)
    finally:
        if process is not None and process.poll() is None:
            stop_owned(process)
        signal.signal(signal.SIGTERM, previous)
        result = dict(command=command, exit_code=None if process is None else process.returncode,
            stop_reason=reason, sampled_tree_rss_peak_kib=peak_rss,
            cgroup_before=initial_cgroup, cgroup_after=cgroup_memory(),
            final_storage=storage_usage(args.storage_root),
            storage_stop_bytes=args.stop_bytes, per_file_limit_bytes=args.file_limit_bytes,
            cache_stop_bytes=args.cache_stop_bytes,
            storage_limit_scope='sampling guard with headroom, not a filesystem quota',
            rss_scope='sampled sum across owned processes; shared pages may be counted more than once',
            elapsed_seconds=time.monotonic()-started)
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if process is not None and process.returncode == 0 and reason is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
