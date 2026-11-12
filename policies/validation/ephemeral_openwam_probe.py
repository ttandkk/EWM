"""Reviewed zero-training OpenWAM probe with one explicitly disposable checkpoint."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024**3
CHECKPOINT = Path('initialization/checkpoint_step_000000.safetensors')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify_reclamation(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    expected = Path('/projects/_hdd/isaachdd/piper-container-build-20261111-v2/data/piper-simulation-v2')
    if (receipt.get('version') != 'piper_probe_storage_release_v1'
            or receipt.get('rootfs_path') != str(expected)
            or not receipt.get('archived_before_removal') or not receipt.get('rootfs_removed')
            or not receipt.get('policy_installation_idle')
            or receipt.get('approved_temporary_budget_bytes', 0) < 26 * GIB
            or expected.exists() or expected.is_symlink()):
        raise ValueError('Rootfs reclamation and exclusive probe storage slot are not confirmed')
    return receipt


def inspect_checkpoint(path):
    """Record a full or interrupted safetensors write without loading tensors."""
    path = Path(path)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    fingerprint = lambda value: [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]
    if fingerprint(before) != fingerprint(after):
        raise ValueError('Checkpoint changed during hashing')
    result = dict(path=str(path), size_bytes=before.st_size, file_identity=fingerprint(before),
                  sha256=digest, complete_checkpoint=False)
    try:
        with path.open('rb') as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError('partial header length')
            length = int.from_bytes(raw_length, 'little')
            if not 0 < length <= 16 * 1024**2:
                raise ValueError('invalid header length')
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError('partial header')
        header = json.loads(raw)
        tensors = {key: value for key, value in header.items() if key != '__metadata__'}
        ranges = sorted(tuple(value['data_offsets']) for value in tensors.values())
        position = 0
        for start, end in ranges:
            if start != position or end < start:
                raise ValueError('invalid tensor data offsets')
            position = end
        result.update(header_sha256=hashlib.sha256(raw).hexdigest(), tensor_count=len(tensors),
                      tensor_manifest=tensors, expected_file_bytes=8+length+position,
                      complete_checkpoint=bool(tensors) and result['size_bytes'] == 8+length+position)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['header_error'] = str(exc)
    return result


def live_recorded_processes(output):
    seen = set()
    for path in Path(output).glob('supervisor_*/resources.jsonl'):
        for line in path.read_text().splitlines():
            if line:
                seen.update(record['pid'] for record in json.loads(line).get('processes', []))
    live = []
    for pid in sorted(seen):
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0]
            if state != 'Z':
                live.append(pid)
        except FileNotFoundError:
            pass
    return live


def cache_directories(output):
    output = Path(output)
    # The unchanged historical service helper directs compiler caches under
    # its plan/prediction output; include those explicit roots in the budget
    # and retirement alongside the common environment caches.
    return [output/'cache', *(output/part/name for part in ('plan', 'prediction')
                             for name in ('inductor_cache', 'triton_cache'))]


def owned_retirement_path(path, output):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(output):
        raise ValueError(f'Resolved retirement path is outside output: {path}')
    return resolved


@contextmanager
def cleanup_signals():
    """Let an interrupted run finish its bounded evidence/retirement phase."""
    previous = {signum: signal.signal(signum, signal.SIG_IGN)
                for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def archive_and_retire(output, stage_results):
    output = Path(output).resolve()
    owner = json.loads((output/'probe_owner.json').read_text())
    if owner != {'kind': 'untrained_openwam_ephemeral_probe', 'output': str(output)}:
        raise ValueError('Output ownership marker mismatch')
    if live_recorded_processes(output):
        raise RuntimeError('A recorded model process is still alive; refuse retirement')
    checkpoint = output / CHECKPOINT
    caches = cache_directories(output)
    if (output/'initialization').is_symlink() or checkpoint.is_symlink() or any(path.is_symlink() for path in caches):
        raise ValueError('Refuse retirement through a symlink')
    for path in [checkpoint, *caches]:
        owned_retirement_path(path, output)
    identity = inspect_checkpoint(checkpoint) if checkpoint.exists() else None
    small_files = {}
    for directory, dirs, files in os.walk(output, followlinks=False):
        dirs[:] = [name for name in dirs if Path(directory)/name not in caches
                   and not (Path(directory)/name).is_symlink()]
        for name in files:
            path = Path(directory)/name
            if path == checkpoint:
                continue
            if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
                raise ValueError(f'Unexpected large or linked evidence file: {path}')
            small_files[path.relative_to(output).as_posix()] = dict(sha256=sha256(path), size_bytes=path.stat().st_size)
    stages_passed = ([stage['stage'] for stage in stage_results] == ['export', 'prepare', 'predict']
                     and all(stage['exit_code'] == 0 for stage in stage_results))
    write_json(output/'checkpoint_audit.json', dict(identity=identity, stages=stage_results,
        strict_reload_and_predict_passed=stages_passed,
        scope='zero-training artifact, not a trained policy; complete header does not establish strict reload success'))
    small_files['checkpoint_audit.json'] = dict(sha256=sha256(output/'checkpoint_audit.json'),
                                               size_bytes=(output/'checkpoint_audit.json').stat().st_size)
    write_json(output/'evidence_manifest.json', dict(sha256=small_files))
    for name, metadata in small_files.items():
        if sha256(output/name) != metadata['sha256']:
            raise ValueError(f'Evidence changed before retirement: {name}')
    # Only this explicit generated file can be removed. The input checkpoint,
    # pretrained paths and original statistics are never cleanup targets.
    if checkpoint.exists():
        resolved_checkpoint = owned_retirement_path(checkpoint, output)
        current = resolved_checkpoint.stat()
        if [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns] != identity['file_identity']:
            raise ValueError('Checkpoint changed before retirement')
        resolved_checkpoint.unlink()
    for cache in caches:
        resolved_cache = owned_retirement_path(cache, output)
        if resolved_cache.is_dir():
            shutil.rmtree(resolved_cache)
    write_json(output/'checkpoint_lifecycle.json', dict(
        original_path=str(checkpoint), identity=identity, retained=False,
        retired_at_unix_s=time.time(), retired_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_is_executable=False,
        strict_reload_and_predict_passed=stages_passed,
        retirement_reason='successful_ephemeral_probe' if stages_passed else 'failed_untrained_ephemeral_probe',
        partial=identity is not None and not identity['complete_checkpoint'],
        evidence_manifest_sha256=sha256(output/'evidence_manifest.json'),
        historical_evidence_only=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--model-python', type=Path, required=True)
    parser.add_argument('--pretrained-root', type=Path, required=True)
    parser.add_argument('--ffmpeg-lib', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reclamation-receipt', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='plan_only_not_executed', checkpoint_copies=1,
                              storage_budget_bytes=26*GIB, original_job_1734785='remains held')))
        return 0
    if args.reclamation_receipt is None:
        parser.error('--execute requires a reviewed --reclamation-receipt')
    receipt = verify_reclamation(args.reclamation_receipt)
    if not args.ffmpeg_lib.is_dir():
        raise FileNotFoundError(args.ffmpeg_lib)
    helpers = Path(__file__).resolve().parent
    sys.path.insert(0, str(helpers))
    from bounded_probe import cache_environment, storage_usage
    source = args.source_root.resolve()
    sys.path.insert(0, str(source/'policies/validation'))
    from service_gpu_probe import verify_source
    source_sha = verify_source(source)
    if source_sha != args.source_manifest_sha256:
        raise ValueError('Source manifest differs from reviewed SHA256')
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'probe_owner.json', dict(kind='untrained_openwam_ephemeral_probe', output=str(output.resolve())))
    write_json(output/'input_review.json', dict(reclamation_receipt=receipt, source_manifest_sha256=source_sha,
        wrapper_sha256=sha256(__file__), optimizer_updates=0))
    environment = dict(os.environ, **cache_environment(output/'cache'))
    environment['PYTHONPATH'] = os.pathsep.join(str(source/path) for path in
        ['policies/src', 'policies/validation', 'benchmark/src', 'OpenWAM', 'dependency_overlay'])
    environment['DIFFSYNTH_MODEL_BASE_PATH'] = str(args.pretrained_root.absolute())
    environment['LD_LIBRARY_PATH'] = str(args.ffmpeg_lib.absolute()) + (
        os.pathsep+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    stage_file = output/'stage.txt'
    stage_file.write_text('setup\n')
    monitor_log = (output/'gpu_monitor.log').open('x')
    monitor = subprocess.Popen([str(args.model_python.absolute()), '-I', '-B', str(helpers/'monitor_gpu.py'),
        '--output', str(output/'gpu_memory.jsonl'), '--stage-file', str(stage_file)], env=environment,
        stdout=monitor_log, stderr=subprocess.STDOUT)
    stages, active, retired = [], None, False
    def interrupt(signum, frame):
        raise InterruptedError(f'OpenWAM probe interrupted by signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        native = source/'policies/validation'
        python = str(args.model_python.absolute())
        commands = [
            ('export', 1080, 25*GIB, ['/usr/bin/time', '-v', '-o', str(output/'export.time.txt'),
             python, '-B', str(native/'export_openwam_initialization.py'),
             '--config', str(source/'review_inputs/openwam_init_config.yaml'), '--normalization',
             str(source/'review_inputs/openwam_normalization_stats.npy'), '--pretrained-root', str(args.pretrained_root),
             '--output', str(output/'initialization'), '--execute-initialization', '--probe-legacy-normalization']),
            ('prepare', 360, 64*1024**2, [python, '-B', str(native/'service_gpu_probe.py'), 'prepare',
             '--source-root', str(source), '--model-python', python, '--entry-module', 'instrumented_policy',
             '--family', 'openwam', '--checkpoint-dir', str(output/'initialization'),
             '--checkpoint', 'checkpoint_step_000000.safetensors', '--checkpoint-step', '0',
             '--pretrained-root', str(args.pretrained_root), '--project-root', str(args.pretrained_root.parent),
             '--pooled-stats', str(source/'review_inputs/pooled_stats.json'), '--metadata-timeout', '350',
             '--service-timeout', '900', '--evidence-label', 'untrained_Piper_action_proprio_native_probe',
             '--output', str(output/'plan')]),
            ('predict', 1080, 64*1024**2, [python, '-B', str(native/'service_gpu_probe.py'), 'run',
             '--plan', str(output/'plan/plan.json'), '--output', str(output/'prediction')]),
        ]
        for name, seconds, file_limit, command in commands:
            stage_file.write_text(name+'\n')
            guarded = [python, '-I', '-B', str(helpers/'bounded_probe.py'),
                '--output', str(output/f'supervisor_{name}'), '--cache-root', str(output/'cache'),
                '--storage-root', str(output), '--stop-bytes', str(int(25.5*GIB)),
                '--cache-stop-bytes', str(512*1024**2), '--file-limit-bytes', str(file_limit),
                '--timeout', str(seconds)]
            for cache in cache_directories(output)[1:]:
                guarded += ['--extra-cache-root', str(cache)]
            guarded += ['--', *command]
            active = subprocess.Popen(guarded, env=environment)
            code = active.wait()
            active = None
            stages.append(dict(stage=name, exit_code=code))
            write_json(output/f'{name}_exit.json', stages[-1])
            if code:
                break
    finally:
        try:
            with cleanup_signals():
                if active is not None and active.poll() is None:
                    active.terminate()
                    try:
                        active.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        active.kill()
                        active.wait()
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
                monitor_log.close()
                try:
                    archive_and_retire(output, stages)
                    retired = True
                except Exception as exc:
                    write_json(output/'retirement_failed.json', dict(error=str(exc),
                        requires_root_audit=True, original_weights_untouched=True))
                write_json(output/'final_storage.json', storage_usage([output]))
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    return 0 if retired and len(stages) == 3 and all(stage['exit_code'] == 0 for stage in stages) else 1


if __name__ == '__main__':
    raise SystemExit(main())
