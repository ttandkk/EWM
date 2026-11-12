"""Prepare or explicitly run one real policy-service request; never execute actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


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


def verify_source(root):
    expected = json.loads((root / 'probe_source_manifest.json').read_text())['sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.name != 'probe_source_manifest.json' and '__pycache__' not in p.parts}
    if actual != set(expected):
        raise ValueError('Frozen source inventory differs')
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    return sha256(root / 'probe_source_manifest.json')


def environment(source, output):
    env = dict(os.environ)
    paths = ['policies/src', 'policies/validation', 'benchmark/src', 'OpenWAM', 'FastWAM/src', 'dependency_overlay']
    env.update(PYTHONPATH=os.pathsep.join(str(source / path) for path in paths),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               DIFFSYNTH_SKIP_DOWNLOAD='true', WANDB_MODE='disabled',
               TORCHINDUCTOR_CACHE_DIR=str(output / 'inductor_cache'),
               TRITON_CACHE_DIR=str(output / 'triton_cache'))
    return env


def observation(descriptor, state):
    import numpy as np
    video = np.zeros((2, 3, 256, 256), dtype=np.uint8)
    video[0, 0] = np.arange(256, dtype=np.uint8)[None, :]
    video[1, 2] = np.arange(256, dtype=np.uint8)[:, None]
    arrays = dict(obs_video=video, proprio=np.asarray(state, dtype=np.float32))
    request = dict(op='predict', raw=True, step=descriptor['checkpoint_step'], seed=42,
                   sampling_steps=4, action_chunk_size=1, observation_step=1,
                   observation_time_s=1/30, control_hz=30,
                   prompt='Move the object into the container.',
                   timing=dict(obs_video_time_s=1/30, proprio_time_s=1/30))
    if descriptor['mode'] == 'joint':
        request['event_time_contract'] = descriptor['event_time_contract']
        request['timing'].update(obs_event_start_s=0., obs_event_end_s=1/30)
        arrays['obs_voxel'] = np.zeros((descriptor['event_cameras'], 2*descriptor['voxel_bins'], 256, 256), np.float32)
        arrays['obs_voxel'][..., 96:160, 96:160] = 1.
    else:
        request['observation_contract'] = descriptor['observation_contract']
    return request, arrays


def validate_optimization(hello, required):
    if not required:
        return False
    import math
    from piper_bench.evaluation.runner import formal_inference_settings
    if not formal_inference_settings(hello):
        raise ValueError('Actual compile/default/no-graphs/KV/warmup3/benchmark3 differs from formal settings')
    settings = hello['optimization']
    if not settings.get('cache_bf16_weights') or settings.get('warmup_sampling_steps') != 4:
        raise ValueError('Formal joint probe requires actual BF16 cache and flow4 warmup')
    benchmark = settings['startup_benchmark']
    for name in ('action_max_abs_error', 'baseline_eager_no_kv_cache_mean_seconds',
                 'eager_kv_cache_mean_seconds', 'optimized_mean_seconds'):
        if not isinstance(benchmark.get(name), (int, float)) or not math.isfinite(benchmark[name]) or benchmark[name] < 0:
            raise ValueError(f'Actual startup benchmark is missing a finite {name}')
    return True


def prepare(args):
    source = args.source_root.resolve()
    snapshot = verify_source(source)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.model_python.expanduser().absolute()), '-B', '-m', getattr(args, 'entry_module', 'piper_policies'), 'describe',
               '--family', args.family, '--checkpoint-dir', str(args.checkpoint_dir.resolve()),
               '--checkpoint', args.checkpoint, '--checkpoint-step', str(args.checkpoint_step),
               '--pretrained-root', str(args.pretrained_root.resolve()), '--project-root', str(args.project_root.resolve()),
               '--device', 'cuda:0']
    result = subprocess.run(command, env=environment(source, output), check=True, capture_output=True, text=True,
                            timeout=getattr(args, 'metadata_timeout', 180))
    descriptor = json.loads(result.stdout)
    state = json.loads(args.pooled_stats.read_text())['observation.state']['mean']
    plan = dict(version='piper_real_gpu_probe_v1', source_root=str(source), snapshot_sha256=snapshot,
                command=command, descriptor=descriptor, evidence_label=args.evidence_label,
                input_kind='synthetic_dual_rgb_events_with_real_training_state_mean',
                state=state, state_stats_path=str(args.pooled_stats.resolve()), state_stats_sha256=sha256(args.pooled_stats),
                formal_optimization=args.formal_optimization, service_timeout_s=args.service_timeout,
                metadata_timeout_s=getattr(args, 'metadata_timeout', 180),
                external_pretrained_content_verified=False,
                scope='one hello and one physical action response; no simulator, training, or success-rate claim')
    write_json(output / 'plan.json', plan)
    print(json.dumps(dict(status='prepared_not_run', plan=str(output / 'plan.json'), checkpoint_sha256=descriptor['checkpoint_sha256'])))


def run(args):
    import numpy as np
    plan = json.loads(args.plan.read_text())
    source = Path(plan['source_root'])
    if verify_source(source) != plan['snapshot_sha256']:
        raise ValueError('Snapshot manifest differs from reviewed plan')
    # The client imports the same frozen protocol sources, never a live checkout.
    for path in reversed(['policies/src', 'benchmark/src', 'dependency_overlay']):
        sys.path.insert(0, str(source / path))
    from piper_bench.evaluation.wire import MAX_MESSAGE, decode_message, encode_message
    from piper_policies.contract import ACTION_CONTRACT, validate_request
    from websockets.sync.client import connect
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = environment(source, output)
    current = subprocess.run(plan['command'], env=env, check=True, capture_output=True, text=True,
                             timeout=plan.get('metadata_timeout_s', 180))
    if json.loads(current.stdout) != plan['descriptor']:
        raise ValueError('Checkpoint/config/statistics/source differs from reviewed descriptor')
    request, arrays = observation(plan['descriptor'], plan['state'])
    validate_request(request, arrays, plan['descriptor'])
    np.savez(output / 'observation.npz', **arrays)
    write_json(output / 'request.json', request)
    ready = output / 'ready.json'
    command = list(plan['command'])
    command[4] = 'serve'
    command += ['--ready-file', str(ready)]
    if plan['descriptor']['family'] == 'ewm' and not plan['formal_optimization']:
        command += ['--no-compile-core', '--warmup-iters', '0', '--benchmark-iters', '0']
    write_json(output / 'execution.json', dict(command=command, scope=plan['scope'], plan_sha256=sha256(args.plan)))
    env['PROBE_RESOURCE_METRICS'] = str(output / 'server_resources.json')
    def terminate(signum, frame):
        raise TimeoutError(f'Probe interrupted by signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    with (output / 'server.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = started + plan['service_timeout_s']
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Policy exited with {process.returncode}; inspect {log.name}')
                if ready.exists():
                    try:
                        metadata = json.loads(ready.read_text())
                        break
                    except json.JSONDecodeError:
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy startup exceeded reviewed timeout')
                time.sleep(.2)
            startup = time.monotonic() - started
            with connect(metadata['url'], max_size=MAX_MESSAGE, compression=None, open_timeout=30) as connection:
                def exchange(header, payload=None):
                    connection.send(encode_message(header, payload or {}))
                    reply, data = decode_message(connection.recv(timeout=600))
                    if not reply.get('ok', True):
                        raise RuntimeError(reply.get('error', reply))
                    return reply, data
                hello, _ = exchange({'op': 'hello'})
                if hello['policy_descriptor'] != plan['descriptor']:
                    raise ValueError('Running descriptor differs from reviewed plan')
                write_json(output / 'hello.json', hello)
                formal_matches = validate_optimization(hello, plan['formal_optimization'])
                call_started = time.monotonic()
                reply, prediction = exchange(request, arrays)
                roundtrip = time.monotonic() - call_started
                actions = prediction['executed_actions']
                if actions.shape != (1, 7) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                    raise ValueError('Physical action must be finite float32[1,7]')
                if (reply['action_contract'] != ACTION_CONTRACT or reply['observation_step'] != 1
                        or abs(reply['observation_time_s'] - 1/30) > 1e-9 or reply['control_hz'] != 30
                        or not np.isfinite(reply['inference_seconds']) or reply['inference_seconds'] < 0):
                    raise ValueError('Action timestamp/contract/inference timing mismatch')
                np.save(output / 'actions.npy', actions, allow_pickle=False)
                write_json(output / 'reply.json', reply)
                exchange({'op': 'shutdown'})
            process.wait(timeout=30)
            if process.returncode != 0:
                raise RuntimeError(f'Service shutdown exit {process.returncode}')
            write_json(output / 'result.json', dict(status='passed', evidence_label=plan['evidence_label'],
                       startup_seconds=startup, roundtrip_seconds=roundtrip,
                       action_max_abs=float(np.abs(actions).max()), action_sha256=sha256(output/'actions.npy'),
                       formal_optimization=plan['formal_optimization'], formal_configuration_verified=formal_matches,
                       actual_optimization=hello.get('optimization', {}),
                       normalization_adapter=hello['policy_descriptor'].get('normalization_adapter'), scope=plan['scope']))
        except BaseException as exc:
            write_json(output / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='CPU metadata only; never starts a model service')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--model-python', type=Path, required=True)
    p.add_argument('--family', choices=['ewm', 'openwam', 'fastwam'], required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-step', type=int, required=True)
    p.add_argument('--pretrained-root', type=Path, required=True)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--pooled-stats', type=Path, required=True)
    p.add_argument('--evidence-label', required=True)
    p.add_argument('--formal-optimization', action='store_true')
    p.add_argument('--service-timeout', type=float, default=1200)
    p.add_argument('--metadata-timeout', type=float, default=180)
    p.add_argument('--entry-module', choices=['piper_policies', 'instrumented_policy'], default='piper_policies')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run', help='EXPLICIT GPU service startup and one request; requires an allocation')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return prepare(args) if args.command == 'prepare' else run(args)


if __name__ == '__main__':
    main()
