"""Checkpoint-based compatibility launcher; model metadata stays model-side."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def main(argv=None):
    from piper_bench.evaluation import suite
    from piper_bench.deployment.config import simulation_python
    from piper_bench.deployment.process import package_environment

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint-dir', type=Path)
    group.add_argument('--checkpoint-map', type=Path)
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--config')
    parser.add_argument('--normalization')
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--family', choices=('ewm', 'openwam', 'fastwam'), default='ewm')
    parser.add_argument('--model-python', default=os.environ.get('PIPER_POLICY_PYTHON', os.environ.get('OWPY', sys.executable)))
    parser.add_argument('--sim-python', default=os.environ.get('PIPER_SIM_PYTHON', os.environ.get('SIMPY')))
    parser.add_argument('--pretrained-root')
    parser.add_argument('--project-root', default=os.environ.get('EWM_PROJECT_ROOT'))
    parser.add_argument('--service-timeout', type=float, default=1800)
    args, remaining = parser.parse_known_args(argv)
    options = suite.parser().parse_args(remaining)
    if options.plan_only:
        return suite.run(options)
    mapping = json.loads(args.checkpoint_map.read_text()) if args.checkpoint_map else {}
    if args.checkpoint_dir:
        mapping = {task: str(args.checkpoint_dir.absolute()) for task in options.tasks}
    if any(task not in mapping for task in options.tasks):
        raise ValueError('execution requires checkpoint-dir or a complete checkpoint-map')
    # The orchestration process runs in simulation Python. Describe and serve
    # always use the explicitly selected, independent model interpreter.
    if not options.preflight:
        interpreter = args.sim_python or str(simulation_python())
        if os.path.abspath(interpreter) != os.path.abspath(sys.executable):
            command = [interpreter, '-B', '-m', 'piper_policies.legacy_suite',
                       *(sys.argv[1:] if argv is None else argv), '--model-python', args.model_python,
                       '--sim-python', interpreter]
            with package_environment(include_policies=True) as environment:
                return subprocess.run(command, check=False, env=environment).returncode
    cache = {}

    def model_args(task):
        result = ['--family', args.family, '--checkpoint-dir', str(Path(mapping[task]).expanduser().absolute())]
        if args.checkpoint_step is not None:
            result += ['--checkpoint-step', str(args.checkpoint_step)]
        for field in ('checkpoint', 'config', 'normalization'):
            if getattr(args, field) is not None:
                result += ['--' + field, getattr(args, field)]
        for entry in args.path_map:
            result += ['--path-map', entry]
        if args.pretrained_root:
            result += ['--pretrained-root', args.pretrained_root]
        if args.project_root:
            result += ['--project-root', args.project_root]
        result += ['--sampling-steps', str(options.sampling_steps)]
        for variable, flag in (
            ('COMPILE_CORE', 'compile-core'), ('COMPILE_CUDA_GRAPHS', 'cuda-graphs'),
            ('CACHE_CONTEXT_KV', 'cache-context-kv'), ('CACHE_BF16_WEIGHTS', 'cache-bf16-weights'),
            ('ALLOW_COMPILE_ACTION_ERROR', 'allow-compile-action-error'),
        ):
            if variable in os.environ:
                value = os.environ[variable].lower()
                if value not in ('true', 'false'):
                    raise ValueError(f'{variable} must be true or false')
                result += ['--' + ('' if value == 'true' else 'no-') + flag]
        for variable, flag in (('COMPILE_MODE', 'compile-mode'), ('WARMUP_ITERS', 'warmup-iters'),
                               ('BENCHMARK_ITERS', 'benchmark-iters')):
            if variable in os.environ:
                result += ['--' + flag, os.environ[variable]]
        return result

    def descriptor(task):
        key = str(Path(mapping[task]).expanduser().absolute())
        if key not in cache:
            with package_environment(include_policies=True) as environment:
                value = subprocess.run([args.model_python, '-B', '-m', 'piper_policies', 'describe', *model_args(task)],
                                       check=True, capture_output=True, text=True, env=environment)
            cache[key] = json.loads(value.stdout)
        return cache[key]

    @contextmanager
    def service(task):
        logs = options.output_dir / 'policy_logs'
        logs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='piper-policy-') as temp, (logs / (task + '.log')).open('a') as log, \
                package_environment(include_policies=True) as environment:
            ready = Path(temp) / 'ready.json'
            command = [args.model_python, '-B', '-m', 'piper_policies', 'serve', *model_args(task),
                       '--ready-file', str(ready)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=environment)
            try:
                deadline = time.monotonic() + args.service_timeout
                while True:
                    if process.poll() is not None:
                        raise ConnectionError(f'policy exited before ready; inspect {log.name}')
                    if ready.exists():
                        try:
                            metadata = json.loads(ready.read_text())
                            break
                        except json.JSONDecodeError:
                            pass  # Writer may still be flushing its first line.
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'policy startup timed out; inspect {log.name}')
                    time.sleep(.2)
                if metadata.get('policy_descriptor') != descriptor(task):
                    raise ValueError('running policy differs from preflight descriptor')
                yield metadata['url']
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    return suite.run(options, descriptor_provider=descriptor, policy_context=service)


if __name__ == '__main__':
    raise SystemExit(main())
