"""Slurm resources are separate from benchmark and model configuration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freeze import freeze, verify


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--partition', required=True)
    p.add_argument('--gres', required=True, help='Cluster-specific GPU resource, e.g. gpu:1')
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('--memory', default='64G')
    p.add_argument('--time', default='08:00:00')
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('evaluation_args', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    argv = args.evaluation_args
    if argv[:1] == ['--']:
        argv = argv[1:]
    if not argv:
        p.error('pass evaluate options after --')
    # A one-task submission is still a one-cell suite with CPU preflight and
    # failure accounting, rather than the low-level simulator-only entrypoint.
    argv = ['--tasks' if arg == '--task' else '--tasks=' + arg.split('=', 1)[1]
            if arg.startswith('--task=') else arg for arg in argv]
    root = Path(__file__).resolve().parents[2]
    snapshot = args.snapshot.expanduser().absolute()
    model_launcher = any(a.split('=', 1)[0] in ('--checkpoint-dir', '--checkpoint-map') for a in argv)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', EWM_PROJECT_ROOT=str(root))
    environment['PYTHONPATH'] = os.pathsep.join((str(snapshot / 'benchmark/src'), str(snapshot / 'policies/src'),
                                               str(snapshot / 'OpenWAM')))
    if model_launcher:
        python = environment.get('PIPER_POLICY_PYTHON')
        if not python or not environment.get('PIPER_SIM_PYTHON'):
            p.error('checkpoint launcher requires PIPER_POLICY_PYTHON and PIPER_SIM_PYTHON')
        command = [python, '-B', '-m', 'piper_policies.legacy_suite', *argv]
    else:
        command = [str(snapshot / 'ewm'), 'evaluate', *argv]
    verify_command = [os.environ.get('PIPER_CLI_PYTHON', 'python3'), str(snapshot / 'launch/freeze.py'),
                      '--output', str(snapshot), '--verify']
    submission = ['sbatch', '--parsable', '--partition=' + args.partition, '--gres=' + args.gres,
                  '--cpus-per-task=' + str(args.cpus), '--mem=' + args.memory, '--time=' + args.time,
                  '--nodes=1', '--ntasks=1', '--job-name=piper-eval', '--output=' + str(args.log.absolute()),
                  '--error=' + str(args.log.absolute()), '--wrap',
                  shlex.join(verify_command) + ' && ' + shlex.join(command)]
    if args.dry_run:
        print(json.dumps(dict(snapshot=str(snapshot), command=submission,
                             pythonpath=environment['PYTHONPATH']), indent=2))
        return 0
    if snapshot.exists():
        verify(snapshot)
    else:
        freeze(root, snapshot)
    # A CPU preflight must succeed before requesting cluster resources.
    subprocess.run(command + ['--preflight'], env=environment, check=True)
    args.log.absolute().parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(submission, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
