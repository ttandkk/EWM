"""Regression tests for snapshot integrity and owned policy launch failures."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_verification_rejects_added_python_file(tmp_path):
    freeze = module_from_file('launch_review_freeze', ROOT / 'launch/freeze.py')
    root, destination = tmp_path / 'root', tmp_path / 'snapshot'
    package = root / 'benchmark/src/piper_bench'
    package.mkdir(parents=True)
    (package / 'cli.py').write_text('pass\n')
    freeze.freeze(root, destination)
    freeze.verify(destination)
    (destination / 'benchmark/src/sitecustomize.py').write_text('changed = True\n')
    with pytest.raises(ValueError):
        freeze.verify(destination)


@pytest.mark.parametrize('task_args', [['--task', 'conveyor'], ['--task=conveyor']])
def test_single_task_submission_uses_suite_cpu_preflight(tmp_path, monkeypatch, task_args):
    from piper_bench.evaluation import suite
    submit = module_from_file('launch_review_submit', ROOT / 'launch/slurm/submit_evaluation.py')
    snapshot = tmp_path / 'frozen source'
    monkeypatch.setattr(submit, 'freeze', lambda root, destination: destination.mkdir())
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] != 'sbatch':
            assert command[:2] == [str(snapshot / 'ewm'), 'evaluate']
            args = suite.parser().parse_args(command[2:])
            assert args.preflight and args.tasks == ['conveyor']
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(submit.subprocess, 'run', run)
    assert submit.main(['--snapshot', str(snapshot), '--partition', 'unit-test', '--gres', 'gpu:1',
                        '--log', str(tmp_path / 'logs/unit.log'), '--', *task_args,
                        '--policy-url', 'ws://test.invalid', '--output-dir', str(tmp_path / 'results')]) == 0
    assert len(calls) == 2 and calls[1][0] == 'sbatch'  # The submission above is mocked.


def test_legacy_policy_startup_failure_keeps_later_cells_and_denominator(tmp_path, monkeypatch):
    pytest.importorskip('piper_policies')
    from piper_policies import legacy_suite
    from piper_policies.metadata import load_metadata
    from piper_bench.evaluation import suite
    bundle = tmp_path / 'model'
    bundle.mkdir()
    (bundle / 'checkpoint_step_80000.pt').write_bytes(b'test checkpoint metadata only')
    (bundle / 'config.yaml').write_text('model: {training_mode: joint}\ndataloader: {chunk_size: 24}\n')
    (bundle / 'normalization.json').write_text('{"voxel":{"num_bins":1,"count_clip":15}}')
    descriptor = load_metadata(bundle)[3]
    processes, simulations, terminated = [], [], []

    class Process:
        def __init__(self, command, **kwargs):
            self.bridge = Path(kwargs['env']['PYTHONPATH'].split(os.pathsep)[0])
            assert (self.bridge / 'piper_policies/__init__.py').is_file()
            self.pid = 100000 + len(processes)
            self.returncode = 1 if not processes else None
            processes.append(self)
            if self.returncode is None:
                ready = Path(command[command.index('--ready-file') + 1])
                ready.write_text(json.dumps(dict(url='ws://test.invalid', policy_descriptor=descriptor)))

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert self.bridge.exists()
            self.returncode = -15
            return self.returncode

    def run(command, **kwargs):
        if 'describe' in command:
            bridge = Path(kwargs['env']['PYTHONPATH'].split(os.pathsep)[0])
            assert (bridge / 'piper_policies/__init__.py').is_file()
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(descriptor))
        assert 'piper_bench.evaluation.runner' in command
        simulations.append(command)
        options = suite.client.build_parser().parse_args(command[command.index('piper_bench.evaluation.runner') + 1:])
        scenes = json.loads(options.scene_manifest.read_text())
        suite.write_json(options.output_dir / 'manifest.json', dict(task=options.task, condition=options.condition,
            scene_manifest_sha256=suite.digest(scenes), policy_descriptor=descriptor))
        suite.write_json(options.output_dir / 'complete.json', dict(complete=True, task=options.task,
            condition=options.condition, trials=50, successes=25, formal_protocol=True))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(legacy_suite.subprocess, 'Popen', Process)
    monkeypatch.setattr(legacy_suite.subprocess, 'run', run)
    monkeypatch.setattr(legacy_suite.os, 'killpg', lambda pid, sig: terminated.append(pid))
    output = tmp_path / 'results'
    assert legacy_suite.main(['--checkpoint-dir', str(bundle), '--model-python', sys.executable,
        '--sim-python', sys.executable, '--tasks', 'conveyor', 'moving_carton', '--conditions', 'id',
        '--output-dir', str(output)]) == 1
    summary = json.loads((output / 'suite_summary.json').read_text())
    assert summary['execution_finished'] and not summary['complete']
    assert summary['requested_trials'] == 100 and summary['completed_trials'] == 50
    assert summary['missing_trials'] == 50 and summary['successes'] == 25
    assert summary['success_rate'] == .25 and summary['infrastructure_error_cells'] == 1
    assert len(processes) == 2 and len(simulations) == 1
    assert terminated == [processes[1].pid]
    assert all(not process.bridge.exists() for process in processes)
