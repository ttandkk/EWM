"""Exercise the public CPU commands without a configured simulator or policy."""
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest


def test_same_simulation_interpreter_applies_runtime_configuration(monkeypatch):
    from piper_bench import cli
    from piper_bench.deployment import config
    monkeypatch.setenv('PIPER_TEST_RUNTIME_CONFIGURED', 'no')
    monkeypatch.setattr(config, 'simulation_python', lambda: Path(sys.executable))
    monkeypatch.setattr(config, 'runtime_environment', lambda: {'PIPER_TEST_RUNTIME_CONFIGURED': 'yes'})
    assert cli._simulation('evaluate', []) is None
    assert os.environ['PIPER_TEST_RUNTIME_CONFIGURED'] == 'yes'

from piper_bench.cli import main


@pytest.mark.parametrize('command', ['setup', 'assets', 'doctor', 'demo', 'collect',
                                    'evaluate', 'validate', 'replay'])
def test_command_help_needs_no_simulation_environment(command, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('PIPER_HOME', str(tmp_path / 'unconfigured'))
    monkeypatch.delenv('PIPER_SIM_PYTHON', raising=False)
    monkeypatch.delenv('PIPER_CONFIG', raising=False)
    with pytest.raises(SystemExit) as exc:
        main([command, '--help'])
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out
    assert not (tmp_path / 'unconfigured').exists()


@pytest.mark.parametrize('task_arg', [['--task', 'conveyor'], ['--task=conveyor']])
def test_single_task_plan_dispatch_supports_argparse_forms(task_arg, capsys):
    assert main(['evaluate', *task_arg, '--condition', 'background_ood', '--plan-only']) in (0, None)
    scene = json.loads(capsys.readouterr().out)
    assert scene['task'] == 'conveyor'
    assert len(scene['episodes']) == 50
    assert scene['imaging']['version'] == 'piper_eight_dynamic_background_ood_v5_grey_green'


def test_full_eight_task_id_and_background_suite_can_plan_without_dataset(tmp_path, capsys):
    output = tmp_path / 'planned'
    assert main(['evaluate', '--conditions', 'id', 'background_ood', '--output-dir', str(output),
                 '--plan-only']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['cells'] == 16 and printed['trials'] == 800
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    assert manifest['requested_trials'] == 800
    assert len(manifest['cells']) == 16
    for cell in manifest['cells']:
        scenes = json.loads((output / cell['task'] / cell['condition'] / 'planned_scenes.json').read_text())
        assert len(scenes['episodes']) == 50
        assert scenes['step_budget']['effective_max_steps'] > 0
        if cell['condition'] == 'background_ood':
            assert scenes['imaging']['version'] == 'piper_eight_dynamic_background_ood_v5_grey_green'
        else:
            assert scenes['imaging']['background'] is None


@pytest.mark.parametrize('override', [['--sampling-steps', '1'], ['--scene-seed', '1'], ['--seed', '1']])
def test_formal_protocol_overrides_require_diagnostic_mode(override, capsys):
    argv = ['evaluate', '--task', 'conveyor', '--plan-only', *override]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    capsys.readouterr()
    assert main([*argv, '--interface-only']) in (0, None)
    assert len(json.loads(capsys.readouterr().out)['episodes']) == 50


def test_real_policy_service_preflight_and_action_roundtrip(tmp_path, capsys):
    pytest.importorskip('piper_policies', reason='optional model-service package is required for this integration')
    pytest.importorskip('websockets', minversion='15',
                        reason='model-service sync server requires websockets >=15; sim is client-only')
    import numpy as np
    from websockets.sync.client import connect
    from piper_policies.contract import action_reply, validate_request
    from piper_policies.metadata import load_metadata
    from piper_policies.server import serve_policy
    from piper_bench.evaluation.policy_client import request, policy_request, validate_hello, validate_reply
    from piper_bench.evaluation.runner import build_parser, load_settings
    from piper_bench.protocols.suite import load_suite

    bundle = tmp_path / 'model'
    bundle.mkdir()
    (bundle / 'checkpoint_step_80000.pt').write_bytes(b'CPU integration metadata; not a real model')
    (bundle / 'config.yaml').write_text('model:\n  training_mode: joint\n  event_camera_mode: dual\n  voxel_bins: 1\ndataloader:\n  chunk_size: 24\n')
    metadata = dict(voxel=dict(num_bins=1, count_clip=15), sources=dict(conveyor=dict(
        collection_sha256=load_suite()['tasks']['conveyor']['collection_file_sha256'])))
    (bundle / 'normalization.json').write_text(json.dumps(metadata))
    descriptor = load_metadata(bundle)[3]
    descriptor['post_trial_visualization_supported'] = False
    owner_threads, errors = [], []

    class Policy:
        def __init__(self):
            self.descriptor = descriptor

        def handle(self, header, arrays):
            owner_threads.append(threading.get_ident())
            observation = validate_request(header, arrays, descriptor)
            return action_reply(np.zeros((24, 7), np.float32), observation, descriptor, .01)

    ready = tmp_path / 'ready.json'
    def serve():
        try:
            serve_policy(Policy(), ready_file=ready)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    endpoint, deadline = None, time.monotonic() + 10
    try:
        while endpoint is None and not errors and time.monotonic() < deadline:
            try:
                endpoint = json.loads(ready.read_text())['url']
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(.02)
        assert not errors and endpoint is not None
        output = tmp_path / 'preflight'
        assert main(['evaluate', '--tasks', 'conveyor', '--conditions', 'background_ood',
                     '--preflight', '--policy-url', endpoint, '--output-dir', str(output)]) == 0
        suite = json.loads((output / 'suite_manifest.json').read_text())
        assert suite['policy_descriptors']['conveyor'] == descriptor
        assert not list(output.rglob('trial_*_result.json'))
        assert not owner_threads  # Preflight inspected metadata; it did not infer.
        args = build_parser().parse_args(['--task', 'conveyor', '--plan-only'])
        load_settings(args)
        with connect(endpoint, compression=None) as conn:
            hello, _ = request(conn, {'op': 'hello'})
            step, cameras = validate_hello(hello, args)
            reply, values = policy_request(conn, args, dict(op='predict', step=step, seed=42,
                sampling_steps=4, action_chunk_size=8, prompt='Move both objects into the basket.',
                event_time_contract='aligned_event_frame_pre_action_v1', observation_row_offset=0,
                timing=dict(obs_video_time_s=0., proprio_time_s=0., obs_event_start_s=-1/30, obs_event_end_s=0.)),
                dict(obs_video=np.zeros((2, 3, 256, 256), np.uint8),
                     proprio=np.zeros(8, np.float32), obs_voxel=np.zeros((2, 2, 256, 256), np.float32)))
            actions, prediction, seconds = validate_reply(reply, values, 8, cameras, args, 15)
            assert actions.shape == (8, 7) and prediction is None and seconds == .01
            assert reply['observation_step'] == 0 and reply['observation_time_s'] == 0.
        assert owner_threads == [thread.ident]
    finally:
        if endpoint is not None and thread.is_alive():
            with connect(endpoint, compression=None) as conn:
                request(conn, {'op': 'shutdown'})
        thread.join(timeout=10)
    assert not thread.is_alive() and not errors and not ready.exists()


def test_demo_only_offers_tasks_with_the_shared_preview_interface():
    import ast
    from pathlib import Path
    import piper_bench
    from piper_bench.cli import DEMO_TASKS
    package = Path(piper_bench.__file__).parent
    for task in DEMO_TASKS:
        tree = ast.parse((package / 'tasks' / task / 'preview.py').read_text())
        assert any(isinstance(node, ast.ImportFrom) and node.module == 'piper_bench.collection.preview'
                   and any(alias.name == 'main' for alias in node.names) for node in ast.walk(tree)), task


@pytest.mark.parametrize('task', ['moving_carton', 'turntable_cups', 'rotating_tube'])
def test_demo_rejects_incompatible_old_preview_before_startup(task, monkeypatch, capsys):
    from piper_bench.deployment import config
    monkeypatch.setattr(config, 'simulation_python', lambda: pytest.fail('unsupported demo must not launch a simulator'))
    with pytest.raises(SystemExit) as exc:
        main(['demo', '--task', task])
    assert exc.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
