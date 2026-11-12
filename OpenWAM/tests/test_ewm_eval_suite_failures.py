"""A failed simulation process must not hide missing trials or block later cells."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    from piper_bench.evaluation import suite
    descriptor = dict(version='piper_policy_v1', family='ewm', mode='joint', action_horizon=24,
        event_horizon=24, action_dim=7, state_dim=8, control_hz=30,
        action_contract='piper_delta_eef_gripper_v1', rgb_shape=[2, 3, 256, 256],
        event_time_contract='aligned_event_frame_pre_action_v1', event_camera_mode='dual',
        event_cameras=2, voxel_bins=1, count_clip=15,
        training_observation_row_offset=0, inference_observation_row_offset=0,
        checkpoint_step=40000, checkpoint_sha256='1'*64, config_sha256='2'*64,
        normalization_sha256='3'*64, training_data={}, post_trial_visualization_supported=True)

    def settings(args):
        args.max_steps = 350
        args.latency_controller = args.latency_controller or 'continuous'
        args.min_execution_steps = args.min_execution_steps or 8
        episodes = [dict(seed=i, imaging=dict(condition=args.condition, group='dark' if i < 25 else 'bright'))
                    for i in range(args.trials)]
        if args.condition == 'speed_ood':
            for i, episode in enumerate(episodes):
                episode['imaging'] = dict(condition='id')
                episode['speed_condition'] = 'slow' if i < 25 else 'fast'
        return None, None, None, dict(task=args.task, condition=args.condition, episodes=episodes)

    monkeypatch.setattr(suite.client, 'load_settings', settings)
    monkeypatch.setattr(suite, 'source_hashes', lambda *args: {'source': 'test'})
    args = suite.parser().parse_args(['--tasks', 'conveyor', 'moving_carton', '--conditions', 'id',
                                      '--output-dir', str(tmp_path/'out')])

    def run():
        return suite.run(args, descriptor_provider=lambda task: descriptor,
                         policy_context=lambda task: nullcontext('ws://test-policy.invalid'))

    def artifacts(command, *, completed=False, successes=1, corrupt=None):
        options = suite.client.build_parser().parse_args(command[command.index('piper_bench.evaluation.runner')+1:])
        directory = options.output_dir
        scenes = json.loads(options.scene_manifest.read_text())
        manifest = dict(task=options.task, condition=options.condition,
            scene_manifest_sha256=suite.digest(scenes), policy_descriptor=dict(descriptor))
        if corrupt == 'checkpoint_sha256':
            manifest['policy_descriptor']['checkpoint_sha256'] = 'wrong'
        elif corrupt:
            manifest[corrupt] = 'wrong'
        suite.write_json(directory/'manifest.json', manifest)
        if completed:
            suite.write_json(directory/'complete.json', dict(complete=True, task=options.task,
                condition=options.condition, trials=50, successes=successes,
                success_rate=successes/50, formal_protocol=True))
        else:
            for i in range(2):
                video = directory/f'trial_{i:03d}.mp4'
                video.write_bytes(b'video')
                suite.write_json(directory/f'trial_{i:03d}_steps.json', [])
                suite.write_json(directory/f'trial_{i:03d}_result.json', dict(trial=i, video=str(video),
                    spec=scenes['episodes'][i], success=i < successes))
        return options

    return suite, args, artifacts, run


@pytest.mark.parametrize('controller', ['serial', 'buffered', 'continuous'])
def test_post_trial_prediction_recording_flag_reaches_every_task(evaluation, monkeypatch, controller):
    suite, args, artifacts, run = evaluation
    args.record_predictions = True
    args.latency_controller = controller
    calls = []

    def execute(command, **kwargs):
        options = artifacts(command, completed=True, successes=25)
        assert options.record_predictions
        assert options.latency_controller == controller
        calls.append(command)

    monkeypatch.setattr(suite.subprocess, 'run', execute)
    assert run() == 0
    assert len(calls) == 2
    manifest = json.loads((args.output_dir/'suite_manifest.json').read_text())
    assert all('--record-predictions' in cell['args'] for cell in manifest['cells'])


def test_exhausted_cell_continues_and_resume_fills_missing_trials(evaluation, monkeypatch):
    suite, args, artifacts, run = evaluation
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if '--task' in command and command[command.index('--task')+1] == 'conveyor':
            artifacts(command)
            raise subprocess.CalledProcessError(1, command)
        artifacts(command, completed=True, successes=25)

    monkeypatch.setattr(suite.subprocess, 'run', execute)
    assert run() == 1
    assert len(calls) == 2
    summary = json.loads((args.output_dir/'suite_summary.json').read_text())
    assert summary['execution_finished'] and not summary['complete'] and not summary['formal_protocol']
    assert summary['requested_trials'] == summary['trials'] == 100
    assert (summary['completed_trials'], summary['successes'], summary['failures'], summary['missing_trials']) == (52, 26, 26, 48)
    assert summary['success_rate'] == .26 and summary['success_rate_is_lower_bound']
    assert summary['by_condition']['id']['trials'] == 100
    assert summary['by_condition']['id']['missing_trials'] == 48
    failed = summary['results'][0]
    assert failed['status'] == 'infrastructure_error' and failed['error']['returncode'] == 1
    assert failed['success_rate'] == .02
    assert (args.output_dir/'conveyor/id/suite_failure.json').is_file()

    calls.clear()

    def resume(command, **kwargs):
        calls.append(command)
        assert '--resume' in command
        artifacts(command, completed=True, successes=20)

    monkeypatch.setattr(suite.subprocess, 'run', resume)
    assert run() == 0
    assert len(calls) == 1  # The already completed second task is not replayed.
    summary = json.loads((args.output_dir/'suite_summary.json').read_text())
    assert summary['complete'] and summary['formal_protocol']
    assert summary['completed_trials'] == 100 and summary['missing_trials'] == 0
    assert summary['successes'] == 45 and summary['success_rate'] == .45
    assert not summary['success_rate_is_lower_bound']


@pytest.mark.parametrize('failure', ['nonzero', 'missing_complete', 'launch_error'])
def test_startup_failures_keep_all_missing_trials_and_continue(evaluation, monkeypatch, failure):
    suite, args, artifacts, run = evaluation
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            if failure == 'nonzero':
                raise subprocess.CalledProcessError(1, command)
            if failure == 'launch_error':
                raise FileNotFoundError('could not launch client')
            return  # Even exit code zero needs a complete.json marker.
        artifacts(command, completed=True, successes=25)

    monkeypatch.setattr(suite.subprocess, 'run', execute)
    assert run() == 1 and len(calls) == 2
    summary = json.loads((args.output_dir/'suite_summary.json').read_text())
    assert summary['results'][0]['missing_trials'] == 50
    assert summary['completed_trials'] == 50 and summary['success_rate'] == .25


@pytest.mark.parametrize('corrupt', [None, 'checkpoint_sha256', 'scene_manifest_sha256'])
@pytest.mark.parametrize('condition,groups', [('lighting_ood', ('dark', 'bright')), ('speed_ood', ('slow', 'fast'))])
def test_partial_strata_use_planned_denominators(evaluation, monkeypatch, corrupt, condition, groups):
    suite, args, artifacts, run = evaluation
    args.tasks = ['conveyor']
    args.conditions = [condition]

    def execute(command, **kwargs):
        artifacts(command, corrupt=corrupt)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(suite.subprocess, 'run', execute)
    assert run() == 1
    result = json.loads((args.output_dir/'suite_summary.json').read_text())['results'][0]
    dark, bright = (result['breakdown'][group] for group in groups)
    assert dark['trials'] == bright['trials'] == 25
    assert bright['missing_trials'] == 25 and bright['success_rate'] == 0
    if corrupt:
        assert result['error']['partial_results_error']
        assert result['completed_trials'] == 0 and result['missing_trials'] == 50
        assert dark['missing_trials'] == 25 and dark['success_rate'] == 0
    else:
        assert result['error']['partial_results_error'] is None
        assert dark['completed_trials'] == 2 and dark['missing_trials'] == 23
        assert dark['success_rate'] == 1/25


def test_user_interrupt_is_not_misreported_as_infrastructure_error(evaluation, monkeypatch):
    suite, args, artifacts, run = evaluation
    calls = []

    def interrupt(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt

    monkeypatch.setattr(suite.subprocess, 'run', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert len(calls) == 1
    summary = json.loads((args.output_dir/'suite_summary.json').read_text())
    assert not summary['execution_finished'] and not summary['infrastructure_error_cells']
