"""Saved terminal truth and complete-action checks without a simulator."""
import copy
import json
from pathlib import Path

import pytest

from piper_bench.evaluation.replay import load_replay, main, rejudge_result, rejudge_trial


CASES = json.loads((Path(__file__).parent / 'fixtures/recorded_terminal_cases.json').read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('case', CASES, ids=('saved_success', 'saved_failure'))
def test_rejudge_physical_terminal_truth_from_formal_v5(case, tmp_path):
    path = tmp_path / 'trial_000_result.json'
    write(path, case['result'])
    before = path.read_bytes()
    result = rejudge_trial(path)
    assert result['terminal_success'] == case['result']['success']
    assert result['agrees']
    assert result['simulation_executed'] is False
    assert path.read_bytes() == before


def test_rejudge_does_not_trust_saved_success_label(tmp_path):
    case = copy.deepcopy(CASES[0]['result'])
    case['terminal']['stable_steps'] = 0
    write(tmp_path / 'trial_000_result.json', case)
    report = rejudge_result(tmp_path)
    assert report['recorded_successes'] == 1
    assert report['terminal_successes'] == 0
    assert len(report['mismatches']) == 1
    assert not report['passed']


def test_rejudge_reports_malformed_results_instead_of_counting_as_failure(tmp_path):
    write(tmp_path / 'trial_000_result.json', {'success': True})
    report = rejudge_result(tmp_path)
    assert report['trials'] == 1 and report['rejudged'] == 0
    assert report['errors'] and not report['passed']


@pytest.fixture
def replay_files(tmp_path):
    result = copy.deepcopy(CASES[0]['result'])
    result.update(trial=0, steps=2)
    spec = result['spec']
    write(tmp_path / 'manifest.json', dict(task=spec['task'], control_hz=30))
    write(tmp_path / 'scene_manifest.json', dict(episodes=[spec]))
    write(tmp_path / 'trial_000_result.json', result)
    write(tmp_path / 'trial_000_initial_state.json', dict(spec=spec, initial={}))
    rows = [dict(step=i, action=[0., 0., 0., 0., 0., 0., 1.], simulation_time_s=i/30)
            for i in (1, 2)]
    write(tmp_path / 'trial_000_steps.json', rows)
    return tmp_path


def test_replay_requires_complete_action_trace(replay_files):
    loaded = load_replay(replay_files, 0)
    assert len(loaded['steps']) == 2
    assert set(loaded['input_sha256']) == {'manifest', 'scenes', 'result', 'initial', 'steps'}
    write(replay_files / 'trial_000_steps.json', loaded['steps'][:1])
    with pytest.raises(ValueError, match='every executed'):
        load_replay(replay_files, 0)


@pytest.mark.parametrize('mutation,expected', [
    ('nan', 'seven finite'), ('gap', 'contiguous'), ('clock', '30 Hz'), ('width', 'seven finite')])
def test_replay_rejects_invalid_action_data(replay_files, mutation, expected):
    path = replay_files / 'trial_000_steps.json'
    rows = json.loads(path.read_text())
    if mutation == 'nan':
        rows[0]['action'][0] = float('nan')
    elif mutation == 'gap':
        rows[1]['step'] = 3
    elif mutation == 'clock':
        rows[1]['simulation_time_s'] = 5.
    else:
        rows[0]['action'].pop()
    write(path, rows)
    with pytest.raises(ValueError, match=expected):
        load_replay(replay_files, 0)


def test_replay_rejects_inconsistent_initial_scene(replay_files):
    path = replay_files / 'trial_000_initial_state.json'
    value = json.loads(path.read_text())
    value['spec']['seed'] += 1
    write(path, value)
    with pytest.raises(ValueError, match='different scenes'):
        load_replay(replay_files, 0)


def test_rejudge_cli_writes_separate_report_and_preserves_source(tmp_path, capsys):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'trial_000_result.json'
    write(path, CASES[0]['result'])
    before = path.read_bytes()
    output = tmp_path / 'audit'
    assert main(['--result-dir', str(source), '--output-dir', str(output), '--rejudge-only']) == 0
    report = json.loads((output / 'rejudge_report.json').read_text())
    assert report['passed'] and not report['simulation_executed']
    assert path.read_bytes() == before
    assert 'terminal_rejudge' in capsys.readouterr().out
    with pytest.raises(ValueError, match='outside'):
        main(['--result-dir', str(source), '--output-dir', str(source / 'new'), '--rejudge-only'])
