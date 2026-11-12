"""Four-action minimum with continuous inference and time-aligned replies."""
import numpy as np
import pytest
from scripts.eval_ewm import eval_ewm_piper_conveyor_sim as client
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as tasks
from scripts.eval_ewm.ewm_eval_summary import plan_statistics


def observation(step):
    return dict(obs_video=np.array([step], np.uint8), obs_voxel=np.array([step], np.float32),
                proprio=np.array([step], np.float32), timing=dict(obs_video_time_s=step / 30))


def predictor(latencies, calls, *, forecast_latency=None):
    def predict(obs, step, horizon):
        assert obs['obs_video'].item() == step
        latency = latencies[min(len(calls), len(latencies) - 1)]
        calls.append(step)
        actions = np.zeros((horizon, 7), np.float32)
        actions[:, 0] = 100 * step + np.arange(horizon)
        actions[:, 6] = -.5
        result = dict(actions=actions, predicted=None, round_trip_seconds=latency,
                      inference_seconds=latency)
        if forecast_latency is not None:
            result.update(event_forecast_ready_round_trip_seconds=forecast_latency,
                          charge_event_forecast_latency=True)
        return result
    return predict


def tick(stream, step):
    stream.ensure_request(observation(step), step)
    return stream.decision(step)


def test_default_four_keeps_inferring_and_uses_freshest_valid_result():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.04], calls), plan_horizon=24, max_steps=20)
    rows = [tick(stream, step) for step in range(11)]
    assert calls == [0, 2, 4, 6, 8, 10]
    assert [r['action_source'] for r in rows[:2]] == ['bootstrap_wait_hold'] * 2
    assert [r['action'][0] for r in rows[2:6]] == [2, 3, 4, 5]
    assert [r['action'][0] for r in rows[6:10]] == [402, 403, 404, 405]
    assert rows[5]['chunk_last'] and rows[9]['chunk_last']
    assert stream.requests[1]['reply_superseded'] and stream.requests[1]['executed_actions'] == 0
    assert stream.requests[2]['received_step'] == stream.requests[2]['activated_step'] == 6
    assert stream.requests[2]['discarded_prefix_steps'] == 2
    assert stream.requests[2]['first_action_step'] == 6
    assert stream.requests[0]['executed_actions'] == 4
    stream.close(11)
    stats = plan_statistics([dict(inference_requests=stream.requests)])
    assert stats['superseded'] >= 2
    assert stats['request_interval_steps']['mean'] == 2
    assert stats['replan_interval_steps']['mean'] == 4
    assert stats['executed_actions_per_received_plan']['mean'] < 4


def test_waiting_candidate_ages_and_drops_prefix_at_switch():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.04, .01, .2], calls), plan_horizon=24, max_steps=20)
    rows = [tick(stream, step) for step in range(7)]
    assert calls == [0, 2, 3]
    assert rows[6]['observation_step'] == 2 and rows[6]['chunk_index'] == 4
    assert rows[6]['action'][0] == 204
    assert stream.requests[1]['received_step'] == 3
    assert stream.requests[1]['activated_step'] == 6
    assert stream.requests[1]['discarded_prefix_steps'] == 4


def test_expired_active_plan_may_end_before_minimum_and_never_repeats_motion():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([0., .2, .01], calls), plan_horizon=2, max_steps=12)
    rows = [tick(stream, step) for step in range(9)]
    assert rows[0]['action_source'] == rows[1]['action_source'] == 'policy'
    for row in rows[2:8]:
        assert row['action_source'] == 'buffer_underflow_hold'
        np.testing.assert_array_equal(row['action'], [0, 0, 0, 0, 0, 0, -.5])
    assert stream.requests[0]['min_execution_unmet_reason'] == 'plan_exhausted'
    assert stream.requests[1]['reply_expired']
    assert rows[8]['action_source'] == 'policy'


def test_fresh_plan_can_replace_exhausted_plan_before_minimum():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([0., .01, .2], calls), plan_horizon=2, max_steps=10)
    rows = [tick(stream, step) for step in range(4)]
    assert rows[0]['action'][0] == 0
    assert rows[1]['action'][0] == 1
    assert rows[2]['action'][0] == 101
    assert stream.requests[0]['min_execution_unmet_reason'] == 'plan_exhausted'


def test_zero_latency_at_most_one_request_per_tick_and_terminal_truncation():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([0.], calls), plan_horizon=24, max_steps=6)
    rows = []
    for step in range(6):
        stream.ensure_request(observation(step), step)
        stream.ensure_request(observation(step), step)
        rows.append(stream.decision(step))
    stream.close(6)
    assert calls == list(range(6))
    assert [r['replan'] for r in rows] == [0, 0, 0, 0, 3, 3]
    assert stream.requests[3]['min_execution_unmet_reason'] == 'terminal'
    assert stream.requests[4]['reply_superseded']
    assert stream.requests[5]['reply_discarded_at_terminal']


def test_deferred_decoder_does_not_delay_action_but_gates_requests():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.01], calls, forecast_latency=.12),
                                                 plan_horizon=24, max_steps=15)
    rows = [tick(stream, step) for step in range(7)]
    assert calls == [0, 4]
    assert rows[1]['action_source'] == 'policy'
    assert rows[1]['action'][0] == 1
    assert [r['replan'] for r in rows[1:5]] == [0] * 4
    assert rows[5]['replan'] == 1


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_rejects_invalid_minimum(value):
    with pytest.raises(ValueError, match='positive integer'):
        client.ContinuousLatencyActionStream(lambda *args: None, plan_horizon=24, max_steps=10,
                                             min_execution_steps=value)


def test_default_cli_and_opt_in_legacy_switching():
    parser = tasks.build_parser()
    args = parser.parse_args(['--task', 'tennis_falling', '--max-steps', '100', '--latency-controller', 'continuous'])
    from piper_bench.protocols.execution import load_execution
    assert args.min_execution_steps is None  # resolved from the explicit profile
    assert load_execution(args.execution_profile).min_execution_steps == 8
    assert load_execution("legacy_buffered_min4_v1").min_execution_steps == 4
    assert parser.parse_args(['--task', 'tennis_falling', '--min-execution-steps', '1']).min_execution_steps == 1
    args.min_execution_steps = 0
    with pytest.raises(ValueError, match='min-execution-steps'):
        client.validate_args(args)


def test_simulated_loop_records_each_plan_once_with_zero_latency(monkeypatch, tmp_path):
    import types
    import torch
    from pathlib import Path

    monkeypatch.setattr(client, 'time', types.SimpleNamespace(perf_counter=lambda: 42.0))
    monkeypatch.setattr(client, 'policy_request', lambda conn, args, request, arrays: (request, {}))
    def validate(reply, values, horizon, event_cameras, args, count_clip):
        actions = np.zeros((horizon, 7), np.float32)
        actions[:, 0] = np.arange(horizon)
        actions[:, 6] = 1.
        return actions, None, 0.
    monkeypatch.setattr(client, 'validate_reply', validate)
    class Runtime:
        def __init__(self):
            self.env = types.SimpleNamespace(device=torch.device('cpu'))
            self.time = 0.
    runtime = Runtime()
    def capture(runtime, dvs, command, cameras):
        step = round(runtime.time * 30) + 1
        runtime.time = step / 30
        obs = observation(step)
        obs['event_counts'] = {}
        return obs, step == 10, None
    class Video:
        def write(self, *args, **kwargs):
            pass
    args = types.SimpleNamespace(seed=42, max_steps=10, sampling_steps=4,
        decode_predicted_events=False, latency_controller='continuous', min_execution_steps=4,
        execution_horizon=10, profile_timing=False, trials=1)
    rows, chunks, step_timings = [], [], []
    outcome = client.run_latency_loop(args, runtime, None, None, {'horizon': 24}, 80000,
        ('cam0', 'cam_wrist'), observation(0), Video(), rows, chunks, step_timings,
        Path(tmp_path / 'heartbeat'), tmp_path, 0, capture=capture, stable_steps=lambda _: 0)
    assert outcome[1] is True and outcome[3] == 10
    assert [(c['executed_horizon'], c['end_reason']) for c in chunks] == [
        (4, 'new_prediction'), (4, 'new_prediction'), (2, 'success')]
    assert sum(c['executed_horizon'] for c in chunks) == len(rows)
    assert [r['step'] for r in rows if r['chunk_end_reason'] == 'new_prediction'] == [4, 8]
    assert outcome[-1]['min_execution_steps'] == 4


@pytest.mark.parametrize('latencies,horizon', [
    ([0.], 24), ([.01], 24), ([.04], 24), ([.1], 24),
    ([.01, .07, .02, .11], 24), ([.8, .01, .12], 24), ([.01, .2], 3),
])
def test_varied_latency_preserves_alignment_and_minimum(latencies, horizon):
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor(latencies, calls), plan_horizon=horizon, max_steps=45)
    rows = [tick(stream, step) for step in range(45)]
    stream.close(45)
    assert len(calls) == len(set(calls))
    for step, row in enumerate(rows):
        if row['action_source'] == 'policy':
            assert row['chunk_index'] == step - row['observation_step']
            assert row['action'][0] == 100 * row['observation_step'] + row['chunk_index']
            assert row['prediction_target_step'] == step + 1
        else:
            np.testing.assert_array_equal(row['action'][:6], np.zeros(6))
    for request in stream.requests:
        used = request.get('executed_actions', 0)
        if 0 < used < 4:
            assert request['min_execution_unmet_reason'] in ('plan_exhausted', 'terminal')
        if request.get('reply_superseded') or request.get('reply_expired'):
            assert used == 0
