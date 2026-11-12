"""Time-aligned continuous inference on the measured-latency simulation clock."""
import numpy as np
import pytest

from scripts.eval_ewm import eval_ewm_piper_conveyor_sim as client
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as tasks


def observation(step):
    return dict(obs_video=np.array([step], np.uint8), obs_voxel=np.array([step], np.float32),
                proprio=np.array([step], np.float32), timing=dict(obs_video_time_s=step/30))


def predictor(latencies, calls, *, decode_latency=None):
    def predict(obs, step, horizon):
        assert obs['obs_video'].item() == step
        assert obs['timing']['obs_video_time_s'] == step/30
        assert set(obs) == {'obs_video', 'obs_voxel', 'proprio', 'timing'}
        latency = latencies[min(len(calls), len(latencies)-1)]
        calls.append((step, horizon))
        actions = np.zeros((horizon, 7), np.float32)
        actions[:, 0] = 100*step + np.arange(horizon)
        actions[:, 6] = -.5
        packet = dict(actions=actions, predicted=None, round_trip_seconds=latency,
                      inference_seconds=latency)
        if decode_latency is not None:
            packet.update(event_forecast_ready_round_trip_seconds=decode_latency,
                          charge_event_forecast_latency=True)
        return packet
    return predict


def tick(stream, step):
    stream.ensure_request(observation(step), step)
    return stream.decision(step)


def test_replans_on_every_reply_using_latest_observation_and_skips_first_prefix():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.04], calls), min_execution_steps=1, plan_horizon=40, max_steps=100)
    rows = [tick(stream, step) for step in range(9)]
    assert calls == [(0, 40), (2, 40), (4, 40), (6, 40), (8, 40)]
    assert all(r['action_source'] == 'bootstrap_wait_hold' for r in rows[:2])
    for step, row in enumerate(rows[2:], 2):
        anchor = (step//2-1)*2
        assert row['observation_step'] == anchor
        assert row['chunk_index'] == step-anchor
        assert row['action'][0] == 100*anchor+step-anchor
        assert row['prediction_target_step'] == step+1
    assert stream.requests[0]['discarded_prefix_steps'] == 2
    assert stream.requests[0]['first_action_step'] == 2
    assert stream.requests[0]['action_anchor_step'] == 0
    assert stream.wait_steps == stream.bootstrap_wait_steps == 2
    assert stream.underflow_steps == 0
    stream.close(9)
    assert stream.requests[-1]['reply_discarded_at_terminal']
    assert stream.pending is None


def test_old_forecast_continues_until_new_reply_replaces_it_at_current_index():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.04, .12, .04], calls),
                                                 min_execution_steps=1, plan_horizon=10, max_steps=30)
    rows = [tick(stream, step) for step in range(7)]
    assert calls == [(0, 10), (2, 10), (6, 10)]
    assert [r['action'][0] for r in rows[2:]] == [2, 3, 4, 5, 204]
    assert rows[5]['chunk_last'] and rows[5]['planned_end_reason'] == 'new_prediction'
    assert stream.requests[1]['discarded_prefix_steps'] == 4


def test_expired_first_chunk_is_never_shifted_forward_or_executed():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.10, .01], calls),
                                                 min_execution_steps=1, plan_horizon=2, max_steps=20)
    stream.gripper = -.75
    rows = [tick(stream, step) for step in range(5)]
    assert calls == [(0, 2), (3, 2), (4, 2)]
    assert stream.requests[0]['reply_expired']
    assert stream.requests[0]['discarded_prefix_steps'] == 2
    assert stream.requests[0]['executed_actions'] == 0
    assert 'first_action_step' not in stream.requests[0]
    for row in rows[:4]:
        np.testing.assert_array_equal(row['action'], [0, 0, 0, 0, 0, 0, -.75])
    assert rows[4]['action'][0] == 301
    assert rows[4]['observation_step'] == 3


def test_underflow_holds_gripper_without_repeating_relative_motion_and_recovers():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.01, .17, .01], calls),
                                                 min_execution_steps=1, plan_horizon=3, max_steps=20)
    rows = [tick(stream, step) for step in range(9)]
    assert calls == [(0, 3), (1, 3), (7, 3), (8, 3)]
    for row in rows[3:8]:
        assert row['action_source'] == 'buffer_underflow_hold'
        np.testing.assert_array_equal(row['action'], [0, 0, 0, 0, 0, 0, -.5])
    assert stream.requests[1]['reply_expired']
    assert rows[8]['action'][0] == 701
    assert stream.underflow_steps == 5


def test_zero_latency_uses_one_request_per_new_observation_and_stops_at_terminal():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([0.], calls), min_execution_steps=1, plan_horizon=40, max_steps=5)
    for step in range(5):
        stream.ensure_request(observation(step), step)
        stream.ensure_request(observation(step), step)
        row = stream.decision(step)
        assert row['chunk_index'] == 0
        assert row['action'][0] == 100*step
    stream.ensure_request(observation(5), 5)
    assert calls == [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)]
    assert stream.wait_steps == 0


@pytest.mark.parametrize('plan_horizon', [2, 10])
def test_decoding_delays_next_request_but_not_action_delivery_even_after_expiry(plan_horizon):
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.01], calls, decode_latency=.12),
                                                 min_execution_steps=1, plan_horizon=plan_horizon, max_steps=30)
    rows = [tick(stream, step) for step in range(5)]
    assert calls == [(0, plan_horizon), (4, plan_horizon)]
    assert rows[1]['action_source'] == 'policy'
    assert rows[1]['action'][0] == 1
    assert stream.requests[0]['request_pipeline_ready_step'] == 4


def test_fully_expired_reply_still_waits_for_decoder_before_replanning():
    calls = []
    stream = client.ContinuousLatencyActionStream(predictor([.04], calls, decode_latency=.17),
                                                 min_execution_steps=1, plan_horizon=2, max_steps=20)
    for step in range(7):
        tick(stream, step)
    assert calls == [(0, 2), (6, 2)]
    assert stream.requests[0]['reply_expired']
    assert stream.requests[0]['executed_actions'] == 0


@pytest.mark.parametrize('latency', [float('nan'), float('inf'), -.1])
def test_invalid_inference_latency_is_rejected(latency):
    stream = client.ContinuousLatencyActionStream(predictor([latency], []), min_execution_steps=1, plan_horizon=40, max_steps=100)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        tick(stream, 0)


def test_continuous_requires_latency_clock():
    args = tasks.build_parser().parse_args(['--task', 'tennis_falling', '--max-steps', '100',
                                           '--latency-controller', 'continuous'])
    from piper_bench.protocols.execution import load_execution
    args.min_execution_steps = load_execution(args.execution_profile).min_execution_steps
    client.validate_args(args)
    args.latency_aware = False
    with pytest.raises(ValueError, match='requires --latency-aware'):
        client.validate_args(args)
