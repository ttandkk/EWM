"""Continuous plans include expiry, episode truncation and censorship."""
from scripts.eval_ewm.ewm_eval_summary import plan_statistics


def test_plan_statistics_counts_zero_execution_and_excludes_pending():
    requests = [
        dict(observation_step=0, received_step=2, discarded_prefix_steps=2, executed_actions=3, round_trip_seconds=.06),
        dict(observation_step=2, received_step=26, discarded_prefix_steps=24, executed_actions=0, reply_expired=True, round_trip_seconds=.8),
        dict(observation_step=26, received_step=28, discarded_prefix_steps=2, executed_actions=1, round_trip_seconds=.06),
        dict(observation_step=28, reply_discarded_at_terminal=True),
    ]
    result = plan_statistics([dict(inference_requests=requests, bootstrap_wait_steps=2, buffer_underflow_steps=4)])
    assert (result["submitted"], result["received"], result["used"], result["censored"], result["expired"]) == (4, 3, 2, 1, 1)
    assert result["executed_actions_per_received_plan"]["mean"] == 4 / 3
    assert result["executed_actions_per_used_plan"]["mean"] == 2
    assert result["executed_actions_per_received_plan"]["p50"] == 1
    assert result["expired_rate"] == 1 / 3
    assert result["bootstrap_wait_steps"] == 2 and result["buffer_underflow_steps"] == 4


def test_plan_statistics_keeps_episode_intervals_separate():
    result = plan_statistics([dict(inference_requests=[dict(observation_step=8)]),
                              dict(inference_requests=[dict(observation_step=0)])])
    assert result["replan_interval_steps"]["count"] == 0
    assert result["executed_actions_per_received_plan"]["mean"] is None
    assert result["censored"] == 2
