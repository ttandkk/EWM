"""Frozen pre-migration controller traces and completed v5 scene fingerprints."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from piper_bench.evaluation import controllers
from piper_bench.evaluation.runner import build_parser, load_settings, formal_inference_settings
from piper_bench.protocols.evaluation_protocol import digest

FIXTURES = Path(__file__).parent / 'fixtures'


def controller_trace(cls, options, latencies):
    calls = []
    def predict(obs, step, horizon):
        assert int(obs['obs_video'][0]) == step
        delay = latencies[len(calls) % len(latencies)]
        calls.append(step)
        actions = np.zeros((horizon, 7), np.float32)
        actions[:, 0] = 100 * step + np.arange(horizon)
        actions[:, 6] = -.5
        return dict(actions=actions, predicted=None, round_trip_seconds=delay, inference_seconds=delay)
    clock = {'now': lambda: 42.} if cls.__name__ == 'LatencyAwareActionStream' else {}
    stream = cls(predict, max_steps=45, **clock, **options)
    rows = []
    for step in range(45):
        obs = dict(obs_video=np.array([step], np.uint8), obs_voxel=np.array([step], np.float32),
                   proprio=np.array([step], np.float32), timing={'obs_video_time_s': step / 30})
        stream.ensure_request(obs, step)
        decision = stream.decision(step)
        rows.append({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in decision.items()})
    stream.close(45)
    return dict(calls=calls, rows=rows, requests=stream.requests)


@pytest.mark.parametrize('case', json.loads((FIXTURES / 'controller_traces.json').read_text())['cases'])
def test_controller_decisions_and_timing_match_old_implementation(case):
    trace = controller_trace(getattr(controllers, case['class']), case['options'], case['latencies'])
    assert digest(trace) == case['trace_sha256']


@pytest.mark.parametrize('task,expected', json.loads((FIXTURES / 'v5_scene_hashes.json').read_text())['tasks'].items())
def test_packaged_default_reproduces_completed_v5_scene(task, expected):
    args = build_parser().parse_args(['--task', task, '--condition', 'background_ood', '--plan-only'])
    scene = load_settings(args)[-1]
    assert digest(scene) == expected
    assert args.min_execution_steps == 8 and args.latency_controller == 'continuous'


@pytest.mark.parametrize('override', [['--scene-seed', '1'], ['--seed', '3'], ['--sampling-steps', '1'],
                                     ['--latency-controller', 'buffered'], ['--min-execution-steps', '4']])
def test_formal_profile_rejects_unmarked_diagnostic_overrides(override):
    args = build_parser().parse_args(['--task', 'conveyor', '--plan-only', *override])
    with pytest.raises(ValueError, match='diagnostics'):
        load_settings(args)
    args.interface_only = True
    load_settings(args)


def test_ewm_formal_label_requires_actual_formal_optimization():
    settings = dict(compile_core_active=True, compile_mode='default', cuda_graphs_enabled=False,
                    cache_context_kv=True, warmup_iters=3, startup_benchmark={'iterations': 3})
    hello = dict(policy_descriptor={'family': 'ewm'}, optimization=settings)
    assert formal_inference_settings(hello)
    for key, value in [('compile_core_active', False), ('compile_mode', 'reduce-overhead'),
                       ('cuda_graphs_enabled', True), ('warmup_iters', 1), ('startup_benchmark', None)]:
        assert not formal_inference_settings(dict(hello, optimization=dict(settings, **{key: value})))
