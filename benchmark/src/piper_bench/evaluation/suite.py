"""Plan and run the versioned eight-task suite against independent services."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

from . import runner as client
from ..collection.sampling import TASKS
from ..collection.protocol import source_hashes, write_json
from ..protocols.dynamic_evaluation import CONDITIONS, DEFAULT_CONDITIONS, supported_conditions, condition_breakdown, numerical_equal
from ..protocols.evaluation_protocol import digest, load_scene_manifest
from ..protocols.execution import DEFAULT_PROFILE, PROFILES, load_execution
from ..protocols.ood_protocol import BACKGROUND_VERSION, BACKGROUND_PROFILES
from ..protocols.suite import DEFAULT_SUITE


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path)
    endpoints = p.add_mutually_exclusive_group()
    endpoints.add_argument('--policy-url', help='one multitask policy service')
    endpoints.add_argument('--policy-map', type=Path, help='JSON mapping task to WebSocket endpoint')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(DEFAULT_CONDITIONS))
    p.add_argument('--suite', default=DEFAULT_SUITE)
    p.add_argument('--execution-profile', choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    p.add_argument('--background-profile', choices=BACKGROUND_PROFILES, default=BACKGROUND_VERSION)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--preflight', action='store_true', help='validate service contracts and write scenes without a GPU')
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--scene-seed', type=int, default=20280002)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--execution-horizon', type=int, default=10)
    p.add_argument('--min-execution-steps', type=int)
    p.add_argument('--sampling-steps', type=int, default=4)
    p.add_argument('--record-predictions', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--latency-controller', choices=('serial', 'buffered', 'continuous'))
    p.add_argument('--latency-aware', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--interface-only', action='store_true')
    p.add_argument('--max-steps', type=int)
    return p


def inspect_policy(endpoint, options, task, dataset, suite):
    from websockets.sync.client import connect
    with connect(endpoint, open_timeout=30, max_size=client.shared.MAX_MESSAGE, compression=None) as conn:
        hello, _ = client.shared.request(conn, {'op': 'hello'})
    client.shared.validate_hello(hello, options)
    if 'policy_descriptor' not in hello:
        raise ValueError('Suite services must implement piper_policy_v1; run ewm serve')
    descriptor = hello['policy_descriptor']
    client.verify_policy_data(descriptor, task, dataset, suite)
    return descriptor


def failed_cell(cell, args, error, policy_descriptors=None):
    directory = Path(cell['output'])
    rows, validation_error = [], None
    try:
        if (directory / 'manifest.json').exists():
            manifest = json.loads((directory / 'manifest.json').read_text())
            if (manifest['task'] != cell['task'] or manifest['condition'] != cell['condition']
                    or manifest['scene_manifest_sha256'] != digest(cell['scenes'])
                    or manifest.get('policy_descriptor') != cell['policy_descriptor']):
                raise ValueError('partial cell used a different task, scene list or policy')
            rows = client.shared.load_completed_results(directory, args.trials)
            for row in rows:
                if (not numerical_equal(row['spec'], cell['scenes']['episodes'][row['trial']])
                        or type(row.get('success')) is not bool):
                    raise ValueError('partial trial scene/result differs')
        elif list(directory.glob('trial_*_result.json')):
            raise ValueError('partial trials have no run manifest')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rows, validation_error = [], f'{type(exc).__name__}: {exc}'
    expected = condition_breakdown([dict(spec=s, success=False) for s in cell['scenes']['episodes']])
    observed = condition_breakdown(rows)
    breakdown = {}
    for name, planned in expected.items():
        actual = observed.get(name, dict(successes=0, trials=0))
        breakdown[name] = trial_counts(actual['successes'], actual['trials'], planned['trials'],
                                       interface_only=args.interface_only)
    result = dict(task=cell['task'], condition=cell['condition'], status='infrastructure_error',
                  complete=False, formal_protocol=False, breakdown=breakdown, output=str(directory),
                  error=dict(type='infrastructure_error', message=str(error),
                             returncode=getattr(error, 'returncode', None), partial_results_error=validation_error),
                  **trial_counts(sum(bool(r['success']) for r in rows), len(rows), args.trials,
                                 interface_only=args.interface_only))
    write_json(directory / 'suite_failure.json', result)
    return result


def run(args, *, descriptor_provider=None, policy_context=None):
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.conditions)) != len(args.conditions):
        raise ValueError('duplicate tasks or conditions')
    if args.trials != 50 and not args.interface_only:
        raise ValueError('formal suite requires 50 trials; use --interface-only for diagnostics')
    endpoints = json.loads(args.policy_map.read_text()) if args.policy_map else {}
    if args.policy_url:
        endpoints = {task: args.policy_url for task in args.tasks}
    if not args.plan_only and policy_context is None and any(task not in endpoints for task in args.tasks):
        raise ValueError('execution/preflight requires policy-url or a complete policy-map')
    out = args.output_dir.expanduser().absolute()
    cells, skipped, descriptors = [], [], {}
    for task in args.tasks:
        for condition in args.conditions:
            if condition not in supported_conditions(task):
                skipped.append(dict(task=task, condition=condition, reason='unsupported task condition'))
                continue
            argv = ['--task', task, '--condition', condition, '--suite', args.suite,
                    '--execution-profile', args.execution_profile, '--background-profile', args.background_profile,
                    '--trials', str(args.trials), '--scene-seed', str(args.scene_seed), '--seed', str(args.seed),
                    '--execution-horizon', str(args.execution_horizon), '--sampling-steps', str(args.sampling_steps),
                    '--latency-aware' if args.latency_aware else '--no-latency-aware']
            if args.dataset_root:
                argv += ['--dataset-dir', str((args.dataset_root / ('piper_' + task)).absolute())]
            for name in ('min_execution_steps', 'latency_controller', 'max_steps'):
                value = getattr(args, name)
                if value is not None:
                    argv += ['--' + name.replace('_', '-'), str(value)]
            if args.record_predictions is not None:
                argv += ['--record-predictions' if args.record_predictions else '--no-record-predictions']
            if args.interface_only:
                argv += ['--interface-only']
            options = client.build_parser().parse_args(argv + ['--plan-only'])
            dataset, _, _, scenes = client.load_settings(options)
            if not args.plan_only and task not in descriptors:
                if descriptor_provider:
                    descriptor = descriptor_provider(task)
                    client.shared.validate_hello({'policy_descriptor': descriptor,
                                                  'horizon': descriptor['action_horizon']}, options)
                    client.verify_policy_data(descriptor, task, dataset, args.suite)
                    descriptors[task] = descriptor
                else:
                    descriptors[task] = inspect_policy(endpoints[task], options, task, dataset, args.suite)
            cells.append(dict(task=task, condition=condition, args=argv, scenes=scenes,
                              policy_descriptor=descriptors.get(task), output=str(out / task / condition),
                              max_steps=options.max_steps))
    if not cells:
        raise ValueError('no supported task/condition combinations')
    manifest = dict(version='piper_eight_eval_suite_v2', suite=args.suite,
                    execution_profile=load_execution(args.execution_profile).metadata(),
                    trials_per_cell=args.trials, requested_trials=len(cells) * args.trials,
                    skipped=skipped, policy_descriptors=descriptors, source_sha256=source_hashes(),
                    cells=[{k: v for k, v in cell.items() if k != 'scenes'} |
                           {'scene_sha256': digest(cell['scenes'])} for cell in cells])
    path = out / 'suite_manifest.json'
    if path.exists() and not numerical_equal(json.loads(path.read_text()), manifest):
        raise ValueError('suite model/data/code/options changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, manifest)
    for cell in cells:
        directory = Path(cell['output'])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'planned_scenes.json'
        if path.exists():
            load_scene_manifest(path, cell['scenes'])
        else:
            write_json(path, cell['scenes'])
    print(json.dumps(dict(cells=len(cells), trials=manifest['requested_trials'], skipped=skipped,
                         plan_only=args.plan_only or args.preflight)), flush=True)
    if args.plan_only or args.preflight:
        return 0
    results = []
    write_json(out / 'suite_summary.json', suite_summary(cells, results, args, finished=False))
    for cell in cells:
        directory = Path(cell['output'])
        complete = directory / 'complete.json'
        if not complete.exists():
            command = [sys.executable, '-B', '-m', 'piper_bench.evaluation.runner', *cell['args'],
                       '--output-dir', str(directory),
                       '--scene-manifest', str(directory / 'planned_scenes.json'),
                       '--headless', '--enable_cameras', '--device', 'cpu']
            if (directory / 'manifest.json').exists():
                command.append('--resume')
            try:
                context = policy_context(cell['task']) if policy_context else nullcontext(endpoints[cell['task']])
                with context as endpoint:
                    subprocess.run([*command, '--policy-url', endpoint], check=True)
                if not complete.exists():
                    raise FileNotFoundError('cell exited without complete.json')
            except (subprocess.CalledProcessError, OSError) as exc:
                results.append(failed_cell(cell, args, exc))
                write_json(out / 'suite_summary.json', suite_summary(cells, results, args, finished=False))
                continue
        result = json.loads(complete.read_text())
        run_manifest = json.loads((directory / 'manifest.json').read_text())
        if (not result.get('complete') or result['trials'] != args.trials
                or result['task'] != cell['task'] or result['condition'] != cell['condition']
                or run_manifest['scene_manifest_sha256'] != digest(cell['scenes'])
                or run_manifest.get('policy_descriptor') != cell['policy_descriptor']):
            raise ValueError('completed cell result or policy differs')
        results.append(dict(task=cell['task'], condition=cell['condition'], status='complete', complete=True,
                            **trial_counts(result['successes'], result['trials'], args.trials,
                                           interface_only=args.interface_only),
                            formal_protocol=result['formal_protocol'], breakdown=result.get('breakdown', {}),
                            output=str(directory)))
        write_json(out / 'suite_summary.json', suite_summary(cells, results, args, finished=False))
    summary = suite_summary(cells, results, args, finished=True)
    write_json(out / 'suite_summary.json', summary)
    return 0 if summary['complete'] else 1


def main(argv=None):
    return run(parser().parse_args(argv))


def trial_counts(successes, completed, requested, *, interface_only):
    if not 0 <= successes <= completed <= requested:
        raise ValueError('invalid evaluation trial counts')
    return dict(trials=requested, requested_trials=requested, completed_trials=completed,
                successes=successes, failures=completed-successes, missing_trials=requested-completed,
                success_rate=None if interface_only else successes/requested,
                success_rate_denominator='requested_trials', success_rate_is_lower_bound=completed < requested)

def suite_summary(cells, results, args, *, finished):
    complete = finished and len(results) == len(cells) and all(r['complete'] for r in results)
    counts = trial_counts(sum(r['successes'] for r in results), sum(r['completed_trials'] for r in results),
                          len(cells)*args.trials, interface_only=args.interface_only)
    by_condition = {}
    for condition in args.conditions:
        planned = [c for c in cells if c['condition'] == condition]
        selected = [r for r in results if r['condition'] == condition]
        if planned:
            by_condition[condition] = dict(tasks=len(planned),
                completed_tasks=sum(r['complete'] for r in selected),
                infrastructure_error_tasks=sum(r['status'] == 'infrastructure_error' for r in selected),
                **trial_counts(sum(r['successes'] for r in selected), sum(r['completed_trials'] for r in selected),
                               len(planned)*args.trials, interface_only=args.interface_only))
    return dict(complete=complete, execution_finished=finished, requested_cells=len(cells),
                completed_cells=sum(r['complete'] for r in results),
                infrastructure_error_cells=sum(r['status'] == 'infrastructure_error' for r in results),
                formal_protocol=complete and all(r['formal_protocol'] for r in results),
                results=results, by_condition=by_condition, **counts)


if __name__ == "__main__":
    raise SystemExit(main())
