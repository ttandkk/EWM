"""Task behavior is checked against independently generated legacy fingerprints."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import piper_bench
from piper_bench.collection.protocol import base_config, episode_spec, make_plan, protocol_camera
from piper_bench.protocols.dynamic_evaluation import numerical_equal
from piper_bench.scenes.task_assets import LAYOUTS, appearance_metadata
from piper_bench.tasks import TASK_MODULES, load_task


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/legacy_task_fingerprints.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'],
                         ids=lambda c: f"{c['task']}-{c['split']}-{c['seed']}")
def test_task_semantics_match_original_source(case):
    task, split, seed = case['task'], case['split'], case['seed']
    plan = make_plan(task, episodes=5, seed=seed, split=split)
    assert plan['version'] == case['version']
    payload = {
        'plan': plan,
        'default_config': load_task(task).Config().metadata(),
        'base_config': base_config(task, plan).metadata(),
        'camera': protocol_camera(task, plan),
        'specs': [episode_spec(plan, slot, retry) for slot in (0, 2, 4)
                  for retry in (0, 1, 3)],
    }
    if task in LAYOUTS:
        payload['appearance'] = appearance_metadata(task, seed)
    # Preserve the independently generated legacy payload's original fingerprint.
    # NumPy 2.2.6 and 2.3.1 differ by <=1.12e-16 in the legacy camera transforms;
    # comparing JSON bytes would incorrectly report a behavior change for all tasks.
    baseline = case['payload']
    assert hashlib.sha256(json.dumps(baseline, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest() == case['sha256']
    actual = json.loads(json.dumps(payload, allow_nan=False))
    assert numerical_equal(actual, baseline, atol=1e-14)


def test_registered_tasks_import_without_simulator_or_repository_path(tmp_path):
    code = """
import sys
from piper_bench.collection.protocol import make_plan, episode_spec
from piper_bench.tasks import TASK_MODULES, load_task
for name in TASK_MODULES:
    task = load_task(name)
    assert task.__name__.startswith('piper_bench.tasks.')
    assert task.Config().metadata()
    assert episode_spec(make_plan(name, episodes=2, seed=42), 0)
assert not any(name.startswith(('isaaclab', 'isaacsim', 'piper_tasks', 'piper_conveyor'))
               for name in sys.modules)
"""
    package_parent = str(Path(piper_bench.__file__).resolve().parent.parent)
    subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                   env={**os.environ, 'PYTHONPATH': package_parent,
                        'PYTHONDONTWRITEBYTECODE': '1'},
                   timeout=60, check=True, capture_output=True, text=True)


def test_canonical_task_imports_do_not_depend_on_legacy_scripts():
    package = Path(piper_bench.__file__).resolve().parent
    for directory in ('tasks', 'sensors', 'collection', 'robots', 'scenes'):
        for path in (package / directory).rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                           else [])
                for module in modules:
                    assert not (module.startswith('piper_') and not module.startswith('piper_bench')), (path, module)


def test_source_hashes_include_installed_canonical_implementation():
    from piper_bench.collection.protocol import source_hashes
    values = source_hashes()
    key = 'piper_bench/tasks/tennis/scene.py'
    path = Path(piper_bench.__file__).resolve().parent / 'tasks/tennis/scene.py'
    assert values[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(name.startswith('piper_bench/') and not Path(name).is_absolute()
               for name in values)


def test_first_party_assets_are_resolved_from_package():
    from piper_bench.assets.resolver import asset_path
    assert asset_path('piper_ball_props/catalog.json').is_file()
    assert asset_path('piper_tabletop_props/catalog.json').is_file()


def test_collection_plan_cli_accepts_explicit_argv_without_simulator(tmp_path, capsys):
    from piper_bench.collection.runner import main
    main(['--task', 'tennis_rolling', '--output', str(tmp_path / 'unused'),
          '--episodes', '3', '--seed', '42', '--plan-only'])
    assert json.loads(capsys.readouterr().out) == make_plan('tennis_rolling', 3, 42)
    assert not (tmp_path / 'unused').exists()


def test_writer_interpreter_comes_from_separate_deployment_profile(tmp_path, monkeypatch):
    from piper_bench.collection.runner import resolve_writer_python
    from piper_bench.deployment import config
    monkeypatch.delenv('PIPER_WRITER_PYTHON', raising=False)
    monkeypatch.setattr(config, 'load_config', lambda: {'profiles': {}})
    with pytest.raises(ValueError, match='setup --profile writer'):
        resolve_writer_python()
    monkeypatch.setattr(config, 'load_config',
                        lambda: {'profiles': {'writer': {'python': sys.executable}}})
    assert resolve_writer_python() == Path(sys.executable).absolute()
    monkeypatch.setenv('PIPER_WRITER_PYTHON', str(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='not found'):
        resolve_writer_python()
    assert resolve_writer_python(sys.executable) == Path(sys.executable).absolute()


def test_dataset_validation_help_has_no_writer_dependency(capsys):
    from piper_bench.collection.validation import main
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert '--episodes' in capsys.readouterr().out
