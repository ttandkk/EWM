"""New collection snapshots include canonical code; historical snapshots stay intact."""
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pytest

from piper_bench.collection.freeze import freeze_collection_source, frozen_pythonpath, hash_snapshot


ROOT = Path(__file__).resolve().parents[2]


def launcher():
    spec = importlib.util.spec_from_file_location('collection_freeze_test_launcher',
                                                ROOT / 'scripts/collect_piper_eight_dynamic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_snapshot_imports_all_tasks_and_assets_without_live_checkout(tmp_path):
    source = freeze_collection_source(ROOT, tmp_path / 'run with spaces/source')
    environment = dict(os.environ, PYTHONPATH=frozen_pythonpath(source), PYTHONDONTWRITEBYTECODE='1')
    code = """
from pathlib import Path
import sys
import piper_bench
from piper_bench.tasks import TASK_MODULES, load_task
from piper_bench.collection.protocol import make_plan, episode_spec
from piper_bench.assets.resolver import asset_path
assert Path(piper_bench.__file__).is_relative_to(Path(sys.argv[1]))
for task in TASK_MODULES:
    assert Path(load_task(task).__file__).is_relative_to(Path(sys.argv[1]))
    assert episode_spec(make_plan(task, 2, 42), 0)
assert asset_path('piper_ball_props/catalog.json').is_file()
assert asset_path('piper_tabletop_props/catalog.json').is_file()
assert 'isaaclab.app' not in sys.modules
"""
    subprocess.run([sys.executable, '-B', '-c', code, str(source)], cwd=tmp_path,
                   env=environment, timeout=60, check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, '-B', str(source / 'scripts/collect_piper_tasks_evis.py'),
                             '--task', 'conveyor', '--output', str(tmp_path / 'unused'), '--plan-only'],
                            cwd=tmp_path, env=environment, timeout=60, check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout)['task'] == 'conveyor'
    assert not (tmp_path / 'unused').exists()
    hashes = hash_snapshot(source)
    resource = 'benchmark/src/piper_bench/resources/assets/piper_ball_props/catalog.json'
    assert resource in hashes and 'benchmark/src/piper_bench/collection/runner.py' in hashes
    (source / resource).write_text((source / resource).read_text() + '\n')
    assert hash_snapshot(source)[resource] != hashes[resource]


def test_slim_checkout_freezes_without_legacy_directories(tmp_path):
    root = tmp_path / 'slim checkout'
    shutil.copytree(ROOT / 'benchmark/src/piper_bench', root / 'benchmark/src/piper_bench',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = freeze_collection_source(root, tmp_path / 'snapshot')
    environment = dict(os.environ, PYTHONPATH=frozen_pythonpath(source), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', str(source / 'scripts/collect_piper_tasks_evis.py'),
                             '--task', 'conveyor', '--output', str(tmp_path / 'unused'), '--plan-only'],
                            cwd=tmp_path, env=environment, timeout=60, check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout)['task'] == 'conveyor'
    subprocess.run([sys.executable, '-B', str(source / 'scripts/validate_piper_tasks_lerobot.py'), '--help'],
                   cwd=tmp_path, env=environment, timeout=60, check=True, capture_output=True, text=True)
    assert not (tmp_path / 'unused').exists()
    assert not (root / 'Event-DataCollection').exists()
    assert (source / 'benchmark/src/piper_bench/resources/assets/piper_ball_props/catalog.json').is_file()
    with pytest.raises(FileExistsError):
        freeze_collection_source(root, source)


def test_old_snapshot_path_and_bytes_are_never_upgraded(tmp_path):
    source = tmp_path / 'old/source'
    scripts = source / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'old.py').write_text('ORIGINAL = True\n')
    before = hash_snapshot(source)
    assert frozen_pythonpath(source) == str(scripts)
    with pytest.raises(FileExistsError):
        freeze_collection_source(ROOT, source)
    assert hash_snapshot(source) == before
    assert not (source / 'benchmark').exists()


def test_existing_launcher_resume_only_validates_old_manifest(tmp_path):
    module = launcher()
    run = tmp_path / 'historical'
    args = module.parser().parse_args(['--run-dir', str(run), '--dataset-root', str(tmp_path / 'data'), '--resume'])
    source = run / 'source/scripts'
    source.mkdir(parents=True)
    (source / 'original.py').write_text('LEGACY = 1\n')
    module.write(run / 'launch.json', module.settings(args))
    module.write(run / 'source_sha256.json', hash_snapshot(run / 'source'))
    before = hash_snapshot(run)
    module.prepare(args)
    assert hash_snapshot(run) == before
    assert module.source_environment(run)['PYTHONPATH'] == str(source)


def test_carton_prepare_hashes_package_resources_and_legacy_assets(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('carton_freeze_test_launcher',
                                                ROOT / 'scripts/recollect_piper_moving_carton.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'checkout'
    files = {
        'benchmark/src/piper_bench/__init__.py': 'VERSION = 1\n',
        'benchmark/src/piper_bench/resources/calibration.json': '{"version": 1}\n',
        'Event-DataCollection/scripts/collector.py': 'ORIGINAL = True\n',
        'Event-DataCollection/assets/legacy.json': '{}\n',
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    monkeypatch.setattr(module, 'ROOT', root)
    run = tmp_path / 'prepared'
    module.prepare(run)
    hashes = json.loads((run / 'launch.json').read_text())['source_sha256']
    assert 'source/benchmark/src/piper_bench/resources/calibration.json' in hashes
    assert 'source/assets/legacy.json' in hashes
    assert not (root / 'datasets').exists()
