"""Cross-interpreter launches forward packages without forwarding dependencies."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import piper_bench
from piper_bench import cli
from piper_bench.deployment.process import package_environment


def test_package_bridge_preserves_sources_cleans_up_and_leaves_target_unchanged(tmp_path):
    source = tmp_path / 'explicit-source'
    source.mkdir()
    parent_site = tmp_path / 'lib/python3.10/site-packages'
    environment = {'PYTHONPATH': os.pathsep.join([str(parent_site), str(source)]),
                   'PYTHONHOME': '/wrong/interpreter'}
    package = Path(piper_bench.__file__).resolve().parent
    original = (package / '__init__.py').read_bytes()
    with pytest.raises(RuntimeError, match='child failed'):
        with package_environment(environment) as child:
            bridge = Path(child['PYTHONPATH'].split(os.pathsep)[0])
            assert sorted(p.name for p in bridge.iterdir()) == ['piper_bench']
            assert (bridge / 'piper_bench').is_symlink()
            assert (bridge / 'piper_bench').resolve() == package
            assert child['PYTHONPATH'].split(os.pathsep)[1:] == [str(source)]
            assert 'PYTHONHOME' not in child
            raise RuntimeError('child failed')
    assert not bridge.exists()
    assert (package / '__init__.py').read_bytes() == original
    assert environment['PYTHONHOME'] == '/wrong/interpreter'


@pytest.mark.parametrize('command', ['simulation', 'demo', 'validate'])
def test_cli_subprocesses_keep_bridge_alive_until_exit(command, monkeypatch, tmp_path):
    from piper_bench.deployment import config
    from piper_bench.collection import runner
    target = tmp_path / 'other-python'
    monkeypatch.setattr(config, 'simulation_python', lambda: target)
    monkeypatch.setattr(config, 'runtime_environment', lambda: {})
    monkeypatch.setattr(runner, 'resolve_writer_python', lambda: target)
    seen = []

    def run(argv, *, env, check):
        bridge = Path(env['PYTHONPATH'].split(os.pathsep)[0])
        assert (bridge / 'piper_bench/__init__.py').is_file()
        seen.append(bridge)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, 'run', run)
    if command == 'simulation':
        result = cli._simulation('collect', [])
    elif command == 'demo':
        result = cli.demo(['--output-dir', str(tmp_path / 'demo')])
    else:
        result = cli.main(['validate', str(tmp_path / 'dataset'), '--episodes', '1'])
    assert result == 7 and len(seen) == 1
    assert not seen[0].exists()


@pytest.mark.parametrize('family_args,profile', [
    ([], 'policy-ewm'), (['--family', 'ewm'], 'policy-ewm'),
    (['--family=openwam'], 'policy-ewm'), (['--family', 'fastwam'], 'policy-fastwam'),
])
@pytest.mark.parametrize('override', [False, True])
def test_serve_selects_profile_and_explicit_interpreter(family_args, profile, override, monkeypatch, tmp_path):
    from piper_bench.deployment import config
    paths = {key: tmp_path / key for key in ('policy-ewm', 'policy-fastwam', 'override')}
    for path in paths.values():
        path.touch()
    monkeypatch.delenv('PIPER_POLICY_PYTHON', raising=False)
    if override:
        monkeypatch.setenv('PIPER_POLICY_PYTHON', str(paths['override']))
    monkeypatch.setattr(config, 'load_config', lambda: {'profiles': {
        key: {'python': str(paths[key])} for key in ('policy-ewm', 'policy-fastwam')}})
    expected = paths['override'] if override else paths[profile]
    calls = []

    def run(argv, *, env, check):
        assert argv[:5] == [str(expected), '-B', '-m', 'piper_policies', 'serve']
        assert argv[5:] == family_args + ['--help']
        bridge = Path(env['PYTHONPATH'].split(os.pathsep)[0])
        assert (bridge / 'piper_bench/__init__.py').is_file()
        calls.append(bridge)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, 'run', run)
    assert cli.main(['serve', *family_args, '--help']) == 0
    assert len(calls) == 1 and not calls[0].exists()


def test_unconfigured_serve_keeps_current_interpreter(monkeypatch):
    from piper_bench.deployment import config
    monkeypatch.delenv('PIPER_POLICY_PYTHON', raising=False)
    monkeypatch.setattr(config, 'load_config', lambda: {'profiles': {}})
    module = ModuleType('piper_policies.cli')
    calls = []
    module.main = lambda argv: calls.append(argv) or 0
    monkeypatch.setitem(sys.modules, 'piper_policies.cli', module)
    assert cli.main(['serve', '--family=fastwam', '--help']) == 0
    assert calls == [['serve', '--family=fastwam', '--help']]


def test_legacy_suite_forwards_policy_package_to_simulation_environment(monkeypatch, tmp_path):
    pytest.importorskip('piper_policies')
    from piper_policies import legacy_suite
    seen = []

    def run(command, *, check, env):
        assert command[:4] == [str(tmp_path / 'sim-python'), '-B', '-m', 'piper_policies.legacy_suite']
        bridge = Path(env['PYTHONPATH'].split(os.pathsep)[0])
        assert (bridge / 'piper_policies/__init__.py').is_file()
        assert (bridge / 'piper_bench/__init__.py').is_file()
        seen.append(bridge)
        return SimpleNamespace(returncode=6)

    monkeypatch.setattr(legacy_suite.subprocess, 'run', run)
    assert legacy_suite.main(['--checkpoint-dir', str(tmp_path / 'model'),
        '--model-python', sys.executable, '--sim-python', str(tmp_path / 'sim-python'),
        '--tasks', 'conveyor', '--conditions', 'id', '--output-dir', str(tmp_path / 'output')]) == 6
    assert len(seen) == 1 and not seen[0].exists()


def test_doctor_probe_forwards_only_package_and_removes_bridge(monkeypatch, tmp_path):
    from piper_bench.deployment import doctor
    seen = []

    def probe(python, environment, lock):
        paths = environment['PYTHONPATH'].split(os.pathsep)
        bridge = Path(paths[0])
        assert (bridge / 'piper_bench/__init__.py').is_file()
        assert all(Path(path).name != 'site-packages' for path in paths)
        seen.append(bridge)
        return {'status': 'passed'}

    monkeypatch.setattr(doctor, '_probe', probe)
    monkeypatch.setattr(doctor, 'verify', lambda *args: {'passed': True})
    monkeypatch.setattr(doctor, 'runtime_environment',
                        lambda cfg: {'PYTHONPATH': str(tmp_path / 'site-packages')})
    report = doctor.diagnose(python=sys.executable, output=tmp_path / 'report.json',
                            config={'prefix': str(tmp_path), 'assets_dir': str(tmp_path / 'assets')})
    assert report['passed'] and not report['deployment_accepted']
    assert len(seen) == 1 and not seen[0].exists()


def test_real_venv_bridge_keeps_numpy_in_the_other_python_environment(tmp_path):
    other = os.environ.get('PIPER_TEST_OTHER_PYTHON')
    if not other:
        pytest.skip('set PIPER_TEST_OTHER_PYTHON for the real different-minor interpreter check')
    other_version = subprocess.check_output([other, '-B', '-c', 'import sys;print(sys.version_info[:2])'], text=True).strip()
    if other_version == str(sys.version_info[:2]):
        pytest.skip('the integration check requires a different Python minor version')
    import numpy
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    venv = tmp_path / 'host-venv'
    subprocess.run([sys.executable, '-B', '-m', 'venv', '--system-site-packages', '--without-pip', str(venv)],
                   check=True, env=env, capture_output=True, timeout=60)
    python = venv / 'bin/python'
    site = Path(subprocess.check_output([str(python), '-B', '-c',
                'import sysconfig;print(sysconfig.get_path("purelib"))'], env=env, text=True).strip())
    shutil.copytree(Path(piper_bench.__file__).parent, site / 'piper_bench',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (site / 'numpy').symlink_to(Path(numpy.__file__).resolve().parent, target_is_directory=True)
    explicit = tmp_path / 'user-source'
    explicit.mkdir()
    (explicit / 'user_marker.py').write_text('VALUE = 42\n')
    code = r'''
import json, os, subprocess, sys
from pathlib import Path
import piper_bench
from piper_bench.deployment.process import package_environment
site = Path(piper_bench.__file__).parent.parent
assert site.name == 'site-packages'
assert str(site).startswith(sys.prefix)
child_code = "import json,numpy,piper_bench,user_marker;print(json.dumps({'numpy':numpy.__file__,'package':piper_bench.__file__,'marker':user_marker.VALUE}))"
bad = subprocess.run([sys.argv[1], '-B', '-c', child_code], env=dict(os.environ, PYTHONPATH=str(site)), text=True, capture_output=True)
assert bad.returncode != 0 and 'numpy' in bad.stderr
with package_environment() as env:
    bridge = Path(env['PYTHONPATH'].split(os.pathsep)[0])
    assert str(site) not in env['PYTHONPATH'].split(os.pathsep)
    result = subprocess.run([sys.argv[1], '-B', '-c', child_code], env=env, text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    assert value['marker'] == 42
    assert not Path(value['numpy']).is_relative_to(site)
    assert Path(value['package']).is_relative_to(bridge)
assert not bridge.exists() and (site / 'piper_bench/__init__.py').is_file()
print(json.dumps(value))
'''
    result = subprocess.run([str(python), '-B', '-c', code, other], cwd=tmp_path,
                            env={**env, 'PYTHONPATH': os.pathsep.join([str(site), str(explicit)])},
                            text=True, capture_output=True, check=True, timeout=60)
    assert json.loads(result.stdout)['marker'] == 42
