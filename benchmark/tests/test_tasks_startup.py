"""Exercise specialized script startup without creating a simulator."""
import builtins
import os
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('module', [
    'moving_carton.preview', 'turntable_cups.preview',
    'rotating_tube.preview', 'rotating_tube.physics',
])
def test_specialized_entrypoint_applies_config_before_simulator_import(module, monkeypatch, tmp_path):
    from piper_bench.backend import isaac

    class ReachedSimulatorConstruction(Exception):
        pass

    class FakeLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device', default='cpu')

        def __init__(self, args):
            assert args.output_dir == tmp_path / 'unused'
            assert args.trials == 1
            raise ReachedSimulatorConstruction

    monkeypatch.setenv('PIPER_TEST_STARTUP_READY', 'no')
    monkeypatch.setattr(isaac, 'runtime_environment', lambda: {'PIPER_TEST_STARTUP_READY': 'yes'})
    original = builtins.__import__
    imported = []

    def importing(name, *args, **kwargs):
        if name == 'isaaclab.app':
            assert os.environ['PIPER_TEST_STARTUP_READY'] == 'yes'
            imported.append(name)
            return SimpleNamespace(AppLauncher=FakeLauncher)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', importing)
    monkeypatch.setattr(sys, 'argv', [module, '--output-dir', str(tmp_path / 'unused'), '--trials', '1'])
    with pytest.raises(ReachedSimulatorConstruction):
        runpy.run_module('piper_bench.tasks.' + module, run_name='__main__')
    assert imported
    assert not (tmp_path / 'unused').exists()
