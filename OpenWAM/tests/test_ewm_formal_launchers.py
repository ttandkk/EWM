"""Archived formal launch recipes: config preservation and no-launch checks."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / 'scripts/train_ewm'


def helper_module():
    path = LAUNCHERS / 'train_ewm_action_only.py'
    spec = importlib.util.spec_from_file_location('formal_action_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_only_preserves_previous_formal_config():
    module = helper_module()
    cfg, changes = module.prepare('991235')
    previous = OmegaConf.load(ROOT / 'outputs/ewm_action_only_d768_a512_ada2_80k/job180957/config.yaml')
    actual = OmegaConf.to_container(cfg, resolve=True)
    actual['training']['output_path'] = previous.training.output_path
    actual['project']['wandb']['run_name'] = previous.project.wandb.run_name
    assert actual == OmegaConf.to_container(previous, resolve=True)
    assert len(changes) == 7
    assert cfg.model.training_mode == 'action_only'
    assert cfg.training.max_steps == 80000 and cfg.training.batch_size == 32
    assert cfg.dataloader.dataset_dir.endswith('piper_stack_fronttight_v3_50/job177683')


def test_action_only_dry_run_has_no_writes(tmp_path):
    fake_id = f'dryrun_{tmp_path.name}'
    output = ROOT / f'outputs/ewm_action_only_d768_a512_ada2_80k/job{fake_id}'
    assert not output.exists()
    environment = dict(os.environ, DRY_RUN='true', SLURM_JOB_ID=fake_id)
    result = subprocess.run(['bash', str(LAUNCHERS / 'train_ewm_action_only.sh')],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    cfg = OmegaConf.create(result.stdout)
    assert cfg.model.training_mode == 'action_only'
    assert cfg.project.wandb.project == 'CVPR2027-EWM'
    assert cfg.training.compile_core is True  # matches the submitted ablation


def test_event_action_dispatches_original_recipe_without_training(tmp_path):
    # Intercept the wrapper's child bash; never enter the actual trainer.
    fake_bash = tmp_path / 'bash'
    fake_bash.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                         'print(json.dumps({"argv": sys.argv[1:], "mode": os.environ["TRAINING_MODE"], '
                         '"wandb": os.environ["WANDB_MODE"], "pure_noise": os.environ["PURE_NOISE_EVAL_ENABLED"]}))\n')
    fake_bash.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}')
    for key in ('TRAINING_MODE', 'WANDB_MODE', 'PURE_NOISE_EVAL_ENABLED'):
        environment.pop(key, None)
    result = subprocess.run(['/bin/bash', str(LAUNCHERS / 'train_ewm_event_action.sh')],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    assert dispatched['argv'] == ['/projects/hdd/myproject/CVPR2027/OpenWAM/scripts/train_ewm/train_ewm.sh']
    assert dispatched['mode'] == 'joint' and dispatched['wandb'] == 'online'
    assert dispatched['pure_noise'] == 'false'
    expected = [line for line in (LAUNCHERS / 'train_ewm.sh').read_text().splitlines() if line.startswith('#SBATCH')]
    actual = [line for line in (LAUNCHERS / 'train_ewm_event_action.sh').read_text().splitlines() if line.startswith('#SBATCH')]
    assert actual == expected


@pytest.mark.parametrize('mode', ['action_only', 'rgb_action', 'event_only'])
def test_event_entry_rejects_another_mode_before_launch(mode):
    result = subprocess.run(['bash', str(LAUNCHERS / 'train_ewm_event_action.sh')],
                            env=dict(os.environ, TRAINING_MODE=mode), capture_output=True, text=True)
    assert result.returncode == 2 and 'requires TRAINING_MODE=joint' in result.stderr


@pytest.mark.parametrize('name', ['train_ewm_event_action.sh', 'train_ewm_action_only.sh'])
def test_shell_syntax_and_no_tmp_runtime_dependency(name):
    launcher = LAUNCHERS / name
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    # Provenance comments may mention tmp; live commands must use promoted paths.
    commands = '\n'.join(line for line in launcher.read_text().splitlines() if not line.startswith('#'))
    assert '/tmp/ewm_action_only_formal/' not in commands
    assert '#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/' in launcher.read_text()
