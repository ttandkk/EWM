"""Explicit zero-training native OpenWAM export; NOT a Piper-trained policy."""
import argparse
import json
import os
from pathlib import Path
import shutil


def export_normalization(source, destination):
    """Translate the legacy Piper state key without changing a statistic."""
    import numpy as np
    from piper_policies.metadata import file_sha256
    source, destination = Path(source), Path(destination)
    raw = np.load(source, allow_pickle=True).item()
    if 'eef' not in raw or not isinstance(raw['eef'], dict):
        raise ValueError('Piper normalization requires an eef action block')
    state_key = 'eef_state' if 'eef_state' in raw else 'state'
    if state_key not in raw or not isinstance(raw[state_key], dict):
        raise ValueError('Piper normalization requires saved 8D state statistics')
    for name, expected in (('eef', 7), (state_key, 8)):
        for key, value in raw[name].items():
            value = np.asarray(value)
            if value.shape != (expected,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid {name}.{key} statistics')
    converted = dict(raw)
    if state_key == 'state':
        converted['eef_state'] = converted.pop('state')
    np.save(destination, converted)
    restored = np.load(destination, allow_pickle=True).item()
    for name, output_name in (('eef', 'eef'), (state_key, 'eef_state')):
        if set(raw[name]) != set(restored[output_name]):
            raise ValueError('Normalization statistic keys changed')
        for key, value in raw[name].items():
            original, exported = np.asarray(value), np.asarray(restored[output_name][key])
            if original.shape != exported.shape or original.dtype != exported.dtype or not np.array_equal(original, exported):
                raise ValueError('Normalization statistic changed during export')
    return dict(input_sha256=file_sha256(source), output_sha256=file_sha256(destination),
                mapping={state_key: 'eef_state'}, exact_shapes_dtypes_and_values_verified=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--normalization', type=Path, required=True)
    p.add_argument('--pretrained-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute-initialization', action='store_true')
    p.add_argument('--probe-legacy-normalization', action='store_true',
                   help='serve the original legacy stats to exercise the public adapter; preserve canonical audit separately')
    args = p.parse_args()
    if not args.execute_initialization:
        print(json.dumps(dict(status='not_run', purpose=__doc__, estimated_output_gb=25,
                              planned_host_memory_gib=66, optimizer_updates=0)))
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='true')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from omegaconf import OmegaConf
    from openwam.model import build_architecture, resolve_architecture_config
    from piper_policies.resources import ResourceResolver
    from piper_policies.metadata import file_sha256
    cfg = ResourceResolver(pretrained_root=args.pretrained_root).config(OmegaConf.load(args.config), 'openwam')
    if cfg.model.architecture.framework != 'dual_system' or cfg.model.architecture.variant != 'joint_self_attn':
        raise ValueError('Only the reviewed native Piper joint-self-attention preset is supported')
    if cfg.model.video_backbone.get('from_scratch', False):
        raise ValueError('This export must load the existing pretrained Wan backbone')
    torch.manual_seed(42)
    resolved = resolve_architecture_config(cfg.model)
    params = OmegaConf.to_container(OmegaConf.create(resolved.params), resolve=True)
    params['video_backbone']['_source'] = str(cfg.model.video_backbone.model_path)
    model = build_architecture(resolved.registry_name, params)
    model.set_dtype_device(torch.bfloat16, torch.device('cpu'))
    model.eval()
    model.save_assets_for_deployment(str(args.output), cfg)
    OmegaConf.save(cfg, args.output / 'config.yaml')
    shutil.copyfile(args.normalization, args.output / 'normalization_source.npy')
    canonical = args.output / ('normalization_canonical_audit.npy' if args.probe_legacy_normalization else 'normalization_stats.npy')
    normalization_conversion = export_normalization(args.normalization, canonical)
    if args.probe_legacy_normalization:
        shutil.copyfile(args.normalization, args.output / 'normalization_stats.npy')
    checkpoint = args.output / 'checkpoint_step_000000.safetensors'
    model.save_checkpoint(str(checkpoint))
    (args.output / 'initialization_provenance.json').write_text(json.dumps(dict(
        evidence_label='native_pretrained_backbone_with_untrained_Piper_action_and_proprio',
        optimizer_updates=0, seed=42, pretrained_content_verified=False, normalization_conversion=normalization_conversion,
        served_normalization_schema='legacy_state' if args.probe_legacy_normalization else 'canonical_eef_state',
        input_config_sha256=file_sha256(args.config), input_normalization_sha256=file_sha256(args.normalization),
        checkpoint_sha256=file_sha256(checkpoint)), indent=2) + '\n')


if __name__ == '__main__':
    main()
