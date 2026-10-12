"""RGB rollout diagnostic and image renderer; training/evaluators are unchanged.

Run with ``python tests/test_rgb_generation.py --help`` from OpenWAM.
Generation runs only through the command-line entry point, not pytest collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'OpenWAM'))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from openwam.dataloader.ewm_normalizer import EWMNormalizer
from openwam.dataloader.registry import build_dataset
from openwam.model.architectures.ewm import EWMModel
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def sample_zero_start(model, batch, normalizer, horizon, sampling_steps):
    """Same reverse Euler/KV-cache path, but both initial flow states are zero.

    This is a nonstandard diagnostic, not a draw from the Gaussian prior.
    Neither future ground truth nor ground-truth actions are accepted here.
    """
    model.eval()
    observed = model.event_codec.encode(batch['obs_rgb'][:, None])[:, 0]
    context = model._context(batch, None)
    kv = model.prepare_context_kv(context)
    video = torch.zeros((observed.shape[0], horizon // 4, *observed.shape[1:]),
                        device=observed.device, dtype=torch.float32)
    action = torch.zeros((observed.shape[0], horizon, 7), device=observed.device)
    scheduler = CosmosFlowSchedulerAdapter(shift_video=1.0)
    scheduler.set_timesteps(sampling_steps)
    for sigma, timestep in zip(scheduler.sigmas, scheduler.timesteps):
        velocity, action_velocity = model.predict_velocity(
            observed, batch['proprio'], video, action, sigma.to(video).expand(video.shape[0]),
            context, context_kv=kv)
        video = scheduler.step(velocity, timestep, video)
        action = scheduler.step(action_velocity, timestep, action)
    action = action.clamp(-1, 1)
    return {'future_rgb': model._decode_future(observed, video),
            'action_normalized': action, 'action': normalizer.denormalize_action(action)}


def rgb_metrics(prediction, target, current):
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError('invalid RGB prediction')
    error = (prediction.float() - target.float()).square()
    mse = error.mean()
    # GT-derived motion ROI is used only for scoring, never for sampling.
    mask = (target - current[None]).abs().amax(dim=2) > 0.05
    roi = error.permute(0, 1, 3, 4, 2)[mask]
    return {'mse': float(mse), 'mae': float((prediction - target).abs().mean()),
            'psnr': float(-10 * torch.log10(mse.clamp_min(1e-12))),
            'motion_roi_mse': float(roi.mean()) if roi.numel() else None,
            'motion_roi_fraction': float(mask.float().mean()),
            'per_camera_psnr': [float(-10 * torch.log10(error[:, c].mean().clamp_min(1e-12)))
                                for c in range(target.shape[1])],
            'per_frame_psnr': [float(-10 * torch.log10(error[t].mean().clamp_min(1e-12)))
                               for t in range(target.shape[0])]}


def pil_frame(tensor):
    return Image.fromarray((tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                            * 255).round().astype(np.uint8))


def render_window(output, index, camera_names, current, pictures, scores):
    frames = (0, 7, 15, 23)
    for camera, name in enumerate(camera_names):
        canvas = Image.new('RGB', (1024, 160 + 280 * len(pictures)), '#f3f3f3')
        draw = ImageDraw.Draw(canvas)
        canvas.paste(pil_frame(current[camera]).resize((128, 128)), (0, 24))
        draw.text((4, 4), f'CURRENT {name} | dataset index={index}', fill='black')
        draw.text((145, 35), '20k RGB-action | 20 Euler steps | future +1,+8,+16,+24', fill='black')
        draw.text((145, 57), 'Gaussian and ZERO start use current RGB/state/text ONLY.', fill='black')
        draw.text((145, 79), 'Clean VAE uses true future: reconstruction reference, NOT forecast.', fill='black')
        for row, (method, video) in enumerate(pictures.items()):
            label = method
            if method in scores:
                label += f" | full-clip {name} PSNR={scores[method]['per_camera_psnr'][camera]:.2f} dB"
            draw.text((5, 160 + row * 280 + 4), label, fill='black')
            for column, frame in enumerate(frames):
                canvas.paste(pil_frame(video[frame, camera]), (column * 256, 184 + row * 280))
        canvas.save(output / f'comparison_{index}_{name}.png')

    # Compact preview keeps both views, exact pixels, and the primary Gaussian draw.
    canvas = Image.new('RGB', (1024, 160 + 4 * 280), '#f3f3f3')
    draw = ImageDraw.Draw(canvas)
    for camera, name in enumerate(camera_names):
        canvas.paste(pil_frame(current[camera]).resize((128, 128)), (camera * 128, 24))
    draw.text((4, 4), f'CURRENT cam0 + wrist | index={index} | checkpoint20k', fill='black')
    draw.text((280, 35), '20-step PURE GAUSSIAN generation, seed1234 + index', fill='black')
    draw.text((280, 60), 'Columns: future +1, +8, +16, +24 (30 Hz)', fill='black')
    for row, (camera, method) in enumerate(((0, 'GT future'), (0, 'Gaussian seed1234'),
                                          (1, 'GT future'), (1, 'Gaussian seed1234'))):
        label = f'{camera_names[camera]} | {method}'
        if method in scores:
            label += f" | PSNR={scores[method]['per_camera_psnr'][camera]:.2f} dB"
        draw.text((5, 164 + row * 280), label, fill='black')
        for column, frame in enumerate(frames):
            canvas.paste(pil_frame(pictures[method][frame, camera]), (column * 256, 184 + row * 280))
    canvas.save(output / f'preview_{index}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=20000)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = args.checkpoint_dir.resolve()
    checkpoint = directory / f'checkpoint_step_{args.step:06d}.pt'
    cfg = OmegaConf.load(directory / 'config.yaml')
    assert cfg.model.training_mode == 'rgb_action' and cfg.model.event_camera_mode == 'dual'
    reference = json.loads(args.reference_manifest.read_text())
    indices = [item['index'] for item in reference['ids']]
    seeds, sampling_steps = reference['seeds'], reference['sampling_steps']
    assert seeds == [1234, 5678] and sampling_steps == 20
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    cfg.dataloader.normalization_stats_path = str(directory / 'normalization.json')
    cfg.dataloader.event_frame_cache_dir = None
    cfg.dataloader.gpu_voxelization = False
    dataset = build_dataset(cfg.dataloader, split='train')
    assert dataset.observation_row_offset == 0 and dataset.chunk_size == 24
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['format'] == 'ewm_rgb_action_latent_v1' and payload['step'] == args.step
    assert payload['horizon'] == 24 and payload['model_config']['training_mode'] == 'rgb_action'
    model = EWMModel.from_config(payload['model_config'])
    model.load_policy_state_dict(payload['model'])
    del payload
    model.cuda().eval()
    normalizer = EWMNormalizer(directory / 'normalization.json')
    assert dataset.normalizer.metadata == normalizer.metadata
    mapping = []
    for index, expected in zip(indices, reference['ids']):
        local = int(np.searchsorted(dataset._cumulative, index, side='right') - 1)
        anchor = int(dataset.first_anchor + (index - dataset._cumulative[local]) * dataset.window_stride)
        record = {'index': index, 'episode': int(dataset.episodes.iloc[local]['episode_index']), 'anchor': anchor}
        assert record == expected, (record, expected)
        mapping.append(record)
    manifest = {'checkpoint_step': args.step, 'checkpoint_sha256': file_hash(checkpoint),
                'config_sha256': file_hash(directory / 'config.yaml'),
                'normalization_sha256': file_hash(directory / 'normalization.json'),
                'checkpoint_dir': str(directory), 'reference_manifest': str(args.reference_manifest),
                'ids': mapping, 'seeds': seeds, 'sampling_steps': sampling_steps,
                'sampling_method': 'reverse Euler, shift=1, sigma 1 to 0',
                'input_keys': ['obs_video', 'obs_rgb', 'proprio', 'prompt'],
                'future_ground_truth_used_for_generation': False,
                'dataset_split': 'training-distribution windows, not held-out',
                'dtype': 'BF16 autocast; FP32 flow state', 'device': torch.cuda.get_device_name(),
                'zero_start': 'nonstandard all-zero video AND action initial flow states',
                'motion_roi': 'scoring only: max RGB abs(target-current)>0.05',
                'visualization_frames': [1, 8, 16, 24], 'camera_names': ['cam0', 'wrist']}
    save_json(output / 'manifest.json', manifest)
    rows = []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            batch = {key: sample[key][None].cuda() for key in ('obs_video', 'obs_rgb', 'proprio')}
            batch['prompt'] = [sample['prompt']]
            assert set(batch) == set(manifest['input_keys'])
            target = sample['future_rgb'].cuda().float() / 255
            current = sample['obs_rgb'].cuda().float() / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):
                full = torch.cat((batch['obs_rgb'][:, None], sample['future_rgb'][None].cuda()), 1)
                oracle = model.event_codec.decode(model.event_codec.encode(full))[0, 1:]
            pictures = {'GT future': target.cpu()}
            scores = {}
            def record(method, prediction, result=None, seed=None):
                metrics = rgb_metrics(prediction, target, current)
                row = {'method': method, 'index': index, 'seed': seed, **metrics}
                if result is not None:
                    row['action_mse'] = float((result['action_normalized'][0] - sample['action'].cuda()).square().mean())
                rows.append(row)
                pictures[method] = prediction.cpu()
                scores[method] = metrics
                print(json.dumps({k: v for k, v in row.items() if not k.startswith('per_')}), flush=True)
            record('Persistence current RGB', current[None].expand_as(target))
            record('Clean VAE (oracle)', oracle)
            for seed in seeds:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model.sample_policy(batch, horizon=24, normalizer=normalizer,
                        num_steps=sampling_steps, generator=torch.Generator(device='cuda').manual_seed(seed + index))
                record(f'Gaussian seed{seed}', result['future_rgb'][0], result, seed)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = sample_zero_start(model, batch, normalizer, 24, sampling_steps)
            record('ZERO start (nonstandard)', result['future_rgb'][0], result)
            render_window(output, index, ['cam0', 'wrist'], current.cpu(), pictures, scores)
            save_json(output / 'rows.json', rows)
    summary = {}
    for method in dict.fromkeys(row['method'] for row in rows):
        selected = [row for row in rows if row['method'] == method]
        summary[method] = {key: statistics.mean(row[key] for row in selected if row[key] is not None)
                           for key in ('mse', 'mae', 'psnr', 'motion_roi_mse', 'motion_roi_fraction')
                           if any(row[key] is not None for row in selected)}
    save_json(output / 'summary.json', summary)
    save_json(output / 'complete.json', {'complete': True, 'rows': len(rows), 'windows': len(indices)})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
