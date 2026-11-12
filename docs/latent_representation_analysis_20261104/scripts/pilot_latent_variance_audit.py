#!/usr/bin/env python3
"""Descriptive paired latent statistics. Reads eight NPZs and writes JSON outside input dir."""
import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube',
         'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')
TRAIN_EPISODES = (0, 10, 20)


def stats(x):
    # Population variance, not an unbiased estimator. Center each coordinate across samples.
    centered = x - x.mean(axis=0, keepdims=True)
    mean_coordinate_variance = float(np.mean(centered * centered))
    # Gram eigenvalues are proportional to sample covariance's nonzero eigenvalues.
    spectrum = np.linalg.eigvalsh(centered @ centered.T).clip(min=0)
    spectrum /= spectrum.sum()
    p = spectrum[spectrum > 1e-12]
    return {
        'n_windows': int(len(x)),
        'dimension': int(x.shape[1]),
        'mean_coordinate_variance_population': mean_coordinate_variance,
        'trace_covariance_population': mean_coordinate_variance * x.shape[1],
        'sample_pca_entropy_effective_rank': float(np.exp(-np.sum(p * np.log(p)))),
        'sample_pca_rank_upper_bound': int(len(x) - 1),
        'global_elements_std_not_sample_variance': float(x.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parents[3]/'tmp/event_rgb_latent_pilot_20261104')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.resolve().is_relative_to(args.input_dir.resolve()):
        ap.error('output must be outside the input directory')
    out = {
        'method': 'Future latent blocks 1:4, flatten all 3x16x32x32 coordinates. Per coordinate center across the specified windows; population variance divides by n, then averages coordinates. Entropy effective rank = exp(-sum p*log p), p = normalized eigenvalues of centered sample Gram matrix. Train-baseline test MSE uses the mean of all 6 training windows. Phase refers to window start=0 versus last available start.',
        'input_dir': str(args.input_dir.resolve()),
        'train_episodes': list(TRAIN_EPISODES),
        'limitations': 'Five trajectories per task; two windows per trajectory are dependent; phase comparisons contain five windows each. This is descriptive for the same fixed encoder and representation scaling, not a general proof of compactness or predictability.',
        'tasks': {},
    }
    for task in TASKS:
        with np.load(args.input_dir / f'{task}_latents.npz', allow_pickle=False) as data:
            episode = data['episode'].copy()
            anchor = data['anchor'].copy()
            arrays = {m: data[m][:, 1:].reshape(len(episode), -1).astype(np.float64) for m in ('event', 'rgb')}
        train = np.isin(episode, TRAIN_EPISODES)
        test = ~train
        groups = {'all': np.ones(len(episode), dtype=bool), 'train': train, 'test': test,
                  'start': anchor == 0, 'end': anchor != 0}
        row = {'episodes': episode.tolist(), 'anchors': anchor.tolist(), 'modalities': {}}
        for modality, x in arrays.items():
            group_stats = {name: stats(x[mask]) for name, mask in groups.items()}
            row['modalities'][modality] = {
                'groups': group_stats,
                'test_mse_to_training_window_mean': float(np.mean((x[test] - x[train].mean(axis=0)) ** 2)),
            }
        row['event_to_rgb_mean_coordinate_variance_ratio'] = {
            group: row['modalities']['event']['groups'][group]['mean_coordinate_variance_population'] /
                   row['modalities']['rgb']['groups'][group]['mean_coordinate_variance_population']
            for group in groups
        }
        out['tasks'][task] = row
        ev = row['modalities']['event']['groups']['all']
        rgb = row['modalities']['rgb']['groups']['all']
        print(f"{task:18s} variance {ev['mean_coordinate_variance_population']:.6f} / {rgb['mean_coordinate_variance_population']:.6f}; effective rank {ev['sample_pca_entropy_effective_rank']:.3f} / {rgb['sample_pca_entropy_effective_rank']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f: f.write(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
