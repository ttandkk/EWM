#!/usr/bin/env python3
"""Train/test episode-split PCA of matched frozen VAE future-block latents."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

TASKS = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube', 'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')
KS = (8, 16, 32, 64)


def evaluate(x, train_idx, test_idx):
    train = x[train_idx].astype(np.float64)
    test = x[test_idx].astype(np.float64)
    mu = train.mean(axis=0)
    train -= mu
    test -= mu
    gram = train @ train.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    numerical_rank = int(np.sum(vals > vals[0] * 1e-12))
    vals = vals[:numerical_rank]
    vecs = vecs[:, :numerical_rank]
    basis = (vecs.T @ train) / np.sqrt(vals[:, None])
    energy = float(np.sum(train * train))
    test_energy = float(np.sum(test * test))
    if energy <= 0 or test_energy <= 0:
        raise ValueError('zero input energy')
    test_scores = test @ basis.T
    explained = np.cumsum(vals) / energy
    heldout = np.cumsum(np.sum(test_scores * test_scores, axis=0)) / test_energy
    components = {}
    for k in KS:
        if k > numerical_rank:
            components[str(k)] = None
        else:
            components[str(k)] = {'train_explained_variance': float(explained[k-1]),
                                  'test_retained_energy': float(heldout[k-1]),
                                  'test_relative_reconstruction_error': float(1.0-heldout[k-1])}
    def k95(v):
        ix = np.flatnonzero(v >= .95)
        return int(ix[0]+1) if len(ix) else None
    return {'numerical_rank': numerical_rank, 'values': components,
            'train_k95': k95(explained), 'test_k95': k95(heldout),
            'train_explained_curve': explained.tolist(),
            'test_retained_curve': heldout.tolist(),
            'max_basis_orthogonality_error': float(np.max(np.abs(basis @ basis.T - np.eye(numerical_rank))))}, mu, basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extra-dir', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = {'protocol': {'train_episodes': 'first 40 sorted episode IDs', 'test_episodes': 'last 10 sorted episode IDs',
                        'input': 'paired current+3 future blocks; retain future blocks 1:4, flatten to 49152',
                        'metrics': 'PCA fit only on train. Train explained variance = sum first K eigenvalues / total training variance. Test retained energy = 1 - ||X_test-mu-truncated_projection||^2 / ||X_test-mu||^2, using training mean; test samples are not used to choose the mean or PCA directions.',
                        'ks': KS, 'variance': 'sample squared deviations; ratios use same normalization',
                        'overlap': 'windows from same episode always in same split', 'train_samples_limit': 'rank at most number of training windows - 1'}, 'tasks': {}}
    for task in args.tasks:
        t = time.monotonic()
        with np.load(args.input_dir / f'{task}_latents.npz', allow_pickle=False) as d:
            arrays = {k: d[k].copy() for k in ('event', 'rgb', 'episode', 'anchor')}
        if args.extra_dir and (args.extra_dir / f'{task}_latents.npz').exists():
            with np.load(args.extra_dir / f'{task}_latents.npz', allow_pickle=False) as d:
                for key in arrays:
                    arrays[key] = np.concatenate([arrays[key], d[key]], axis=0)
        ep = arrays['episode']
        ids = np.unique(ep)
        if len(ids) != 50:
            raise ValueError(f'{task}: unexpected number of episodes: {len(ids)}')
        for eid in ids:
            anchors = arrays['anchor'][ep == eid]
            if len(anchors) != len(np.unique(anchors)):
                raise ValueError(f'{task}: duplicate anchors in episode {eid}')
        tr = np.flatnonzero(np.isin(ep, ids[:40]))
        te = np.flatnonzero(np.isin(ep, ids[40:]))
        task_out = {'train_ids': ids[:40].tolist(), 'test_ids': ids[40:].tolist(),
                    'train_windows': len(tr), 'test_windows': len(te), 'modalities': {}}
        save = {'train_indices':tr, 'test_indices':te, 'episode':ep, 'anchor':arrays['anchor']}
        for mod in ('event', 'rgb'):
            raw = arrays[mod]
            if raw.shape != (len(ep),4,16,32,32) or not np.isfinite(raw).all():
                raise ValueError(f'{task}: wrong latent shape for {mod}: {raw.shape}')
            x = raw[:,1:].reshape(len(ep), -1)
            metric, mu, basis = evaluate(x, tr, te)
            task_out['modalities'][mod] = metric
            if task == 'tennis_falling':
                save[f'{mod}_mean'] = mu.astype(np.float32)
                save[f'{mod}_basis'] = basis.astype(np.float32)
        if task == 'tennis_falling':
            pending = args.output_dir / 'tennis_falling_pca.pending.npz'
            np.savez_compressed(pending, **save)
            pending.replace(args.output_dir / 'tennis_falling_pca.npz')
        out['tasks'][task] = task_out
        pending_result = args.output_dir / 'results.pending.json'
        pending_result.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
        pending_result.replace(args.output_dir / 'results.json')
        print(f'{task}: train {len(tr)}, test {len(te)}, {time.monotonic()-t:.1f}s', flush=True)

if __name__ == '__main__':
    main()
