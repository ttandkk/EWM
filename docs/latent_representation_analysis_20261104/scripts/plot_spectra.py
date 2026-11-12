#!/usr/bin/env python3
"""Visualize normalized sample spectra and stage variance decomposition."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('EWM_LATENT_INPUT', ROOT / 'tmp/latent_expanded_20261104'))
REFERENCE = HERE / 'data/latent_expanded_20261104_results.json'
OUT = Path(os.environ['EWM_PLOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)

def main():
    reference = json.loads(REFERENCE.read_text())['tasks']
    results = {}
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for ax, task, group, title in zip(axs,
            ['tennis_falling', 'conveyor'], ['all', 'end'],
            ['Falling ball — both windows', 'Conveyor — end window']):
        with np.load(DATA / f'{task}_latents.npz', allow_pickle=False) as ds:
            episode, anchor = ds['episode'], ds['anchor']
            ix = np.arange(len(episode)) if group == 'all' else np.array([
                np.flatnonzero(episode == ep)[np.argmax(anchor[episode == ep])]
                for ep in np.unique(episode)])
            assert len(ix) == (100 if group == 'all' else 50)
            group_out = {}
            for mod, color in [('event', '#0072B2'), ('rgb', '#D55E00')]:
                a = np.asarray(ds[mod][ix, 1:], dtype=np.float64).reshape(len(ix), -1)
                a -= a.mean(axis=0, keepdims=True)
                gram = a @ a.T
                eig = np.linalg.eigvalsh((gram + gram.T) / 2)[::-1].clip(min=0)
                p = eig / eig.sum()
                pp = p[p > 1e-12]
                erank = float(np.exp(-np.sum(pp * np.log(pp))))
                cum = np.cumsum(p)
                n90 = int(np.searchsorted(cum, .9) + 1)
                var = float(np.trace(gram) / (a.shape[0] * a.shape[1]))
                ref = reference[task]['groups'][group]
                assert np.isclose(erank, ref['effective_rank'][mod], rtol=1e-9, atol=1e-10)
                assert np.isclose(var, ref['variance'][mod], rtol=1e-9, atol=1e-10)
                ax.plot(np.arange(1, len(cum)+1), cum * 100, color=color, lw=2.5,
                        label=f'{mod.upper()}: effective rank {erank:.2f}; 90% at {n90} PCs')
                ax.scatter([n90], [cum[n90-1]*100], color=color, s=32, zorder=3)
                group_out[mod] = {'effective_rank': erank, 'variance': var,
                                  'n90': n90, 'cumulative': cum.tolist()}
            results[f'{task}_{group}'] = group_out
        ax.axhline(90, color='#777777', lw=1, ls=':')
        ax.set(title=f'{title}\n{len(ix)} paired windows, 50 trajectories',
               xlabel='Number of principal components', ylabel='Cumulative sample variance explained (%)',
               xlim=(1, len(ix)-1), ylim=(0, 102))
        ax.grid(True, alpha=.22)
        ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('Normalized sample PCA spectra: concentration within each matched group', fontsize=14, weight='bold')
    fig.savefig(OUT / 'spectral_curves.png', dpi=180)
    fig.savefig(OUT / 'spectral_curves.pdf')
    plt.close(fig)

    result = {}
    task = reference['tennis_falling']['groups']
    for mod in ('event', 'rgb'):
        total = task['all']['variance'][mod]
        within = (task['start']['variance'][mod] + task['end']['variance'][mod]) / 2
        between = total - within
        assert between >= 0
        result[mod] = {'total_variance': total, 'within': within, 'between': between,
                       'between_fraction': between / total}
    results['falling_variance_decomposition'] = result
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    positions = np.arange(2)
    for i, mod in enumerate(('event', 'rgb')):
        r = result[mod]
        ax.bar(positions[i], 100*r['within']/r['total_variance'], color='#8da0cb',
               label='Within-window-stage variation' if i == 0 else None)
        ax.bar(positions[i], 100*r['between_fraction'], bottom=100*r['within']/r['total_variance'],
               color='#fc8d62', label='Difference between stage means' if i == 0 else None)
        ax.text(positions[i], 100 - 50*r['between_fraction'], f"{100*r['between_fraction']:.1f}%",
                ha='center', va='center', fontsize=12, weight='bold')
    ax.set_xticks(positions, [f"Event latent\nvariance = {result['event']['total_variance']:.4f}", f"RGB latent\nvariance = {result['rgb']['total_variance']:.4f}"])
    ax.set(ylim=(0, 106), ylabel='Share of each modality’s pooled variance (%)',
           title='Falling ball: pooled variance by stage\n50 paired trajectories; start and end windows')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, fontsize=9)
    fig.savefig(OUT / 'falling_variance_decomposition.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'falling_variance_decomposition.pdf', bbox_inches='tight')
    plt.close(fig)
    (OUT / 'spectral_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: {m: {key: v for key, v in x.items() if key != 'cumulative'} for m, x in val.items()}
                      for k, val in results.items() if k != 'falling_variance_decomposition'}, indent=2))
    print(json.dumps(results['falling_variance_decomposition'], indent=2))

if __name__ == '__main__':
    main()
