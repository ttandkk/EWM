#!/usr/bin/env python3
"""Render existing paired latent statistics, without re-encoding or resampling."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'data' / 'latent_expanded_20261104_results.json'
import os
OUT = Path(os.environ['EWM_PLOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
D = json.loads(SOURCE.read_text())['tasks']
TASKS = ['conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube', 'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container']
LABELS = ['Conveyor', 'Moving carton', 'Turntable cups', 'Rotating tube', 'Rolling ball', 'Falling ball', 'Parabolic ball', 'Ball in container']
assert set(TASKS) == set(D)
assert all(D[t]['groups']['all']['episodes'] == 50 and D[t]['groups']['all']['windows'] == 100 for t in TASKS)
assert all(D[t]['groups'][g]['episodes'] == 50 and D[t]['groups'][g]['windows'] == 50 for t in TASKS for g in ['start', 'end'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.labelcolor': '#243447', 'text.color': '#243447', 'xtick.color': '#536476', 'ytick.color': '#243447', 'axes.edgecolor': '#c9d2dc'})
EVENT, RGB = '#0072B2', '#D55E00'

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)

# Direct values: no projection and no normalization applied.
fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.0), gridspec_kw={'wspace': .24})
y = np.arange(len(TASKS))
for ax, metric, title, unit, xmax in zip(axes, ['variance', 'effective_rank'], ['A  Mean coordinate variance', 'B  PCA spectral effective rank'], ['Latent variance (population divisor N)', 'Effective rank = exp(entropy of PCA spectrum)'], [0.40, 69]):
    event = np.array([D[t]['groups']['all'][metric]['event'] for t in TASKS])
    rgb = np.array([D[t]['groups']['all'][metric]['rgb'] for t in TASKS])
    for i, (a, b) in enumerate(zip(event, rgb)):
        ax.plot([a, b], [i, i], color='#bcc8d3', lw=2.3, zorder=1)
        ax.scatter(a, i, s=77, color=EVENT, edgecolor='white', linewidth=.7, zorder=3)
        ax.scatter(b, i, s=77, color=RGB, edgecolor='white', linewidth=.7, zorder=3)
        fmt = '{:.3f}' if metric == 'variance' else '{:.1f}'
        gap = .007 if metric == 'variance' else 1.2
        ax.text(a-gap, i-.20, fmt.format(a), ha='right', va='center', color=EVENT, fontsize=9)
        ax.text(b+gap, i+.20, fmt.format(b), ha='left', va='center', color=RGB, fontsize=9)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(TASKS)-.5, -.7)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(unit, fontsize=9, labelpad=10)
    ax.xaxis.grid(True, color='#e8edf1', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
fig.suptitle('Event and RGB latent distributions: direct comparison', x=.075, ha='left', y=.98, fontsize=17, fontweight='bold')
fig.text(.075, .92, 'Eight tasks  |  50 trajectories per task  |  2 paired windows per trajectory  |  future latent blocks only', fontsize=10, color='#536476')
fig.legend(handles=[plt.Line2D([],[], marker='o', color=EVENT, ls='', label='Event'), plt.Line2D([],[], marker='o', color=RGB, ls='', label='RGB')], loc='upper right', bbox_to_anchor=(.98,.98), frameon=False, ncol=2)
fig.text(.075, .025, 'Each point is a statistic of 100 paired windows, not an individual trajectory. Variance depends on latent scale; effective rank depends on sample size.', fontsize=9, color='#536476')
fig.subplots_adjust(top=.81, bottom=.14, left=.17, right=.96)
save(fig, 'latent_metrics_all_tasks')

# Phase decomposition: ratios are computed from the original unrounded values.
stages = ['all', 'start', 'end']
stage_labels = ['Combined\n100 windows', 'Start\n50 windows', 'End\n50 windows']
metrics = ['variance', 'effective_rank']
fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.2), gridspec_kw={'wspace': .24})
cmap = LinearSegmentedColormap.from_list('ratio', ['#2764a5', '#77aad1', '#edf3f6', '#f0a66a', '#bc492c'])
norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=1.2)
for ax, metric, title in zip(axes, metrics, ['A  Variance ratio: Event / RGB', 'B  Effective-rank ratio: Event / RGB']):
    vals = np.array([[D[t]['groups'][s][metric]['event_over_rgb'] for s in stages] for t in TASKS])
    assert np.all(np.isfinite(vals))
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(3), stage_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_yticks(range(8), LABELS)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    for r in range(8):
        for c in range(3):
            v = vals[r,c]
            ax.text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=10, fontweight='bold', color='white' if v < .48 or v > 1.12 else '#172c3d')
    if metric == 'effective_rank':
        assert vals[0,2] > 1 and (vals > 1).sum() == 1
        ax.add_patch(Rectangle((1.515,-.485), .97, .97, fill=False, edgecolor='#a82d19', lw=2.7, zorder=10))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=50)
fig.suptitle('Does the pattern hold within each trajectory stage?', x=.075, ha='left', y=.98, fontsize=17, fontweight='bold')
fig.text(.075, .925, 'Ratios below 1 mean Event < RGB. The highlighted cell is the only effective-rank reversal.', fontsize=10, color='#536476')
fig.text(.075, .075, 'Conveyor / end: effective rank Event 47.54 > RGB 45.37  (ratio 1.048).', fontsize=10, color='#a63820', fontweight='bold')
fig.text(.075, .040, 'Combined uses 2 correlated windows per trajectory; start/end use 1. Effective ranks have sample-dependent upper bounds 99 / 49.', fontsize=9, color='#536476')
fig.subplots_adjust(top=.74, bottom=.15, left=.17, right=.96)
# A discrete legend clarifies the threshold, without implying uncertainty or a truncated scale.
fig.legend(handles=[Patch(facecolor='#5590be', label='Ratio < 1: Event lower'), Patch(facecolor='#e88751', label='Ratio > 1: Event higher')], loc='upper right', bbox_to_anchor=(.965,.925), frameon=False, ncol=2, fontsize=9)
save(fig, 'latent_metrics_by_stage')
print('Generated 2 PNG and 2 SVG figures from', SOURCE)
