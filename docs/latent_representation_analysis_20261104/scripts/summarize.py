#!/usr/bin/env python3
"""Summarize window-macro and stage-macro decoder quality."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
import os
OUT = Path(os.environ['EWM_PLOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
out = OUT
r = json.loads((HERE/'data'/'decoder_results.json').read_text())
rows = r['metrics']
episodes = r['evaluated_test_episodes']
first = {ep: min(row['anchor'] for row in rows if row['episode'] == ep) for ep in episodes}
groups = {'all': rows, 'start': [row for row in rows if row['anchor'] == first[row['episode']]],
          'end': [row for row in rows if row['anchor'] != first[row['episode']]]}
summary = {}
for group, group_rows in groups.items():
    summary[group] = {}
    for modality in ('event','rgb'):
        summary[group][modality] = {}
        for k in [None]+r['ks']:
            selected = [row['metrics'] for row in group_rows if row['modality'] == modality and row['k'] == k]
            assert selected
            keys = ['input_mse', 'input_mae', 'decoder_delta_mse']
            if modality == 'event': keys += ['active_fraction', 'active_mse', 'precision', 'recall']
            vals = {key: float(np.mean([m[key] for m in selected])) for key in keys}
            if modality == 'event':
                vals['f1'] = float(np.mean([2*m['tp']/(2*m['tp']+m['fp']+m['fn']) if 2*m['tp']+m['fp']+m['fn'] else 0 for m in selected]))
            vals['n_windows'] = len(selected)
            summary[group][modality]['full' if k is None else str(k)] = vals
(out/'decoder_summary.json').write_text(json.dumps(summary, indent=2))

fig, axes = plt.subplots(1,3,figsize=(14,4.2))
specs = [('event','active_mse','Event: active-pixel MSE ↓'),
         ('event','f1','Event: activity F1 ↑'), ('rgb','input_mse','RGB: pixel MSE ↓')]
for ax, (mod, metric, title) in zip(axes,specs):
    for group, color in [('all','#343434'),('start','#2875a8'),('end','#db7c26')]:
        vals = summary[group][mod]
        x = r['ks']
        ax.plot(x,[vals[str(k)][metric] for k in x],marker='o',label=group,color=color)
        ax.axhline(vals['full'][metric],linestyle='--',color=color,alpha=.5)
    ax.set_title(title)
    ax.set_xlabel('PCA components K')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Mean over windows')
axes[-1].legend(title='solid: PCA; dashed: no compression',fontsize=8)
fig.suptitle('Held-out falling-ball trajectories, 4 trajectories / 8 windows')
fig.tight_layout()
fig.savefig(out/'decoder_quality.png',dpi=180)
for path in (HERE/'figures').glob('event_example_ep*.png'):
    img = np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8).copy()
    # Event values are count/15 and visually dark. Brighten every panel by the
    # same factor for presentation only; numerical metrics use original values.
    img[24:] = np.minimum(img[24:].astype(np.uint16)*4,255).astype(np.uint8)
    Image.fromarray(img).save(out/'event_example_brightness_x4.png')
print(json.dumps(summary,indent=2))
