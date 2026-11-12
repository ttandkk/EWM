#!/usr/bin/env python3
"""Plot trajectory-disjoint PCA results for falling ball overall and per phase."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent/'data'
import os
OUT = Path(os.environ['EWM_PLOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
all_task=json.loads((root/'results.json').read_text())['tasks']['tennis_falling']['modalities']
phase=json.loads((root/'phase_results.json').read_text())['tasks']['tennis_falling']
groups=[('Combined windows',all_task),('Start only',phase['start']),('End only',phase['end'])]
metrics=[('Train explained variance (%)','train_explained_variance'),('Test reconstruction retention (%)','test_retained_energy')]
fig,axes=plt.subplots(2,3,figsize=(12.2,6.9),sharey=True,layout='constrained')
for row,(ylabel,key) in enumerate(metrics):
    for col,(name,values) in enumerate(groups):
        ax=axes[row,col]
        ks=[8,16,32,64] if col == 0 else [8,16,32]
        for mod,clr,marker in [('event','#1572b6','o'),('rgb','#d87622','s')]:
            vals=[100*values[mod]['values'][str(k)][key] for k in ks]
            ax.plot(ks,vals,label=mod.upper(),color=clr,marker=marker,lw=2)
            for x,y in zip(ks,vals):
                ax.annotate(f'{y:.1f}',(x,y),xytext=(0,6 if mod=='event' else -13),textcoords='offset points',ha='center',fontsize=8,color=clr)
        ax.set_xticks([8,16,32,64])
        ax.set_xlim(4,68)
        ax.set_ylim(0,105)
        ax.grid(axis='y',alpha=.25)
        if row==0:
            ax.set_title(name,weight='bold')
        if col==0:
            ax.set_ylabel(ylabel)
        if row==1:
            ax.set_xlabel('Number of PCA components K')
        if col>0:
            ax.text(.99,.04,'K=64 unavailable\n40 train windows',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#555')
axes[0,0].legend(loc='lower right',frameon=False)
fig.suptitle('Falling ball: PCA fit on 40 trajectories, tested on 10 held-out trajectories',weight='bold',fontsize=13)
fig.savefig(OUT/'falling_pca_reconstruction.png',dpi=180)
fig.savefig(OUT/'falling_pca_reconstruction.pdf')
print('saved falling_pca_reconstruction.png and .pdf')
