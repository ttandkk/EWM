#!/usr/bin/python3
"""Standalone falling figure and independent SVD check of PCA."""
from pathlib import Path
import numpy as np
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualize import ROOT, OUT, gram_pca, dots
with np.load(ROOT/'tennis_falling_latents.npz',allow_pickle=False) as z:
    ep=z['episode']; anchor=z['anchor']
    arrays={m:z[m][:,1:].reshape(len(ep),-1).astype(np.float64) for m in ('event','rgb')}
assert len(ep)==100 and len(np.unique(ep))==50
scores={}; ratios={}
for mod,x in arrays.items():
    scores[mod],ratios[mod],_=gram_pca(x)
    x-=x.mean(axis=0)
    u,s,v=svd(x,full_matrices=False,lapack_driver='gesdd')
    exp=s[:2]**2/np.sum(s**2)
    np.testing.assert_allclose(ratios[mod],exp,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.abs(np.sum(scores[mod]*(u[:,:2]*s[:2]),axis=0))/np.sum(scores[mod]**2,axis=0),1,rtol=1e-8)
    print(mod,'independent SVD explained variance',exp,'passed',flush=True)
lim=max(np.abs(np.concatenate(list(scores.values()))).max()*1.08,1e-3)
fig,axes=plt.subplots(1,2,figsize=(11,5.5),constrained_layout=True)
for ax,mod in zip(axes,('event','rgb')):
    dots(ax,scores[mod],anchor); ax.set(xlim=(-lim,lim),ylim=(-lim,lim))
    ax.set_title(f'{mod.upper()} | independent PCA')
    ax.set_xlabel(f'PC1 ({100*ratios[mod][0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100*ratios[mod][1]:.1f}%)')
axes[0].legend(loc='upper right')
fig.suptitle('Tennis falling: 50 trajectories × 2 windows, future latents\nAxes fit separately; matching plot scale, but PCA directions differ')
fig.savefig(OUT/'pca_tennis_falling.png',dpi=180)
