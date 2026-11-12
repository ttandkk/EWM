from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
base=Path(os.environ.get('EWM_PILOT_LATENT_INPUT', '/tmp/ewm-event-rgb-study'))
outdir=Path(os.environ['EWM_PLOT_OUTPUT'])
outdir.mkdir(parents=True, exist_ok=False)
tasks=['conveyor','moving_carton','turntable_cups','rotating_tube','tennis_rolling','tennis_falling','tennis_parabolic','tennis_container']
fig,axes=plt.subplots(len(tasks),2,figsize=(11,24),constrained_layout=True)
for row,task in enumerate(tasks):
 d=np.load(base/(task+'_latents.npz'))
 ep=d['episode']; train=np.isin(ep,[0,10,20]);test=~train
 for col,name in enumerate(['event','rgb']):
  z=d[name][:,1:].reshape(len(ep),-1).astype(np.float64)
  mu=z[train].mean(axis=0);x=z-mu
  _,s,vt=np.linalg.svd(x[train],full_matrices=False); proj=x@vt[:2].T
  frac=s[:2]**2 / np.sum(s**2)
  ax=axes[row,col];ax.scatter(proj[train,0],proj[train,1],marker='o',color='#2377b9',label='train',s=22)
  ax.scatter(proj[test,0],proj[test,1],marker='x',color='#dd6734',label='held-out',s=32)
  for i,e in enumerate(ep):ax.annotate(str(e),proj[i],xytext=(3,2),textcoords='offset points',fontsize=6)
  ax.axhline(0,color='#bbbbbb',linewidth=.5);ax.axvline(0,color='#bbbbbb',linewidth=.5)
  ax.set_title(f'{task} / {name}  PC1+2={sum(frac)*100:.1f}%')
  ax.set_xlabel('PC1, train-only fit');ax.set_ylabel('PC2, train-only fit')
  if row==0:ax.legend(fontsize=7)
fig.suptitle('LeanVAE future latent PCA; 3 train + 2 held-out episodes/task, 2 anchors/episode\nEach pane has an independent PCA basis and scale; clustering is not learnability',fontsize=13)
out=outdir/'latent_pca.png';fig.savefig(out,dpi=150);print(out)
