import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
p=Path(os.environ['EWM_PLOT_OUTPUT'])
p.mkdir(parents=True, exist_ok=False)
data=Path(__file__).resolve().parent.parent/'data'
d=json.loads((data/'pilot_ridge_probe_complete.json').read_text())['tasks']
names=list(d); event=np.array([d[t]['event']['test_relative_mse'] for t in names]); rgb=np.array([d[t]['rgb']['test_relative_mse'] for t in names])
y=np.arange(len(names));fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
ax.barh(y-.18,event,.34,label='event latent',color='#d06d38');ax.barh(y+.18,rgb,.34,label='RGB latent',color='#397ab1')
ax.axvline(1,color='#555555',linestyle='--',linewidth=1,label='training-mean baseline')
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('Test MSE / train-mean baseline MSE (lower is better relative to each target)')
ax.set_title('Exploratory paired ridge probe: common current RGB latent, cam0\n3 train + 2 test trajectories per task; 2 windows per trajectory')
ax.legend(loc='lower right')
for i,(e,r) in enumerate(zip(event,rgb)):
 ax.text(e+.01,i-.18,f'{e:.3f}',va='center',fontsize=8);ax.text(r+.01,i+.18,f'{r:.3f}',va='center',fontsize=8)
ax.set_xlim(0,1.15);fig.savefig(p/'ridge_relative_mse.png',dpi=160);print(p/'ridge_relative_mse.png')
