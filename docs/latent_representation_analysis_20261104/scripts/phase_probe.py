#!/usr/bin/env python3
"""Phase-matched PCA: train/test remains trajectory-disjoint."""
import json
from pathlib import Path
import numpy as np
from pca_probe import TASKS, evaluate
import os
root=Path(os.environ.get('EWM_LATENT_INPUT', Path(__file__).resolve().parents[3]/'tmp/latent_expanded_20261104'))
out={'protocol':'PCA fit separately for each task, phase, modality; train first 40 sorted trajectories and test last 10; one start or end window per episode. Test retained energy relative to training mean, not test-centered explained variance.','tasks':{}}
for task in TASKS:
    with np.load(root / f'{task}_latents.npz') as f:
        data={k:f[k].copy() for k in ('episode','anchor','event','rgb')}
    eps=data['episode']
    ids=np.unique(eps)
    pairs=[]
    for eid in ids:
        idx=np.flatnonzero(eps==eid)
        if len(idx)!=2 or len(np.unique(data['anchor'][idx])) != 2:
            raise ValueError(f'{task} episode {eid}: expected two distinct windows')
        pairs.append(idx[np.argsort(data['anchor'][idx])])
    pairs=np.asarray(pairs)
    task_result={}
    for phase,pi in [('start',0),('end',1)]:
        tr=pairs[:40,pi]
        te=pairs[40:,pi]
        task_result[phase]={}
        for mod in ('event','rgb'):
            a=data[mod][:,1:].reshape(len(eps),-1)
            metric,_,_=evaluate(a,tr,te)
            task_result[phase][mod]=metric
    out['tasks'][task]=task_result
    print(task,flush=True)
with Path(os.environ['EWM_ANALYSIS_OUTPUT']).open('x') as f: json.dump(out,f,indent=2,allow_nan=False)
