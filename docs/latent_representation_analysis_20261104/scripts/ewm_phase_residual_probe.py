"""Episode-held-out stage-demeaned ridge; phase label is diagnostic only."""
import json,time
from pathlib import Path
import numpy as np
import os
root=Path(os.environ.get('EWM_PILOT_LATENT_INPUT', Path(__file__).resolve().parents[3]/'tmp/event_rgb_latent_pilot_20261104'))
tasks=['conveyor','moving_carton','turntable_cups','rotating_tube','tennis_rolling','tennis_falling','tennis_parabolic','tennis_container']
reg=[.001,.01,.1,1.,10.,100.]
def predict(x,y,phase,x_test,phase_test,alpha):
    xmeans={p:x[phase==p].mean(0) for p in (False,True)}
    ymeans={p:y[phase==p].mean(0) for p in (False,True)}
    xc=x-np.stack([xmeans[p] for p in phase]); yc=y-np.stack([ymeans[p] for p in phase])
    xt=x_test-np.stack([xmeans[p] for p in phase_test]); base=np.stack([ymeans[p] for p in phase_test])
    k=xc@xc.T; scale=np.trace(k)/len(k)
    if scale<1e-20: return base,base
    return base+(xt@xc.T)@np.linalg.solve(k+alpha*scale*np.eye(len(k)),yc),base
start=time.monotonic(); out={}
for task in tasks:
    with np.load(root/(task+'_latents.npz')) as d:
        ep=d['episode']; phase=d['anchor']==0; n=len(ep)
        x=d['rgb'][:,0].reshape(n,-1).astype('float64')
        ys={m:d[m][:,1:].reshape(n,-1).astype('float64') for m in ('event','rgb')}
    assert n==10 and len(np.unique(ep))==5
    result={}
    for mode,y in ys.items():
        folds=[]
        for held in np.unique(ep):
            train=ep!=held;test=ep==held
            choices=[]
            for a in reg:
                ss=0.;bs=0.
                for innerheld in np.unique(ep[train]):
                    fit=train & (ep!=innerheld);valid=train & (ep==innerheld)
                    pred,base=predict(x[fit],y[fit],phase[fit],x[valid],phase[valid],a)
                    ss+=np.square(pred-y[valid]).sum();bs+=np.square(base-y[valid]).sum()
                choices.append(ss/bs)
            a=reg[int(np.argmin(choices))]
            pred,base=predict(x[train],y[train],phase[train],x[test],phase[test],a)
            num=float(np.square(pred-y[test]).mean());den=float(np.square(base-y[test]).mean())
            folds.append({'held_episode':int(held),'alpha':a,'selected_cv_ratio':min(choices),'conditioned_mse':num,'stage_baseline_mse':den,'ratio':num/den})
        result[mode]={'folds':folds,'pooled_ratio':sum(f['conditioned_mse'] for f in folds)/sum(f['stage_baseline_mse'] for f in folds)}
    out[task]=result
    print(task, *(f"{m}={result[m]['pooled_ratio']:.4f}" for m in ('event','rgb')),flush=True)
result={'elapsed_seconds':time.monotonic()-start,'method':'5-fold leave-one-episode-out and nested train-only episode-CV; known first-versus-last phase input used to center X and Y; stage oracle is diagnostic and not available at deployment','regularization':reg,'tasks':out}
with Path(os.environ['EWM_ANALYSIS_OUTPUT']).open('x') as f: json.dump(result,f,indent=2)
print('elapsed',round(result['elapsed_seconds'],2))
