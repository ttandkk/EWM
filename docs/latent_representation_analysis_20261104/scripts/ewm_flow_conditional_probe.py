"""CPU-only grouped latent-prediction / flow proxy; reuses frozen LeanVAE caches.
Run OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /tmp/ewm_flow_conditional_probe.py
Not an EWM checkpoint evaluation. Never fits on held-out episodes.
"""
import json, time
from pathlib import Path
import numpy as np
import os
root=Path(os.environ.get('EWM_PILOT_LATENT_INPUT', Path(__file__).resolve().parents[3]/'tmp/event_rgb_latent_pilot_20261104'))
tasks=['conveyor','moving_carton','turntable_cups','rotating_tube','tennis_rolling','tennis_falling','tennis_parabolic','tennis_container']
grid=[.001,.01,.1,1.,10.,100.]
times=np.arange(.025,1.,.05)

def predict(x,y,xt,alpha):
    mx=x.mean(0); my=y.mean(0); xc=x-mx; k=xc@xc.T; scale=np.trace(k)/len(k)
    if scale < 1e-20: return np.broadcast_to(my,(len(xt),len(my))).copy()
    return my+((xt-mx)@xc.T)@np.linalg.solve(k+alpha*scale*np.eye(len(k)), y-my)

def fit_alpha(x,y,episode,train):
    best=None
    for a in grid:
        err=0.; base=0.
        for held in np.unique(episode[train]):
            fit=train & (episode!=held); val=train & (episode==held)
            p=predict(x[fit],y[fit],x[val],a)
            err+=np.sum((p-y[val])**2);base+=np.sum((y[fit].mean(0)-y[val])**2)
        r=err/base
        if best is None or r<best[0]: best=(r,a)
    return best[1]

def cv_variances(x,y,ep,train,a):
    ce=0.;ue=0.;n=0
    for held in np.unique(ep[train]):
        fit=train & (ep!=held); val=train & (ep==held)
        ce+=np.square(predict(x[fit],y[fit],x[val],a)-y[val]).sum()
        ue+=np.square(y[fit].mean(0)-y[val]).sum(); n+=y[val].size
    return ce/n,ue/n

t0=time.monotonic(); results={}
for task_idx,task in enumerate(tasks):
    with np.load(root/(task+'_latents.npz')) as d:
        ep=d['episode']; anchors=d['anchor']; n=len(ep)
        x=d['rgb'][:,0].reshape(n,-1).astype('float64')
        targets={m:d[m][:,1:].reshape(n,-1).astype('float64') for m in ('event','rgb')}
    assert len(np.unique(ep))==5 and n==10
    metrics={}
    for m,y in targets.items():
        folds=[]
        for held in np.unique(ep):
            train=ep!=held;test=ep==held
            assert train.sum()==8 and test.sum()==2
            a=fit_alpha(x,y,ep,train)
            cond_var,uncond_var=cv_variances(x,y,ep,train,a)
            cond=predict(x[train],y[train],x[test],a)
            base=np.broadcast_to(y[train].mean(0),cond.shape)
            yt=y[test];de=np.square(cond-yt).mean();be=np.square(base-yt).mean()
            phase_base=np.stack([y[train & ((anchors==0)==(anchor==0))].mean(0) for anchor in anchors[test]])
            pe=np.square(phase_base-yt).mean()
            rng=np.random.default_rng(10000+100*task_idx+int(held))
            noise=rng.standard_normal(yt.shape)
            # Flow z=(1-t)*y+t*noise, target v=noise-y. Posterior-style
            # scalar Gaussian gains, calibrated via training-only grouped CV.
            fcond=[];fbase=[]
            for t in times:
                z=(1-t)*yt+t*noise; velocity=noise-yt
                def flow(pred,var):
                    gain=(1-t)*var/((1-t)**2*var+t*t)
                    yhat=pred+gain*(z-(1-t)*pred)
                    return (z-yhat)/t
                fcond.append(np.mean((flow(cond,cond_var)-velocity)**2))
                fbase.append(np.mean((flow(base,uncond_var)-velocity)**2))
            folds.append({'held_episode':int(held),'alpha':a,'direct_cond_mse':de,'direct_mean_mse':be,'direct_phase_mse':pe,
                          'cond_train_group_cv_mse':cond_var,'mean_train_group_cv_mse':uncond_var,
                          'flow_cond_mse':float(np.mean(fcond)),'flow_mean_mse':float(np.mean(fbase)),
                          'direct_cond_over_mean':de/be,'flow_cond_over_mean':float(np.mean(fcond)/np.mean(fbase)),
                          'flow_cond_by_t':fcond,'flow_mean_by_t':fbase})
        metrics[m]={'folds':folds,'pooled_direct_cond_over_mean':sum(f['direct_cond_mse'] for f in folds)/sum(f['direct_mean_mse'] for f in folds),
                    'pooled_direct_phase_over_mean':sum(f['direct_phase_mse'] for f in folds)/sum(f['direct_mean_mse'] for f in folds),
                    'pooled_flow_cond_over_mean':sum(f['flow_cond_mse'] for f in folds)/sum(f['flow_mean_mse'] for f in folds),
                    'pooled_flow_cond_mse':float(np.mean([f['flow_cond_mse'] for f in folds])),
                    'pooled_flow_mean_mse':float(np.mean([f['flow_mean_mse'] for f in folds]))}
    results[task]=metrics
    print(task, 'direct',*[f"{m}={metrics[m]['pooled_direct_cond_over_mean']:.4f}" for m in ('event','rgb')], 'flow',*[f"{m}={metrics[m]['pooled_flow_cond_over_mean']:.4f}" for m in ('event','rgb')],flush=True)
out={'description':'5-fold held-out episode, two windows per episode, current RGB VAE latent -> future event / RGB VAE latents; nested grouped CV kernel ridge, Gaussian flow proxy calibrated from train-only grouped CV residuals; 20 evenly spaced noise levels, common random noise across modalities. Ratio conditioned / train-mean; <1 helps. No DINO/action conditioning; not trained EWM.', 'times':times.tolist(),'alpha_grid':grid,'elapsed_seconds':time.monotonic()-t0,'tasks':results}
with Path(os.environ['EWM_ANALYSIS_OUTPUT']).open('x') as f: json.dump(out,f,indent=2)
print('elapsed_sec',round(out['elapsed_seconds'],2))
