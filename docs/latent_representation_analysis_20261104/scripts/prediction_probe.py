import json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import os
T=time.monotonic();root=Path(__file__).resolve().parents[3]; base=root/'datasets/piper_tennis_falling'
d=np.load(root/'tmp/event_rgb_latent_pilot_20261104/tennis_falling_latents.npz'); ep=d['episode']; train=np.isin(ep,[0,10,20]); test=np.isin(ep,[30,40]);
assert (train.sum(),test.sum())==(6,4)
cols=['episode_index','frame_index','observation.object_active','observation.object_pose','observation.object_velocity'];t=pq.read_table(sorted((base/'data').rglob('*.parquet')),columns=cols).to_pandas()
lookup={(int(a),int(b)):(float(np.asarray(c).reshape(-1)[0]),np.asarray(p,dtype=float)[:3],np.asarray(v,dtype=float)[:3]) for a,b,c,p,v in zip(*(t[col] for col in cols))}
r=[]
for i,(episode,anchor) in enumerate(zip(d['episode'],d['anchor'])):
 for b,off in enumerate((8,16,24),start=1):
  active,pos,vel=lookup[(int(episode),int(anchor)+off)]
  if active>.5:r.append((i,b,int(episode),(0 if anchor==0 else 1)*3+b-1,pos,vel))
indices=np.array([v[0] for v in r]);blocks=np.array([v[1] for v in r]);eps=np.array([v[2] for v in r]);stages=np.array([v[3] for v in r]);tr=np.isin(eps,[0,10,20]);te=np.isin(eps,[30,40]);targets={m:np.stack([z[k] for z in r]) for m,k in [('position',4),('velocity',5)]};reg=np.array([.001,.01,.1,1,10,100,1000.]); x=d['rgb'][:,0].reshape(10,-1).astype(float)
# Stage is known input; choose phase-specific estimates only from each fitting split.
def phase_mean(y,fit,val):
 sf=stages[fit]; yf=y[fit]; return np.stack([yf[sf==s].mean(0) for s in stages[val]])
def physical_pred(X,y,fit,val,a,xv=None):
 sf=stages[fit];sv=stages[val];xf=X[fit];yf=y[fit];xv=X[val] if xv is None else xv
 mx={s:xf[sf==s].mean(0) for s in np.unique(sf)};my={s:yf[sf==s].mean(0) for s in np.unique(sf)}
 xc=xf-np.stack([mx[s] for s in sf]);yc=yf-np.stack([my[s] for s in sf]);xt=xv-np.stack([mx[s] for s in sv]);ym=np.stack([my[s] for s in sv]);K=xc@xc.T;scale=np.trace(K)/len(K)
 return ym if scale<1e-20 else ym+(xt@xc.T)@np.linalg.solve(K+a*scale*np.eye(len(K)),yc)
def latent_pred(x,y,fit,val,a):
 xm=x[fit].mean(0);ym=y[fit].mean(0);xc=x[fit]-xm;xt=x[val]-xm;K=xc@xc.T;scale=np.trace(K)/len(K)
 return ym if scale<1e-20 else ym+(xt@xc.T)@np.linalg.solve(K+a*scale*np.eye(len(K)),y[fit]-ym)
def error(y,z):return float(np.linalg.norm(y-z,axis=-1).mean())
out={'train_windows':int(train.sum()),'test_windows':int(test.sum()),'active_train_targets':int(tr.sum()),'active_test_targets':int(te.sum()),'tasks':{},'notes':'current RGB latent to future event/RGB latent; physical readout fit only on ground-truth train latent; test readout on predicted future latent; physical model given phase; both regularizers selected using leave-one-training-trajectory-out CV; active points only'}
for modality in ('event','rgb'):
 yall=d[modality][:,1:].reshape(10,-1).astype(float)
 scores=[]
 for alpha in reg:
  numer=denom=0.
  for hold in [0,10,20]:
   fit=train&(ep!=hold);val=train&(ep==hold);p=latent_pred(x,yall,fit,val,alpha); numer+=np.square(p-yall[val]).sum();denom+=np.square(yall[val]-yall[fit].mean(0)).sum()
  scores.append(float(numer/denom))
 best=float(reg[np.argmin(scores)]);yhat=latent_pred(x,yall,train,test,best).reshape(test.sum(),3,16,32,32);test_window_indices=np.flatnonzero(test);wmap={int(k):j for j,k in enumerate(test_window_indices)}; X=np.stack([d[modality][i,b].reshape(-1) for i,b in zip(indices,blocks)]).astype(float); Xpredict=np.stack([yhat[wmap[int(i)],b-1].reshape(-1) for i,b in zip(indices[te],blocks[te])]).astype(float)
 result={'latent_alpha':best,'latent_cv_relative_mse':scores,'physical':{}}
 for name,y in targets.items():
  cvs=[]
  for a in reg:
   ys=[];ps=[]
   for hold in [0,10,20]:
    fit=tr&(eps!=hold);val=tr&(eps==hold);ps.extend(physical_pred(X,y,fit,val,a));ys.extend(y[val])
   cvs.append(error(np.array(ys),np.array(ps)))
  a=float(reg[np.argmin(cvs)]);pp=physical_pred(X,y,tr,te,a,xv=Xpredict);oracle=physical_pred(X,y,tr,te,a);basep=phase_mean(y,tr,te)
  result['physical'][name]={'chosen_alpha':a,'cv_error_on_ground_truth_latent':cvs,'phase_baseline':error(y[te],basep),'ground_truth_latent_probe':error(y[te],oracle),'predicted_latent_probe':error(y[te],pp),'by_phase':{str(int(s)):{'phase_baseline':error(y[te][stages[te]==s],basep[stages[te]==s]),'ground_truth':error(y[te][stages[te]==s],oracle[stages[te]==s]),'predicted':error(y[te][stages[te]==s],pp[stages[te]==s])} for s in np.unique(stages[te])}}
 out['tasks'][modality]=result
out['elapsed_s']=time.monotonic()-T
print(json.dumps(out,indent=2),flush=True);Path(os.environ['EWM_ANALYSIS_OUTPUT']).open('x').write(json.dumps(out,indent=2))
