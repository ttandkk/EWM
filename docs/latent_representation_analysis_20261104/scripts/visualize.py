#!/usr/bin/python3
"""Paired future latents: independent PCA and exact t-SNE. No training."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
ROOT=Path(os.environ.get('EWM_LATENT_INPUT', Path(__file__).resolve().parents[3]/'tmp/latent_expanded_20261104'))
OUT=Path(os.environ['EWM_PLOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
TASKS=('conveyor','moving_carton','turntable_cups','rotating_tube','tennis_rolling','tennis_falling','tennis_parabolic','tennis_container')
SEED=20261104

def gram_pca(x):
    x=x.astype(np.float64); x-=x.mean(axis=0)
    g=x@x.T; g=(g+g.T)/2
    e,u=np.linalg.eigh(g); order=np.argsort(e)[::-1]; e=np.maximum(e[order],0); u=u[:,order]
    return u[:,:2]*np.sqrt(e[:2]), e[:2]/e.sum(), g

def distances(g):
    d=np.maximum(np.diag(g)[:,None]+np.diag(g)[None,:]-2*g,0)
    np.fill_diagonal(d,0)
    return d

def joint_probabilities(dist,perplexity):
    n=len(dist); p=np.zeros_like(dist); target=np.log(perplexity)
    for i in range(n):
        mask=np.arange(n)!=i; d=dist[i,mask]; beta=1.0/max(np.median(d),1e-12); lo=0.; hi=np.inf
        for _ in range(80):
            logits=-d*beta; logits-=logits.max(); q=np.exp(logits); q/=q.sum()
            entropy=-np.sum(q*np.log(np.maximum(q,1e-300)))
            if abs(entropy-target)<1e-8: break
            if entropy>target:
                lo=beta; beta=2*beta if not np.isfinite(hi) else (beta+hi)/2
            else:
                hi=beta; beta=(beta+lo)/2
        p[i,mask]=q
    p=(p+p.T)/(2*n)
    assert np.isclose(p.sum(),1) and np.allclose(np.diag(p),0)
    return p

def loss_grad(y,p):
    diff=y[:,None,:]-y[None,:,:]; num=1/(1+np.einsum('ijk,ijk->ij',diff,diff)); np.fill_diagonal(num,0)
    q=num/num.sum(); q=np.maximum(q,1e-300)
    loss=np.sum(p[p>0]*np.log(p[p>0]/q[p>0]))
    gradient=4*np.sum(((p-q)*num)[:,:,None]*diff,axis=1)
    return loss,gradient

def tsne(d,perplexity):
    p=joint_probabilities(d,perplexity); rng=np.random.default_rng(SEED); y=rng.normal(0,1e-4,(len(d),2))
    velocity=np.zeros_like(y); gains=np.ones_like(y)
    for i in range(1000):
        _,grad=loss_grad(y,p*12 if i<250 else p)
        gains=np.where(np.sign(grad)!=np.sign(velocity),gains+0.2,gains*0.8); gains=np.maximum(gains,0.01)
        velocity=(0.5 if i<250 else 0.8)*velocity-100*gains*grad
        y+=velocity; y-=y.mean(axis=0)
    loss,_=loss_grad(y,p)
    if not np.isfinite(y).all() or not np.isfinite(loss): raise ValueError('nonfinite t-SNE')
    return y, float(loss)

def gradient_check():
    rng=np.random.default_rng(741); a=rng.normal(size=(7,5)); d=distances((a-a.mean(0))@(a-a.mean(0)).T)
    p=joint_probabilities(d,3); y=rng.normal(size=(7,2)); _,g=loss_grad(y,p); num=np.empty_like(y); h=1e-6
    for i in range(len(y)):
        for j in range(2):
            yp=y.copy(); ym=y.copy(); yp[i,j]+=h; ym[i,j]-=h
            num[i,j]=(loss_grad(yp,p)[0]-loss_grad(ym,p)[0])/(2*h)
    err=np.max(np.abs(g-num)); assert err<1e-7, err
    return float(err)

def dots(ax,xy,anchor):
    for mask,name,color,mark in [(anchor==0,'start','#2468ad','o'),(anchor!=0,'end','#e17730','^')]:
        ax.scatter(xy[mask,0],xy[mask,1],s=17,alpha=.75,color=color,marker=mark,label=name,edgecolors='none')
    ax.grid(alpha=.15); ax.set_aspect('equal',adjustable='box')

def main():
    gradient_error=gradient_check(); print('gradient max abs error',gradient_error,flush=True)
    summary={'seed':SEED,'tsne_perplexities':[15,30], 'tsne_gradient_max_abs_error':gradient_error,'tasks':{}}
    fig,axes=plt.subplots(8,2,figsize=(12,34),constrained_layout=True)
    all_tsne={}
    for i,task in enumerate(TASKS):
        path=ROOT/f'{task}_latents.npz'
        with np.load(path,allow_pickle=False) as z:
            ep=z['episode']; anchor=z['anchor']; arrays={m:z[m][:,1:].reshape(len(ep),-1).copy() for m in ('event','rgb')}
        assert len(ep)==100 and len(np.unique(ep))==50 and sum(anchor==0)==50
        summary['tasks'][task]={}; scores={}; gs={}
        for mod in ('event','rgb'):
            scores[mod],explained,gs[mod]=gram_pca(arrays[mod]); summary['tasks'][task][mod]={'pca_explained_ratio':explained.tolist()}
        lim=max(np.abs(np.concatenate(list(scores.values()))).max()*1.08,1e-3)
        for j,mod in enumerate(('event','rgb')):
            ax=axes[i,j]; dots(ax,scores[mod],anchor); ax.set(xlim=(-lim,lim),ylim=(-lim,lim))
            ev=summary['tasks'][task][mod]['pca_explained_ratio']
            ax.set_title(f'{task} | {mod.upper()} independent PCA',fontsize=11)
            ax.set_xlabel(f'PC1 ({100*ev[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100*ev[1]:.1f}%)')
        if i==0: axes[i,0].legend(loc='upper right')
        if task in ('tennis_falling','conveyor'):
            for mod in ('event','rgb'):
                for p in (15,30):
                    coords,loss=tsne(distances(gs[mod]),p)
                    all_tsne[task,mod,p]=coords; summary['tasks'][task][mod][f'tsne_p{p}_kl']=loss
        print(task,'done',flush=True)
    fig.suptitle('Independent PCA per task and modality; matching axis scales within each task\nFuture latent only; 50 paired trajectories × 2 windows. Axes differ across modalities.',fontsize=14)
    fig.savefig(OUT/'pca_all_tasks.png',dpi=150); plt.close(fig)
    for task in ('tennis_falling','conveyor'):
        with np.load(ROOT/f'{task}_latents.npz',allow_pickle=False) as z: anchor=z['anchor']
        fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
        for i,p in enumerate((15,30)):
            for j,mod in enumerate(('event','rgb')):
                ax=axes[i,j]; dots(ax,all_tsne[task,mod,p],anchor)
                ax.set_title(f'{mod.upper()} | perplexity={p} | KL={summary["tasks"][task][mod][f"tsne_p{p}_kl"]:.3f}')
                ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        axes[0,0].legend(); fig.suptitle(f'{task}: independent exact t-SNE, seed {SEED}\nAxes and apparent area are not comparable across panels.',fontsize=13)
        fig.savefig(OUT/f'tsne_{task}.png',dpi=150); plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
