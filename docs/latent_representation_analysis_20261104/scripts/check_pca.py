#!/usr/bin/env python3
"""Independent direct SVD and explicit reconstruction audit of falling PCA."""
import json
from pathlib import Path
import numpy as np
repo = Path(__file__).resolve().parents[3]
root = repo/'tmp/latent_pca_reconstruction_20261104/pca'
with np.load(repo/'tmp/latent_expanded_20261104/tennis_falling_latents.npz') as d:
    arr = {k:d[k].copy() for k in ('event','rgb','episode','anchor')}
with np.load(root / 'tennis_falling_pca.npz') as p:
    saved = {k:p[k].copy() for k in p.files}
results = json.loads((root/'results.json').read_text())['tasks']['tennis_falling']
for mod in ('event','rgb'):
    x=arr[mod][:,1:].reshape(100,-1).astype(np.float64)
    tr,te=saved['train_indices'],saved['test_indices']
    mu=x[tr].mean(0)
    centered=x[tr]-mu
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    spectrum=s*s / np.sum(centered**2)
    for k in (8,16,32,64):
        ref=results['modalities'][mod]['values'][str(k)]
        explained=float(np.sum(spectrum[:k]))
        test=x[te]-mu
        recon=(test@vh[:k].T)@vh[:k]
        retained=1-float(np.sum((test-recon)**2)/np.sum(test**2))
        np.testing.assert_allclose(explained,ref['train_explained_variance'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(retained,ref['test_retained_energy'],rtol=1e-9,atol=1e-10)
        basis=saved[f'{mod}_basis'][:k].astype(np.float64)
        smu=saved[f'{mod}_mean'].astype(np.float64)
        test2=x[te]-smu
        recon2=(test2@basis.T)@basis
        value=1-float(np.sum((test2-recon2)**2)/np.sum(test2**2))
        np.testing.assert_allclose(value,ref['test_retained_energy'],rtol=2e-7,atol=1e-7)
        print(mod,k,'train explained',explained,'test retained',retained,'saved float32 retained',value,flush=True)
print('PASS: independent direct SVD and explicit reconstructions')
