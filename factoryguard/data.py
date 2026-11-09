"""Synthetic dataset for FactoryGuard-Maintenance — Predictive Maintenance Prototype; no public dataset is used."""
from __future__ import annotations
import numpy as np
FEATURES=['age', 'hours', 'load', 'wear', 'vibration', 'temperature', 'pressure', 'acoustic']
GROUP_NAME="entity_id"
GROUP_SPLIT=True
def make_dataset(n=2400,seed=42):
    if n<160: raise ValueError("n must be at least 160")
    rng=np.random.default_rng(seed)
    machines=max(24,n//40); machine=np.arange(n)%machines; age=rng.uniform(1,15,machines)[machine]; hours=rng.uniform(0,9000,machines)[machine]; load=np.clip(rng.normal(.65,.18,n),.05,1.2); wear=np.clip(.012*age+.000035*hours+.45*(load-.5)+rng.normal(0,.12,n),0,1.5); vibration=np.maximum(.02,.18+.82*wear+rng.normal(0,.10,n)); temperature=55+15*wear+9*load+rng.normal(0,3,n); pressure=100-11*wear+4*load+rng.normal(0,3,n); acoustic=20+18*wear+rng.normal(0,5,n); z=-4.2+2*wear+2.8*vibration+.035*(temperature-60)+.045*(100-pressure)+.025*(acoustic-20); p=1/(1+np.exp(-z)); y=(rng.random(n)<p).astype(int); groups=machine
    X=np.column_stack([age, hours, load, wear, vibration, temperature, pressure, acoustic]).astype(float)
    if X.shape!=(n,len(FEATURES)) or not np.isfinite(X).all(): raise ValueError("invalid feature matrix")
    if np.asarray(groups).shape!=(n,): raise ValueError("invalid group vector")
    return X,y.astype(int),np.asarray(groups)
