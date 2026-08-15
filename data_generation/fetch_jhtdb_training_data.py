"""
Fetch JHTDB training data (sequential, conservative)
====================================================
"""

import numpy as np
import h5py
from zeep import Client
import time

WS = 'https://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL'
TOKEN = 'edu.jhu.pha.turbulence.testing-201406'
DATASET = 'isotropic1024coarse'

print("Loading WSDL...")
client = Client(WS)
print("OK")


def fetch(T, x, y, z):
    try:
        r = client.service.GetAnyCutoutWeb(
            TOKEN, DATASET, 'u', T, x, y, z, x+63, y+63, z, 1, 1, 1, 1, '')
        d = np.frombuffer(r, dtype=np.float32)
        return np.stack([d[0::3].reshape(64,64), d[1::3].reshape(64,64)], axis=0)
    except Exception as e:
        print(f"  ERR {T},{x},{y}: {str(e)[:50]}")
        return None


# 10 positions x 4 time pairs = 40 requests (~2 min with 3s delay)
positions = [(100,100),(300,200),(500,400),(700,300),(200,600),
             (400,800),(600,500),(800,700),(150,450),(350,650)]
time_pairs = [(5,15),(10,20),(15,25),(20,30)]

inputs, targets = [], []
for i, (x,y) in enumerate(positions):
    for t1, t2 in time_pairs:
        print(f"Fetch pos{i+1} ({x},{y}) T={t1}...", end='', flush=True)
        u1 = fetch(t1, x, y, 512)
        time.sleep(0.5)
        print(f" T={t2}...", end='', flush=True)
        u2 = fetch(t2, x, y, 512)
        time.sleep(0.5)
        if u1 is not None and u2 is not None:
            inputs.append(u1)
            targets.append(u2)
            print(f" OK ({len(inputs)})")
        else:
            print(" FAIL")

print(f"\nFetched: {len(inputs)} pairs")
if len(inputs) == 0:
    exit(1)

# Normalize
E = 0.5 * (np.array(inputs)**2).mean(axis=(1,2,3), keepdims=True)
scale = np.sqrt(1.0 / (E + 1e-8))
inputs = np.array(inputs) * scale
targets = np.array(targets) * scale

# Split
n = len(inputs)
p = np.random.permutation(n)
n1, n2 = int(0.7*n), int(0.15*n)

with h5py.File('data/jhtdb_training_64x64.h5','w') as f:
    for split, idx in [('train',p[:n1]),('val',p[n1:n1+n2]),('test',p[n1+n2:])]:
        g = f.create_group(split)
        g.create_dataset('input', data=inputs[idx].astype(np.float32))
        g.create_dataset('target', data=targets[idx].astype(np.float32))

print(f"Saved: train={n1}, val={n2}, test={n-n1-n2}")
