"""
Generate high-resolution 2D turbulence dataset (128x128).
Uses SpectralDataGenerator with Re=1000.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import h5py
import torch
from tqdm import tqdm

from src.data_generation import SpectralDataGenerator

# Configuration
RESOLUTION = 128
REYNOLDS = 1000.0
DT = 0.005
WARMUP = 300
N_TRAIN = 400
N_VAL = 50
N_TEST = 50
OUTPUT = 'data/turbulence_re1000_128x128.h5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Generating {N_TRAIN} train + {N_VAL} val + {N_TEST} test samples")
print(f"Resolution: {RESOLUTION}x{RESOLUTION}, Re={REYNOLDS}, dt={DT}")

gen = SpectralDataGenerator(
    resolution=RESOLUTION,
    reynolds=REYNOLDS,
    dt=DT,
    warmup_steps=WARMUP,
    random_seed=42
)

# Generate all splits
train_inputs = []
train_targets = []
val_inputs = []
val_targets = []
test_inputs = []
test_targets = []

for split_name, n_samples, inp_list, tgt_list in [
    ('train', N_TRAIN, train_inputs, train_targets),
    ('val', N_VAL, val_inputs, val_targets),
    ('test', N_TEST, test_inputs, test_targets),
]:
    print(f"\nGenerating {split_name} set ({n_samples} samples)...")
    for i in tqdm(range(n_samples)):
        u0 = gen.generate_initial_condition()
        # Warmup
        u = u0.copy()
        for _ in range(WARMUP):
            u = gen.evolve_step(u)
        # Store input
        inp_list.append(u.copy())
        # Evolve one step for target
        u_next = gen.evolve_step(u)
        tgt_list.append(u_next.copy())

# Convert to numpy arrays and save
def to_array(lst):
    arr = np.stack(lst, axis=0)  # (N, 2, H, W)
    return arr

with h5py.File(OUTPUT, 'w') as f:
    f.create_dataset('train/input', data=to_array(train_inputs))
    f.create_dataset('train/target', data=to_array(train_targets))
    f.create_dataset('val/input', data=to_array(val_inputs))
    f.create_dataset('val/target', data=to_array(val_targets))
    f.create_dataset('test/input', data=to_array(test_inputs))
    f.create_dataset('test/target', data=to_array(test_targets))
    
    # Metadata
    meta = f.create_group('metadata')
    meta.attrs['resolution'] = RESOLUTION
    meta.attrs['reynolds'] = REYNOLDS
    meta.attrs['dt'] = DT
    meta.attrs['n_train'] = N_TRAIN
    meta.attrs['n_val'] = N_VAL
    meta.attrs['n_test'] = N_TEST

print(f"\nSaved to {OUTPUT}")
print(f"Train: {to_array(train_inputs).shape}")
print(f"Val: {to_array(val_inputs).shape}")
print(f"Test: {to_array(test_inputs).shape}")
