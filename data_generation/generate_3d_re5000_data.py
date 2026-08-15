"""Generate valid Re=5000 3D NS turbulence data for OOD experiments."""
import numpy as np
import h5py
import sys
from generate_3d_ns_correct import NS3DSolver, generate_trajectory

print("="*70)
print("Generating Re=5000 3D Turbulence Data")
print("="*70)

solver = NS3DSolver(N=64, Re=5000, dt=0.001)

# Training data
print("\n[1] Training data (5 trajectories, 200 steps each)")
train_samples = []
for i in range(5):
    print(f"  Trajectory {i+1}/5")
    traj = generate_trajectory(solver, n_steps=200, n_burnin=300, seed=5000+i)
    train_samples.append(traj)
    print(f"    Got {len(traj)} samples")

train_data = np.concatenate(train_samples, axis=0)
train_input = train_data[:-1]
train_target = train_data[1:]
print(f"\n  Train: {train_input.shape}")

# Validation data
print("\n[2] Validation data (2 trajectories, 100 steps each)")
val_samples = []
for i in range(2):
    print(f"  Trajectory {i+1}/2")
    traj = generate_trajectory(solver, n_steps=100, n_burnin=400, seed=6000+i)
    val_samples.append(traj)

val_data = np.concatenate(val_samples, axis=0)
val_input = val_data[:-1]
val_target = val_data[1:]
print(f"\n  Val: {val_input.shape}")

# Save
print("\n[3] Saving...")
with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
    f.create_dataset('train/input', data=train_input)
    f.create_dataset('train/target', data=train_target)
    f.create_dataset('val/input', data=val_input)
    f.create_dataset('val/target', data=val_target)

print("  Saved: data/turbulence3d_re5000_64x64x64.h5")
print("\nDONE")
