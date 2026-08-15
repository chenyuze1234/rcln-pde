"""Fast generation of Re=5000 3D data with reduced burn-in for OOD testing."""
import numpy as np
import h5py
from generate_3d_ns_correct import NS3DSolver, generate_trajectory

print("Generating Re=5000 3D data (fast mode, reduced burn-in)")
solver = NS3DSolver(N=64, Re=5000, dt=0.001)

# Reduced data for quick generation
train_samples = []
for i in range(3):
    print(f"  Train traj {i+1}/3")
    traj = generate_trajectory(solver, n_steps=100, n_burnin=150, seed=5000+i)
    train_samples.append(traj)

train_data = np.concatenate(train_samples, axis=0)
train_input = train_data[:-1]
train_target = train_data[1:]

val_samples = []
for i in range(1):
    print(f"  Val traj {i+1}/1")
    traj = generate_trajectory(solver, n_steps=50, n_burnin=200, seed=6000+i)
    val_samples.append(traj)

val_data = np.concatenate(val_samples, axis=0)
val_input = val_data[:-1]
val_target = val_data[1:]

print(f"Train: {train_input.shape}, Val: {val_input.shape}")

with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
    f.create_dataset('train/input', data=train_input)
    f.create_dataset('train/target', data=train_target)
    f.create_dataset('val/input', data=val_input)
    f.create_dataset('val/target', data=val_target)

print("Saved.")
