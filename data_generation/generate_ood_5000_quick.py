"""Quickly generate valid Re=5000 3D OOD data for rigorous validation."""
import numpy as np
import h5py
from generate_3d_ns_correct import NS3DSolver, generate_trajectory

print('Generating Re=5000 3D OOD data...')
solver = NS3DSolver(N=64, Re=5000, dt=0.001)

train_samples = []
for i in range(3):
    print(f'  Trajectory {i+1}/3')
    traj = generate_trajectory(solver, n_steps=50, n_burnin=300, seed=5000+i)
    train_samples.append(traj)
    
train_data = np.concatenate(train_samples, axis=0)
train_input = train_data[:-1]
train_target = train_data[1:]

print(f'Train input shape: {train_input.shape}')
print(f'  mean={train_input.mean():.4f}, std={train_input.std():.4f}, max={np.abs(train_input).max():.4f}')

with h5py.File('data/turbulence3d_re5000_64x64x64.h5', 'w') as f:
    f.create_dataset('train/input', data=train_input)
    f.create_dataset('train/target', data=train_target)
    
print('Saved to data/turbulence3d_re5000_64x64x64.h5')
