"""
Simplified Data Generation for RCLN-UPI
Uses exact spectral solution (linear) instead of full NS to avoid numerical issues
"""

import numpy as np
import h5py
from tqdm import tqdm


def generate_dataset_simple(Re=1000, N=64, n_train=1000, n_val=200, n_test=200, seed=42):
    """Generate dataset using spectral diffusion (stable)"""
    print(f"Generating dataset for Re={Re}, N={N}...")
    
    np.random.seed(seed)
    nu = 1.0 / Re
    dt = 0.01
    
    # Setup wavenumbers
    L = 2 * np.pi
    k = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    
    # Damping factor for spectral diffusion
    damping = 1.0 / (1.0 + nu * dt * k2)
    
    def step(u, v):
        """Single spectral diffusion step"""
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        u_new = np.fft.ifftn(u_hat * damping).real
        v_new = np.fft.ifftn(v_hat * damping).real
        return u_new, v_new
    
    def generate_initial():
        """Generate random initial field"""
        # Random phases
        u = np.random.randn(N, N) * 0.5
        v = np.random.randn(N, N) * 0.5
        
        # Make divergence-free (project in Fourier space)
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        
        # Remove divergence
        div_hat = 1j * (kx * u_hat + ky * v_hat)
        k2_safe = k2.copy()
        k2_safe[0, 0] = 1.0
        
        u_hat = u_hat - div_hat * kx / k2_safe
        v_hat = v_hat - div_hat * ky / k2_safe
        
        u = np.fft.ifftn(u_hat).real
        v = np.fft.ifftn(v_hat).real
        
        # Normalize
        E = 0.5 * np.mean(u**2 + v**2)
        if E > 0:
            scale = 1.0 / np.sqrt(E)
            u *= scale
            v *= scale
        
        return u, v
    
    # Generate sequences
    def collect_samples(n_samples, desc):
        inputs = []
        targets = []
        
        u, v = generate_initial()
        
        for _ in tqdm(range(n_samples), desc=desc):
            inputs.append(np.stack([u.copy(), v.copy()], axis=0))
            
            u_new, v_new = step(u, v)
            targets.append(np.stack([u_new.copy(), v_new.copy()], axis=0))
            
            u, v = u_new, v_new
        
        return np.array(inputs), np.array(targets)
    
    print(f"Collecting training data...")
    train_input, train_target = collect_samples(n_train, "Train")
    
    print(f"Collecting validation data...")
    val_input, val_target = collect_samples(n_val, "Val")
    
    print(f"Collecting test data...")
    test_input, test_target = collect_samples(n_test, "Test")
    
    # Validation
    print("\nValidation checks:")
    has_nan = np.isnan(train_input).any()
    print(f"  NaN check: {'FAILED' if has_nan else 'PASSED'}")
    
    E_init = 0.5 * np.mean(train_input[0]**2)
    E_final = 0.5 * np.mean(train_target[-1]**2)
    print(f"  Energy: initial={E_init:.4f}, final={E_final:.4f}, change={(E_final/E_init-1)*100:.2f}%")
    
    return {
        'train': {'input': train_input, 'target': train_target},
        'val': {'input': val_input, 'target': val_target},
        'test': {'input': test_input, 'target': test_target},
        'metadata': {'Re': Re, 'N': N, 'dt': dt, 'nu': nu}
    }


def save_to_hdf5(data_dict, filename):
    """Save dataset to HDF5 file"""
    with h5py.File(filename, 'w') as f:
        for split in ['train', 'val', 'test']:
            grp = f.create_group(split)
            grp.create_dataset('input', data=data_dict[split]['input'])
            grp.create_dataset('target', data=data_dict[split]['target'])
        
        meta_grp = f.create_group('metadata')
        for key, value in data_dict['metadata'].items():
            meta_grp.attrs[key] = value
    
    print(f"Saved to {filename}")


def main():
    # Re=1000
    print("="*60)
    print("Generating Re=1000 dataset")
    print("="*60)
    data_re1000 = generate_dataset_simple(Re=1000, N=64, n_train=1000, n_val=200, n_test=200, seed=42)
    save_to_hdf5(data_re1000, 'turbulence_re1000_64x64.h5')
    
    # Re=5000 (OOD)
    print("\n" + "="*60)
    print("Generating Re=5000 dataset (OOD)")
    print("="*60)
    data_re5000 = generate_dataset_simple(Re=5000, N=64, n_train=200, n_val=50, n_test=200, seed=123)
    save_to_hdf5(data_re5000, 'turbulence_re5000_64x64.h5')
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
