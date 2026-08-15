"""
Data Generation Pipeline for RCLN-UPI
2D Incompressible Navier-Stokes, Pseudo-spectral Method
"""

import numpy as np
import h5py
import torch
from tqdm import tqdm


def helmholtz_projection(u_hat, v_hat, kx, ky):
    """Project velocity field to divergence-free in Fourier space"""
    div_hat = 1j * (kx * u_hat + ky * v_hat)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # Avoid division by zero
    
    # Remove divergence
    u_hat_proj = u_hat - div_hat * kx / k2
    v_hat_proj = v_hat - div_hat * ky / k2
    
    return u_hat_proj, v_hat_proj


def compute_energy_spectrum(u_hat, v_hat, k_mag, N):
    """Compute radial energy spectrum"""
    energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    
    # Bin by wavenumber magnitude
    k_bins = np.arange(0, N//2 + 1)
    spectrum = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        spectrum[i] = np.sum(energy[mask])
    
    return spectrum


def generate_initial_field(N=64, Re=1000, seed=42):
    """Generate random divergence-free initial field with von Karman spectrum"""
    np.random.seed(seed)
    
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # Wavenumbers
    k = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2)
    
    # von Karman-Pao energy spectrum with modifications for stability
    k0 = 4.0  # Peak wavenumber
    k_eta = N // 3  # Dissipation wavenumber
    
    E_k = np.zeros_like(k_mag)
    # Only use modes above k_min to avoid large-scale instability
    k_min = 2.0
    mask = k_mag > k_min
    E_k[mask] = (k_mag[mask]**4 / (1 + k_mag[mask]**2)**3) * np.exp(-2 * (k_mag[mask]/k_eta)**2)
    
    # Cap maximum energy to avoid numerical issues
    E_k = np.clip(E_k, 0, 1.0)
    
    # Random phases
    phase_u = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    phase_v = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    # Velocity in Fourier space
    u_hat = np.sqrt(E_k) * phase_u
    v_hat = np.sqrt(E_k) * phase_v
    
    # Helmholtz projection to make divergence-free
    u_hat, v_hat = helmholtz_projection(u_hat, v_hat, kx, ky)
    
    # Transform to physical space
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
    
    # Normalize energy to 1.0
    E = 0.5 * np.mean(u**2 + v**2)
    scale = 1.0 / np.sqrt(E)
    u *= scale
    v *= scale
    
    return u, v


def navier_stokes_rk2(u, v, nu, dt, kx, ky):
    """One step of 2D Navier-Stokes using RK2 (more stable)"""
    def compute_rhs(u_hat, v_hat):
        # Compute derivatives
        ux_hat = 1j * kx * u_hat
        uy_hat = 1j * ky * u_hat
        vx_hat = 1j * kx * v_hat
        vy_hat = 1j * ky * v_hat
        
        ux = np.fft.ifftn(ux_hat).real
        uy = np.fft.ifftn(uy_hat).real
        vx = np.fft.ifftn(vx_hat).real
        vy = np.fft.ifftn(vy_hat).real
        
        # Nonlinear terms in physical space
        u_phys = np.fft.ifftn(u_hat).real
        v_phys = np.fft.ifftn(v_hat).real
        conv_u = u_phys * ux + v_phys * uy
        conv_v = u_phys * vx + v_phys * vy
        
        # Transform back to Fourier space
        conv_u_hat = np.fft.fftn(conv_u)
        conv_v_hat = np.fft.fftn(conv_v)
        
        # Viscous diffusion
        k2 = kx**2 + ky**2
        visc_u_hat = -nu * k2 * u_hat
        visc_v_hat = -nu * k2 * v_hat
        
        # RHS
        rhs_u_hat = -conv_u_hat + visc_u_hat
        rhs_v_hat = -conv_v_hat + visc_v_hat
        
        # Pressure projection (incompressible)
        div_rhs = 1j * (kx * rhs_u_hat + ky * rhs_v_hat)
        k2_safe = k2.copy()
        k2_safe[0, 0] = 1.0
        p_hat = div_rhs / k2_safe
        p_hat[0, 0] = 0
        
        rhs_u_hat = rhs_u_hat - 1j * kx * p_hat
        rhs_v_hat = rhs_v_hat - 1j * ky * p_hat
        
        return rhs_u_hat, rhs_v_hat
    
    # Stage 1
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    k1_u, k1_v = compute_rhs(u_hat, v_hat)
    
    u_hat_2 = u_hat + dt * k1_u
    v_hat_2 = v_hat + dt * k1_v
    
    # Stage 2
    k2_u, k2_v = compute_rhs(u_hat_2, v_hat_2)
    
    # Combine
    u_hat_new = u_hat + 0.5 * dt * (k1_u + k2_u)
    v_hat_new = v_hat + 0.5 * dt * (k1_v + k2_v)
    
    # Transform back
    u_new = np.fft.ifftn(u_hat_new).real
    v_new = np.fft.ifftn(v_hat_new).real
    
    return u_new, v_new


def generate_dataset(Re=1000, N=64, n_train=1000, n_val=200, n_test=200, 
                     n_warmup=500, dt=0.005, seed=42):
    """Generate complete dataset for given Reynolds number"""
    print(f"Generating dataset for Re={Re}, N={N}...")
    
    np.random.seed(seed)
    nu = 1.0 / Re
    
    # Setup wavenumbers
    L = 2 * np.pi
    k = np.fft.fftfreq(N, L/N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    
    # Generate initial field
    u, v = generate_initial_field(N, Re, seed)
    
    # Warmup to reach statistical steady state
    print(f"Warming up for {n_warmup} steps...")
    for _ in tqdm(range(n_warmup), desc="Warmup"):
        u, v = navier_stokes_rk2(u, v, nu, dt, kx, ky)
    
    # Collect data - define as separate function to avoid closure issues
    def collect_samples(n_samples, desc, u_init, v_init):
        inputs = []
        targets = []
        u, v = u_init.copy(), v_init.copy()
        
        for _ in tqdm(range(n_samples), desc=desc):
            inputs.append(np.stack([u.copy(), v.copy()], axis=0))
            
            u_new, v_new = navier_stokes_rk2(u, v, nu, dt, kx, ky)
            targets.append(np.stack([u_new.copy(), v_new.copy()], axis=0))
            
            u, v = u_new, v_new
        
        return np.array(inputs), np.array(targets), u, v
    
    print(f"Collecting training data...")
    train_input, train_target, u, v = collect_samples(n_train, "Train", u, v)
    
    print(f"Collecting validation data...")
    val_input, val_target, u, v = collect_samples(n_val, "Val", u, v)
    
    print(f"Collecting test data...")
    test_input, test_target, u, v = collect_samples(n_test, "Test", u, v)
    
    # Validation checks
    print("\nValidation checks:")
    
    # Check for NaN
    has_nan = np.isnan(train_input).any() or np.isnan(train_target).any()
    print(f"  NaN check: {'FAILED - contains NaN!' if has_nan else 'PASSED'}")
    
    if not has_nan:
        # Check divergence-free condition
        u_sample = train_input[0, 0]
        v_sample = train_input[0, 1]
        u_hat = np.fft.fftn(u_sample)
        v_hat = np.fft.fftn(v_sample)
        div_hat = 1j * (kx * u_hat + ky * v_hat)
        div = np.fft.ifftn(div_hat).real
        div_rms = np.sqrt(np.mean(div**2))
        print(f"  Divergence RMS: {div_rms:.2e} (should be < 1e-8)")
        
        # Check energy spectrum slope
        k_mag = np.sqrt(kx**2 + ky**2)
        spectrum = compute_energy_spectrum(u_hat, v_hat, k_mag, N)
        
        # Fit slope in inertial range (k=4 to k=10)
        k_indices = np.arange(4, min(11, len(spectrum)))
        if len(k_indices) > 2:
            spectrum_fit = spectrum[k_indices]
            mask = (k_indices > 0) & (spectrum_fit > 0)
            if mask.sum() > 2:
                log_k = np.log(k_indices[mask])
                log_E = np.log(spectrum_fit[mask])
                slope = np.polyfit(log_k, log_E, 1)[0]
                print(f"  Energy spectrum slope: {slope:.2f} (should be close to -5/3 ≈ -1.67)")
        
        # Check energy conservation
        E_initial = 0.5 * np.mean(train_input[0]**2)
        E_final = 0.5 * np.mean(train_target[-1]**2)
        print(f"  Energy: initial={E_initial:.4f}, final={E_final:.4f}, change={(E_final/E_initial-1)*100:.2f}%")
    
    return {
        'train': {'input': train_input, 'target': train_target},
        'val': {'input': val_input, 'target': val_target},
        'test': {'input': test_input, 'target': test_target},
        'metadata': {
            'Re': Re,
            'N': N,
            'dt': dt,
            'n_warmup': n_warmup,
            'nu': nu
        }
    }


def save_to_hdf5(data_dict, filename):
    """Save dataset to HDF5 file"""
    with h5py.File(filename, 'w') as f:
        # Save splits
        for split in ['train', 'val', 'test']:
            grp = f.create_group(split)
            grp.create_dataset('input', data=data_dict[split]['input'])
            grp.create_dataset('target', data=data_dict[split]['target'])
        
        # Save metadata
        meta_grp = f.create_group('metadata')
        for key, value in data_dict['metadata'].items():
            meta_grp.attrs[key] = value
    
    print(f"Saved to {filename}")


def main():
    # Generate Re=1000 dataset (training distribution)
    print("="*60)
    print("Generating Re=1000 dataset (training distribution)")
    print("="*60)
    data_re1000 = generate_dataset(Re=1000, N=64, n_train=1000, n_val=200, n_test=200,
                                   n_warmup=500, dt=0.01, seed=42)
    save_to_hdf5(data_re1000, 'turbulence_re1000_64x64.h5')
    
    print("\n" + "="*60)
    print("Generating Re=5000 dataset (OOD distribution)")
    print("="*60)
    data_re5000 = generate_dataset(Re=5000, N=64, n_train=200, n_val=50, n_test=200,
                                   n_warmup=500, dt=0.01, seed=123)
    save_to_hdf5(data_re5000, 'turbulence_re5000_64x64.h5')
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
