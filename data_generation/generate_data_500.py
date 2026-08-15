"""
Generate CORRECT 500-sample 3D turbulence dataset.
Based on validated generate_v3_correct.py parameters.

dt = 0.20, n_steps = 5  =>  correlation ~0.93
dt = 0.01, n_steps = 5  =>  correlation ~0.999 (TOO SMALL)
"""

import os
import time
import h5py
import numpy as np
import torch
import torch.fft as fft
from tqdm import tqdm

N = 64
Re = 1000
nu = 1.0 / Re
dt = 0.20
n_steps = 5
n_warmup = 300
n_train = 500
n_val = 100
output_path = 'data/turbulence_re1000_n64_500samples.h5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Generating {n_train} train + {n_val} val samples...")
print(f"dt={dt}, n_steps={n_steps}, total_evolution={dt*n_steps:.2f}")
print(f"warmup={n_warmup}, Re={Re}, N={N}\n")

# Precompute wavenumbers
kx_1d = torch.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
kz_1d = torch.fft.rfftfreq(N, d=1.0/N) * 2 * np.pi
kx, ky, kz = torch.meshgrid(kx_1d, kx_1d, kz_1d, indexing='ij')
k2 = kx**2 + ky**2 + kz**2
kx = kx.to(device)
ky = ky.to(device)
kz = kz.to(device)
k2 = k2.to(device)
k2_safe = k2.clone()
k2_safe[0, 0, 0] = 1.0


def pseudo_spectral_step(u_hat, dt_step):
    u = fft.irfftn(u_hat[0], s=(N, N, N))
    v = fft.irfftn(u_hat[1], s=(N, N, N))
    w = fft.irfftn(u_hat[2], s=(N, N, N))

    dudx = fft.irfftn(1j * kx * u_hat[0], s=(N, N, N))
    dudy = fft.irfftn(1j * ky * u_hat[0], s=(N, N, N))
    dudz = fft.irfftn(1j * kz * u_hat[0], s=(N, N, N))
    dvdx = fft.irfftn(1j * kx * u_hat[1], s=(N, N, N))
    dvdy = fft.irfftn(1j * ky * u_hat[1], s=(N, N, N))
    dvdz = fft.irfftn(1j * kz * u_hat[1], s=(N, N, N))
    dwdx = fft.irfftn(1j * kx * u_hat[2], s=(N, N, N))
    dwdy = fft.irfftn(1j * ky * u_hat[2], s=(N, N, N))
    dwdz = fft.irfftn(1j * kz * u_hat[2], s=(N, N, N))

    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz

    conv_u_hat = fft.rfftn(conv_u)
    conv_v_hat = fft.rfftn(conv_v)
    conv_w_hat = fft.rfftn(conv_w)

    u_hat_new = torch.stack([
        u_hat[0] - dt_step * (conv_u_hat + nu * k2 * u_hat[0]),
        u_hat[1] - dt_step * (conv_v_hat + nu * k2 * u_hat[1]),
        u_hat[2] - dt_step * (conv_w_hat + nu * k2 * u_hat[2]),
    ], dim=0)

    div = kx * u_hat_new[0] + ky * u_hat_new[1] + kz * u_hat_new[2]
    phi = div / k2_safe
    u_hat_new[0] -= kx * phi
    u_hat_new[1] -= ky * phi
    u_hat_new[2] -= kz * phi

    return u_hat_new


def generate_initial_field(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    k_mag = torch.sqrt(k2)
    k0 = 4.0
    E_k = (k_mag / k0) ** 4 / (1 + (k_mag / k0) ** 2) ** (17 / 6)
    E_k[0, 0, 0] = 0

    amplitude = torch.sqrt(E_k / (4 * np.pi * k_mag ** 2 + 1e-10))
    amplitude[0, 0, 0] = 0

    u_hat = torch.stack([
        amplitude * torch.exp(1j * torch.rand_like(amplitude) * 2 * np.pi),
        amplitude * torch.exp(1j * torch.rand_like(amplitude) * 2 * np.pi),
        amplitude * torch.exp(1j * torch.rand_like(amplitude) * 2 * np.pi),
    ], dim=0).to(device)

    # Project to divergence-free (NO energy normalization - keep natural scale)
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    phi = div / k2_safe
    u_hat[0] -= kx * phi
    u_hat[1] -= ky * phi
    u_hat[2] -= kz * phi

    return u_hat


def generate_sample(seed):
    u_hat = generate_initial_field(seed)

    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, dt_step=0.001)
        if not torch.isfinite(u_hat).all():
            raise ValueError(f"NaN during warmup at seed {seed}")

    u_input = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])

    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, dt_step=dt)
        if not torch.isfinite(u_hat).all():
            raise ValueError(f"NaN during evolution at seed {seed}")

    u_target = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])

    return u_input.cpu().numpy(), u_target.cpu().numpy()


def main():
    os.makedirs('data', exist_ok=True)
    start_time = time.time()

    train_inputs = np.zeros((n_train, 3, N, N, N), dtype=np.float32)
    train_targets = np.zeros((n_train, 3, N, N, N), dtype=np.float32)

    print(f"[1/3] Generating {n_train} training samples...")
    for i in tqdm(range(n_train)):
        inp, tgt = generate_sample(i)
        train_inputs[i] = inp
        train_targets[i] = tgt

    print(f"\n[2/3] Generating {n_val} validation samples...")
    val_inputs = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
    val_targets = np.zeros((n_val, 3, N, N, N), dtype=np.float32)
    for i in tqdm(range(n_val)):
        inp, tgt = generate_sample(n_train + i)
        val_inputs[i] = inp
        val_targets[i] = tgt

    print(f"\n[3/3] Computing statistics...")
    all_data = np.concatenate([train_inputs, train_targets, val_inputs, val_targets], axis=0)
    mean = all_data.mean()
    std = all_data.std()

    print(f"  Raw mean: {mean:.6e}, raw std: {std:.6e}")

    # Save raw data (training code will normalize)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        f.attrs['mean'] = float(mean)
        f.attrs['std'] = float(std)
        f.attrs['N'] = N
        f.attrs['Re'] = Re
        f.attrs['dt'] = dt
        f.attrs['n_steps'] = n_steps

    elapsed = time.time() - start_time
    print(f"\nSaved: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1e9:.2f} GB")
    print(f"Time: {elapsed / 60:.1f} min")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    with h5py.File(output_path, 'r') as f:
        inp = f['train/input'][:]
        tgt = f['train/target'][:]

        corrs = [np.corrcoef(inp[i].flatten(), tgt[i].flatten())[0, 1]
                 for i in range(min(50, len(inp)))]
        mean_corr = np.mean(corrs)
        print(f"\nMean correlation (50 samples): {mean_corr:.4f}")

        div_rms = []
        for i in range(min(10, len(inp))):
            u = inp[i]
            du_dx = (u[0, 2:, :, :] - u[0, :-2, :, :]) / 2.0
            dv_dy = (u[1, :, 2:, :] - u[1, :, :-2, :]) / 2.0
            dw_dz = (u[2, :, :, 2:] - u[2, :, :, :-2]) / 2.0
            div = du_dx[:, :-2, :-2] + dv_dy[:-2, :, :-2] + dw_dz[:-2, :-2, :]
            div_rms.append(np.sqrt(np.mean(div ** 2)))
        print(f"Mean divergence RMS: {np.mean(div_rms):.4e}")

        E_in = 0.5 * np.mean(inp ** 2, axis=(1, 2, 3, 4))
        E_tgt = 0.5 * np.mean(tgt ** 2, axis=(1, 2, 3, 4))
        ratio = np.mean(E_tgt / (E_in + 1e-10))
        print(f"Mean energy ratio (target/input): {ratio:.4f}")

        print("\n" + "-" * 60)
        if 0.88 <= mean_corr <= 0.97:
            print(f"  Correlation {mean_corr:.3f}: PASS")
        elif mean_corr > 0.97:
            print(f"  Correlation {mean_corr:.3f}: TOO HIGH")
        else:
            print(f"  Correlation {mean_corr:.3f}: TOO LOW")

        if np.mean(div_rms) < 1e-4:
            print(f"  Divergence: PASS")
        else:
            print(f"  Divergence: FAIL")
        print("-" * 60)

    # Cleanup old temp files
    for f in os.listdir('data'):
        if f.startswith('turbulence_re1000_n64_500samples_temp_'):
            os.remove(os.path.join('data', f))
            print(f"Cleaned temp: {f}")


if __name__ == '__main__':
    main()
