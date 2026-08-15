"""
Incremental 500-sample generation.
Builds on the validated experiment_v3_correct.h5 (100 samples).
Generates additional 400 train + 80 val samples.
"""
import numpy as np
import torch
import torch.fft as fft
import h5py
import os
from tqdm import tqdm

N = 64
Re = 1000
nu = 1.0 / Re
dt = 0.20
n_steps = 5
n_warmup = 300
output_path = 'data/turbulence_re1000_n64_500samples.h5'

# Check if we have existing valid data
existing_train = 0
existing_val = 0
if os.path.exists('data/experiment_v3_correct.h5'):
    with h5py.File('data/experiment_v3_correct.h5', 'r') as f:
        existing_train_in = f['train/input'][:]
        existing_train_tgt = f['train/target'][:]
        existing_val_in = f['val/input'][:]
        existing_val_tgt = f['val/target'][:]
    existing_train = len(existing_train_in)
    existing_val = len(existing_val_in)
    print(f"Found existing valid data: {existing_train} train, {existing_val} val")
else:
    existing_train_in = np.zeros((0, 3, N, N, N), dtype=np.float32)
    existing_train_tgt = np.zeros((0, 3, N, N, N), dtype=np.float32)
    existing_val_in = np.zeros((0, 3, N, N, N), dtype=np.float32)
    existing_val_tgt = np.zeros((0, 3, N, N, N), dtype=np.float32)

target_train = 500
target_val = 100
need_train = target_train - existing_train
need_val = target_val - existing_val

print(f"Need to generate: {need_train} train, {need_val} val")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

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


def generate_field(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    k_mag = torch.sqrt(k2)
    k0 = 4.0
    E_k = (k_mag / k0) ** 4 / (1 + (k_mag / k0) ** 2) ** (17 / 6)
    E_k[0, 0, 0] = 0
    amplitude = torch.sqrt(E_k / (4 * np.pi * k_mag ** 2 + 1e-10))
    amplitude[0, 0, 0] = 0
    phase_u = torch.rand_like(amplitude) * 2 * np.pi
    phase_v = torch.rand_like(amplitude) * 2 * np.pi
    phase_w = torch.rand_like(amplitude) * 2 * np.pi
    u_hat = torch.stack([
        amplitude * torch.exp(1j * phase_u),
        amplitude * torch.exp(1j * phase_v),
        amplitude * torch.exp(1j * phase_w),
    ], dim=0).to(device)
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    phi = div / k2_safe
    u_hat[0] -= kx * phi
    u_hat[1] -= ky * phi
    u_hat[2] -= kz * phi
    return u_hat


def generate_sample(seed):
    u_hat = generate_field(seed)
    for _ in range(n_warmup):
        u_hat = pseudo_spectral_step(u_hat, 0.001)
    u_input = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    for _ in range(n_steps):
        u_hat = pseudo_spectral_step(u_hat, dt)
    u_target = torch.stack([fft.irfftn(u_hat[i], s=(N, N, N)) for i in range(3)])
    return u_input.cpu().numpy(), u_target.cpu().numpy()


# Generate additional training samples
new_train_in = []
new_train_tgt = []
if need_train > 0:
    print(f"Generating {need_train} additional training samples...")
    for i in tqdm(range(need_train)):
        seed = existing_train + i
        inp, tgt = generate_sample(seed)
        new_train_in.append(inp)
        new_train_tgt.append(tgt)
    new_train_in = np.array(new_train_in, dtype=np.float32)
    new_train_tgt = np.array(new_train_tgt, dtype=np.float32)
else:
    new_train_in = np.zeros((0, 3, N, N, N), dtype=np.float32)
    new_train_tgt = np.zeros((0, 3, N, N, N), dtype=np.float32)

# Generate additional validation samples
new_val_in = []
new_val_tgt = []
if need_val > 0:
    print(f"Generating {need_val} additional validation samples...")
    for i in tqdm(range(need_val)):
        seed = target_train + existing_val + i
        inp, tgt = generate_sample(seed)
        new_val_in.append(inp)
        new_val_tgt.append(tgt)
    new_val_in = np.array(new_val_in, dtype=np.float32)
    new_val_tgt = np.array(new_val_tgt, dtype=np.float32)
else:
    new_val_in = np.zeros((0, 3, N, N, N), dtype=np.float32)
    new_val_tgt = np.zeros((0, 3, N, N, N), dtype=np.float32)

# Combine
all_train_in = np.concatenate([existing_train_in, new_train_in], axis=0)
all_train_tgt = np.concatenate([existing_train_tgt, new_train_tgt], axis=0)
all_val_in = np.concatenate([existing_val_in, new_val_in], axis=0)
all_val_tgt = np.concatenate([existing_val_tgt, new_val_tgt], axis=0)

# Normalize
all_data = np.concatenate([all_train_in, all_train_tgt, all_val_in, all_val_tgt], axis=0)
mean = all_data.mean()
std = all_data.std()

train_in_norm = (all_train_in - mean) / (std + 1e-8)
train_tgt_norm = (all_train_tgt - mean) / (std + 1e-8)
val_in_norm = (all_val_in - mean) / (std + 1e-8)
val_tgt_norm = (all_val_tgt - mean) / (std + 1e-8)

# Save
with h5py.File(output_path, 'w') as f:
    f.create_dataset('train/input', data=train_in_norm)
    f.create_dataset('train/target', data=train_tgt_norm)
    f.create_dataset('val/input', data=val_in_norm)
    f.create_dataset('val/target', data=val_tgt_norm)
    f.attrs['mean'] = float(mean)
    f.attrs['std'] = float(std)
    f.attrs['N'] = N
    f.attrs['Re'] = Re
    f.attrs['dt'] = dt
    f.attrs['n_steps'] = n_steps

print(f"\nSaved: {output_path}")
print(f"Train: {len(train_in_norm)}, Val: {len(val_in_norm)}")
print(f"Mean: {mean:.6e}, Std: {std:.6e}")

# Verify
print("\nVerification:")
corrs = [np.corrcoef(train_in_norm[i].flatten(), train_tgt_norm[i].flatten())[0, 1]
         for i in range(min(50, len(train_in_norm)))]
print(f"Correlation: {np.mean(corrs):.4f}")
