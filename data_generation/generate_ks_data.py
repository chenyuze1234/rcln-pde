"""
KS: u_t = -u*u_x - u_{xx} - u_{xxxx}, L=22, N=64, chaotic regime.
ETDRK4: linear part exact, nonlinear explicit. dt=0.001, 10000 steps.
"""

import numpy as np, h5py, os

L, N, dt, T = 22.0, 64, 0.001, 10.0
steps, warmup = int(T/dt), 2000
k = 2j * np.pi * np.fft.fftfreq(N, d=L/N)
Lop = -k**2 - k**4  # linear: -u_xx - u_{xxxx}

np.random.seed(42)
x = np.linspace(0, L, N, endpoint=False)
u = 0.5*(np.cos(2*np.pi*x/L + 0.3) + np.cos(4*np.pi*x/L - 0.5) +
         0.3*np.sin(6*np.pi*x/L + 0.7) + 0.2*np.cos(8*np.pi*x/L))

print(f'KS L={L} N={N} dt={dt} steps={steps} warmup={warmup}')

def rhs_nl(u):
    """Nonlinear: -u*u_x. Spectrally accurate."""
    u_hat = np.fft.fft(u)
    u_x = np.fft.ifft(k * u_hat).real
    return -u * u_x

def etdrk4_step(u):
    u_hat = np.fft.fft(u)
    E  = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)

    # Stage 1
    N_a = rhs_nl(u)
    N_a_hat = np.fft.fft(N_a)
    a_hat = u_hat * E2 + N_a_hat * (E2 - 1.0) / (Lop + 1e-12)

    # Stage 2
    a = np.fft.ifft(a_hat).real
    N_b = rhs_nl(a)
    N_b_hat = np.fft.fft(N_b)
    b_hat = u_hat * E2 + N_b_hat * (E2 - 1.0) / (Lop + 1e-12)

    # Stage 3
    b = np.fft.ifft(b_hat).real
    N_c = rhs_nl(b)
    N_c_hat = np.fft.fft(N_c)
    c_hat = a_hat * E2 + (2.0*N_c_hat - N_a_hat) * (E2 - 1.0) / (Lop + 1e-12)

    # Stage 4
    c = np.fft.ifft(c_hat).real
    N_d = rhs_nl(c)
    N_d_hat = np.fft.fft(N_d)

    u_new_hat = u_hat * E + (
        N_a_hat * (-4.0 - Lop*dt + E*(4.0 - 3.0*Lop*dt + (Lop*dt)**2)) +
        2.0 * (N_b_hat + N_c_hat) * (2.0 + Lop*dt + E*(-2.0 + Lop*dt)) +
        N_d_hat * (-4.0 - 3.0*Lop*dt - (Lop*dt)**2 + E*(4.0 - Lop*dt))
    ) / (Lop**3 * dt**2 + 1e-12)

    # Handle k=0 mode separately
    u_hat_zero = u_hat[0]
    dt_N_avg = dt * (N_a_hat[0] + 2*N_b_hat[0] + 2*N_c_hat[0] + N_d_hat[0]) / 6.0
    u_new_hat[0] = u_hat_zero + dt_N_avg

    return np.fft.ifft(u_new_hat).real

# Warmup
for s in range(warmup):
    u = etdrk4_step(u)
    if s % 500 == 0:
        print(f'  warmup {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')
        if not np.isfinite(u).all(): print(f'BLOW-UP!'); exit(1)

# Production — sample every 100 solver steps (dt_effective = 0.1)
sample_every = 100
fields_raw = []
for s in range(steps):
    u = etdrk4_step(u)
    if s % sample_every == 0:
        fields_raw.append(u.astype(np.float32))
    if s % 2000 == 0:
        print(f'  step {s}: umax={np.abs(u).max():.3f} rms={np.std(u):.4f}')

fields = np.stack(fields_raw, axis=0)  # [T_eff, N]
print(f'\n{fields.shape} {fields.nbytes/1e6:.1f}MB [{fields.min():.3f},{fields.max():.3f}] RMS={np.std(fields):.4f}')

os.makedirs('data_generated', exist_ok=True)
p = f'data_generated/ks_L{L:.0f}_N{N}_T{len(fields)*dt*sample_every:.1f}.h5'
with h5py.File(p, 'w') as f:
    f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
    f.create_dataset('times', data=np.arange(len(fields))*dt*sample_every)
    f.attrs['equation']='KS'; f.attrs['L']=L; f.attrs['N']=N; f.attrs['dt_effective']=dt*sample_every
print(f'Saved: {p}')
