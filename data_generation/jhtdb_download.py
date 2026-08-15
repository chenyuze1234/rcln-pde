"""
JHTDB Isotropic Turbulence Data Downloader (v2 — HTTP API)
============================================================
Downloads 3D velocity cutouts from JHTDB isotropic1024coarse using HTTP API.
No Python library dependencies — uses urllib directly.
Output: HDF5 with fields [T, 3, H, W, D] matching RCLN-UPI v5 format.

JHTDB isotropic1024coarse:
  Reλ = 433 (strong forced isotropic turbulence)
  1024^3 DNS, 1024 time frames
  Variables: 3 velocity components + pressure
  Free token: https://turbulence.pha.jhu.edu/authtoken.aspx
"""

import urllib.request
import urllib.error
import urllib.parse
import ssl
import numpy as np
import h5py
import os
import time
import sys
import json

# Windows/AV MITM networks break the cert chain ("self-signed certificate in
# certificate chain"). Build a tolerant SSL context for the JHTDB endpoint.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def get_velocity_cutout(
    token: str,
    t_idx: int,
    size: int = 64,
    x0: int = 0, y0: int = 0, z0: int = 0,
    dataset: str = "isotropic1024coarse",
    retries: int = 3,
) -> np.ndarray:
    """
    Download a single 3D velocity cutout from JHTDB.

    Args:
        token: JHTDB auth token
        t_idx: Time frame index (0-1023)
        size: Cutout size in each dimension
        x0, y0, z0: Starting coordinates
        dataset: JHTDB dataset name
        retries: Number of retry attempts

    Returns:
        velocity: numpy array shape (3, size, size, size), float32
            velocity[0] = x-component, velocity[1] = y, velocity[2] = z
    """
    # JHTDB web cutout service endpoint
    base_url = "https://turbulence.idies.jhu.edu/cutout"

    params = {
        'dataset': dataset,
        'get': 'Velocity',
        'x_start': x0, 'x_end': x0 + size,
        'y_start': y0, 'y_end': y0 + size,
        'z_start': z0, 'z_end': z0 + size,
        'x_step': 1, 'y_step': 1, 'z_step': 1,
        'time': t_idx,
        'authToken': token,
        'format': 'hdf5',
    }

    url = base_url + '?' + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120, context=_SSL_CTX) as resp:
                data = resp.read()

            # Parse HDF5 from bytes
            import io
            with h5py.File(io.BytesIO(data), 'r') as f:
                # JHTDB returns velocity as dataset named 'Velocity'
                # shape: (3, z, y, x) or (3, x, y, z) depending on version
                vel = f['Velocity'][()]

            # Standardize to (3, size, size, size) in (x, y, z) order
            # JHTDB typically returns (3, z, y, x) — transpose to (3, x, y, z)
            if vel.shape != (3, size, size, size):
                if vel.shape == (3, size, size, size):
                    pass
                elif len(vel.shape) == 4:
                    # Try common orderings
                    if vel.shape[3] == size and vel.shape[1] == size:
                        vel = vel.transpose(0, 3, 2, 1)  # (3, z, y, x) → (3, x, y, z)
                    elif vel.shape[1] == size and vel.shape[2] == size and vel.shape[3] == size:
                        pass  # already (3, x, y, z)

            return vel.astype(np.float32)

        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise ValueError(
                    f"Authentication failed. Get free token at:\n"
                    f"  https://turbulence.pha.jhu.edu/authtoken.aspx\n"
                    f"Current token: {token[:8]}..."
                )
            if attempt < retries - 1:
                print(f"    HTTP {e.code}, retrying ({attempt+2}/{retries})...")
                time.sleep(2)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                print(f"    {e}, retrying ({attempt+2}/{retries})...")
                time.sleep(2)
            else:
                raise

    raise RuntimeError(f"Failed after {retries} retries")


def download_full(
    output_path: str,
    token: str,
    spatial_size: int = 64,
    n_frames: int = 200,
    frame_start: int = 0,
    frame_step: int = 5,
    x0: int = 100,
    y0: int = 100,
    z0: int = 100,
    dataset: str = "isotropic1024coarse",
):
    """
    Download multiple 3D velocity cutouts.

    Args:
        output_path: Output HDF5 file path
        token: JHTDB auth token
        spatial_size: Spatial cutout size (cube side, default 64)
        n_frames: Number of frames
        frame_start: First frame index (0-1023)
        frame_step: Step between frames (δt per frame ≈ 0.002)
        x0, y0, z0: Starting position in 1024^3 grid
        dataset: JHTDB dataset name
    """
    print(f"{'='*60}")
    print("JHTDB Isotropic Turbulence Downloader")
    print(f"{'='*60}")
    print(f"Dataset:  {dataset}")
    print(f"Size:     {spatial_size}^3")
    print(f"Frames:   {n_frames} (start={frame_start}, step={frame_step})")
    print(f"Position: ({x0}, {y0}, {z0}) in 1024^3 grid")
    print(f"Output:   {output_path}")
    print(f"Volume:   {n_frames} x {spatial_size}^3 = {n_frames * spatial_size**3:,} pts")
    print(f"{'='*60}")

    fields = []
    times_used = []
    n_failures = 0
    t0 = time.time()

    for i in range(n_frames):
        t_idx = frame_start + i * frame_step
        if t_idx >= 1024:
            print(f"  Frame {i}: t_idx={t_idx} exceeds max (1023), stopping.")
            break

        try:
            vel = get_velocity_cutout(
                token, t_idx=t_idx, size=spatial_size,
                x0=x0, y0=y0, z0=z0, dataset=dataset,
            )
            fields.append(vel)
            times_used.append(float(t_idx))

            # Progress
            elapsed = time.time() - t0
            rate = elapsed / max(1, i + 1)
            eta = rate * (n_frames - i - 1)
            print(f"  [{i+1:3d}/{n_frames}] t={t_idx:4d}  "
                  f"shape={vel.shape}  range=[{vel.min():+.3f}, {vel.max():+.3f}]  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

        except Exception as e:
            n_failures += 1
            print(f"  [{i+1:3d}/{n_frames}] t={t_idx:4d}  FAIL: {e}")
            if n_failures > 5:
                print(f"  Too many failures ({n_failures}), aborting.")
                break
            continue

    if not fields:
        raise RuntimeError("No data downloaded! Check token and network.")

    # Stack
    fields = np.stack(fields, axis=0)  # [T, 3, H, W, D]
    times_used = np.array(times_used)

    print(f"\nDownloaded: {fields.shape}, {fields.nbytes / 1e9:.2f} GB")
    print(f"Range:      [{fields.min():.4f}, {fields.max():.4f}]")
    print(f"Mean±std:   {fields.mean():.6f} ± {fields.std():.6f}")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
        f.create_dataset('times', data=times_used)
        f.attrs['dataset'] = dataset
        f.attrs['source'] = 'JHTDB isotropic turbulence DNS (HTTP API)'
        f.attrs['Re_lambda'] = 433
        f.attrs['grid_size_source'] = 1024
        f.attrs['cutout_size'] = spatial_size
        f.attrs['n_frames'] = len(fields)
        f.attrs['frame_start'] = frame_start
        f.attrs['frame_step'] = frame_step
        f.attrs['position'] = [x0, y0, z0]

    total_time = (time.time() - t0) / 60
    print(f"Saved: {output_path} ({total_time:.1f} min)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download JHTDB turbulence data')
    parser.add_argument('--token', type=str, default=None,
                       help='JHTDB auth token (free: https://turbulence.pha.jhu.edu/authtoken.aspx)')
    parser.add_argument('--output', type=str,
                       default='data_generated/jhtdb_isotropic_N64_T200.h5')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--n_frames', type=int, default=200)
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_step', type=int, default=5)
    parser.add_argument('--x0', type=int, default=100)
    parser.add_argument('--y0', type=int, default=100)
    parser.add_argument('--z0', type=int, default=100)
    args = parser.parse_args()

    token = args.token or os.environ.get('JHTDB_TOKEN', '')
    if not token:
        print("=" * 60)
        print("JHTDB AUTH TOKEN REQUIRED (free, 30 seconds)")
        print("=" * 60)
        print()
        print("1. Go to: https://turbulence.pha.jhu.edu/authtoken.aspx")
        print("2. Enter your email, click Submit")
        print("3. Copy the token from the page")
        print("4. Run:")
        print(f"   set JHTDB_TOKEN=your-token-here")
        print(f"   python scripts/data_generation/jhtdb_download.py")
        print()
        print("Or pass directly:")
        print(f"   python scripts/data_generation/jhtdb_download.py --token your-token-here")
        sys.exit(1)

    download_full(
        output_path=args.output,
        token=token,
        spatial_size=args.size,
        n_frames=args.n_frames,
        frame_start=args.frame_start,
        frame_step=args.frame_step,
        x0=args.x0,
        y0=args.y0,
        z0=args.z0,
    )
