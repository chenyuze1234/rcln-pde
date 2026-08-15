"""
JHTDB Downloader v3 — giverny/REST API (2026-08)
=================================================
Downloads 3D velocity cutouts from JHTDB via the official givernylocal
client (the old /cutout HTTP endpoint and the legacy SOAP service are dead).

Output: one HDF5 per spatial position, key 'fields' [T,3,64,64,64],
matching FlowSequenceDataset / FlowDataset in train_high_diversity_v5.py.

Usage:
    python scripts/data_generation/jhtdb_download_giverny.py \
        --token <JHTDB_TOKEN> --preset small

Env fallback: JHTDB_TOKEN.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

# giverny loads its dataset config from raw.githubusercontent.com, which is
# unreachable from some networks. A local copy was saved on 2026-08-08;
# the ghproxy mirror works as a live fallback.
LOCAL_CONFIG = 'data/downloaded/jhtdb_config.json'
CONFIG_URLS = [
    'https://ghproxy.net/https://raw.githubusercontent.com/sciserver/giverny/main/metadata/configs/jhtdb_config.json',
    'https://raw.githubusercontent.com/sciserver/giverny/main/metadata/configs/jhtdb_config.json',
]

DATASET = 'isotropic1024coarse'   # Reλ=433 forced isotropic DNS, 1024³, t∈[0,1023]

# Presets: (positions, n_frames, frame_step)
#   small: pipeline validation (~6 files, ~40 frames each)
#   medium: real training run (~16 files)
PRESETS = {
    'small': dict(
        positions=[(100, 100, 100), (300, 200, 400), (500, 400, 200),
                   (700, 300, 600), (200, 600, 300), (400, 800, 500)],
        n_frames=40, frame_step=10, frame_start=0),
    'medium': dict(
        positions=[(100, 100, 100), (300, 200, 400), (500, 400, 200),
                   (700, 300, 600), (200, 600, 300), (400, 800, 500),
                   (600, 500, 100), (800, 700, 300), (150, 450, 700),
                   (350, 650, 150), (550, 100, 550), (250, 350, 800),
                   (750, 850, 250), (50, 250, 450), (450, 550, 750),
                   (650, 750, 50)],
        n_frames=100, frame_step=5, frame_start=0),
}


def probe_service(token: str) -> bool:
    """One tiny authenticated request; True iff the cutout backend is up."""
    import requests
    url = 'https://web.idies.jhu.edu/turbulence-svc/cutout/api/local'
    params = dict(token=token, function='velocity', dataset=DATASET,
                  xs=100, xe=101, ys=100, ye=101, zs=100, ze=101, ts=1, te=1,
                  stridet=1, stridex=1, stridey=1, stridez=1, filter_width=1)
    try:
        r = requests.get(url, params=params, timeout=60)
        print(f'probe: HTTP {r.status_code}')
        return r.status_code == 200
    except Exception as e:
        print(f'probe: {type(e).__name__} {str(e)[:80]}')
        return False


def make_cube(token: str):
    from givernylocal.turbulence_dataset import turb_dataset
    if os.path.exists(LOCAL_CONFIG):
        # requests cannot read file:// — serve the local copy over an in-process
        # URL is overkill; giverny accepts any URL, so point it at a data: URI is
        # not supported either. Simplest: monkeypatch requests.get for the config.
        import requests
        _orig_get = requests.get
        local_text = open(LOCAL_CONFIG, encoding='utf-8').read()

        def patched(url, *a, **kw):
            if 'jhtdb_config.json' in url:
                class R:
                    text = local_text
                    status_code = 200
                    def raise_for_status(self): pass
                return R()
            return _orig_get(url, *a, **kw)
        requests.get = patched
    return turb_dataset(dataset_title=DATASET,
                        output_path='data/downloaded/jhtdb_cutouts',
                        auth_token=token,
                        json_url=CONFIG_URLS[0])


def fetch_cube(cube, pos, t_idx, size=64, retries=4):
    """One [3,size,size,size] velocity cutout at grid position pos, time t_idx."""
    from givernylocal.turbulence_toolkit import getCutout
    x0, y0, z0 = pos
    axes = np.array([[x0, x0 + size - 1], [y0, y0 + size - 1],
                     [z0, z0 + size - 1], [t_idx, t_idx]])
    strides = np.array([1, 1, 1, 1])
    for attempt in range(retries):
        try:
            out = getCutout(cube, 'velocity', axes, strides, verbose=False)
            arr = out[0] if isinstance(out, tuple) else out
            if hasattr(arr, 'values'):
                arr = arr.values
            arr = np.asarray(arr, dtype=np.float32)
            # normalize to [3, size, size, size]
            if arr.shape == (size, size, size, 3):
                arr = arr.transpose(3, 0, 1, 2)
            elif arr.shape != (3, size, size, size):
                arr = arr.reshape(3, size, size, size)
            return arr
        except Exception as e:
            print(f'    attempt {attempt+1}/{retries} fail: {str(e)[:100]}')
            time.sleep(5 * (attempt + 1))
    return None


def download_position(cube, pos, out_path, n_frames, frame_start, frame_step):
    fields, times = [], []
    t0 = time.time()
    for i in range(n_frames):
        t_idx = frame_start + i * frame_step
        if t_idx > 1023:
            print(f'  t_idx={t_idx} > 1023, stop')
            break
        arr = fetch_cube(cube, pos, t_idx)
        if arr is None:
            print(f'  [{i+1}/{n_frames}] t={t_idx} FAILED, skipping')
            continue
        fields.append(arr)
        times.append(float(t_idx))
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - t0) / (i + 1)
            eta = rate * (n_frames - i - 1)
            print(f'  [{i+1}/{n_frames}] t={t_idx} range=[{arr.min():+.3f},{arr.max():+.3f}] '
                  f'elapsed={time.time()-t0:.0f}s ETA={eta:.0f}s')
    if not fields:
        return False
    fields = np.stack(fields, 0)  # [T,3,64,64,64]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip', compression_opts=4)
        f.create_dataset('times', data=np.array(times))
        f.attrs.update(dataset=DATASET, source='JHTDB via givernylocal',
                       Re_lambda=433, grid_size_source=1024, cutout_size=64,
                       position=list(pos), frame_start=frame_start, frame_step=frame_step)
    print(f'  saved {out_path} {fields.shape}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token', default=os.environ.get('JHTDB_TOKEN', ''))
    ap.add_argument('--preset', choices=list(PRESETS), default='small')
    ap.add_argument('--output_dir', default='data_generated')
    ap.add_argument('--probe_only', action='store_true',
                    help='only check whether the cutout service is back up')
    args = ap.parse_args()
    if not args.token:
        sys.exit('token required (--token or JHTDB_TOKEN)')

    if args.probe_only:
        ok = probe_service(args.token)
        print('SERVICE_UP' if ok else 'SERVICE_DOWN')
        sys.exit(0 if ok else 1)

    cfg = PRESETS[args.preset]
    print(f'preset={args.preset}: {len(cfg["positions"])} positions x '
          f'{cfg["n_frames"]} frames (step {cfg["frame_step"]})')
    cube = make_cube(args.token)
    n_ok = 0
    for pos in cfg['positions']:
        out_path = os.path.join(
            args.output_dir,
            f'jhtdb_iso433_N64_x{pos[0]}y{pos[1]}z{pos[2]}_T{cfg["n_frames"]}.h5')
        if os.path.exists(out_path):
            print(f'{out_path} exists, skip')
            n_ok += 1
            continue
        print(f'position {pos}:')
        if download_position(cube, pos, out_path,
                             cfg['n_frames'], cfg['frame_start'], cfg['frame_step']):
            n_ok += 1
    print(f'DONE: {n_ok}/{len(cfg["positions"])} position files')


if __name__ == '__main__':
    main()
