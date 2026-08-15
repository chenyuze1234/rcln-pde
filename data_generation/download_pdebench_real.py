"""
Download PDEBench 2D CFD real data from Zenodo.
"""
import os
import requests
from pathlib import Path

URL = "https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5"
SAVE_PATH = Path("D:/PDEBench_Data/2D/CFD/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5")

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

def download():
    print(f"Downloading PDEBench 2D CFD data...")
    print(f"URL: {URL}")
    print(f"Save to: {SAVE_PATH}")
    print(f"Expected size: ~2.3 GB")
    print("=" * 60)
    
    if SAVE_PATH.exists():
        existing_size = SAVE_PATH.stat().st_size
        print(f"File exists, size: {existing_size / 1024**3:.2f} GB")
        if existing_size > 2 * 1024**3:
            print("File appears complete. Skipping download.")
            return True
        print("Resuming download...")
        headers = {"Range": f"bytes={existing_size}-"}
    else:
        existing_size = 0
        headers = {}
    
    response = requests.get(URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0)) + existing_size
    mode = 'ab' if existing_size > 0 else 'wb'
    
    downloaded = existing_size
    chunk_size = 1024 * 1024  # 1MB
    
    with open(SAVE_PATH, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) == 0:  # every 50MB
                    pct = (downloaded / total_size * 100) if total_size else 0
                    print(f"  Downloaded: {downloaded/1024**3:.2f} GB / {total_size/1024**3:.2f} GB ({pct:.1f}%)")
    
    print(f"\n[OK] Download complete: {SAVE_PATH}")
    print(f"Final size: {SAVE_PATH.stat().st_size / 1024**3:.2f} GB")
    return True

if __name__ == '__main__':
    try:
        download()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
