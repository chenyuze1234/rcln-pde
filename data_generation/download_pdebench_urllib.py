"""
Download PDEBench 2D CFD with urllib + SSL workaround + retry logic.
"""
import os
import ssl
import urllib.request
import urllib.error
from pathlib import Path
import time

URL = "https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5"
SAVE_PATH = Path("D:/PDEBench_Data/2D/CFD/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create unverified SSL context (workaround for SSL issues)
ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def download_with_retry(max_retries=5):
    existing_size = SAVE_PATH.stat().st_size if SAVE_PATH.exists() else 0
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            req = urllib.request.Request(URL)
            if existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")
                print(f"  Resuming from {existing_size / 1024**2:.1f} MB")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=120) as response:
                total_size = int(response.headers.get('Content-Length', 0)) + existing_size
                mode = 'ab' if existing_size > 0 else 'wb'
                
                downloaded = existing_size
                chunk_size = 1024 * 1024  # 1MB
                
                with open(SAVE_PATH, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (50 * 1024 * 1024) == 0:
                            pct = (downloaded / total_size * 100) if total_size else 0
                            print(f"  {downloaded/1024**3:.2f} GB / {total_size/1024**3:.2f} GB ({pct:.1f}%)")
                
                print(f"[OK] Download complete: {downloaded/1024**3:.2f} GB")
                return True
                
        except Exception as e:
            print(f"  Error: {e}")
            existing_size = SAVE_PATH.stat().st_size if SAVE_PATH.exists() else 0
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("[FAILED] All retries exhausted.")
                return False
    
    return False

if __name__ == '__main__':
    print("Downloading PDEBench 2D CFD (urllib + SSL workaround)...")
    print(f"Save to: {SAVE_PATH}")
    print("=" * 60)
    success = download_with_retry()
    if not success:
        print("\n[SUGGESTION] Zenodo download unstable. Consider:")
        print("  1. Manual download from browser: https://zenodo.org/record/6993294")
        print("  2. Use OpenXLab mirror: OpenDataLab/Cylinder_in_Crossflow")
        exit(1)
