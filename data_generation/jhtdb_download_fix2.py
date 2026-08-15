"""JHTDB 下载器（v5-fix2 训练数据）——SOAP GetAnyCutoutWeb、串行、可续传

数据集: isotropic1024coarse (Reλ≈433, 1024³ DNS, 1024 时间帧, δt_frame≈0.002)
通道: TurbulenceService SOAP (https://turbulence.pha.jhu.edu/service/turbulence.asmx)
  - 旧 HTTP cutout 端点已转为 SciServer 登录制；pyJHTDB/turblib SOAP 因
    HTTP→HTTPS 重定向失效，zeep + HTTPS 为官方建议的替代路径
  - GetAnyCutoutWeb: field='u', **1-based 索引**（JHTDB 2023-09 起）,
    返回 base64 二进制 float32，布局 [z][y][x][component]（分量交错，
    已由散度判据验证: interleaved div_rms=0.085 vs component-major 1.09）
实测吞吐 ~7-12 KB/s（用户网络到 JHU 链路瓶颈），单帧 64³≈270s。
规模: 4 原点 × 30 帧 × frame_step=5（有效 dt≈0.01, rollout 8 步=0.08）
  = 120 查询 ≈ 9h（后台可续传，逐帧缓存）
礼仪: 严格串行、查询间隔 0.5s、失败重试间隔 ≥5s（指数退避）。
输出: data_generated/jhtdb_iso_re433_N64_traj{00..03}.h5
  fields [T,3,64,64,64]（FlowSequenceDataset 直接可读）。
"""
import os, sys, time
import numpy as np
import h5py

import urllib3
urllib3.disable_warnings()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN_PATH = os.path.join(ROOT, "jhtdb_token.txt")
CACHE_DIR = os.path.join(ROOT, "data", "downloaded", "jhtdb_fix2_cache")
OUT_DIR = os.path.join(ROOT, "data_generated")

DATASET = "isotropic1024coarse"
WSDL = "https://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL"
SIZE = 64
N_FRAMES = 30
FRAME_START = 1       # 1-based
FRAME_STEP = 5        # δt_frame≈0.002 → 有效 dt≈0.01
ORIGINS = [           # 4 个散布原点（0-based 逻辑坐标；查询时 +1 转 1-based）
    (64, 64, 64),
    (512, 64, 512),
    (64, 512, 512),
    (512, 512, 64),
]
QUERY_GAP = 0.5
RETRY_GAP = 5.0
RETRIES = 4


def load_token():
    with open(TOKEN_PATH) as f:
        return f.read().strip()


def make_client():
    import zeep
    from zeep.transports import Transport
    from requests import Session
    s = Session()
    s.verify = False
    return zeep.Client(WSDL, transport=Transport(session=s, timeout=600))


def fetch_frame(client, token, t_idx, origin, traj_idx):
    """带缓存与礼仪重试的单帧下载（1-based SOAP）"""
    cache = os.path.join(CACHE_DIR, f"traj{traj_idx:02d}")
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, f"t{t_idx:04d}.npy")
    if os.path.exists(fp):
        return np.load(fp)
    x0, y0, z0 = origin[0] + 1, origin[1] + 1, origin[2] + 1  # → 1-based
    last_err = None
    for attempt in range(RETRIES):
        try:
            t0 = time.time()
            r = client.service.GetAnyCutoutWeb(
                authToken=token, dataset=DATASET, field='u', T=t_idx,
                x_start=x0, y_start=y0, z_start=z0,
                x_end=x0 + SIZE - 1, y_end=y0 + SIZE - 1, z_end=z0 + SIZE - 1,
                x_step=1, y_step=1, z_step=1, filter_width=1, addr='')
            arr = np.frombuffer(r, dtype=np.float32)
            assert arr.size == 3 * SIZE ** 3, f"bad size {arr.size}"
            # 布局 [z][y][x][component] → [3,64,64,64]
            vel = arr.reshape(SIZE, SIZE, SIZE, 3).transpose(3, 0, 1, 2).copy()
            assert np.isfinite(vel).all(), "non-finite values"
            np.save(fp, vel)
            el = time.time() - t0
            print(f"    t={t_idx} ok ({el:.0f}s, {len(r)/1024:.0f}KB)", flush=True)
            time.sleep(QUERY_GAP)
            return vel
        except Exception as e:
            last_err = e
            wait = RETRY_GAP * (attempt + 1)
            print(f"    retry {attempt+1}/{RETRIES} after {wait:.0f}s: {str(e)[:150]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"frame t={t_idx} origin={origin} failed: {last_err}")


def div_rms(vel):
    du = vel[0, 1:, :-1, :-1] - vel[0, :-1, :-1, :-1]
    dv = vel[1, :-1, 1:, :-1] - vel[1, :-1, :-1, :-1]
    dw = vel[2, :-1, :-1, 1:] - vel[2, :-1, :-1, :-1]
    return float(np.sqrt(((du + dv + dw) ** 2).mean()))


def spectrum_slope(u):
    N = u.shape[-1]
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    psd = (np.abs(uh) ** 2).sum(axis=0)
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kr = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    E = np.zeros(N // 2 + 1)
    for kk in range(1, N // 2 + 1):
        m = (kr >= kk - 0.5) & (kr < kk + 0.5)
        if m.sum() > 0:
            E[kk] = psd[m].mean()
    pts = [(k_, E[k_]) for k_ in range(3, 20) if E[k_] > 0]
    return float(np.polyfit(np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1)[0])


def assemble_trajectory(traj_idx, origin):
    """从缓存组装单轨迹 h5（全部帧到齐才调用）"""
    fields = []
    for i in range(N_FRAMES):
        t_idx = FRAME_START + i * FRAME_STEP
        fp = os.path.join(CACHE_DIR, f"traj{traj_idx:02d}", f"t{t_idx:04d}.npy")
        fields.append(np.load(fp))
    fields = np.stack(fields, axis=0)
    u0 = fields[0]
    print(f"  sanity: rms={float(np.sqrt((u0**2).mean())):.4f} "
          f"div_rms={div_rms(u0):.2e} slope={spectrum_slope(u0):.2f}", flush=True)
    out_path = os.path.join(OUT_DIR, f"jhtdb_iso_re433_N64_traj{traj_idx:02d}.h5")
    times = np.arange(FRAME_START, FRAME_START + N_FRAMES * FRAME_STEP, FRAME_STEP, dtype=np.float64)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("fields", data=fields, compression="gzip", compression_opts=4)
        f.create_dataset("times", data=times)
        f.attrs["dataset"] = DATASET
        f.attrs["source"] = "JHTDB isotropic turbulence DNS (SOAP GetAnyCutoutWeb)"
        f.attrs["Re_lambda"] = 433
        f.attrs["origin"] = list(origin)
        f.attrs["n_frames"] = N_FRAMES
        f.attrs["frame_start"] = FRAME_START
        f.attrs["frame_step"] = FRAME_STEP
        f.attrs["dt_effective"] = 0.002 * FRAME_STEP
        f.attrs["traj_idx"] = traj_idx
        f.attrs["layout"] = "fields[T, component, z, y, x] (axes equivalent for isotropic)"
    print(f"  saved -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


def main():
    token = load_token()
    print(f"token: ****{token[-6:]}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    client = make_client()
    t_all = time.time()
    total = len(ORIGINS) * N_FRAMES
    done = 0

    for tj, origin in enumerate(ORIGINS):
        out_path = os.path.join(OUT_DIR, f"jhtdb_iso_re433_N64_traj{tj:02d}.h5")
        if os.path.exists(out_path):
            with h5py.File(out_path, "r") as f:
                if f.attrs.get("n_frames") == N_FRAMES:
                    print(f"[traj{tj:02d}] exists, skip", flush=True)
                    done += N_FRAMES
                    continue
        print(f"[traj{tj:02d}] origin={origin} frames={N_FRAMES}", flush=True)
        for i in range(N_FRAMES):
            t_idx = FRAME_START + i * FRAME_STEP
            fetch_frame(client, token, t_idx, origin, tj)
            done += 1
            el = time.time() - t_all
            print(f"  [{i+1}/{N_FRAMES}] total {done}/{total} elapsed={el/60:.0f}m", flush=True)
        assemble_trajectory(tj, origin)

    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
