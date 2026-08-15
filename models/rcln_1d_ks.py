"""
RCLN-UPI v5-1D for Kuramoto-Sivashinsky (System C, NC P0-compliant)
====================================================================
NC P0-2: Constraint-Preserving Physical Anchor (NOT claiming PDE solver).
NC P0-1: One-sided anti-explosion safety net — dissipation allowed.
Same principle as v5-3D, different PhysicsSpec (1D chaotic KS).
KS: u_t + u*u_x + u_xx + u_xxxx = 0
Physics anchor: bounded energy envelope (anti-explosion safety net)
Residual learner: learns chaotic corrections from data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def kinetic_energy_1d(u):
    return 0.5 * (u ** 2).sum(dim=(1, 2))


# ── Encoder: 1D CNN ──
class Encoder1D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.level0 = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv1d(base, base, 3, padding=1), nn.GroupNorm(4, base), nn.SiLU(),
        )
        self.down1 = nn.Conv1d(base, base*2, 4, stride=2, padding=1)
        self.level1 = nn.Sequential(
            nn.Conv1d(base*2, base*2, 3, padding=1), nn.GroupNorm(4, base*2), nn.SiLU(),
        )
        self.down2 = nn.Conv1d(base*2, base*4, 4, stride=2, padding=1)
        self.level2 = nn.Sequential(
            nn.Conv1d(base*4, base*4, 3, padding=1), nn.GroupNorm(4, base*4), nn.SiLU(),
        )
        self.down3 = nn.Conv1d(base*4, base*8, 4, stride=2, padding=1)
        self.level3 = nn.Sequential(
            nn.Conv1d(base*8, base*8, 3, padding=1), nn.GroupNorm(4, base*8), nn.SiLU(),
        )

    def forward(self, x):
        f0 = self.level0(x)              # [B, 32, 64]
        f1 = self.level1(self.down1(f0)) # [B, 64, 32]
        f2 = self.level2(self.down2(f1)) # [B, 128, 16]
        f3 = self.level3(self.down3(f2)) # [B, 256, 8]
        return f0, f3


# ── Structured Observer (1D) ──
class Observer1D(nn.Module):
    def __init__(self, deep_ch=256, latent_dim=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(2)
        self.max_pool = nn.AdaptiveMaxPool1d(2)
        pool_flat = deep_ch * 4
        self.to_latent = nn.Sequential(
            nn.Linear(pool_flat, latent_dim*2), nn.SiLU(),
            nn.Linear(latent_dim*2, latent_dim),
        )
        self.z_struct = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(), nn.Linear(64, latent_dim),
        )
        self.cert_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.Sigmoid(),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, 8), nn.Sigmoid(),
        )

    def forward(self, f3):
        B = f3.shape[0]
        p_avg = self.avg_pool(f3)
        p_max = self.max_pool(f3)
        pooled = torch.cat([p_avg, p_max], dim=1).view(B, -1)
        z = self.to_latent(pooled)
        return (self.z_struct(z),
                self.cert_head(z).view(B, 1, 64),  # [B, 1, 64] — per-point cert
                self.energy_head(z))


# ── Soft Shell (1D MoE) ──
class SoftShell1D(nn.Module):
    def __init__(self, local_ch=32, global_dim=64, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        gate_in = local_ch + global_dim
        self.gate = nn.Sequential(
            nn.Conv1d(gate_in, local_ch, 1), nn.GroupNorm(4, local_ch), nn.SiLU(),
            nn.Conv1d(local_ch, n_experts, 1),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(local_ch + global_dim, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                nn.Conv1d(32, 1, 3, padding=1),
            ) for _ in range(n_experts)
        ])

    def forward(self, f0, z_global):
        B, _, N = f0.shape
        z_b = z_global.view(B, -1, 1).expand(-1, -1, N)
        gate_in = torch.cat([f0, z_b], dim=1)
        pi = F.softmax(self.gate(gate_in), dim=1)
        outputs = [expert(torch.cat([f0, z_b], dim=1)) for expert in self.experts]
        u_comp = sum(pi[:, k:k+1] * outputs[k] for k in range(self.n_experts))
        load_balance = self.n_experts * (pi.mean(dim=(0, 2)) ** 2).sum()
        return u_comp, pi, torch.tensor(0.0), load_balance


# ── Channel Arbiter (1D) ──
class Arbiter1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv1d(2, 8, 1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv1d(8, 1, 1), nn.Sigmoid(),
        )

    def forward(self, cert, sigma_comp):
        cert_full = cert  # already at full resolution
        w = self.weight_net(torch.cat([cert_full, sigma_comp], dim=1))
        uncertainty = 1.0 - cert_full
        stim_map = torch.sigmoid((uncertainty + sigma_comp - 0.5) * 5.0)
        stim_signal = stim_map.mean(dim=(1, 2))
        risk_flag = (stim_signal > 0.7).float()
        return w, stim_map, stim_signal, risk_flag


# ── Spatial FiLM (1D) ──
class SpatialFiLM1D(nn.Module):
    def __init__(self, struct_dim=64):
        super().__init__()
        self.global_seed = nn.Sequential(
            nn.Linear(struct_dim, 64), nn.SiLU(), nn.Linear(64, 2),
        )
        self.spatial_refine = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv1d(16, 2, 3, padding=1),
        )

    def forward(self, u_comp, z_struct, cert, w):
        B, C, N = u_comp.shape
        seed = self.global_seed(z_struct).view(B, 2, 1).expand(-1, -1, N)
        spatial = torch.cat([cert, seed], dim=1)
        film_map = self.spatial_refine(spatial)
        gamma = film_map[:, :1]
        beta = film_map[:, 1:]
        u_cond = gamma * u_comp + beta
        return (1 - w) * u_cond + w * u_comp


# ── Energy Safety Net (1D anti-explosion) ──
def energy_safetynet_1d(u_final, u_input):
    E_pred = kinetic_energy_1d(u_final)
    E_in = kinetic_energy_1d(u_input)
    explosion = E_pred > E_in * 1.05
    if explosion.any():
        scale = torch.where(
            explosion,
            torch.sqrt(E_in * 1.05 / (E_pred + 1e-8)),
            torch.ones_like(E_in)
        )
        return u_final * scale.view(-1, 1, 1), {'explosions_prevented': int(explosion.sum().item())}
    return u_final, {'explosions_prevented': 0}


# ============================================================
# Full v8-1D model
# ============================================================

class RCLN_UPI_v8_1D(nn.Module):
    """v8 architecture for 1D PDEs (KS). Same principle, different PhysicsSpec."""

    def __init__(self, in_ch=1, base=32, latent_dim=64):
        super().__init__()
        self.encoder = Encoder1D(in_ch, base)
        self.observer = Observer1D(base * 8, latent_dim)
        self.soft_shell = SoftShell1D(base, latent_dim)
        self.arbiter = Arbiter1D()
        self.fusion = SpatialFiLM1D(latent_dim)

    def forward(self, u, target=None, return_components=False):
        f0, f3 = self.encoder(u)
        z_struct, cert, energy = self.observer(f3)
        u_comp, pi, me, lb = self.soft_shell(f0, z_struct)
        sigma_comp = F.softplus((u_comp - u_comp.mean(dim=2, keepdim=True)).abs().mean(dim=1, keepdim=True))
        w, stim, stim_sig, risk = self.arbiter(cert, sigma_comp)
        u_fused = self.fusion(u_comp, z_struct, cert, w)

        if not self.training:
            u_final, energy_info = energy_safetynet_1d(u_fused, u)
        else:
            u_final = u_fused
            energy_info = {}

        if return_components:
            return u_final, {'u_comp': u_comp, 'z_struct': z_struct, 'cert': cert, 'w': w}
        return u_final


# ── Test ──
if __name__ == '__main__':
    m = RCLN_UPI_v8_1D()
    x = torch.randn(2, 1, 64)
    y, c = m(x, return_components=True)
    n = sum(p.numel() for p in m.parameters())
    print(f'v8-1D: {n/1e6:.2f}M params')
    print(f'Output: {y.shape}')
    print(f'w mean: {c["w"].mean().item():.3f}')
    print(f'cert shape: {c["cert"].shape}')
    print('All tests passed')
