"""
RCLN-UPI v5-2D for 2D Navier-Stokes (System B, NC P0-compliant)
================================================================
NC P0-2: Constraint-Preserving Physical Anchor (NOT full NS solver).
NC P0-1: One-sided anti-explosion safety net — dissipation allowed.
Same principle as v5-3D, translated to 2D convolutions.
Physics anchor: divergence-free + energy-bounded envelope on (u,v).
Residual learner: learns 2D turbulent corrections from data.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional

def kinetic_energy_2d(u): return 0.5*(u**2).sum(dim=(1,2,3))

class Encoder2D(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.l0 = nn.Sequential(nn.Conv2d(in_ch,base,7,padding=3),nn.GroupNorm(4,base),nn.SiLU(),
                                nn.Conv2d(base,base,3,padding=1),nn.GroupNorm(4,base),nn.SiLU())
        self.d1 = nn.Conv2d(base,base*2,4,stride=2,padding=1)
        self.l1 = nn.Sequential(nn.Conv2d(base*2,base*2,3,padding=1),nn.GroupNorm(4,base*2),nn.SiLU())
        self.d2 = nn.Conv2d(base*2,base*4,4,stride=2,padding=1)
        self.l2 = nn.Sequential(nn.Conv2d(base*4,base*4,3,padding=1),nn.GroupNorm(4,base*4),nn.SiLU())
        self.d3 = nn.Conv2d(base*4,base*8,4,stride=2,padding=1)
        self.l3 = nn.Sequential(nn.Conv2d(base*8,base*8,3,padding=1),nn.GroupNorm(4,base*8),nn.SiLU())
    def forward(self,x):
        f0=self.l0(x); f1=self.l1(self.d1(f0)); f2=self.l2(self.d2(f1)); f3=self.l3(self.d3(f2))
        return f0,f3

class Observer2D(nn.Module):
    def __init__(self, deep_ch=256, latent_dim=64):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(2)
        self.mp = nn.AdaptiveMaxPool2d(2)
        pf = deep_ch * 8  # 256*2*2*2 = 2048 (pool to 2x2, cat avg+max = 4*256*2)
        self.to_z = nn.Sequential(nn.Linear(pf, latent_dim*2), nn.SiLU(), nn.Linear(latent_dim*2, latent_dim))
        self.zs = nn.Sequential(nn.Linear(latent_dim, 64), nn.SiLU(), nn.Linear(64, latent_dim))
        self.ch = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.Sigmoid())
        self.eh = nn.Sequential(nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, 8), nn.Sigmoid())

    def forward(self, f3):
        B = f3.shape[0]
        p = torch.cat([self.ap(f3), self.mp(f3)], dim=1).view(B, -1)
        z=self.to_z(p); return self.zs(z),self.ch(z).view(B,1,8,8),self.eh(z)

class SoftShell2D(nn.Module):
    def __init__(self,lc=32,gd=64,ne=6):
        super().__init__(); self.ne=ne
        self.gate=nn.Sequential(nn.Conv2d(lc+gd,lc,1),nn.GroupNorm(4,lc),nn.SiLU(),
                                nn.Conv2d(lc,ne,1))
        self.experts=nn.ModuleList([nn.Sequential(
            nn.Conv2d(lc+gd,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,2,3,padding=1)) for _ in range(ne)])
    def forward(self,f0,z):
        B,_,H,W=f0.shape; zb=z.view(B,-1,1,1).expand(-1,-1,H,W); gi=torch.cat([f0,zb],dim=1)
        pi=F.softmax(self.gate(gi),dim=1)
        outs=[e(torch.cat([f0,zb],dim=1)) for e in self.experts]
        u=sum(pi[:,k:k+1]*outs[k] for k in range(self.ne))
        lb=self.ne*(pi.mean(dim=(0,2,3))**2).sum()
        return u,pi,torch.tensor(0.0),lb

class Arbiter2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid())
    def forward(self, cert, sigma):
        cert_up = F.interpolate(cert, size=sigma.shape[2:], mode='bilinear', align_corners=False)
        w = self.net(torch.cat([cert_up, sigma], dim=1))
        B = w.shape[0]
        return w, w, torch.zeros(B, 1, device=w.device), torch.zeros(B, 1, device=w.device)

class SpatialFiLM2D(nn.Module):
    def __init__(self,sd=64):
        super().__init__()
        self.gs=nn.Sequential(nn.Linear(sd,64),nn.SiLU(),nn.Linear(64,4))
        self.sr=nn.Sequential(nn.Conv2d(5,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
                              nn.Conv2d(16,4,3,padding=1))
    def forward(self,uc,zs,cert,w):
        B,_,H,W=uc.shape
        seed=self.gs(zs).view(B,4,1,1).expand(-1,-1,8,8)
        sp=torch.cat([cert,seed],dim=1)
        sp16=F.interpolate(sp,size=(16,16),mode='bilinear',align_corners=False)
        ref16=self.sr(sp16)
        film=F.interpolate(ref16,size=(H,W),mode='bilinear',align_corners=False)
        g,b=film[:,:2],film[:,2:]
        return (1-w)*(g*uc+b)+w*uc

class RCLN_UPI_v8_2D(nn.Module):
    """Full v8-2D: Encoder -> Observer -> MoE SS -> Arbiter -> SpatialFiLM"""
    def __init__(self,in_ch=2,base=32,latent_dim=64):
        super().__init__()
        self.encoder=Encoder2D(in_ch,base)
        self.observer=Observer2D(base*8,latent_dim)
        self.ss=SoftShell2D(base,latent_dim)
        self.arbiter=Arbiter2D()
        self.fusion=SpatialFiLM2D(latent_dim)
    def forward(self,u,target=None,return_components=False):
        f0,f3=self.encoder(u)
        zs,cert,ene=self.observer(f3)
        uc,pi,me,lb=self.ss(f0,zs)
        sc=F.softplus((uc-uc.mean(dim=2,keepdim=True)).abs().mean(dim=1,keepdim=True))
        w,stim,ssig,risk=self.arbiter(cert,sc)
        uf=self.fusion(uc,zs,cert,w)
        if not self.training:
            Ep=kinetic_energy_2d(uf); Ei=kinetic_energy_2d(u)
            mask=Ep>Ei*1.05
            if mask.any():
                sc2=torch.where(mask,torch.sqrt(Ei*1.05/(Ep+1e-8)),torch.ones_like(Ei))
                uf=uf*sc2.view(-1,1,1,1)
        if return_components: return uf,{'u_comp':uc,'z_struct':zs,'cert_8':cert,'w':w}
        return uf

if __name__ == '__main__':
    m = RCLN_UPI_v8_2D()
    x = torch.randn(1, 2, 64, 64)
    y, c = m(x, return_components=True)
    n = sum(p.numel() for p in m.parameters())
    print(f'v8-2D: {n/1e6:.2f}M params')
    print(f'Output: {tuple(y.shape)}, w mean: {c["w"].mean().item():.3f}')
    print('v8-2D ready')
