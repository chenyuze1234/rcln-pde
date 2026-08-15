"""
UNet-3D Baseline Model
=======================
Canonical 3D U-Net (Cicek et al. MICCAI 2016) adapted for flow prediction.
Architecture: 4-level encoder-decoder, skip connections, double 3x3x3 convs.

Configurations:
  base=24 → ~3.15M params (matches v5 CNN)
  base=48 → ~12.5M params
  base=34 → ~6.3M params  (matches v8 TFM)

Extracted and cleaned from experiments/baselines.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groupnorm(channels: int, max_groups: int = 8):
    """Return GroupNorm with largest divisor ≤ max_groups, or Identity."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.Identity()


class DoubleConv3D(nn.Module):
    """Two 3x3x3 convs + GroupNorm + SiLU — standard UNet building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _safe_groupnorm(out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _safe_groupnorm(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    Canonical 3D U-Net for velocity field prediction.

    64³ → 32³ → 16³ → 8³  (encoder, MaxPool3d downsample)
      8³ → 16³ → 32³ → 64³  (decoder, ConvTranspose3d upsample + skip)

    Args:
        in_ch:  input channels (3 for velocity)
        out_ch: output channels (3 for velocity)
        base:   base channel width — scales param count
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 34):
        super().__init__()
        w = base

        # ── Encoder ──
        self.enc0 = DoubleConv3D(in_ch, w)
        self.down0 = nn.MaxPool3d(2)
        self.enc1 = DoubleConv3D(w, w * 2)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(w * 2, w * 4)
        self.down2 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(w * 4, w * 8)

        # ── Decoder ──
        self.up2 = nn.ConvTranspose3d(w * 8, w * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(w * 8, w * 4)  # concat with e2 → doubled input
        self.up1 = nn.ConvTranspose3d(w * 4, w * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(w * 4, w * 2)  # concat with e1
        self.up0 = nn.ConvTranspose3d(w * 2, w, 2, stride=2)
        self.dec0 = nn.Sequential(
            DoubleConv3D(w * 2, w),              # concat with e0
            nn.Conv3d(w, out_ch, 1),
        )

    def forward(self, x):
        e0 = self.enc0(x)                               # [B, w,   64, 64, 64]
        e1 = self.enc1(self.down0(e0))                  # [B, 2w,  32, 32, 32]
        e2 = self.enc2(self.down1(e1))                  # [B, 4w,  16, 16, 16]
        b  = self.bottleneck(self.down2(e2))            # [B, 8w,   8,  8,  8]

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # [B, 4w, 16, 16, 16]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 2w, 32, 32, 32]
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))  # [B, C,  64, 64, 64]
        return d0


# ── Param count test ──
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for base in [24, 30, 34, 48, 64]:
        model = UNet3D(in_ch=3, out_ch=3, base=base).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        tag = ""
        if abs(n_params - 3_430_000) < 200_000:
            tag = " ← v5 CNN (3.4M)"
        elif abs(n_params - 6_500_000) < 300_000:
            tag = " ← v8 TFM (6.5M)"
        print(f"UNet-3D base={base:3d}: {n_params:>10,} params ({n_params/1e6:.2f}M){tag}")

    # Forward test with best-match config
    model = UNet3D(in_ch=3, out_ch=3, base=34).to(device)
    x = torch.randn(1, 3, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"MSE random: {F.mse_loss(out, torch.randn_like(out)).item():.4f}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("All tests passed!")
