"""Compact Cartesian/polar fusion network and post-hoc calibration.

The polar view unwraps radial signatures: rings become nearly vertical bands while
edge/local defects remain localized.  This is an engineering prior, not a claim
that the network identifies a physical process root cause.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def cartesian_to_polar(x: torch.Tensor, radial_bins: Optional[int] = None,
                       angular_bins: Optional[int] = None) -> torch.Tensor:
    """Differentiably unwrap ``B,C,H,W`` images to radius x angle coordinates."""
    if x.ndim != 4:
        raise ValueError("expected an image tensor with shape B,C,H,W")
    b, _, h, w = x.shape
    nr, nt = radial_bins or h, angular_bins or w
    radius = torch.linspace(0, 1, nr, device=x.device, dtype=x.dtype)
    theta = torch.linspace(-math.pi, math.pi, nt, device=x.device, dtype=x.dtype)
    rr, tt = torch.meshgrid(radius, theta, indexing="ij")
    # grid_sample coordinates are x,y in [-1,1]; radius=1 reaches image edge.
    grid = torch.stack((rr * torch.cos(tt), rr * torch.sin(tt)), dim=-1)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, width: int = 24, out_dim: int = 96):
        super().__init__()
        layers = []
        c = in_channels
        for nxt in (width, width * 2, width * 4):
            layers += [nn.Conv2d(c, nxt, 3, padding=1, bias=False),
                       nn.BatchNorm2d(nxt), nn.SiLU(), nn.MaxPool2d(2)]
            c = nxt
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(c, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x).flatten(1))


class WaferFusionCNN(nn.Module):
    """Fuses Cartesian pixels, polar pixels, and optional spatial features."""
    def __init__(self, num_classes: int, in_channels: int = 1,
                 num_features: int = 0, width: int = 24, embedding_dim: int = 96,
                 dropout: float = 0.2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes, self.num_features = num_classes, num_features
        self.cartesian = ConvEncoder(in_channels, width, embedding_dim)
        self.polar = ConvEncoder(in_channels, width, embedding_dim)
        self.feature_net = (nn.Sequential(nn.LayerNorm(num_features),
                            nn.Linear(num_features, 32), nn.SiLU())
                            if num_features else None)
        fused = embedding_dim * 2 + (32 if num_features else 0)
        self.fusion = nn.Sequential(nn.Linear(fused, 128), nn.SiLU(),
                                    nn.Dropout(dropout))
        self.classifier = nn.Linear(128, num_classes)

    def embed(self, image: torch.Tensor,
              features: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [self.cartesian(image), self.polar(cartesian_to_polar(image))]
        if self.feature_net is not None:
            if features is None or features.shape[-1] != self.num_features:
                raise ValueError(f"expected {self.num_features} engineered features")
            parts.append(self.feature_net(features))
        return self.fusion(torch.cat(parts, dim=1))

    def forward(self, image: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.embed(image, features))


class TemperatureScaler(nn.Module):
    """Single-parameter calibration fitted on validation logits only."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 50) -> "TemperatureScaler":
        logits, labels = logits.detach(), labels.detach()
        opt = torch.optim.LBFGS([self.log_temperature], lr=0.05,
                                max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self

