"""Interpretable spatial signatures for wafer-process triage."""
from __future__ import annotations
import numpy as np
from .data import defect_mask

FEATURE_NAMES = [
    "defect_fraction", "centroid_x", "centroid_y", "radial_mean", "radial_std",
    "core_density", "middle_density", "edge_density", "left_density", "right_density",
    "top_density", "bottom_density", "anisotropy", "orientation_sin", "orientation_cos",
    "angular_entropy", "radial_entropy",
]


def spatial_features(image: np.ndarray) -> dict[str, float]:
    """Return normalized, engineer-readable descriptors of a defect signature."""
    mask = defect_mask(image); h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]; x = (xx - (w - 1) / 2) / max((w - 1) / 2, 1); y = (yy - (h - 1) / 2) / max((h - 1) / 2, 1)
    r = np.sqrt(x*x + y*y); wafer = r <= 1.05; active = mask & wafer
    n = int(active.sum())
    def density(region):
        denom = np.sum(region & wafer); return float(np.sum(active & region) / denom) if denom else 0.
    if not n:
        return {name: 0. for name in FEATURE_NAMES}
    xa, ya, ra = x[active], y[active], r[active]; cx, cy = xa.mean(), ya.mean()
    cov = np.cov(np.stack([xa, ya]), bias=True); eig = np.linalg.eigvalsh(cov)
    anisotropy = float((eig[-1] - eig[0]) / (eig.sum() + 1e-8))
    angle = .5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    def entropy(values, bins, domain):
        counts, _ = np.histogram(values, bins=bins, range=domain); p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / np.log(bins))
    return dict(zip(FEATURE_NAMES, map(float, [
        n / wafer.sum(), cx, cy, ra.mean(), ra.std(), density(r < .33), density((r >= .33) & (r < .70)),
        density(r >= .70), density(x < 0), density(x >= 0), density(y < 0), density(y >= 0),
        anisotropy, np.sin(2*angle), np.cos(2*angle), entropy(np.arctan2(ya, xa), 8, (-np.pi, np.pi)),
        entropy(ra, 5, (0, 1.05)),
    ])))


def feature_vector(image: np.ndarray) -> np.ndarray:
    values = spatial_features(image)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)

