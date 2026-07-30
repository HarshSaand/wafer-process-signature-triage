"""Dataset discovery, decoding, duplicate grouping, and leakage-safe splitting."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(root: str | Path) -> list[Path]:
    """Return supported images in deterministic class/path order."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def decode_wafer(path: str | Path, size: tuple[int, int] | None = (32, 32)) -> np.ndarray:
    """Decode an image to RGB uint8, tolerating grayscale/RGBA inputs."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8).copy()


def grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxW or HxWx3 image, got {image.shape}")
    return np.tensordot(image[..., :3], np.array([0.299, 0.587, 0.114]), axes=1).astype(np.float32)


def defect_mask(image: np.ndarray) -> np.ndarray:
    """Infer foreground defect pixels from rendered wafer maps.

    JPEG compression makes exact RGB rules brittle. Saturated/dark colored pixels are
    treated as defects while the near-white background and pale wafer disk are ignored.
    """
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    spread = rgb.max(2) - rgb.min(2)
    brightness = rgb.mean(2)
    return ((spread > 28) & (brightness < 245)) | (brightness < 75)


def difference_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Perceptual dHash used to identify exact/near duplicate renderings."""
    gray = Image.fromarray(grayscale(image).astype(np.uint8)).resize(
        (hash_size + 1, hash_size), Image.Resampling.BILINEAR
    )
    values = np.asarray(gray)
    bits = (values[:, 1:] > values[:, :-1]).ravel()
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):0{hash_size * hash_size // 4}x}"


def build_manifest(root: str | Path) -> list[dict[str, object]]:
    """Inspect all images and create serializable provenance/audit records."""
    root = Path(root)
    records: list[dict[str, object]] = []
    for path in image_files(root):
        raw = path.read_bytes()
        with Image.open(path) as probe:
            width, height = probe.size
            mode = probe.mode
        image = decode_wafer(path, None)
        records.append({
            "path": path.relative_to(root).as_posix(), "label": path.parent.name,
            "width": width, "height": height, "mode": mode,
            "sha256": hashlib.sha256(raw).hexdigest(), "dhash": difference_hash(image),
        })
    return records


def assign_duplicate_groups(records: list[dict[str, object]], max_hamming: int = 3) -> list[str]:
    """Connected-component grouping of perceptual hashes within each class.

    Comparisons are class-local so visually similar but semantically different labels
    remain visible as a label-quality issue rather than being silently merged.
    """
    parent = list(range(len(records)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    by_label: dict[str, list[int]] = {}
    for i, row in enumerate(records): by_label.setdefault(str(row["label"]), []).append(i)
    for indices in by_label.values():
        for pos, i in enumerate(indices):
            hi = int(str(records[i]["dhash"]), 16)
            for j in indices[pos + 1:]:
                if (hi ^ int(str(records[j]["dhash"]), 16)).bit_count() <= max_hamming:
                    union(i, j)
    roots: dict[int, str] = {}
    groups = []
    for i in range(len(records)):
        r = find(i); roots.setdefault(r, f"dup_{len(roots):04d}"); groups.append(roots[r])
    return groups


def stratified_group_split(labels: Iterable[str], groups: Iterable[str], seed: int = 42,
                           train_fraction: float = .70, val_fraction: float = .15) -> np.ndarray:
    """Assign whole groups to train/val/test while greedily preserving class ratios."""
    labels, groups = np.asarray(list(labels)), np.asarray(list(groups))
    if len(labels) != len(groups) or not len(labels): raise ValueError("labels/groups must be non-empty and equal length")
    fractions = np.array([train_fraction, val_fraction, 1 - train_fraction - val_fraction])
    if np.any(fractions <= 0): raise ValueError("split fractions must be positive and sum below one")
    rng = np.random.default_rng(seed); result = np.empty(len(labels), dtype=object)
    names = np.array(["train", "val", "test"])
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label); unique = np.unique(groups[idx]); rng.shuffle(unique)
        sizes = np.array([np.sum(groups[idx] == group) for group in unique])
        target = fractions * len(idx); counts = np.zeros(3)
        for group, size in sorted(zip(unique, sizes), key=lambda x: -x[1]):
            # Largest relative deficit; deterministic RNG shuffle above breaks ties.
            choice = int(np.argmax((target - counts) / np.maximum(target, 1)))
            result[(labels == label) & (groups == group)] = names[choice]; counts[choice] += size
    return result


def write_manifest(records: list[dict[str, object]], output: str | Path) -> None:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

