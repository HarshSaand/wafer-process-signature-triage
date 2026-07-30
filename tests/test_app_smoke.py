from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def test_app_imports_without_starting_server():
    module = importlib.import_module("app")
    assert callable(module.main)


def test_image_tensor_shape_and_range():
    from app import image_tensor

    image = Image.fromarray(np.full((32, 32), 127, dtype=np.uint8))
    tensor = image_tensor(image)
    assert tensor.shape == (1, 1, 64, 64)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_load_wafer_image_detaches_from_file(tmp_path: Path):
    from app import load_wafer_image

    path = tmp_path / "wafer.png"
    Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(path)
    loaded = load_wafer_image(path)
    path.unlink()
    assert loaded.mode == "RGB"
    assert loaded.size == (9, 8)
