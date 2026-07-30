import numpy as np
from PIL import Image
from wafer_tcad.data import decode_wafer, defect_mask, stratified_group_split
from wafer_tcad.features import FEATURE_NAMES, feature_vector, spatial_features

def test_decode_converts_grayscale_and_resizes(tmp_path):
    path = tmp_path / "gray.png"; Image.fromarray(np.zeros((9, 7), np.uint8)).save(path)
    assert decode_wafer(path).shape == (32, 32, 3)

def test_center_and_edge_descriptors():
    center = np.full((32, 32, 3), 255, np.uint8); center[14:18, 14:18] = [255, 0, 0]
    edge = np.full_like(center, 255); edge[14:18, 28:31] = [255, 0, 0]
    fc, fe = spatial_features(center), spatial_features(edge)
    assert fc["core_density"] > fc["edge_density"]
    assert fe["edge_density"] > fe["core_density"] and fe["centroid_x"] > .7
    assert feature_vector(center).shape == (len(FEATURE_NAMES),)

def test_empty_map_is_finite():
    values = feature_vector(np.full((32, 32, 3), 255, np.uint8))
    assert np.isfinite(values).all() and not values.any()

def test_groups_never_cross_splits_and_is_reproducible():
    labels = ["a"] * 12 + ["b"] * 12; groups = [f"a{i//2}" for i in range(12)] + [f"b{i//2}" for i in range(12)]
    split = stratified_group_split(labels, groups, seed=7)
    assert np.array_equal(split, stratified_group_split(labels, groups, seed=7))
    for group in set(groups): assert len(set(split[np.array(groups) == group])) == 1
    assert set(split) == {"train", "val", "test"}
