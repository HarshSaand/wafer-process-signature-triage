#!/usr/bin/env python3
"""Create a reproducible dataset manifest and human-readable audit summary."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_tcad.data import assign_duplicate_groups, build_manifest, stratified_group_split, write_manifest, decode_wafer
from wafer_tcad.features import spatial_features

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=Path("data/raw/WM811k_Dataset")); parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv")); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); records = build_manifest(args.data); groups = assign_duplicate_groups(records)
    splits = stratified_group_split([r["label"] for r in records], groups, args.seed)
    for row, group, split in zip(records, groups, splits):
        row["duplicate_group"], row["split"] = group, split
        # The Kaggle derivative contains rendered JPEG heatmaps, not original
        # ternary die arrays. These descriptors therefore operate on a clearly
        # documented intensity/color proxy and are not exact die-failure counts.
        row.update({f"feat_{k}": float(v) for k, v in spatial_features(decode_wafer(args.data / row["path"])).items()})
    write_manifest(records, args.output)
    summary = {"source": str(args.data), "samples": len(records), "classes": dict(sorted(Counter(r["label"] for r in records).items())), "dimensions": dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)), "unique_sha256": len(set(r["sha256"] for r in records)), "perceptual_groups": len(set(groups)), "split_samples": dict(Counter(splits)), "split_groups": {s: len({g for g, x in zip(groups, splits) if x == s}) for s in ("train", "val", "test")}}
    summary_path = args.output.with_suffix(".summary.json"); summary_path.write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
