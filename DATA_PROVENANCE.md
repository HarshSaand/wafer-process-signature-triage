# Data provenance and responsible use

## Source

This prototype uses the Kaggle dataset **WM811K Silicon Wafer Map Dataset
Image**, published by Muhammed Junayed:

https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image

The locally audited image release contains 902 JPEG images organized into nine
folders: Center, Donut, Edge Local, Edge Ring, Local, Scratch, near full, none,
and random. Every audited image is 32 × 32 pixels. The machine-readable audit
is generated at `data/processed/manifest.summary.json`; rerun the audit rather
than copying these statements to a different dataset release.

## License boundary

The repository's MIT license covers original source code and documentation
only. It does **not** grant rights to the dataset, downloaded images, model
weights derived from them, or third-party packages. Obtain the dataset from
its source, review the current Kaggle usage terms and dataset-page license, and
follow any attribution or redistribution limits shown there. Raw data is
excluded from Git by default.

## Integrity and limitations

- Images are a curated 902-image derivative, not the complete 811,457-wafer
  WM-811K pickle commonly discussed in research.
- Folder names are treated as labels; they are not independently verified
  process diagnoses.
- Exact hashes and perceptual-hash groups are kept together during splitting to
  reduce duplicate leakage. Similar-looking wafers can still remain.
- The images do not include lot, tool, recipe, sensor, metrology, or ground-truth
  root-cause metadata. Pattern classification cannot establish causality.
- Do not use this prototype for production disposition, safety-critical action,
  or unsupervised recipe changes.

## Reproduce the audit

```bash
python scripts/audit_dataset.py
```

Check the script's `--help` if your dataset is stored in a non-default folder.
