#!/usr/bin/env python3
"""Download the public Kaggle archive through Kaggle's official API endpoint."""
import argparse, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--destination", type=Path, default=Path("data/raw")); args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image"
    archive = args.destination / "wm811k-silicon-wafer-map-dataset-image.zip"
    try:
        print(f"Downloading public archive from {url}")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf: zf.extractall(args.destination)
    except Exception as exc:
        if not shutil.which("kaggle"):
            sys.exit(f"Public download failed: {exc}\nInstall the Kaggle CLI (`pip install kaggle`) and configure kaggle.json, or download manually from the dataset page.")
        print(f"Public endpoint failed ({exc}); trying authenticated Kaggle CLI.")
        subprocess.run(["kaggle", "datasets", "download", "-d", "muhammedjunayed/wm811k-silicon-wafer-map-dataset-image", "-p", str(args.destination), "--unzip"], check=True)
    print(f"Dataset extracted under {args.destination.resolve()}")
if __name__ == "__main__": main()
