#!/usr/bin/env python3
"""Evaluate a checkpoint on its untouched test split."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from train import rows, ManifestDataset, predict
from wafer_tcad.model import WaferFusionCNN
from wafer_tcad.metrics import classification_metrics
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",default="data/processed/manifest.csv"); ap.add_argument("--data-root",default=None); ap.add_argument("--checkpoint",default="artifacts/model.pt"); ap.add_argument("--output",default="outputs/metrics.json"); ap.add_argument("--batch-size",type=int,default=64); a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); rec=[r for r in rows(a.manifest) if r["split"]=="test"]
    ds=ManifestDataset(rec,ck["classes"],ck["feature_cols"],data_root=a.data_root or ck.get("data_root","data/raw/WM811k_Dataset")); model=WaferFusionCNN(**ck["config"]); model.load_state_dict(ck["state_dict"])
    logits,y,_,_=predict(model,torch.utils.data.DataLoader(ds,batch_size=a.batch_size),torch.device("cpu")); probs=torch.softmax(logits/ck.get("temperature",1.),1).numpy(); result=classification_metrics(y.numpy(),probs,ck["classes"])
    baseline=Path(a.checkpoint).with_name("baseline_logistic.joblib")
    if baseline.exists():
        import joblib, numpy as np
        clf=joblib.load(baseline); X=np.stack([ds[i][1].numpy() for i in range(len(ds))])
        bp=clf.predict_proba(X); result["baseline_logistic"] = classification_metrics(y.numpy(),bp,ck["classes"])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
