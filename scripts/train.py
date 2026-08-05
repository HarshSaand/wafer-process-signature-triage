#!/usr/bin/env python3
"""Train the fusion CNN and engineered-feature logistic baseline."""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from wafer_process_ai.model import WaferFusionCNN, TemperatureScaler
from wafer_process_ai.data import decode_wafer
from wafer_process_ai.features import FEATURE_NAMES, feature_vector


def rows(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

class ManifestDataset(Dataset):
    def __init__(self, records, classes, feature_cols, size=64, data_root="data/raw/WM811k_Dataset"):
        self.records, self.lookup, self.feature_cols, self.size = records, {x:i for i,x in enumerate(classes)}, feature_cols, size
        self.data_root = Path(data_root)
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r=self.records[i]; p=Path(r["path"]); p=p if p.is_absolute() else ROOT/self.data_root/p
        rgb=decode_wafer(p, (self.size,self.size)); im=Image.fromarray(rgb).convert("L")
        x=torch.from_numpy(np.asarray(im,dtype=np.float32)/255.).unsqueeze(0)
        f=torch.tensor(([float(r[c]) for c in self.feature_cols] if self.feature_cols[0].startswith("feat_") else feature_vector(rgb)),dtype=torch.float32)
        return x,f,self.lookup[r["label"]],str(p)

def predict(model, loader, device):
    model.eval(); ls=[]; ys=[]; es=[]; paths=[]
    with torch.no_grad():
        for x,f,y,p in loader:
            x,f=x.to(device),f.to(device); ls.append(model(x,f)); es.append(model.embed(x,f)); ys.append(y); paths.extend(p)
    return torch.cat(ls),torch.cat(ys).to(device),torch.cat(es),paths

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",default="data/processed/manifest.csv"); ap.add_argument("--data-root",default="data/raw/WM811k_Dataset"); ap.add_argument("--output",default="artifacts/model.pt"); ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rec=rows(a.manifest); classes=sorted({r["label"] for r in rec}); csv_features=sorted(c for c in rec[0] if c.startswith("feat_")); features=csv_features or list(FEATURE_NAMES); splits={s:[r for r in rec if r["split"]==s] for s in ("train","val","test")}
    if not all(splits.values()): raise ValueError("manifest must contain nonempty train, val, and test splits")
    ds={s:ManifestDataset(v,classes,features,data_root=a.data_root) for s,v in splits.items()}; loaders={s:DataLoader(v,batch_size=a.batch_size,shuffle=s=="train") for s,v in ds.items()}
    device=torch.device(a.device); model=WaferFusionCNN(len(classes),num_features=len(features)).to(device)
    counts=np.bincount([ds["train"].lookup[r["label"]] for r in splits["train"]],minlength=len(classes)); weights=torch.tensor(len(splits["train"])/(len(classes)*counts),dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); criterion=torch.nn.CrossEntropyLoss(weight=weights)
    best=None
    for epoch in range(a.epochs):
        model.train(); total=0.
        for x,f,y,_ in loaders["train"]:
            x,f,y=x.to(device),f.to(device),y.to(device); opt.zero_grad(); loss=criterion(model(x,f),y); loss.backward(); opt.step(); total+=loss.item()*len(y)
        val_logits,val_y,_,_=predict(model,loaders["val"],device); score=torch.nn.functional.cross_entropy(val_logits,val_y).item()
        print(json.dumps({"epoch":epoch+1,"train_loss":total/len(ds['train']),"val_nll":score}))
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); val_logits,val_y,_,_=predict(model,loaders["val"],device); scaler=TemperatureScaler().to(device).fit(val_logits,val_y)
    train_logits,train_y,train_emb,train_paths=predict(model,DataLoader(ds["train"],batch_size=a.batch_size),device)
    baseline=None
    if features:
        from sklearn.linear_model import LogisticRegression
        import joblib
        X=np.stack([ds["train"][i][1].numpy() for i in range(len(ds["train"]))]); y=np.array([classes.index(r["label"]) for r in splits["train"]])
        baseline=Path(a.output).with_name("baseline_logistic.joblib"); baseline.parent.mkdir(parents=True,exist_ok=True); joblib.dump(LogisticRegression(max_iter=2000,class_weight="balanced").fit(X,y),baseline)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"classes":classes,"feature_cols":features,"temperature":float(scaler.temperature.detach()),"config":{"num_classes":len(classes),"num_features":len(features)},"reference_embeddings":train_emb.cpu(),"reference_labels":[classes[int(x)] for x in train_y.cpu()],"reference_paths":train_paths,"seed":a.seed,"data_root":a.data_root},out)
    print(f"saved {out} (baseline: {baseline})")
if __name__=="__main__": main()
