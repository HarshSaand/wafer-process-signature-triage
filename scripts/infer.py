#!/usr/bin/env python3
"""Infer one wafer image and retrieve similar training cases."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from wafer_process_ai.model import WaferFusionCNN
from wafer_process_ai.metrics import nearest_neighbors
from wafer_process_ai.data import decode_wafer
from wafer_process_ai.features import feature_vector
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--checkpoint",default="artifacts/model.pt"); ap.add_argument("--features",nargs="*",type=float,default=[]); ap.add_argument("--top-k",type=int,default=5); a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); n=ck["config"]["num_features"]
    rgb=decode_wafer(a.image,(64,64))
    values=a.features or feature_vector(rgb).tolist()
    if len(values)!=n: raise SystemExit(f"checkpoint expects {n} --features values in order: {ck['feature_cols']}")
    x=torch.from_numpy(np.asarray(Image.fromarray(rgb).convert("L"),dtype=np.float32)/255.).view(1,1,64,64); f=torch.tensor([values],dtype=torch.float32)
    model=WaferFusionCNN(**ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad(): logits=model(x,f)/ck.get("temperature",1.); p=torch.softmax(logits,1)[0]; emb=model.embed(x,f)[0]
    order=torch.argsort(p,descending=True)[:3]; result={"prediction":ck["classes"][int(order[0])],"confidence":float(p[order[0]]),"top3":[{"class":ck["classes"][int(i)],"probability":float(p[i])} for i in order]}
    if "reference_embeddings" in ck:
        nn=nearest_neighbors(emb.numpy(),ck["reference_embeddings"].numpy(),ck["reference_labels"],a.top_k)
        for item in nn: item["path"]=ck["reference_paths"][item["index"]]
        result["similar_training_cases"]=nn
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
