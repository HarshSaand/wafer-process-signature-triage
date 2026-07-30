#!/usr/bin/env python3
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
sns.set_theme(style='whitegrid'); navy='#082B4C'; teal='#00A6A6'; gold='#E4B363'; red='#D1495B'
m=json.loads((ROOT/'outputs/metrics.json').read_text()); classes=list(m['per_class'])

fig,ax=plt.subplots(figsize=(8.3,6.5)); sns.heatmap(np.array(m['confusion_matrix']),annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes,ax=ax,cbar=False)
ax.set(xlabel='Predicted class',ylabel='True class',title='Cartesian-polar fusion CNN: held-out test confusion matrix'); plt.xticks(rotation=40,ha='right'); plt.tight_layout(); fig.savefig(OUT/'confusion_matrix.png',dpi=180); plt.close(fig)

f1=[m['per_class'][c]['f1'] for c in classes]; fig,ax=plt.subplots(figsize=(8.3,4.6)); ax.bar(classes,f1,color=[teal if x>=.8 else gold for x in f1]); ax.axhline(m['macro_f1'],color=navy,ls='--',label=f"macro-F1 = {m['macro_f1']:.3f}"); ax.set_ylim(0,1.05); ax.set_ylabel('F1'); ax.set_title('Per-class held-out performance'); ax.legend(); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); fig.savefig(OUT/'per_class_f1.png',dpi=180); plt.close(fig)

base=m['baseline_logistic']; fig,ax=plt.subplots(figsize=(6.8,4.2)); names=['Engineered-feature\nlogistic baseline','Cartesian + polar\nfusion CNN']; vals=[base['macro_f1'],m['macro_f1']]; bars=ax.bar(names,vals,color=[gold,teal]); ax.set_ylim(0,1); ax.set_ylabel('Held-out macro-F1'); ax.set_title('Measured model comparison');
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.3f}',ha='center',weight='bold'); plt.tight_layout(); fig.savefig(OUT/'model_comparison.png',dpi=180); plt.close(fig)

counts={};
with open(ROOT/'data/processed/manifest.csv') as f:
 for r in csv.DictReader(f): counts[r['label']]=counts.get(r['label'],0)+1
fig,ax=plt.subplots(figsize=(8.3,4.3)); ax.bar(counts.keys(),counts.values(),color=navy); ax.set_ylabel('Images'); ax.set_title('Shared Kaggle derivative: class counts (n=902)'); ax.set_ylim(0,115); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); fig.savefig(OUT/'class_counts.png',dpi=180); plt.close(fig)

# Top-to-bottom implemented workflow illustration.
fig,ax=plt.subplots(figsize=(7.2,8)); ax.axis('off'); boxes=[('1  Kaggle image derivative','902 labeled 32x32 RGB JPEG heatmaps'),('2  Audit + grouped split','Hash/dHash groups; 627 / 137 / 138'),('3  Two representations','Cartesian image + differentiable polar unwrap'),('4  Fusion model','Dual CNN embeddings + proxy spatial descriptors'),('5  Validation-only calibration','Temperature scaling; no test tuning'),('6  Engineer review output','Class, confidence, spatial signatures, similar wafers')]
ys=np.linspace(.92,.08,len(boxes))
for i,((title,sub),y) in enumerate(zip(boxes,ys)):
 ax.add_patch(plt.Rectangle((.08,y-.055),.84,.095,facecolor='#EAF4F4',edgecolor=teal,lw=2)); ax.text(.11,y+.012,title,fontsize=12,weight='bold',color=navy,va='center'); ax.text(.11,y-.022,sub,fontsize=9.5,color='#334E68',va='center')
 if i<len(boxes)-1: ax.annotate('',xy=(.5,ys[i+1]+.045),xytext=(.5,y-.06),arrowprops=dict(arrowstyle='-|>',color=navy,lw=1.5))
ax.set_title('Wafer Process-Signature Triage - implemented system flow',fontsize=16,weight='bold',color=navy,pad=14); fig.savefig(OUT/'system_flow.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Dataset examples, clearly presented as rendered images.
dirs=sorted((ROOT/'data/raw/WM811k_Dataset').iterdir()); fig,axs=plt.subplots(3,3,figsize=(7.5,7.5))
for ax,d in zip(axs.flat,dirs): ax.imshow(Image.open(next(d.glob('*.jpg')))); ax.set_title(d.name); ax.axis('off')
fig.suptitle('Representative rendered JPEGs (not original ternary die arrays)',fontsize=14,weight='bold'); plt.tight_layout(); fig.savefig(OUT/'dataset_examples.png',dpi=180); plt.close(fig)
print(OUT)
