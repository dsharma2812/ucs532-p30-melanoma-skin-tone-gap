"""Score saved EfficientNet-B0 checkpoints on DDI (Diverse Dermatology Images) and report sensitivity by FST group.
NOT YET RUN. Needs DDI images + ddi_metadata.csv from Stanford AIMI/Redivis (access pending). Column names below follow the public
DDI-Code loader (ddi_metadata.csv: DDI_file, skin_tone in {12,34,56}, malignant, disease); verify against the real file before use.

Label mapping (declare in the report): our models were trained on ISIC melanoma vs everything else. DDI's label is malignant vs benign
(includes melanoma, SCC, BCC and others). We report (a) malignant=1 as the positive class, the closest available label, and
(b) melanoma-only positives vs benign negatives (diseases containing 'melanoma', non-melanoma malignant rows dropped).
Threshold: the same 90%-specificity-on-ISIC-validation cut-off stored per run (from results/tables/per_run_group_metrics.csv); no tuning on DDI.
Usage: python score_ddi.py --ddi-dir /content/ddi --ckpt results/ckpt --thresholds results/tables/per_run_group_metrics.csv --out results/ddi
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import models
from metrics import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--ddi-dir", required=True); ap.add_argument("--ckpt", default="results/ckpt")
ap.add_argument("--thresholds", default="results/tables/per_run_group_metrics.csv"); ap.add_argument("--out", default="results/ddi")
ap.add_argument("--size", type=int, default=224); ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = pd.read_csv(os.path.join(a.ddi_dir, "ddi_metadata.csv"))
meta["malignant"] = meta["malignant"].astype(str).str.lower().isin(["true", "1", "yes"])
meta["fst_group"] = meta["skin_tone"].map({12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"})
meta["is_melanoma"] = meta["disease"].str.contains("melanoma", case=False, na=False)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def load(f):  # same preprocessing idea as ISIC cache: resize to size x size RGB, scale 0-1, ImageNet norm
    im = Image.open(os.path.join(a.ddi_dir, f)).convert("RGB").resize((a.size, a.size), Image.BILINEAR)
    return np.asarray(im)
X = np.stack([load(f) for f in meta["DDI_file"]])
thr = pd.read_csv(a.thresholds).query("group == 'all'").set_index("run")["threshold"].to_dict()
rng = np.random.default_rng(0); rows = []
def sens_ci(s, y, t):
    p = s[y == 1] > t
    if len(p) == 0: return np.nan, np.nan, np.nan
    b = [rng.choice(p, len(p)).mean() for _ in range(a.n_boot)]
    return p.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
for ck in sorted(glob.glob(f"{a.ckpt}/*.pt")):
    tag = os.path.basename(ck)[:-3]
    m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(ck, map_location=dev)); m.to(dev).eval(); sc = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = torch.from_numpy(X[i:i + 64]).permute(0, 3, 1, 2).float().to(dev) / 255.0
            sc.append(m((x - mean) / std).squeeze(1).cpu().numpy())
    sc = np.concatenate(sc); meta[f"score_{tag}"] = sc
    for lab, mask, y in [("malignant_vs_benign", np.ones(len(meta), bool), meta.malignant.values.astype(int)),
                         ("melanoma_vs_benign", (meta.is_melanoma | ~meta.malignant).values, meta.is_melanoma.values.astype(int))]:
        for g in ["all", "FST I-II", "FST III-IV", "FST V-VI"]:
            k = mask & ((meta.fst_group == g).values if g != "all" else True)
            s_, y_ = sc[k], y[k]
            if y_.sum() == 0 or (y_ == 0).sum() == 0: continue
            se, lo, hi = sens_ci(s_, y_, thr[tag]) if tag in thr else (np.nan,) * 3
            fpr = (s_[y_ == 0] > thr[tag]).mean() if tag in thr else np.nan
            rows.append(dict(run=tag, label=lab, group=g, n=len(y_), n_pos=int(y_.sum()), auroc=auroc(y_, s_), sensitivity=se, ci_low=lo, ci_high=hi, fpr=fpr))
pd.DataFrame(rows).to_csv(f"{a.out}/ddi_group_metrics.csv", index=False); meta.to_csv(f"{a.out}/ddi_scores.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
