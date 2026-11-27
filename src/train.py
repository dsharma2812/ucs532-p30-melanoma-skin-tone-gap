"""Train one arm of the experiment and write validation/test scores.

arm=floor    frozen ImageNet EfficientNet-B0 features + a trained linear head (no fine-tuning)
arm=A        fine-tuned EfficientNet-B0, class-balanced loss weights (melanoma vs rest)
arm=B        A + sample weights = inverse frequency of the (ITA) tone group in the TRAINING set
arm=C        A + colour jitter (brightness/contrast/saturation 0.2, hue 0.05) on training images only (exploratory)
arm=control  A + same weighting scheme but group labels randomly permuted (tone-blind control)
--shuffle-labels   sanity check: permute training labels; test AUROC must be about 0.5
The model never sees the tone group as input; it is used only for the loss weights and for reporting.
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import models
from torchvision.transforms import ColorJitter
cj = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
from seed import set_seed
from metrics import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["floor", "A", "B", "control", "C"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--cache", default="cache"); ap.add_argument("--size", type=int, default=224)
ap.add_argument("--out", default="results/runs"); ap.add_argument("--shuffle-labels", action="store_true")
ap.add_argument("--pretrained", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); set_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={dev} torch={torch.__version__} args={vars(a)}", flush=True)

d = pd.read_csv(f"{a.cache}/subset_with_ita.csv"); X = np.load(f"{a.cache}/images_{a.size}.npy", mmap_mode="r")
if a.limit: d = d.groupby("split", group_keys=False).head(a.limit); 
idx = {s: d.index[d.split == s].to_numpy() for s in ["train", "val", "test"]}
y = d["label"].to_numpy().copy()
rng = np.random.default_rng(a.seed)
if a.shuffle_labels:
    y[idx["train"]] = rng.permutation(y[idx["train"]])
grp = d["ita_group"].to_numpy().copy()
if a.arm == "control":                      # same group sizes, meaningless membership
    grp[idx["train"]] = rng.permutation(grp[idx["train"]])

# ---- loss weights ----
w = np.ones(len(d))
tr = idx["train"]
pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)            # class balance (arms A, B, control)
if a.arm in ("B", "control"):
    freq = pd.Series(grp[tr]).value_counts(normalize=True)
    w[tr] = np.array([1.0 / freq[g] for g in grp[tr]]); w[tr] /= w[tr].mean()   # inverse group frequency, mean 1
    print("[weights] group freq in train:", freq.round(3).to_dict(), flush=True)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def batch(ids, train):
    x = torch.from_numpy(np.stack([X[i] for i in ids])).permute(0, 3, 1, 2).float().to(dev) / 255.0
    if train:
        if np.random.rand() < 0.5: x = x.flip(3)
        if np.random.rand() < 0.5: x = x.flip(2)
        if a.arm == "C": x = torch.stack([cj(xi) for xi in x]).clamp(0, 1)
    return (x - mean) / std

m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if a.pretrained else None)
m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
if a.arm == "floor":
    for p in m.features.parameters(): p.requires_grad = False
m.to(dev)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(reduction="none")

def predict(ids):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ids), 32):
            out.append(m(batch(ids[i:i + 32], False)).squeeze(1).cpu().numpy())
    return np.concatenate(out)

best, tag = -1, f"{a.arm}{'_shuffled' if a.shuffle_labels else ''}_seed{a.seed}"
os.makedirs(a.out, exist_ok=True)
for ep in range(a.epochs):
    m.train(); perm = rng.permutation(tr); t0 = time.time(); tot = 0
    for i in range(0, len(perm), a.bs):
        ids = perm[i:i + a.bs]
        xb = batch(ids, True); yb = torch.tensor(y[ids], dtype=torch.float32, device=dev)
        wb = torch.tensor(w[ids], dtype=torch.float32, device=dev) * (1 + (pos_w - 1) * yb)
        loss = (lossf(m(xb).squeeze(1), yb) * wb).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ids)
    sv = predict(idx["val"]); va = auroc(d["label"].to_numpy()[idx["val"]], sv)   # model selection uses VAL only
    print(f"[epoch {ep + 1}/{a.epochs}] train_loss={tot / len(tr):.4f} val_auroc={va:.4f} time={time.time() - t0:.0f}s", flush=True)
    if va > best:
        best = va; st = predict(idx["test"]); best_sv = sv
        os.makedirs(f"{a.out}/../ckpt", exist_ok=True); torch.save(m.state_dict(), f"{a.out}/../ckpt/{tag}.pt")   # best-epoch weights (for external eval, e.g. DDI)
out = pd.concat([pd.DataFrame({"image": d.image.to_numpy()[idx["val"]], "split": "val", "score": best_sv}),
                 pd.DataFrame({"image": d.image.to_numpy()[idx["test"]], "split": "test", "score": st})])
out.to_csv(f"{a.out}/{tag}_scores.csv", index=False)
print(f"[done] {tag} best_val_auroc={best:.4f} -> {a.out}/{tag}_scores.csv", flush=True)
