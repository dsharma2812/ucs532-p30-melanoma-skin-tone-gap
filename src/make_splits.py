"""Build the ISIC 2019 subset and a leakage-safe train/val/test split.

Rules (from the proposal):
- Binary label: melanoma (MEL) = 1, everything else = 0.
- Rows that share a lesion_id (same lesion photographed more than once) always go to the SAME split.
  Images with no lesion_id are treated as their own lesion (a declared limitation: ISIC 2019 gives
  no patient id, so we can only group by lesion, not by patient).
- Subset: a random sample of N images, fixed seed, natural melanoma rate kept.
Output: results/splits.csv with columns image, label, group, split
"""
import argparse, os
import numpy as np, pandas as pd
from seed import set_seed


def build(meta_csv, gt_csv, n_subset, seed, frac=(0.70, 0.15, 0.15)):
    m = pd.read_csv(meta_csv); g = pd.read_csv(gt_csv)
    d = m.merge(g, on="image")
    d["label"] = (d["MEL"] == 1).astype(int)
    d["group"] = d["lesion_id"].fillna(d["image"])
    rng = np.random.default_rng(seed)
    # subsample whole groups until we reach n_subset images, so groups are never cut in half
    groups = d["group"].unique(); rng.shuffle(groups)
    sizes = d.groupby("group").size()
    keep, total = [], 0
    for gr in groups:
        if total >= n_subset: break
        keep.append(gr); total += sizes[gr]
    d = d[d["group"].isin(set(keep))].copy()
    # group-level split, stratified by "group contains a melanoma"
    gdf = d.groupby("group").agg(has_mel=("label", "max")).reset_index()
    split_of = {}
    for hm in (0, 1):
        ids = gdf.loc[gdf.has_mel == hm, "group"].to_numpy().copy(); rng.shuffle(ids)
        a = int(frac[0] * len(ids)); b = a + int(frac[1] * len(ids))
        for i, gr in enumerate(ids):
            split_of[gr] = "train" if i < a else ("val" if i < b else "test")
    d["split"] = d["group"].map(split_of)
    return d[["image", "label", "group", "split", "age_approx", "anatom_site_general", "sex"]]


def check_no_leak(d):
    g = d.groupby("group")["split"].nunique()
    assert (g == 1).all(), "a lesion group appears in more than one split"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/splits.csv")
    a = ap.parse_args(); set_seed(a.seed)
    d = build(a.meta, a.gt, a.n, a.seed); check_no_leak(d)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); d.to_csv(a.out, index=False)
    print(d.groupby("split").agg(images=("image", "size"), melanomas=("label", "sum"),
                                 lesions=("group", "nunique")).to_string())
    print("[ok] no lesion group crosses splits; subset images =", len(d))
