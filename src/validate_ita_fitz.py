"""Validate the ITA tone estimate on retrievable Fitzpatrick17k images (clinical photos with human Fitzpatrick skin-type labels).
Samples up to --per-fst images per FST (1-6), tries to download each, computes ITA with src/ita.py exactly as used on ISIC
(ring mask + median Lab), and reports: retrievable counts per FST (selection bias), ITA vs FST correlation, and the
lighter/medium/darker confusion against FST bands (1-2 lighter, 3-4 medium, 5-6 darker).
Nothing is extrapolated: all numbers are for the retrieved images only. Cut-offs are NOT changed here."""
import argparse, io, sys, time, os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ita import ita_value, ita_group
ap = argparse.ArgumentParser()
ap.add_argument("--per-fst", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default="../results/fitz_ita")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = pd.read_csv("https://www.sfu.ca/~kabhishe/SkinDatasetsCritique/Fitzpatrick17k/metadata_updated.csv")
print("metadata rows", len(d), "columns", list(d.columns)[:12])
fc = [c for c in d.columns if "fitz" in c.lower()][0]
d = d[d.url.notna() & d[fc].isin([1, 2, 3, 4, 5, 6])]
print("rows with url and FST 1-6:", len(d)); print(d[fc].value_counts().sort_index().to_string())
samp = pd.concat([g.sample(min(len(g), a.per_fst), random_state=a.seed) for _, g in d.groupby(fc)]).reset_index(drop=True)

def work(row):
    try:
        r = urllib.request.urlopen(urllib.request.Request(row.url, headers={"User-Agent": "Mozilla/5.0"}), timeout=12)
        if not r.headers.get_content_type().startswith("image"):
            return None
        im = Image.open(io.BytesIO(r.read())).convert("RGB").resize((224, 224))
        x = np.asarray(im)
        return float(ita_value(x))
    except Exception:
        return None

t = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    vals = list(ex.map(work, [r for r in samp.itertuples()]))
samp["ita"] = vals; samp["fst"] = samp[fc]
samp["retrieved"] = samp.ita.notna() | [v is not None for v in vals]
samp["ok"] = samp.ita.notna()
samp["ita_group"] = [ita_group(v) if v == v else "unknown" for v in samp.ita]
samp["fst_group"] = samp.fst.map(lambda f: "lighter" if f <= 2 else ("medium" if f <= 4 else "darker"))
samp[["md5hash", "fst", "ita", "ita_group", "fst_group", "ok"]].to_csv(f"{a.out}/fitz_ita_per_image.csv", index=False)
print(f"elapsed {time.time()-t:.0f}s")
print("\nRETRIEVAL by FST (sampled / ITA computed):")
print(samp.groupby("fst").agg(sampled=("ok", "size"), ita_computed=("ok", "sum")).assign(rate=lambda x: (x.ita_computed / x.sampled).round(3)).to_string())
ok = samp[samp.ok]
print("\ntotal sampled", len(samp), "ITA computed", len(ok))
if len(ok) > 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr(ok.ita, ok.fst)
    print(f"\nSpearman(ITA, FST) = {rho:.3f} (p={p:.2g}); expected negative if ITA tracks FST")
    print("\nMedian / IQR ITA by FST:")
    print(ok.groupby("fst").ita.describe()[["count", "25%", "50%", "75%"]].round(1).to_string())
    print("\nITA group (rows) vs FST band (cols), counts:")
    print(pd.crosstab(ok.ita_group, ok.fst_group).to_string())
    print("\nITA group vs FST band, row %:")
    print((pd.crosstab(ok.ita_group, ok.fst_group, normalize="index") * 100).round(1).to_string())
    print("\nFST band recall (share of band assigned the same ITA group):")
    print((pd.crosstab(ok.fst_group, ok.ita_group, normalize="index") * 100).round(1).to_string())
