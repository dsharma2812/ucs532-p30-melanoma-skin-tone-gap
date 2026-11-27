"""Check how many Fitzpatrick17k image URLs still return an image (follows redirects). Writes counts to stdout."""
import argparse, io, urllib.request
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
d = pd.read_csv("https://www.sfu.ca/~kabhishe/SkinDatasetsCritique/Fitzpatrick17k/metadata_updated.csv")
d = d[d.url.notna()].sample(a.n, random_state=a.seed)
ok = 0
for u in d.url:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=10)
        ok += r.headers.get_content_type().startswith("image")
    except Exception:
        pass
print(f"[fitz links] seed={a.seed} sampled={a.n} returned_image={ok}")
