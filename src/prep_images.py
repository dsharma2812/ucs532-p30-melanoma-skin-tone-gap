"""Read only the subset images from the big ISIC zip, resize to 224x224, cache as one uint8 array,
and compute the ITA tone estimate for each image. Nothing here is committed to git (data stays out of the repo)."""
import argparse, io, zipfile
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from ita import ita_value, ita_group

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--splits", default="results/splits.csv")
ap.add_argument("--out", default="cache"); ap.add_argument("--size", type=int, default=224)
a = ap.parse_args()
import os; os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(a.splits)
z = zipfile.ZipFile(a.zip)
names = {os.path.basename(n).rsplit(".", 1)[0]: n for n in z.namelist() if n.lower().endswith(".jpg")}
X = np.zeros((len(d), a.size, a.size, 3), dtype=np.uint8); ita = np.zeros(len(d))
for i, img in enumerate(tqdm(d["image"])):
    im = Image.open(io.BytesIO(z.read(names[img]))).convert("RGB").resize((a.size, a.size))
    X[i] = np.asarray(im); ita[i] = ita_value(X[i])
np.save(f"{a.out}/images_{a.size}.npy", X)
d["ita"] = ita; d["ita_group"] = [ita_group(v) for v in ita]
d.to_csv(f"{a.out}/subset_with_ita.csv", index=False)
print(d.groupby(["split", "ita_group"]).agg(n=("image", "size"), mel=("label", "sum")).to_string())
