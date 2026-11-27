"""Turn saved scores into the report tables.

For each run: AUROC (test), threshold fixed on VALIDATION at 90% specificity, then per tone group on TEST:
sensitivity, false-positive rate, number of melanomas, bootstrap 95% CI (resampling lesion groups).
Gap = sensitivity(lighter) - sensitivity(darker). Also lighter-group change B - A.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from metrics import auroc, threshold_at_specificity, sensitivity, false_positive_rate, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache"); ap.add_argument("--runs", default="results/runs")
ap.add_argument("--out", default="results/tables"); ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(f"{a.cache}/subset_with_ita.csv").drop(columns=["split"]).set_index("image")
rows = []
for f in sorted(glob.glob(f"{a.runs}/*_scores.csv")):
    tag = os.path.basename(f).replace("_scores.csv", ""); arm, seed = re.match(r"(.*)_seed(\d+)", tag).groups()
    s = pd.read_csv(f); v = s[s.split == "val"].join(d, on="image"); t = s[s.split == "test"].join(d, on="image")
    thr = threshold_at_specificity(v.label.values, v.score.values, 0.90)
    base = dict(run=tag, arm=arm, seed=int(seed), threshold=thr, test_auroc=auroc(t.label.values, t.score.values))
    for g in ["all", "lighter", "medium", "darker"]:
        tg = t if g == "all" else t[t.ita_group == g]
        r = dict(base, group=g, n_images=len(tg), n_melanoma=int(tg.label.sum()),
                 sensitivity=sensitivity(tg.label.values, tg.score.values, thr),
                 fpr=false_positive_rate(tg.label.values, tg.score.values, thr))
        lo, hi = bootstrap_ci(tg.label.values, tg.score.values, None, thr, sensitivity, a.boot, 0, tg["group"].values)
        r.update(sens_ci_low=lo, sens_ci_high=hi); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{a.out}/per_run_group_metrics.csv", index=False)
piv = R.pivot_table(index=["arm", "seed"], columns="group", values="sensitivity")
if {"lighter", "darker"} <= set(piv.columns):
    piv["gap_lighter_minus_darker"] = piv["lighter"] - piv["darker"]
piv.to_csv(f"{a.out}/sensitivity_and_gap.csv")
print(R.round(3).to_string(index=False)); print(); print(piv.round(3).to_string())
