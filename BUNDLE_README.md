# UCS532 P30 results bundle (AI-assisted first version)

All numbers come from real training runs on a free Colab T4. Nothing here is synthetic. DDI is NOT YET RUN and no DDI data is in this bundle.

Run counts: 18 training jobs = 12 core matrix (floor, A, B, control x seeds 0-2) + 3 shuffled-label (A, seeds 0-2) + 3 Arm C (A + colour jitter, exploratory).
"All 15 unshuffled runs" = the 12 core + 3 Arm C.

- tables/per_run_group_metrics.csv, sensitivity_and_gap.csv : core matrix + shuffled, seeds 0-2
- tables/arm_C/ : Arm C tables
- tables/merged_all_arms_per_run_group.csv : all 18 runs, per tone group (sensitivity, FPR, bootstrap CI)
- scores/ : per-image test scores for all 18 runs
- logs/ : data prep, splits, training, evaluation logs
- fitzpatrick/ : ITA vs Fitzpatrick17k per-image output and log (829 retrievable of 3,000 sampled)
- splits.csv : lesion-grouped split (train 5,635 / val 1,201 / test 1,164)
- repo/ : code. src/train.py = original + Arm C + checkpoint saving. src/score_ddi.py is NOT YET RUN.

Seed ranges are min to max over three seeds, not confidence intervals. Darker test group: 52 images, 17 melanomas.
Arm C result: no clear change in the gap in these runs. That is not evidence of no effect.
Not done: DDI, manual ITA review on ISIC, GitHub publishing, student confirmation. Checkpoints were not saved for the 18 runs (DDI would need retraining).
