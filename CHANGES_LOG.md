# Changes from the proposal (keep updating; copy into the report)
- 24 Nov: schedule is tight (final deadline 29 Nov per brief). DDI, Arm C and a third seed stay in the plan; any drop will be decided on measured runtime or coverage and recorded here as a proposed fallback until approved.
- 24 Nov: ISIC 2019 has lesion_id for 91.8% of images but no patient id, so grouping is by lesion.
- 25 Nov: third seed (seeds 0-2) run for floor, A, B, control and shuffled-label. All real runs on a Colab T4.
- 25 Nov: Arm C (A + ColorJitter 0.2/0.2/0.2/0.05) added and run for 3 seeds, exploratory. 18 training runs in total (12 core + 3 shuffled + 3 Arm C).
- 25 Nov: src/train.py now contains Arm C and saves checkpoints (results/ckpt). The original runs saved no weights; src/train_original_no_checkpoints.py is the previous version.
- 25 Nov: src/validate_ita_fitz.py added (ITA vs Fitzpatrick17k check; 829 of 3,000 sampled images retrievable).
- 26 Nov: src/score_ddi.py added. NOT YET RUN. DDI access and terms acceptance are pending with the student; DDI scoring needs retrained checkpoints.
