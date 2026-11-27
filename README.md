# UCS532 P30: Skin-tone gap in melanoma detection (AI-assisted FIRST VERSION for professor review)

**Status: first version prepared with AI assistance for instructor review. Real results for 18 training runs are in tables and scores. DDI is NOT YET RUN. Student review and confirmation are pending.**

Student: Dishita Sharma (roll 1024170081), solo. Problem P30, Category E.

## Problem
Does an EfficientNet-B0 melanoma classifier trained on ISIC 2019 detect fewer melanomas in darker-skin groups, how much does reweighting reduce the gap, and how far can the ITA tone groups be trusted? (See the proposal; hypotheses are fixed there and in `docs/PREDECLARED_EXPECTATIONS.md`.)

## Data (not stored in this repo)
- ISIC 2019 Training: https://challenge.isic-archive.com/data/ (CC-BY-NC, cite as the ISIC 2019 aggregate: Tschandl 2018, Codella 2018, Combalia 2019). 25,331 images, 4,522 melanoma, 23,247 with a `lesion_id` (11,847 lesions), 2,084 without.
- Subset used: 8,000 images chosen by whole lesion groups with seed 0 (`results/splits.csv`; counts in `logs/splits.log`).
- Fitzpatrick17k metadata (human skin-type labels, image URLs): https://github.com/mattgroh/fitzpatrick17k ; check status of image links is logged in `logs/` (TO BE FILLED from a real run).

## Reproduce
```
pip install -r requirements.txt
export ISIC_ZIP=/path/to/ISIC_2019_Training_Input.zip
SEEDS="0 1 2" bash run_all.sh   # script default is SEEDS="0 1"; EPOCHS=5 N=8000. Arm C is in src/train.py (see CHANGES_LOG)
```
Seeds are fixed and printed (`[seed] SEED=...`) at the top of every log.

## Hardware and runtime
Colab free T4, about 5 minutes per run.

## Results
See the report PDF and tables. Gap (lighter minus darker sensitivity) is negative in all 15 unshuffled runs; darker test group has 17 melanomas.

## Credits (reused code, models, data)
- EfficientNet-B0 ImageNet weights: torchvision (Tan & Le 2019).
- ISIC 2019 data: ISIC archive; BCN20000 and HAM10000 components.
- Fitzpatrick17k: Groh et al. 2021.
- ITA bands: Del Bino & Bernerd 2013 (literature cut-offs). Everything else in `src/` was written for this project with AI help.

## AI declaration
The proposal, the code in `src/`, `tests/`, `run_all.sh`, the notebook, and the report/slide drafts were generated with AI assistance (an AI assistant working for the student) and have NOT yet been confirmed by the student. The student must review them and be able to explain every file. Synthetic-data smoke tests were used only to check that scripts run; no synthetic number appears in any result.

## Limits
ISIC 2019 has no patient id, only lesion id, so splits are by lesion. ITA is an estimate from pixels and is affected by lighting. See the report's limitations.
