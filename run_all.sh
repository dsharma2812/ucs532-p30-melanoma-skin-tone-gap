#!/usr/bin/env bash
# Reproduces the main results table. Needs a GPU (tested target: Colab T4). Set ISIC_ZIP to the downloaded ISIC_2019_Training_Input.zip.
set -euo pipefail
cd "$(dirname "$0")"
: "${ISIC_ZIP:?set ISIC_ZIP=/path/to/ISIC_2019_Training_Input.zip}"
SEEDS="${SEEDS:-0 1}"; EPOCHS="${EPOCHS:-5}"; N="${N:-8000}"
mkdir -p data results/runs logs
# Optional durable store: set PERSIST=/path (for example a mounted Google Drive folder). Finished runs and the image cache are copied
# there after each step and restored at start, so a dropped Colab runtime only loses the run in progress.
PERSIST="${PERSIST:-}"
if [ -n "$PERSIST" ]; then mkdir -p "$PERSIST"; cp -ru "$PERSIST"/results "$PERSIST"/cache . 2>/dev/null || true; fi
save() { [ -z "$PERSIST" ] || { mkdir -p "$PERSIST"; cp -ru results cache logs "$PERSIST"/ 2>/dev/null || true; }; }
run() { # run <tag> <train args...>: skip if scores already exist
  local tag="$1"; shift
  if [ -f "../results/runs/${tag}_scores.csv" ]; then echo "[skip] $tag already done"; return; fi
  python train.py "$@" --cache ../cache --out ../results/runs | tee ../logs/train_${tag}.log; save
}
[ -f data/ISIC_2019_Training_Metadata.csv ] || curl -L -o data/ISIC_2019_Training_Metadata.csv https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
[ -f data/ISIC_2019_Training_GroundTruth.csv ] || curl -L -o data/ISIC_2019_Training_GroundTruth.csv https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
python -m pytest -q tests
cd src
python make_splits.py --meta ../data/ISIC_2019_Training_Metadata.csv --gt ../data/ISIC_2019_Training_GroundTruth.csv --n "$N" --seed 0 --out ../results/splits.csv | tee ../logs/splits.log
if [ ! -f ../cache/images_224.npy ]; then python prep_images.py --zip "$ISIC_ZIP" --splits ../results/splits.csv --out ../cache | tee ../logs/prep.log; save; fi
for s in $SEEDS; do
  for arm in floor A B control; do run "${arm}_seed${s}" --arm $arm --seed $s --epochs $EPOCHS; done
  run "A_shuffled_seed${s}" --arm A --seed $s --epochs $EPOCHS --shuffle-labels
done
python evaluate.py --cache ../cache --runs ../results/runs --out ../results/tables | tee ../logs/evaluate.log; save
