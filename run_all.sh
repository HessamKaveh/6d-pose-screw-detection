#!/usr/bin/env bash
# Reproduce every number and figure in the README (CPU is fine; ~4 h, dominated by training).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DEVICE=${DEVICE:-cpu}

# 1) data: UT57/synthetic_screws (MIT) -> data/ut57
[ -d data/ut57/images/test ] || $PY scripts/download_data.py data/ut57

# 2) detector
[ -f weights/screw_yolo11n.pt ] || $PY scripts/train.py --model yolo11n.pt --epochs 40 --imgsz 640 --batch 16 --device "$DEVICE"
$PY scripts/evaluate_detector.py --weights weights/screw_yolo11n.pt --split test --out results/detection

# 3) synthetic RGB-D benchmark with exact 6D ground truth (+ noise ablation sets)
$PY scripts/make_rgbd_benchmark.py --images data/ut57/images/test --labels data/ut57/labels/test \
    --out data/rgbd_bench --views 2 --seed 0
for ns in 0.5 1 2 4; do
  $PY scripts/make_rgbd_benchmark.py --images data/ut57/images/test --labels data/ut57/labels/test \
      --out data/rgbd_noise_$ns --views 1 --seed 1 --noise-scale $ns
done

# 4) 6D pose evaluation (GT boxes + detector boxes) and noise ablation (GT boxes)
$PY scripts/evaluate_pose.py --bench data/rgbd_bench --weights weights/screw_yolo11n.pt --out results/pose
for ns in 0.5 1 2 4; do
  $PY scripts/evaluate_pose.py --bench data/rgbd_noise_$ns --out results/pose_noise/$ns
done

# 5) figures + tests
$PY scripts/make_figures.py
$PY -m pytest -q tests
