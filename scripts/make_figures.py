"""Produce the figures used in the README from results/.

    python scripts/make_figures.py
"""
import glob
import json
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"ours": "#2a78d6", "box_median": "#eb6834", "center_depth": "#1baf7a"}
LAB = {"ours": "Ours (local plane + ray)", "box_median": "Box median depth", "center_depth": "Centre-pixel depth"}
MARK = {"ours": "o", "box_median": "s", "center_depth": "^"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})


def success(df):
    return ((df.pos_err_mm.fillna(1e9) < 5) & (df.normal_err_deg < 5)).mean() * 100


# 1) position-error CDF + success vs tilt (GT boxes)
df = pd.read_csv("results/pose/per_object.csv")
g = df[df["mode"] == "gt_box"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for m in ("ours", "box_median", "center_depth"):
    e = np.sort(g[g.method == m].pos_err_mm.fillna(np.inf).values)
    y = np.arange(1, len(e) + 1) / len(e) * 100
    ax[0].plot(e, y, color=COL[m], lw=2, label=LAB[m])
ax[0].set_xlim(0, 12); ax[0].set_ylim(0, 100)
ax[0].set_xlabel("3D position error (mm)"); ax[0].set_ylabel("objects within error (%)")
ax[0].set_title("Position error CDF (ground-truth boxes)", color=INK, loc="left")
ax[0].legend(loc="lower right")
bins = [0, 10, 20, 30, 40]
centers = [5, 15, 25, 35]
for m in ("ours", "box_median", "center_depth"):
    s = g[g.method == m]
    ys = [success(s[(s.tilt_deg >= lo) & (s.tilt_deg < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
    ax[1].plot(centers, ys, color=COL[m], lw=2, marker=MARK[m], ms=8, label=LAB[m])
ax[1].set_xticks(centers, ["0-10", "10-20", "20-30", "30-40"]); ax[1].set_ylim(0, 102)
ax[1].set_xlabel("surface tilt w.r.t. camera (deg)"); ax[1].set_ylabel("success: <5 mm and <5° (%)")
ax[1].set_title("Success rate vs. surface tilt", color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/pose_error.png", dpi=150); plt.close(fig)

# 2) depth-noise ablation
scales = sorted(float(os.path.basename(p)) for p in glob.glob("results/pose_noise/*"))
if scales:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for m in ("ours", "box_median", "center_depth"):
        med, suc = [], []
        for sc in scales:
            key = str(sc).rstrip("0").rstrip(".") if sc != int(sc) else str(int(sc))
            d = json.load(open(f"results/pose_noise/{key}/summary.json"))[f"gt_box/{m}"]
            med.append(d["pos_err_mm"]["median"]); suc.append(d["success_5mm_5deg"] * 100)
        ax[0].plot(scales, med, color=COL[m], lw=2, marker=MARK[m], ms=8, label=LAB[m])
        ax[1].plot(scales, suc, color=COL[m], lw=2, marker=MARK[m], ms=8, label=LAB[m])
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(scales, [f"×{s:g}" for s in scales])
        a.set_xlabel("depth-noise multiplier (×1: σ = 0.5 mm + 2.5 mm·z², ≈1.4 mm at 0.6 m)")
    ax[0].set_ylabel("median position error (mm)"); ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel("success <5 mm & <5° (%)"); ax[1].set_ylim(0, 102)
    ax[0].set_title("Robustness to depth noise", color=INK, loc="left"); ax[1].legend(loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/noise_ablation.png", dpi=150); plt.close(fig)

# 3) qualitative montage of full-pipeline results
vis = sorted(glob.glob("results/pose/vis_*.jpg"))[:4]
if vis:
    ims = [cv2.resize(cv2.imread(v), (640, 360)) for v in vis]
    while len(ims) < 4:
        ims.append(np.full_like(ims[0], 252))
    cv2.imwrite(f"{OUT}/qualitative_pose.jpg", np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:4])]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

# 4) training curves
rc = "runs/screw_yolo11n/results.csv"
if os.path.exists(rc):
    r = pd.read_csv(rc)
    r.columns = [c.strip() for c in r.columns]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].plot(r.epoch, r["train/box_loss"], color=COL["ours"], lw=2, label="train box loss")
    ax[0].plot(r.epoch, r["val/box_loss"], color=COL["box_median"], lw=2, label="val box loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Detector training", color=INK, loc="left")
    ax[1].plot(r.epoch, r["metrics/mAP50(B)"] * 100, color=COL["ours"], lw=2, label="mAP@0.5")
    ax[1].plot(r.epoch, r["metrics/mAP50-95(B)"] * 100, color=COL["box_median"], lw=2, label="mAP@0.5:0.95")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation mAP (%)"); ax[1].set_ylim(0, 100); ax[1].legend()
    ax[1].set_title("Validation mAP", color=INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/training.png", dpi=150); plt.close(fig)
print("figures ->", OUT)
