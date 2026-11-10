"""Evaluate 6D lifting on the synthetic RGB-D benchmark.

Compares the proposed local-plane method against two baselines that use a
camera-facing normal with (a) the depth at the box centre or (b) the median depth in
the box. Uses either ground-truth boxes (isolates the 3D part) or detector boxes
(full pipeline).

    python scripts/evaluate_pose.py --bench data/rgbd_bench --weights weights/screw_yolo11n.pt --out results/pose
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from screwpose import Detection, Intrinsics, PoseEstimator  # noqa: E402
from screwpose.geometry import angle_between  # noqa: E402
from screwpose.pose import box_median_pose, naive_pose  # noqa: E402

NAMES = {0: "screw", 1: "nut"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gts, preds, thr=0.5):
    """Greedy (by confidence) one-to-one matching. Returns {gt_index: pred}."""
    out, used = {}, set()
    for p in sorted(preds, key=lambda d: -d.confidence):
        best, bi = thr, None
        for i, g in enumerate(gts):
            if i in used or g["class_id"] != p.class_id:
                continue
            v = iou(g["bbox_xyxy"], p.xyxy)
            if v >= best:
                best, bi = v, i
        if bi is not None:
            used.add(bi)
            out[bi] = p
    return out


def summarize(rows):
    def stats(key):
        v = np.array([r[key] for r in rows], float)
        f = v[np.isfinite(v)]
        return dict(mean=float(f.mean()) if len(f) else None, median=float(np.median(f)) if len(f) else None,
                    p90=float(np.percentile(f, 90)) if len(f) else None, valid=float(len(f) / max(len(v), 1)))
    pe, ne = np.array([r["pos_err_mm"] for r in rows]), np.array([r["normal_err_deg"] for r in rows])
    ok = np.nan_to_num(pe, nan=1e9)
    return dict(n=len(rows), pos_err_mm=stats("pos_err_mm"), normal_err_deg=stats("normal_err_deg"),
                success_5mm_5deg=float(np.mean((ok < 5) & (np.nan_to_num(ne, nan=1e9) < 5))),
                success_10mm_10deg=float(np.mean((ok < 10) & (np.nan_to_num(ne, nan=1e9) < 10))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/pose")
    ap.add_argument("--save-vis", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = Intrinsics.from_json(os.path.join(a.bench, "intrinsics.json"))
    est = PoseEstimator(K)
    det = None
    if a.weights:
        from screwpose import ScrewDetector
        det = ScrewDetector(a.weights, conf=a.conf, imgsz=a.imgsz)
    rows, n_gt, n_pred, n_tp = [], 0, 0, 0
    method_axis_log = []
    scenes = sorted(glob(os.path.join(a.bench, "*_gt.json")))
    for si, gp in enumerate(scenes):
        g = json.load(open(gp))
        base = gp[: -len("_gt.json")]
        rgb = cv2.imread(base + "_rgb.png")
        depth = np.load(base + "_depth.npz")["depth"]
        gts = g["objects"]
        modes = {"gt_box": {i: Detection(tuple(o["bbox_xyxy"]), 1.0, o["class_id"], NAMES[o["class_id"]])
                            for i, o in enumerate(gts)}}
        if det is not None:
            preds = det(rgb)
            m = match(gts, preds)
            n_gt += len(gts); n_pred += len(preds); n_tp += len(m)
            modes["pred_box"] = m
            if si < a.save_vis:
                from screwpose.viz import draw_poses
                vis = draw_poses(rgb, est.estimate(rgb, depth, preds), K)
                cv2.imwrite(os.path.join(a.out, f"vis_{g['scene']}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for mode, dets in modes.items():
            for i, d in dets.items():
                o = gts[i]
                gpos, gn = np.array(o["position_m"]), np.array(o["normal"])
                p = est.estimate_one(i, rgb, depth, d)
                cands = (("ours", np.array(p.position_m), np.array(p.normal)),
                         ("center_depth", *naive_pose(depth, K, d)),
                         ("box_median", *box_median_pose(depth, K, d)))
                if method_axis_log is not None and mode == "gt_box" and p.axis_angle_px_deg is not None:
                    x_plane = np.array(g["plane_R"]).T @ np.array(p.rotation)[:, 0]
                    loc = np.array(g["plane_R"]).T @ (gpos - np.array(g["plane_t"])) / g["metres_per_px"]
                    method_axis_log.append(dict(source=g["source"], obj=(round(loc[0]), round(loc[1])),
                                                class_id=o["class_id"],
                                                angle=float(np.degrees(np.arctan2(x_plane[1], x_plane[0])))))
                for method, pos, nrm in cands:
                    rows.append(dict(scene=g["scene"], mode=mode, method=method, class_id=o["class_id"],
                                     tilt_deg=g["tilt_deg"], distance_m=g["distance_m"],
                                     pos_err_mm=float(np.linalg.norm(pos - gpos) * 1000),
                                     normal_err_deg=angle_between(nrm, gn),
                                     status=p.status if method == "ours" else "-"))
        if si % 50 == 0:
            print(f"{si}/{len(scenes)}", flush=True)

    with open(os.path.join(a.out, "per_object.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {}
    for mode in ("gt_box", "pred_box"):
        for method in ("ours", "center_depth", "box_median"):
            sel = [r for r in rows if r["mode"] == mode and r["method"] == method]
            if sel:
                summary[f"{mode}/{method}"] = summarize(sel)
                for c, nm in NAMES.items():
                    sc = [r for r in sel if r["class_id"] == c]
                    if sc:
                        summary[f"{mode}/{method}/{nm}"] = summarize(sc)
    if det is not None:
        summary["detection_on_rgbd_bench"] = dict(gt=n_gt, pred=n_pred, tp_iou50=n_tp,
                                                  recall=n_tp / max(n_gt, 1), precision=n_tp / max(n_pred, 1))
    # in-plane axis view-consistency: same object, two random views -> same axis in the plane frame
    by = {}
    for r in method_axis_log:
        by.setdefault((r["source"], r["obj"]), []).append(r)
    d_line, d_dir = [], []
    for v in by.values():
        if len(v) == 2 and v[0]["class_id"] == 0:
            d = (v[0]["angle"] - v[1]["angle"] + 180) % 360 - 180
            d_dir.append(abs(d))
            d_line.append(min(abs(d), 180 - abs(d)))
    if d_line:
        d_line, d_dir = np.array(d_line), np.array(d_dir)
        summary["axis_view_consistency_screws"] = dict(
            pairs=len(d_line), line_err_median_deg=float(np.median(d_line)),
            line_within_5deg=float(np.mean(d_line < 5)), line_within_10deg=float(np.mean(d_line < 10)),
            head_direction_agreement=float(np.mean(d_dir[d_line < 10] < 90)) if np.any(d_line < 10) else None)
    summary["n_scenes"] = len(scenes)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k.count("/") <= 1}, indent=2))


if __name__ == "__main__":
    main()
