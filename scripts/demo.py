"""Run detection (+ 6D pose if a depth map is given) on one image.

    # RGB only -> 2D boxes + in-plane screw axis
    python scripts/demo.py --rgb img.png --weights weights/screw_yolo11n.pt
    # RGB-D -> full 6D poses (depth in metres: .npy/.npz, or 16-bit PNG in mm with --depth-scale 0.001)
    python scripts/demo.py --rgb img.png --depth depth.npy --intrinsics configs/camera_example.json \
        --weights weights/screw_yolo11n.pt
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from screwpose import Intrinsics, PoseEstimator, ScrewDetector  # noqa: E402
from screwpose.orientation import estimate_axis  # noqa: E402
from screwpose.viz import draw_poses  # noqa: E402


def load_depth(path, scale):
    if path.endswith(".npz"):
        d = np.load(path)
        d = d[d.files[0]]
    elif path.endswith(".npy"):
        d = np.load(path)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    d = d.astype(np.float32) * scale
    d[d <= 0] = np.nan
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--rgb", required=True)
ap.add_argument("--depth")
ap.add_argument("--depth-scale", type=float, default=1.0, help="multiply depth by this to get metres")
ap.add_argument("--intrinsics", default="configs/camera_example.json")
ap.add_argument("--weights", default="weights/screw_yolo11n.pt")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="outputs")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
img = cv2.imread(a.rgb)
dets = ScrewDetector(a.weights, conf=a.conf, imgsz=a.imgsz)(img)
stem = os.path.splitext(os.path.basename(a.rgb))[0]
if a.depth:
    K = Intrinsics.from_json(a.intrinsics)
    poses = PoseEstimator(K).estimate(img, load_depth(a.depth, a.depth_scale), dets)
    out = [p.to_dict() for p in poses]
    vis = draw_poses(img, poses, K)
else:
    out, vis = [], img.copy()
    for i, d in enumerate(dets):
        ax = estimate_axis(img, d.xyxy)
        out.append(dict(id=i, class_name=d.class_name, confidence=round(d.confidence, 4),
                        bbox_xyxy=[round(t, 2) for t in d.xyxy], axis_angle_px_deg=ax.angle_deg))
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 200, 60) if d.class_id == 0 else (40, 160, 255), 2)
        if ax.angle_deg is not None:
            c = np.array(ax.center_px); L = 0.6 * max(x2 - x1, y2 - y1)
            t = np.radians(ax.angle_deg); e = c + L * np.array([np.cos(t), np.sin(t)])
            cv2.arrowedLine(vis, tuple(map(int, c)), tuple(map(int, e)), (0, 0, 255), 2, tipLength=0.2)
cv2.imwrite(os.path.join(a.out, f"{stem}_vis.jpg"), vis)
json.dump(dict(image=a.rgb, detections=out), open(os.path.join(a.out, f"{stem}.json"), "w"), indent=2)
print(f"{len(dets)} objects -> {a.out}/{stem}_vis.jpg, {a.out}/{stem}.json")
