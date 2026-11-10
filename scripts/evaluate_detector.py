"""Evaluate the detector on the held-out test split (COCO-style mAP via Ultralytics).

    python scripts/evaluate_detector.py --weights weights/screw_yolo11n.pt
"""
import argparse
import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from screwpose.data import resolve_data_yaml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="configs/ut57_screws.yaml")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--split", default="test")
ap.add_argument("--out", default="results/detection")
a = ap.parse_args()
a.data = resolve_data_yaml(a.data)

m = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, batch=8, conf=0.001, iou=0.6,
                        project=os.path.abspath(a.out), name=a.split, exist_ok=True, plots=True)
res = {"split": a.split, "imgsz": a.imgsz,
       "all": {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
               "mAP50-95": float(m.box.map)}}
for i, c in enumerate(m.box.ap_class_index):
    name = m.names[int(c)]
    p, r, ap50, ap = m.box.class_result(i)
    res[name] = {"precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}
res["speed_ms_per_img"] = {k: float(v) for k, v in m.speed.items()}
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, f"metrics_{a.split}.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
