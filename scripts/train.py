"""Train the screw/nut detector (Ultralytics YOLO11).

    python scripts/train.py --model yolo11n.pt --epochs 40 --imgsz 640 --device cpu
"""
import argparse
import os
import shutil
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from screwpose.data import resolve_data_yaml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="configs/ut57_screws.yaml")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default=None)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--name", default="screw_yolo11n")
ap.add_argument("--project", default="runs")
a = ap.parse_args()
a.data = resolve_data_yaml(a.data)

model = YOLO(a.model)
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
    project=os.path.abspath(a.project), name=a.name, exist_ok=True, seed=0, deterministic=True,
    patience=20, cos_lr=True, close_mosaic=10,
    # top-down parts on a table: any in-plane rotation and flips are valid
    degrees=180.0, flipud=0.5, fliplr=0.5, scale=0.5, translate=0.1, mosaic=1.0,
    hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, plots=True,
)
best = os.path.join(a.project, a.name, "weights", "best.pt")
os.makedirs("weights", exist_ok=True)
shutil.copy(best, os.path.join("weights", f"{a.name}.pt"))
print("saved", os.path.join("weights", f"{a.name}.pt"))
