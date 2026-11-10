"""Thin wrapper around an Ultralytics YOLO detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class ScrewDetector:
    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.6, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def __call__(self, image_bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(image_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            c = int(b.cls[0])
            out.append(Detection(tuple(float(t) for t in b.xyxy[0].tolist()), float(b.conf[0]), c, self.names[c]))
        return out


def detections_from_yolo_labels(label_path: str, width: int, height: int, names: dict[int, str]) -> list[Detection]:
    """Ground-truth YOLO txt labels -> Detection objects (confidence = 1)."""
    dets = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
            dets.append(Detection(((xc - bw / 2) * width, (yc - bh / 2) * height,
                                   (xc + bw / 2) * width, (yc + bh / 2) * height), 1.0, c, names.get(c, str(c))))
    return dets
