"""Drawing helpers."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Intrinsics, project

PALETTE = {0: (60, 200, 60), 1: (40, 160, 255)}


def draw_poses(img: np.ndarray, poses, K: Intrinsics, axis_len: float = 0.015, show_label: bool = True) -> np.ndarray:
    out = img.copy()
    for p in poses:
        col = PALETTE.get(0 if p.class_name == "screw" else 1, (255, 255, 0))
        x1, y1, x2, y2 = map(int, p.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        o = np.array(p.position_m)
        if not np.all(np.isfinite(o)):
            continue
        R = np.array(p.rotation)
        pts = project(np.stack([o, o + R[:, 0] * axis_len, o + R[:, 1] * axis_len, o + R[:, 2] * axis_len]), K)
        o2 = tuple(int(t) for t in pts[0])
        for k, c in zip((1, 2, 3), ((0, 0, 255), (0, 220, 0), (255, 80, 0))):
            cv2.arrowedLine(out, o2, tuple(int(t) for t in pts[k]), c, 2, tipLength=0.25)
        if show_label:
            cv2.putText(out, f"{p.class_name} z={o[2]:.3f}m", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out
