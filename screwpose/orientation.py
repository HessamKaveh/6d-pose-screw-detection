"""In-plane orientation of a screw from its RGB crop (no learning needed).

The object is separated from the local background by colour distance to the
crop border, then a PCA of the foreground pixels gives the long axis. The
head/tip ambiguity is resolved with the skewness of the pixel distribution along
the axis: the head is the heavy end, the thread is the long thin tail.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class InPlaneAxis:
    angle_deg: float | None  # image-plane angle of the tip->head direction, None if symmetric
    elongation: float  # sqrt(lambda1 / lambda2) of the foreground pixel cloud
    center_px: tuple[float, float]  # centroid of the foreground in full-image pixels
    n_pixels: int


def estimate_axis(
    image_bgr: np.ndarray,
    xyxy: tuple[float, float, float, float],
    margin: float = 0.15,
    min_elongation: float = 1.6,
) -> InPlaneAxis:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    X1 = int(max(0, np.floor(x1 - margin * bw)))
    Y1 = int(max(0, np.floor(y1 - margin * bh)))
    X2 = int(min(w, np.ceil(x2 + margin * bw)))
    Y2 = int(min(h, np.ceil(y2 + margin * bh)))
    crop = image_bgr[Y1:Y2, X1:X2]
    cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return InPlaneAxis(None, 1.0, (cx0, cy0), 0)

    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=-1)
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # keep the connected component closest to the box centre
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return InPlaneAxis(None, 1.0, (cx0, cy0), 0)
    c_local = np.array([cx0 - X1, cy0 - Y1])
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    dists = np.linalg.norm(cents[1:] - c_local, axis=1)
    score = areas / (1.0 + dists)
    k = 1 + int(np.argmax(score))
    ys, xs = np.nonzero(lbl == k)
    if len(xs) < 10:
        return InPlaneAxis(None, 1.0, (cx0, cy0), int(len(xs)))

    pts = np.stack([xs, ys], 1).astype(float)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    elong = float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9)))
    center = (float(mu[0] + X1), float(mu[1] + Y1))
    if elong < min_elongation:
        return InPlaneAxis(None, elong, center, len(xs))

    axis = evecs[:, 1]
    proj = (pts - mu) @ axis
    skew = float(np.mean(proj**3) / (np.std(proj) ** 3 + 1e-9))
    if skew > 0:  # long tail on the + side => tail = thread, so head is on the - side
        axis = -axis
    return InPlaneAxis(float(np.degrees(np.arctan2(axis[1], axis[0]))), elong, center, len(xs))
