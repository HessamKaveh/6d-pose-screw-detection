"""Lift 2D screw detections to 6D poses using a registered depth map.

For every detection:
  1. Sample depth on a ring around the box (the supporting surface, not the shiny
     metal part, whose depth is usually missing or noisy).
  2. Fit a local plane with RANSAC -> surface normal = approach axis (z).
  3. Intersect the viewing ray through the object centre with that plane ->
     3D position. This is robust to holes in the depth map on the object itself.
  4. Lift the in-plane image axis (from :mod:`screwpose.orientation`) onto the
     plane -> screw long axis (x). Symmetric objects (nuts, washers, screws seen
     head-on) get x from the camera x-axis.

No surface model, no hard-coded normal: each object gets its own local plane, so
the method works on tilted, curved (piecewise-planar) and multi-surface scenes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .detector import Detection
from .geometry import (
    Intrinsics,
    backproject,
    fit_plane_ransac,
    intersect_ray_plane,
    ray,
    rotation_from_normal_and_axis,
    rotmat_to_quat,
)
from .orientation import estimate_axis


@dataclass
class ScrewPose:
    id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    position_m: list[float]  # contact point on the supporting surface, camera frame
    rotation: list[list[float]]  # 3x3, columns = x (screw axis), y, z (approach)
    quaternion_xyzw: list[float]
    normal: list[float]  # surface normal, pointing to the camera
    axis_angle_px_deg: float | None  # in-image tip->head angle, None if symmetric
    plane_inlier_ratio: float
    n_depth_points: int
    status: str  # "ok" | "fallback"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PoseConfig:
    ring_scale: float = 2.2  # outer ring = box scaled by this factor
    inner_scale: float = 1.15  # exclude the (slightly dilated) box itself
    min_ring_px: int = 12  # minimum half-size of the ring in pixels
    plane_thresh_m: float = 0.003
    ransac_iters: int = 150
    max_points: int = 2000
    min_points: int = 30
    min_depth_m: float = 0.05
    max_depth_m: float = 5.0
    perspective_center: bool = True  # see PoseEstimator._perspective_center


class PoseEstimator:
    def __init__(self, K: Intrinsics, cfg: PoseConfig | None = None, seed: int = 0):
        self.K = K
        self.cfg = cfg or PoseConfig()
        self.rng = np.random.default_rng(seed)

    def _ring_points(self, depth: np.ndarray, xyxy) -> np.ndarray:
        c = self.cfg
        h, w = depth.shape
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw = max((x2 - x1) / 2 * c.ring_scale, c.min_ring_px)
        hh = max((y2 - y1) / 2 * c.ring_scale, c.min_ring_px)
        X1, X2 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
        Y1, Y2 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
        sub = depth[Y1:Y2, X1:X2]
        vv, uu = np.mgrid[Y1:Y2, X1:X2]
        iw, ih = (x2 - x1) / 2 * c.inner_scale, (y2 - y1) / 2 * c.inner_scale
        outside = (np.abs(uu - cx) > iw) | (np.abs(vv - cy) > ih)
        valid = outside & np.isfinite(sub) & (sub > c.min_depth_m) & (sub < c.max_depth_m)
        u, v, z = uu[valid], vv[valid], sub[valid]
        if len(z) > c.max_points:
            idx = self.rng.choice(len(z), c.max_points, replace=False)
            u, v, z = u[idx], v[idx], z[idx]
        return backproject(u.astype(float), v.astype(float), z.astype(float), self.K)

    def _perspective_center(self, xyxy, normal, d, axis):
        """Perspective-correct object centre on the plane.

        Under perspective the 2D box centre is *not* the image of the 3D centre. For an
        elongated part the two ends project (approximately) onto opposite box corners, so
        back-projecting that diagonal onto the plane and taking the 3D midpoint removes the
        bias. For symmetric parts the midpoints of both diagonals are averaged.
        """
        if not self.cfg.perspective_center:
            return None
        x1, y1, x2, y2 = xyxy
        P = [intersect_ray_plane(ray(px, py, self.K), normal, d) for px, py in ((x1, y1), (x2, y2), (x2, y1), (x1, y2))]
        if any(p is None for p in P):
            return None
        m_main, m_anti = (P[0] + P[1]) / 2, (P[2] + P[3]) / 2  # TL-BR and TR-BL diagonals
        if axis is not None and axis.angle_deg is not None:
            a = np.radians(axis.angle_deg)
            return m_main if np.cos(a) * np.sin(a) >= 0 else m_anti
        return (m_main + m_anti) / 2

    def estimate_one(self, i: int, image_bgr: np.ndarray, depth_m: np.ndarray, det: Detection) -> ScrewPose:
        c = self.cfg
        x1, y1, x2, y2 = det.xyxy
        axis = estimate_axis(image_bgr, det.xyxy) if image_bgr is not None else None
        u, v = ((x1 + x2) / 2, (y1 + y2) / 2)

        P = self._ring_points(depth_m, det.xyxy)
        status, inl_ratio = "ok", 0.0
        normal, d = None, None
        if len(P) >= c.min_points:
            normal, d, inl = fit_plane_ransac(P, c.plane_thresh_m, c.ransac_iters, self.rng)
            inl_ratio = float(inl.mean())
        r = ray(u, v, self.K)
        pos = None
        if normal is not None:
            pos = self._perspective_center(det.xyxy, normal, d, axis)
            if pos is None:
                pos = intersect_ray_plane(r, normal, d)
        if pos is None:
            # fallback: median depth in/around the box, camera-facing normal
            status = "fallback"
            z = float(np.median(P[:, 2])) if len(P) else float("nan")
            if not np.isfinite(z):
                box = depth_m[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
                box = box[np.isfinite(box) & (box > c.min_depth_m)]
                z = float(np.median(box)) if box.size else float("nan")
            pos = r * z
            normal, d = np.array([0.0, 0.0, -1.0]), z

        axis_hint = None
        if axis is not None and axis.angle_deg is not None:
            a = np.radians(axis.angle_deg)
            L = max(x2 - x1, y2 - y1) / 2
            p_head = intersect_ray_plane(ray(u + L * np.cos(a), v + L * np.sin(a), self.K), normal, d)
            p_tail = intersect_ray_plane(ray(u - L * np.cos(a), v - L * np.sin(a), self.K), normal, d)
            if p_head is not None and p_tail is not None:
                axis_hint = p_head - p_tail
        R = rotation_from_normal_and_axis(normal, axis_hint)
        return ScrewPose(
            id=i,
            class_name=det.class_name,
            confidence=round(float(det.confidence), 4),
            bbox_xyxy=[round(float(t), 2) for t in det.xyxy],
            position_m=[round(float(t), 5) for t in pos],
            rotation=np.round(R, 6).tolist(),
            quaternion_xyzw=np.round(rotmat_to_quat(R), 6).tolist(),
            normal=np.round(normal, 6).tolist(),
            axis_angle_px_deg=None if axis is None or axis.angle_deg is None else round(axis.angle_deg, 2),
            plane_inlier_ratio=round(inl_ratio, 3),
            n_depth_points=int(len(P)),
            status=status,
        )

    def estimate(self, image_bgr: np.ndarray, depth_m: np.ndarray, detections: list[Detection]) -> list[ScrewPose]:
        return [self.estimate_one(i, image_bgr, depth_m, d) for i, d in enumerate(detections)]


def naive_pose(depth_m: np.ndarray, K: Intrinsics, det: Detection) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: depth at the box centre + fixed camera-facing normal."""
    x1, y1, x2, y2 = det.xyxy
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    z = depth_m[int(round(v)), int(round(u))]
    if not np.isfinite(z) or z <= 0:
        return np.full(3, np.nan), np.array([0.0, 0.0, -1.0])
    return ray(u, v, K) * z, np.array([0.0, 0.0, -1.0])


def box_median_pose(depth_m: np.ndarray, K: Intrinsics, det: Detection) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: median of the valid depth inside the box + fixed camera-facing normal."""
    x1, y1, x2, y2 = det.xyxy
    box = depth_m[int(y1):int(np.ceil(y2)) + 1, int(x1):int(np.ceil(x2)) + 1]
    box = box[np.isfinite(box) & (box > 0)]
    if box.size == 0:
        return np.full(3, np.nan), np.array([0.0, 0.0, -1.0])
    return ray((x1 + x2) / 2, (y1 + y2) / 2, K) * float(np.median(box)), np.array([0.0, 0.0, -1.0])
