"""Camera geometry helpers: back-projection, robust plane fitting, rotations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_json(cls, path: str) -> "Intrinsics":
        import json

        with open(path) as f:
            d = json.load(f)
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], d.get("width"), d.get("height"))


def ray(u: float, v: float, K: Intrinsics) -> np.ndarray:
    """Unit-less viewing ray through pixel (u, v), with z = 1."""
    return np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Pixels + metric depth (z along optical axis) -> Nx3 points in camera frame."""
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=-1)


def project(P: np.ndarray, K: Intrinsics) -> np.ndarray:
    P = np.atleast_2d(P)
    return np.stack([K.fx * P[:, 0] / P[:, 2] + K.cx, K.fy * P[:, 1] / P[:, 2] + K.cy], axis=-1)


def fit_plane_lsq(P: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane n.x + d = 0 (unit n) through Nx3 points."""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def fit_plane_ransac(
    P: np.ndarray,
    thresh: float = 0.002,
    iters: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit followed by a least-squares refit on the inliers.

    Returns (normal, d, inlier_mask). The normal is oriented towards the camera
    (i.e. n.z < 0, since the camera looks along +z).
    """
    if len(P) < 3:
        raise ValueError("need at least 3 points")
    rng = rng or np.random.default_rng(0)
    best = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -n @ a
        inl = np.abs(P @ n + d) < thresh
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best = cnt, inl
    if best is None or best_count < 3:
        best = np.ones(len(P), bool)
    n, d = fit_plane_lsq(P[best])
    inl = np.abs(P @ n + d) < thresh
    if inl.sum() >= 3:
        n, d = fit_plane_lsq(P[inl])
    if n[2] > 0:
        n, d = -n, -d
    return n, d, inl


def intersect_ray_plane(r: np.ndarray, n: np.ndarray, d: float) -> np.ndarray | None:
    denom = n @ r
    if abs(denom) < 1e-9:
        return None
    t = -d / denom
    return t * r if t > 0 else None


def rotation_from_normal_and_axis(normal: np.ndarray, axis_hint: np.ndarray | None) -> np.ndarray:
    """Build a right-handed frame [x y z] (columns) in the camera frame.

    z: approach direction = -normal (points *into* the surface, i.e. away from the camera)
    x: the screw's long axis projected onto the surface (or the camera x-axis when the
       object is rotationally symmetric / no hint is available)
    """
    z = -normal / np.linalg.norm(normal)
    hint = np.array([1.0, 0.0, 0.0]) if axis_hint is None else np.asarray(axis_hint, float)
    x = hint - (hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
