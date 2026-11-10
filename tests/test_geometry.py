import numpy as np

from screwpose import Detection, Intrinsics, PoseConfig, PoseEstimator
from screwpose.geometry import project
from screwpose.orientation import InPlaneAxis
from screwpose.geometry import (angle_between, fit_plane_ransac, intersect_ray_plane, ray,
                                rotation_from_normal_and_axis)

K = Intrinsics(900.0, 900.0, 640.0, 360.0, 1280, 720)


def tilted_plane_depth(n_away, d0=0.5):
    """Depth map of the plane n.x = d0 (n pointing away from the camera)."""
    uu, vv = np.meshgrid(np.arange(1280.0), np.arange(720.0))
    r = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu)], -1)
    return (d0 / (r @ n_away)).astype(np.float32)


def test_ransac_recovers_plane_with_outliers():
    rng = np.random.default_rng(1)
    n = np.array([0.2, -0.3, 0.93]); n /= np.linalg.norm(n)
    pts = rng.uniform(-0.1, 0.1, (500, 3)); pts[:, 2] = 0
    # put points on the plane n.x = 0.6
    e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    P = 0.6 * n + pts[:, :1] * e1 + pts[:, 1:2] * e2
    P[:100] += rng.uniform(-0.05, 0.05, (100, 3))  # 20 % outliers
    nh, d, inl = fit_plane_ransac(P, 0.002, 300, rng)
    assert angle_between(nh, -n) < 0.5
    assert abs(d - 0.6) < 1e-3
    assert inl[100:].mean() > 0.99


def test_rotation_is_proper():
    R = rotation_from_normal_and_axis(np.array([0.1, 0.2, -0.97]), np.array([1.0, 0.5, 0.0]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_pose_on_tilted_plane_with_hole():
    n = np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])
    depth = tilted_plane_depth(n, 0.5)
    box = (600.0, 330.0, 680.0, 390.0)
    depth[335:385, 605:675] = np.nan  # no depth on the (shiny) object itself
    est = PoseEstimator(K, PoseConfig(perspective_center=False))
    p = est.estimate(None, depth, [Detection(box, 0.9, 0, "screw")])[0]
    gt = intersect_ray_plane(ray(640, 360, K), -n, 0.5)
    assert p.status == "ok"
    assert np.linalg.norm(np.array(p.position_m) - gt) < 1e-4
    assert angle_between(np.array(p.normal), -n) < 0.1


def test_perspective_center_recovers_segment_midpoint():
    """A thin screw on a 35 deg tilted plane: the 2D box centre is biased, the diagonal fix is exact."""
    n = np.array([0.0, np.sin(np.radians(35)), np.cos(np.radians(35))])
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.cross(n, e1)
    c3 = intersect_ray_plane(ray(500, 250, K), -n, 0.4)
    a, b = c3 - 0.03 * (e1 + e2) / np.sqrt(2), c3 + 0.03 * (e1 + e2) / np.sqrt(2)
    uv = project(np.stack([a, b]), K)
    box = (*uv.min(0), *uv.max(0))
    ang = float(np.degrees(np.arctan2(*(uv[1] - uv[0])[::-1])))
    est = PoseEstimator(K)
    naive = intersect_ray_plane(ray((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, K), -n, 0.4)
    fixed = est._perspective_center(box, -n, 0.4, InPlaneAxis(ang, 10.0, (0, 0), 100))
    assert np.linalg.norm(naive - c3) > 5e-4
    assert np.linalg.norm(fixed - c3) < 1e-6
