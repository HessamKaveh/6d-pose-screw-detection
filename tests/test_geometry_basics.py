"""Camera model, plane fitting, ray-plane intersection and rotation/quaternion helpers."""
import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from screwpose import Intrinsics
from screwpose.geometry import (angle_between, backproject, fit_plane_lsq, fit_plane_ransac, intersect_ray_plane,
                                project, ray, rotation_from_normal_and_axis, rotmat_to_quat)

K = Intrinsics(900.0, 910.0, 640.0, 360.0, 1280, 720)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_intrinsics_matrix_and_json(tmp_path):
    assert np.allclose(K.K, [[900, 0, 640], [0, 910, 360], [0, 0, 1]])
    Kj = Intrinsics.from_json(os.path.join(ROOT, "configs", "camera_example.json"))
    assert (Kj.fx, Kj.fy, Kj.cx, Kj.cy, Kj.width, Kj.height) == (900.0, 900.0, 640.0, 360.0, 1280, 720)
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0}))
    assert Intrinsics.from_json(str(p)) == Intrinsics(1.0, 2.0, 3.0, 4.0)


def test_backproject_project_roundtrip():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 1280, 50), rng.uniform(0, 720, 50)
    z = rng.uniform(0.2, 3.0, 50)
    P = backproject(u, v, z, K)
    assert np.allclose(P[:, 2], z)
    assert np.allclose(project(P, K), np.stack([u, v], 1))


def test_ray_through_principal_point_is_optical_axis():
    assert np.allclose(ray(K.cx, K.cy, K), [0, 0, 1])
    r = ray(100.0, 200.0, K)
    assert np.allclose(project(2.5 * r, K), [[100.0, 200.0]])


def test_intersect_ray_plane():
    n = np.array([0.0, 0.0, -1.0])  # camera-facing plane z = 0.8  <=>  n.x + 0.8 = 0
    r = ray(900.0, 100.0, K)
    p = intersect_ray_plane(r, n, 0.8)
    assert p[2] == pytest.approx(0.8)
    assert np.allclose(np.cross(p, r), 0)  # lies on the ray
    assert intersect_ray_plane(np.array([1.0, 0, 0]), n, 0.8) is None  # parallel
    assert intersect_ray_plane(r, n, -0.8) is None  # plane behind the camera


def test_fit_plane_lsq_exact():
    rng = np.random.default_rng(2)
    n = np.array([0.3, -0.2, -0.93]); n /= np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    ab = rng.uniform(-1, 1, (100, 2))
    P = 0.7 * -n + ab[:, :1] * e1 + ab[:, 1:] * e2  # n.x + 0.7 = 0
    nh, d = fit_plane_lsq(P)
    s = np.sign(nh @ n)
    assert np.allclose(s * nh, n, atol=1e-9) and s * d == pytest.approx(0.7)


def test_ransac_normal_points_to_camera_and_rejects_degenerate():
    P = backproject(np.array([0.0, 1280, 0, 1280, 640]), np.array([0.0, 0, 720, 720, 360]), np.full(5, 1.2), K)
    n, d, inl = fit_plane_ransac(P, 1e-4, 20)
    assert n[2] < 0 and np.allclose(n, [0, 0, -1]) and d == pytest.approx(1.2) and inl.all()
    with pytest.raises(ValueError):
        fit_plane_ransac(P[:2])


@pytest.mark.parametrize("normal,hint", [([0, 0, -1.0], None), ([0.2, -0.4, -0.9], [1.0, 1.0, 0.0]),
                                         ([0, 0, -1.0], [0, 0, 1.0])])  # last: hint parallel to the normal
def test_rotation_frame(normal, hint):
    normal = np.array(normal) / np.linalg.norm(normal)
    R = rotation_from_normal_and_axis(normal, None if hint is None else np.array(hint))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(R[:, 2], -normal)  # z = approach direction, into the surface
    if hint is not None and abs(np.dot(hint, normal)) < 0.99 * np.linalg.norm(hint):
        h = np.array(hint) - np.dot(hint, R[:, 2]) * R[:, 2]
        assert angle_between(R[:, 0], h) < 1e-4  # x = hint projected onto the surface


def test_fronto_parallel_frame_is_identity():
    assert np.allclose(rotation_from_normal_and_axis(np.array([0, 0, -1.0]), None), np.eye(3))


def test_rotmat_to_quat_xyzw():
    Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.allclose(rotmat_to_quat(np.eye(3)), [0, 0, 0, 1])
    q = rotmat_to_quat(Rz90)
    q = q * np.sign(q[3])
    assert np.allclose(q, [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    R = Rotation.random(random_state=3).as_matrix()
    assert np.allclose(Rotation.from_quat(rotmat_to_quat(R)).as_matrix(), R)


def test_angle_between():
    assert angle_between(np.array([1.0, 0, 0]), np.array([0, 2.0, 0])) == pytest.approx(90.0)
    assert angle_between(np.array([1.0, 1, 0]), np.array([2.0, 2, 0])) == pytest.approx(0.0, abs=1e-4)
    assert angle_between(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) == pytest.approx(180.0)
