"""In-plane axis from the RGB crop, the full 6D lifting on synthetic RGB-D, and the baselines."""
import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from screwpose import Detection, Intrinsics, PoseEstimator
from screwpose.detector import detections_from_yolo_labels
from screwpose.geometry import angle_between, ray
from screwpose.orientation import estimate_axis
from screwpose.pose import box_median_pose, naive_pose

K = Intrinsics(900.0, 900.0, 320.0, 240.0, 640, 480)


def synthetic_screw(theta_deg, centre=(320.0, 240.0), shape=(480, 640)):
    """Dark screw (thin shaft + round head) on a bright surface; the head is at +theta."""
    img = np.full((*shape, 3), 200, np.uint8)
    c = np.array(centre); a = np.radians(theta_deg); d = np.array([np.cos(a), np.sin(a)])
    tail, head = c - 45 * d, c + 35 * d
    cv2.line(img, tuple(np.int32(tail)), tuple(np.int32(head)), (40, 40, 40), 6)
    cv2.circle(img, tuple(np.int32(head)), 14, (40, 40, 40), -1)
    pts = np.stack([tail, head])
    return img, (*(pts.min(0) - 16), *(pts.max(0) + 16))


@pytest.mark.parametrize("theta", [0, 30, 90, -120, 170])
def test_axis_points_from_tip_to_head(theta):
    img, box = synthetic_screw(theta)
    ax = estimate_axis(img, box)
    assert ax.elongation > 3
    assert abs((ax.angle_deg - theta + 180) % 360 - 180) < 2


def test_round_object_is_symmetric():
    img = np.full((240, 240, 3), 200, np.uint8)
    cv2.circle(img, (120, 120), 25, (40, 40, 40), -1)
    ax = estimate_axis(img, (90, 90, 150, 150))
    assert ax.angle_deg is None
    assert ax.center_px == pytest.approx((120, 120), abs=1)


def test_degenerate_box_is_handled():
    ax = estimate_axis(np.zeros((50, 50, 3), np.uint8), (10, 10, 11, 11))
    assert ax.angle_deg is None and ax.n_pixels == 0


def test_full_pose_on_fronto_parallel_surface():
    theta = 30.0
    img, box = synthetic_screw(theta)
    depth = np.full(img.shape[:2], 0.5, np.float32)
    depth[img[..., 0] < 100] = np.nan  # no depth on the metal part
    p = PoseEstimator(K).estimate(img, depth, [Detection(box, 0.8, 0, "screw")])[0]
    assert p.status == "ok" and p.plane_inlier_ratio > 0.99
    assert p.position_m[2] == pytest.approx(0.5, abs=1e-4)
    R = np.array(p.rotation)
    assert np.allclose(R[:, 2], [0, 0, 1], atol=1e-4)  # approach = into the table
    a = np.radians(theta)
    assert angle_between(R[:, 0], np.array([np.cos(a), np.sin(a), 0])) < 2  # x = tail->head on the plane
    assert np.allclose(Rotation.from_quat(p.quaternion_xyzw).as_matrix(), R, atol=1e-5)
    json.dumps(p.to_dict())  # serialisable output


def test_fallback_without_surface_depth():
    depth = np.full((480, 640), np.nan, np.float32)
    box = (300.0, 220.0, 340.0, 260.0)
    depth[220:261, 300:341] = 0.4  # only the object itself has depth
    p = PoseEstimator(K).estimate(None, depth, [Detection(box, 0.5, 1, "nut")])[0]
    assert p.status == "fallback" and p.n_depth_points == 0
    assert np.allclose(p.position_m, ray(320, 240, K) * 0.4)
    assert np.allclose(p.normal, [0, 0, -1])


def test_baselines():
    depth = np.full((480, 640), 0.6, np.float32)
    det = Detection((310.0, 230.0, 330.0, 250.0), 1.0, 0, "screw")
    for f in (naive_pose, box_median_pose):
        pos, n = f(depth, K, det)
        assert np.allclose(pos, [0, 0, 0.6]) and np.allclose(n, [0, 0, -1])
    depth[:] = np.nan
    for f in (naive_pose, box_median_pose):
        assert np.isnan(f(depth, K, det)[0]).all()


def test_detections_from_yolo_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.25 0.5\n1 0.1 0.2 0.1 0.1\n\n")
    dets = detections_from_yolo_labels(str(f), 640, 480, {0: "screw"})
    assert len(dets) == 2
    assert dets[0].xyxy == pytest.approx((240, 120, 400, 360)) and dets[0].class_name == "screw"
    assert dets[1].class_name == "1" and dets[1].confidence == 1.0
