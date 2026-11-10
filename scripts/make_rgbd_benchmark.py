"""Build a synthetic RGB-D pose benchmark with exact 6D ground truth.

Each top-down test image of the detection dataset is used as the texture of a
physical plane (the work surface) that is placed in front of a virtual 720p
camera at a random distance, tilt and in-plane rotation. We render:

* RGB  - perspective warp of the texture (bilinear),
* depth - exact ray/plane depth, plus
    - object relief: every labelled object sticks out of the surface by 2-8 mm,
    - specular dropout: 30-70 % of the object pixels are missing (NaN),
    - stereo-like noise: sigma(z) = a + b*z^2, spatially correlated,
    - random holes elsewhere in the map.

Ground truth per object: contact point on the surface (box centre on the plane),
surface normal, projected bounding box and class id.

Usage:
    python scripts/make_rgbd_benchmark.py --images data/ut57/images/test \
        --labels data/ut57/labels/test --out data/rgbd_bench --views 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from screwpose.geometry import Intrinsics, project  # noqa: E402

W, H = 1280, 720
K = Intrinsics(900.0, 900.0, 640.0, 360.0, W, H)


def rot_from_axis_angle(axis, ang):
    return cv2.Rodrigues(np.asarray(axis, float) * ang / np.linalg.norm(axis))[0]


def sample_plane(rng, tex_w, tex_h, tilt_max):
    """Return (R_plane, t_plane, metres_per_px, tilt_deg, dist) with the plane covering the whole view.

    The tilt is drawn uniformly first; the texture scale is then chosen as the smallest one that
    covers the image (times 1-1.15), so high tilts are not under-sampled by rejection.
    """
    tilt_deg = rng.uniform(0, tilt_max)
    for _ in range(1000):
        dist = rng.uniform(0.25, 0.5)
        tilt = np.radians(tilt_deg)
        az = rng.uniform(0, 2 * np.pi)
        psi = rng.uniform(0, 2 * np.pi)
        # plane frame: columns = s-axis, t-axis, normal (normal points away from camera, +z)
        R = rot_from_axis_angle([np.cos(az), np.sin(az), 0], tilt) @ rot_from_axis_angle([0, 0, 1], psi)
        t = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), dist])
        n = R[:, 2]
        need = 0.0
        for (u, v) in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]:
            r = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
            p = R.T @ ((n @ t) / (n @ r) * r - t)
            need = max(need, abs(p[0]) / (tex_w / 2 - 3), abs(p[1]) / (tex_h / 2 - 3))
        mpp = need * rng.uniform(1.0, 1.15)
        if 0.40 <= mpp * tex_w <= 0.90:  # texture (work surface) 40-90 cm wide
            return R, t, mpp, tilt_deg, dist
    return None


def render(tex, labels, R, t, mpp, rng, noise_scale=1.0):
    th, tw = tex.shape[:2]
    n = R[:, 2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rays = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu)], -1)
    lam = (n @ t) / (rays @ n)
    P = rays * lam[..., None]
    loc = (P - t) @ R  # plane coords
    mapx = (loc[..., 0] / mpp + tw / 2).astype(np.float32)
    mapy = (loc[..., 1] / mpp + th / 2).astype(np.float32)
    rgb = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    depth = P[..., 2].copy()

    def tex2cam(s, q):
        p = np.array([(s - tw / 2) * mpp, (q - th / 2) * mpp, 0.0])
        return R @ p + t

    objects = []
    for (c, xc, yc, bw, bh) in labels:
        xc, yc, bw, bh = xc * tw, yc * th, bw * tw, bh * th
        corners = np.array([tex2cam(xc + dx * bw / 2, yc + dy * bh / 2) for dx in (-1, 1) for dy in (-1, 1)])
        uv = project(corners, K)
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        if x1 < 1 or y1 < 1 or x2 > W - 2 or y2 > H - 2:
            continue
        center = tex2cam(xc, yc)
        # relief + specular dropout inside the object footprint (ellipse in texture space)
        inside = (((mapx - xc) / (bw / 2)) ** 2 + ((mapy - yc) / (bh / 2)) ** 2) <= 1.0
        height = rng.uniform(0.002, 0.008)
        depth[inside] -= height * n[2]  # object surface is closer to the camera
        drop = inside & (rng.random(inside.shape) < rng.uniform(0.3, 0.7))
        depth[drop] = np.nan
        objects.append(dict(class_id=int(c), bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                            position_m=center.tolist(), normal=(-n).tolist(), height_m=height))

    # stereo-like, spatially correlated noise
    sigma = noise_scale * (0.0005 + 0.0025 * np.nan_to_num(depth, nan=0.5) ** 2)
    g = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 2.0)
    g /= g.std() + 1e-9
    depth = depth + sigma * g
    # random holes (blobs)
    holes = cv2.GaussianBlur(rng.random((H // 8, W // 8)).astype(np.float32), (0, 0), 1.5)
    holes = cv2.resize(holes, (W, H)) > np.quantile(holes, 0.95)
    depth[holes] = np.nan
    return rgb, depth.astype(np.float32), objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--tilt-max", type=float, default=40.0)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "intrinsics.json"), "w") as f:
        json.dump(dict(fx=K.fx, fy=K.fy, cx=K.cx, cy=K.cy, width=W, height=H), f, indent=2)
    files = sorted(glob(os.path.join(a.images, "*.jpg")) + glob(os.path.join(a.images, "*.png")))
    n_scenes = 0
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        tex = cv2.imread(fp)
        lab = np.loadtxt(os.path.join(a.labels, stem + ".txt"), ndmin=2)
        for v in range(a.views):
            sp = sample_plane(rng, tex.shape[1], tex.shape[0], a.tilt_max)
            if sp is None:
                continue
            R, t, mpp, tilt, dist = sp
            rgb, depth, objs = render(tex, lab, R, t, mpp, rng, a.noise_scale)
            if not objs:
                continue
            name = f"{stem}_v{v}"
            cv2.imwrite(os.path.join(a.out, name + "_rgb.png"), rgb)
            np.savez_compressed(os.path.join(a.out, name + "_depth.npz"), depth=depth)
            with open(os.path.join(a.out, name + "_gt.json"), "w") as f:
                json.dump(dict(scene=name, source=os.path.basename(fp), tilt_deg=tilt, distance_m=dist,
                               metres_per_px=mpp, plane_R=R.tolist(), plane_t=t.tolist(), objects=objs), f)
            n_scenes += 1
    print(f"wrote {n_scenes} scenes to {a.out}")


if __name__ == "__main__":
    main()
