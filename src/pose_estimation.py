import json
import numpy as np
import cv2
import os
from ultralytics import YOLO

# ─── تنظیمات ───────────────────────────────────────────
INTRINSICS_PATH = "config/camera_intrinsics.json"
YOLO_WEIGHTS    = "weights/best.pt"
RGB_PATH        = "data/raw/rgb/000003.png"
DEPTH_NPY_PATH  = "data/raw/depth_npy/000003.npy"
CONFIDENCE      = 0.1
EXPAND_FACTOR   = 2.5
BOX_W = 0.012
BOX_H = 0.012
BOX_D = 0.008

# normal ثابت سطح ریل — از میانگین اندازه‌گیری‌های قبلی
# اگر دوربین جابجا شد، این را دوباره کالیبره کن
RAIL_NORMAL = np.array([0.084, -0.980, -0.179])
RAIL_NORMAL = RAIL_NORMAL / np.linalg.norm(RAIL_NORMAL)
# ────────────────────────────────────────────────────────

COLORS = [
    (0,255,0),(0,200,255),(255,100,0),
    (255,0,200),(0,255,200),(180,180,0),
]


def np_encoder(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    cam = data["color_camera"]["camera_matrix"]
    return {"fx":cam[0][0],"fy":cam[1][1],"cx":cam[0][2],"cy":cam[1][2]}


def pixel_to_3d(u, v, depth_mm, intr):
    z = depth_mm / 1000.0
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z])


def project_3d_to_2d(pt3d, intr):
    x, y, z = pt3d
    if z <= 0: return None
    return (int(x*intr["fx"]/z + intr["cx"]),
            int(y*intr["fy"]/z + intr["cy"]))


def get_rail_depth(depth_npy, x1, y1, x2, y2, expand_factor,
                   img_h, img_w, screw_depth_mm):
    """فقط depth میانگین سطح ریل اطراف screw"""
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ex1 = int(max(0,     cx - bw*expand_factor/2))
    ex2 = int(min(img_w, cx + bw*expand_factor/2))
    ey1 = int(max(0,     cy - bh*expand_factor/2))
    ey2 = int(min(img_h, cy + bh*expand_factor/2))

    mask_expand = np.zeros(depth_npy.shape, dtype=bool)
    mask_expand[ey1:ey2, ex1:ex2] = True
    mask_screw  = np.zeros(depth_npy.shape, dtype=bool)
    mask_screw[int(y1):int(y2), int(x1):int(x2)] = True
    mask_rail   = mask_expand & ~mask_screw & (depth_npy > 0)

    depths = depth_npy[mask_rail].astype(float)
    valid  = np.abs(depths - screw_depth_mm) < 80.0
    depths = depths[valid]

    if len(depths) < 5:
        return screw_depth_mm, (ex1,ey1,ex2,ey2)

    d_low, d_high = np.percentile(depths, 15), np.percentile(depths, 85)
    depths = depths[(depths >= d_low) & (depths <= d_high)]

    return float(depths.mean()) if len(depths) > 0 else screw_depth_mm, (ex1,ey1,ex2,ey2)


def rotation_from_normal(normal):
    """
    normal = جهت عمود بر سطح (به سمت دوربین)
    Z محور = جهت فرود ابزار (عمود بر سطح)
    X محور = در صفحه سطح، در جهت افقی
    Y محور = در صفحه سطح، در جهت عمودی
    """
    # Z محور ابزار = به سمت سطح
    z_axis = -normal

    # X محور: cross product با world_up
    # از world up استفاده می‌کنیم تا X همیشه تقریباً افقی باشد
    world_up = np.array([0.0, 1.0, 0.0])
    x_axis   = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 0.01:
        world_up = np.array([1.0, 0.0, 0.0])
        x_axis   = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_to_euler(R):
    """ZYX convention"""
    pitch = np.degrees(np.arcsin(-R[2,0]))
    if abs(R[2,0]) < 0.9999:
        roll = np.degrees(np.arctan2(R[2,1], R[2,2]))
        yaw  = np.degrees(np.arctan2(R[1,0], R[0,0]))
    else:
        roll = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        yaw  = 0.0
    return roll, pitch, yaw


def draw_3d_box(img, center_3d, R, intr, color, screw_id):
    w, h, d = BOX_W/2, BOX_H/2, BOX_D/2
    corners_local = np.array([
        [-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],
        [-w,-h, d],[w,-h, d],[w,h, d],[-w,h, d],
    ])
    corners_3d = (R @ corners_local.T).T + center_3d
    corners_2d = [project_3d_to_2d(pt, intr) for pt in corners_3d]
    if any(p is None for p in corners_2d):
        return

    def line(i,j,t=2):
        if corners_2d[i] and corners_2d[j]:
            cv2.line(img, corners_2d[i], corners_2d[j], color, t)

    # وجه جلو (روشن‌تر)
    for a,b in [(4,5),(5,6),(6,7),(7,4)]:
        line(a,b,2)
    # وجه عقب (کم‌رنگ‌تر)
    dark = tuple(int(c*0.4) for c in color)
    for a,b in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.line(img, corners_2d[a], corners_2d[b], dark, 1)
    # اتصال
    for a,b in [(0,4),(1,5),(2,6),(3,7)]:
        line(a,b,1)

    center_2d = project_3d_to_2d(center_3d, intr)
    if center_2d:
        cv2.circle(img, center_2d, 5, color, -1)
        cv2.putText(img, f"#{screw_id}",
                    (center_2d[0]+7, center_2d[1]-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # محورها X=قرمز Y=سبز Z=آبی
    axis_len = 0.020
    for vec,col in [(R[:,0],(0,0,255)),
                    (R[:,1],(0,255,0)),
                    (R[:,2],(255,150,0))]:
        tip = project_3d_to_2d(center_3d + vec*axis_len, intr)
        if center_2d and tip:
            cv2.arrowedLine(img, center_2d, tip, col, 2, tipLength=0.25)


def run(rgb_path, depth_npy_path):
    intr      = load_intrinsics(INTRINSICS_PATH)
    model     = YOLO(YOLO_WEIGHTS)
    rgb       = cv2.imread(rgb_path)
    depth_npy = np.load(depth_npy_path)

    if rgb is None:
        print(f"❌ تصویر پیدا نشد: {rgb_path}")
        return []

    img_h, img_w = depth_npy.shape
    results      = model(rgb, conf=CONFIDENCE)[0]
    detections   = []

    # rotation ثابت از normal ریل (همه screwها روی یک سطح)
    R_rail = rotation_from_normal(RAIL_NORMAL)
    roll_r, pitch_r, yaw_r = rotation_to_euler(R_rail)
    print(f"Normal ریل: {RAIL_NORMAL}")
    print(f"Orientation ریل: roll={roll_r:.1f}° pitch={pitch_r:.1f}° yaw={yaw_r:.1f}°\n")

    for i, box in enumerate(results.boxes):
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        u = (x1+x2)/2
        v = (y1+y2)/2

        # depth خود screw
        depth_screw = float(depth_npy[int(v), int(u)])
        if depth_screw == 0:
            patch   = depth_npy[max(0,int(v)-5):int(v)+5,
                                max(0,int(u)-5):int(u)+5]
            nonzero = patch[patch > 0]
            if len(nonzero) == 0:
                print(f"  ⚠️  id={i} depth صفر، رد شد")
                continue
            depth_screw = float(nonzero.mean())

        # depth سطح ریل اطراف screw
        depth_rail, (ex1,ey1,ex2,ey2) = get_rail_depth(
            depth_npy, x1,y1,x2,y2, EXPAND_FACTOR, img_h, img_w, depth_screw)

        # موقعیت 3D با depth ریل
        pos = pixel_to_3d(u, v, depth_rail, intr)

        print(f"  id={i} | rail={depth_rail:.1f}mm  screw={depth_screw:.1f}mm | "
              f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

        color = COLORS[i % len(COLORS)]
        cv2.rectangle(rgb, (ex1,ey1), (ex2,ey2), color, 1)
        draw_3d_box(rgb, pos, R_rail, intr, color, i)

        det = {
            "id": i,
            "confidence": round(conf, 4),
            "center_px": [round(float(u),2), round(float(v),2)],
            "depth_screw_mm": round(depth_screw, 1),
            "depth_rail_mm":  round(depth_rail, 1),
            "position_m": {
                "x": round(float(pos[0]),4),
                "y": round(float(pos[1]),4),
                "z": round(float(pos[2]),4),
            },
            "orientation_deg": {
                "roll":  round(float(roll_r),2),
                "pitch": round(float(pitch_r),2),
                "yaw":   round(float(yaw_r),2),
            },
            "normal_vector": [round(float(n),4) for n in RAIL_NORMAL],
        }
        detections.append(det)

    os.makedirs("results", exist_ok=True)
    basename = os.path.splitext(os.path.basename(rgb_path))[0]
    out_img  = f"results/{basename}_pose.png"
    out_json = f"results/{basename}_pose.json"

    cv2.imwrite(out_img, rgb)
    with open(out_json,"w") as f:
        json.dump({"image":rgb_path,"detections":detections},
                  f, indent=2, default=np_encoder)

    print(f"\n✅ {len(detections)} screw پیدا شد")
    print(f"📁 {out_img}  |  {out_json}")
    return detections


if __name__ == "__main__":
    run(RGB_PATH, DEPTH_NPY_PATH)
