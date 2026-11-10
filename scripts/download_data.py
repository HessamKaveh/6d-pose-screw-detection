"""Download UT57/synthetic_screws (MIT licence) from the Hugging Face Hub as a YOLO dataset.

Images are resized so that the longest side is 1280 px and stored as JPEG
(~0.4 GB instead of ~5.8 GB of 1932x1440 PNGs). Labels are copied unchanged
(class 0 = screw/bolt, class 1 = nut/washer).

    python scripts/download_data.py data/ut57
"""
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

REPO = "UT57/synthetic_screws"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/yolo_data"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/yolo_data"
MAX_SIDE = 1280


def list_dir(path):
    with urllib.request.urlopen(f"{API}/{path}") as r:
        return [x["path"] for x in json.load(r)]


def fetch(root, split, name):
    stem = os.path.splitext(name)[0]
    out_img = f"{root}/images/{split}/{stem}.jpg"
    out_lab = f"{root}/labels/{split}/{stem}.txt"
    if os.path.exists(out_img) and os.path.exists(out_lab):
        return
    err = None
    for _ in range(4):
        try:
            img = urllib.request.urlopen(f"{BASE}/images/{split}/{name}", timeout=120).read()
            lab = urllib.request.urlopen(f"{BASE}/labels/{split}/{stem}.txt", timeout=60).read()
            break
        except Exception as e:  # network hiccup -> retry
            err = e
    else:
        print("FAILED", name, err)
        return
    im = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    s = MAX_SIDE / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_img, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_lab, "wb") as f:
        f.write(lab)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/ut57"
    for split in ("train", "val", "test"):
        os.makedirs(f"{root}/images/{split}", exist_ok=True)
        os.makedirs(f"{root}/labels/{split}", exist_ok=True)
        names = [os.path.basename(p) for p in list_dir(f"images/{split}") if p.endswith(".png")]
        with cf.ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda n: fetch(root, split, n), names))
        print(split, len(os.listdir(f"{root}/images/{split}")), "images", flush=True)


if __name__ == "__main__":
    main()
