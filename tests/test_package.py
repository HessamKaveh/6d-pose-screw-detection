"""Packaging, config helpers and a smoke test of the released detector weights."""
import os

import numpy as np
import pytest
import yaml

import screwpose
from screwpose.data import resolve_data_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights", "screw_yolo11n.pt")


def test_version_matches_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        assert f'version = "{screwpose.__version__}"' in f.read()


def test_resolve_data_yaml(tmp_path):
    cfg = tmp_path / "cfg" / "d.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump({"path": "../data/x", "names": {0: "screw"}}))
    out = resolve_data_yaml(str(cfg))
    try:
        with open(out) as f:
            d = yaml.safe_load(f)
        assert d["path"] == str((tmp_path / "data" / "x").resolve())
        assert d["names"] == {0: "screw"}
    finally:
        os.remove(out)


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="detector weights not present")
def test_detector_smoke():
    pytest.importorskip("ultralytics")
    det = screwpose.ScrewDetector(WEIGHTS, device="cpu", imgsz=320)
    assert set(det.names.values()) >= {"screw"}
    out = det(np.full((240, 320, 3), 127, np.uint8))
    assert isinstance(out, list)
