"""Dataset-yaml helper: make the relative `path:` in configs/*.yaml resolve against the yaml file."""
import os
import tempfile

import yaml


def resolve_data_yaml(path: str) -> str:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        cfg["path"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), root))
    fd, out = tempfile.mkstemp(suffix=".yaml", prefix="screwpose_data_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return out
