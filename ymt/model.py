from __future__ import annotations

import os
from pathlib import Path

from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG


def _default_yaml() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / "third_party/ultralytics/ultralytics/cfg/models/26/yolo26-multitask.yaml")


def load_multitask_model(
    yaml_path: str | None = None,
    ch: int = 3,
    nc: int | None = None,
    verbose: bool = False,
) -> MultiTaskModel:
    """Build :class:`ultralytics.nn.tasks.MultiTaskModel` from YAML (vendored by default)."""
    path = yaml_path or os.environ.get("YMT_YAML", _default_yaml())
    model = MultiTaskModel(cfg=path, ch=ch, nc=nc, verbose=verbose)
    if not hasattr(model, "args") or model.args is None:
        model.args = DEFAULT_CFG
    return model
