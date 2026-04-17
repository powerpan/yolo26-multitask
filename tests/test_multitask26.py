"""Smoke tests for vendored Ultralytics multitask head (no dataset / no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRA_ROOT = ROOT / "third_party" / "ultralytics"
if ULTRA_ROOT.is_dir() and str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from ultralytics.nn.modules.head import MultiTask26  # noqa: E402
from ultralytics.nn.tasks import MultiTaskModel, guess_model_task  # noqa: E402


YAML = ULTRA_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-multitask.yaml"


@pytest.mark.skipif(not YAML.is_file(), reason="vendored ultralytics yaml missing")
def test_multitask_model_build_and_forward():
    model = MultiTaskModel(cfg=str(YAML), ch=3, verbose=False)
    assert isinstance(model.model[-1], MultiTask26)
    x = torch.randn(1, 3, 256, 256)
    model.train()
    out = model(x)
    assert isinstance(out, dict)
    assert set(out) == {"det", "pose", "seg"}
    for k in out:
        assert isinstance(out[k], dict)
        assert "one2many" in out[k] and "one2one" in out[k]

    model.eval()
    out_e = model(x)
    # Eval / val path unwraps the detection branch for compatibility with detection NMS.
    assert not isinstance(out_e, dict)


def test_guess_model_task_multitask_yaml():
    assert guess_model_task(YAML) == "multitask"


def test_multitask_predict_unwraps_det():
    model = MultiTaskModel(cfg=str(YAML), ch=3, verbose=False)
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    out = model.predict(x)
    assert not isinstance(out, dict)


def test_multitask_head_shapes():
    ch = (64, 128, 256)
    feats = [torch.randn(1, c, 32 // (2**i), 32 // (2**i)) for i, c in enumerate(ch)]
    head = MultiTask26(3, 4, (2, 3), 5, 16, 128, reg_max=1, end2end=True, ch=ch)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    for sub in (head.det, head.pose, head.seg):
        sub.stride = head.stride.clone()
    head.train()
    y = head(feats)
    assert y["det"]["one2many"]["boxes"].shape[1] == 4
