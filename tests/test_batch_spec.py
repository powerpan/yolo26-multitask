"""Batch layout validation and multitask loss batch isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "third_party" / "ultralytics"
if ULTRA.is_dir() and str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.nn.tasks import MultiTaskModel  # noqa: E402
from ultralytics.utils.loss import E2EMultiTaskLoss  # noqa: E402

from ymt.batch import assert_multitask_batch, validate_multitask_batch  # noqa: E402

YAML = ULTRA / "ultralytics" / "cfg" / "models" / "26" / "yolo26-multitask.yaml"
KPT = (8, 3)


@pytest.mark.skipif(not YAML.is_file(), reason="yaml missing")
def test_validate_multitask_batch_ok():
    batch = _minimal_batch()
    assert validate_multitask_batch(batch, KPT) == []


def test_validate_detects_mismatched_cls_rows():
    batch = _minimal_batch()
    batch["cls_det"] = torch.zeros(2, 1)  # wrong vs 1 bbox row
    err = validate_multitask_batch(batch, KPT)
    assert any("cls_det" in e for e in err)


def test_validate_detects_bad_batch_idx():
    batch = _minimal_batch()
    batch["batch_idx"] = torch.tensor([[99]], dtype=torch.float32)
    err = validate_multitask_batch(batch, KPT)
    assert any("batch_idx" in e for e in err)


@pytest.mark.skipif(not YAML.is_file(), reason="yaml missing")
def test_e2e_multitask_loss_branch_cls_independent():
    model = MultiTaskModel(cfg=str(YAML), ch=3, verbose=False)
    crit = E2EMultiTaskLoss(model)
    img = torch.zeros(1, 3, 128, 128)
    model.train()
    preds = model(img)

    batch = _minimal_batch()
    batch["cls_det"] = torch.tensor([[0.0]])
    batch["cls_pose"] = torch.tensor([[1.0]])
    batch["cls_seg"] = torch.tensor([[2.0]])
    # Should not raise; branch dicts must not alias each other's cls slot
    crit(preds, batch)


@pytest.mark.skipif(not YAML.is_file(), reason="yaml missing")
def test_multitask_model_set_head_attr_propagates():
    model = MultiTaskModel(cfg=str(YAML), ch=3, verbose=False)
    model.set_head_attr(max_det=77)
    h = model.model[-1]
    assert h.det.max_det == 77 and h.pose.max_det == 77 and h.seg.max_det == 77


def _minimal_batch() -> dict:
    # sem_masks: per-image dense class id map (H,W), required when Segment26 Proto returns (proto, semseg) in training
    return {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.zeros(1, 1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.1, 0.2, 0.05, 0.06]]),
        "keypoints": torch.zeros(1, KPT[0], KPT[1]),
        "masks": torch.zeros(1, 64, 64),
        "sem_masks": torch.zeros(1, 64, 64, dtype=torch.long),
    }


def test_assert_multitask_batch_raises():
    bad = {"img": torch.zeros(1, 3, 8, 8), "batch_idx": torch.zeros(0, 1)}
    with pytest.raises(ValueError):
        assert_multitask_batch(bad, KPT)
