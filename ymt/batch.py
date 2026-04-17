"""Multitask batch layout checks aligned with Ultralytics v8 losses."""

from __future__ import annotations

from typing import Any

import torch


def validate_multitask_batch(batch: dict[str, Any], kpt_shape: tuple[int, int]) -> list[str]:
    """Return a list of human-readable issues (empty if OK)."""
    issues: list[str] = []
    if "img" not in batch:
        issues.append("missing key: img")
        return issues
    img = batch["img"]
    if not isinstance(img, torch.Tensor) or img.dim() != 4:
        issues.append("img must be a 4D tensor (B,3,H,W)")
    b = int(img.shape[0]) if isinstance(img, torch.Tensor) else 0

    for k in ("batch_idx", "bboxes"):
        if k not in batch:
            issues.append(f"missing key: {k}")

    bi = batch.get("batch_idx")
    bb = batch.get("bboxes")
    if isinstance(bi, torch.Tensor) and isinstance(bb, torch.Tensor):
        if bi.numel() == 0 and bb.numel() == 0:
            pass
        elif bi.shape[0] != bb.shape[0]:
            issues.append(f"batch_idx length {bi.shape[0]} != bboxes rows {bb.shape[0]}")
        if bb.numel() and bb.shape[-1] != 4:
            issues.append("bboxes must be (N, 4) xywh normalized")
        if bi.numel():
            bad = (bi.view(-1) < 0) | (bi.view(-1) >= b)
            if bad.any():
                issues.append("batch_idx out of range for batch size B")

    nk, nd = kpt_shape
    kp = batch.get("keypoints")
    if kp is not None and isinstance(kp, torch.Tensor):
        if kp.numel() and kp.shape[-2:] != (nk, nd):
            issues.append(f"keypoints trailing shape expected ({nk}, {nd}), got {tuple(kp.shape[-2:])}")

    cls_keys = ("cls", "cls_det", "cls_pose", "cls_seg")
    for ck in cls_keys:
        t = batch.get(ck)
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            issues.append(f"{ck} must be a tensor")
            continue
        if ck == "cls_pose" and kp is not None and isinstance(kp, torch.Tensor) and kp.numel() and t.numel():
            if t.shape[0] != kp.shape[0]:
                issues.append(f"cls_pose rows {t.shape[0]} != keypoints rows {kp.shape[0]}")
        elif ck in ("cls", "cls_det", "cls_seg") and isinstance(bb, torch.Tensor) and bb.numel() and t.numel():
            if t.shape[0] != bb.shape[0]:
                issues.append(f"{ck} rows {t.shape[0]} != bboxes rows {bb.shape[0]}")

    if "masks" in batch:
        m = batch["masks"]
        if not isinstance(m, torch.Tensor):
            issues.append("masks must be a tensor")
        elif m.numel() and "sem_masks" not in batch:
            issues.append(
                "masks present but sem_masks missing — Segment26 training + v8SegmentationLoss expects "
                "batch['sem_masks'] (B,H,W) when proto returns semseg; see docs/BATCH_SPEC.md"
            )

    sm = batch.get("sem_masks")
    if sm is not None:
        if not isinstance(sm, torch.Tensor):
            issues.append("sem_masks must be a tensor")
        elif isinstance(img, torch.Tensor) and sm.dim() == 3 and sm.shape[0] != b:
            issues.append(f"sem_masks batch dim {sm.shape[0]} != img batch B={b}")
    return issues


def assert_multitask_batch(batch: dict[str, Any], kpt_shape: tuple[int, int]) -> None:
    """Raise ValueError if validate_multitask_batch reports issues."""
    err = validate_multitask_batch(batch, kpt_shape)
    if err:
        raise ValueError("multitask batch invalid: " + "; ".join(err))
