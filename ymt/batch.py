"""Multitask batch layout checks and collators aligned with Ultralytics v8 losses."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_2d(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype)
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        t = t.view(-1, 1)
    return t


def _tensor_1d(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype)
    return t.view(-1)


def collate_multitask_samples(
    samples: list[dict[str, Any]],
    kpt_shape: tuple[int, int],
    *,
    legacy_cls: bool = False,
) -> dict[str, torch.Tensor]:
    """Collate per-image multitask samples into one Ultralytics-compatible batch.

    Expected input format per sample:
      - ``img``: ``(3,H,W)`` tensor
      - ``sem_masks``: optional ``(H,W)`` dense class-id map
      - ``instances``: list of object dicts, each with at least ``bbox`` and task labels

    Each instance dict should contain:
      - ``bbox``: normalized xywh tensor with 4 values
      - ``cls_det`` / ``cls_pose`` / ``cls_seg``: task-specific class ids
      - ``keypoints``: optional ``(K, D)`` tensor matching ``kpt_shape``
      - ``mask``: optional instance mask tensor

    If ``legacy_cls=True``, the helper accepts ``cls`` instead of task-specific labels and mirrors it into
    ``cls`` / ``cls_det`` / ``cls_pose`` / ``cls_seg`` for compatibility with older code paths.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    images: list[torch.Tensor] = []
    batch_idx: list[torch.Tensor] = []
    bboxes: list[torch.Tensor] = []
    keypoints: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    sem_masks: list[torch.Tensor] = []
    cls: list[torch.Tensor] = []
    cls_det: list[torch.Tensor] = []
    cls_pose: list[torch.Tensor] = []
    cls_seg: list[torch.Tensor] = []

    nk, nd = kpt_shape
    for image_index, sample in enumerate(samples):
        if "img" not in sample:
            raise ValueError(f"sample {image_index} missing required key 'img'")
        if "instances" not in sample:
            raise ValueError(f"sample {image_index} missing required key 'instances'")

        img = sample["img"]
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            raise ValueError(f"sample {image_index} img must be a 3D tensor (3,H,W)")
        images.append(img)

        if "sem_masks" in sample and sample["sem_masks"] is not None:
            sm = sample["sem_masks"]
            if not isinstance(sm, torch.Tensor) or sm.ndim != 2:
                raise ValueError(f"sample {image_index} sem_masks must be a 2D tensor (H,W)")
            sem_masks.append(sm.long())

        for inst_index, inst in enumerate(sample["instances"]):
            if "bbox" not in inst:
                raise ValueError(f"sample {image_index} instance {inst_index} missing required key 'bbox'")
            bbox = _tensor_1d(inst["bbox"], dtype=torch.float32)
            if bbox.numel() != 4:
                raise ValueError(f"sample {image_index} instance {inst_index} bbox must have 4 values")
            bboxes.append(bbox)
            batch_idx.append(torch.tensor([image_index], dtype=torch.long))

            if "keypoints" in inst and inst["keypoints"] is not None:
                kp = torch.as_tensor(inst["keypoints"], dtype=torch.float32)
                if kp.shape != (nk, nd):
                    raise ValueError(
                        f"sample {image_index} instance {inst_index} keypoints shape must be {(nk, nd)}, got {tuple(kp.shape)}"
                    )
                keypoints.append(kp)

            if "mask" in inst and inst["mask"] is not None:
                mask = torch.as_tensor(inst["mask"])
                if mask.ndim != 2:
                    raise ValueError(f"sample {image_index} instance {inst_index} mask must be 2D")
                masks.append(mask)

            if legacy_cls:
                c = inst.get("cls", inst.get("cls_det", inst.get("cls_pose", inst.get("cls_seg"))))
                if c is None:
                    raise ValueError(f"sample {image_index} instance {inst_index} missing cls for legacy_cls=True")
                c2 = _tensor_2d(c, dtype=torch.float32)
                cls.append(c2)
                cls_det.append(c2)
                cls_pose.append(c2)
                cls_seg.append(c2)
            else:
                for key, bucket in (("cls_det", cls_det), ("cls_pose", cls_pose), ("cls_seg", cls_seg)):
                    if key not in inst:
                        raise ValueError(
                            f"sample {image_index} instance {inst_index} missing required key '{key}' (or enable legacy_cls=True)"
                        )
                    bucket.append(_tensor_2d(inst[key], dtype=torch.float32))

    batch: dict[str, torch.Tensor] = {
        "img": torch.stack(images, dim=0),
        "batch_idx": torch.stack(batch_idx, dim=0),
        "bboxes": torch.stack(bboxes, dim=0),
    }

    if legacy_cls:
        batch["cls"] = torch.cat(cls, dim=0)

    if cls_det:
        batch["cls_det"] = torch.cat(cls_det, dim=0)
    if cls_pose:
        batch["cls_pose"] = torch.cat(cls_pose, dim=0)
    if cls_seg:
        batch["cls_seg"] = torch.cat(cls_seg, dim=0)
    if keypoints:
        batch["keypoints"] = torch.stack(keypoints, dim=0)
    if masks:
        batch["masks"] = torch.stack(masks, dim=0)
    if sem_masks:
        batch["sem_masks"] = torch.stack(sem_masks, dim=0)

    return batch


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
