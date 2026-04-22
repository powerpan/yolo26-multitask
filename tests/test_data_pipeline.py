"""Multitask data pipeline helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymt.batch import collate_multitask_samples  # noqa: E402

KPT = (8, 3)


def test_collate_multitask_samples_builds_ultralytics_batch():
    samples = [
        {
            "img": torch.full((3, 4, 4), 1.0),
            "sem_masks": torch.zeros(4, 4, dtype=torch.long),
            "instances": [
                {
                    "bbox": torch.tensor([0.10, 0.20, 0.30, 0.40]),
                    "cls_det": 1,
                    "cls_pose": 2,
                    "cls_seg": 3,
                    "keypoints": torch.ones(*KPT),
                    "mask": torch.ones(4, 4),
                },
                {
                    "bbox": torch.tensor([0.50, 0.60, 0.20, 0.10]),
                    "cls_det": 4,
                    "cls_pose": 5,
                    "cls_seg": 6,
                    "keypoints": torch.zeros(*KPT),
                    "mask": torch.zeros(4, 4),
                },
            ],
        },
        {
            "img": torch.full((3, 4, 4), 2.0),
            "sem_masks": torch.ones(4, 4, dtype=torch.long),
            "instances": [
                {
                    "bbox": torch.tensor([0.25, 0.25, 0.50, 0.50]),
                    "cls_det": 7,
                    "cls_pose": 8,
                    "cls_seg": 9,
                    "keypoints": torch.full(KPT, 2.0),
                    "mask": torch.full((4, 4), 2.0),
                }
            ],
        },
    ]

    batch = collate_multitask_samples(samples, KPT)

    assert batch["img"].shape == (2, 3, 4, 4)
    assert batch["batch_idx"].tolist() == [[0], [0], [1]]
    assert torch.allclose(batch["bboxes"], torch.tensor([[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.20, 0.10], [0.25, 0.25, 0.50, 0.50]]))
    assert batch["cls_det"].tolist() == [[1.0], [4.0], [7.0]]
    assert batch["cls_pose"].tolist() == [[2.0], [5.0], [8.0]]
    assert batch["cls_seg"].tolist() == [[3.0], [6.0], [9.0]]
    assert batch["keypoints"].shape == (3, *KPT)
    assert batch["masks"].shape == (3, 4, 4)
    assert batch["sem_masks"].shape == (2, 4, 4)
    assert batch["sem_masks"].dtype == torch.long


def test_collate_multitask_samples_can_add_shared_cls_for_legacy_paths():
    samples = [
        {
            "img": torch.zeros(3, 2, 2),
            "instances": [
                {
                    "bbox": torch.tensor([0.1, 0.1, 0.2, 0.2]),
                    "cls": 11,
                }
            ],
        }
    ]

    batch = collate_multitask_samples(samples, KPT, legacy_cls=True)

    assert batch["cls"].tolist() == [[11.0]]
    assert batch["cls_det"].tolist() == [[11.0]]
    assert batch["cls_pose"].tolist() == [[11.0]]
    assert batch["cls_seg"].tolist() == [[11.0]]


def test_collate_multitask_samples_requires_instances_key():
    with pytest.raises(ValueError):
        collate_multitask_samples([{"img": torch.zeros(3, 2, 2)}], KPT)
