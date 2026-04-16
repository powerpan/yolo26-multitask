# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class MultiTaskTrainer(DetectionTrainer):
    """Trainer for :class:`~ultralytics.nn.tasks.MultiTaskModel` (detection branch dataset + multitask loss)."""

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a multitask model built from YAML or weights."""
        model = MultiTaskModel(cfg, ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a detection validator (detection metrics on the det branch)."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_model_attributes(self):
        """Set model attributes; multitask keeps per-branch class counts from YAML."""
        self.model.args = self.args
        if not isinstance(self.model, MultiTaskModel):
            super().set_model_attributes()
            return
        head = self.model.model[-1]
        nc_det = int(getattr(head, "nc_det", self.data["nc"]))
        nc_pose = int(getattr(head, "nc_pose", nc_det))
        nc_seg = int(getattr(head, "nc_seg", nc_det))
        self.model.nc = nc_det
        self.model.names = self.data["names"]
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)
        # Detection losses read model.nc; TaskViewModel branches use these attrs.
        head.det.nc = nc_det
        head.pose.nc = nc_pose
        head.seg.nc = nc_seg


def multitask_train(cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
    """Entry point matching other task trainers for dynamic loading."""
    return MultiTaskTrainer(cfg, overrides, _callbacks)
