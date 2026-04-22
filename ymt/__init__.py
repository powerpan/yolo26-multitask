"""Project helpers around vendored Ultralytics YOLO26 multitask stack."""

from ymt.batch import assert_multitask_batch, collate_multitask_samples, validate_multitask_batch


def load_multitask_model(*args, **kwargs):
    """Lazily import and build the multitask model.

    Keeping the import lazy lets ``ymt.batch`` be used in lightweight environments that do not have the full
    Ultralytics dependency stack installed yet.
    """
    from ymt.model import load_multitask_model as _load_multitask_model

    return _load_multitask_model(*args, **kwargs)


__all__ = ["assert_multitask_batch", "collate_multitask_samples", "load_multitask_model", "validate_multitask_batch"]
