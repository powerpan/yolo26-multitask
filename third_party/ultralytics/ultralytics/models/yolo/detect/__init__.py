# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor, MultiTaskPredictor
from .train import DetectionTrainer, MultiTaskTrainer
from .val import DetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "MultiTaskPredictor", "MultiTaskTrainer"
