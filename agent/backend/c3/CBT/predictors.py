"""Optional shared CBT predictor instance for application-level imports."""

from .cbt_model import CBTPredictor


cbt_predictor = CBTPredictor()

__all__ = ["CBTPredictor", "cbt_predictor"]
