"""Explicit stages used by the pipeline runner."""

from src.pipeline.steps.dataset import DatasetStage
from src.pipeline.steps.detection import DetectorStage
from src.pipeline.steps.embedding import EmbeddingStage
from src.pipeline.steps.tracking import TrackingStage

__all__ = ["DatasetStage", "EmbeddingStage", "DetectorStage", "TrackingStage"]
