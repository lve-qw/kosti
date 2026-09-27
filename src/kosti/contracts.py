"""Shared interfaces. Medical decisions must come from an explicit predictor."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

SPINE = "Поясничный отдел позвоночника"
HIP = "Проксимальный отдел бедра"
VIOLATIONS = {
    SPINE: ("Некорректная укладка", "Не выравнена ось позвоночника", "Присутствуют посторонние предметы"),
    HIP: ("Некорректная укладка", "Некорректная область интереса"),
}
CSV_COLUMNS = (
    "path_to_study", "study_uid", "image_uid", "anatomical_region",
    "quality_class", "violation_type", "processing_status", "time_of_processing", "quality_prob",
)

@dataclass
class DicomImage:
    path: Path
    pixels: np.ndarray
    study_uid: str
    image_uid: str
    pixel_hash: str
    spacing_mm: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    projection: str = "unknown"
    projection_source: str = "unavailable"

@dataclass(frozen=True)
class Prediction:
    anatomical_region: str
    quality_prob: float
    violation_probabilities: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)

class Predictor(Protocol):
    def predict(self, image: DicomImage) -> Prediction: ...

@dataclass
class ResultRow:
    path_to_study: str
    study_uid: str = ""
    image_uid: str = ""
    anatomical_region: str = ""
    quality_class: int | None = None
    violation_type: str = ""
    processing_status: str = "Failure"
    time_of_processing: float = 0.0
    quality_prob: float | None = None

class ModelUnavailableError(RuntimeError):
    """A validated quality model has not been configured."""
