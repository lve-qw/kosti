"""Offline, explicit model bundles. Importing this module does not import torch."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import HIP, SPINE, VIOLATIONS, DicomImage, ModelUnavailableError, Prediction

REGIONS = (SPINE, HIP)
# Fixed semantic indices, separate from configurable anatomy class order.
VIOLATION_LABELS = tuple((region, name) for region in REGIONS for name in VIOLATIONS[region])


def validate_metadata(meta: dict, *, kind: str = "quality") -> dict:
    """Validate decisions that cannot safely be inferred from a checkpoint."""
    if not isinstance(meta, dict):
        raise ValueError("Bundle metadata must be a JSON object")
    if meta.get("schema_version") != 1 or meta.get("kind") != kind:
        raise ValueError("Expected schema_version=1 and explicit bundle kind")
    if meta.get("architecture") != "resnet18":
        raise ValueError("Only resnet18 bundles are supported")
    classes = meta.get("anatomy_classes")
    if not isinstance(classes, list) or len(classes) != 2 or not all(isinstance(c, str) for c in classes) or set(classes) != set(REGIONS):
        raise ValueError("anatomy_classes must give the two exact region names in output order")
    prep = meta.get("preprocessing", {})
    if not isinstance(prep, dict):
        raise ValueError("preprocessing must be an object")
    if prep.get("mode") != "minmax_letterbox_rgb" or prep.get("interpolation") != "bilinear":
        raise ValueError("Explicit minmax_letterbox_rgb / bilinear preprocessing required")
    size = prep.get("size")
    if type(size) is not int or not 32 <= size <= 2048:
        raise ValueError("size must be an integer in [32, 2048]")
    for key in ("mean", "std"):
        values = np.asarray(prep.get(key), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"preprocessing.{key} must contain three finite numbers")
        if key == "std" and (values <= 0).any():
            raise ValueError("std must be positive")
    if prep.get("padding_value") != 0:
        raise ValueError("Explicit padding_value=0 required (before standardization)")
    if not isinstance(meta.get("weights"), str) or not meta["weights"]:
        raise ValueError("weights must name a local file")
    sha = meta.get("weights_sha256", "")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("weights_sha256 must be a lowercase SHA256 digest")
    if kind == "quality":
        expected = [[region, name] for region, name in VIOLATION_LABELS]
        if meta.get("violation_labels") != expected or meta.get("quality_regions") != list(REGIONS):
            raise ValueError("Explicit fixed quality and violation output order required")
        for key, count in (("quality_thresholds", 2), ("violation_thresholds", 5)):
            values = np.asarray(meta.get(key), dtype=float)
            if values.shape != (count,) or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
                raise ValueError(f"{key} must contain {count} thresholds strictly between 0 and 1")
    return meta


def preprocess(pixels: np.ndarray, config: dict) -> np.ndarray:
    """Minmax finite grayscale -> centered square letterbox -> repeated RGB CHW."""
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2 or min(pixels.shape) == 0 or not np.isfinite(pixels).all():
        raise ValueError("Expected a nonempty finite grayscale image")
    minimum, maximum = float(pixels.min()), float(pixels.max())
    if maximum <= minimum:
        raise ValueError("Constant image cannot be assessed")
    pixels = (pixels - minimum) / (maximum - minimum)
    size = config["size"]
    scale = size / max(pixels.shape)
    height, width = [max(1, min(size, round(n * scale))) for n in pixels.shape]
    resized = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.BILINEAR))
    padded = np.zeros((size, size), dtype=np.float32)
    top, left = (size - height) // 2, (size - width) // 2
    padded[top:top + height, left:left + width] = resized
    rgb = np.repeat(padded[None, :, :], 3, axis=0)
    return ((rgb - np.asarray(config["mean"], dtype=np.float32)[:, None, None]) /
            np.asarray(config["std"], dtype=np.float32)[:, None, None])


def build_quality_model():
    """Trainable ResNet18: fc logits = anatomy(2), violations(5), quality(2)."""
    from torchvision.models import resnet18
    import torch
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    return model


def _load(bundle_path: str | Path, kind: str):
    path = Path(bundle_path).resolve()
    meta = validate_metadata(json.loads(path.read_text(encoding="utf-8")), kind=kind)
    weights = (path.parent / meta["weights"]).resolve()
    if not weights.is_relative_to(path.parent):
        raise ValueError("Weights must be inside the bundle directory")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != meta["weights_sha256"]:
        raise ValueError("Weight checksum mismatch")
    import torch
    from torchvision.models import resnet18
    if kind == "quality":
        model = build_quality_model()
    else:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return meta, model


class BundlePredictor:
    def __init__(self, meta: dict, model, device="cpu"):
        self.meta, self.model, self.device = meta, model.to(device), device

    def predict(self, image: DicomImage) -> Prediction:
        import torch
        tensor = torch.from_numpy(preprocess(image.pixels, self.meta["preprocessing"]))[None].to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        if tuple(logits.shape) != (1, 9) or not torch.isfinite(logits).all():
            raise ValueError("Invalid quality model output")
        region = self.meta["anatomy_classes"][int(logits[0, :2].argmax())]
        probabilities = torch.sigmoid(logits[0, 2:7]).tolist()
        if region == REGIONS[0]:
            from .geometry import spine_axis_probability
            geometry = spine_axis_probability(image.pixels, image.spacing_mm)
            if geometry is not None:
                # Equal fixed blend: the CNN captures appearance while the
                # deterministic branch measures the explicit five-degree rule.
                probabilities[1] = (probabilities[1] + geometry) / 2
        violations = {name: probabilities[i] for i, (r, name) in enumerate(VIOLATION_LABELS) if r == region}
        thresholds = {name: self.meta["violation_thresholds"][i]
                      for i, (r, name) in enumerate(VIOLATION_LABELS) if r == region}
        # The public quality class is the OR of applicable violations. Report
        # its score from those same probabilities, preventing contradictory
        # output such as quality_class=1 together with quality_prob < 0.5.
        # Auxiliary quality logits remain in existing checkpoints for research
        # evaluation and backward-compatible state dictionaries.
        quality = max(violations.values())
        return Prediction(region, quality, violations, thresholds)


class AnatomyPredictor:
    """Anatomy-only diagnostic adapter, deliberately not a quality Predictor."""
    def __init__(self, meta: dict, model, device="cpu"):
        self.meta, self.model, self.device = meta, model.to(device), device

    def predict_region(self, pixels: np.ndarray) -> str:
        import torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(preprocess(pixels, self.meta["preprocessing"]))[None].to(self.device))
        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Invalid anatomy model output")
        return self.meta["anatomy_classes"][int(logits.argmax())]


def load_quality_predictor(bundle_path: str | Path, device="cpu") -> BundlePredictor:
    try:
        return BundlePredictor(*_load(bundle_path, "quality"), device=device)
    except (OSError, ValueError, ImportError, RuntimeError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        raise ModelUnavailableError(f"Cannot load quality bundle: {exc}") from exc


def load_anatomy_predictor(bundle_path: str | Path, device="cpu") -> AnatomyPredictor:
    return AnatomyPredictor(*_load(bundle_path, "anatomy"), device=device)
