import copy
import numpy as np
import pytest
from kosti.contracts import SPINE, HIP, ModelUnavailableError
from kosti.ml import validate_metadata, preprocess, load_quality_predictor, VIOLATION_LABELS


def metadata():
    return {"schema_version": 1, "kind": "quality", "architecture": "resnet18",
            "anatomy_classes": [SPINE, HIP], "quality_regions": [SPINE, HIP],
            "violation_labels": [list(pair) for pair in VIOLATION_LABELS],
            "weights": "weights.pth", "weights_sha256": "a" * 64,
            "preprocessing": {"mode": "minmax_letterbox_rgb", "size": 32,
                              "interpolation": "bilinear", "padding_value": 0,
                              "mean": [0, 0, 0], "std": [1, 1, 1]},
            "quality_thresholds": [0.5, 0.5], "violation_thresholds": [0.5] * 5}


def test_explicit_metadata_and_anatomy_order():
    m = metadata()
    assert validate_metadata(m) == m
    m["anatomy_classes"].reverse()
    validate_metadata(m)
    del m["preprocessing"]["std"]
    with pytest.raises(ValueError):
        validate_metadata(m)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("kind", "anatomy"),
    ("quality_regions", [HIP, SPINE]), ("violation_thresholds", [0.5] * 4),
    ("weights_sha256", "unknown"), ("anatomy_classes", [SPINE, SPINE])])
def test_reject_ambiguous_bundle(field, value):
    m = metadata()
    m[field] = value
    with pytest.raises(ValueError):
        validate_metadata(m)


def test_preprocessing_preserves_aspect_and_repeats_grayscale():
    result = preprocess(np.arange(32, dtype=float).reshape(4, 8), metadata()["preprocessing"])
    assert result.shape == (3, 32, 32)
    assert np.all(result[:, :8] == 0) and np.all(result[:, 24:] == 0)
    np.testing.assert_array_equal(result[0], result[1])
    assert result.dtype == np.float32
    for bad in (np.ones((2, 2)), np.array([[0, np.nan], [1, 2]])):
        with pytest.raises(ValueError):
            preprocess(bad, metadata()["preprocessing"])


def test_missing_model_is_explicit_failure(tmp_path):
    with pytest.raises(ModelUnavailableError):
        load_quality_predictor(tmp_path / "missing.json")


def test_checksum_rejected_before_torch_import(tmp_path):
    import json
    (tmp_path / "weights.pth").write_bytes(b"not weights")
    (tmp_path / "model.json").write_text(json.dumps(metadata()))
    with pytest.raises(ModelUnavailableError, match="checksum"):
        load_quality_predictor(tmp_path / "model.json")


@pytest.mark.parametrize("bad", [None, [], "metadata"])
def test_metadata_requires_object(bad):
    with pytest.raises(ValueError):
        validate_metadata(bad)
