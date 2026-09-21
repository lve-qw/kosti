import csv
from pathlib import Path
import numpy as np
import pytest
from kosti import pipeline
from kosti.contracts import CSV_COLUMNS, DicomImage, Prediction, SPINE, VIOLATIONS


class FakePredictor:
    def predict(self, image):
        return Prediction(SPINE, 0.73, dict(zip(VIOLATIONS[SPINE], [0.1, 0.5, 0.8])))


def fake_read(path):
    if path.name == "bad":
        raise ValueError("not DICOM")
    return DicomImage(path, np.zeros((2, 2)), "1.2", "1.3", "hash")


def test_preserves_duplicates_and_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "read_dicom", fake_read)
    rows, errors = pipeline.process_files([tmp_path / "good", tmp_path / "bad", tmp_path / "good"], FakePredictor(), tmp_path)
    assert [row.processing_status for row in rows] == ["Success", "Failure", "Success"]
    assert rows[0].quality_class == 1
    assert rows[0].quality_prob == 0.73
    assert rows[0].violation_type == "; ".join(VIOLATIONS[SPINE][1:])
    assert rows[1].quality_class is None and rows[1].anatomical_region == ""
    assert len(errors) == 1
    destination = tmp_path / "out.csv"
    pipeline.write_csv(rows, destination)
    assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
    with destination.open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CSV_COLUMNS)
        result = list(reader)
    assert result[1]["quality_class"] == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, None, True, "0.5"])
def test_rejects_invalid_probability(value):
    with pytest.raises(ValueError):
        pipeline.validate_prediction(Prediction(SPINE, value, dict.fromkeys(VIOLATIONS[SPINE], 0.1)))


def test_rejects_missing_violations_and_unknown_region():
    with pytest.raises(ValueError):
        pipeline.validate_prediction(Prediction(SPINE, 0.5, {}))
    with pytest.raises(ValueError):
        pipeline.validate_prediction(Prediction("spine", 0.5, {}))


def test_invalid_threshold_is_isolated(monkeypatch):
    monkeypatch.setattr(pipeline, "read_dicom", fake_read)
    class Invalid:
        def predict(self, image):
            return Prediction(SPINE, 0.5, dict.fromkeys(VIOLATIONS[SPINE], 0.1), {VIOLATIONS[SPINE][0]: float("nan")})
    rows, errors = pipeline.process_files([Path("good")], Invalid())
    assert rows[0].processing_status == "Failure"
    assert rows[0].quality_prob is None
    assert errors[0]["error_type"] == "ValueError"


def test_xlsx_strings_are_not_formulas(tmp_path):
    from openpyxl import load_workbook
    from kosti.contracts import ResultRow
    target = tmp_path / "out.xlsx"
    pipeline.write_xlsx([ResultRow("=1+1")], target)
    book = load_workbook(target)
    assert book.active["A2"].data_type == "s"
    assert book.active["A2"].value == "=1+1"
