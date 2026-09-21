"""Real DICOM decode and torch forward; synthetic weights are not a quality benchmark."""
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kosti.api import create_app
from kosti.contracts import CSV_COLUMNS, SPINE, VIOLATIONS, ModelUnavailableError
from kosti.ml import build_quality_model, load_quality_predictor
from test_dicom import write_dicom
from test_ml import metadata


@pytest.fixture
def quality_bundle(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = build_quality_model()
    with torch.no_grad():
        model.fc.weight.zero_()
        # Explicit synthetic fixture: spine, axis+foreign body; unrelated hip heads negative.
        model.fc.bias.copy_(torch.tensor([2., -2., -2., 2., 2., -2., -2., 2., -2.]))
    weights = tmp_path / "weights.pth"
    torch.save(model.state_dict(), weights)
    meta = metadata()
    meta["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_real_dicom_torch_api_csv_roundtrip(tmp_path, quality_bundle):
    image = tmp_path / "снимок.dcm"
    ds = write_dicom(image, np.arange(64, dtype=np.uint16).reshape(8, 8))
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("снимок.dcm", image.read_bytes())
        archive.writestr("копия.dcm", image.read_bytes())
        archive.writestr("broken.dcm", b"invalid")
    predictor = load_quality_predictor(quality_bundle)
    with TestClient(create_app(predictor)) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/v1/batch", files={"files": ("input.zip", source.getvalue())})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        reader = csv.DictReader(io.StringIO(archive.read("results.csv").decode("utf-8-sig")))
        assert reader.fieldnames == list(CSV_COLUMNS)
        rows = list(reader)
    assert len(rows) == 3
    for row in rows[:2]:
        assert row["study_uid"] == ds.StudyInstanceUID
        assert row["image_uid"] == ds.SOPInstanceUID
        assert row["anatomical_region"] == SPINE
        assert row["quality_class"] == "1"
        assert row["violation_type"] == "; ".join(VIOLATIONS[SPINE][1:])
        assert float(row["quality_prob"]) == pytest.approx(.880797, abs=1e-5)
    assert rows[0]["path_to_study"] != rows[1]["path_to_study"]
    assert rows[2]["processing_status"] == "Failure"
    assert rows[2]["quality_prob"] == ""


def test_wrong_head_dimensions_fail_closed(tmp_path, quality_bundle):
    torch = pytest.importorskip("torch")
    state = torch.load(tmp_path / "weights.pth", weights_only=True)
    state["fc.weight"] = state["fc.weight"][:2]
    state["fc.bias"] = state["fc.bias"][:2]
    torch.save(state, tmp_path / "weights.pth")
    meta = json.loads(quality_bundle.read_text())
    meta["weights_sha256"] = hashlib.sha256((tmp_path / "weights.pth").read_bytes()).hexdigest()
    quality_bundle.write_text(json.dumps(meta))
    with pytest.raises(ModelUnavailableError, match="size mismatch"):
        load_quality_predictor(quality_bundle)
