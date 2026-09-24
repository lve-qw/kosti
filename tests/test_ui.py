"""The browser contract uses the same real DICOM/CSV path as batch export."""
import base64
import csv
import io
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from kosti.api import create_app
from kosti.contracts import CSV_COLUMNS, HIP, Prediction, VIOLATIONS


class FakePredictor:
    def predict(self, image):
        return Prediction(HIP, 0.1, dict.fromkeys(VIOLATIONS[HIP], 0.1))


def _dicom_bytes():
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("synthetic.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SOPInstanceUID = "1.2.3", "1.2.4"
    ds.PatientName = "MustNotBeExported"
    ds.Rows, ds.Columns = 2, 2
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.array([[0, 10], [20, 30]], dtype=np.uint16).tobytes()
    stream = io.BytesIO()
    ds.save_as(stream)
    return stream.getvalue()


def test_interface_assets_and_unconfigured_state():
    with TestClient(create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'lang="ru"' in page.text
        assert "/ui/app.js" in page.text and "/ui/style.css" in page.text
        assert page.headers["cache-control"] == "no-store"
        assert "default-src 'none'" in page.headers["content-security-policy"]
        assert client.get("/ui/app.js").status_code == 200
        assert client.get("/ui/style.css").status_code == 200
        assert client.get("/ui/other.js").status_code == 404
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/review", files={"files": ("test.dcm", _dicom_bytes())}).status_code == 503


def test_review_rows_previews_diagnostics_and_csv():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/review", files=[
            ("files", ("good.dcm", _dicom_bytes())),
            ("files", ("bad.dcm", b"not a dicom")),
        ])
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "MustNotBeExported" not in response.text
    result = response.json()
    assert result["columns"] == list(CSV_COLUMNS)
    assert [row["processing_status"] for row in result["rows"]] == ["Success", "Failure"]
    assert [row["quality_class"] for row in result["rows"]] == [0, None]
    error = next(item for item in result["diagnostics"] if "error_type" in item)
    assert error["path_to_study"] == result["rows"][1]["path_to_study"]
    assert error["error_type"] == "ValueError"
    csv_rows = list(csv.DictReader(io.StringIO(result["csv"].lstrip("\ufeff"))))
    assert list(csv_rows[0]) == list(CSV_COLUMNS)
    assert [row["processing_status"] for row in csv_rows] == ["Success", "Failure"]
    assert list(result["previews"]) == ["0"]
    header, encoded = result["previews"]["0"].split(",", 1)
    assert header == "data:image/png;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.format == "PNG" and image.size == (2, 2)


def test_review_uses_upload_limit():
    with TestClient(create_app(FakePredictor(), max_upload_bytes=100)) as client:
        response = client.post("/v1/review", files={"files": ("x.dcm", b"x" * 101)})
    assert response.status_code == 413


def test_review_accepts_zip_and_preserves_inner_filenames():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("study/image.dcm", _dicom_bytes())
        archive.writestr("study/broken.dcm", b"not a dicom")
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/v1/review", files={"files": ("study.zip", source.getvalue())})
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert [row["processing_status"] for row in rows] == ["Success", "Failure"]
    assert [row["path_to_study"].split("/")[-1] for row in rows] == ["image.dcm", "broken.dcm"]
